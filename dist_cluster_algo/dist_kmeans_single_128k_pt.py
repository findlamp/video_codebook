#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-layer K-Means (K=128k, d=16) — PT variant
- 读取 .pt 文件 (torch.save)；文件里是 dict，使用 --latent_key（默认 'latent'）
- 支持通道优先 [16, T, H, W]（默认 --layout cthw）；也支持通道最后 [T, H, W, 16]（--layout thwc）
- 3D 网格下采样 (--subsample "st,sh,sw")；展开为 [N,16]
- 其他与 dist_kmeans_single_128k.py 相同：多机多卡、Faiss IVF/Flat、可选白化、bf16 输入、W&B

示例（单机单卡 debug）:
  CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=test --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
    dist_kmeans_single_128k_pt.py \
    --data_glob "/data/vae_latents/*.pt" \
    --latent_key latent --layout cthw \
    --dim 16 --num_clusters 1024 --iters 2 \
    --batch_size 200000 --index_type flat \
    --subsample "3,3,5" --use_bf16 \
    --save_dir ./debug_ckpt

正式多机示例同上脚本，把 num_clusters=128000、iters/nlist/nprobe 调大即可。
"""

import os, glob, json, time, math, argparse, gc
import numpy as np
import torch
import torch.distributed as dist
from typing import Iterator, Tuple
import multiprocessing as mp
from queue import Empty as QueueEmpty
import faiss
import faiss.contrib.torch_utils

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False


try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# -----------------------------
# Dist utils
# -----------------------------

def setup_dist():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

def is_rank0():
    return dist.get_rank() == 0

def log0(*a):
    if is_rank0():
        print(*a, flush=True)

# -----------------------------
# File loader: .pt dict -> [N,16] numpy (float32)
# -----------------------------

def pt_to_numpy(path: str, latent_key: str, layout: str, subsample: Tuple[int,int,int] | None):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if latent_key not in obj:
            raise KeyError(f"{path} missing key '{latent_key}' (keys={list(obj.keys())[:8]}...)")
        x = obj[latent_key]
    elif torch.is_tensor(obj):
        x = obj
    else:
        raise TypeError(f"{path} is neither dict nor tensor (type={type(obj)})")

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x[:, 1:]
    # ensure float32 for kmeans domain (we'll cast to bf16/fp16 on GPU later)
    x = x.to(torch.float32)

    if x.ndim == 4:
        # layout either cthw [16,T,H,W] or thwc [T,H,W,16]
        if layout == "cthw":
            C,T,H,W = x.shape
            assert C == 16, f"Expect C=16, got {C}"
            if subsample is not None:
                st,sh,sw = subsample
                x = x[:, 0:T:st, 0:H:sh, 0:W:sw]
            # [16,T',H',W'] -> [T',H',W',16]
            x = x.permute(1,2,3,0).contiguous()
        elif layout == "thwc":
            T,H,W,C = x.shape
            assert C == 16, f"Expect last dim 16, got {C}"
            if subsample is not None:
                st,sh,sw = subsample
                x = x[0:T:st, 0:H:sh, 0:W:sw, :]
        else:
            raise ValueError("layout must be 'cthw' or 'thwc'")
        x = x.reshape(-1, 16)
    elif x.ndim == 2 and x.shape[1] == 16:
        # already flat tokens [N,16]
        pass
    else:
        raise ValueError(f"Unsupported tensor shape {tuple(x.shape)} in {path}")

    return x.numpy()  # float32 numpy [N,16]

# -----------------------------
# Iterator over batches across many small .pt files (sharded by rank)
# -----------------------------
def iter_batches(files, batch_size, device, dtype, latent_key, layout, subsample):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    shard_files = [f for i,f in enumerate(sorted(files)) if i % world_size == rank]
    if rank == 0:
        print(f"[rank0] files assigned: {len(shard_files)}, batch_size={batch_size}", flush=True)

    gpu_dtype = dtype
    buf = []                 # 累计多个文件
    buf_n = 0
    iterator = tqdm(shard_files, desc="rank0: loading .pt", dynamic_ncols=True) if (rank==0 and _HAS_TQDM) else shard_files

    for i, f in enumerate(iterator):
        arr = pt_to_numpy(f, latent_key, layout, subsample)  # numpy float32 [N,16]
        buf.append(arr); buf_n += arr.shape[0]

        # 跨文件拼到 >= batch_size 再上 GPU
        while buf_n >= batch_size:
            need = batch_size
            take = []
            while need > 0:
                a = buf[0]
                if a.shape[0] <= need:
                    take.append(a); need -= a.shape[0]; buf.pop(0)
                else:
                    take.append(a[:need]); buf[0] = a[need:]; need = 0
            xb = np.concatenate(take, axis=0, dtype=np.float32, casting='no')
            xb = torch.from_numpy(xb).to(device=device, dtype=gpu_dtype, non_blocking=True)
            yield xb
            del xb

            buf_n = sum(x.shape[0] for x in buf)
            torch.cuda.empty_cache()

        if rank != 0 and (i % 200 == 0):
            print(f"[rank{rank}] processed {i}/{len(shard_files)} files", flush=True)

    # flush 余量
    if buf_n > 0:
        xb = np.concatenate(buf, axis=0, dtype=np.float32, casting='no')
        xb = torch.from_numpy(xb).to(device=device, dtype=gpu_dtype, non_blocking=True)
        yield xb
        del xb
    torch.cuda.empty_cache()

# -----------------------------
# Whitening stats (μ, Σ)
# -----------------------------

def dist_mean_cov(files, batch_size, device, dtype, latent_key, layout, subsample):
    d = 16
    n_local = 0
    s1 = torch.zeros(d, device=device, dtype=torch.float64)
    s2 = torch.zeros(d, d, device=device, dtype=torch.float64)
    for xb in iter_batches(files, batch_size, device, dtype, latent_key, layout, subsample):
        x = xb.to(torch.float32)
        n_local += x.shape[0]
        s1 += x.sum(0, dtype=torch.float64)
        s2 += (x.t().to(torch.float64) @ x.to(torch.float64))
    n = torch.tensor([n_local], device=device, dtype=torch.float64)
    dist.all_reduce(n); dist.all_reduce(s1); dist.all_reduce(s2)
    n = n.item()
    mu = (s1 / n).to(torch.float32)
    cov = (s2 / n - torch.outer(mu, mu).to(torch.float64)).to(torch.float32)
    return mu, cov


def compute_whitener(cov: torch.Tensor, eps: float, mode: str):
    d = cov.size(0)
    if mode == "none":
        W = torch.eye(d, device=cov.device, dtype=torch.float32)
        Winv = torch.eye(d, device=cov.device, dtype=torch.float32)
        return W, Winv
    if mode == "diag":
        std = torch.sqrt(torch.clamp(torch.diag(cov), min=eps))
        W = torch.diag(1.0/std)
        Winv = torch.diag(std)
        return W.to(torch.float32), Winv.to(torch.float32)
    if mode == "pca":
        S, U = torch.linalg.eigh(cov + eps*torch.eye(cov.size(0), device=cov.device))
        inv_sqrt = torch.diag(1.0/torch.sqrt(torch.clamp(S, min=eps)))
        sqrt = torch.diag(torch.sqrt(torch.clamp(S, min=eps)))
        W = inv_sqrt @ U.t()
        Winv = U @ sqrt
        return W.to(torch.float32), Winv.to(torch.float32)
    raise ValueError("whitening must be one of: none, diag, pca")

# -----------------------------
# Init centers with k-means++ via Faiss (niter=0)
# -----------------------------

def init_centers_kmeanspp(files, args, device, dtype, W, mu):
    d = args.dim; K = args.num_clusters
    per_rank = max(1, args.init_samples // dist.get_world_size())
    got = 0
    bufs = []
    for xb in iter_batches(files, args.batch_size, device, dtype, args.latent_key, args.layout, args.subsample):
        x = xb.to(torch.float32)
        if args.whitening != "none":
            x = (x - mu) @ W.t()
        take = min(per_rank - got, x.size(0))
        bufs.append(x[:take].float().cpu())
        got += take
        if got >= per_rank:
            break
    local = torch.cat(bufs, 0) if bufs else torch.empty(0, d)
    size = torch.tensor([local.size(0)], device=device, dtype=torch.int64)
    all_sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, size)
    all_sizes = [int(s.item()) for s in all_sizes]
    maxn = max(all_sizes)
    pad = maxn - local.size(0)
    if pad>0:
        local_pad = torch.cat([local.to(device), torch.zeros(pad, d, device=device)], 0)
    else:
        local_pad = local.to(device)
    gather_buf = [torch.zeros(maxn, d, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_buf, local_pad)
    if is_rank0():
        cat = [g[:n].cpu().numpy().astype('float32') for g,n in zip(gather_buf, all_sizes)]
        data = np.concatenate(cat, 0)
        km = faiss.Kmeans(d, K, niter=0, verbose=True, seed=123, gpu=True)
        km.train(data)
        C0 = torch.from_numpy(km.centroids.astype('float32')).to(device)
    else:
        C0 = torch.empty(K, d, device=device, dtype=torch.float32)
    dist.broadcast(C0, src=0)
    return C0

# -----------------------------
# Build GPU index (IVF / Flat)
# -----------------------------

def build_index_gpu(centers: torch.Tensor, index_type: str, nlist: int, nprobe: int):
    d = centers.size(1)
    C = centers.detach().float().cpu().numpy()
    if index_type == "flat":
        cpu = faiss.IndexFlatL2(d)
        cpu.add(C)
    elif index_type == "ivf":
        quant = faiss.IndexFlatL2(d)
        cpu = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
        if not cpu.is_trained:
            cpu.train(C)
        cpu.add(C)
        cpu.nprobe = nprobe
    else:
        raise ValueError("index_type must be flat/ivf")
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    gpu_index = faiss.index_cpu_to_gpu(res, torch.cuda.current_device(), cpu, co)
    if index_type == "ivf":
        gpu_index.nprobe = nprobe
    return gpu_index

# -----------------------------
# One iteration
# -----------------------------
@torch.no_grad()
def one_iter(files, args, device, centers, W, mu):
    K, d = centers.size(0), centers.size(1)
    sum_local = torch.zeros(K, d, device=device, dtype=torch.float32)
    cnt_local = torch.zeros(K, device=device, dtype=torch.int64)
    sse_local = torch.zeros(1, device=device, dtype=torch.float64)

    index = build_index_gpu(centers, args.index_type, args.nlist, args.nprobe)

    files_sorted = sorted(files)
    for xb in iter_batches(files_sorted, args.batch_size, device, torch.bfloat16 if args.use_bf16 else torch.float16, args.latent_key, args.layout, args.subsample):
        x = xb.to(torch.float32)
        if args.whitening != "none":
            x = (x - mu) @ W.t()
        D, I = index.search(x, 1)
        labels = I.view(-1).to(torch.int64)
        sse_local += D.sum(dtype=torch.float64)

        uniq, inv = torch.unique(labels, return_inverse=True)
        sums = torch.zeros(uniq.numel(), d, device=device, dtype=torch.float32)
        cnts = torch.zeros(uniq.numel(), device=device, dtype=torch.int64)
        sums.scatter_add_(0, inv.view(-1,1).expand(-1,d), x)
        cnts.scatter_add_(0, inv, torch.ones_like(inv, dtype=torch.int64, device=device))
        sum_local.index_add_(0, uniq, sums)
        cnt_local.index_add_(0, uniq, cnts)

        del xb, x, D, I, labels, uniq, inv, sums, cnts
        torch.cuda.empty_cache()

    dist.all_reduce(sum_local, op=dist.ReduceOp.SUM)
    dist.all_reduce(cnt_local, op=dist.ReduceOp.SUM)
    dist.all_reduce(sse_local, op=dist.ReduceOp.SUM)

    new_centers = centers.clone()
    nz = cnt_local > 0
    new_centers[nz] = sum_local[nz] / cnt_local[nz].unsqueeze(1).float()

    delta = torch.norm(new_centers - centers) / (torch.norm(centers) + 1e-12)
    return new_centers, float(delta), float(sse_local.item()), cnt_local

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", type=str, required=True)
    ap.add_argument("--latent_key", type=str, default="latent")
    ap.add_argument("--layout", choices=["cthw","thwc"], default="cthw")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--num_clusters", type=int, default=128000)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=6_000_000)
    ap.add_argument("--index_type", choices=["ivf","flat"], default="ivf")
    ap.add_argument("--nlist", type=int, default=131072)
    ap.add_argument("--nprobe", type=int, default=48)
    ap.add_argument("--init_samples", type=int, default=200_000_000)
    ap.add_argument("--whitening", choices=["none","diag","pca"], default="none")
    ap.add_argument("--whiten_eps", type=float, default=1e-5)
    ap.add_argument("--subsample", type=str, default="")  # e.g. "3,3,5"
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--resume_centers", type=str, default="")
    ap.add_argument("--wandb_project", type=str, default="")
    ap.add_argument("--wandb_run", type=str, default="")
    args = ap.parse_args()

    setup_dist()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = True

    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise RuntimeError("No files matched data_glob")

    subsample = None
    if args.subsample:
        st,sh,sw = map(int, args.subsample.split(","))
        subsample = (st,sh,sw)
    args.subsample = subsample

    # Whitening
    if args.whitening != "none":
        t0 = time.time()
        mu, cov = dist_mean_cov(files, args.batch_size, device, torch.bfloat16 if args.use_bf16 else torch.float16, args.latent_key, args.layout, args.subsample)
        W, Winv = compute_whitener(cov, args.whiten_eps, args.whitening)
        t1 = time.time()
        if is_rank0():
            log0(f"[whitening] mode={args.whitening} took {t1-t0:.1f}s")
    else:
        mu = torch.zeros(16, device=device)
        W = torch.eye(16, device=device)
        Winv = torch.eye(16, device=device)

    # Init centers
    if args.resume_centers:
        centers = torch.from_numpy(np.load(args.resume_centers)).to(device=device, dtype=torch.float32)
    else:
        centers = init_centers_kmeanspp(files, args, device, torch.bfloat16 if args.use_bf16 else torch.float16, W, mu)

    # W&B
    if is_rank0() and args.wandb_project and WANDB_OK:
        wandb.init(project=args.wandb_project, name=args.wandb_run or None,
                   config={"K":args.num_clusters,"dim":args.dim,"index":args.index_type,
                           "nlist":args.nlist,"nprobe":args.nprobe,"iters":args.iters,
                           "batch_size":args.batch_size,"whitening":args.whitening,
                           "subsample":args.subsample, "layout":args.layout,
                           "latent_key":args.latent_key})

    os.makedirs(args.save_dir, exist_ok=True)

    # rough inertia denominator (optional, for display only)
    total_points_est = 0
    if is_rank0():
        for f in files[:min(len(files), 8)]:
            x = pt_to_numpy(f, args.latent_key, args.layout, args.subsample)
            total_points_est += x.shape[0] * (len(files)//len(files[:min(len(files), 8)]))

    for it in range(1, args.iters+1):
        t0 = time.time()
        new_centers, delta, sse, cnt = one_iter(files, args, device, centers, W, mu)
        centers = new_centers
        t1 = time.time()

        empty = int((cnt==0).sum().item())
        nonempty = int((cnt>0).sum().item())
        maxc = int(cnt.max().item())
        minc = int(cnt[cnt>0].min().item()) if nonempty>0 else 0
        if is_rank0():
            inertia = sse / max(1, total_points_est)
            log0(f"[iter {it:02d}] delta={delta:.3e} sse={sse:.3e} inertia~={inertia:.6e} empty={empty} time={t1-t0:.1f}s index={args.index_type}")
            if args.whitening != "none":
                center_origin = (centers @ Winv.T) + mu
                torch.save(os.path.join(args.save_dir, f"centers_iter_orig{it}.pt"), center_origin.detach().cpu().astype('float32'))
            torch.save(os.path.join(args.save_dir, f"centers_iter{it}.pt"), centers.detach().cpu().astype('float32'))
            with open(os.path.join(args.save_dir, "state.json"), "w") as f:
                json.dump({"iter":it, "delta":delta, "sse":sse, "empty":empty}, f)
            if args.wandb_project and WANDB_OK:
                wandb.log({"iter":it, "delta":delta, "sse":sse, "inertia_est":inertia,
                           "empty_clusters":empty, "nonempty_clusters":nonempty,
                           "min_cluster_size":minc, "max_cluster_size":maxc,
                           "time_sec": (t1-t0)})

        if (args.index_type == "ivf") and (it == args.iters - 2):
            args.index_type = "flat"
        if delta < 1e-3:
            if is_rank0():
                log0(f"Converged at iter {it}, delta={delta:.3e}")
            break

    dist.barrier()
    if is_rank0() and args.wandb_project and WANDB_OK:
        wandb.finish()

if __name__ == "__main__":
    main()
