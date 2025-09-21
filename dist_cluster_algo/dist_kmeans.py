#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-layer K-Means (K=128k, d=16) for massive latents.
- 8x8 multi-node multi-GPU via torch.distributed (NCCL)
- Faiss GPU for nearest-centroid search (IVF for speed, Flat for refine)
- Optional PCA whitening (and exact inverse for de-whitening)
- bf16 input, FP32 accumulation/centroids
- Double-buffer streaming I/O from .npy shards (+ deterministic subsampling on a 3D grid)
- W&B metrics: delta, SSE, inertia, cluster size stats, usage histogram

Start example:
  torchrun \
    --nnodes=8 --nproc_per_node=8 \
    --rdzv_id=kmeans-128k \
    --rdzv_backend=c10d \
    --rdzv_endpoint=<MASTER_IP>:29400 \
    dist_kmeans_single_128k.py \
    --data_glob "/mnt/datasets/latents/*.npy" \
    --dim 16 --num_clusters 128000 \
    --iters 12 --batch_size 6000000 \
    --index_type ivf --nlist 131072 --nprobe 48 \
    --subsample "3,3,5" \
    --whitening pca \
    --init_samples 200000000 \
    --save_dir "/mnt/checkpoints/km128k" \
    --wandb_project kmeans-128k --wandb_run single-ivf-bf16
"""

import os, glob, json, time, math, argparse, gc
import numpy as np
import torch
import torch.distributed as dist
from typing import Iterator, Tuple

import faiss
from faiss.contrib import torch_utils as faiss_t

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

# -----------------------------
# Utils: distributed init/log
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
# Subsampling: deterministic 3D grid downsample for (T,H,W,16)
# If your .npy is [N,16] flat tokens, subsample is ignored.
# -----------------------------

def grid_subsample(arr: np.ndarray, subsample: Tuple[int,int,int]) -> np.ndarray:
    """arr can be [T,H,W,16] or [N,16]. Returns [M,16]."""
    if arr.ndim == 4 and arr.shape[-1] == 16:
        st, sh, sw = subsample
        T,H,W,_ = arr.shape
        sl_t = slice(0, T, st)
        sl_h = slice(0, H, sh)
        sl_w = slice(0, W, sw)
        picked = arr[sl_t, sl_h, sl_w, :]  # [T',H',W',16]
        return picked.reshape(-1, 16)
    elif arr.ndim == 2 and arr.shape[1] == 16:
        return arr  # no-op for flat
    else:
        raise ValueError(f"Unexpected array shape {arr.shape}")

# -----------------------------
# Double-buffer stream loader with optional grid subsample
# -----------------------------

def iter_batches(files, batch_size, device, dtype, subsample) -> Iterator[torch.Tensor]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = [f for i,f in enumerate(sorted(files)) if i % world_size == rank]
    for f in files:
        arr = np.load(f, mmap_mode="r")
        if subsample is not None:
            arr = grid_subsample(arr, subsample)
        n = arr.shape[0]
        off = 0
        next_buf = None
        def to_gpu(start, b):
            x = torch.from_numpy(arr[start:start+b])
            return x.to(device=device, dtype=dtype, non_blocking=True)
        while off < n:
            b = min(batch_size, n-off)
            if next_buf is None:
                next_buf = to_gpu(off, b)
                off += b
            cur = next_buf
            next_buf = None
            if off < n:
                b2 = min(batch_size, n-off)
                next_buf = to_gpu(off, b2)
                off += b2
            yield cur  # [B,16]
            del cur
            torch.cuda.empty_cache()
        del arr
        gc.collect()

# -----------------------------
# Global stats for whitening (μ, Σ) with FP64 accumulation
# -----------------------------

def dist_mean_cov(files, batch_size, device, dtype, subsample):
    d = 16
    n_local = 0
    s1 = torch.zeros(d, device=device, dtype=torch.float64)
    s2 = torch.zeros(d, d, device=device, dtype=torch.float64)
    for xb in iter_batches(files, batch_size, device, dtype, subsample):
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
# Initialization with k-means++ via Faiss (niter=0)
# -----------------------------

def init_centers_kmeanspp(files, args, device, dtype, W, mu):
    d = args.dim; K = args.num_clusters
    per_rank = max(1, args.init_samples // dist.get_world_size())
    got = 0
    bufs = []
    for xb in iter_batches(files, args.batch_size, device, dtype, args.subsample):
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
# Build GPU index over centers (IVF or Flat)
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
# One iteration: assignment + local reduce + global allreduce
# Also returns global SSE and cluster counts for metrics
# -----------------------------
@torch.no_grad()
def one_iter(files, args, device, centers, W, Winv, mu):
    K, d = centers.size(0), centers.size(1)
    sum_local = torch.zeros(K, d, device=device, dtype=torch.float32)
    cnt_local = torch.zeros(K, device=device, dtype=torch.int64)
    sse_local = torch.zeros(1, device=device, dtype=torch.float64)

    # Build index in whitened space (centers are assumed in whitened space already)
    index = build_index_gpu(centers, args.index_type, args.nlist, args.nprobe)

    files_sorted = sorted(files)
    for xb in iter_batches(files_sorted, args.batch_size, device, torch.bfloat16 if args.use_bf16 else torch.float16, args.subsample):
        x = xb.to(torch.float32)  # search in fp32
        if args.whitening != "none":
            x = (x - mu) @ W.t()
        # 1-NN in whitened space
        D, I = faiss_t.search(index, x, 1)  # D:[B,1] sqdist, I:[B,1]
        labels = I.view(-1).to(torch.int64)
        sse_local += D.sum(dtype=torch.float64)

        # local sum/count
        uniq, inv = torch.unique(labels, return_inverse=True)
        sums = torch.zeros(uniq.numel(), d, device=device, dtype=torch.float32)
        cnts = torch.zeros(uniq.numel(), device=device, dtype=torch.int64)
        sums.scatter_add_(0, inv.view(-1,1).expand(-1,d), x)
        cnts.scatter_add_(0, inv, torch.ones_like(inv, dtype=torch.int64, device=device))
        sum_local.index_add_(0, uniq, sums)
        cnt_local.index_add_(0, uniq, cnts)

        del xb, x, D, I, labels, uniq, inv, sums, cnts
        torch.cuda.empty_cache()

    # Global reduce
    dist.all_reduce(sum_local, op=dist.ReduceOp.SUM)
    dist.all_reduce(cnt_local, op=dist.ReduceOp.SUM)
    dist.all_reduce(sse_local, op=dist.ReduceOp.SUM)

    # Update centers (in whitened space)
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
    ap.add_argument("--use_bf16", action="store_true", help="use bf16 input (default fp16)")
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

    # parse subsample
    subsample = None
    if args.subsample:
        st,sh,sw = map(int, args.subsample.split(","))
        subsample = (st,sh,sw)
    args.subsample = subsample

    # Whitening stats
    if args.whitening != "none":
        t0 = time.time()
        mu, cov = dist_mean_cov(files, args.batch_size, device, torch.bfloat16 if args.use_bf16 else torch.float16, subsample)
        W, Winv = compute_whitener(cov, args.whiten_eps, args.whitening)
        t1 = time.time()
        if is_rank0():
            log0(f"[whitening] mode={args.whitening} took {t1-t0:.1f}s")
    else:
        mu = torch.zeros(16, device=device)
        W = torch.eye(16, device=device)
        Winv = torch.eye(16, device=device)

    # Init centers in whitened space
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
                           "subsample":args.subsample})

    os.makedirs(args.save_dir, exist_ok=True)

    # Iterations
    total_points_est = 0
    if is_rank0():
        # rough estimate for inertia normalization (only for display)
        for f in files[:min(len(files), 8)]:
            arr = np.load(f, mmap_mode="r")
            if subsample is not None:
                arr = grid_subsample(arr, subsample)
            total_points_est += arr.shape[0] * (len(files)//len(files[:min(len(files), 8)]))

    for it in range(1, args.iters+1):
        t0 = time.time()
        new_centers, delta, sse, cnt = one_iter(files, args, device, centers, W, Winv, mu)
        centers = new_centers
        t1 = time.time()

        # Metrics/logging
        empty = int((cnt==0).sum().item())
        nonempty = int((cnt>0).sum().item())
        maxc = int(cnt.max().item())
        minc = int(cnt[cnt>0].min().item()) if nonempty>0 else 0
        if is_rank0():
            inertia = sse / max(1, total_points_est)
            log0(f"[iter {it:02d}] delta={delta:.3e} sse={sse:.3e} inertia~={inertia:.6e} empty={empty} time={t1-t0:.1f}s index={args.index_type}")
            # Save checkpoint
            np.save(os.path.join(args.save_dir, f"centers_iter{it}.npy"), centers.detach().cpu().numpy().astype('float32'))
            with open(os.path.join(args.save_dir, "state.json"), "w") as f:
                json.dump({"iter":it, "delta":delta, "sse":sse, "empty":empty}, f)
            # W&B
            if args.wandb_project and WANDB_OK:
                wandb.log({"iter":it, "delta":delta, "sse":sse, "inertia_est":inertia,
                           "empty_clusters":empty, "nonempty_clusters":nonempty,
                           "min_cluster_size":minc, "max_cluster_size":maxc,
                           "time_sec": (t1-t0)})

        # switch to flat for last 2-3 iters (refine)
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
