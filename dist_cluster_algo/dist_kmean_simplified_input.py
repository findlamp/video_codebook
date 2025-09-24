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
from collections import OrderedDict
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
class BatchIterator:
    def __init__(self, files, batch_size, device, dtype, latent_key, layout, subsample, cache_max_gb):
        """
        初始化批处理迭代器

        :param files: 输入文件列表
        :param batch_size: 每个批次的大小
        :param device: 设备 (e.g. 'cuda', 'cpu')
        :param dtype: 数据类型（例如 torch.float32）
        :param latent_key: 用于从文件中提取 latent 的键
        :param layout: 数据布局
        :param subsample: 是否对数据进行下采样
        :param cache_size: 缓存大小（默认10个文件）
        """
        self.files = files
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.latent_key = latent_key
        self.layout = layout
        self.subsample = subsample


        self.buf = []      # 缓存文件
        self.buf_n = 0     # 当前缓存的样本数
        self.cache_max_bytes = int(cache_max_gb * (1024**3))
        self._cache = None     # OrderedDict[path -> np.ndarray]
        self._cache_bytes = 0
        
    def _rank_shard(self, files):
        if dist.is_initialized():
            world = dist.get_world_size()
            rank = dist.get_rank()
            self.rank = rank
            files = [f for i,f in enumerate(files) if i % world == rank]
        return files
    
    def _cache_get(self, path):
        if self._cache is None or self.cache_max_bytes <= 0:
            return None
        arr = self._cache.get(path)
        if arr is not None:
            # LRU：刷新到队尾
            self._cache.move_to_end(path)
        return arr

    def _cache_put(self, path, arr: np.ndarray):
        if self.cache_max_bytes <= 0:
            return
        if self._cache is None:
            self._cache = OrderedDict()
            self._cache_bytes = 0
        size = int(arr.nbytes)
        if size > self.cache_max_bytes:
            return
        # 淘汰到放得下为止
        while (self._cache_bytes + size) > self.cache_max_bytes and len(self._cache) > 0:
            _, old = self._cache.popitem(last=False)  # LRU pop
            self._cache_bytes -= int(old.nbytes)
        self._cache[path] = arr
        self._cache_bytes += size
        
        
    def iter_batches(self):
        """
        批处理迭代器，返回一个批次的张量。
        """
        shard_files = self._rank_shard(self.files)
        if self.rank == 0:
            iterator = tqdm(shard_files, desc="Loading .pt files", dynamic_ncols=True)
        else:
            iterator = shard_files
        for i, f in enumerate(iterator):
            # 从文件中读取并转换为 numpy 数组
            try:
                arr = self._cache_get(f)
                #if arr is not None:
                    #print(f"rank{self.rank}[cache] hit {f}", flush=True)
                if arr is None:
                    arr = pt_to_numpy(f, self.latent_key, self.layout, self.subsample)  # np.float32 [N,16]
                    # 保证只读，避免下游意外 in-place 修改污染缓存
                    arr.setflags(write=False)
                    self._cache_put(f, arr)
            except Exception as e:
                bad += 1
                print(f"rank{self.rank}[count] skip {f}: {e}", flush=True)
                continue
            
            self.buf.append(arr)
            self.buf_n += arr.shape[0]

            # 当缓存数据量大于等于 batch_size 时，处理并送到 GPU
            while self.buf_n >= self.batch_size:
                need = self.batch_size
                take = []
                while need > 0:
                    a = self.buf[0]
                    if a.shape[0] <= need:
                        take.append(a)
                        need -= a.shape[0]
                        self.buf.pop(0)
                    else:
                        take.append(a[:need])
                        self.buf[0] = a[need:]
                        need = 0
                xb = np.concatenate(take, axis=0, dtype=np.float32, casting='no')
                xb = torch.from_numpy(xb).to(device=self.device, dtype=self.dtype, non_blocking=True)
                yield xb
                del xb

                # 更新缓存大小
                self.buf_n = sum(x.shape[0] for x in self.buf)
                torch.cuda.empty_cache()

            # 每隔200个文件输出一次进度
            #if i % 200 == 0:
                #print(f"[rank{self.rank}] Processed {i}/{len(shard_files)} files", flush=True)

        # 最后的余量数据处理
        if self.buf_n > 0:
            xb = np.concatenate(self.buf, axis=0, dtype=np.float32, casting='no')
            xb = torch.from_numpy(xb).to(device=self.device, dtype=self.dtype, non_blocking=True)
            yield xb
            del xb

        torch.cuda.empty_cache()



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
    
    #print("before slice", x.shape)
    x = x[:, 1:]
    #print("after slice", x.shape)
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
# Whitening stats (μ, Σ)
# -----------------------------


def dist_mean_cov(batch_iterator, args, device, dtype):
    d = 16
    n_local = 0
    s1 = torch.zeros(d, device=device, dtype=torch.float64)
    s2 = torch.zeros(d, d, device=device, dtype=torch.float64)
    for xb in batch_iterator.iter_batches():
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

def init_centers_kmeanspp(batch_iterator, args, device, dtype, W, mu):
    d = args.dim; K = args.num_clusters
    per_rank = max(1, args.init_samples // dist.get_world_size())
    got = 0
    bufs = []
    for xb in batch_iterator.iter_batches():
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

def init_centers_kmeanspp_reservoir(batch_iterator, args, device, dtype, W, mu,
                          oversample_factor=1.0, reservoir_seed=123):
    import math, random
    d = args.dim; K = args.num_clusters
    world_size = dist.get_world_size()
    # ceil 而不是 floor，并支持 oversample
    target_total = max(K * 10, int(args.init_samples * oversample_factor))
    per_rank = max(1, math.ceil(target_total / world_size))

    # --- 每个 rank 做蓄水池采样（对全数据均匀采样）---
    rng = random.Random(reservoir_seed + dist.get_rank())
    reservoir = []   # list of CPU float32 tensors (row vectors)
    seen = 0
    for xb in batch_iterator.iter_batches():
        x = xb.to(torch.float32)
        if args.whitening != "none":
            x = (x - mu) @ W.t()
        x = x.detach()
        n = x.size(0)
        if len(reservoir) < per_rank:
            take = min(per_rank - len(reservoir), n)
            reservoir.extend(x[:take].cpu().split(1, dim=0))
            seen += take
        # 蓄水池替换
        for i in range(seen, seen + (n - max(0, per_rank - len(reservoir)))):
            j = rng.randint(0, i)
            if j < per_rank:
                # 随机替换为当前样本中的某个
                idx_in_batch = max(0, i - seen)  # 当前批内索引（近似）
                # 为避免逐元素循环，简单选一个随机行替换
                ridx = rng.randint(0, n-1)
                reservoir[j] = x[ridx:ridx+1].cpu()
        seen += n

        if len(reservoir) >= per_rank:
            # 已达到目标就可以提前结束（也可不 break，多看数据公平性更好）
            pass

    if len(reservoir) == 0:
        local = torch.empty(0, d, dtype=torch.float32)
    else:
        local = torch.cat(reservoir, dim=0)[:per_rank].contiguous()  # [per_rank, d]

    # --- gather 变长 ---
    size = torch.tensor([local.size(0)], device=device, dtype=torch.int64)
    all_sizes = [torch.zeros_like(size) for _ in range(world_size)]
    dist.all_gather(all_sizes, size)
    all_sizes = [int(s.item()) for s in all_sizes]
    maxn = max(all_sizes)
    pad = maxn - local.size(0)
    local_pad = (torch.cat([local.to(device), torch.zeros(pad, d, device=device)], 0)
                 if pad > 0 else local.to(device))

    gather_buf = [torch.empty(maxn, d, device=device) for _ in range(world_size)]
    dist.all_gather(gather_buf, local_pad)

    if is_rank0():
        cat = [g[:n].cpu().numpy().astype('float32') for g, n in zip(gather_buf, all_sizes)]
        data = np.concatenate(cat, 0)  # 这里 data 的行数≈ target_total
        # 用更多点来做 kmeans++ 初始化（niter=0 只做初始化）
        km = faiss.Kmeans(d, K, niter=0, verbose=True, seed=reservoir_seed, gpu=True)
        km.train(data)
        C0 = torch.from_numpy(km.centroids.astype('float32')).to(device)
    else:
        C0 = torch.empty(K, d, device=device, dtype=torch.float32)
    dist.broadcast(C0, src=0)
    return C0

@torch.no_grad()
def init_centers_kmeans_parallel(batch_iterator, args, device, dtype, W, mu,
                                 l=None, rounds=3, max_candidates=200000,
                                 use_fp16_comm=True, seed=123):
    """
    分布式 k-means|| 初始化。
    返回: C0 [K, d] torch.float32 on device
    参数:
      - l: 每轮过采样数（总数）。默认 2K~10K 之间较稳：若 None 用 8*K。
      - rounds: 轮数（2~5 通常够用）
      - max_candidates: 候选上限，避免极端场景撑爆内存
      - use_fp16_comm: all_gather 时用半精度减少带宽
    依赖:
      - batch_iterator.iter_batches() 迭代器，返回 [N, d]
      - args.dim, args.num_clusters, args.whitening
    """
    torch.manual_seed(seed + dist.get_rank())
    d = args.dim
    K = args.num_clusters
    world = dist.get_world_size()
    rng = torch.Generator(device=device).manual_seed(seed + dist.get_rank())

    # ---------- Step 0: 选一个初始中心 ----------
    # 简单做法：rank0 从首批里随便取1个，broadcast 出去
    first_point = None
    for xb in batch_iterator.iter_batches():
        x = xb.to(torch.float32)
        if args.whitening != "none":
            x = (x - mu) @ W.t()
        if x.numel() > 0:
            first_point = x[0:1].to(device)
        break
    if first_point is None:
        # 空数据保护
        C0 = torch.zeros(K, d, device=device, dtype=torch.float32)
        return C0

    # 广播这一个初始中心
    one = torch.zeros(1, d, device=device)
    if is_rank0():
        one.copy_(first_point)
    dist.broadcast(one, src=0)
    centers = one.clone()  # [m, d]，动态扩展

    # 过采样数 l 默认设为 8K（经验上挺稳），可按需改成 2K~10K
    if l is None:
        l = max(2*K, min(8*K, 20000))

    # ---------- 定义一个函数：计算到当前中心集合的 d2 ----------
    def min_dist2(x, centers):
        # x: [n, d], centers: [m, d]
        # 使用 (x - c)^2 = |x|^2 + |c|^2 - 2 x·c
        # 避免一次性展开太大，可分块；这里先简单实现
        x2 = (x * x).sum(dim=1, keepdim=True)      # [n,1]
        c2 = (centers * centers).sum(dim=1)        # [m]
        # [n, m]
        prod = x @ centers.t()
        dist2 = x2 + c2.unsqueeze(0) - 2.0 * prod
        d2 = dist2.min(dim=1).values.clamp_min_(0)
        return d2  # [n]

    # ---------- k-means|| 主循环 ----------
    for r in range(rounds):
        # Pass A: 全局距离和 S = sum_x D(x)^2
        local_sum = torch.zeros(1, device=device)
        for xb in batch_iterator.iter_batches():
            x = xb.to(torch.float32)
            if args.whitening != "none":
                x = (x - mu) @ W.t()
            d2 = min_dist2(x.to(device), centers)
            local_sum += d2.sum()

        global_sum = local_sum.clone()
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)

        # 边界：如果全局和接近 0，说明所有点都几乎被覆盖，提前退出
        if float(global_sum.item()) <= 1e-20:
            break

        # Pass B: 按 p(x)=min(1, l * d2 / S) 进行伯努利抽样，收集候选
        # 为了减少通信，先在各自 rank 收集本地候选，再 all_gather 变长拼接
        local_cand = []

        for xb in batch_iterator.iter_batches():
            x = xb.to(torch.float32)
            if args.whitening != "none":
                x = (x - mu) @ W.t()
            x = x.to(device)
            d2 = min_dist2(x, centers)
            p = (l * d2 / global_sum).clamp_max_(1.0)
            # 采样
            mask = torch.rand_like(p, generator=rng) < p
            if mask.any():
                local_cand.append(x[mask])

        local_cand = (torch.cat(local_cand, dim=0)
                      if len(local_cand) > 0 else torch.empty(0, d, device=device))

        # 变长 all_gather
        size = torch.tensor([local_cand.size(0)], device=device, dtype=torch.int64)
        sizes = [torch.zeros_like(size) for _ in range(world)]
        dist.all_gather(sizes, size)
        sizes = [int(s.item()) for s in sizes]
        maxn = max(sizes)

        # 通信精度压缩（可选）
        send = local_cand
        if use_fp16_comm:
            send = send.to(torch.float16)

        pad = maxn - send.size(0)
        if pad > 0:
            send = torch.cat([send, torch.zeros(pad, d, device=device, dtype=send.dtype)], dim=0)

        recv_buf = [torch.empty(maxn, d, device=device, dtype=send.dtype) for _ in range(world)]
        dist.all_gather(recv_buf, send)

        # rank0 汇总候选并回传“是否继续”的小信号（这里直接在各 rank 上扩展 centers 更简单）
        # 为了减少中心集膨胀，这里在本地（各rank）先拼起来再广播新增中心数
        new_cand = []
        for g, n in zip(recv_buf, sizes):
            if n > 0:
                cc = g[:n].to(torch.float32)
                new_cand.append(cc)
        if len(new_cand) > 0:
            new_cand = torch.cat(new_cand, dim=0)
        else:
            new_cand = torch.empty(0, d, device=device, dtype=torch.float32)

        # 合并到 centers（注意去重/上限控制）
        # 简化处理：随机打乱后拼接，再裁到 max_candidates
        if new_cand.numel() > 0:
            centers = torch.cat([centers, new_cand], dim=0)
            # 打乱
            perm = torch.randperm(centers.size(0), device=device, generator=rng)
            centers = centers[perm]
            if centers.size(0) > max_candidates:
                centers = centers[:max_candidates].contiguous()

    # ---------- 最终在“候选集合”上做一次小规模 kmeans（或 kmeans++） ----------
    # gather centers 到 rank0
    size = torch.tensor([centers.size(0)], device=device, dtype=torch.int64)
    sizes = [torch.zeros_like(size) for _ in range(world)]
    dist.all_gather(sizes, size)
    sizes = [int(s.item()) for s in sizes]
    maxn = max(sizes)

    send = centers
    if use_fp16_comm:
        send = send.to(torch.float16)
    pad = maxn - send.size(0)
    if pad > 0:
        send = torch.cat([send, torch.zeros(pad, d, device=device, dtype=send.dtype)], dim=0)

    recv_buf = [torch.empty(maxn, d, device=device, dtype=send.dtype) for _ in range(world)]
    dist.all_gather(recv_buf, send)

    if is_rank0():
        cat = []
        for g, n in zip(recv_buf, sizes):
            if n > 0:
                cat.append(g[:n].to(torch.float32).cpu().numpy().astype('float32'))
        cand_all = np.concatenate(cat, axis=0) if len(cat) else np.zeros((K, d), dtype='float32')

        # 再做一次抽样以防候选过多（可选）
        if cand_all.shape[0] > max_candidates:
            idx = np.random.RandomState(seed).choice(cand_all.shape[0], size=max_candidates, replace=False)
            cand_all = cand_all[idx]

        # 用小规模kmeans得到 K 个初始中心。
        # niter=0 => 只做 kmeans++ 初始化；也可以给个小的 niter=5~20 让它更稳。
        km = faiss.Kmeans(d, K, niter=0, verbose=True, seed=seed, gpu=True)
        km.train(cand_all)
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
def one_iter(batch_iterator, args, device, centers, W, mu):
    K, d = centers.size(0), centers.size(1)
    dtype = centers.dtype
    sum_local = torch.zeros(K, d, device=device, dtype=torch.float32)
    cnt_local = torch.zeros(K, device=device, dtype=torch.int64)
    sse_local = torch.zeros(1, device=device, dtype=torch.float64)

    index = build_index_gpu(centers, args.index_type, args.nlist, args.nprobe)

    for xb in batch_iterator.iter_batches():
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
    ap.add_argument("--cache_max_gb", type=float, default=32.0, help="Max GB to use for caching .pt files in RAM")
    ap.add_argument("--init_method", choices=["init_batch","reservoir","kmeans||"], default="init_batch",)
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

    
    batch_iterator = BatchIterator(files, batch_size=args.batch_size, 
                                   device=device, dtype=torch.float32, 
                                   latent_key=args.latent_key, 
                                   layout=args.layout, subsample=args.subsample,
                                   cache_max_gb=args.cache_max_gb,)
    
    
    # Whitening
    if args.whitening != "none":
        t0 = time.time()
        mu, cov = dist_mean_cov(batch_iterator, args, device, torch.bfloat16 if args.use_bf16 else torch.float16)
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
        centers = torch.load(args.resume_centers).to(device=device, dtype=torch.float32)
    else:
        if args.init_method == "init_batch":
            centers = init_centers_kmeanspp(batch_iterator,  args, device, torch.bfloat16 if args.use_bf16 else torch.float16, W, mu)
        elif args.init_method == "reservoir":
            centers = init_centers_kmeanspp_reservoir(batch_iterator,  args, device, torch.bfloat16 if args.use_bf16 else torch.float16, W, mu,
                                                      oversample_factor=1.0, reservoir_seed=123)
        elif args.init_method == "kmeans||":
            centers = init_centers_kmeans_parallel(batch_iterator, args, device, torch.bfloat16 if args.use_bf16 else torch.float16, W, mu,
                                                   l=None, rounds=3, max_candidates=200000,
                                                   use_fp16_comm=True, seed=123)
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

    if args.init_method == "init_batch":
        batch_iterator = BatchIterator(files, batch_size=args.batch_size, 
                                    device=device, dtype=torch.float32, 
                                    latent_key=args.latent_key, 
                                    layout=args.layout, subsample=args.subsample,
                                    cache_max_gb=args.cache_max_gb,)
    
    
    for it in range(1, args.iters+1):
        t0 = time.time()
        new_centers, delta, sse, cnt = one_iter(batch_iterator, args, device, centers, W, mu)
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
                torch.save(center_origin.detach().cpu(), os.path.join(args.save_dir, f"centers_iter_orig{it}.pt"))
            torch.save(centers.detach().cpu(), os.path.join(args.save_dir, f"centers_iter{it}.pt"))
            record = {
                "iter": it,
                "delta": float(delta),
                "sse": float(sse),
                "inertia": float(inertia),
                "empty": int(empty)
            }

            log_path = os.path.join(args.save_dir, "state.json")
            # 以追加模式写入，每条记录一行 JSON
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
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
