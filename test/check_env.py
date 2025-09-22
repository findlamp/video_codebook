import torch, faiss
import faiss.contrib.torch_utils  # 关键一步

d = 16
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, d)   # GPU 索引
xb = torch.randn(10000, d, device='cuda', dtype=torch.float32)
xq = torch.randn(100,   d, device='cuda', dtype=torch.float32)

index.add(xb)               # 直接喂 torch CUDA tensor
D_np, I_np = index.search(xq, 1)  # 直接喂 torch CUDA tensor
print(type(D_np), D_np.shape)      # numpy a