import torch
import time
from tu_pth_dist.perf_tools import measure_and_print_torch, measure_duration_torch
device = "cuda:0"
dtype = torch.bfloat16
generate_latent = torch.randn((16, 3 * 60 * 104), device=device, dtype=dtype)
codebook = torch.randn((128000, 16), device=device, dtype=dtype)    
durs = []

for i in range(10):
    with measure_and_print_torch("ref-bwd"):
        distance = generate_latent.T @ codebook.T  # (18720, 128000)
        indices = distance.argmax(dim=-1)          # (18720,)

    print(indices.shape, indices.min(), indices.max())  # (18720,) 0 127999