import torch, glob
p = sorted(glob.glob("/mnt/teams/algo-teams/ming.gao/latent/spatialvidHQ_vae_latent_5s/*.pt"))[0]
d = torch.load(p, map_location="cpu")
print(list(d.keys())[:10])          # 应该包含 'latents'
print(d["latents"].shape) 