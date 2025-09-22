torchrun \
  --standalone --nproc_per_node=8 \
  dist_cluster_algo/dist_kmeans_single_128k_pt.py \
  --data_glob "/mnt/algo-fastnas/ming_gao/data/latents/mnt/teams/algo-teams/ming.gao/latent/spatialvidHQ_vae_latent_5s/*.pt" \
  --latent_key latents \
  --layout cthw \
  --dim 16 --num_clusters 128000 \
  --iters 40 --batch_size 200000 \
  --index_type flat \
  --subsample "3,3,5" \
  --use_bf16 \
  --save_dir "./debug_ckpt"