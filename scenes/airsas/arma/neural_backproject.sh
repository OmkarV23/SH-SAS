#!/bin/bash

echo "System data path: $1"
echo "Fit folder path: $2"
echo "Experiment name: $3"

python ../../../inr_reconstruction/reconstruct_scene_dir_sh.py \
  --scene_inr_config ./nbp_config.json \
  --fit_folder $2 \
  --system_data $1 \
  --output_dir ./nbp_output \
  --plot_thresh 2. \
  --learning_rate 1e-3 \
  --num_epochs 26000 \
  --num_rays 5000 \
  --info_every 25 \
  --scene_every 1000 \
  --accum_grad 5 \
  --scale_factor 3e1 \
  --max_weights 200 \
  --use_up_to 120 \
  --sampling_distribution_uniformity 1.0 \
  --lambertian_ratio 0. \
  --occlusion \
  --occlusion_scale 5e2 \
  --num_layers 2 \
  --num_neurons 32 \
  --reg_start 500 \
  --thresh .15 \
  --smooth_loss 5e1 \
  --smooth_delta 1.0 \
  --sparsity 1e1 \
  --point_at_center \
  --transmit_from_tx \
  --normalize_scene_dims \
  --expname $3 \
  --beamwidth 30 \
  --phase_loss 1e-1 \
  --no_reload \
  --sh_levels 3 \
  # --resample_measurements \
  # --skip_every_n 5



