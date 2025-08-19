#!/bin/bash

python ../../../inr_reconstruction/novel_view_synthesis.py \
  --scene_inr_config ./nbp_config.json \
  --fit_folder ./deconvolved_measurements \
  --system_data ./system_data.pik \
  --output_dir ./nbp_output \
  --num_rays 5000 \
  --scale_factor 3e1 \
  --max_weights 200 \
  --use_up_to 120 \
  --sampling_distribution_uniformity 1.0 \
  --lambertian_ratio 0. \
  --occlusion \
  --occlusion_scale 5e2 \
  --num_layers 2 \
  --num_neurons 32 \
  --point_at_center \
  --transmit_from_tx \
  --normalize_scene_dims \
  --expname arma_20k_albert \
  --beamwidth 30 \
  --sh_levels 3 \
  --rand_batch 25000 \
  --model_type nvrcsas \
  --ft_path ./nbp_output/arma_20k_albert/models/020000.tar
