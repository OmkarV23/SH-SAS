#!/bin/bash

python main_render.py \
  --n_azimuth 10 \
  --mesh_name budda \
  --expname sim_arma_20db_20k_1_new \
  --render_output_dir ../reconstructed_mesh/render_output_arma \
  --recon_mesh_dir ../reconstructed_mesh \
  --gt_mesh_dir ../data/gt_meshes \
  --thresh 0.2 \
  --elevation 0.0 \
  --distance 0.3 \
  --fov 60