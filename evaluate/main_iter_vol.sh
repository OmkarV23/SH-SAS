#!/bin/bash

python main_iter_vol.py \
  --output_dir ../iter_volume \
  --system_data_path ../scenes/simulated/xyz_dragon/system_data_20db.pik \
  --object albert_xyz_dragon \
  --comp_albedo_path ../scenes/simulated/xyz_dragon/nbp_output/sim_xyz_dragon_20db_20k_1_albert/numpy/comp_albedo30000.npy \