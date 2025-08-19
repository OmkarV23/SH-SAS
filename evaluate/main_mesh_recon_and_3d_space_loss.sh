#!/bin/bash

python main_mesh_recon_and_3d_space_loss.py \
  --output_dir ../reconstructed_mesh \
  --system_data_path ../scenes/simulated/buddha/system_data_20db.pik \
  --expname sim_buddha_20db_20k_1_new_60 \
  --mesh_name budda \
  --comp_albedo_path ../scenes/simulated/buddha/nbp_output/sim_buddha_20db_20k_1_new/numpy/comp_albedo60000.npy \
  --csv_file_name buddha \
  --gt_mesh_dir /home/omkar/Desktop/All_Desktop_files/Projetcs/Acoustic_fields/data/gt_meshes \
  --thresh 0.1
  #--scene_inr_result_directory ../experiments/scene_inr_result \
