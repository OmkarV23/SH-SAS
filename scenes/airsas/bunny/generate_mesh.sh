
#old model

# python ../../../inr_reconstruction/upsample_network_with_input_args.py \
#   --exp_name $2 \
#   --experiment_dir ./ \
#   --inr_config ./nbp_config.json \
#   --num_layers 2 \
#   --num_neurons 32 \
#   --output_scene_file_name final_upsampled_scene \
#   --output_dir_name reconstructed_scenes \
#   --system_data $1 \
#   --normalize_scene_dims \
#   --sf 2 \
#   --max_voxels 15000 \
#   --view_model \
#   --threshold 0.1 \
#   --model_path /home/omkar/Desktop/All_Desktop_files/Projetcs/Acoustic_fields/scenes/airsas/bunny_20k/nbp_output/bunny_20k_view_sh_3/models/025000.tar




#new model

python ../../../inr_reconstruction/upsample_network_with_input_args.py \
  --exp_name $2 \
  --experiment_dir ./ \
  --inr_config ./nbp_config.json \
  --num_layers 2 \
  --num_neurons 64 \
  --output_scene_file_name final_upsampled_scene \
  --output_dir_name reconstructed_scenes \
  --system_data $1 \
  --normalize_scene_dims \
  --sf 2 \
  --max_voxels 15000 \
  --view_model \
  --threshold 0.05 \
  --model_path /home/omkar/Desktop/All_Desktop_files/Projetcs/Acoustic_fields/scenes/airsas/bunny_20k/nbp_output/bunny_20k_view_sh_6/models/025000.tar