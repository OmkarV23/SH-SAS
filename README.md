# SH-SAS

## Installation

### Create Conda environment
```bash
conda create -n sh_sas python=3.9
conda activate sh_sas
```

### gcc and gxx
```bash
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11
# chmod +x ./extras/env_mods.sh
# ./extras/env_mods.sh
# conda deactivate && conda activate sh_sas
```

### Cuda-toolkit
```bash
conda install conda-forge::cudatoolkit nvidia/label/cuda-11.8.0::cuda-toolkit
```

### PyTorch
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

### Packages install
```bash
pip install -e .
```

### Install tiny-cuda
```bash
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH}"

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Install pytorch3d
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Compile custom CUDA kernels for Spherical Harmonic basis projection
```bash
cd ./inr_reconstruction/eval-sh
pip install .
cd ../../
```

<!-- Install custom PyMCubes
```bash
cd ./extras/PyMCubes
pip install .
cd ../../
``` -->

## Data
We are providing one example dataset for testing purposes. You can find it in the `SH-SAS-data` directory.
There are two folders: `airsas` and `simulated`. The `airsas` folder contains real AirSAS data, while the `simulated` folder contains synthetic data. Both contains a `system_data.pik` file and precomputed pulse deconvolved measurements in the `deconvolved_measurements` folder.

## Running experiments

### Pulse Deconvolution (Optional)
If you still want to run pulse deconvolution and test it by yourself, 
```bash
cd ./scenes/simulated/xyz_dragon
chmod +x pulse_deconvolve.sh

# Run pulse deconvolution script
./pulse_deconvolve.sh <path_to_system_data_in_SH-SAS-data>
# e.g. ./pulse_deconvolve.sh ../../../../SH-SAS-data/simulated/xyz_dragon/system_data.pik
```

### Neural Backprojection
For simplicity, we have already provided pulse deconvolved measurements for the synthetic xyz dragon and real AirSAS scene. You can run the neural backprojection script directly.
```bash
cd ./scenes/simulated/xyz_dragon
chmod +x neural_backproject.sh

# Run neural backprojection script
./neural_backproject.sh <path_to_system_data> <path_to_fit_folder> <experiment_name>

# If you choose to run the pulse deconvolution script, you can use the following command:
# e.g. ./neural_backproject.sh ../../../../SH-SAS-data/simulated/xyz_dragon/system_data.pik ./deconvolved_measurements simulated_xyz_dragon

# If you want to use the provided data, you can use the following command:
# e.g. ./neural_backproject.sh ../../../../SH-SAS-data/simulated/xyz_dragon/system_data.pik ../../../../SH-SAS-data/simulated/xyz_dragon/deconvolved_measurements simulated_xyz_dragon
```

Outputs a folder `nbp_output` with the following structure:
```
nbp_output/
├── simulated_xyz_dragon
│   ├── models
│   │   ├── 000000.tar
│   │   ├── 001000.tar
│   │   ├── 002000.tar
│   │   ├── ...
│   ├── images
│   │   ├── albedo_abs_00.png
│   │   ├── albedo_abs_01000.png
│   │   ├── albedo_abs_02000.png
│   │   ├── ...
│   │   ├── comp_albedo0.png
│   │   ├── comp_albedo1000.png
│   │   ├── comp_albedo2000.png
│   │   ├── ...
│   │   ├── sigma_est0.png
│   │   ├── sigma_est1000.png
│   │   ├── sigma_est2000.png
│   │   ├── ...
│   ├── numpy
│   │   ├── comp_albedo0.npy
│   │   ├── comp_albedo1000.npy
│   │   ├── comp_albedo2000.npy
│   │   ├── ...
```

### Generate Mesh
```bash
cd ./scenes/simulated/xyz_dragon
chmod +x generate_mesh.sh

# Run mesh generation script
./generate_mesh.sh <path_to_system_data> <experiment_name> <path_to_model>
# e.g. ./generate_mesh.sh ../../../../SH-SAS-data/simulated/xyz_dragon/system_data.pik simulated_xyz_dragon ./nbp_output/simulated_xyz_dragon/models/025000.tar
```

Mesh will be generated in `./nbp_output/simulated_xyz_dragon/reconstructed_scenes/mesh_smooth.obj`. Use blender to visualize the mesh.
