<!-- # SH-SAS: An Implicit Neural Representation for Complex Spherical-Harmonic Scattering Fields for 3D Synthetic Aperture Sonar

Codebase for the paper: [SH-SAS: An Implicit Neural Representation for Complex Spherical-Harmonic Scattering Fields for 3D Synthetic Aperture Sonar](https://arxiv.org/abs/2309.07510) by Omkar S. Vengurlekar, Adithya K. Pediredla, and Suren Jayasuriya, published in 3D Vision (3DV) 2026 (Oral). -->

# 🌊 SH-SAS: An Implicit Neural Representation for Complex Spherical-Harmonic Scattering Fields for 3D Synthetic Aperture Sonar

### 🎤 Oral Presentation at 3D Vision (3DV) 2026

**Omkar S. Vengurlekar** · **Adithya K. Pediredla** · **Suren Jayasuriya**

[[📄 Paper](https://www.arxiv.org/abs/2509.11087?context=cs.LG)] [[🌐 Project Page](https://omkarv23.github.io/SH-SAS-website/)] [[🎥 Video](https://drive.google.com/file/d/1IMgEknGVkRSvuteTG90Xd_XO35C034wi/view?usp=sharing)]

## Installation

### Create Conda environment
```bash
conda create -n sh_sas python=3.9
conda activate sh_sas
```

### gcc and gxx
```bash
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11
```

### Cuda-toolkit
```bash
conda install -y -c nvidia cuda-toolkit=11.8
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

# Data
<!-- We are providing one example dataset for testing purposes. You can find it in the `SH-SAS-data` directory.
There are two folders: `airsas` and `simulated`. The `airsas` folder contains real AirSAS data, while the `simulated` folder contains synthetic data. Both contains a `system_data.pik` file and precomputed pulse deconvolved measurements in the `deconvolved_measurements` folder. -->

## AirSAS data

### Download the data:
```bash
gdown --folder "https://drive.google.com/drive/folders/1KpOeGXd5d_2vNXmyjipPQCOYB1fNScuM" -O SH-SAS-data/airsas
```
Files are organized as follows:
```
SH-SAS-data/
└── airsas
    ├── system_data_arma_20k.pik
    ├── system_data_bunny_20k.pik
```
Copy to designated folders in scenes:
```bash
cp SH-SAS-data/airsas/system_data_arma_20k.pik scenes/airsas/arma/system_data.pik
cp SH-SAS-data/airsas/system_data_bunny_20k.pik scenes/airsas/bunny/system_data.pik
```

**(Showing example for AirSAS Arma scene, similar steps can be followed for Bunny scene)**


### Run Pulse Deconvolution

```bash
cd ./scenes/airsas/arma
chmod +x pulse_deconvolve.sh

# Run pulse deconvolution script
./pulse_deconvolve.sh ../../../../scenes/airsas/arma/system_data.pik ./deconvolved_measurements
```
        
### Run SH-SAS Neural Backprojection

```bash
cd ./scenes/airsas/arma
chmod +x neural_backproject.sh

# Run neural backprojection script
./neural_backproject.sh ../../../../scenes/airsas/arma/system_data.pik ./deconvolved_measurements airsas_arma
```


Outputs a folder `nbp_output` with the following structure:
```
nbp_output/
├── airsas_arma
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
cd ./scenes/airsas/arma
chmod +x generate_mesh.sh

# Run mesh generation script
# Format ---> ./generate_mesh.sh <path_to_system_data> <experiment_name> <path_to_model>
./generate_mesh.sh ../../../../SH-SAS-data/airsas/arma/system_data.pik airsas_arma ./nbp_output/airsas_arma/models/025000.tar
```

Mesh will be generated in `./nbp_output/airsas_arma/reconstructed_scenes/mesh_smooth.obj`. Use blender to visualize the mesh.

## Sediment Volume Search Sonar (SVSS) Reconstructions

### Download the data
In accordance with funding agency guidelines: request access to the SVSS dataset by first contacting Omkar S. Vengurlekar at ovengurl@asu.edu or Albert W. Reed at awreed@asu.edu .

We will walk through an example of reconstructing the cylindrical target. The other SVSS scenes will follow the same steps. Create a system_data.pik file for the svss cylinder scene:

```bash
cd ./scenes/svss/cylinder
chmod +x create_system_data.sh
./create_system_data.sh <path/to/downloaded_data> ./system_data
```

This will output a system_data.pik file as well as backprojected imagery to a new directory named ./system_data.

### Run Pulse Deconvolution

```bash
cd ./scenes/svss/cylinder
chmod +x pulse_deconvolve.sh

# Run pulse deconvolution script
./pulse_deconvolve.sh ./system_data/system_data.pik ./deconvolved_measurements
```

### Run SH-SAS Neural Backprojection

```bash
cd ./scenes/svss/cylinder
chmod +x neural_backproject.sh

# Run neural backprojection script
./neural_backproject.sh ./system_data/system_data.pik ./deconvolved_measurements svss_cylinder
``` 

Outputs a folder `nbp_output` with the following structure:
```nbp_output/
├── svss_cylinder
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
cd ./scenes/svss/cylinder
chmod +x generate_mesh.sh

# Run mesh generation script
# Format ---> ./generate_mesh.sh <path_to_system_data> <experiment_name> <path_to_model>
./generate_mesh.sh ../../../../scenes/svss/cylinder/system_data.pik svss_cylinder ./nbp_output/svss_cylinder/models/005000.tar
```

## Simulated data

Download the folder containing the transient measurements and system_data.pik file. Note this command will download approx. 100 gb of data. It may be preferrable to download a single scene by directly opening the google drive link below and downloading data for a single scene.

```
gdown https://drive.google.com/drive/folders/1bWUpcjJhro5m035W98DHDBYv13PGidRF -O ./simulated_data --folder
```

https://drive.google.com/drive/folders/1bWUpcjJhro5m035W98DHDBYv13PGidRF?usp=share_link

The folder contains data for 6 simulated scenes: bunny, xyz_dragon, lucy, dragon, buddha, and aramdillo, and the
`system_data.pik` file which defines the scene geometry. Each scene folder contains transients from the 
scene using rendered with our [ToF renderer](https://github.com/juhyeonkim95/MitsubaPyOptiXTransient). We simulate sonar
measurements by convolving transients with the sonar transmit waveform. Finally, the folder contains a `gt_meshes` folder used for evaluation. 

We provide an example for reconstructing the xyz_dragon object in `scenes/simulated/xyz_dragon`. 

### Simulate the sonar waveforms from the transients:
```bash
cd ./scenes/simulated/xyz_dragon
chmod +x simulate_waveforms.sh

# Run waveform simulation script
./simulate_waveforms <path-to-downloaded-simulated-data>/system_data.pik <path-to-downloaded-simulated-data>/xyz_dragon/data_full.npy
```

### Run Pulse Deconvolution

```bash
chmod +x pulse_deconvolve.sh

# Run pulse deconvolution script
./pulse_deconvolve.sh system_data_20db.pik ./deconvolved_measurements
```

### Run SH-SAS Neural Backprojection

```bash
chmod +x neural_backproject.sh

# Run neural backprojection script
./neural_backproject.sh system_data_20db.pik ./deconvolved_measurements simulated_xyz_dragon
``` 

Outputs a folder `nbp_output` with the following structure:
```nbp_output/
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
chmod +x generate_mesh.sh

# Run mesh generation script
# Format ---> ./generate_mesh.sh <path_to_system_data> <experiment_name> <path_to_model>
./generate_mesh.sh ../../../../simulated_data/system_data.pik simulated_xyz_dragon ./nbp_output/simulated_xyz_dragon/models/050000.tar
```