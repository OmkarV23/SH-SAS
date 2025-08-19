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