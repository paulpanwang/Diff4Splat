#!/bin/bash

PROJ_DIR=$(pwd)

# Pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -U xformers==0.0.29.post1
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# Others
pip install -U gpustat
pip install -U -r settings/requirements.txt
pip install --upgrade imageio imageio[ffmpeg]
sudo apt-get install -y ffmpeg tmux

# GSplat
MAX_JOBS=128 pip3 install git+https://github.com/nerfstudio-project/gsplat.git@v1.5.0
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
