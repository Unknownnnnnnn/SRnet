#!/bin/zsh

############## make sure you can write in \tmp; Or you should set TORCH_EXTENSIONS_DIR
# e.g. export TORCH_EXTENSIONS_DIR=/mnt/lustre/$YourUserName$/tmp

conda create -n mvp python=3.7 -y
conda activate mvp
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# setup completion
cd completion
pip install -r requirements.txt


cd ../utils/mm3d_pn2/
sh setup.sh

############## make sure NVCC is in your environment
# SLURM users
# sh run_build.sh
# or 
pip install -v -e . 
# python setup.py develop

cd ../../

# setup registration


