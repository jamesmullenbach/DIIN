#!/bin/bash
source /usr/usc/cuda/8.0/setup.sh
source /usr/usc/cuDNN/v6.0-cuda8.0/setup.sh
PYTHONHASHSEED=0 python train_mnli.py DIIN $1 --train_pw_only --alpha 0 
