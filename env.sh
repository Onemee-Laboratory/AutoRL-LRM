#!/bin/bash
# set path and env

export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6
export AUTORL_HOME=/home/oz/.autorl
export AUTORL_DATA=$AUTORL_HOME/data
export AUTORL_CHECKPOINTS=$AUTORL_HOME/checkpoints
#export VIRTUAL_ENV=$AUTORL_HOME/venv/.venv
export VIRTUAL_ENV=/home/oz/.autorl/venv/.venv
export PATH=$VIRTUAL_ENV/bin:$PATH
#source $VIRTUAL_ENV/bin/activate
