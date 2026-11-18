#!/bin/bash
# Shared environment for the Newton scripts. Source it from the project folder:
#   source scripts/newton/env.sh
# Override any of these by exporting them before sourcing.
export TD_CONDA_MODULE=${TD_CONDA_MODULE:-anaconda/anaconda-2024.10}
export TD_ENV_DIR=${TD_ENV_DIR:-$HOME/envs/turing}      # Python 3.11 env with requirements.txt
export HF_HOME=${HF_HOME:-$HOME/hf_cache}               # model weights (~130 GB for Llama 2 70B)

module load "$TD_CONDA_MODULE"
eval "$(conda shell.bash hook)"
if [ -d "$TD_ENV_DIR" ]; then
    conda activate "$TD_ENV_DIR"
fi
