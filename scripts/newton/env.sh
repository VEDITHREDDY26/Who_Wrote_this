#!/bin/bash
# Shared environment for the Newton scripts. Source it from the project folder:
#   source scripts/newton/env.sh
# Override any of these by exporting them before sourcing.
export TD_CONDA_MODULE=${TD_CONDA_MODULE:-anaconda/anaconda-2024.10}
export TD_ENV_DIR=${TD_ENV_DIR:-$HOME/envs/turing}      # Python 3.11 env with requirements.txt
export HF_HOME=${HF_HOME:-$HOME/hf_cache}               # model weights (~130 GB for Llama 2 70B)

# Use the env's own python directly instead of `conda activate`: batch jobs inherit the
# submitting shell's conda state, and the anaconda module then fails on `conda deactivate`.
if [ -x "$TD_ENV_DIR/bin/python" ]; then
    export PATH="$TD_ENV_DIR/bin:$PATH"
fi
