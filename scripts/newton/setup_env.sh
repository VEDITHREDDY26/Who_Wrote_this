#!/bin/bash
# One-time setup on a Newton login node, from the project folder:
#   bash scripts/newton/setup_env.sh
# Creates the Python env, logs in to Hugging Face and downloads the rewriting model.
# Safe to rerun: an existing env is reused and the download resumes where it stopped.
set -eo pipefail  # no -u: conda's activate scripts read unset variables
source scripts/newton/env.sh
MODEL=${TD_PARAPHRASE_MODEL:-meta-llama/Llama-2-70b-chat-hf}

if [ ! -x "$TD_ENV_DIR/bin/python" ]; then  # conda is only needed to create the env
    module load "$TD_CONDA_MODULE"
    conda create -y -p "$TD_ENV_DIR" python=3.11
    source scripts/newton/env.sh  # puts the new env first on PATH
fi
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if ! hf auth whoami >/dev/null 2>&1; then
    hf auth login  # paste the Read token; Meta must have approved your Llama 2 request
fi
mkdir -p "$HF_HOME" logs
# One --include per pattern: extra positional values would be read as file names instead.
hf download "$MODEL" --include "*.json" --include "*.safetensors" --include "tokenizer.model"
echo "Setup done: env $TD_ENV_DIR, weights in $HF_HOME"
