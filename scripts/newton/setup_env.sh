#!/bin/bash
# One-time setup on a Newton login node, from the project folder:
#   bash scripts/newton/setup_env.sh
# Creates the Python env, logs in to Hugging Face and downloads the rewriting model.
# Safe to rerun: an existing env is reused and the download resumes where it stopped.
set -eo pipefail  # no -u: conda's activate scripts read unset variables
source scripts/newton/env.sh
MODEL=${TD_PARAPHRASE_MODEL:-meta-llama/Llama-2-70b-chat-hf}

if [ ! -d "$TD_ENV_DIR" ]; then
    conda create -y -p "$TD_ENV_DIR" python=3.11
fi
conda activate "$TD_ENV_DIR"
pip install --upgrade pip
pip install -r requirements.txt

if ! hf auth whoami >/dev/null 2>&1; then
    hf auth login  # paste the Read token; Meta must have approved your Llama 2 request
fi
mkdir -p "$HF_HOME" logs
hf download "$MODEL" --include "*.json" "*.safetensors" "tokenizer.model"
echo "Setup done: env $TD_ENV_DIR, weights in $HF_HOME"
