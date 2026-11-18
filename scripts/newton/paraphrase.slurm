#!/bin/bash
# Llama rewrites for the paraphrase experiment (python -m src.experiments.paraphrase generate).
# Usage, from the project folder after scripts/newton/setup_env.sh:
#   sbatch scripts/newton/paraphrase.slurm             # all passages
#   sbatch scripts/newton/paraphrase.slurm --limit 32  # quick check first
# If the job stops for any reason, submit it again: finished batches are kept.
#SBATCH --job-name=td-paraphrase
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -eo pipefail
source scripts/newton/env.sh
export HF_HUB_OFFLINE=1  # weights were downloaded by setup_env.sh; compute nodes may be offline

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; n = torch.cuda.device_count(); assert n > 0, 'no GPU visible to torch'; \
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, {n} GPUs')"
python -m src.experiments.paraphrase generate "$@"
