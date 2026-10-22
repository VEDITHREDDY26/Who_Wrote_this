#!/bin/bash
# Train one model on UCF Newton. Usage: sbatch scripts/newton/train.slurm <model_name>
# TODO: fill in partition/modules/env activation and the train.py call.
#SBATCH --job-name=turing-detectives
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# export TD_DATA_DIR=...  TD_CHECKPOINT_DIR=...  TD_RESULTS_DIR=...
