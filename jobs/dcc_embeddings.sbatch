#!/usr/bin/env bash
#SBATCH --job-name=cxr-embeddings
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=courses-gpu
#SBATCH --account=coursess26
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:p100:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/dcc_simclr.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH before submitting the job.}"
CHECKPOINT_TYPE="${CHECKPOINT_TYPE:-simclr}"

mkdir -p "$PROJECT_DIR/logs"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
python -m cxr_project.visualize_embeddings --config "$CONFIG_PATH" --checkpoint "$CHECKPOINT_PATH" --checkpoint-type "$CHECKPOINT_TYPE"
