#!/usr/bin/env bash
#SBATCH --job-name=cxr-simclr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=courses-gpu
#SBATCH --account=coursess26
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:p100:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/dcc_simclr.yaml}"

mkdir -p "$PROJECT_DIR/logs"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
python -m cxr_project.pretrain_simclr --config "$CONFIG_PATH"

