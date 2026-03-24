#!/usr/bin/env bash
#SBATCH --job-name=cxr-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=courses-gpu
#SBATCH --account=coursess26
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1

set -euo pipefail

# Change `courses-gpu` to `gpu-common` if you are running outside the course partition.
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/dcc_train.yaml}"
BOOTSTRAP="${BOOTSTRAP:-0}"

mkdir -p "$PROJECT_DIR/logs"
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
python -m cxr_project.train --config "$CONFIG_PATH" --bootstrap "$BOOTSTRAP"
