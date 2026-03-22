#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/dcc_simclr.yaml}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-$PROJECT_DIR/configs/dcc_train.yaml}"
SIMCLR_OUTPUT_DIR="${SIMCLR_OUTPUT_DIR:-$PROJECT_DIR/outputs/dcc_simclr}"
SIMCLR_CHECKPOINT="${SIMCLR_CHECKPOINT:-$SIMCLR_OUTPUT_DIR/checkpoints/last.ckpt}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"
python -m cxr_project.pretrain_simclr --config "$CONFIG_PATH"
python -m cxr_project.linear_eval --config "$TRAIN_CONFIG_PATH" --simclr-checkpoint "$SIMCLR_CHECKPOINT"
python -m cxr_project.visualize_embeddings --config "$TRAIN_CONFIG_PATH" --checkpoint "$SIMCLR_CHECKPOINT" --checkpoint-type simclr --max-samples 200
