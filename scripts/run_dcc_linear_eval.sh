#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <simclr-checkpoint>" >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/dcc_linear_eval.yaml}"
SIMCLR_CHECKPOINT="$1"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
python -m cxr_project.linear_eval --config "$CONFIG_PATH" --simclr-checkpoint "$SIMCLR_CHECKPOINT"
