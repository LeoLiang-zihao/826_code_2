#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/dcc_train.yaml}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"
python -m cxr_project.train --config "$CONFIG_PATH"

