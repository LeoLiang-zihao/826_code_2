#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/configs/dcc_train.yaml}"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"
python -m cxr_project.train --config "$CONFIG_PATH"

