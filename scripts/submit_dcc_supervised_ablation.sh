#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
BOOTSTRAP="${BOOTSTRAP:-0}"

CONFIGS=(
  "configs/dcc_train_head_only.yaml"
  "configs/dcc_train_last1.yaml"
  "configs/dcc_train_last2.yaml"
  "configs/dcc_train_full.yaml"
)

cd "$PROJECT_DIR"

for config_path in "${CONFIGS[@]}"; do
  echo "Submitting $config_path"
  sbatch --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG_PATH="$PROJECT_DIR/$config_path",BOOTSTRAP="$BOOTSTRAP" jobs/dcc_train.sbatch
done
