#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found. Run ./scripts/bootstrap_env.sh first." >&2
  exit 1
fi

source .venv/bin/activate
python ./scripts/prepare_synthetic_data.py --output-dir data/synthetic --num-subjects 36 --seed 826
python -m cxr_project.train --config ./configs/local_synthetic.yaml
python -m cxr_project.pretrain_simclr --config ./configs/local_simclr.yaml
python -m pytest tests -q

