$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv\Scripts\Activate.ps1)) {
  throw "Virtual environment not found. Run .\\scripts\\bootstrap_env.ps1 first."
}

. .\.venv\Scripts\Activate.ps1
python .\scripts\prepare_synthetic_data.py --output-dir data/synthetic --num-subjects 36 --seed 826
python -m cxr_project.train --config .\configs\local_synthetic.yaml
python -m cxr_project.pretrain_simclr --config .\configs\local_simclr_synthetic.yaml
python -m pytest tests -q
