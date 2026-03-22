$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv\Scripts\Activate.ps1)) {
  throw "Virtual environment not found. Run .\\scripts\\bootstrap_env.ps1 first."
}

. .\.venv\Scripts\Activate.ps1
python .\scripts\prepare_synthetic_data.py --output-dir data/synthetic --num-subjects 36 --seed 826
python -m cxr_project.train --config .\configs\local_synthetic.yaml
python -m cxr_project.pretrain_simclr --config .\configs\local_simclr_synthetic.yaml
python -m cxr_project.linear_eval --config .\configs\local_linear_eval_synthetic.yaml --simclr-checkpoint .\outputs\local_simclr_synthetic\checkpoints\last.ckpt --bootstrap 25
python -m cxr_project.extract_attributions --config .\configs\local_synthetic.yaml --checkpoint .\outputs\local_synthetic\checkpoints\best.ckpt --num-positive 2 --num-negative 2
python -m cxr_project.visualize_embeddings --config .\configs\local_simclr_synthetic.yaml --checkpoint .\outputs\local_simclr_synthetic\checkpoints\last.ckpt --checkpoint-type simclr --max-samples 12
python -m pytest tests -q
