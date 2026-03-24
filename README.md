# CXR Project Bootstrap

This repository is set up to make the Assignment 2 workflow reproducible in stages:

1. Run a full local smoke test on synthetic chest X-ray style images.
2. Swap the manifest to a real MIMIC-CXR-JPG subset on DCC without changing the training loop.
3. Reuse the same codebase for Grad-CAM attribution, SimCLR pretraining, frozen linear evaluation, and t-SNE embeddings.

## Local quick start

Use Python 3.11 for this project. The default `python` on this machine is 3.14, which is not a safe target for `torch` wheels.

```powershell
.\scripts\bootstrap_env.ps1
.\scripts\run_local_smoke.ps1
.\scripts\run_local_part4_smoke.ps1
```

The local scripts cover two levels:

- `run_local_smoke.ps1`: supervised train/valid/test plus pytest
- `run_local_part4_smoke.ps1`: SimCLR pretraining, frozen linear eval, embedding export, and Grad-CAM export

## Main Python entrypoints

```powershell
python -m cxr_project.train --config .\configs\local_synthetic.yaml
python -m cxr_project.evaluate --config .\configs\local_synthetic.yaml --checkpoint .\outputs\local_synthetic\checkpoints\best.ckpt
python -m cxr_project.extract_attributions --config .\configs\local_synthetic.yaml --checkpoint .\outputs\local_synthetic\checkpoints\best.ckpt
python -m cxr_project.pretrain_simclr --config .\configs\local_simclr_synthetic.yaml
python -m cxr_project.linear_eval --config .\configs\local_linear_eval_synthetic.yaml --simclr-checkpoint .\outputs\local_simclr_synthetic\checkpoints\last.ckpt
python -m cxr_project.visualize_embeddings --config .\configs\local_simclr_synthetic.yaml --checkpoint .\outputs\local_simclr_synthetic\checkpoints\last.ckpt --checkpoint-type simclr
```

## Real MIMIC subset preparation

```bash
python scripts/prepare_mimic_subset.py \
  --labels-path /path/to/mimic-cxr-2.0.0-chexpert.csv.gz \
  --metadata-path /path/to/mimic-cxr-2.0.0-metadata.csv.gz \
  --image-root /path/to/mimic-cxr-jpg/2.1.0 \
  --output-manifest /path/to/mimic_subset_manifest.csv \
  --pathology "Pleural Effusion"
```

## DCC environment setup

For DCC, mirror the local environment with:

```bash
bash scripts/bootstrap_env_dcc.sh
```

Useful environment overrides:

- `PYTHON_MODULE`: optional module name to `module load` before creating the venv
- `PYTHON_BIN`: exact Python binary to use, default `python3.11`
- `VENV_DIR`: virtualenv path, default `./.venv`

## DCC job flow

```bash
sbatch jobs/dcc_train.sbatch
sbatch jobs/dcc_pretrain_simclr.sbatch
sbatch jobs/dcc_linear_eval.sbatch
sbatch jobs/dcc_attribute.sbatch
sbatch jobs/dcc_embeddings.sbatch
```

All DCC jobs use the same Python entrypoints as local runs. Before submission, update the manifest path in `configs/dcc_train.yaml` or `configs/dcc_simclr.yaml`, and set required environment variables such as `SIMCLR_CHECKPOINT` or `CHECKPOINT_PATH` where needed.

For the Part 2 supervised ablation required by the assignment, submit all four fine-tuning depths with:

```bash
bash scripts/submit_dcc_supervised_ablation.sh
```

This submits `head_only`, `last1`, `last2`, and `full` as separate jobs with separate output directories under `outputs/`.

To submit the main DCC pipeline with job dependencies so downstream experiments run automatically, use:

```bash
bash scripts/submit_dcc_pipeline.sh
```

This submits:

- the four supervised ablations
- Grad-CAM attribution after the `full` supervised run completes
- SimCLR pretraining
- frozen linear evaluation after SimCLR completes
- embedding export after SimCLR completes

To also schedule the expensive bootstrap reruns for `head_only` and `full`, use:

```bash
ENABLE_BOOTSTRAP=1 bash scripts/submit_dcc_pipeline.sh
```
