# CLAUDE.md

## Project Overview

BIOSTAT 826 Code Project 2: Binary classification of chest X-rays using the MIMIC-CXR-JPG dataset. The codebase is local-first (synthetic data on CPU) and DCC-ready (real MIMIC on GPU via SLURM).

**Target pathology**: Pleural Effusion (PA view, 3:1 neg/pos ratio)
**User NetID**: zl467
**DCC data path**: `/work/zl467/mimic-cxr-jpg-2.1.0`

## Architecture

```
configs/          YAML configs (local_* for dev, dcc_* for cluster)
jobs/             SLURM .sbatch scripts
scripts/          Bootstrap, data prep, smoke tests
src/cxr_project/  Main Python package (installed editable)
  data/           DataModule, Dataset, manifest builder, transforms
  models/         ResNet backbones, classifier, SimCLR, GradCAM
  utils/          Seed utility
tests/            Pytest smoke tests (synthetic data)
skills/           PhysioNet download skill
```

## Five Pipelines

| # | Pipeline | Entrypoint | DCC Job | Config |
|---|----------|-----------|---------|--------|
| 1 | Supervised Training | `python -m cxr_project.train` | `dcc_train.sbatch` | `dcc_train.yaml` |
| 2 | SimCLR Pretraining | `python -m cxr_project.pretrain_simclr` | `dcc_pretrain_simclr.sbatch` | `dcc_simclr.yaml` |
| 3 | Frozen Linear Eval | `python -m cxr_project.linear_eval` | `dcc_linear_eval.sbatch` | `dcc_linear_eval.yaml` |
| 4 | Grad-CAM Attribution | `python -m cxr_project.extract_attributions` | `dcc_attribute.sbatch` / `dcc_attribution.sbatch` | uses train/linear_eval config |
| 5 | Embedding Visualization | `python -m cxr_project.visualize_embeddings` | `dcc_embeddings.sbatch` | `dcc_simclr.yaml` |

## DCC Job Submission Quick Reference

All commands run from the project root on DCC (`/work/zl467/cxr_project` or wherever the repo is cloned).

### Prerequisites
1. Python venv ready: `bash scripts/bootstrap_env_dcc.sh`
2. MIMIC-CXR-JPG images downloaded to `/work/zl467/mimic-cxr-jpg-2.1.0`
3. Manifest built with `prepare_mimic_subset.py`
4. `logs/` directory exists (auto-created by sbatch scripts)
5. Config YAML `manifest_path` updated to actual path

### Step-by-step

```bash
# 0. Prepare manifest (one-time, on login node or interactive session)
python scripts/prepare_mimic_subset.py \
  --labels-path /work/zl467/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz \
  --metadata-path /work/zl467/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --image-root /work/zl467/mimic-cxr-jpg-2.1.0 \
  --output-manifest data/manifests/mimic_subset_manifest.csv \
  --pathology "Pleural Effusion"

# 1. Supervised training
sbatch jobs/dcc_train.sbatch

# 2. SimCLR pretraining
sbatch jobs/dcc_pretrain_simclr.sbatch

# 3. Frozen linear eval (after SimCLR completes)
export SIMCLR_CHECKPOINT=outputs/dcc_simclr/checkpoints/simclr-best.ckpt
sbatch jobs/dcc_linear_eval.sbatch

# 4a. Attribution from supervised checkpoint
export CHECKPOINT_PATH=outputs/dcc_train/checkpoints/best.ckpt
sbatch jobs/dcc_attribution.sbatch

# 4b. Attribution from linear eval checkpoint
export CHECKPOINT_PATH=outputs/dcc_linear_eval/linear_eval/checkpoints/best.ckpt
sbatch jobs/dcc_attribute.sbatch

# 5. Embedding visualization
export CHECKPOINT_PATH=outputs/dcc_simclr/checkpoints/last.ckpt
export CHECKPOINT_TYPE=simclr
sbatch jobs/dcc_embeddings.sbatch
```

### Monitoring
```bash
squeue -u zl467                    # Check job status
cat logs/cxr-train-<JOBID>.out     # View stdout
cat logs/cxr-train-<JOBID>.err     # View stderr
scancel <JOBID>                    # Cancel a job
```

## SLURM Resources Summary

| Job | Partition | GPU | Mem | Time | CPUs |
|-----|-----------|-----|-----|------|------|
| train | courses-gpu | p100:1 | 32G | 8h | 4 |
| simclr | courses-gpu | p100:1 | 48G | 12h | 4 |
| linear_eval | courses-gpu | p100:1 | 32G | 8h | 4 |
| attribution | courses-gpu | p100:1 | 16G | 2h | 2 |
| embeddings | courses-gpu | p100:1 | 16G | 2h | 2 |

## Config Paths to Update

The DCC configs ship with placeholder `manifest_path: /work/your_netid/...`. Before first run, update:
- `configs/dcc_train.yaml` line 4
- `configs/dcc_simclr.yaml` line 4
- `configs/dcc_linear_eval.yaml` line 4

Replace `/work/your_netid/cxr_project/data/manifests/mimic_subset_manifest.csv` with the actual manifest path.

## Data Download Context

User downloads MIMIC-CXR-JPG P10 subset on DCC at `/work/zl467/mimic-cxr-jpg-2.1.0` using parallel `nohup wget` workers with PhysioNet `~/.netrc` auth. The download skill is at `skills/dcc-physionet-download/SKILL.md`.

P10 has ~36,681 images. Use `check_manifest_progress.sh` or:
```bash
echo "$(find mimic-cxr-jpg/2.1.0/files/p10 -type f | wc -l) / $(grep '^files/p10/' IMAGE_FILENAMES | wc -l)"
```

## Key Decisions

- **Python 3.11** required (3.14 default on local machine lacks PyTorch wheels)
- **ResNet50** for DCC runs, **ResNet18** for local synthetic
- **Patient-level splits** (not image-level) to prevent data leakage
- Labels from CheXpert format: only use definite 0/1, exclude uncertain (-1) and NaN
- `image_path` resolution tries both `files/pXX/...` and `pXX/...` prefixes

## Running Tests

```bash
python -m pytest tests -q
```

Tests use synthetic data generated into `tmp_path` (no real MIMIC needed).

## When User Pastes DCC Terminal Output

The user (zl467) runs Claude Code from their local Windows machine but also operates on DCC via JupyterLab terminal. When they paste terminal output:
- `(base) zl467@dcc-courses-XX` = DCC login/compute node
- `/work/zl467/...` = DCC work directory
- `nohup wget` with `Exit 8` = download workers that finished (exit 8 = some files already existed, normal)
- `squeue`, `sbatch`, `scancel` = SLURM job management

Help them interpret output, suggest next steps, and reference the correct paths/configs for DCC operations.
