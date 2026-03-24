#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
BOOTSTRAP="${BOOTSTRAP:-0}"
ENABLE_BOOTSTRAP="${ENABLE_BOOTSTRAP:-0}"
ATTR_SUPERVISED_CONFIG="${ATTR_SUPERVISED_CONFIG:-$PROJECT_DIR/configs/dcc_train_full.yaml}"
ATTR_CHECKPOINT_PATH="${ATTR_CHECKPOINT_PATH:-$PROJECT_DIR/outputs/dcc_train_full/checkpoints/best.ckpt}"
SIMCLR_CONFIG_PATH="${SIMCLR_CONFIG_PATH:-$PROJECT_DIR/configs/dcc_simclr.yaml}"
LINEAR_EVAL_CONFIG_PATH="${LINEAR_EVAL_CONFIG_PATH:-$PROJECT_DIR/configs/dcc_linear_eval.yaml}"
SIMCLR_CHECKPOINT_PATH="${SIMCLR_CHECKPOINT_PATH:-$PROJECT_DIR/outputs/dcc_simclr/checkpoints/simclr-best.ckpt}"
EMBEDDING_CHECKPOINT_PATH="${EMBEDDING_CHECKPOINT_PATH:-$PROJECT_DIR/outputs/dcc_simclr/checkpoints/last.ckpt}"

SUPERVISED_CONFIGS=(
  "configs/dcc_train_head_only.yaml"
  "configs/dcc_train_last1.yaml"
  "configs/dcc_train_last2.yaml"
  "configs/dcc_train_full.yaml"
)

cd "$PROJECT_DIR"

declare -A supervised_jobs

echo "Submitting supervised ablation jobs..."
for rel_config in "${SUPERVISED_CONFIGS[@]}"; do
  job_id="$(sbatch --parsable \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG_PATH="$PROJECT_DIR/$rel_config",BOOTSTRAP="$BOOTSTRAP" \
    jobs/dcc_train.sbatch)"
  supervised_jobs["$rel_config"]="$job_id"
  echo "  $rel_config -> $job_id"
done

full_job_id="${supervised_jobs[configs/dcc_train_full.yaml]}"

echo "Submitting attribution job after full fine-tuning..."
attr_job_id="$(sbatch --parsable \
  --dependency=afterok:"$full_job_id" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG_PATH="$ATTR_SUPERVISED_CONFIG",CHECKPOINT_PATH="$ATTR_CHECKPOINT_PATH" \
  jobs/dcc_attribution.sbatch)"
echo "  attribution -> $attr_job_id (after $full_job_id)"

if [[ "$ENABLE_BOOTSTRAP" == "1" ]]; then
  echo "Submitting bootstrap reruns for head_only and full..."
  bootstrap_head_id="$(sbatch --parsable \
    --dependency=afterok:"${supervised_jobs[configs/dcc_train_head_only.yaml]}" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG_PATH="$PROJECT_DIR/configs/dcc_train_head_only.yaml",BOOTSTRAP=1000 \
    jobs/dcc_train.sbatch)"
  echo "  bootstrap head_only -> $bootstrap_head_id"

  bootstrap_full_id="$(sbatch --parsable \
    --dependency=afterok:"$full_job_id" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG_PATH="$PROJECT_DIR/configs/dcc_train_full.yaml",BOOTSTRAP=1000 \
    jobs/dcc_train.sbatch)"
  echo "  bootstrap full -> $bootstrap_full_id"
fi

echo "Submitting SimCLR pretraining..."
simclr_job_id="$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG_PATH="$SIMCLR_CONFIG_PATH" \
  jobs/dcc_pretrain_simclr.sbatch)"
echo "  simclr -> $simclr_job_id"

echo "Submitting linear eval after SimCLR..."
linear_eval_job_id="$(sbatch --parsable \
  --dependency=afterok:"$simclr_job_id" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG_PATH="$LINEAR_EVAL_CONFIG_PATH",SIMCLR_CHECKPOINT="$SIMCLR_CHECKPOINT_PATH" \
  jobs/dcc_linear_eval.sbatch)"
echo "  linear eval -> $linear_eval_job_id (after $simclr_job_id)"

echo "Submitting embedding export after SimCLR..."
embeddings_job_id="$(sbatch --parsable \
  --dependency=afterok:"$simclr_job_id" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",CONFIG_PATH="$SIMCLR_CONFIG_PATH",CHECKPOINT_PATH="$EMBEDDING_CHECKPOINT_PATH",CHECKPOINT_TYPE=simclr \
  jobs/dcc_embeddings.sbatch)"
echo "  embeddings -> $embeddings_job_id (after $simclr_job_id)"
