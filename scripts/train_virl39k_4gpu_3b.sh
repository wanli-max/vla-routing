#!/bin/bash
# Full training: 4x GPU, Qwen2.5-VL-3B-Instruct, ViRL39K+MMK12.
# Data must be pre-adapted and pre-filtered (offline).
#
# Usage:
#   cd /path/to/EasyR1
#   bash scripts/train_virl39k_4gpu_3b.sh [EXPERIMENT_NAME] [MODEL_PATH] [DATASET_ROOT]

set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]] || [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
      awk -F', ' -v uuids="$CUDA_VISIBLE_DEVICES" \
      'BEGIN{split(uuids,u,",")} {for(i in u) if($2==u[i]) printf "%s%s",(n++?",":""),$1}')
    echo "[INFO] Converted CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"
fi

BASE_EXPERIMENT_NAME=${1:-"virl39k_4gpu_3b"}
MODEL_PATH=${2:-"Qwen/Qwen2.5-VL-3B-Instruct"}
DATASET_ROOT=${3:-"/projects_vol/gp_boan/easyr1_assets/datasets/virl39k_mmk12_easyr1"}
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${RUN_TIMESTAMP}"

TRAIN_FILES="${DATASET_ROOT}/train"
VAL_FILES="${DATASET_ROOT}/val"

echo "============================================"
echo "  ViRL39K Full Training - 4x GPU (3B)"
echo "============================================"
echo "  Experiment : ${EXPERIMENT_NAME}"
echo "  Model      : ${MODEL_PATH}"
echo "  Data       : ${DATASET_ROOT}"
echo "============================================"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.format_prompt=null \
    data.filter_overlong_prompts=false \
    data.max_prompt_length=8192 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.rollout.gpu_memory_utilization=0.4 \
    data.rollout_batch_size=64 \
    data.val_batch_size=64 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size=64 \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.logger='["file","tensorboard"]' \
    2>&1 | tee "${EXPERIMENT_NAME}.log"

echo "[FINISH] Training completed: ${EXPERIMENT_NAME}"
