#!/bin/bash
# Smoke test: 4x GPU, Qwen2.5-VL-3B-Instruct, ViRL39K+MMK12, 3 steps.
# Data must be pre-adapted and pre-filtered (offline).
#
# Usage:
#   cd /path/to/EasyR1
#   bash scripts/train_virl39k_smoke_4gpu_3b.sh [EXPERIMENT_NAME] [MODEL_PATH] [DATASET_ROOT]

set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]] || [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
      awk -F', ' -v uuids="$CUDA_VISIBLE_DEVICES" \
      'BEGIN{split(uuids,u,",")} {for(i in u) if($2==u[i]) printf "%s%s",(n++?",":""),$1}')
    echo "[INFO] Converted CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"
fi

BASE_EXPERIMENT_NAME=${1:-"virl39k_smoke_4gpu_3b"}
MODEL_PATH=${2:-"Qwen/Qwen2.5-VL-3B-Instruct"}
DATASET_ROOT=${3:-"/projects_vol/gp_boan/easyr1_assets/datasets/virl39k_mmk12_easyr1"}
SMOKE_ROOT="/projects_vol/gp_boan/easyr1_assets/datasets/virl39k_mmk12_easyr1_smoke"
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${RUN_TIMESTAMP}"
TENSORBOARD_RUN_DIR="tensorboard_log/easy_r1/${EXPERIMENT_NAME}"

TRAIN_FILES="${SMOKE_ROOT}/train"
VAL_FILES="${SMOKE_ROOT}/val"

echo "============================================"
echo "  ViRL39K Smoke Test - 4x GPU (3B)"
echo "============================================"
echo "  Experiment : ${EXPERIMENT_NAME}"
echo "  Base name  : ${BASE_EXPERIMENT_NAME}"
echo "  Model      : ${MODEL_PATH}"
echo "  Data       : ${DATASET_ROOT}"
echo "  TB run dir : ${TENSORBOARD_RUN_DIR}"
echo "============================================"

# Build smoke subset (8 train, 4 val)
export DATASET_ROOT SMOKE_ROOT
python3 - <<'PY'
import glob, os
from pathlib import Path
from datasets import load_dataset

src  = Path(os.environ["DATASET_ROOT"])
dst  = Path(os.environ["SMOKE_ROOT"])

train_files = sorted(glob.glob(str(src / "train" / "*.parquet")))
val_files   = sorted(glob.glob(str(src / "val"   / "*.parquet")))
if not train_files: raise FileNotFoundError(f"No train parquet under {src}/train")
if not val_files:   raise FileNotFoundError(f"No val parquet under {src}/val")

train_ds = load_dataset("parquet", data_files=train_files, split="train").select(range(8))
val_ds   = load_dataset("parquet", data_files=val_files,   split="train").select(range(4))

(dst / "train").mkdir(parents=True, exist_ok=True)
(dst / "val").mkdir(parents=True, exist_ok=True)
train_ds.to_parquet(str(dst / "train" / "part-00000.parquet"))
val_ds.to_parquet(str(dst / "val"   / "part-00000.parquet"))
print(f"Smoke subset ready: {len(train_ds)} train, {len(val_ds)} val")
PY

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.format_prompt=null \
    data.filter_overlong_prompts=false \
    data.rollout_batch_size=4 \
    data.val_batch_size=4 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size=4 \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.logger='["file","tensorboard"]' \
    trainer.max_steps=3 \
    trainer.val_freq=3 \
    trainer.save_freq=3 \
    2>&1 | tee "${EXPERIMENT_NAME}.log"

echo "[FINISH] Smoke run completed successfully."
echo "[FINISH] Experiment name: ${EXPERIMENT_NAME}"
echo "[FINISH] TensorBoard dir: ${TENSORBOARD_RUN_DIR}"
echo "[FINISH] Log file: ${EXPERIMENT_NAME}.log"
