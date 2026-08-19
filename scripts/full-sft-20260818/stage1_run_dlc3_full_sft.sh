#!/usr/bin/env bash
set -u

: "${WORLD_SIZE:?DLC must provide WORLD_SIZE (node count)}"
: "${RANK:?DLC must provide RANK (node rank)}"
: "${MASTER_ADDR:?DLC must provide MASTER_ADDR}"

STAGE_ROOT=${STAGE_ROOT:-/mnt/luojunkun/stage1}
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-${STAGE_ROOT}/ms-swift}
MODEL=${MODEL:-${STAGE_ROOT}/model}
MANIFEST=${MANIFEST:-${MS_SWIFT_ROOT}/scripts/dlc_ready_entrypoints.stage1.json}
OUTPUT_DIR=${OUTPUT_DIR:-${STAGE_ROOT}/sft-model/qwen3-vl-instruct4b-20260813}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:?Set PER_DEVICE_TRAIN_BATCH_SIZE from benchmark}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:?Set GRADIENT_ACCUMULATION_STEPS from benchmark}
SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE:?Set SEQUENCE_PARALLEL_SIZE from benchmark}
CELOSS_PARALLEL_SIZE=${CELOSS_PARALLEL_SIZE:-0}

if [[ "${WORLD_SIZE}" != "3" ]]; then
  echo "Expected 3 DLC nodes, got WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi

export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NPROC_PER_NODE=16
export NNODES="${WORLD_SIZE}"
export NODE_RANK="${RANK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_MAX_TOKEN_NUM=1024
export CELOSS_PARALLEL_SIZE
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_SHOW_CPP_STACKTRACES=1

# Keep the Qwen3-VL default video sampling and spatial budget.
unset VIDEO_MAX_TOKEN_NUM VIDEO_MIN_TOKEN_NUM VIDEO_MAX_PIXELS VIDEO_MIN_PIXELS
unset VIDEO_TOTAL_PIXELS FPS FPS_MAX_FRAMES FPS_MIN_FRAMES

NCCL_LOG_DIR=${STAGE_ROOT}/logs/nccl
ELASTIC_LOG_DIR=${STAGE_ROOT}/logs/elastic
mkdir -p "${NCCL_LOG_DIR}" "${ELASTIC_LOG_DIR}" "${OUTPUT_DIR}"
export NCCL_DEBUG_FILE="${NCCL_LOG_DIR}/nccl-%h-%p.log"
export TORCHELASTIC_ERROR_FILE="${ELASTIC_LOG_DIR}/error_${HOSTNAME}_${RANK}.json"

cd "${MS_SWIFT_ROOT}" || exit 2
python -c 'import swift; print("swift package:", swift.__file__)' || exit 2
python -c 'import torch; n=torch.cuda.device_count(); print("visible GPU count:", n); assert n == 16' || exit 2
python -c 'import flash_attn; print("flash_attn:", flash_attn.__version__)' || exit 2

DATASET_LIST=$(python3 - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for config in manifest["datasets"].values():
    if config.get("enabled", True):
        print(config["train"])
PY
)

echo "SP=${SEQUENCE_PARALLEL_SIZE} BS=${PER_DEVICE_TRAIN_BATCH_SIZE} GA=${GRADIENT_ACCUMULATION_STEPS}"
echo "CELOSS_PARALLEL_SIZE=${CELOSS_PARALLEL_SIZE}"
echo "Video overrides:"
env | grep -E '^(VIDEO_|FPS)' || true

set +e
swift sft \
  --model "${MODEL}" \
  --model_type qwen3_vl \
  --tuner_type full \
  --freeze_llm false \
  --freeze_vit false \
  --freeze_aligner false \
  --deepspeed zero2 \
  --dataset ${DATASET_LIST} \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --torch_dtype bfloat16 \
  --attn_impl flash_attention_2 \
  --num_train_epochs 3 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing true \
  --padding_free true \
  --packing false \
  --sequence_parallel_size "${SEQUENCE_PARALLEL_SIZE}" \
  --save_steps 900 \
  --save_total_limit 50 \
  --logging_steps 5 \
  --max_length 50000 \
  --truncation_strategy right \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --dataloader_persistent_workers true \
  --dataloader_prefetch_factor 2 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to wandb
status=$?

echo "swift exit status: ${status}"
find "${ELASTIC_LOG_DIR}" -maxdepth 1 -type f -name 'error_*.json' -print -exec cat {} \; || true
exit "${status}"
