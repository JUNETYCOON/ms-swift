#!/usr/bin/env bash
set -u

# Alibaba Cloud DLC exports WORLD_SIZE as the node count and RANK as node rank.
: "${WORLD_SIZE:?DLC must provide WORLD_SIZE (node count)}"
: "${RANK:?DLC must provide RANK (node rank)}"
: "${MASTER_ADDR:?DLC must provide MASTER_ADDR}"

STAGE_ROOT=${STAGE_ROOT:-/mnt/luojunkun/stage1}
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-${STAGE_ROOT}/ms-swift}
MODEL_PATH=${MODEL_PATH:-${STAGE_ROOT}/model}
MANIFEST=${MANIFEST:-${MS_SWIFT_ROOT}/scripts/dlc_ready_entrypoints.stage1.json}
OUTPUT_DIR=${OUTPUT_DIR:-${STAGE_ROOT}/sft-model/qwen3-vl-stage1-dlc1x16-full}

NPROC_PER_NODE=${NPROC_PER_NODE:-16}
EXPECTED_NNODES=${EXPECTED_NNODES:-1}
EXPECTED_GPU_COUNT=${EXPECTED_GPU_COUNT:-16}
MASTER_PORT=${MASTER_PORT:-29500}

NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:--1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-65536}
SAVE_STEPS=${SAVE_STEPS:-900}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-50}
REPORT_TO=${REPORT_TO:-wandb}

# The one-node training job normally occupies all GPUs. Post-training c-IoU
# backfill is safe by default; online evaluation requires a separately reserved GPU.
GROUNDING_EVAL_DURING_TRAIN=${GROUNDING_EVAL_DURING_TRAIN:-0}
GROUNDING_EVAL_AFTER_TRAIN=${GROUNDING_EVAL_AFTER_TRAIN:-1}

if [[ "${WORLD_SIZE}" != "${EXPECTED_NNODES}" ]]; then
  echo "Expected ${EXPECTED_NNODES} DLC node, got WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi
if [[ "${RANK}" != "0" ]]; then
  echo "A single-node DLC job must use RANK=0, got RANK=${RANK}" >&2
  exit 2
fi
if (( NPROC_PER_NODE % SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} must be divisible by SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE}" >&2
  exit 2
fi

export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE
export EXPECTED_GPU_COUNT
export MASTER_ADDR
export MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
export FPS=${FPS:-25}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-128}
export QWENVL_BBOX_FORMAT=${QWENVL_BBOX_FORMAT:-legacy}
export CELOSS_PARALLEL_SIZE=${CELOSS_PARALLEL_SIZE:-2048}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export WANDB_PROJECT=${WANDB_PROJECT:-ms-swift}
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
  WANDB_RUN_ID="stage1-$(basename "${OUTPUT_DIR}" | tr -c 'A-Za-z0-9_-' '-')"
  export WANDB_RUN_ID
fi

NCCL_LOG_DIR=${STAGE_ROOT}/logs/nccl
ELASTIC_LOG_DIR=${STAGE_ROOT}/logs/elastic
mkdir -p "${NCCL_LOG_DIR}" "${ELASTIC_LOG_DIR}" "${OUTPUT_DIR}"
export NCCL_DEBUG_FILE="${NCCL_LOG_DIR}/single-%h-%p.log"
export TORCHELASTIC_ERROR_FILE="${ELASTIC_LOG_DIR}/single_error_${HOSTNAME}.json"

cd "${MS_SWIFT_ROOT}" || exit 2
command -v swift || exit 2
python3 -c 'import swift; print("swift package:", swift.__file__)' || exit 2
python3 -c 'import torch; n=torch.cuda.device_count(); print("visible GPU count:", n); assert n == int(__import__("os").environ["EXPECTED_GPU_COUNT"])' || exit 2
python3 -c 'import flash_attn; print("flash_attn:", flash_attn.__version__)' || exit 2

TRAIN_DATA_LIST=$(mktemp)
trap 'rm -f "${TRAIN_DATA_LIST}"' EXIT
python3 - "${MANIFEST}" >"${TRAIN_DATA_LIST}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if "global" in manifest_path.name.lower():
    raise SystemExit(f"global manifest is forbidden: {manifest_path}")
policy = manifest.get("global_dedup") or {}
if policy.get("deduplicate_cross_dataset_train") is not False:
    raise SystemExit("ready manifest must retain valid cross-dataset train supervision")
report_path = Path(policy.get("report", "")).resolve()
if not report_path.is_file():
    raise SystemExit(f"decontamination report is missing: {report_path}")
report = json.loads(report_path.read_text(encoding="utf-8"))
verification = report.get("verification") or {}
if report.get("status") != "complete" or verification.get("status") != "complete":
    raise SystemExit(f"decontamination report is incomplete: {report_path}")
if verification.get("train_eval_overlap_rows") != 0:
    raise SystemExit("train/eval contamination is not zero")
for name, config in manifest["datasets"].items():
    if not config.get("enabled", True):
        continue
    path = Path(config["train"]).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty train dataset: {name}: {path}")
    print(path)
PY
mapfile -t TRAIN_DATA <"${TRAIN_DATA_LIST}"
rm -f "${TRAIN_DATA_LIST}"
trap - EXIT
if [[ ${#TRAIN_DATA[@]} -eq 0 ]]; then
  echo "No enabled training datasets found in ${MANIFEST}" >&2
  exit 2
fi

echo "HOSTNAME=${HOSTNAME} WORLD_SIZE=${WORLD_SIZE} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "GPUS=${NPROC_PER_NODE} BS=${PER_DEVICE_TRAIN_BATCH_SIZE} GA=${GRADIENT_ACCUMULATION_STEPS} SP=${SEQUENCE_PARALLEL_SIZE} MAX_LENGTH=${MAX_LENGTH}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "MANIFEST=${MANIFEST} DATASET_COUNT=${#TRAIN_DATA[@]}"
echo "OUTPUT_DIR=${OUTPUT_DIR} WANDB_PROJECT=${WANDB_PROJECT} WANDB_RUN_ID=${WANDB_RUN_ID}"
echo "GROUNDING_EVAL_DURING_TRAIN=${GROUNDING_EVAL_DURING_TRAIN} GROUNDING_EVAL_AFTER_TRAIN=${GROUNDING_EVAL_AFTER_TRAIN}"

grounding_eval_pid=""
if [[ "${GROUNDING_EVAL_DURING_TRAIN}" == "1" ]]; then
  if [[ -z "${GROUNDING_EVAL_CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Refusing online c-IoU on a fully occupied node. Set GROUNDING_EVAL_CUDA_VISIBLE_DEVICES to an actually reserved GPU." >&2
    exit 2
  fi
  OUTPUT_DIR="${OUTPUT_DIR}" STAGE_ROOT="${STAGE_ROOT}" MS_SWIFT_ROOT="${MS_SWIFT_ROOT}" MANIFEST="${MANIFEST}" \
    bash "${MS_SWIFT_ROOT}/scripts/run_grounding_ciou_wandb_online.sh" &
  grounding_eval_pid=$!
fi

set +e
swift sft \
  --model "${MODEL_PATH}" \
  --model_type qwen3_vl \
  --tuner_type full \
  --freeze_llm false \
  --freeze_vit false \
  --freeze_aligner false \
  --deepspeed zero3 \
  --dataset "${TRAIN_DATA[@]}" \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --torch_dtype bfloat16 \
  --attn_impl flash_attention_2 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing true \
  --padding_free true \
  --packing false \
  --sequence_parallel_size "${SEQUENCE_PARALLEL_SIZE}" \
  --use_logits_to_keep false \
  --max_length "${MAX_LENGTH}" \
  --truncation_strategy right \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --logging_steps 5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --dataset_num_proc 16 \
  --dataloader_num_workers 8 \
  --dataloader_persistent_workers true \
  --dataloader_prefetch_factor 2 \
  --report_to "${REPORT_TO}" \
  --output_dir "${OUTPUT_DIR}"
status=$?
set -e

if [[ -n "${grounding_eval_pid}" ]]; then
  kill "${grounding_eval_pid}" 2>/dev/null || true
  wait "${grounding_eval_pid}" 2>/dev/null || true
fi

echo "swift exit status: ${status}"
find "${ELASTIC_LOG_DIR}" -maxdepth 1 -type f -name 'single_error_*.json' -print -exec cat {} \; || true

if [[ "${status}" == "0" && "${GROUNDING_EVAL_AFTER_TRAIN}" == "1" ]]; then
  echo "Starting post-training grounding IoU/c-IoU checkpoint backfill"
  OUTPUT_DIR="${OUTPUT_DIR}" STAGE_ROOT="${STAGE_ROOT}" MS_SWIFT_ROOT="${MS_SWIFT_ROOT}" MANIFEST="${MANIFEST}" WATCH=0 \
    bash "${MS_SWIFT_ROOT}/scripts/run_grounding_iou_checkpoint_eval.sh" || {
      echo "WARNING: c-IoU evaluation failed; training completed successfully" >&2
    }
fi
exit "${status}"
