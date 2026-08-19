#!/usr/bin/env bash
set -euo pipefail

# Canonical stage-1 full SFT launcher for DLC (3 nodes x 16 GPUs).
#
# Proven smoke baseline (2026-08-17): bs8 / ga1 / sp4 / ml50000 / workers=1
#   gpu_memory_mib_max=96123, gpu_util_avg=78.5%
# Full run default: bs8 / ga8 / sp4, effective global batch = 8*8*12 = 768
#   on 48 GPUs with sequence parallel size 4.

cd /mnt/luojunkun/stage1/ms-swift
export PYTHONPATH="/mnt/luojunkun/stage1/ms-swift:${PYTHONPATH:-}"

: "${WORLD_SIZE:?DLC must provide WORLD_SIZE (node count)}"
: "${RANK:?DLC must provide RANK (node rank)}"
: "${MASTER_ADDR:?DLC must provide MASTER_ADDR}"

STAGE_ROOT=${STAGE_ROOT:-/mnt/luojunkun/stage1}
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-${STAGE_ROOT}/ms-swift}
TOOL_ROOT=${TOOL_ROOT:-${MS_SWIFT_ROOT}/scripts}
MODEL_PATH=${MODEL_PATH:-${STAGE_ROOT}/model}
OUTPUT_DIR=${OUTPUT_DIR:-${STAGE_ROOT}/sft-model/qwen3-vl-stage1-dlc3x16-full}
MANIFEST=${MANIFEST:-${TOOL_ROOT}/dlc_ready_entrypoints.stage1.json}

NPROC_PER_NODE=${NPROC_PER_NODE:-16}
EXPECTED_NNODES=${EXPECTED_NNODES:-3}
EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-48}
TOTAL_PROCESSES=$((WORLD_SIZE * NPROC_PER_NODE))
if [[ ${WORLD_SIZE} -ne ${EXPECTED_NNODES} ]]; then
  echo "Expected ${EXPECTED_NNODES} nodes, got WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 2
fi
if [[ ${TOTAL_PROCESSES} -ne ${EXPECTED_WORLD_SIZE} ]]; then
  echo "Expected ${EXPECTED_WORLD_SIZE} accelerators, got WORLD_SIZE=${WORLD_SIZE} x NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 2
fi

MAX_LENGTH=${MAX_LENGTH:-50000}
SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE:-4}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:--1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
SAVE_STEPS=${SAVE_STEPS:-2000}
MASTER_PORT=${MASTER_PORT:-29500}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-1}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-1}
DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS:-false}

if (( TOTAL_PROCESSES % SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "World size ${TOTAL_PROCESSES} must be divisible by SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE}" >&2
  exit 2
fi
if ! command -v swift >/dev/null 2>&1; then
  echo "swift CLI is unavailable in the active DLC environment" >&2
  exit 2
fi

TRAIN_DATA_LIST=$(mktemp)
trap 'rm -f "${TRAIN_DATA_LIST}"' EXIT
python3 - "${MANIFEST}" >"${TRAIN_DATA_LIST}" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
policy = manifest["global_dedup"]
report_path = Path(policy["report"]).resolve()
report = json.loads(report_path.read_text(encoding="utf-8"))
audit_path = report_path.parent / "dlc_sft_audit_report.json"
audit = json.loads(audit_path.read_text(encoding="utf-8"))
assert policy.get("deduplicate_cross_dataset_train") is False
assert report.get("status") == "complete"
assert report.get("verification", {}).get("status") == "complete"
assert report["verification"]["train_eval_overlap_rows"] == 0
assert audit.get("status") == "passed", audit.get("hard_failures")
manifest_fingerprint = report.get("manifest_fingerprint", {})
assert Path(manifest_fingerprint.get("path", "")).resolve() == manifest_path
assert manifest_fingerprint.get("sha256") == hashlib.sha256(
    manifest_path.read_bytes()
).hexdigest(), "ready manifest changed after decontamination"
for name, config in manifest["datasets"].items():
    if config.get("enabled", True):
        path = Path(config["train"]).resolve()
        evidence = report["datasets"][name]
        fingerprint = evidence["fingerprints"]["train"]
        stat = path.stat()
        assert Path(evidence["train"]).resolve() == path, (name, path)
        assert Path(fingerprint["path"]).resolve() == path, (name, path)
        assert stat.st_size == fingerprint["size"] > 0, (name, path, "size changed")
        assert stat.st_mtime_ns <= report_path.stat().st_mtime_ns, (
            name,
            path,
            "changed after decontamination report",
        )
        print(path)
PY
mapfile -t TRAIN_DATA <"${TRAIN_DATA_LIST}"
rm -f "${TRAIN_DATA_LIST}"
trap - EXIT
if [[ ${#TRAIN_DATA[@]} -eq 0 ]]; then
  echo "The ready manifest contains no enabled training datasets" >&2
  exit 2
fi
echo "DLC train entrypoints and audit gates are ready (${#TRAIN_DATA[@]} datasets)."

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
export FPS=${FPS:-25}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-128}
export QWENVL_BBOX_FORMAT=${QWENVL_BBOX_FORMAT:-legacy}
export CELOSS_PARALLEL_SIZE=${CELOSS_PARALLEL_SIZE:-2048}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export WANDB_PROJECT=${WANDB_PROJECT:-ms-swift}
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
  WANDB_RUN_ID="stage1-full-$(basename "${OUTPUT_DIR}" | tr -c 'A-Za-z0-9_-' '-')"
  export WANDB_RUN_ID
fi

echo "WORLD_SIZE=${WORLD_SIZE} RANK=${RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "BS=${PER_DEVICE_TRAIN_BATCH_SIZE} GA=${GRADIENT_ACCUMULATION_STEPS} SP=${SEQUENCE_PARALLEL_SIZE} MAX_LENGTH=${MAX_LENGTH}"
echo "MODEL=${MODEL_PATH}"
echo "MANIFEST=${MANIFEST} DATASET_COUNT=${#TRAIN_DATA[@]}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

NNODES=${WORLD_SIZE} \
NODE_RANK=${RANK} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
swift sft \
  --model "${MODEL_PATH}" \
  --model_type qwen3_vl \
  --tuner_type full \
  --dataset "${TRAIN_DATA[@]}" \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --torch_dtype bfloat16 \
  --attn_impl flash_attention_2 \
  --freeze_llm false \
  --freeze_vit false \
  --freeze_aligner false \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing true \
  --max_grad_norm 0.5 \
  --padding_free true \
  --packing false \
  --sequence_parallel_size "${SEQUENCE_PARALLEL_SIZE}" \
  --use_logits_to_keep false \
  --max_length "${MAX_LENGTH}" \
  --truncation_strategy right \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 50 \
  --logging_steps 5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --deepspeed zero2 \
  --report_to wandb \
  --output_dir "${OUTPUT_DIR}"
