#!/usr/bin/env bash
set -euo pipefail

# Alibaba Cloud DLC exports WORLD_SIZE as the node count and RANK as node rank.
: "${WORLD_SIZE:?DLC must provide WORLD_SIZE (node count)}"
: "${RANK:?DLC must provide RANK (node rank)}"
: "${MASTER_ADDR:?DLC must provide MASTER_ADDR}"
: "${MODEL_PATH:?Set MODEL_PATH to the base/full checkpoint shared by every node}"

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

STAGE_ROOT=${STAGE_ROOT:-/mnt/luojunkun/stage1}
TOOL_ROOT=${TOOL_ROOT:-${STAGE_ROOT}/sft-model/scripts}
OUTPUT_DIR=${OUTPUT_DIR:-${STAGE_ROOT}/sft-model/qwen3-vl-stage1-dlc3x16-full}
MAX_LENGTH=${MAX_LENGTH:-65536}
SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
SAVE_STEPS=${SAVE_STEPS:-1000}
MASTER_PORT=${MASTER_PORT:-29500}

if (( TOTAL_PROCESSES % SEQUENCE_PARALLEL_SIZE != 0 )); then
  echo "World size ${TOTAL_PROCESSES} must be divisible by SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE}" >&2
  exit 2
fi
if ! command -v swift >/dev/null 2>&1; then
  echo "swift CLI is unavailable in the active DLC environment" >&2
  exit 2
fi

MANIFEST=${TOOL_ROOT}/dlc_ready_entrypoints.stage1.json
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
manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert (
    Path(audit.get("manifest", "")).resolve() == manifest_path
    or manifest_fingerprint.get("sha256") == manifest_sha256
)
assert manifest_fingerprint.get("sha256") == manifest_sha256, "ready manifest changed after decontamination"
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

NNODES=${WORLD_SIZE} \
NODE_RANK=${RANK} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
swift sft \
  --model "${MODEL_PATH}" \
  --tuner_type full \
  --dataset "${TRAIN_DATA[@]}" \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --freeze_llm false \
  --freeze_vit false \
  --freeze_aligner false \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
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
  --save_total_limit 3 \
  --logging_steps 5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --dataset_num_proc 16 \
  --dataloader_num_workers 8 \
  --deepspeed zero3 \
  --report_to tensorboard \
  --output_dir "${OUTPUT_DIR}"
