#!/usr/bin/env bash
set -euo pipefail

# Run from a per-invocation snapshot so editing the canonical script while a
# job is running cannot corrupt the running bash process mid-run.
if [ -z "${MEM_SMOKE_SNAPSHOT_ACTIVE:-}" ]; then
  SNAP_ROOT=${SNAP_ROOT:-/tmp}
  SNAP="${SNAP_ROOT}/stage1_mem_smoke_16bs_$(date +%Y%m%d_%H%M%S)_$$.sh"
  if cp "$0" "${SNAP}" 2>/dev/null; then
    chmod +x "${SNAP}" 2>/dev/null || true
    export MEM_SMOKE_SNAPSHOT_ACTIVE=1
    exec bash "${SNAP}" "$@"
  fi
fi

# Single-node 16bs memory/loss smoke for stage-1 SFT.
# Requires: MANIFEST, RUN_TAG, OUTPUT_DIR, LOG_DIR
# Monitors nvidia-smi and host memory in the background, logs loss to W&B, and
# writes a result.json/report.md with max GPU memory and average utilization.

STAGE_ROOT=${STAGE_ROOT:-/mnt/luojunkun/stage1}
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-${STAGE_ROOT}/ms-swift}
MODEL=${MODEL:-${STAGE_ROOT}/model}

: "${MANIFEST:?MANIFEST is required}"
: "${RUN_TAG:?RUN_TAG is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"

MAX_STEPS=${MAX_STEPS:-300}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-100}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE:-4}
MAX_LENGTH=${MAX_LENGTH:-50000}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-linear}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
EXPECTED_GPU_COUNT=${EXPECTED_GPU_COUNT:-16}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
MASTER_PORT=${MASTER_PORT:-29851}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-1}
DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS:-true}
SAVE_STRATEGY=${SAVE_STRATEGY:-no}
SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-}
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-false}

export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-${STAGE_ROOT}/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
export FPS=${FPS:-25}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-128}
export QWENVL_BBOX_FORMAT=${QWENVL_BBOX_FORMAT:-legacy}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT
unset WORLD_SIZE RANK LOCAL_WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE

export TMPDIR=${TMPDIR:-/mnt/workspace/stage1/tmp/${RUN_TAG}}
mkdir -p "${TMPDIR}"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CELOSS_PARALLEL_SIZE=${CELOSS_PARALLEL_SIZE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE:-1}

export WANDB_PROJECT=${WANDB_PROJECT:-ms-swift}
export WANDB_RUN_ID=${WANDB_RUN_ID:-${RUN_TAG}}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_TAG}}
export WANDB_DIR=${WANDB_DIR:-${STAGE_ROOT}/wandb}
mkdir -p "${WANDB_DIR}" "${WANDB_DIR}/config" "${WANDB_DIR}/cache"
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-${WANDB_DIR}/config}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
export WANDB_SILENT=true

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
TRAIN_LOG=${LOG_DIR}/train.log
GPU_LOG=${LOG_DIR}/gpu.csv
MEM_LOG=${LOG_DIR}/mem.csv
TMP_GPU_LOG=${TMPDIR}/${RUN_TAG}-gpu.csv
RESULT_JSON=${LOG_DIR}/result.json
REPORT_MD=${LOG_DIR}/report.md

cd "${MS_SWIFT_ROOT}" || exit 2

{
  echo "HOSTNAME=${HOSTNAME}"
  echo "RUN_TAG=${RUN_TAG}"
  echo "MANIFEST=${MANIFEST}"
  echo "MODEL=${MODEL}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "TMPDIR=${TMPDIR}"
  echo "MAX_STEPS=${MAX_STEPS}"
  echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
  echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
  echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
  echo "SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE}"
  echo "MAX_LENGTH=${MAX_LENGTH}"
  echo "LEARNING_RATE=${LEARNING_RATE}"
  echo "LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE}"
  echo "WARMUP_RATIO=${WARMUP_RATIO}"
  echo "DATASET_NUM_PROC=${DATASET_NUM_PROC}"
  echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}"
  echo "DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR}"
  echo "DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS}"
  echo "SAVE_STRATEGY=${SAVE_STRATEGY}"
  echo "SAVE_STEPS=${SAVE_STEPS}"
  echo "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
  echo "SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"
  echo "WANDB_RUN_ID=${WANDB_RUN_ID}"
  echo "WANDB_DIR=${WANDB_DIR}"
} | tee "${TRAIN_LOG}"

python -c 'import swift; print("swift package:", swift.__file__)' | tee -a "${TRAIN_LOG}" || exit 2
python - "${EXPECTED_GPU_COUNT}" <<'PY' | tee -a "${TRAIN_LOG}" || exit 2
import sys
import torch

expected = int(sys.argv[1])
count = torch.cuda.device_count()
print("visible GPU count:", count)
assert count == expected, (count, expected)
PY

DATASET_LIST=$(python3 - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for _, cfg in manifest["datasets"].items():
    if cfg.get("enabled", True):
        print(cfg["train"])
PY
)
echo "DATASETS=${DATASET_LIST}" | tee -a "${TRAIN_LOG}"

{
  printf '%s\n' "timestamp,index,memory.used,memory.total,utilization.gpu,power.draw"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits || true
    sleep 5
  done
} >"${TMP_GPU_LOG}" &
monitor_pid=$!

{
  printf '%s\n' "timestamp,mem_total_gib,mem_used_gib,mem_available_gib,swap_total_gib,swap_used_gib" 2>/dev/null || true
  while true; do
    mem=$(free -g | awk '/^Mem:/{printf "%s %s %s", $2, $3, $7}') || true
    swap=$(free -g | awk '/^Swap:/{printf "%s %s", $2, $3}') || true
    echo "$(date '+%Y/%m/%d %H:%M:%S'),${mem},${swap}" 2>/dev/null || true
    sleep 5
  done
} >"${MEM_LOG}" &
mem_monitor_pid=$!

trap 'kill "${monitor_pid}" "${mem_monitor_pid}" 2>/dev/null || true; pkill -f "wandb.*service" 2>/dev/null || true' EXIT

SAVE_ARGS=(--save_strategy "${SAVE_STRATEGY}")
if [[ "${SAVE_STRATEGY}" != "no" ]]; then
  SAVE_ARGS+=(--save_steps "${SAVE_STEPS}")
  if [[ -n "${SAVE_TOTAL_LIMIT}" ]]; then
    SAVE_ARGS+=(--save_total_limit "${SAVE_TOTAL_LIMIT}")
  fi
  if [[ "${SAVE_ONLY_MODEL}" == "true" ]]; then
    SAVE_ARGS+=(--save_only_model true)
  fi
fi

CALLBACK_ARGS=()
if [[ -n "${CALLBACKS:-}" ]]; then
  CALLBACK_ARGS=(--callbacks)
  read -r -a CALLBACKS_LIST <<< "${CALLBACKS}"
  CALLBACK_ARGS+=("${CALLBACKS_LIST[@]}")
fi

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
  --logging_steps 1 \
  --max_length "${MAX_LENGTH}" \
  --truncation_strategy right \
  --eval_strategy no \
  "${SAVE_ARGS[@]}" \
  "${CALLBACK_ARGS[@]}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --include_num_input_tokens_seen true \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_TAG}" \
  --report_to wandb >> "${TRAIN_LOG}" 2>&1
status=$?
set -e

kill "${monitor_pid}" "${mem_monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" "${mem_monitor_pid}" 2>/dev/null || true
cp "${TMP_GPU_LOG}" "${GPU_LOG}" 2>/dev/null || true

python3 - "${TRAIN_LOG}" "${GPU_LOG}" "${RESULT_JSON}" "${REPORT_MD}" "${RUN_TAG}" "${status}" \
  "${PER_DEVICE_TRAIN_BATCH_SIZE}" "${NPROC_PER_NODE}" "${SEQUENCE_PARALLEL_SIZE}" \
  "${GRADIENT_ACCUMULATION_STEPS}" "${MAX_STEPS}" <<'PY' | tee -a "${TRAIN_LOG}"
import csv
import json
import re
import sys
from pathlib import Path

train_log = Path(sys.argv[1])
gpu_log = Path(sys.argv[2])
result_json = Path(sys.argv[3])
report_md = Path(sys.argv[4])
run_tag = sys.argv[5]
status = int(sys.argv[6])
bs = int(sys.argv[7])
nproc = int(sys.argv[8])
sp = int(sys.argv[9])
ga = int(sys.argv[10])
max_steps = int(sys.argv[11])

text = train_log.read_text(encoding="utf-8", errors="ignore") if train_log.exists() else ""

def parse_metric(text, key):
    values = {}
    current_step = None
    pattern = re.compile(rf"['\"]{re.escape(key)}['\"]\s*:\s*['\"]?([-+0-9.eE]+)")
    for line in text.splitlines():
        step_match = re.search(r"['\"]global_step/max_steps['\"]\s*:\s*['\"]?(\d+)/", line)
        if step_match:
            current_step = int(step_match.group(1))
        metric_match = pattern.search(line)
        if metric_match and current_step is not None:
            try:
                values[current_step] = float(metric_match.group(1))
            except ValueError:
                pass
    return values

loss_by_step = parse_metric(text, "loss")
grad_by_step = parse_metric(text, "grad_norm")
lr_by_step = parse_metric(text, "learning_rate")

gpu_mem = []
gpu_util = []
gpu_rows = 0
if gpu_log.exists():
    with gpu_log.open(encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            gpu_rows += 1
            try:
                gpu_mem.append(float(row[2]))
            except ValueError:
                pass
            try:
                gpu_util.append(float(row[4]))
            except ValueError:
                pass

losses = [v for _, v in sorted(loss_by_step.items())]
grads = [v for _, v in sorted(grad_by_step.items())]
result = {
    "run_tag": run_tag,
    "status": status,
    "max_steps": max_steps,
    "per_device_train_batch_size": bs,
    "gradient_accumulation_steps": ga,
    "nproc_per_node": nproc,
    "global_micro_batch_size": bs * nproc,
    "sequence_parallel_size": sp,
    "loss_last": losses[-1] if losses else None,
    "loss_min": min(losses) if losses else None,
    "grad_norm_max": max(grads) if grads else None,
    "gpu_monitor_rows": gpu_rows,
    "gpu_memory_mib_max": max(gpu_mem) if gpu_mem else None,
    "gpu_util_percent_avg": sum(gpu_util) / len(gpu_util) if gpu_util else None,
    "gpu_log": str(gpu_log),
    "train_log": str(train_log),
}
result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    f"# Memory Smoke Report: {run_tag}",
    "",
    "## Config",
    "",
    "| key | value |",
    "|---|---|",
]
for key, value in result.items():
    lines.append(f"| {key} | {value} |")
lines.append("")
lines.append("## Loss by step")
lines.append("")
lines.append("| step | loss | grad_norm | lr |")
lines.append("|---:|---:|---:|---:|")
steps = sorted(set(loss_by_step) | set(grad_by_step) | set(lr_by_step))
for step in steps:
    lines.append(
        f"| {step} | {loss_by_step.get(step)} | {grad_by_step.get(step)} | {lr_by_step.get(step)} |"
    )
lines.append("")
report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

echo "swift exit status: ${status}" | tee -a "${TRAIN_LOG}"
echo "GPU_LOG=${GPU_LOG}" | tee -a "${TRAIN_LOG}"
echo "MEM_LOG=${MEM_LOG}" | tee -a "${TRAIN_LOG}"
echo "RESULT_JSON=${RESULT_JSON}" | tee -a "${TRAIN_LOG}"
echo "REPORT_MD=${REPORT_MD}" | tee -a "${TRAIN_LOG}"
if [ "${status:-0}" -eq 0 ]; then
  : > "${OUTPUT_DIR}/training_done"
  echo "training_done marker written: ${OUTPUT_DIR}/training_done" | tee -a "${TRAIN_LOG}"
fi
if [ "${status:-0}" -eq 0 ]; then
  exit 0
fi
exit "${status}"
