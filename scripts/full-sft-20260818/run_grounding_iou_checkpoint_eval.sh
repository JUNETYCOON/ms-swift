#!/usr/bin/env bash
set -euo pipefail

# Run checkpoint-level grounding IoU evaluation and log curves to W&B.
#
# Default mode is one-shot backfill: scan OUTPUT_DIR/checkpoint-* once.
# Set WATCH=1 to poll for new checkpoints. For 48-card full training, prefer
# post-training backfill unless a separate idle GPU node is available.

: "${OUTPUT_DIR:?Set OUTPUT_DIR to the ms-swift training output directory}"

STAGE_ROOT=${STAGE_ROOT:-/mnt/luojunkun/stage1}
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-${STAGE_ROOT}/ms-swift}
MANIFEST=${MANIFEST:-${MS_SWIFT_ROOT}/scripts/dlc_ready_entrypoints.stage1.json}
EVAL_ROOT=${GROUNDING_EVAL_ROOT:-${STAGE_ROOT}/sft-model/grounding-iou-eval}
EVAL_SET=${GROUNDING_EVAL_SET:-${EVAL_ROOT}/grounding_iou_eval.jsonl}
EVAL_SET_REPORT=${GROUNDING_EVAL_SET_REPORT:-${EVAL_ROOT}/grounding_iou_eval_set_report.json}
MAX_PER_DATASET=${GROUNDING_EVAL_MAX_PER_DATASET:-64}
POLL_SECONDS=${GROUNDING_EVAL_POLL_SECONDS:-300}
WATCH=${WATCH:-0}
WAIT_FOR_GPU=${GROUNDING_EVAL_WAIT_FOR_GPU:-0}
MIN_FREE_MIB=${GROUNDING_EVAL_MIN_FREE_MIB:-30000}
PREDICTION_SPACE=${GROUNDING_PREDICTION_SPACE:-norm1000}
GROUND_TRUTH_SPACE=${GROUNDING_GT_SPACE:-objects}
THRESHOLDS=${GROUNDING_IOU_THRESHOLDS:-0.25 0.5 0.75}
WANDB_PROJECT=${WANDB_PROJECT:-ms-swift}
WANDB_RESUME=${WANDB_RESUME:-allow}

export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${EVAL_ROOT}"
cd "${MS_SWIFT_ROOT}" || exit 2

if [[ ! -s "${EVAL_SET}" ]]; then
  python3 scripts/data-process/prepare_grounding_iou_eval_set.py \
    --manifest "${MANIFEST}" \
    --output "${EVAL_SET}" \
    --report "${EVAL_SET_REPORT}" \
    --max-per-dataset "${MAX_PER_DATASET}"
fi

step_from_checkpoint() {
  local path="$1"
  basename "${path}" | sed -E 's/^checkpoint-([0-9]+)$/\1/'
}

wait_for_gpu_gate() {
  if [[ "${WAIT_FOR_GPU}" != "1" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-gate] nvidia-smi unavailable; skip GPU gate"
    return 0
  fi
  while true; do
    local max_free
    max_free="$(
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m}'
    )"
    if [[ -n "${max_free}" && "${max_free}" -ge "${MIN_FREE_MIB}" ]]; then
      echo "[gpu-gate] open: max_free=${max_free} MiB required=${MIN_FREE_MIB} MiB"
      return 0
    fi
    echo "[gpu-gate] waiting: max_free=${max_free:-unknown} MiB required=${MIN_FREE_MIB} MiB"
    sleep "${POLL_SECONDS}"
  done
}

run_one_checkpoint() {
  local checkpoint="$1"
  local step="$2"
  local marker="${EVAL_ROOT}/checkpoint-${step}.done"
  local run_dir="${EVAL_ROOT}/checkpoint-${step}"
  local pred="${run_dir}/grounding_pred.jsonl"
  local report="${run_dir}/iou_report.json"
  local details="${run_dir}/iou_errors.jsonl"
  mkdir -p "${run_dir}"
  if [[ -f "${marker}" ]]; then
    echo "[skip] checkpoint-${step} already evaluated: ${marker}"
    return 0
  fi

  echo "[eval] checkpoint=${checkpoint} step=${step}"
  wait_for_gpu_gate
  swift infer \
    --model "${checkpoint}" \
    --val_dataset "${EVAL_SET}" \
    --result_path "${pred}"

  python3 scripts/evaluate_grounding_iou.py \
    --input "${pred}" \
    --prediction-space "${PREDICTION_SPACE}" \
    --ground-truth-space "${GROUND_TRUTH_SPACE}" \
    --thresholds ${THRESHOLDS} \
    --group-by dataset_name objects.bbox_type \
    --report "${report}" \
    --details-output "${details}" \
    --overwrite

  local wandb_run_args=()
  if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    wandb_run_args+=(--run-id "${WANDB_RUN_ID}")
  fi
  python3 scripts/log_grounding_iou_to_wandb.py \
    --report "${report}" \
    --project "${WANDB_PROJECT}" \
    "${wandb_run_args[@]}" \
    --resume "${WANDB_RESUME}" \
    --step "${step}"

  date -u +"%Y-%m-%dT%H:%M:%SZ" >"${marker}"
  echo "[done] checkpoint-${step}"
}

scan_once() {
  shopt -s nullglob
  local checkpoints=("${OUTPUT_DIR}"/checkpoint-*)
  shopt -u nullglob
  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "[scan] no checkpoints under ${OUTPUT_DIR}"
    return 0
  fi
  printf '%s\n' "${checkpoints[@]}" | sort -V | while read -r checkpoint; do
    [[ -d "${checkpoint}" ]] || continue
    local step
    step="$(step_from_checkpoint "${checkpoint}")"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    run_one_checkpoint "${checkpoint}" "${step}"
  done
}

if [[ "${WATCH}" == "1" ]]; then
  echo "[watch] polling ${OUTPUT_DIR}/checkpoint-* every ${POLL_SECONDS}s"
  while true; do
    scan_once
    sleep "${POLL_SECONDS}"
  done
else
  scan_once
fi
