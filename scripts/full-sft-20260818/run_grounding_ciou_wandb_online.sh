#!/usr/bin/env bash
set -euo pipefail

# Online W&B grounding c-IoU watcher.
#
# Run this while training is running. It polls OUTPUT_DIR/checkpoint-* and logs
# c-IoU/IoU metrics to the same W&B run. Prefer launching it on an idle GPU node
# or restrict it to free GPUs via GROUNDING_EVAL_CUDA_VISIBLE_DEVICES.

: "${OUTPUT_DIR:?Set OUTPUT_DIR to the ms-swift training output directory}"

STAGE_ROOT=${STAGE_ROOT:-/mnt/luojunkun/stage1}
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-${STAGE_ROOT}/ms-swift}

export WATCH=1
export GROUNDING_EVAL_WAIT_FOR_GPU=${GROUNDING_EVAL_WAIT_FOR_GPU:-1}
export GROUNDING_EVAL_MIN_FREE_MIB=${GROUNDING_EVAL_MIN_FREE_MIB:-30000}
export GROUNDING_EVAL_POLL_SECONDS=${GROUNDING_EVAL_POLL_SECONDS:-300}
export GROUNDING_EVAL_ROOT=${GROUNDING_EVAL_ROOT:-${STAGE_ROOT}/sft-model/grounding-ciou-online-eval}
export GROUNDING_EVAL_MAX_PER_DATASET=${GROUNDING_EVAL_MAX_PER_DATASET:-32}
export GROUNDING_IOU_THRESHOLDS=${GROUNDING_IOU_THRESHOLDS:-"0.25 0.5 0.75"}
export GROUNDING_PREDICTION_SPACE=${GROUNDING_PREDICTION_SPACE:-norm1000}
export GROUNDING_GT_SPACE=${GROUNDING_GT_SPACE:-objects}
export WANDB_PROJECT=${WANDB_PROJECT:-ms-swift}

if [[ -n "${GROUNDING_EVAL_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GROUNDING_EVAL_CUDA_VISIBLE_DEVICES}"
fi

echo "ONLINE_GROUNDING_CIOU_WATCH=1"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MS_SWIFT_ROOT=${MS_SWIFT_ROOT}"
echo "GROUNDING_EVAL_ROOT=${GROUNDING_EVAL_ROOT}"
echo "GROUNDING_EVAL_POLL_SECONDS=${GROUNDING_EVAL_POLL_SECONDS}"
echo "GROUNDING_EVAL_WAIT_FOR_GPU=${GROUNDING_EVAL_WAIT_FOR_GPU}"
echo "GROUNDING_EVAL_MIN_FREE_MIB=${GROUNDING_EVAL_MIN_FREE_MIB}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_RUN_ID=${WANDB_RUN_ID:-<not set; set it to match training run>}"
echo "Expected W&B scalar: grounding/c_iou@0.5"

exec bash "${MS_SWIFT_ROOT}/scripts/run_grounding_iou_checkpoint_eval.sh"
