#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT=/mnt/workspace/stage1
STAGE_ROOT=/mnt/luojunkun/stage1
CLI=${REPO_ROOT}/scripts/benchmark-tool/cli.py
VAL_DATASET=${STAGE_ROOT}/dataset_ms-swift/robovqa/robovqa_train_sft_eval.jsonl
OUTPUT_ROOT=${STAGE_ROOT}/benchmark-eval-result/self-valdataset/robovqa-smoke-256-b4-f16-t64
LOG_ROOT=${STAGE_ROOT}/benchmark-eval-result/stage1-autoeval/logs/robovqa-smoke-256-b4-f16-t64

BASELINE_WEIGHTS=${STAGE_ROOT}/model
OURS_WEIGHTS=${STAGE_ROOT}/sft-model/7-30_stage1_modelv1
BATCH_SIZE=${BATCH_SIZE:-4}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
RUN_TAG=${RUN_TAG:-initial}
STATUS_FILE=${LOG_ROOT}/run-${RUN_TAG}-status.txt

export LD_LIBRARY_PATH=/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_one() {
    local label=$1
    local weights=$2
    local output_dir=${OUTPUT_ROOT}/${label}
    local log_file=${LOG_ROOT}/${label}-${RUN_TAG}.log

    python "${CLI}" eval-custom \
        --val-dataset "${VAL_DATASET}" \
        --model-weights "${weights}" \
        --model_type qwen3_vl \
        --model-name "${label}" \
        --dataset-name robovqa_smoke_256_b4_f16_t64 \
        --task vqa \
        --output-dir "${output_dir}" \
        --batch-size "${BATCH_SIZE}" \
        --max-new-tokens 64 \
        --max-video-frames 16 \
        --group-video-batches \
        --limit 256 \
        --dtype bfloat16 \
        --device cuda:0 \
        --attn-implementation "${ATTN_IMPLEMENTATION}" \
        --progress-every 16 \
        --continue-on-error \
        --resume \
        --retry-errors \
        >"${log_file}" 2>&1
}

printf 'started_at=%s\nbatch_size=%s\nattn_implementation=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${BATCH_SIZE}" "${ATTN_IMPLEMENTATION}" \
    > "${STATUS_FILE}"

run_one baseline "${BASELINE_WEIGHTS}" &
baseline_pid=$!
run_one ours "${OURS_WEIGHTS}" &
ours_pid=$!
printf 'baseline_pid=%s\nours_pid=%s\n' "${baseline_pid}" "${ours_pid}" >> "${STATUS_FILE}"

wait "${baseline_pid}"
baseline_exit=$?
wait "${ours_pid}"
ours_exit=$?

printf 'baseline_exit=%s\nours_exit=%s\nfinished_at=%s\n' \
    "${baseline_exit}" "${ours_exit}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    >> "${STATUS_FILE}"

if (( baseline_exit != 0 || ours_exit != 0 )); then
    exit 1
fi
