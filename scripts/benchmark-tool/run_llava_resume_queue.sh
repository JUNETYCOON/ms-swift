#!/usr/bin/env bash

set -uo pipefail

STAGE_ROOT=/mnt/luojunkun/stage1
TOOL_ROOT=/mnt/workspace/stage1/scripts/benchmark-tool
DATASET=${STAGE_ROOT}/dataset_ms-swift/llava-instruct/llava_v1_5_mix665k_sft_msswift_val.jsonl
BASELINE_WEIGHTS=${STAGE_ROOT}/model
OURS_WEIGHTS=${STAGE_ROOT}/sft-model/qwen3-vl-instruct4b/v30-20260728-003837/checkpoint-400
OUTPUT_ROOT=${STAGE_ROOT}/benchmark-eval-result/self-valdataset/llava-instruct
LOG_ROOT=${STAGE_ROOT}/benchmark-eval-result/stage1-autoeval/logs/llava-resume
GPU=${LLAVA_CUDA_VISIBLE_DEVICES:-7}

mkdir -p "${LOG_ROOT}"

run_model() {
    local model=$1
    local weights=$2
    local output_dir=${OUTPUT_ROOT}/${model}
    local log_file=${LOG_ROOT}/${model}.log
    if [[ -f "${output_dir}/scores.json" ]]; then
        printf '[llava-resume] skip existing scores: %s\n' "${model}"
        return
    fi
    printf '[llava-resume] resuming %s on physical GPU %s\n' "${model}" "${GPU}"
    CUDA_VISIBLE_DEVICES="${GPU}" python "${TOOL_ROOT}/cli.py" eval-custom \
        --val-dataset "${DATASET}" \
        --model-weights "${weights}" \
        --model_type qwen3_vl \
        --model-name "${model}" \
        --dataset-name llava-instruct \
        --task auto \
        --output-dir "${output_dir}" \
        --batch-size 32 \
        --max-new-tokens 512 \
        --dtype bfloat16 \
        --device cuda:0 \
        --attn-implementation sdpa \
        --progress-every 100 \
        --continue-on-error \
        --resume \
        --retry-errors > "${log_file}" 2>&1
}

run_model baseline "${BASELINE_WEIGHTS}"
run_model ours "${OURS_WEIGHTS}"
