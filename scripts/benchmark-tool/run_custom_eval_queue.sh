#!/usr/bin/env bash

set -uo pipefail

STAGE_ROOT=/mnt/luojunkun/stage1
REPO_ROOT=/mnt/workspace/stage1
DATA_ROOT=${STAGE_ROOT}/dataset_ms-swift
RESULT_ROOT=${STAGE_ROOT}/benchmark-eval-result/self-valdataset
CONTROL_ROOT=${STAGE_ROOT}/benchmark-eval-result/stage1-autoeval
BASELINE_WEIGHTS=${STAGE_ROOT}/model
OURS_WEIGHTS=${STAGE_ROOT}/sft-model/qwen3-vl-instruct4b/v30-20260728-003837/checkpoint-400
CLI=${REPO_ROOT}/scripts/benchmark-tool/cli.py
ONLY_DATASET=${1:-}

LOG_ROOT=${CONTROL_ROOT}/logs/custom
STATUS_FILE=${CONTROL_ROOT}/custom-status.tsv
GPU_LOG=${CONTROL_ROOT}/gpu-memory.csv
PLOT_EVERY=${PLOT_EVERY:-200}
WANDB_PROJECT=${WANDB_PROJECT:-}
mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}" "${CONTROL_ROOT}"

if [[ ! -f "${STATUS_FILE}" ]]; then
    printf 'timestamp\tdataset\tmodel\tstate\texit_code\toutput_dir\n' > "${STATUS_FILE}"
fi
if [[ ! -f "${GPU_LOG}" ]]; then
    printf 'timestamp,dataset,index,memory_total_mb,memory_used_mb,memory_free_mb,gpu_util_percent\n' > "${GPU_LOG}"
fi

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

is_clean_complete() {
    local score_file=$1
    [[ -f "${score_file}" ]] && python -c \
        'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if d.get("status") == "complete" and all(int(r.get("failed_samples", 0)) == 0 for r in d.get("scores", [])) else 1)' \
        "${score_file}" 2>/dev/null
}

wait_for_memory() {
    local minimum_free_mb=$1
    local free_mb
    while true; do
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
        if [[ "${free_mb}" =~ ^[0-9]+$ ]] && (( free_mb >= minimum_free_mb )); then
            return
        fi
        printf '[%s] waiting for GPU memory: free=%s MB, required=%s MB\n' \
            "$(timestamp)" "${free_mb:-unknown}" "${minimum_free_mb}"
        sleep 30
    done
}

monitor_gpu() {
    local dataset=$1
    shift
    while true; do
        local alive=0
        local pid
        for pid in "$@"; do
            if kill -0 "${pid}" 2>/dev/null; then
                alive=1
                break
            fi
        done
        (( alive == 1 )) || return
        nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits -i 0 | \
            while IFS= read -r row; do
                printf '%s,%s,%s\n' "$(timestamp)" "${dataset}" "${row// /}" >> "${GPU_LOG}"
            done
        sleep 10
    done
}

run_one() {
    local dataset=$1
    local val_file=$2
    local task=$3
    local batch_size=$4
    local max_tokens=$5
    local max_video_frames=$6
    local model_label=$7
    local weights=$8
    local output_dir=${RESULT_ROOT}/${dataset}/${model_label}
    local attempt
    attempt=$(date -u '+%Y%m%dT%H%M%SZ')
    local log_file=${LOG_ROOT}/${dataset}-${model_label}-${attempt}.log

    if is_clean_complete "${output_dir}/scores.json"; then
        printf '[%s] skip complete: %s/%s\n' "$(timestamp)" "${dataset}" "${model_label}"
        printf '%s\t%s\t%s\tskipped_complete\t0\t%s\n' \
            "$(timestamp)" "${dataset}" "${model_label}" "${output_dir}" >> "${STATUS_FILE}"
        return 0
    fi

    mkdir -p "${output_dir}"
    printf '[%s] start: %s/%s batch=%s tokens=%s\n' \
        "$(timestamp)" "${dataset}" "${model_label}" "${batch_size}" "${max_tokens}"
    printf '%s\t%s\t%s\tstarted\t\t%s\n' \
        "$(timestamp)" "${dataset}" "${model_label}" "${output_dir}" >> "${STATUS_FILE}"

    local command=(
        python "${CLI}" eval-custom
        --val-dataset "${val_file}"
        --model-weights "${weights}"
        --model_type qwen3_vl
        --model-name "${model_label}"
        --dataset-name "${dataset}"
        --task "${task}"
        --output-dir "${output_dir}"
        --batch-size "${batch_size}"
        --max-new-tokens "${max_tokens}"
        --dtype bfloat16
        --device cuda:0
        --attn-implementation sdpa
        --progress-every 200
        --plot-every "${PLOT_EVERY}"
        --plot-dir "${output_dir}/plots"
        --continue-on-error
        --resume
        --retry-errors
    )
    if [[ -n "${WANDB_PROJECT}" ]]; then
        command+=(--wandb-project "${WANDB_PROJECT}")
        command+=(--wandb-run-name "${model_label}-${dataset}-${attempt}")
        if [[ -n "${WANDB_RUN_ID:-}" ]]; then
            command+=(--wandb-run-id "${WANDB_RUN_ID}")
        fi
    fi
    if [[ -n "${max_video_frames}" ]]; then
        command+=(--max-video-frames "${max_video_frames}")
    fi

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
        "${command[@]}" > "${log_file}" 2>&1
    local exit_code=$?
    local state=failed
    if (( exit_code == 0 )) && is_clean_complete "${output_dir}/scores.json"; then
        state=complete
    elif (( exit_code == 0 )); then
        state=complete_with_errors
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(timestamp)" "${dataset}" "${model_label}" "${state}" "${exit_code}" "${output_dir}" >> "${STATUS_FILE}"
    printf '[%s] %s: %s/%s exit=%s\n' \
        "$(timestamp)" "${state}" "${dataset}" "${model_label}" "${exit_code}"
    return "${exit_code}"
}

run_pair() {
    local dataset=$1
    local val_file=$2
    local task=$3
    local batch_size=$4
    local max_tokens=$5
    local max_video_frames=$6

    if [[ -n "${ONLY_DATASET}" && "${ONLY_DATASET}" != "${dataset}" ]]; then
        return
    fi
    if [[ ! -f "${val_file}" ]]; then
        printf '%s\t%s\tall\tmissing_dataset\t2\t%s\n' \
            "$(timestamp)" "${dataset}" "${val_file}" >> "${STATUS_FILE}"
        printf '[%s] missing dataset: %s\n' "$(timestamp)" "${val_file}"
        return
    fi
    if is_clean_complete "${RESULT_ROOT}/${dataset}/baseline/scores.json" && \
            is_clean_complete "${RESULT_ROOT}/${dataset}/ours/scores.json"; then
        printf '[%s] skip clean pair: %s\n' "$(timestamp)" "${dataset}"
        return
    fi

    wait_for_memory 30000
    run_one "${dataset}" "${val_file}" "${task}" "${batch_size}" "${max_tokens}" "${max_video_frames}" \
        baseline "${BASELINE_WEIGHTS}" &
    local baseline_pid=$!
    run_one "${dataset}" "${val_file}" "${task}" "${batch_size}" "${max_tokens}" "${max_video_frames}" \
        ours "${OURS_WEIGHTS}" &
    local ours_pid=$!
    monitor_gpu "${dataset}" "${baseline_pid}" "${ours_pid}" &
    local monitor_pid=$!

    wait "${baseline_pid}" || true
    wait "${ours_pid}" || true
    wait "${monitor_pid}" || true
}

# Small datasets run first so processor/model compatibility fails fast.
run_pair ai2d "${DATA_ROOT}/ai2d/ai2d_pretrain_msswift_eval.jsonl" description 16 64 ''
run_pair chartqa "${DATA_ROOT}/chartqa/chartqa_val_sft_msswift.jsonl" vqa 32 16 ''
run_pair textvqa "${DATA_ROOT}/textvqa/textvqa_validation_sft_msswift.jsonl" vqa 16 16 ''
run_pair robo2vlm "${DATA_ROOT}/robo2vlm/robo2vlm_sft_eval.jsonl" vqa 16 64 ''
run_pair visualgenome_qa "${DATA_ROOT}/visualgenome/visualgenome_qa_val.jsonl" vqa 24 16 ''
run_pair vlm_r1_grounding "${DATA_ROOT}/vlm-r1/vlm_r1_sft_grounding_msswift_eval.jsonl" grounding 16 64 ''
run_pair vg-grounding "${DATA_ROOT}/visualgenome_grounding/visualgenome_regions_grounding_val.jsonl" grounding 32 16 ''
run_pair vg-grounded "${DATA_ROOT}/visualgenome/visualgenome_regions_val.jsonl" grounded 32 128 ''
run_pair llava "${DATA_ROOT}/llava-instruct/llava_v1_5_mix665k_sft_msswift_val.jsonl" description 8 64 ''
run_pair gqa "${DATA_ROOT}/gqa/gqa_val_balanced_sft_msswift.jsonl" vqa 24 16 ''
run_pair vqav2 "${DATA_ROOT}/VQAv2/vqav2_validation_sft_msswift.jsonl" vqa 24 16 ''
run_pair robovqa "${DATA_ROOT}/robovqa/robovqa_train_sft_eval.jsonl" vqa 1 192 32

printf '[%s] custom evaluation queue finished\n' "$(timestamp)"
if [[ -z "${ONLY_DATASET}" && "${SKIP_OFFICIAL:-0}" != 1 ]]; then
    bash "${REPO_ROOT}/scripts/benchmark-tool/run_official_eval_queue.sh"
fi
