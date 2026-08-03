#!/usr/bin/env bash

set -uo pipefail

STAGE_ROOT=/mnt/luojunkun/stage1
REPO_ROOT=/mnt/workspace/stage1
CONTROL_ROOT=${STAGE_ROOT}/benchmark-eval-result/stage1-autoeval
SOURCE_ROOT=${SOURCE_ROOT_OVERRIDE:-${STAGE_ROOT}/benchmark-stage1}
PREPARED_ROOT=${PREPARED_ROOT_OVERRIDE:-${CONTROL_ROOT}/prepared}
RESULT_ROOT=${RESULT_ROOT_OVERRIDE:-${CONTROL_ROOT}/official-local}
BASELINE_WEIGHTS=${STAGE_ROOT}/model
OURS_WEIGHTS=${STAGE_ROOT}/sft-model/qwen3-vl-instruct4b/v30-20260728-003837/checkpoint-400
TOOL_ROOT=${REPO_ROOT}/scripts/benchmark-tool
CLI=${TOOL_ROOT}/cli.py
PREPARE=${TOOL_ROOT}/prepare_local_benchmarks.py
SCORER=${TOOL_ROOT}/score_local_benchmark.py
SCORER_PYTHON=${SCORER_PYTHON:-/mnt/workspace/benchmark-eval-venv/bin/python}
ONLY_BENCHMARK=${1:-}
ONLY_MODEL=${ONLY_MODEL:-}
EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0}
MIN_FREE_MB=${MIN_FREE_MB:-30000}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-30}
MONITOR_INTERVAL_SECONDS=${MONITOR_INTERVAL_SECONDS:-15}
PREPARE_LIMIT=${PREPARE_LIMIT:-}
PREPARE_OVERWRITE=${PREPARE_OVERWRITE:-0}
ATTENTION_IMPLEMENTATION=${ATTENTION_IMPLEMENTATION:-flash_attention_2}
VIDEO_BACKEND=${VIDEO_BACKEND:-decord}

if [[ -d /usr/local/PPU_SDK/lib ]]; then
    export LD_LIBRARY_PATH="/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -n "${ONLY_MODEL}" && "${ONLY_MODEL}" != baseline && "${ONLY_MODEL}" != ours ]]; then
    printf 'ONLY_MODEL must be baseline or ours, got: %s\n' "${ONLY_MODEL}" >&2
    exit 2
fi

LOG_ROOT=${CONTROL_ROOT}/logs/official
STATUS_FILE=${CONTROL_ROOT}/official-status.tsv
GPU_LOG=${CONTROL_ROOT}/gpu-memory.csv
mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}" "${CONTROL_ROOT}"

if [[ ! -f "${STATUS_FILE}" ]]; then
    printf 'timestamp\tbenchmark\tmodel\tstate\texit_code\toutput_dir\n' > "${STATUS_FILE}"
fi
if [[ ! -f "${GPU_LOG}" ]]; then
    printf 'timestamp,dataset,index,memory_total_mb,memory_used_mb,memory_free_mb,gpu_util_percent\n' > "${GPU_LOG}"
fi

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

json_status_is_clean() {
    local score_file=$1
    [[ -f "${score_file}" ]] && python -c \
        'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if d.get("status") == "complete" and all(int(r.get("failed_samples", 0)) == 0 for r in d.get("scores", [])) else 1)' \
        "${score_file}" 2>/dev/null
}

gpu_free_mb() {
    if command -v gpustat >/dev/null 2>&1; then
        gpustat --json 2>/dev/null | python -c \
            'import json,sys; g=json.load(sys.stdin)["gpus"][0]; print(int(g["memory.total"])-int(g["memory.used"]))'
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' '
        return
    fi
    return 1
}

wait_for_memory() {
    local minimum_free_mb=$1
    local free_mb
    while true; do
        if ! free_mb=$(gpu_free_mb); then
            free_mb=unknown
        fi
        if [[ "${free_mb}" =~ ^[0-9]+$ ]] && (( free_mb >= minimum_free_mb )); then
            printf '[%s] GPU gate open: free=%s MB, required=%s MB\n' \
                "$(timestamp)" "${free_mb}" "${minimum_free_mb}"
            return
        fi
        printf '[%s] waiting for GPU memory: free=%s MB, required=%s MB\n' \
            "$(timestamp)" "${free_mb:-unknown}" "${minimum_free_mb}"
        sleep "${WAIT_INTERVAL_SECONDS}"
    done
}

monitor_gpu() {
    local benchmark=$1
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
        if command -v gpustat >/dev/null 2>&1; then
            gpustat --json 2>/dev/null | python -c \
                'import json,sys; ts,label=sys.argv[1:3]; data=json.load(sys.stdin); [print("{},{},{},{},{},{},{}".format(ts,label,g.get("index"),g.get("memory.total"),g.get("memory.used"),int(g.get("memory.total",0))-int(g.get("memory.used",0)),g.get("utilization.gpu"))) for g in data.get("gpus",[])]' \
                "$(timestamp)" "official_${benchmark}" >> "${GPU_LOG}" || true
        elif command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu \
                --format=csv,noheader,nounits -i 0 | \
                while IFS= read -r row; do
                    printf '%s,official_%s,%s\n' "$(timestamp)" "${benchmark}" "${row// /}" >> "${GPU_LOG}"
                done
        fi
        sleep "${MONITOR_INTERVAL_SECONDS}"
    done
}

prepare_benchmark() {
    local benchmark=$1
    local log_file=${LOG_ROOT}/${benchmark}-prepare-$(date -u '+%Y%m%dT%H%M%SZ').log
    printf '[%s] preparing local benchmark: %s\n' "$(timestamp)" "${benchmark}"
    local command=(
        python "${PREPARE}" "${benchmark}"
        --source-root "${SOURCE_ROOT}"
        --output-root "${PREPARED_ROOT}"
    )
    if [[ -n "${PREPARE_LIMIT}" ]]; then
        command+=(--limit "${PREPARE_LIMIT}")
    fi
    if [[ "${PREPARE_OVERWRITE}" == 1 ]]; then
        command+=(--overwrite)
    fi
    "${command[@]}" > "${log_file}" 2>&1
}

run_one() {
    local benchmark=$1
    local task=$2
    local batch_size=$3
    local max_tokens=$4
    local max_video_frames=$5
    local max_image_pixels=$6
    local model_label=$7
    local weights=$8
    local val_file=${PREPARED_ROOT}/${benchmark}/eval.jsonl
    local output_dir=${RESULT_ROOT}/${model_label}/${benchmark}
    local log_file=${LOG_ROOT}/${benchmark}-${model_label}-$(date -u '+%Y%m%dT%H%M%SZ').log

    if [[ -f "${output_dir}/official_result.json" ]] && \
            json_status_is_clean "${output_dir}/scores.json"; then
        printf '[%s] skip scored: %s/%s\n' "$(timestamp)" "${benchmark}" "${model_label}"
        printf '%s\t%s\t%s\tskipped_complete\t0\t%s\n' \
            "$(timestamp)" "${benchmark}" "${model_label}" "${output_dir}" >> "${STATUS_FILE}"
        return 0
    fi
    mkdir -p "${output_dir}"
    if ! json_status_is_clean "${output_dir}/scores.json"; then
        wait_for_memory "${MIN_FREE_MB}"
        printf '[%s] start: %s/%s batch=%s tokens=%s\n' \
            "$(timestamp)" "${benchmark}" "${model_label}" "${batch_size}" "${max_tokens}"
        printf '%s\t%s\t%s\tstarted\t\t%s\n' \
            "$(timestamp)" "${benchmark}" "${model_label}" "${output_dir}" >> "${STATUS_FILE}"
        local command=(
            python "${CLI}" eval-custom
            --val-dataset "${val_file}"
            --model-weights "${weights}"
            --model_type qwen3_vl
            --model-name "${model_label}"
            --dataset-name "${benchmark}"
            --task "${task}"
            --output-dir "${output_dir}"
            --batch-size "${batch_size}"
            --max-new-tokens "${max_tokens}"
            --dtype bfloat16
            --device cuda:0
            --attn-implementation "${ATTENTION_IMPLEMENTATION}"
            --progress-every 100
            --continue-on-error
            --resume
            --retry-errors
        )
        if [[ -n "${max_video_frames}" ]]; then
            command+=(--max-video-frames "${max_video_frames}")
            command+=(--video-backend "${VIDEO_BACKEND}")
        fi
        if [[ "${benchmark}" == egoplan ]]; then
            command+=(--allow-empty-video-fallback)
            command+=(--min-video-frames "${EGOPLAN_MIN_VIDEO_FRAMES:-8}")
        fi
        if [[ "${benchmark}" == openeqa && "${OPENEQA_GROUP_VIDEO_BATCHES:-1}" == 1 ]]; then
            command+=(--group-video-batches)
        fi
        if [[ "${benchmark}" == flickr30k && "${FLICKR30K_GROUP_IMAGE_BATCHES:-0}" == 1 ]]; then
            command+=(--group-image-batches)
            if [[ -n "${FLICKR30K_BATCHED_IMAGE_GRIDS:-}" ]]; then
                command+=(--batched-image-grids "${FLICKR30K_BATCHED_IMAGE_GRIDS}")
            fi
            if [[ -n "${FLICKR30K_IMAGE_GRID_FALLBACK:-}" ]]; then
                command+=(--image-grid-fallback "${FLICKR30K_IMAGE_GRID_FALLBACK}")
            fi
            if [[ -n "${FLICKR30K_MAX_BATCHED_IMAGE_GRID_AREA:-}" ]]; then
                command+=(--max-batched-image-grid-area "${FLICKR30K_MAX_BATCHED_IMAGE_GRID_AREA}")
            fi
        fi
        if [[ -n "${max_image_pixels}" ]]; then
            command+=(--max-image-pixels "${max_image_pixels}")
        fi
        CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" "${command[@]}" > "${log_file}" 2>&1
        local eval_exit=$?
        if (( eval_exit != 0 )); then
            printf '%s\t%s\t%s\teval_failed\t%s\t%s\n' \
                "$(timestamp)" "${benchmark}" "${model_label}" "${eval_exit}" "${output_dir}" >> "${STATUS_FILE}"
            return "${eval_exit}"
        fi
        if ! json_status_is_clean "${output_dir}/scores.json"; then
            printf '%s\t%s\t%s\teval_incomplete\t1\t%s\n' \
                "$(timestamp)" "${benchmark}" "${model_label}" "${output_dir}" >> "${STATUS_FILE}"
            printf '[%s] eval_incomplete: %s/%s has failed samples\n' \
                "$(timestamp)" "${benchmark}" "${model_label}"
            return 1
        fi
    fi

    "${SCORER_PYTHON}" "${SCORER}" "${benchmark}" --predictions "${output_dir}/predictions.jsonl" \
        --output-file "${output_dir}/official_result.json" >> "${log_file}" 2>&1
    local score_exit=$?
    local state=scored
    if (( score_exit != 0 )); then
        state=score_failed
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(timestamp)" "${benchmark}" "${model_label}" "${state}" "${score_exit}" "${output_dir}" >> "${STATUS_FILE}"
    printf '[%s] %s: %s/%s exit=%s\n' \
        "$(timestamp)" "${state}" "${benchmark}" "${model_label}" "${score_exit}"
    return "${score_exit}"
}

run_with_monitor() {
    local benchmark=$1
    run_one "$@" &
    local eval_pid=$!
    monitor_gpu "${benchmark}" "${eval_pid}" &
    local monitor_pid=$!
    local exit_code=0
    wait "${eval_pid}" || exit_code=$?
    wait "${monitor_pid}" || true
    return "${exit_code}"
}

run_pair() {
    local benchmark=$1
    local task=$2
    local batch_size=$3
    local max_tokens=$4
    local max_video_frames=$5
    local max_image_pixels=$6

    if [[ -n "${ONLY_BENCHMARK}" && "${ONLY_BENCHMARK}" != "${benchmark}" ]]; then
        return
    fi
    if [[ "${SKIP_PREPARE:-0}" != 1 ]] && ! prepare_benchmark "${benchmark}"; then
        printf '%s\t%s\tall\tprepare_failed\t2\t%s\n' \
            "$(timestamp)" "${benchmark}" "${PREPARED_ROOT}/${benchmark}" >> "${STATUS_FILE}"
        return
    fi

    if [[ -z "${ONLY_MODEL}" || "${ONLY_MODEL}" == baseline ]]; then
        run_with_monitor "${benchmark}" "${task}" "${batch_size}" "${max_tokens}" "${max_video_frames}" \
            "${max_image_pixels}" baseline "${BASELINE_WEIGHTS}" || true
    fi
    if [[ -z "${ONLY_MODEL}" || "${ONLY_MODEL}" == ours ]]; then
        run_with_monitor "${benchmark}" "${task}" "${batch_size}" "${max_tokens}" "${max_video_frames}" \
            "${max_image_pixels}" ours "${OURS_WEIGHTS}" || true
    fi
}

if [[ -z "${ONLY_BENCHMARK}" && "${SKIP_CUSTOM_AUDIT:-0}" != 1 ]]; then
    printf '[%s] auditing failed custom-eval samples before official benchmarks\n' "$(timestamp)"
    SKIP_OFFICIAL=1 bash "${TOOL_ROOT}/run_custom_eval_queue.sh"
fi

run_pair video_mme vqa 3 8 32 262144
run_pair ocrbench_v2 vqa 32 128 '' 1048576
run_pair robospatial vqa 32 96 '' 1048576
run_pair egoplan vqa "${EGOPLAN_BATCH_SIZE:-1}" 8 32 262144
run_pair openeqa description "${OPENEQA_BATCH_SIZE:-2}" 64 32 262144
run_pair flickr30k description "${FLICKR30K_BATCH_SIZE:-32}" 64 '' 524288

if [[ -n "${ONLY_BENCHMARK}" ]]; then
    printf '[%s] targeted benchmark queue finished: %s/%s\n' \
        "$(timestamp)" "${ONLY_BENCHMARK}" "${ONLY_MODEL:-both}"
    exit 0
fi

if ! python "${TOOL_ROOT}/validate_stage1_results.py" --result-root "${RESULT_ROOT}"; then
    printf '[%s] official/local benchmark validation failed; report not generated\n' "$(timestamp)"
    exit 3
fi

printf '[%s] official/local benchmark queue finished\n' "$(timestamp)"
python "${TOOL_ROOT}/generate_stage1_report.py" \
    > "${LOG_ROOT}/report-$(date -u '+%Y%m%dT%H%M%SZ').log" 2>&1
