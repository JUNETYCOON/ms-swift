#!/usr/bin/env bash

set -uo pipefail

STAGE_ROOT=/mnt/luojunkun/stage1
TOOL_ROOT=/mnt/workspace/stage1/scripts/benchmark-tool
CONTROL_ROOT=${STAGE_ROOT}/benchmark-eval-result/stage1-autoeval
RESULT_ROOT=${CONTROL_ROOT}/official-local
LOG_ROOT=${CONTROL_ROOT}/logs/parallel-resume
SCORER_PYTHON=${SCORER_PYTHON:-/mnt/workspace/benchmark-eval-venv/bin/python}

mkdir -p "${LOG_ROOT}"

pids=()
labels=()

launch() {
    local benchmark=$1
    local model=$2
    local gpu=$3
    local label=${benchmark}-${model}-gpu${gpu}
    local log_file=${LOG_ROOT}/${label}.log
    printf '[parallel-resume] launching %s\n' "${label}"
    ONLY_MODEL="${model}" \
    EVAL_CUDA_VISIBLE_DEVICES="${gpu}" \
    MIN_FREE_MB=0 \
    PREPARED_ROOT_OVERRIDE=/tmp/stage1-official-prepared \
    SKIP_PREPARE=1 \
    SKIP_CUSTOM_AUDIT=1 \
    SCORER_PYTHON="${SCORER_PYTHON}" \
        bash "${TOOL_ROOT}/run_official_eval_queue.sh" "${benchmark}" > "${log_file}" 2>&1 &
    pids+=("$!")
    labels+=("${label}")
}

launch robospatial ours 0
launch egoplan baseline 1
launch egoplan ours 2
launch openeqa baseline 3
launch openeqa ours 4
launch flickr30k baseline 5
launch flickr30k ours 6

for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf '[parallel-resume] process exited: %s\n' "${labels[$index]}"
    else
        printf '[parallel-resume] process failed: %s\n' "${labels[$index]}" >&2
    fi
done

if ! python "${TOOL_ROOT}/validate_stage1_results.py" --result-root "${RESULT_ROOT}"; then
    printf '[parallel-resume] validation failed; report not generated\n' >&2
    exit 3
fi

python "${TOOL_ROOT}/generate_stage1_report.py"
