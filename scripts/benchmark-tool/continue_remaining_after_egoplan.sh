#!/usr/bin/env bash

set -uo pipefail

WAIT_PID=${1:?usage: continue_remaining_after_egoplan.sh WAIT_PID}
TOOL_ROOT=/mnt/workspace/stage1/scripts/benchmark-tool
QUEUE=${TOOL_ROOT}/run_official_eval_queue.sh
VALIDATOR=${TOOL_ROOT}/validate_stage1_results.py
REPORTER=${TOOL_ROOT}/generate_stage1_report.py
LOG=/mnt/workspace/post-egoplan-official-gpu0.log

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
done

run_queue() {
    local benchmark=$1
    printf '[%s] starting %s\n' "$(timestamp)" "${benchmark}" >> "${LOG}"
    EVAL_CUDA_VISIBLE_DEVICES=0 \
        PREPARED_ROOT_OVERRIDE=/tmp/stage1-official-prepared \
        SKIP_PREPARE=1 \
        bash "${QUEUE}" "${benchmark}" >> "${LOG}" 2>&1
}

run_queue openeqa
run_queue flickr30k

printf '[%s] validating all official results\n' "$(timestamp)" >> "${LOG}"
if python "${VALIDATOR}" >> "${LOG}" 2>&1; then
    python "${REPORTER}" >> "${LOG}" 2>&1
    printf '[%s] validation and report generation complete\n' "$(timestamp)" >> "${LOG}"
else
    printf '[%s] strict validation failed; report generation skipped\n' "$(timestamp)" >> "${LOG}"
    exit 1
fi
