#!/usr/bin/env bash

set -uo pipefail

STAGE_ROOT=/mnt/luojunkun/stage1
TOOL_ROOT=/mnt/workspace/stage1/scripts/benchmark-tool
CONTROL_ROOT=${STAGE_ROOT}/benchmark-eval-result/stage1-autoeval
SMOKE_LIMIT=${SMOKE_LIMIT:-3}
SMOKE_PREPARED_ROOT=${CONTROL_ROOT}/smoke-prepared
SMOKE_RESULT_ROOT=${CONTROL_ROOT}/smoke-results
STATUS_FILE=${CONTROL_ROOT}/target-queue-status.tsv
LOCK_FILE=/tmp/stage1-target-benchmark-queue.lock

mkdir -p "${CONTROL_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    printf '[target-queue] another queue already holds %s\n' "${LOCK_FILE}"
    exit 0
fi

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

record_status() {
    printf '%s\t%s\n' "$(timestamp)" "$1" >> "${STATUS_FILE}"
}

smoke_is_clean() {
    python -c \
        'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); expected=int(sys.argv[2]); reports=[json.load(open(root/model/"video_mme"/"scores.json", encoding="utf-8")) for model in ("baseline","ours")]; ok=all(r.get("status")=="complete" and int(r.get("processed_samples",0))==expected and all(int(s.get("failed_samples",0))==0 for s in r.get("scores",[])) for r in reports) and all((root/model/"video_mme"/"official_result.json").is_file() for model in ("baseline","ours")); raise SystemExit(0 if ok else 1)' \
        "${SMOKE_RESULT_ROOT}" "${SMOKE_LIMIT}"
}

record_status "video_mme_smoke_prepare_started"
PREPARED_ROOT_OVERRIDE="${SMOKE_PREPARED_ROOT}" \
RESULT_ROOT_OVERRIDE="${SMOKE_RESULT_ROOT}" \
PREPARE_LIMIT="${SMOKE_LIMIT}" \
PREPARE_OVERWRITE=1 \
SKIP_CUSTOM_AUDIT=1 \
bash "${TOOL_ROOT}/run_official_eval_queue.sh" video_mme

if ! smoke_is_clean; then
    record_status "video_mme_smoke_failed"
    printf '[target-queue] Video-MME smoke test failed; full queue was not started.\n'
    exit 2
fi

record_status "video_mme_smoke_complete"
record_status "full_queue_started"
SKIP_CUSTOM_AUDIT=1 bash "${TOOL_ROOT}/run_official_eval_queue.sh"
queue_exit=$?
record_status "full_queue_finished_exit_${queue_exit}"
exit "${queue_exit}"
