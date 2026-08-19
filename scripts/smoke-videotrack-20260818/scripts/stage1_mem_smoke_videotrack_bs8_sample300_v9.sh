#!/usr/bin/env bash
set -euo pipefail

# Molmo2-VideoTrack bs8 smoke on the fixed 300-row norm1000 sample.
# Video/model are copied to /dev/shm first; dataloader workers are reduced to
# 1 so decode memory does not OOM the 300Gi host memory.

export WANDB_PROJECT=ms-swift
export WANDB_RUN_ID=mem-smoke-videotrack-bs8-sample300-v8-20260817
export WANDB_RUN_NAME=mem-smoke-videotrack-bs8-sample300-v8-20260817
export RUN_TAG=mem-smoke-videotrack-bs8-sample300-v8-20260817
export OUTPUT_DIR=/mnt/luojunkun/stage1/sft-model/mem-smoke-videotrack-bs8-sample300-v8-20260817
export LOG_DIR=/mnt/luojunkun/stage1/tmp/mem-smoke-16bs-20260815/videotrack-bs8-sample300-v8

export PER_DEVICE_TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export MAX_LENGTH=50000
export LEARNING_RATE=4e-5
export FPS_MAX_FRAMES=128
export DATASET_NUM_PROC=4
export DATALOADER_NUM_WORKERS=1
export DATALOADER_PREFETCH_FACTOR=1
export DATALOADER_PERSISTENT_WORKERS=false
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=WARN

SOURCE_MODEL=/mnt/luojunkun/stage1/model
SOURCE_VIDEOS=/mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack/videos
SOURCE_TRAIN_JSONL=/mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack/norm1000/ready_train_norm1000.jsonl
SAMPLE_SCRIPT=/mnt/luojunkun/stage1/ms-swift/scripts/data-process/smoke/sample_norm1000_smoke_300.py
REWRITE_SCRIPT=/mnt/luojunkun/stage1/ms-swift/scripts/data-process/smoke/prepare_local_video_dataset.py
SAMPLE_COUNT=300
SAMPLE_SEED=20260817
SAMPLE_ROOT=/mnt/luojunkun/stage1/dataset_ms-swift/smoke-data
SAMPLED_TRAIN_JSONL=${SAMPLE_ROOT}/ready_train_norm1000_sample300.jsonl
LOCAL_VIDEO_LIST=${SAMPLE_ROOT}/video_list.txt
LOCAL_ROOT=/dev/shm/stage1_bs8_sample300
LOCAL_MODEL=${LOCAL_ROOT}/model-qwen3vl
LOCAL_VIDEOS=${LOCAL_ROOT}/videos
LOCAL_DATA=${LOCAL_ROOT}/data
FAIL_LOG=${LOCAL_DATA}/video_copy_failures.txt

mkdir -p "${LOCAL_ROOT}" "${LOCAL_DATA}" "${SAMPLE_ROOT}"

echo "[sample-prep] sampling ${SAMPLE_COUNT} rows from norm1000 with seed ${SAMPLE_SEED} ..."
python3 "${SAMPLE_SCRIPT}" \
  "${SOURCE_TRAIN_JSONL}" \
  "${SOURCE_VIDEOS}" \
  "${SAMPLE_COUNT}" \
  "${SAMPLE_SEED}" \
  "${SAMPLED_TRAIN_JSONL}" \
  "${LOCAL_VIDEO_LIST}"

if [[ ! -d "${LOCAL_MODEL}" ]]; then
  echo "[model-prep] copying main model files to ${LOCAL_MODEL} ..."
  MODEL_TMP=${LOCAL_ROOT}/model-qwen3vl.tmp.$$
  mkdir -p "${MODEL_TMP}"
  find "${SOURCE_MODEL}" -maxdepth 1 -type f -exec cp {} "${MODEL_TMP}"/ \;
  mv "${MODEL_TMP}" "${LOCAL_MODEL}"
  echo "[model-prep] done: $(du -sh "${LOCAL_MODEL}" | awk '{print $1}')"
fi
export MODEL="${LOCAL_MODEL}"
echo "[model-prep] MODEL=${MODEL}"

if [[ ! -f "${LOCAL_VIDEOS}/.copy_done" ]]; then
  echo "[video-prep] copying sampled videos to ${LOCAL_VIDEOS} with 16 parallel streams ..."
  VIDEOS_TMP=${LOCAL_ROOT}/videos.tmp.$$
  mkdir -p "${VIDEOS_TMP}"
  : > "${FAIL_LOG}"
  cd "${SOURCE_VIDEOS}"
  while IFS= read -r f; do
    f="${f%$'\r'}"
    [[ -z "${f}" ]] && continue
    printf '%s\0' "${f}"
  done < "${LOCAL_VIDEO_LIST}" | xargs -0 -P 16 -I{} bash -c 'f="$1"; d="$2"; out="$d/$f"; mkdir -p "$(dirname "$out")"; if ! timeout 120 cp "$f" "$out" 2>/dev/null; then echo "$f" >> "$3"; fi' _ {} "${VIDEOS_TMP}" "${FAIL_LOG}"
  cd -
  mv "${VIDEOS_TMP}" "${LOCAL_VIDEOS}"
  touch "${LOCAL_VIDEOS}/.copy_done"
  echo "[video-prep] copy finished, failed_files=$(wc -l < "${FAIL_LOG}")"
fi

LOCAL_TRAIN_JSONL=${LOCAL_DATA}/ready_train_norm1000_sample300_local.jsonl
LOCAL_MANIFEST=${LOCAL_DATA}/manifest_local.json
python3 "${REWRITE_SCRIPT}" \
  "${SAMPLED_TRAIN_JSONL}" \
  "${SOURCE_VIDEOS}" \
  "${LOCAL_VIDEOS}" \
  "${FAIL_LOG}" \
  "${LOCAL_TRAIN_JSONL}"
cat > "${LOCAL_MANIFEST}" <<'JSON'
{
  "description": "Molmo2-VideoTrack norm1000 300-row bs8 smoke local-video manifest.",
  "datasets": {
    "molmo2-video-track": {
      "enabled": true,
      "train": "/dev/shm/stage1_bs8_sample300/data/ready_train_norm1000_sample300_local.jsonl",
      "eval": "/mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack/norm1000/ready_eval_norm1000.jsonl"
    }
  }
}
JSON
export MANIFEST="${LOCAL_MANIFEST}"
echo "[video-prep] using local manifest: ${MANIFEST}"
echo "[video-prep] disk: $(df -h "${LOCAL_ROOT}" | tail -1)"

exec /mnt/luojunkun/stage1/ms-swift/scripts/smoke-videotrack-20260818/scripts/stage1_mem_smoke_single_node_16bs_fixed.sh
