#!/usr/bin/env bash
set -u

: "${WORLD_SIZE:?DLC must provide WORLD_SIZE, expected 3}"
: "${RANK:?DLC must provide RANK, expected 0/1/2}"

DLC_WORLD_SIZE=${WORLD_SIZE}
DLC_RANK=${RANK}

STAGE_ROOT=${STAGE_ROOT:-/mnt/luojunkun/stage1}
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-${STAGE_ROOT}/ms-swift}
MODEL=${MODEL:-${STAGE_ROOT}/model}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-${MS_SWIFT_ROOT}/scripts/dlc_ready_entrypoints.stage1.json}
ABLATION_ROOT=${ABLATION_ROOT:-${STAGE_ROOT}/tmp/task_type_ablation_20260813}
SAMPLE_PER_DATASET=${SAMPLE_PER_DATASET:-512}
SAMPLE_SEED=${SAMPLE_SEED:-20260813}

if [[ "${DLC_WORLD_SIZE}" != "3" ]]; then
  echo "Expected WORLD_SIZE=3 for this ablation, got WORLD_SIZE=${DLC_WORLD_SIZE}" >&2
  exit 2
fi

case "${DLC_RANK}" in
  0)
    EXP_NAME=rank0_vqa_linear_lr1e-5
    TASK_GROUP=vqa
    ENABLED_DATASETS="ai2d chartqa gqa textvqa visualgenome-qa vqav2 robo2vlm robovqa"
    ;;
  1)
    EXP_NAME=rank1_description_linear_lr1e-5
    TASK_GROUP=description
    ENABLED_DATASETS="llava pixmo-cap coco visualgenome-regions"
    ;;
  2)
    EXP_NAME=rank2_grounding_video_robotics_linear_lr1e-5
    TASK_GROUP=grounding_video_robotics
    ENABLED_DATASETS="vlm-r1 spatialvlm pixmo-points molmo2-video-track robo2vlm robovqa"
    ;;
  *)
    echo "Unexpected RANK=${DLC_RANK}; expected 0, 1, or 2" >&2
    exit 2
    ;;
esac

LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-linear}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
MAX_STEPS=${MAX_STEPS:-500}

SAMPLED_DIR=${ABLATION_ROOT}/${EXP_NAME}/sampled_jsonl_seed${SAMPLE_SEED}_n${SAMPLE_PER_DATASET}
SAMPLED_MANIFEST=${ABLATION_ROOT}/${EXP_NAME}/sampled_manifest_seed${SAMPLE_SEED}_n${SAMPLE_PER_DATASET}.json
OUTPUT_DIR=${OUTPUT_DIR:-${STAGE_ROOT}/sft-model/task-type-ablation-20260813/${EXP_NAME}}
LOG_DIR=${ABLATION_ROOT}/logs/${EXP_NAME}

export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-${STAGE_ROOT}/cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-128}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-16}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export NPROC_PER_NODE=16
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29610 + DLC_RANK))

# Make every DLC node a local 16-GPU job, not a 3-node DDP job.
unset WORLD_SIZE RANK LOCAL_WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CELOSS_PARALLEL_SIZE=${CELOSS_PARALLEL_SIZE:-2048}

mkdir -p "${SAMPLED_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}"

echo "HOSTNAME=${HOSTNAME}"
echo "DLC_WORLD_SIZE=${DLC_WORLD_SIZE}"
echo "DLC_RANK=${DLC_RANK}"
echo "EXP_NAME=${EXP_NAME}"
echo "TASK_GROUP=${TASK_GROUP}"
echo "ENABLED_DATASETS=${ENABLED_DATASETS}"
echo "LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "SAMPLED_MANIFEST=${SAMPLED_MANIFEST}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

cd "${MS_SWIFT_ROOT}" || exit 2

python3 - "${SOURCE_MANIFEST}" "${SAMPLED_DIR}" "${SAMPLED_MANIFEST}" "${SAMPLE_PER_DATASET}" "${SAMPLE_SEED}" "${ENABLED_DATASETS}" <<'PY'
import json
import random
import sys
from pathlib import Path

source_manifest = Path(sys.argv[1])
sampled_dir = Path(sys.argv[2])
sampled_manifest = Path(sys.argv[3])
n = int(sys.argv[4])
seed = int(sys.argv[5])
enabled = set(sys.argv[6].split())

manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
sampled_dir.mkdir(parents=True, exist_ok=True)

out = {"datasets": {}}
rng = random.Random(seed)

for name, cfg in manifest.get("datasets", {}).items():
    new_cfg = dict(cfg)
    if name not in enabled or not cfg.get("enabled", True):
        new_cfg["enabled"] = False
        out["datasets"][name] = new_cfg
        continue

    src = Path(cfg["train"])
    if not src.exists():
        raise FileNotFoundError(src)

    lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) <= n:
        indices = list(range(len(lines)))
    else:
        indices = sorted(rng.sample(range(len(lines)), n))

    chosen = [lines[i] for i in indices]
    dst = sampled_dir / f"{name}.sample{len(chosen)}.seed{seed}.jsonl"
    dst.write_text("\n".join(chosen) + "\n", encoding="utf-8")

    new_cfg["train"] = str(dst)
    new_cfg["enabled"] = True
    new_cfg["sample_source"] = str(src)
    new_cfg["sample_seed"] = seed
    new_cfg["sample_size"] = len(chosen)
    new_cfg["sample_indices_head"] = indices[:20]
    out["datasets"][name] = new_cfg
    print(f"[sample] {name}: {len(chosen)}/{len(lines)} -> {dst}")

sampled_manifest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[sample] manifest -> {sampled_manifest}")
PY

python -c 'import swift; print("swift package:", swift.__file__)' || exit 2
python -c 'import torch; n=torch.cuda.device_count(); print("visible GPU count:", n); assert n == 16' || exit 2
python - <<'PY'
try:
    import flash_attn
    print("flash_attn:", flash_attn.__version__)
except Exception as e:
    print("flash_attn check failed:", repr(e))
PY

DATASET_LIST=$(python3 - "${SAMPLED_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for config in manifest["datasets"].values():
    if config.get("enabled", True):
        print(config["train"])
PY
)

set +e
swift sft \
  --model "${MODEL}" \
  --model_type qwen3_vl \
  --tuner_type full \
  --freeze_llm false \
  --freeze_vit false \
  --freeze_aligner false \
  --deepspeed zero2 \
  --dataset ${DATASET_LIST} \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --torch_dtype bfloat16 \
  --attn_impl flash_attention_2 \
  --max_steps "${MAX_STEPS}" \
  --save_strategy no \
  --per_device_train_batch_size 8 \
  --learning_rate "${LEARNING_RATE}" \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing true \
  --padding_free true \
  --packing false \
  --sequence_parallel_size 8 \
  --logging_steps 1 \
  --max_length 50000 \
  --truncation_strategy right \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --dataloader_persistent_workers true \
  --dataloader_prefetch_factor 2 \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "task-type-ablation-${EXP_NAME}" \
  --report_to wandb 2>&1 | tee "${LOG_DIR}/train.log"
status=${PIPESTATUS[0]}

echo "swift exit status: ${status}" | tee -a "${LOG_DIR}/train.log"
exit "${status}"
