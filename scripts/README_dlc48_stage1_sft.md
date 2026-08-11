# Stage 1 DLC 48-card full SFT

This run uses the 16 enabled entries in
`dlc_ready_entrypoints.stage1.json`. It does not use any `*_global_train.jsonl`
file. The ready entrypoints remove exact duplicate rows and invalid real
bounding boxes, then globally reserve eval media while retaining shared
cross-dataset train media and its distinct supervision.

## Required gates

Run these after decontamination and the DLC audit complete:

```bash
python3 /mnt/workspace/stage1/scripts/validate_sft_entrypoints.py \
  --manifest /mnt/workspace/stage1/scripts/dlc_ready_entrypoints.stage1.json

python3 - <<'PY'
import json
p = "/mnt/luojunkun/stage1/dataset_ms-swift/dlc_sft_audit_report.json"
r = json.load(open(p, encoding="utf-8"))
assert r["status"] == "passed", r["hard_failures"]
assert r["decontamination_verification"]["train_eval_overlap_rows"] == 0
assert r["totals"]["schema_errors"] == 0
assert r["totals"]["exact_duplicate_rows"] == 0
print(r["totals"])
PY
```

## Verified ready population

The final audit on 2026-08-12 passed for 9,181,393 training rows and
738,770 reserved eval rows. It found zero schema errors, zero exact duplicate
rows, zero train/eval media overlap, and zero grounding visualization errors.
The non-global policy retained 4,200,782 rows whose media is also used by a
different training dataset.

| Dataset | Train | Eval |
| --- | ---: | ---: |
| AI2D | 717 | 79 |
| ChartQA | 28,268 | 4,420 |
| GQA | 907,276 | 131,968 |
| TextVQA | 34,588 | 5,000 |
| VisualGenome QA | 1,257,817 | 14,686 |
| VisualGenome Regions | 2,901,093 | 34,493 |
| VLM-R1 | 230,676 | 29,826 |
| VQAv2 | 368,095 | 211,165 |
| Robo2VLM | 614,282 | 70,408 |
| RoboVQA | 1,025,741 | 112,770 |
| LLaVA | 319,065 | 39,293 |
| SpatialVLM | 78,458 | 8,641 |
| PixMo Cap | 311,010 | 16,232 |
| PixMo Points | 1,015,196 | 52,998 |
| Molmo2 VideoTrack | 23,567 | 1,011 |
| COCO | 65,544 | 5,780 |

The training task distribution is 51.67% VQA/instruction, 31.54% grounded
description, 9.35% point grounding, 4.08% caption/description, 2.65% bbox
grounding, 0.25% video tracking, 0.39% text-only, and 0.07% box-to-text.
The machine-readable reports are:

- `/mnt/luojunkun/stage1/dataset_ms-swift/dlc_ready_decontamination_report.json`
- `/mnt/luojunkun/stage1/dataset_ms-swift/dlc_sft_audit_report.json`
- `/mnt/luojunkun/stage1/dataset_ms-swift/dlc_sft_audit_visualizations/`

## DLC launch command

The DLC job should have 6 nodes with 8 GPUs per node. DLC injects
`WORLD_SIZE`, `RANK`, and `MASTER_ADDR`. Set the shared input checkpoint and
output path in the job command:

```bash
export MODEL_PATH=/shared/path/to/Qwen3-VL-4B-Instruct-or-your-full-checkpoint
export OUTPUT_DIR=/mnt/luojunkun/stage1/sft-model/qwen3-vl-stage1-dlc48-full
bash /mnt/workspace/stage1/scripts/run_dlc_48card_full_sft.sh
```

The defaults are full-parameter BF16, ZeRO-3, 8-way sequence parallelism,
65,536 tokens, one epoch, and effective global batch 24
(`6` data-parallel groups times micro-batch `1` times accumulation `4`). Change
`GRADIENT_ACCUMULATION_STEPS` to adjust the global batch.

VideoTrack labels cover up to 128 source frames, and the converted clips use
6, 12, 20, or 25 FPS. The launch defaults therefore use `FPS=25`,
`FPS_MAX_FRAMES=128`, and `VIDEO_MAX_TOKEN_NUM=128` so that frame-indexed
targets are not trained against a 2 FPS/16-frame subsample. Do not lower these
values for the full run unless VideoTrack is removed from the manifest.

`MAX_LENGTH=65536` is selected because VideoTrack contains unusually long
frame-by-frame targets. Before the paid 48-card run, execute a short 8-card
smoke run in the exact DLC image to verify Qwen3-VL sequence parallelism,
FlashAttention, DeepSpeed, video decoding, and the longest-sample memory use.
Do not reduce `MAX_LENGTH` without first measuring token lengths with the exact
model processor. `truncation_strategy=right` keeps records instead of silently
dropping overlength samples, but an overlength target would still lose its tail.

Use the same script for the required one-node 8-card smoke test before submitting
the full job:

```bash
export WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export MODEL_PATH=/shared/path/to/model
export EXPECTED_WORLD_SIZE=8 MAX_STEPS=2 SAVE_STEPS=2
export OUTPUT_DIR=/mnt/luojunkun/stage1/sft-model/dlc48-preflight-smoke
bash /mnt/workspace/stage1/scripts/run_dlc_48card_full_sft.sh
```

Run this in the exact 48-card DLC image, not in the current data-processing
container. The latter has a broken accelerator runtime (`libhggcrt1.so` is
missing) and is not valid evidence that the paid training image can launch.

The training command deliberately disables inline evaluation. The reserved eval
population is large and should be evaluated separately through the benchmark
queue instead of repeatedly consuming training time.
