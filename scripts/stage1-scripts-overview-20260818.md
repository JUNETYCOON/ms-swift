# Stage1 脚本整理与性能结论（2026-08-18）

## 去重结果

对比迁移副本与 `/mnt/luojunkun/stage1` 原目录后，已删除 9 个 MD5 完全一致的
迁移副本，原目录均未删除：

```text
/mnt/luojunkun/stage1/scripts/workspace-20260818/
rchive//d
  audit_dlc_sft.py
  audit_molmo2_videotrack_swift.py
  dlc_non_global_entrypoints.stage1.json
  dlc_ready_entrypoints.stage1.json
  global_media_dedup.py
  materialize_molmo2_videotrack_media.py
  sanitize_sft_jsonl.py
  test_audit_molmo2_videotrack_swift.py
  test_materialize_molmo2_videotrack_media.py
```

工具仓库 `lmms-eval`、`open-eqa`、`personpath22-tracking-dataset`、
`benchmark-analysis`、`benchmark-report-tools` 与 `VLMEvalKit` 不是同一仓库，
未按重名清理。

## Smoke 目录

```text
/mnt/luojunkun/stage1/ms-swift/scripts/smoke-videotrack-20260818/
rchive//d
  README.md
  scripts/stage1_mem_smoke_videotrack_bs8_sample300_v10.sh
rchive//d
  scripts/stage1_mem_smoke_single_node_16bs_fixed.sh
rchive//d
  data-process/smoke/sample_norm1000_smoke_300.py
rchive//d
  data-process/smoke/prepare_local_video_dataset.py
rchive//d
  scripts/point_accuracy_formula.py
rchive//d
  scripts/point_accuracy_from_trace.py
rchive//d
  results/videotrack-bs8-sample300-v8.result.json
  results/videotrack-bs2-ml30000.result.json
```

smoke 数据统一放在：

```text
/mnt/luojunkun/stage1/dataset_ms-swift/smoke-data/
  ready_train_norm1000_sample300.jsonl
  video_list.txt
```

## 正式全量训练目录

```text
/mnt/luojunkun/stage1/ms-swift/scripts/full-sft-20260818/
rchive//d
  README.md
  run_dlc_48card_full_sft.bs8.ga8.20260818.sh
  dlc_ready_entrypoints.stage1.json
  stage1_run_dlc3_full_sft.sh
  stage1_run_dlc_single_node_sft.sh
  README_dlc48_stage1_sft.md
```

推荐正式版本已同时安装到：

```text
/mnt/luojunkun/stage1/ms-swift/scripts/run_dlc_48card_full_sft.sh
rchive//d
```

旧版 ZeRO-3 / bs1 历史快照已删除（2026-08-18）：

```text
rchive//d
```

旧的 `/mnt/luojunkun/stage1/ms-swift/scripts/smoke-813` 已删除，新 smoke
rchive//d
脚本已迁移到上述 `smoke-videotrack-20260818` 目录。

2026-08-18 已进一步清理：

- 删除旧 4-GPU smoke 输出 `sft-model/smoke-qwen3vl-4gpu`（约 602G）。
- 删除所有 `sft-model/mem-smoke-*`、`tmp/mem-smoke-*`、`smoke-data` 和
  `tmp/bbox-only-ciou-smoke-300step-20260814`。
rchive//d
rchive//d

## 最高性能配置

```text
per_device_train_batch_size=8
gradient_accumulation_steps=8
sequence_parallel_size=4
max_length=50000
FPS_MAX_FRAMES=128
dataset_num_proc=4
dataloader_num_workers=1
dataloader_persistent_workers=false
deepspeed=zero2
packing=false
```

实测：

| 配置 | 结果 | 显存峰值 MiB | GPU 利用率 |
| --- | --- | ---: | ---: |
| bs8 / ga1 / ml30000 | 300 step 有 loss，loss_last=0.1521 | 96123 | 78.5% |
| bs16 / ga1 / ml50000 | 未进入训练，宿主机 300Gi 内存 OOM | - | - |

结论：不使用 `per_device_train_batch_size=16`。若需要等效更大 global batch，
优先 bs8 + ga8（48 卡等效 768），或 bs8 + ga2 等效 bs16 的 micro 吞吐。
