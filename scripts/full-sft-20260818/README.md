# Stage-1 正式全量训练脚本（2026-08-18 整理）

## 推荐配置

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

48 卡等效 global batch：

```text
8 * 8 * (48 / 4) = 768
```

不要直接使用 per_device_train_batch_size=16：

- bs16 smoke 在 300Gi 宿主机内存下 DataLoader worker 被 OOM kill，未进入训练。
- bs8 smoke 稳定完成，显存峰值约 96 GiB，GPU 利用率约 78.5%。
- 若想等效 bs16，使用 bs8 + ga2，而不是改成 per_device_train_batch_size=16。

## 启动

```bash
export WORLD_SIZE=3 RANK=0 MASTER_ADDR=xxx MASTER_PORT=29500
bash /mnt/luojunkun/stage1/ms-swift/scripts/full-sft-20260818/run_dlc_48card_full_sft.bs8.ga8.20260818.sh
```

每个 DLC 节点分别设置自己的 `RANK=0/1/2`。

## 文件

- `run_dlc_48card_full_sft.bs8.ga8.20260818.sh`：推荐正式版本。
- `stage1_run_dlc3_full_sft.sh`：旧版三节点入口，可对照。
- `stage1_run_dlc_single_node_sft.sh`：单节点全量入口，可对照。
- `dlc_ready_entrypoints.stage1.json`：全量 ready manifest 副本。
- `README_dlc48_stage1_sft.md`：2026-08-12 全量审计与训练说明。

## 视频加载

全量 3795 个视频建议先打成 tar 放 OSS，DLC 启动后在节点本地解压，再训练；
不要逐文件从 OSS 冷读。当前 smoke 使用 `/dev/shm` 本地视频副本。
