# Molmo2-VideoTrack bs8 smoke（2026-08-18 整理）

## 抽样数据

- `/mnt/luojunkun/stage1/dataset_ms-swift/smoke-data/ready_train_norm1000_sample300.jsonl`：
  固定 seed=20260817 的 300 条 norm1000 抽样。
- `/mnt/luojunkun/stage1/dataset_ms-swift/smoke-data/video_list.txt`：
  对应视频相对路径列表。

## 推荐脚本

`scripts/stage1_mem_smoke_videotrack_bs8_sample300_v10.sh` 是当前功能最全的
smoke 入口：

- bs8 / ga1 / sp4 / ml50000 / FPS_MAX_FRAMES=128
- dataloader_num_workers=1，dataloader_persistent_workers=false
- 关闭 checkpoint 保存，保留 batch trace
- 每步写 point_acc，结果落在 `results/pointacc-per-step/`

底层训练脚本：`scripts/stage1_mem_smoke_single_node_16bs_fixed.sh`。

## 实测结论

| 配置 | 状态 | loss_last | 显存峰值 MiB | GPU 利用率 |
| --- | --- | ---: | ---: | ---: |
| bs2 / ga1 / ml30000 | 完成 300 step | 0.1433 | 41646 | 81.1% |
| bs8 / ga1 / ml30000 | 完成，300 step 有 loss | 0.1521 | 96123 | 78.5% |
| bs16 / ga1 / ml50000 | 未进入训练，宿主机 OOM | - | - | - |

推荐：单节点 smoke 用 bs8 + ga1；正式全量用 bs8 + ga8。

## 依赖脚本

- `data-process/smoke/sample_norm1000_smoke_300.py`：抽样
- `data-process/smoke/prepare_local_video_dataset.py`：把视频复制到本地后重写 jsonl
- `point_accuracy_formula.py` / `point_accuracy_from_trace.py`：逐 step point_acc
- `dlc_create_mem800_smoke_videotrack_bs8_sample300_v10.py`：DLC 任务创建脚本
