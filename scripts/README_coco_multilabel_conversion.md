# COCO-MODELSCOPE 转换说明

`prepare_coco_multilabel_swift.py` 将
`/mnt/luojunkun/stage1/dataset/COCO/COCO-MODELSCOPE` 中的 Hugging Face Arrow 数据清洗并转换为
ms-swift JSONL。

源数据边界：

- `train` 每行包含一张内嵌图片和 COCO 多标签 ID；
- 官方 `test` 只有图片，没有训练答案，因此只输出推理文件；
- 不从图片、文件名或模型预测中补造标签。

## 运行

先执行小规模检查：

```bash
python scripts/prepare_coco_multilabel_swift.py \
  --output-dir /mnt/luojunkun/stage1/dataset_ms-swift/coco-smoke \
  --max-rows-per-split 100 \
  --num-workers 2
```

全量转换：

```bash
python scripts/prepare_coco_multilabel_swift.py \
  --input-dir /mnt/luojunkun/stage1/dataset/COCO/COCO-MODELSCOPE \
  --output-dir /mnt/luojunkun/stage1/dataset_ms-swift/coco \
  --val-ratio 0.05 \
  --seed 42 \
  --num-workers 8
```

输出文件：

- `coco_train_sft_msswift.jsonl`：清洗、去重后的训练记录；
- `coco_val_sft_msswift.jsonl`：按媒体 SHA-256 确定性划分的验证记录；
- `coco_test_inference_msswift.jsonl`：官方无标签 test，仅用于推理；
- `media_groups.tsv`：每条源记录的媒体、lineage、划分与状态；
- `rejected.jsonl`：所有被拒绝或去重的源记录及原因；
- `schema_report.json`、`conversion_report.json`、`validation.json`：schema、全量对账与验证；
- `source-readonly-evidence.json`：转换前后源目录快照对比。

训练时只传入 SFT 文件：

```bash
swift sft \
  --model /path/to/model \
  --dataset /mnt/luojunkun/stage1/dataset_ms-swift/coco/coco_train_sft_msswift.jsonl \
  --val_dataset /mnt/luojunkun/stage1/dataset_ms-swift/coco/coco_val_sft_msswift.jsonl
```

`test` 文件末尾没有 assistant 回答，不得作为 SFT 数据使用。
