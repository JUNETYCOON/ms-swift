# scripts/data-process

数据处理脚本统一目录，包含转换、清洗、抽样、校验、审计等脚本。

## 目录结构

- `processors.py`：统一处理器注册表。每个数据集对应一个
  `BaseDatasetProcessor` 子类，登记名称、描述、脚本、任务类型和
  `bbox_type`，公共执行逻辑由基类统一处理。
- `convert_datasets.py`：统一入口 CLI，注册新的数据集类后无需改动入口。
- `prepare_*` / `convert_*` / `curate_*` / `clean_*`：数据集转换与清洗的
  实际实现。
- `test_*.py`：与脚本对应的单元测试，删除前必须确认逻辑已迁移。
- `split/`：数据集划分。
- `universal_dataset/`：通用 S1-UDF 数据模型与 adapter 框架，适合需要
  正式发布链路的数据集。
- `audit_*` / `validate_*` / `verify_*` / `sanitize_*`：质量审计与格式校验。
- `smoke/`：smoke 抽样与本地视频准备脚本。

## 为什么不建议现在删除 prepare/test

`processors.py` 目前是“注册表 + 调度器”，不是实现替换：
`LegacyScriptProcessor.run()` 会调用对应 `prepare_*.py` 完成真实转换，
删除脚本会导致统一入口立即失败。`test_*.py` 覆盖这些脚本的行为，是回归
测试。

若希望彻底删除 `prepare_*`，需要先把每个数据集的转换逻辑迁入对应
processor 类或 `universal_dataset` source adapter，跑通测试并产出等价
JSONL 后再删，建议按数据集分批迁移。

## 统一入口用法

```bash
python scripts/data-process/convert_datasets.py list
python scripts/data-process/convert_datasets.py list --json
python scripts/data-process/convert_datasets.py run gqa --gqa-root /data/gqa --output out.jsonl
```

`run` 会把 `dataset` 后面的参数原样传给对应脚本，`--workdir` 控制底层
脚本的工作目录。

## 新增数据集

1. 在 `processors.py` 新增一个 `BaseDatasetProcessor` 子类并注册。
2. 指定 `name`、`description`、`script`、`task_types`，grounding/point
   类显式写 `bbox_type="norm1000"`。
3. 在 `convert_datasets.py list` 中确认已出现。
4. 为新逻辑补充 `test_*.py`。

复杂来源建议优先使用 `universal_dataset/ADAPTER_GUIDE.md` 的 source
adapter 插件，通过 `universal_dataset/cli.py convert-source` 做转换、
切分和校验。

## Manifest

`dlc_ready_entrypoints.stage1.json`、`curated_dataset_entrypoints.stage1.json`、
`dlc_non_global_entrypoints.stage1.json` 是训练入口 manifest，保留在
`scripts/` 根目录，不随数据处理脚本移动。

## 清洗、去重与过滤规则

Stage 1 训练数据的清洗、去重和过滤不在单一脚本里，而是按链路落在 `scripts/data-process/`：各源先由 `prepare_*.py` / `convert_*.py` / `curate_*.py` / `clean_*.py`（统一入口 `convert_datasets.py` + `processors.py`）拒绝缺媒体、非法文本、坐标无法归一到 `norm1000` 等记录并写入 `rejected.jsonl`；`sanitize_sft_jsonl.py` 再丢掉完全相同的 JSONL 行、非法 `messages`、`<image|video|audio>` 与媒体列表数量不一致、图片无法解码，以及 `bbox_type=real` 且越出解码宽高的框；`split/grouped_jsonl_split.py` 按图片/视频家族做 train/eval 切分，禁止同一媒体或 lineage 跨 split；最后 `global_media_dedup.py` 读取 `scripts/curated_dataset_entrypoints.stage1.json` 的 `global_dedup`：先冻结全部 eval 媒体身份（路径、URL、文件 SHA256、COCO 图号、`episode_id`/`video_id` 等 lineage），再按 `training_priority` 写各集 `*_global_train.jsonl`。同一数据集内部的 train/eval 媒体重叠一律排除（`eval_media_overlap`）；`eval_overlap_exempt_groups` 中的数据集家族（默认 GQA 与 Visual Genome）允许共享同一张图，各自保留在自己的 train 里。跨训练集媒体在 `deduplicate_cross_dataset_train=false` 时归各源自己保留、其余不再记 `higher_priority_train_overlap`，同一数据集内同图不同 QA 保留。Visual Genome 的 train/val 由 `split/resplit_visualgenome_gqa_shared.py` 重建：与 GQA 同图强制进 VG train，仅 VG 独有图像按 `val_ratio` 哈希进 val。入口合法性由 `validate_sft_entrypoints.py` 与 `audit_dlc_sft.py` 核对，排除明细在 `global_media_dedup_exclusions.jsonl`。
