# scripts/data-process

数据转换、清洗、抽样、校验、审计等数据处理脚本统一放在本目录。

- `prepare_*` / `convert_*` / `curate_*` / `clean_*`：数据集转换与清洗。
- `split/`：数据集划分。
- `universal_dataset/`：通用 S1-UDF 数据模型与 adapter 框架。
- `audit_*` / `validate_*` / `verify_*` / `sanitize_*`：质量审计与格式校验。
- `download_pixmo_media.py` / `materialize_molmo2_videotrack_media.py`：媒体下载与落地。
- `smoke/`：smoke 抽样与本地视频准备脚本。

`dlc_ready_entrypoints.stage1.json`、`curated_dataset_entrypoints.stage1.json`、
`dlc_non_global_entrypoints.stage1.json` 属于训练入口 manifest，仍保留在
`scripts/` 根目录，不随数据处理脚本移动。

## 统一入口

`convert_datasets.py` 是数据处理脚本的统一注册入口。每个数据集对应
`processors.py` 中的一个 `BaseDatasetProcessor` 子类；子类记录数据集名称、
描述、对应脚本、任务类型和 `bbox_type`，公共执行逻辑由基类统一处理。

```bash
python scripts/data-process/convert_datasets.py list
python scripts/data-process/convert_datasets.py list --json
python scripts/data-process/convert_datasets.py run gqa --gqa-root /data/gqa --output out.jsonl
```

新增数据集时，在 `processors.py` 中新增一个子类并注册即可，不需要改动
`convert_datasets.py`。若来源格式复杂、需要进入正式发布链路，优先参考
`universal_dataset/ADAPTER_GUIDE.md`，把来源适配器写成 S1-UDF source adapter
插件，再通过 `universal_dataset/cli.py convert-source` 统一转换、切分和校验。

