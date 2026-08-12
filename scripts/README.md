# Stage 1 脚本使用说明

本目录包含 Stage 1 数据准备、清洗切分、训练入口校验、grounding 评估和 benchmark 工具。除非脚本说明另有要求，命令都应在仓库根目录执行：

```bash
cd /mnt/workspace/stage1
python scripts/<script>.py --help
```

多数数据脚本的默认路径指向服务器 `/mnt/luojunkun/stage1`。在其他机器运行时必须显式传入输入和输出路径。带 `--overwrite` 的命令会替换已有结果，先用独立测试目录和 `--limit`、`--max-samples`、`--check-only` 或 `--skip-convert` 做小规模检查。

## 推荐执行顺序

```text
原始数据
  -> prepare_*.py                 转成 ms-swift JSONL，并解析/提取媒体
  -> curate_*.py 或 split/*.py   清洗、去重、按媒体分组切分
  -> audit_*.py / validate_*.py  检查污染、泄漏和训练入口
  -> swift sft                   训练
  -> benchmark-tool/             推理、评分、断点续跑和对比报告
```

`curate_robovqa_sft.py` 和 `curate_llava_sft.py` 已同时完成清洗/去重和分组切分，不要再对其输出做一次普通随机切分。训练时只传 curated train 文件，评测时只传对应 eval 文件。

## ms-swift 数据约定

普通图像或视频 SFT 行使用 `messages` 加媒体列表：

```json
{"messages":[{"role":"user","content":"<video>\nQuestion: Is the robot moving?"},{"role":"assistant","content":"yes"}],"videos":["/path/clip.mp4"]}
```

Grounding 行额外使用 `objects`。`objects.ref` 按出现顺序替换 `<ref-object>`，`objects.bbox` 独立地按出现顺序替换 `<bbox>`，二者不建立隐式一一对应关系：

```json
{"messages":[{"role":"user","content":"<image>Locate <ref-object>."},{"role":"assistant","content":"<bbox>"}],"images":["/path/image.jpg"],"objects":{"ref":["red cup"],"bbox":[[20,30,100,160]],"bbox_type":"real","image_id":[0]}}
```

`bbox_type` 可为 `real` 或 `norm1`，默认 `real`。`image_id` 仅在多图且真实像素坐标时需要，索引从 0 开始。每个占位符的数量必须分别与相应数组长度一致。

Visual Genome 原始 regions 转换遵循“一条 region annotation -> 一条样本、一个 bbox”。即使同一张图里有多个同名物体，单数名称也不会让转换器自动聚合多个框；只有源记录的 `objects.bbox` 本来就包含多个框时，后续 grounding 转换才会输出多个 `<bbox>`。

如果任务标注明确要求定位同类的全部实例，可以只保留一个名称占位符，同时输出多个框占位符。例如：

```json
{"messages":[{"role":"user","content":"<image>Locate <ref-object>."},{"role":"assistant","content":"<bbox><bbox>"}],"images":["/path/image.jpg"],"objects":{"ref":["sheep"],"bbox":[[90.9,160.8,135,212.8],[360.9,480.8,495,532.8]],"bbox_type":"real"}}
```

Qwen2.5-VL/Qwen3-VL 如采用官方 cookbook 的 `bbox_2d` JSON 输出格式，应设置 `QWENVL_BBOX_FORMAT=new`，并在训练与推理时保持一致；默认仍为 `legacy`。benchmark 解析器可识别 `bbox_2d`。

## RoboVQA 完整流程

### 1. 转换并移除推理污染

`prepare_robovqa_swift.py` 读取 `robovqa_reasoning_*.json` 和 `robovqa_understanding.json`，解析或提取视频片段，生成 ms-swift JSONL。默认 `--reasoning-policy answer-only` 会提取最终答案并移除 `<think>/<answer>` 等合成推理格式；超长答案默认压缩到 512 字符以内。

先在独立目录做 smoke 检查：

```bash
python scripts/prepare_robovqa_swift.py \
  --input-dir /mnt/luojunkun/stage1/dataset/robovqa \
  --output-dir /tmp/robovqa-prepare-smoke \
  --reasoning-policy answer-only \
  --max-answer-chars 512 \
  --overlong-answer-policy compact \
  --missing-media-policy error \
  --max-samples 100 \
  --num-workers 4 \
  --overwrite
```

确认后运行全量：

```bash
python scripts/prepare_robovqa_swift.py \
  --input-dir /mnt/luojunkun/stage1/dataset/robovqa \
  --output-dir /mnt/luojunkun/stage1/dataset_ms-swift/robovqa \
  --reasoning-policy answer-only \
  --max-answer-chars 512 \
  --overlong-answer-policy compact \
  --missing-media-policy skip \
  --num-workers 8 \
  --overwrite
```

主要输出为 `robovqa_train_sft.jsonl` 和 `robovqa_conversion_report.json`。`--relative-paths` 可写相对媒体路径；移动 JSONL 时必须同步保持相对目录结构。

### 2. 清洗旧数据并按完整视频切分

`curate_robovqa_sft.py` 适合重新处理已经生成的 RoboVQA JSONL。它会再次提取最终答案、清理用户提示中的格式要求，并以视频为原子组确定性切分，保证同一视频不会同时进入 train/eval。

```bash
python scripts/curate_robovqa_sft.py \
  --input-json /mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_train_sft.jsonl \
  --train-output /mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_train_sft_train.jsonl \
  --eval-output /mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_train_sft_eval.jsonl \
  --report-output /mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_reasoning_cleanup_report.json \
  --eval-ratio 0.10 \
  --seed 42 \
  --max-answer-chars 512 \
  --overlong-answer-policy compact \
  --overwrite
```

报告中的 `video_overlap` 和 `group_overlap` 必须为 0。输出通过同目录临时文件原子替换；中断后遗留的 `.*.tmp` 不是正式数据，确认正式文件完整且没有进程仍在写入后再处理。

如果不需要清理，只需对已经干净的 JSONL 按视频切分，可运行：

```bash
python scripts/split/split_robovqa.py \
  --input-json /path/robovqa_train_sft.jsonl \
  --eval-ratio 0.10 \
  --seed 42 \
  --overwrite
```

### 3. 审计残留污染

`audit_robovqa_contamination.py` 检查残留 reasoning 标签、格式说明和 assistant 元推理。`--check-cleaner` 还会验证每条回答能否被当前清洗器稳定处理。

```bash
python scripts/audit_robovqa_contamination.py \
  /mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_train_sft_train.jsonl \
  /mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_train_sft_eval.jsonl \
  --check-cleaner \
  --max-examples 20 \
  > /mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_contamination_audit.json
```

检查输出中的 `contaminated_rows == 0`、`cleaner_failure_rows == 0` 和 `cleaner_failures == 0`。该脚本把 JSON 报告写到标准输出，不会因发现污染自动返回非零退出码。

### 4. 抽样评测

现有 `benchmark-tool/run_robovqa_smoke_256.sh` 使用 `--limit 256` 测试 eval 文件的前 256 行；这是固定前缀 smoke，不是随机抽样。脚本当前会同时启动 baseline 和 ours，显存紧张或底层视频算子不稳定时，应按 [benchmark-tool/README.md](benchmark-tool/README.md#robovqa-快速对比) 的命令顺序运行两个模型，并为二者使用相同参数和不同输出目录。

## 数据准备脚本

所有 `prepare_*.py` 都支持 `--help`。下表列出其主要职责；输出文件名可通过各脚本的 `--output-*`、`--jsonl-suffix` 或 `--output-template` 调整。

| 脚本 | 输入与用途 | 关键注意事项 |
| --- | --- | --- |
| `prepare_ai2d_swift.py` | AI2D annotations/questions/images -> PT 或 SFT JSONL | `--mode pretrain|sft`；可保留 ID、相对路径和 fallback 行 |
| `prepare_chartqa_swift.py` | ChartQA 原始 JSON 或 parquet -> train/val/test JSONL | `--source-format auto|parquet|json`；原始 JSON 模式可选 human/augmented |
| `prepare_cococaption_swift.py` | COCO Caption 风格 Arrow/JSON/JSONL -> caption SFT | 当前默认 input/output 都误指 RoboVQA 目录，实际使用时必须同时显式传 `--input-dir` 和 `--output-dir`；多 caption 可选 first/all |
| `prepare_gqa_swift.py` | GQA parquet configs -> PT/SFT JSONL | 常用 configs 为 `train_balanced val_balanced`；回答可选 `fullAnswer` 或 `answer` |
| `prepare_llava_instruct_swift.py` | LLaVA v1.5 mix665k -> 对齐媒体的 SFT JSONL | 涉及 COCO/OCR-VQA/GQA/TextVQA/VG；先 `--check-only`，缺图策略默认应显式选择；支持断点式 OCR 提取 checkpoint |
| `prepare_robo2vlm_swift.py` | Robo2VLM parquet -> PT/SFT JSONL，并提取图像 | 文本调整后可 `--reuse-existing-images` 避免重读 parquet 图片字节 |
| `prepare_robovqa_swift.py` | RoboVQA JSON + clips/archives -> 视频 SFT JSONL | 默认只监督最终答案；详见上方完整流程 |
| `prepare_textvqa_swift.py` | TextVQA shards/images -> SFT JSONL | 多答案策略 majority/first/all；`--strict-counts` 可把行数漂移变为错误 |
| `prepare_visualgenome_swift.py` | Visual Genome QA/regions -> PT/SFT JSONL | 按 image ID 切分，避免同图泄漏；regions 输出是 grounded region caption |
| `prepare_visualgenome_regions_grounding.py` | 已转换的 VG region JSONL -> locate/describe/grounded 数据 | `locate` 是 phrase-to-box，`describe` 是 box-to-phrase，`both` 为每个源行写两条 |
| `prepare_visualgenome_grounded_graph.py` | VG region graph -> 一图多对象 grounded caption | `--max-regions` 限制全局读取的 region 总数，不是单图对象数；生成后运行对应 validator |
| `prepare_vlmr1_swift.py` | VLM-R1 SFT/REC JSON + ZIP 图像 -> grounding JSONL | `--check-only` 先验证映射；可分别控制缺图、歧义图和非法记录策略 |
| `prepare_vqav2_swift.py` | VQAv2 parquet -> PT/SFT JSONL | 已有图片时优先 `--json-only`；只有检查确认零缺图后才使用 `--trust-image-paths` |

典型小规模转换模式：

```bash
python scripts/prepare_vqav2_swift.py \
  --input-dir /path/VQAv2 \
  --output-dir /tmp/vqav2-smoke \
  --splits validation \
  --mode sft \
  --limit 100 \
  --overwrite
```

先检查具体脚本生成的 `*_report.json`，再把 smoke 参数原样迁移到正式输出目录并移除样本上限。

## 清洗与切分脚本

| 脚本 | 用途 |
| --- | --- |
| `curate_llava_sft.py` | 优先保留独立 GQA/TextVQA/VisualGenome 数据，移除 LLaVA 中对应副本和 VQAv2 精确问题重叠，再按完整图片切分 |
| `curate_robovqa_sft.py` | 清理 RoboVQA 合成推理并按完整视频切分 |
| `split/split_ai2d.py` | 按图像字节 SHA-256 切分；可用现有 eval 作为保留集合，严格保证同图不跨 split |
| `split/split_robo2vlm.py` | 合并 Robo2VLM 源 split，去掉尾部 `_qN` 后按底层 episode/frame lineage 保留全部问题 |
| `split/split_robovqa.py` | 按 `videos` 分组切分并验证媒体零重叠 |
| `split/split_vlmr1.py` | 按图像字节 SHA-256 分组切分，并分别报告 QA/grounding 行数 |
| `split/grouped_jsonl_split.py` | 上述分组切分器共享实现，不是面向用户的命令入口 |

数据集包含同图、同视频或同轨迹的多条问答时，不要使用逐行随机切分。分组切分的 eval 行数是按组逼近 `--eval-ratio`，不保证恰好等于总行数乘比例。

已有 eval 已用于历史评测时，使用 `--reserved-eval-json` 和 `--reserve-only` 保留其媒体成员，再从完整 source 重建严格 train。例如 AI2D：

```bash
python scripts/split/split_ai2d.py \
  --reserved-eval-json /mnt/luojunkun/stage1/dataset_ms-swift/ai2d/ai2d_pretrain_msswift_eval.jsonl \
  --reserve-only \
  --overwrite
```

训练只能使用 `ai2d_pretrain_msswift_train.jsonl`。禁止把完整 `ai2d_pretrain_msswift.jsonl` 与 eval 同时使用。报告中的 `train_eval_overlap` 必须为 0。

VLM-R1 和 Robo2VLM 同样保留已经用于历史评测的 eval 媒体组：

```bash
python scripts/split/split_vlmr1.py \
  --reserved-eval-json /mnt/luojunkun/stage1/dataset_ms-swift/vlm-r1/vlm_r1_sft_grounding_msswift_eval.jsonl \
  --reserve-only \
  --overwrite

python scripts/split/split_robo2vlm.py \
  --reserved-eval-json /mnt/luojunkun/stage1/dataset_ms-swift/robo2vlm/robo2vlm_sft_eval.jsonl \
  --reserve-only \
  --overwrite
```

Robo2VLM 的官方 source `test` 与 `train` 共享底层 episode，不能直接把 source test 当成独立验证集。必须先用 `prepare_robo2vlm_swift.py` 把 parquet 中可能嵌套的选项展开为逐行 `A/B/C...`，再合并两个 source JSONL，按去掉 `_qN` 后的完整 lineage 重切；该数据是多选 VQA / 具身状态理解，不是 caption。评分器将 `B`、选项文本和 `B. 选项文本` 视为同一答案，但拒绝整份选项列表及标签/文本冲突的输出。

## 全局媒体去重

单数据集内零交叉还不够。`global_media_dedup.py` 读取 v2 curated manifest，先保留所有 eval 媒体，再按 `global_dedup.training_priority` 为训练媒体分配唯一数据集归属：

- 图像：SHA-256、COCO image ID、规范路径/URL；
- 视频：规范路径/URL、视频文件 ID、显式 episode/video/trajectory 字段；
- Robo2VLM：额外使用去掉 `_qN` 后的 episode/frame lineage；
- 同一数据集同一媒体上的多条 QA 会全部保留，只有跨数据集重复或 train/eval 重复会被排除。

```bash
python scripts/global_media_dedup.py \
  --manifest /mnt/luojunkun/stage1/dataset_ms-swift/curated_dataset_entrypoints.json \
  --overwrite

python scripts/validate_sft_entrypoints.py \
  --manifest /mnt/luojunkun/stage1/dataset_ms-swift/curated_dataset_entrypoints.json
```

当 manifest 设置 `global_dedup.required=true` 时，校验器只接受其中列出的全局 clean `train`，拒绝 `source_only`、`source_train`、`split_train` 和任意未登记训练路径。去重报告必须是 `complete`，且后验 `train_eval_overlap_rows`、`cross_dataset_train_overlap_rows` 均为 0；任何输入在报告生成后发生变化，也必须重新去重。

去重运行期间不得修改 JSONL 或其媒体文件。同一路径首次访问时校验文件类型、size 和 mtime，并计算或复用 SHA-256；同次运行后续访问走进程内缓存，跨运行复用 SQLite 缓存时重新校验 size 和 mtime。

OSSFS 等高延迟挂载可设置 `--hash-workers 16 --hash-prefetch-rows 2000` 并发读取每批本地图片。记录过滤、SQLite cache/owner 写入和输出顺序仍由主线程串行执行；并发只用于文件 `stat` 与 SHA-256 计算。

`global_dedup.require_media_identity` 默认开启。任一启用数据集的 train/eval 行若无法解析出图像、视频或 episode/trajectory 身份，全局去重会直接失败；入口校验器也要求报告中的 `train_rows_without_identity` 和 `eval_rows_without_identity` 均为 0，禁止用“无法审计的行”得到表面上的零重叠结论。

对于明确混合纯文本对话的数据集，可在单个数据集配置中设置 `allow_text_only: true`。这类行使用 canonical `messages` 的 SHA-256 作为全局 `text:sha256` 身份；输入消息包含 `<image>`、`<video>`、`<audio>`，或任意消息包含媒体内容块但缺少媒体路径时仍会失败，不能借纯文本开关绕过缺失媒体检查。assistant 回答中的字面 HTML 标签（例如讲解 `<video>` 元素）按纯文本处理。

SpatialVLM 转换必须保留 parquet 文件声明的官方 split：`test-*` 只进入 val，`train-*` 只进入 train。转换器会预扫描完整 official test 图像 SHA-256 集，并排除 train 中命中该集合的记录。旧转换把约 8,222 条 test 来源记录写入 train，旧入口已失效，不能用于训练。

完整 Stage 1 整改和验收顺序如下。先重建四个受影响的数据集，再安装最终 manifest、生成全局 clean train，最后校验所有训练入口：

```bash
python scripts/split/split_ai2d.py \
  --reserved-eval-json /mnt/luojunkun/stage1/dataset_ms-swift/ai2d/ai2d_pretrain_msswift_eval.jsonl \
  --reserve-only \
  --overwrite

python scripts/split/split_vlmr1.py \
  --reserved-eval-json /mnt/luojunkun/stage1/dataset_ms-swift/vlm-r1/vlm_r1_sft_grounding_msswift_eval.jsonl \
  --reserve-only \
  --overwrite

python scripts/prepare_robo2vlm_swift.py \
  --input-dir /mnt/luojunkun/stage1/dataset/robo2vlm \
  --output-dir /mnt/luojunkun/stage1/dataset_ms-swift/robo2vlm \
  --dataset-mode sft \
  --reuse-existing-images \
  --validate-reused-images \
  --missing-image-policy error \
  --overwrite

python scripts/split/split_robo2vlm.py \
  --reserved-eval-json /mnt/luojunkun/stage1/dataset_ms-swift/robo2vlm/robo2vlm_sft_eval.jsonl \
  --reserve-only \
  --overwrite

python scripts/prepare_molmo_pixmo_spatial_swift.py \
  --datasets spatialvlm \
  --input-root /mnt/luojunkun/stage1/dataset \
  --output-root /mnt/luojunkun/stage1/dataset_ms-swift \
  --overwrite

python scripts/verify_stage1_data_remediation.py \
  --data-root /mnt/luojunkun/stage1/dataset_ms-swift \
  --manifest scripts/curated_dataset_entrypoints.stage1.json \
  --output /mnt/luojunkun/stage1/dataset_ms-swift/stage1_data_remediation_report.json

install -m 0644 scripts/curated_dataset_entrypoints.stage1.json \
  /mnt/luojunkun/stage1/dataset_ms-swift/curated_dataset_entrypoints.json

python scripts/global_media_dedup.py \
  --manifest /mnt/luojunkun/stage1/dataset_ms-swift/curated_dataset_entrypoints.json \
  --overwrite

python scripts/validate_sft_entrypoints.py \
  --manifest /mnt/luojunkun/stage1/dataset_ms-swift/curated_dataset_entrypoints.json
```

`stage1_data_remediation_report.json` 必须记录 AI2D 79 个、VLM-R1 2,825 个历史 eval 图像组，以及 Robo2VLM 1,582 个历史 eval lineage 和 5,239 个官方 train/test 共享 lineage，并验证 split 文件 SHA-256。SpatialVLM 的转换报告必须绑定 train、val 和媒体组清单的 SHA-256，三个官方 split 泄漏计数必须全为 0。

### DLC 训练入口：只清除 train/eval 污染

48 卡全量训练不使用上述 `*_global_train.jsonl`。本次策略在 manifest 中设置
`deduplicate_cross_dataset_train=false`：所有 eval 媒体仍在全局保留，任何训练行只要命中任一 eval
媒体就会被排除；不同训练数据集共享同一媒体时，各自的不同任务和监督全部保留。最终训练入口使用
`*_ready_train.jsonl` 命名，避免和 strict global 结果及清洗前的 `*_dlc_train.jsonl` 混淆。

```bash
python scripts/global_media_dedup.py \
  --manifest scripts/dlc_ready_entrypoints.stage1.json \
  --hash-workers 16 \
  --hash-prefetch-rows 4000 \
  --overwrite

python scripts/validate_sft_entrypoints.py \
  --manifest scripts/dlc_ready_entrypoints.stage1.json

python scripts/audit_dlc_sft.py \
  --manifest scripts/dlc_ready_entrypoints.stage1.json \
  --workers 8 \
  --samples-per-dataset 3
```

最终报告
`/mnt/luojunkun/stage1/dataset_ms-swift/dlc_sft_audit_report.json` 必须为
`status=passed`，其中 `train_eval_overlap_rows=0`、schema error 和精确重复行均为 0。
Grounding 抽样不是只打印坐标：`dlc_sft_audit_visualizations/` 会把真值 bbox/point
画回原图，VideoTrack 会抽取对应视频帧并叠加逐帧真值点。完整的 48 卡命令和参数说明见
[README_dlc48_stage1_sft.md](README_dlc48_stage1_sft.md)。

## 校验与独立评分

| 脚本 | 用途和典型入口 |
| --- | --- |
| `audit_robovqa_contamination.py` | 审计 RoboVQA 最终答案中的推理污染；详见 RoboVQA 流程 |
| `audit_dlc_sft.py` | 流式校验 DLC JSONL schema、占位符、媒体、objects、重复行和分布，并在原图/视频帧绘制 grounding 真值抽样 |
| `validate_sft_entrypoints.py` | 根据 curated manifest 拒绝重复路径、train/eval 混用、源文件与 curated split 同时启用 |
| `validate_visualgenome_grounded_graph.py` | 校验 grounded graph 数据、媒体和 bbox 边界；split 校验硬编码对应生成器默认 `val_ratio=0.01`、`seed=42`，自定义切分不能直接使用；`--repair-bounds` 会修改数据，使用前备份 |
| `evaluate_grounding_iou.py` | 对 swift infer JSONL 计算多框 IoU、阈值准确率和分组指标；支持 line-aligned ground truth、optimal/ordered 匹配和错误明细 |

训练前入口校验：

```bash
python scripts/validate_sft_entrypoints.py \
  --manifest /mnt/luojunkun/stage1/dataset_ms-swift/curated_dataset_entrypoints.json
```

独立 grounding 评分示例：

```bash
python scripts/evaluate_grounding_iou.py \
  --input /path/predictions.jsonl \
  --ground-truth /path/eval.jsonl \
  --prediction-space norm1000 \
  --ground-truth-space objects \
  --thresholds 0.5 0.75 \
  --matching optimal \
  --report /path/iou_report.json \
  --details-output /path/iou_errors.jsonl \
  --details-mode errors \
  --overwrite
```

`--task-filter auto` 只评价参考答案要求输出 bbox 的行；仅仅存在 `objects.bbox` 不代表任务方向是 phrase-to-box grounding。

## Benchmark 工具

`benchmark-tool/` 是当前 Stage 1 推理与评分入口，支持：

- 官方 benchmark runner 和 VLMEvalKit；
- 自建 image/video VQA、description、grounding、grounded 数据；
- RoboVQA 类型感知答案指标、断点续跑、失败样本重试和离线重评分；
- baseline/ours 汇总、结果完整性校验和 HTML/CSV/Markdown 报告。

完整用法见 [benchmark-tool/README.md](benchmark-tool/README.md)。`benchmark-tool/*.pre-*-20260801`、`README copy.md` 和测试归档是历史快照，不应作为运行入口。

### Hy-Embodied benchmark 下载

`benchmark-tool/hy_benchmark_sources.json` 固定了 38 个 Hy-Embodied 表项的来源和 revision。相同仓库只下载一次：SITE-Bench 的 Image/Video、ShareRobot 的 Affordance/Trajectory、RoboBench 的 MCQ/Planning 分别共享一份数据。下载前先检查映射和 dry-run：

```bash
python scripts/benchmark-tool/download_hy_benchmarks.py --list

python scripts/benchmark-tool/download_hy_benchmarks.py \
  --dry-run \
  --target-root /mnt/luojunkun/stage1/benchmark-stage1
```

全量执行：

```bash
python scripts/benchmark-tool/download_hy_benchmarks.py \
  --target-root /mnt/luojunkun/stage1/benchmark-stage1 \
  --jobs 4
```

ModelScope 通过公共 HTTP API 递归分页下载，并按文件大小和 SHA-256 校验；Hugging Face 使用 `hf download` 的 LFS/断点续传能力。每个完整来源写入 `.hy-benchmark-download.json`，总报告写入 `_hy_benchmark_download_report.json`。重跑会复用匹配的完成标记，`--refresh` 会重新检查来源。

报告状态必须按字面理解：`complete` 才表示本地数据可用；`metadata_only` 表示媒体未完整落盘；`restricted` 表示需要授权；`manual_required` 表示还需拼装官方外部媒体；`unavailable` 表示没有可验证的公开源。目前 PixMo-Points 是已验 SHA-256 的本地 parquet，但图片仍是未验证远程 URL；EgoPlan2 缺少需 Ego4D 授权的视频；CrossHOI-Bench 仍需 HICO-DET、V-COCO、SWiG-HOI；Depth-InHouse 无公开源。原版 EgoPlan 不能替代 EgoPlan2。

## 其他目录

`benchmark/` 是 ms-swift 的旧实验调度器：`exp.py --config ... --save_dir ...` 根据 JSON 配置申请空闲 GPU 并执行 sft/rlhf/export，`generate_report.py` 汇总其 `experiment` 目录。它与当前 `benchmark-tool/` 不是同一套评测入口，新 Stage 1 模型对比优先使用后者。

`utils/` 是仓库维护脚本，而不是训练数据流水线：

| 脚本 | 用途 |
| --- | --- |
| `utils/plot_loss.py` | 从 TensorBoard 日志画 loss；先在文件内设置 `ckpt_dir` |
| `utils/run_dataset_info.py` | 重新生成支持数据集文档和 token 统计，可能下载/遍历大量数据 |
| `utils/run_model_info.py` | 重新生成支持模型表 |
| `utils/run_template.py` | 列出注册的 generation/chat template |
| `utils/test_link_valid.py` | 检查 Markdown 本地链接和 HTTP 链接，需要网络 |

这些维护脚本会更新仓库文档或访问外部资源，运行前先查看源码和 `git diff`。

## 常见问题

- `FileExistsError`：输出已存在。确认目标正确后再加 `--overwrite`，不要用它覆盖仍在写入的正式文件。
- 媒体找不到：先确认 JSONL 中是绝对路径还是相对 JSONL 的路径；跨机器复制时同步媒体目录。
- train/eval 行数比例不精确：分组切分以零媒体泄漏优先，比例只做近似。
- grounding 全为 0：确认任务方向、assistant 参考答案中确有 `<bbox>`、`objects.bbox` 数量正确，并核对 `--prediction-space`。
- 视频评测退出或显存不足：降低 batch、`--max-video-frames` 和生成长度；baseline/ours 顺序运行，并使用相同参数保证可比性。
