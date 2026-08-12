# Stage 1 DLC 48 卡（3 节点 × 16 卡）全量 SFT 抽检与训练说明

本次训练使用 `dlc_ready_entrypoints.stage1.json` 中启用的 16 个数据集，
不使用任何 `*_global_train.jsonl`。最终 ready 入口已删除精确重复行和非法
真实坐标 bbox，全局保留所有 eval 媒体，同时保留不同训练数据集共享媒体上的
不同任务与监督。

## 训练前门禁

去污染和全量抽检完成后运行：

```bash
python3 /mnt/luojunkun/stage1/sft-model/scripts/validate_sft_entrypoints.py \
  --manifest /mnt/luojunkun/stage1/sft-model/scripts/dlc_ready_entrypoints.stage1.json

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

## 最终数据分布

2026-08-12 最终全量审计通过，共包含 9,181,393 条训练数据和 738,770 条
保留评测数据。检查结果为：

- ms-swift schema 错误：0
- 精确重复行：0
- train/eval 媒体重叠：0
- grounding 可视化错误：0
- 跨训练数据集共享媒体的保留行数：4,200,782

最后一项不是污染。当前策略设置
`deduplicate_cross_dataset_train=false`，同一媒体在不同训练数据集中的不同任务和
监督均会保留；只有命中任一 eval 媒体的训练行才会被排除。

| 数据集 | Train | Eval |
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

## 训练任务分布

任务类型根据每行实际的 `messages`、媒体字段和 `objects` 内容判断，不根据数据集名称
推测。

| 任务类型 | Train 行数 | 占比 |
| --- | ---: | ---: |
| VQA / instruction | 4,744,480 | 51.67% |
| Grounded description | 2,895,538 | 31.54% |
| Point grounding | 858,802 | 9.35% |
| Caption / description | 374,447 | 4.08% |
| Bbox grounding | 243,036 | 2.65% |
| 纯文本 | 35,442 | 0.39% |
| Video tracking | 22,975 | 0.25% |
| Box-to-text | 6,673 | 0.07% |

## 抽检方法与结果

全量审计会解析每一行 JSONL，检查：

- `messages`、role 和 content 是否符合 ms-swift 格式；
- `<image>`、`<video>`、`<audio>` 占位符数量是否与媒体列表一致；
- 媒体是否为存在的本地绝对路径；
- `objects.ref`、`objects.bbox`、`image_id` 与对应占位符数量是否一致；
- bbox/point 坐标是否合法，真实坐标 bbox 是否超出原图边界；
- 每个入口内是否仍有精确重复 JSONL 行；
- 所有 train 媒体是否与任一 eval 媒体重叠。

可视化样本不是抽取文件前几行，而是在完整文件扫描过程中，根据每行 128-bit
BLAKE2b 哈希做确定性 bottom-k 抽样。每个 grounding 数据集和 split 最多抽 3 条，
最终生成 30 组可视化，全部通过。

抽检必须把真值画回原始像素，不能只打印坐标：

- bbox 任务在原图绘制真实框、对象编号和 ref 文本；
- point 任务在原图绘制点、点编号和 ref 文本；
- VideoTrack 使用 PyAV 精确解码目标帧，在首帧、中间帧和末帧绘制 norm1000 真值点。

人工复核的代表样例：

- VisualGenome Regions：两个框分别覆盖长颈鹿整体和腿部区域；
- VLM-R1：单个 bbox 覆盖人物手中的黑色物体；
- PixMo Points：5 个点均落在地面大写字母的对应笔画上；
- Molmo2 VideoTrack：第 0、40、127 帧的点均落在同一球门员上，并随目标位置变化。

报告和可视化目录：

- `/mnt/luojunkun/stage1/dataset_ms-swift/dlc_ready_decontamination_report.json`
- `/mnt/luojunkun/stage1/dataset_ms-swift/dlc_sft_audit_report.json`
- `/mnt/luojunkun/stage1/dataset_ms-swift/dlc_sft_audit_visualizations/`

## 持久化文件位置

`/mnt/workspace/stage1/scripts` 是临时工作区，正式训练不要依赖该目录。当前已将
训练入口、manifest 和门禁脚本同步到持久化目录：

- 训练相关脚本和 manifest：`/mnt/luojunkun/stage1/sft-model/scripts/`
- 模型输出和 smoke test 输出：`/mnt/luojunkun/stage1/sft-model/`
- 转换后的 ms-swift 数据、去污染报告和抽检可视化：`/mnt/luojunkun/stage1/dataset_ms-swift/`

其中 `dlc_ready_entrypoints.stage1.json` 可以从临时工作区迁移到持久化目录使用；
校验只允许 manifest 路径变化，文件内容 SHA256 必须与去污染报告记录一致。

## DLC 正式启动命令

DLC 任务使用 3 个节点，每个节点 16 张 GPU，总计 48 卡。DLC 会注入
`WORLD_SIZE`、`RANK` 和 `MASTER_ADDR`。在任务命令中设置共享模型权重和输出目录：

```bash
export MODEL_PATH=/shared/path/to/Qwen3-VL-4B-Instruct-or-your-full-checkpoint
export OUTPUT_DIR=/mnt/luojunkun/stage1/sft-model/qwen3-vl-stage1-dlc3x16-full
export EXPECTED_NNODES=3 NPROC_PER_NODE=16 EXPECTED_WORLD_SIZE=48
bash /mnt/luojunkun/stage1/sft-model/scripts/run_dlc_48card_full_sft.sh
```

默认参数为：全参数 BF16、ZeRO-3、8 路 sequence parallel、最大长度 65,536、
训练 1 个 epoch，有效全局 batch size 为 24，即 48 卡 / 8 路 sequence parallel =
6 个 data-parallel group，乘以 micro-batch 1 和 gradient accumulation 4。可通过
`GRADIENT_ACCUMULATION_STEPS` 调整全局 batch size。

VideoTrack 标签最多覆盖 128 个源视频帧，转换后视频帧率包括 6、12、20 和 25 FPS。
启动脚本因此默认设置 `FPS=25`、`FPS_MAX_FRAMES=128` 和
`VIDEO_MAX_TOKEN_NUM=128`，避免逐帧真值只对应到 2 FPS 或 16 帧的稀疏采样。
只要训练中包含 VideoTrack，就不要降低这些值。

`MAX_LENGTH=65536` 是针对 VideoTrack 较长的逐帧输出设置的。不能在未使用实际模型
processor 测量 token 长度前直接降低。`truncation_strategy=right` 可以避免丢弃整条
超长样本，但超出最大长度的 assistant 输出尾部仍会被截断。

## 8 卡 Smoke Test

提交付费 48 卡任务前，必须在相同 DLC 镜像中先运行单节点 8 卡、2 step smoke，
验证 Qwen3-VL sequence parallel、FlashAttention、DeepSpeed、视频解码和最长样本显存：

```bash
export WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export MODEL_PATH=/shared/path/to/model
export EXPECTED_NNODES=1 NPROC_PER_NODE=8 EXPECTED_WORLD_SIZE=8
export MAX_STEPS=2 SAVE_STEPS=2
export OUTPUT_DIR=/mnt/luojunkun/stage1/sft-model/dlc3x16-preflight-smoke
bash /mnt/luojunkun/stage1/sft-model/scripts/run_dlc_48card_full_sft.sh
```

必须在正式 DLC 训练镜像中运行，不能使用当前数据处理容器作为训练启动验证依据。
当前数据处理容器缺少 `libhggcrt1.so`，其加速器运行时不可用。

训练脚本关闭了训练过程中的 inline eval。保留评测集体量较大，应通过 benchmark
队列独立评测，避免在训练过程中反复消耗计算资源。
