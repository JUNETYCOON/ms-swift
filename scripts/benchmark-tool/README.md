# VLM Benchmark Tool 使用说明

本工具提供三套相互独立的流程：

1. `eval-custom`：直接用 Transformers 加载模型，对自建 image/video VQA、description、grounding 或 grounded JSONL 推理并评分。
2. `run`：调用 ms-swift/VLMEvalKit 或外部官方仓库，收集 benchmark 已计算的正式指标。
3. `prepare_local_benchmarks.py` + `score_local_benchmark.py`：把服务器本地下载的数据适配成统一 JSONL，再用带 `metric_scope` 的确定性评分器评测。

不要混淆三者的指标。特别是本地适配的 Flickr30k caption generation 不是 retrieval Recall@K，本地 OpenEQA 词面分数也不是官方 LLM Judge。最终报告必须检查 `metric_scope`。

## 安装和入口

在仓库根目录安装 ms-swift 及评测依赖：

```bash
cd /mnt/workspace/stage1
pip install -e '.[eval]'
```

安装后可使用 `vlm-eval`；不安装 console entry point 时，直接运行同一实现：

```bash
vlm-eval --help
python scripts/benchmark-tool/cli.py --help
```

视频随机访问建议安装与当前环境兼容的 `decord`。模型、CUDA/PPU SDK、PyTorch 和 Transformers 必须使用训练时验证过的兼容版本；工具不会自动替换这些运行时依赖。

## 命令速查

| 目标 | 命令 |
| --- | --- |
| 查看注册的正式 benchmark | `vlm-eval list-benchmarks --details` |
| 对自建验证集推理并评分 | `vlm-eval eval-custom ...` |
| 对保存的 custom predictions 重新评分 | `python scripts/benchmark-tool/rescore_custom_predictions.py ...` |
| 对通用 response/labels 文件做文本评分 | `vlm-eval score-custom ...` |
| 对分离的 prediction/annotation 文件算 IoU | `vlm-eval score-grounding ...` |
| 运行 VLMEvalKit/外部官方接口 | `vlm-eval run ...` |
| 汇总已有正式结果并画图 | `vlm-eval visual ...` |
| 准备服务器本地 benchmark 数据 | `python scripts/benchmark-tool/prepare_local_benchmarks.py ...` |
| 生成 Stage 1 汇总报告 | `python scripts/benchmark-tool/generate_stage1_report.py` |

## RoboVQA 快速对比

当前对比路径：

```text
dataset  /mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_train_sft_eval.jsonl
baseline /mnt/luojunkun/stage1/model
ours     /mnt/luojunkun/stage1/sft-model/7-30_stage1_modelv1
```

在评测前先运行 `scripts/audit_robovqa_contamination.py`，确认参考答案没有 `<think>/<answer>`、格式说明或元推理残留。`predictions.jsonl` 会保存推理当时的 reference，之后只修改源数据不会自动修正旧预测文件里的 reference。

下面的 smoke 参数为 batch 2、最多生成 64 tokens、最多抽 16 帧、前 256 行。两个模型按命令调用顺序执行，不会同时占用显存；输出目录彼此隔离：

```bash
cd /mnt/workspace/stage1

export LD_LIBRARY_PATH=/opt/accl-p:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0

VAL=/mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_train_sft_eval.jsonl
OUT=/mnt/luojunkun/stage1/benchmark-eval-result/self-valdataset/robovqa-smoke-256-type-aware

run_robovqa_smoke() {
  label=$1
  weights=$2
  python scripts/benchmark-tool/cli.py eval-custom \
    --val-dataset "${VAL}" \
    --model-weights "${weights}" \
    --model_type qwen3_vl \
    --model-name "${label}" \
    --dataset-name robovqa_smoke_256 \
    --task vqa \
    --output-dir "${OUT}/${label}" \
    --batch-size 2 \
    --max-new-tokens 64 \
    --max-video-frames 16 \
    --group-video-batches \
    --limit 256 \
    --dtype bfloat16 \
    --device cuda:0 \
    --attn-implementation sdpa \
    --progress-every 16 \
    --continue-on-error
}

run_robovqa_smoke baseline /mnt/luojunkun/stage1/model
run_robovqa_smoke ours /mnt/luojunkun/stage1/sft-model/7-30_stage1_modelv1
```

`--limit 256` 取数据文件前 256 行，不是随机抽样。需要统计代表性时，应先生成固定 seed、按视频分组且两模型共用的 sample JSONL，再去掉 `--limit` 或把 limit 设为该文件总行数。

`run_robovqa_smoke_256.sh` 是服务器快捷脚本，但当前实现会把 baseline 和 ours 放到后台并行运行，默认 batch 4。显存或视频底层算子不稳定时优先使用上面的顺序命令；使用脚本前必须检查其中的数据、权重和输出路径。

### 如何判断运行有效

分别检查：

```text
${OUT}/baseline/scores.json
${OUT}/ours/scores.json
```

必须同时满足：

- `status == "complete"`；
- `processed_samples == 256`；
- VQA 汇总行的 `failed_samples == 0`；
- 两边 `results.csv` 的 `sample_id`、`line_number` 和样本数一致；
- 两边使用相同 frames、tokens、batch/task 和数据文件。

`status=complete` 只表示数据循环正常结束；如果启用了 `--continue-on-error`，仍需单独确认 `failed_samples` 为 0。

## RoboVQA 类型感知指标

RoboVQA 同时包含 yes/no 和开放式动作回答。仅用整句 exact match 会把“答案正确但附带解释”的输出判错，因此 VQA 汇总增加以下指标：

| 字段 | 定义 |
| --- | --- |
| `answer_type` | 单样本类型：reference 是一致的规范化 `yes`/`no` 时为 `yes_no`，否则为 `freeform` |
| `answer_accuracy` | 主准确率。yes/no 为极性是否正确；freeform 为语义代理分数是否达到 0.68 |
| `semantic_similarity` | yes/no 样本取 0/1；freeform 样本取 `[0, 1]` 的离线代理分数；汇总时取全部 VQA 样本均值 |
| `yes_no_samples` | yes/no 样本数 |
| `yes_no_accuracy` | yes/no 子集准确率 |
| `freeform_samples` | 开放回答样本数 |
| `freeform_accuracy` | 开放回答中离线代理分数 `>= 0.68` 的比例 |
| `freeform_similarity` | 开放回答子集的平均离线代理分数 |

yes/no 评分先移除 `<answer>` 或 `</think>` 之前的内容，再取模型最终答案中第一个独立的 `yes`/`no`。因此 `yes, because ...` 可正确计分，`no ... yes ...` 按第一个明确极性计分。

开放回答使用无需额外模型、无需联网的 RoboVQA 启发式代理：规范化常见动作同义词和词形，综合 token/entity 覆盖与 chrF，并对动作缺失、实体覆盖不足、否定或方向/开关等语义冲突做门控。阈值 0.68 是当前 action/entity/conflict 规则后的校准值，并写入 `scores.json.freeform_similarity_threshold`。

该代理适合快速比较短英文机器人动作答案，但不等同于 embedding 相似度、LLM Judge、人工语义真值或 benchmark 官方指标。论文级结论应抽查边界样本和 baseline/ours 分歧样本，必要时增加盲审或独立 judge。

旧指标继续保留用于诊断：

- `exact_match`：清理后的最终预测文本与任一 reference 去首尾空白后完全一致，不做大小写、标点或同义容错；
- `vqa_accuracy`：VQA 规范化后的一致性/多参考软准确率；
- `token_f1`：规范化 token 重叠。

比较 RoboVQA 时优先报告 `answer_accuracy`，并同时展开 `yes_no_accuracy`、`freeform_accuracy`、`freeform_similarity` 和样本数。不要只报告单个总体分数掩盖数据类型占比。

## 自建数据格式和任务

输入支持 JSONL、JSON 和 CSV。推荐直接使用训练时的 JSONL：

```json
{"messages":[{"role":"user","content":"<image>What color is the cup?"},{"role":"assistant","content":"red"}],"images":["/path/image.jpg"]}
```

视频使用 `videos` 和 `<video>`：

```json
{"messages":[{"role":"user","content":"<video>\nWhat does the robot do next?"},{"role":"assistant","content":"pick up the cup"}],"videos":["/path/clip.mp4"]}
```

Grounding 使用 `objects` 占位符协议：

```json
{"messages":[{"role":"user","content":"<image>Locate <ref-object>."},{"role":"assistant","content":"<bbox>"}],"images":["/path/image.jpg"],"objects":{"ref":["red cup"],"bbox":[[20,30,100,160]],"bbox_type":"real","image_id":[0]}}
```

任务方向按模型需要输出的内容判断：

| `--task` | 输入/输出方向 | 指标 |
| --- | --- | --- |
| `vqa` | 问题 -> 短答案或开放答案 | 类型感知 VQA、exact、token F1 |
| `description` | 图像/视频/区域 -> 自然语言描述 | ROUGE-L、sentence BLEU-4、chrF、token F1 |
| `grounding` | phrase -> bbox | IoU、precision/F1、框解析成功率 |
| `grounded` | 描述 + bbox 输出 | description 与 grounding 两组指标 |
| `auto` | 每行自动判断上述类型 | 混合数据；非典型 prompt 建议不要依赖自动判断 |

`objects.bbox` 可能只是 box-to-phrase 任务的输入，字段存在本身不会强制判为 grounding。phrase-to-box 的参考 assistant 必须真正要求输出 `<bbox>`。对单一数据集建议显式传 `--task`。

## `eval-custom` 主要参数

基础命令：

```bash
python scripts/benchmark-tool/cli.py eval-custom \
  --val-dataset /path/val.jsonl \
  --model-weights /path/checkpoint \
  --model_type qwen3_vl \
  --model-name my_model \
  --dataset-name my_val \
  --task auto \
  --output-dir /path/eval_output \
  --batch-size 1 \
  --dtype bfloat16 \
  --device cuda:0
```

模型和推理参数：

| 参数 | 说明 |
| --- | --- |
| `--model-weights` | 完整模型目录或 LoRA adapter |
| `--base-model` | `--model-weights` 为 adapter 时的基础模型 |
| `--model-type` / `--model_type` | 校验 `config.json`；`qwen3_vl` 显式加载对应 generation class |
| `--max-new-tokens` | 短 VQA 常用 16-64；描述任务通常需要 128 或更多 |
| `--dtype` | `auto|float32|float16|bfloat16` |
| `--attn-implementation` | `sdpa` 默认；Flash SDP 不稳定时试 `eager`；已正确安装时可用 `flash_attention_2` |
| `--max-image-pixels` | 图像像素上限；常用 262144，降低可提速但会损害 OCR/小目标 |
| `--trust-remote-code` | 只对可信模型仓库启用 |

视频和批处理参数：

| 参数 | 说明 |
| --- | --- |
| `--max-video-frames` | 每个视频最多采样帧数；模型对比必须保持一致 |
| `--min-video-frames` | 为极短视频设置最少帧数 |
| `--video-backend` | `auto|decord|opencv|pyav|torchvision|torchcodec`；长视频优先测试 decord |
| `--group-video-batches` | 只把处理后时空 grid 一致的视频放在同一 batch |
| `--min-video-batch-size` | 同构视频 batch 太小时用 padding 项达到最小 batch，padding 预测不计分 |
| `--allow-empty-video-fallback` | 无法解码时用黑帧，仅用于明确允许该退化的兼容测试，不建议用于 RoboVQA 正式分数 |
| `--group-image-batches` | 只合并处理后空间 grid 一致的图像 |
| `--batched-image-grids` | 可批处理 grid 白名单，如 `18x32,24x32` |
| `--image-grid-fallback` | 把白名单外图像 letterbox 到指定 grid |
| `--max-batched-image-grid-area` | grid 超过阈值时退回 singleton，控制峰值显存 |

不要为了某个模型单独降低分辨率、帧数或生成长度后直接比较分数。先用相同样本和参数确认可运行，再统一调整两边配置。

## 断点恢复和失败重试

推理中断后，用原命令追加：

```bash
--resume --retry-errors
```

行为如下：

- `--resume` 同时读取 `predictions.jsonl` 和 `results.csv`，按数据文件行号跳过已完成行并恢复累计指标；
- `--retry-errors` 只与 `--resume` 配合，先从两个文件中移除先前失败行，再重新生成这些行；
- `--resume` 与 `--overwrite` 互斥；
- 恢复时必须使用同一数据文件、行序、模型、task 和关键推理配置；
- 只有一个中间文件存在、两个文件行数不一致或 line number 重复时会拒绝恢复。

如果只是评分逻辑升级而模型预测已经完整，不要重新加载模型，使用离线 rescore。

## 输出文件

每个 `eval-custom` 输出目录固定包含：

```text
eval_output/
  predictions.jsonl  原始记录 + sample_id/task/response/labels；可离线重评分
  results.csv         每样本 prediction/reference/error 和各项分数
  scores.csv          按 task 汇总的一行或多行指标
  scores.json         状态、配置、阈值、输出路径和汇总指标
```

异常中止但已处理过样本时，工具会先写部分 `scores.csv/json`，此时 `status` 为 `partial`。正常走完时为 `complete`。启用 `--continue-on-error` 后，坏样本的错误写入 `eval_error`/`error`，并以 0 分计入汇总分母，不会被静默跳过。

## 离线重新评分

`rescore_custom_predictions.py` 读取 `eval-custom` 的 `predictions.jsonl`，不加载模型，按当前评分逻辑重建完整四件套。输出到新目录，保留原始结果便于审计：

```bash
python scripts/benchmark-tool/rescore_custom_predictions.py \
  --predictions-file /path/old_run/predictions.jsonl \
  --output-dir /path/rescored_run \
  --model-name ours \
  --dataset-name robovqa_smoke_256 \
  --task vqa \
  --progress-every 100 \
  --overwrite
```

Grounding/grounded 重评分还需保证 `objects` 和相对媒体路径仍可访问；必要时传 `--dataset-dir`、`--prediction-space` 和 `--iou-threshold`。

对于其他已有 prediction 文件，可使用 `score-custom`。输入记录需包含一个预测字段（如 `response`/`prediction`）和一个参考字段（如 `labels`/`answer`）：

```bash
vlm-eval score-custom \
  --model ours \
  --dataset robovqa_val \
  --input-file /path/predictions.jsonl \
  --metric answer_accuracy semantic_similarity yes_no_accuracy freeform_accuracy freeform_similarity \
  --output-file /path/robovqa_scores.csv
```

`score-custom` 输出标准化单行 CSV，适合 `visual --result-file`；它不生成 `eval-custom` 的逐样本 `results.csv`。旧 `acc` 是严格文本相等，`rouge/nlg` 调用 `swift.metrics.compute_rouge_bleu`。

## Description 和 Grounding 指标

Description：

- `rouge_l`：最长公共子序列 F1；
- `bleu_4`：平滑 sentence BLEU-4；
- `chrf`：字符 n-gram F 分数；
- `token_f1`：token 重叠诊断值。

Grounding：

- `mean_iou`：每个 GT 框的最佳一对一匹配 IoU 均值；
- `iou_accuracy`：IoU 达到 `--iou-threshold` 的 GT 框比例；
- `grounding_precision` / `grounding_f1`：按阈值命中的框级指标；
- `sample_accuracy`：预测框数量和全部命中均满足要求的整样本准确率；
- `parse_success_rate`：模型输出至少解析到一个框的样本比例。

无法解析的框按 0 分进入分母。Qwen-VL 默认预测空间为 `norm1000`；其他模型按实际输出传 `--prediction-space norm1|real`。`grounded` 同时报告 description 和 grounding，不人为合并成一个加权总分。

## 正式 benchmark runner

当前注册表：

| Benchmark | 默认指标 | 后端 |
| --- | --- | --- |
| Flickr30k | Recall@1/5/10 | 外部官方 runner |
| OpenEQA | LLM Judge；可选 Human Judge | 外部官方 runner |
| EgoPlan-Bench | Accuracy | 外部官方 runner |
| RoboSpatial | Accuracy、IoU | 外部官方 runner |
| OCRBench | Score | ms-swift/VLMEvalKit |
| RealWorldQA | Accuracy | ms-swift/VLMEvalKit |
| Video-MME | Accuracy | ms-swift/VLMEvalKit |

VLMEvalKit 内置项示例：

```bash
vlm-eval run \
  --model baseline=/mnt/luojunkun/stage1/model \
          ours=/mnt/luojunkun/stage1/sft-model/7-30_stage1_modelv1 \
  --benchmark ocrbench realworldqa video-mme \
  --infer-backend transformers \
  --output-dir /mnt/luojunkun/stage1/benchmark-eval-result/official
```

外部 benchmark 先复制 `benchmark-runners.example.json` 并替换仓库 root、正式入口参数和结果 selector：

```bash
vlm-eval run \
  --model ours=/path/model \
  --benchmark flickr30k openeqa egoplan-bench robospatial \
  --runner-config scripts/benchmark-tool/benchmark-runners.json \
  --output-dir /path/official-results \
  --dry-run
```

确认 `--dry-run` 打印的 cwd、命令和结果路径正确后移除该参数。可用占位符：

```text
{python}           当前 Python
{model}            模型显示名
{model_name}       模型显示名
{model_path}       权重路径
{base_model}       可选基础模型
{benchmark}        标准 benchmark 名
{benchmark_name}   benchmark 显示名
{benchmark_root}   外部仓库根目录
{output_dir}       总输出目录
{run_dir}          当前 model/benchmark 输出目录
{official_result}  默认官方 JSON 路径
{stdout_log}       标准输出日志
{stderr_log}       标准错误日志
```

runner 始终以参数数组和 `shell=False` 启动。API key 等敏感值通过启动进程的环境变量传入。

结果 selector 支持 JSON Pointer、CSV 列和文本正则。例如：

```json
{
  "result": {
    "file": "{run_dir}/official_result.json",
    "format": "json",
    "metrics": {"accuracy": "/metrics/accuracy"}
  }
}
```

未配置 selector 时，工具只接受按注册别名找到的唯一数值；同名指标存在多个不同值时会要求显式配置，不会擅自求平均。

## 服务器本地 benchmark 流程

`prepare_local_benchmarks.py` 支持 `flickr30k`、`ocrbench_v2`、`realworldqa`、`egoplan`、`openeqa`、`robospatial` 和 `video_mme`：

```bash
python scripts/benchmark-tool/prepare_local_benchmarks.py video_mme \
  --source-root /mnt/luojunkun/stage1/benchmark-stage1 \
  --output-root /tmp/stage1-benchmark-smoke \
  --limit 3 \
  --overwrite
```

它生成 `<output-root>/<benchmark>/eval.jsonl` 并按需提取媒体。模型推理仍用 `eval-custom`；完成后再运行：

```bash
python scripts/benchmark-tool/score_local_benchmark.py video_mme \
  --predictions /path/model/video_mme/predictions.jsonl \
  --output-file /path/model/video_mme/official_result.json
```

输出中的 `metric_scope` 说明该分数是 official-compatible、caption-generation 还是 diagnostic。没有相应官方 judge 的本地代理不能写成官方主指标。

## 队列脚本

这些 shell 脚本记录了服务器特定路径、GPU 和阶段状态，不是通用 CLI：

| 脚本 | 用途 |
| --- | --- |
| `run_robovqa_smoke_256.sh` | RoboVQA 256 行 baseline/ours 快测；当前并行启动两模型 |
| `run_custom_eval_queue.sh` | 依次跑 Stage 1 自建验证集，随后可接官方队列 |
| `run_official_eval_queue.sh [benchmark]` | 准备、本地推理、确定性评分；指定参数时只跑一个 benchmark |
| `run_target_benchmark_queue.sh` | 先做 Video-MME smoke，通过后再启动完整队列，并用文件锁防重复 |
| `run_parallel_resume_queue.sh` | 在多张 GPU 上恢复多个官方本地任务 |
| `run_llava_resume_queue.sh` | LLaVA 自建验证集恢复脚本 |
| `validate_stage1_results.py` | 严格检查预期 benchmark、模型、状态、样本数和 metric scope |
| `generate_stage1_report.py` | 输出 `all-results.csv`、HTML、Markdown 和 benchmark CSV |

多个队列文件中的 `OURS_WEIGHTS` 仍指向历史 `v30/.../checkpoint-400`，`validate_stage1_results.py` 的 `EXPECTED_WEIGHTS` 也硬编码了该旧路径。本次目标权重是 `/mnt/luojunkun/stage1/sft-model/7-30_stage1_modelv1`；这些服务器特化脚本不能原样用于本次模型，启动前必须同步核对权重、数据、结果根目录和 validator 预期。`.pre-*-20260801`、`README copy.md` 和 `tests/results.zip` 是历史快照，不要执行。

## 汇总和可视化

对 `vlm-eval run` 或标准化 CSV 结果：

```bash
vlm-eval visual \
  --model baseline ours \
  --benchmark flickr30k robospatial ocrbench \
  --results-dir /path/results \
  --output-dir /path/comparison
```

输出 `summary.csv` 和 `comparison.png`；没有 matplotlib 时回退 SVG。Stage 1 服务器报告使用：

```bash
python scripts/benchmark-tool/validate_stage1_results.py \
  --result-root /mnt/luojunkun/stage1/benchmark-eval-result/stage1-autoeval/official-local

python scripts/benchmark-tool/generate_stage1_report.py \
  --stage-root /mnt/luojunkun/stage1
```

严格校验失败时不要生成或发布最终报告，先修复缺失、失败或 scope 不正确的任务。

## 常见问题

- `exact_match=0` 但回答含义正确：查看 `answer_accuracy` 和对应类型子指标，不要用 exact match 作为 RoboVQA 主指标。
- PPU/CUDA 退出码 134 或固定视频形状崩溃：先用 batch 1-2、16 帧、`eager` 定位；这不一定是显存不足。保留输出后用 `--resume --retry-errors`。
- OOM：依次降低 batch、视频帧数、图像像素和生成长度；不要只改变某一个模型的参数。
- `status=complete` 但结果不可用：检查每个 task 汇总的 `failed_samples`，以及 baseline/ours 的 sample ID 是否对齐。
- grounding 全为 0：确认 `--task grounding|grounded`、参考答案确实要求输出 bbox，并核对 `objects.bbox` 与 `--prediction-space`。
- resume 被拒绝：确认 `predictions.jsonl` 和 `results.csv` 同时存在且行数一致；不要把其他模型或数据集的输出目录拿来恢复。
- 重评分后分数没变化：确认输入是包含 `response` 和 `labels` 的 `predictions.jsonl`，并查看 `scores.json` 中的阈值和 `rescore_source`。
