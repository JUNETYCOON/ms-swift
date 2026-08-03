# VLM Benchmark 评测与可视化工具最新开发方案

> 版本日期：2026-07-28
>
> 代码目录：`ms-swift/scripts/benchmark-tool`
>
> 命令名称：`vlm-eval`

## 1. 项目定位

开发一个以训练权重为输入的 VLM benchmark 调度、结果标准化和可视化工具。

平台使用者只需要提供：

- 模型名称和训练权重路径。
- 一个或多个 benchmark 名称。
- 外部 benchmark 已配置好的 runner 文件。

平台负责：

- 调用 ms-swift/VLMEvalKit 或 benchmark 官方程序。
- 保存每次运行的 stdout、stderr 和官方原始报告。
- 从官方报告中提取指标，生成统一 `score.csv`。
- 汇总多个模型、多个 benchmark 的 `summary.csv`。
- 生成柱状对比图。

正式 benchmark 的评分公式只能来自官方实现。本工具不根据逐样本输出重新定义或近似官方分数。

## 2. 当前实现状态

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| benchmark/metric 注册表 | 已完成 | 已登记 7 个 benchmark、指标别名、默认指标和后端 |
| 多模型、多 benchmark CLI | 已完成 | `vlm-eval run` 顺序执行模型与 benchmark 组合 |
| ms-swift/VLMEvalKit 后端 | 已完成 | OCRBench、RealWorldQA、Video-MME |
| 外部官方命令执行框架 | 已完成 | 支持工作目录、命令模板、环境变量、超时和日志 |
| Flickr30k 官方命令配置 | 待接入 | 需要根据已下载仓库的实际入口填写配置 |
| OpenEQA 官方命令配置 | 待接入 | 需要配置 judge 模型/API 和结果字段 |
| EgoPlan-Bench 官方命令配置 | 优先接入 | 仓库已下载，需要确认真实评测入口和结果文件 |
| RoboSpatial 官方命令配置 | 待接入 | 需要确认 QA 与 bbox 两类官方评测入口 |
| JSON/JSONL/CSV/文本结果提取 | 已完成 | 支持 JSON Pointer、CSV 列、行过滤和正则 |
| 指标多义性检查 | 已完成 | 同一指标出现多个不同值时拒绝自行平均 |
| 标准 CSV 与柱状图 | 已完成 | 无 matplotlib 时自动输出 SVG |
| 自建 val 文本与 grounding | 已完成 | 与 7 个正式 benchmark 隔离 |
| 跳过已成功任务/断点恢复 | 待开发 | 当前只有 `--continue-on-error` |
| HTML 与错误分析报告 | 待开发 | 当前输出 CSV 和 PNG/SVG |

当前代码框架已经可运行，但四个外部 benchmark 的样例命令仍是占位符。在真实官方入口、参数和结果路径完成配置前，
不能把样例配置视为可直接执行的官方评测。

## 3. Benchmark 范围

| Benchmark | 任务 | 官方输出 | 注册指标 | 当前后端 |
| --- | --- | --- | --- | --- |
| Flickr30k | Caption 短语定位 | bbox | Recall@1/5/10 | 外部官方接口 |
| OpenEQA | 第一视角环境问答 | 开放式文本 | LLM/Human Judge | 外部官方接口 |
| EgoPlan-Bench | 第一视角视频下一步规划 | 动作选择/文本 | Accuracy | 外部官方接口 |
| RoboSpatial | 机器人场景空间关系和定位 | QA、bbox | Accuracy、IoU | 外部官方接口 |
| OCRBench | OCR 和文档视觉问答 | 自由文本 | Score，可选 Accuracy | ms-swift/VLMEvalKit |
| RealWorldQA | 真实世界图像问答 | 选项/短答案 | Accuracy | ms-swift/VLMEvalKit |
| Video-MME | 视频多选理解 | 选项 | Accuracy | ms-swift/VLMEvalKit |

查看代码中实际注册内容：

```bash
vlm-eval list-benchmarks --details
```

## 4. 职责边界

允许的标准化行为：

- 将 `Recall@1`、`R@1` 等官方字段统一为 `recall_at_1`。
- 将 `75%` 或明确的百分制 Accuracy 标准化为内部数值 `0.75`。
- 按配置定位嵌套 JSON、CSV 行或日志中的官方分数。
- 对同一模型、同一 benchmark 的标准记录进行展示。

禁止的行为：

- 自行实现 Flickr30k Recall@K 并冒充官方结果。
- 自行替代 OpenEQA 的 LLM/Human Judge。
- 自动把 OCRBench 1000 分制 Score 换算为 Accuracy。
- 对 Video-MME 不同字幕模式或时长子集擅自求平均。
- 用自建 grounding IoU 结果替代 Flickr30k 或 RoboSpatial 官方评分。

## 5. 系统架构

```text
训练权重
   |
   v
vlm-eval run
   |
   +-- ms-swift/VLMEvalKit ---------- OCRBench / RealWorldQA / Video-MME
   |
   +-- 外部 runner 配置 ------------- Flickr30k / OpenEQA / EgoPlan / RoboSpatial
   |
   v
官方预测与官方评分报告
   |
   v
official_results.py
   |
   v
score.csv -> summary.csv -> comparison.png/svg
```

主要模块：

| 文件 | 职责 |
| --- | --- |
| `cli.py` | `run`、`visual`、`score-custom`、`score-grounding`、`list-benchmarks` |
| `registry.py` | benchmark、指标、别名、默认指标、调用后端和参考资料 |
| `benchmark_runner.py` | 生成命令、注入权重、启动子进程、超时和错误处理 |
| `swift_eval_runner.py` | 在独立进程调用 `EvalArguments + eval_main` |
| `official_results.py` | 提取官方报告并输出单项 `score.csv` |
| `loader.py` | 加载和聚合标准 CSV/JSON/JSONL |
| `visualizer.py` | 输出汇总表和有效 benchmark/metric 面板 |
| `benchmark_metrics.py` | 自建 val 的 ms-swift acc/rouge |
| `grounding.py` | 自建 grounding val 的 IoU |

## 6. 目标环境和目录

建议 Linux/GPU 环境目录：

```text
/mnt/luojunkun/stage1/
  ms-swift/
  model/                                  # 基础模型或合并后的完整权重
  benchmark-stage1/
    egoplan/                              # 已下载的 EgoPlan 仓库
    flickr30k/                            # 以实际目录为准
    openeqa/
    robospatial/
  benchmark-runner-config/
    benchmark-runners.json
  benchmark-results/
```

环境安装：

```bash
cd /mnt/luojunkun/stage1/ms-swift
python -m pip install -e '.[eval]'
```

如果镜像中只有 `/usr/local/bin/swift`，但 Python 无法导入本地 `swift`，启动前设置：

```bash
export PYTHONPATH="/mnt/luojunkun/stage1/ms-swift:${PYTHONPATH:-}"
python -c 'import swift; print(swift.__file__)'
```

Qwen3-VL-4B 使用 BF16 推理时，单张 48GB L20 足以运行普通图像 benchmark。多卡优先部署多个单卡副本或并行运行
不同 benchmark，不建议为 4B 模型默认使用跨卡 Tensor Parallel。Video-MME 需要单独限制帧数、像素和并发。

## 7. 标准运行流程

### 7.1 内置 VLMEvalKit benchmark

```bash
vlm-eval run \
  --model qwen3_vl_4b=/mnt/luojunkun/stage1/model \
  --benchmark ocrbench,realworldqa,video-mme \
  --infer-backend vllm \
  --output-dir /mnt/luojunkun/stage1/benchmark-results/qwen3_vl_4b
```

如果输入是 LoRA adapter：

```bash
vlm-eval run \
  --model qwen3_vl_4b=/path/to/adapter \
  --base-model /mnt/luojunkun/stage1/model \
  --benchmark realworldqa
```

### 7.2 外部 benchmark

外部 benchmark 的 runner 配置由平台维护者准备一次，普通使用者运行时仍只需要提供权重。

```bash
vlm-eval run \
  --model qwen3_vl_4b=/mnt/luojunkun/stage1/model \
  --benchmark egoplan-bench \
  --runner-config /mnt/luojunkun/stage1/benchmark-runner-config/benchmark-runners.json \
  --output-dir /mnt/luojunkun/stage1/benchmark-results/qwen3_vl_4b
```

正式执行前必须先运行：

```bash
vlm-eval run \
  --model qwen3_vl_4b=/mnt/luojunkun/stage1/model \
  --benchmark egoplan-bench \
  --runner-config /mnt/luojunkun/stage1/benchmark-runner-config/benchmark-runners.json \
  --dry-run
```

### 7.3 已有结果可视化

```bash
vlm-eval visual \
  --model model_a,model_b \
  --benchmark flickr30k,egoplan-bench,robospatial,ocrbench \
  --results-dir /mnt/luojunkun/stage1/benchmark-results \
  --output-dir /mnt/luojunkun/stage1/benchmark-results/comparison
```

## 8. EgoPlan-Bench 优先接入方案

已知仓库路径：

```text
/mnt/luojunkun/stage1/benchmark-stage1/egoplan
```

第一步先确认该版本官方入口，而不是直接使用样例中的 `OFFICIAL_EVAL_ENTRY.py`：

```bash
cd /mnt/luojunkun/stage1/benchmark-stage1/egoplan
find . -maxdepth 3 -type f \( -iname '*eval*.py' -o -iname '*test*.py' -o -iname '*infer*.py' \)
find . -maxdepth 3 -type f \( -iname 'README*' -o -iname '*.yaml' -o -iname '*.json' \)
```

需要从官方仓库确认四项信息：

1. 模型如何接入：本地权重、Transformers 类、OpenAI 兼容 API，还是官方模型 adapter。
2. 数据集路径和 split 如何传入。
3. 推理和官方 Accuracy 计算由哪个命令完成。
4. 最终官方指标写入哪个 JSON/CSV/日志字段。

确认后将配置落为：

```json
{
  "benchmarks": {
    "egoplan_bench": {
      "root": "/mnt/luojunkun/stage1/benchmark-stage1/egoplan",
      "command": [
        "{python}",
        "实际官方入口.py",
        "--model-path",
        "{model_path}",
        "--output-file",
        "{run_dir}/official_result.json"
      ],
      "result": {
        "file": "{run_dir}/official_result.json",
        "format": "json",
        "metrics": {
          "accuracy": "/实际官方Accuracy路径"
        }
      }
    }
  }
}
```

如果 EgoPlan 官方代码不能直接加载 Qwen3-VL 权重，需要增加一个薄适配层。优先顺序：

1. 复用官方已支持的 Hugging Face/VLM 接口。
2. 通过 `swift deploy` 暴露 OpenAI 兼容服务，由 benchmark 调用 API。
3. 最后才在 EgoPlan 仓库内增加 Qwen3-VL adapter，避免复制官方评分逻辑。

## 9. Runner 配置约定

命令必须使用字符串数组，执行器使用 `shell=False`：

```json
"command": ["{python}", "evaluate.py", "--model-path", "{model_path}"]
```

主要占位符：

```text
{python}             当前 Python
{model}/{model_name} 模型显示名
{model_path}         权重路径
{base_model}         可选基础模型
{benchmark}          标准 benchmark 名
{benchmark_root}     官方仓库根目录
{run_dir}            当前模型/benchmark 输出目录
{official_result}    默认官方报告路径
{stdout_log}         stdout 日志
{stderr_log}         stderr 日志
```

API Key 等敏感配置通过父进程环境变量传入，不写入 runner 配置或日志。

## 10. 官方结果契约

每个 scorer 最终生成：

```csv
model,benchmark,accuracy
qwen3_vl_4b,egoplan_bench,0.68
```

支持的提取方式：

- JSON/JSONL：JSON Pointer，例如 `/metrics/accuracy`。
- CSV：列名、`row_filter`、`row_index` 和可选 `scale`。
- 文本日志：捕获一个数值分组的正则表达式。
- 文件名包含通配符时，选择最新匹配结果。

未配置 selector 时会按注册别名自动查找。若同一指标匹配出多个不同值，立即失败并要求明确 selector。

## 11. 输出规范

```text
benchmark-results/
  qwen3_vl_4b/
    egoplan_bench/
      stdout.log
      stderr.log
      official_result.json
      score.csv
  summary.csv
  comparison.png 或 comparison.svg
```

可视化规则：

- 一个有效 benchmark/metric 组合对应一个面板。
- 每个面板用柱状图比较不同模型。
- 百分比指标显示为 0% 到 100%。
- Score、IoU 等保留注册表定义的数值尺度。
- `summary.csv` 中缺失指标保持空值。
- 整个面板无数据时显示 `missing`；单个模型缺值时当前为零高度柱，不能解读为真实 0 分。

## 12. 自建 val

自建验证集不经过七个正式 benchmark 的官方 runner：

```bash
vlm-eval score-custom \
  --model qwen3_vl_4b \
  --dataset val \
  --input-file outputs/val_predictions.jsonl \
  --metric acc
```

- `acc` 对齐 ms-swift 推理逻辑 `response == labels`。
- `rouge/nlg` 直接调用 `swift.metrics.compute_rouge_bleu`。
- 自建 bbox grounding 使用 `score-grounding`。

## 13. 验证与验收

已完成的自动验证：

- Python 编译检查。
- CLI 和模块包装入口检查。
- benchmark 别名与默认指标测试。
- JSON、CSV、文本结果 selector 测试。
- 多值冲突拒绝测试。
- 两个模型、两个模拟 benchmark 的端到端子进程测试。
- 无 matplotlib 环境的 SVG 输出测试。

真实 benchmark 验收标准：

1. 固定官方仓库 commit/tag 和依赖版本。
2. `--dry-run` 展示的命令、工作目录和结果路径正确。
3. 单模型、小样本运行成功并保留完整日志。
4. 工具提取值与官方报告逐项一致。
5. 同一模型重复运行结果可解释且不存在错误聚合。
6. 两个模型结果可生成统一 CSV 和柱状图。
7. 失败时返回非零退出码，并能从对应 stderr 定位原因。

建议执行：

```bash
python -m compileall scripts/benchmark-tool vlm_eval
python -m unittest discover -s scripts/benchmark-tool/tests -v
python scripts/benchmark-tool/cli.py list-benchmarks --details
python scripts/benchmark-tool/cli.py run --help
python scripts/benchmark-tool/cli.py visual --help
```

## 14. 下一阶段计划

### P0：打通一个真实闭环

1. 检查已下载 EgoPlan 仓库版本、依赖和官方入口。
2. 明确 Qwen3-VL-4B 的直接权重或 API 接入方式。
3. 完成 EgoPlan runner 配置。
4. 用少量样本核对官方 Accuracy。
5. 完成单模型全量 EgoPlan 评测。

### P1：补齐全部 benchmark

1. 接入 Flickr30k Recall@K。
2. 接入 OpenEQA LLM Judge，单独管理 judge 配置和费用。
3. 接入 RoboSpatial Accuracy/IoU。
4. 在目标环境验证 OCRBench、RealWorldQA、Video-MME。

### P2：生产化

1. 增加 `--resume` 或 `--skip-existing`，跳过已有成功结果。
2. 增加受控并发和 GPU 分配，避免多个 runner 抢占同一设备。
3. 记录 benchmark commit、模型 hash、依赖、命令和运行时间。
4. 增加多次运行的均值、方差和置信区间。
5. 增加 HTML 报告和 per-sample 错误分析链接。

## 15. 最终交付定义

平台达到可交付状态需要满足：

- 七个 benchmark 均有固定版本、可直接运行的 runner 或内置后端。
- 用户运行时只提供模型名称、权重路径和 benchmark 列表。
- 所有正式分数都能追溯到官方原始报告。
- 多模型结果可以统一汇总和可视化。
- 外部程序失败不会产生伪造或部分成功的标准分数。
- 文档覆盖安装、运行、结果解释、错误定位和扩展新 benchmark 的流程。
