# S1-UDF 来源与目标 Adapter 指南

本文说明如何用版本化 source adapter 把来源数据确定性地转换为 S1-UDF 1.0.0，再用 target
adapter 投影到训练或评估格式。当前包提供双向 adapter 契约、schema registry、通用转换 runner、
manifest 构建、校验、切分，以及内置 `ms-swift` source/target adapter；它**没有声明服务器上
各个原始数据集的 dataset-specific adapter 已实现**。下方模板中的来源解析和目标渲染仍必须
按实际 schema 实现并测试。

正式契约以以下代码为准：

- `schemas/record-1.0.0.schema.json`：单条 record 的字段和类型；
- `schemas/manifest-1.0.0.schema.json`：数据集 manifest；
- `validation.py`：JSON Schema 之外的引用、坐标、时间和 episode 语义检查；
- `schema_registry.py`：当前 record/manifest schema 版本注册；
- `split.py`：按 group、共享 asset 和相同 episode ID 的连通分量切分；
- `manifest.py`：扫描最终 record 文件并构建可对账 manifest；
- `ms_swift.py`：S1-UDF 与 ms-swift 间的有损投影边界；
- `adapter.py`：`AdapterSpec`、source/target 契约、注册表和显式插件加载；
- `conversion.py`：有界并行 runner、diagnostic、错误策略、报告和 writer 协议。

## 1. 推荐流水线

```text
只读 profile 来源
  -> 写下 schema 假设和允许损失
  -> convert-source 生成并校验未切分 S1-UDF JSONL
  -> split（共享 asset/episode 连通分量 + train/val）
  -> 分别 validate train/val
  -> build-manifest + validate-manifest --check-files
  -> convert-target 按目标显式投影
```

从仓库根目录执行：

```bash
python scripts/universal_dataset/cli.py profile /path/to/source \
  --sample-rows 100 --max-depth 8 --output source-profile.json

python scripts/universal_dataset/cli.py convert-source /path/to/source \
  --adapter my-source --plugin /path/to/plugin.py \
  --output converted.jsonl --dataset-name my-dataset \
  --workers 8 --batch-size 64 --max-pending-batches 16 \
  --error-policy error --report source-conversion.json

python scripts/universal_dataset/cli.py split converted.jsonl \
  --train-output train.jsonl --val-output val.jsonl \
  --val-ratio 0.02 --seed 42

python scripts/universal_dataset/cli.py validate train.jsonl --check-assets
python scripts/universal_dataset/cli.py validate val.jsonl --check-assets
python scripts/universal_dataset/cli.py build-manifest train.jsonl val.jsonl \
  --output manifest.json --dataset-name my-dataset --dataset-version v1
python scripts/universal_dataset/cli.py validate-manifest manifest.json --check-files

python scripts/universal_dataset/cli.py convert-target train.jsonl \
  --adapter my-target --plugin /path/to/plugin.py \
  --output target-output --workers 8 --batch-size 64 \
  --max-pending-batches 16 --error-policy error \
  --report target-conversion.json
```

不要把 profile 的采样结果当作完整 schema。实现前还应核对所有 shard、来源 split、
缺失值分支、媒体布局、坐标定义和时间单位。来源 schema 改变时，adapter 应失败并提示，
不能用宽泛的 `.get()` 静默产生空监督。

`--plugin` 加载的是可执行 Python，不是被动数据。主进程与多进程 worker 都会执行插件模块；
只运行经过审阅并固定 revision/hash 的可信代码，不要把数据集、网页或未知仓库中的 Python 文件
直接交给 CLI。`--option KEY=JSON` 可重复传入强类型选项，key 必须非空且唯一，value 必须是合法
JSON。

## 2. 身份契约

### 2.1 `id`、`group_id` 和 asset identity 不是一回事

| 身份 | 作用域 | 必须满足的性质 |
| --- | --- | --- |
| record `id` | 整个转换数据集 | 每条输出唯一、稳定、可由来源逻辑键重建 |
| `group_id` | 切分原子 | 共享图片、视频、episode 或任一底层媒体的记录不能跨组 |
| `assets[].id` | 单条 record 内 | 唯一，并供 region、track、conversation、episode 引用 |
| asset identity | 整个 JSONL | 用于发现不同 record 中其实相同的媒体 |

推荐用带数据集命名空间的来源逻辑键，而不是文件遍历序号：

```text
record id:  dataset:source-record-id
group id:   image:dataset:source-image-id
            video:dataset:source-video-id
            episode:dataset:source-episode-id
local id:   asset:source-media-id, entity:source-object-id, region:source-region-id
```

`stable_record_id(dataset_name, source_id)` 当前只拼接
`"<dataset_name>:<source_id>"`。调用方仍需保证 `source_id` 在数据集内稳定且唯一。复合键
应先用规范 JSON 序列化或 SHA-256 编码，避免字符串分隔符歧义。禁止使用：

- Python `hash()`，其结果可能随进程和运行环境变化；
- `uuid4()`、当前时间、进程号或 worker 编号；
- 未冻结文件列表上的行号、遍历顺序或 parquet row-group 顺序；
- QA ID、caption ID、bbox ID、逐帧 ID 作为媒体级 `group_id`。

同一来源记录展开成多条 QA、caption 或帧样本时，record `id` 可以加入稳定的子项 ID，
但 `group_id` 必须保持媒体或 episode 级一致。

### 2.2 媒体身份优先级

为每个 asset 保留稳定的 direct `uri`，或在资源仍位于容器/切片中时写 external `source_ref`；能承担成本时
填写文件内容的 `sha256`。身份优先级建议为：

1. 官方内容校验和或本地计算的 SHA-256；
2. 来源提供、跨 shard 唯一且版本内不变的 media ID，同时用于生成 `group_id`，并写入带 release
   范围且显式 `verified: true` 的 `assets[].source_identities[]`；
3. 规范化后的永久 URI、绝对路径或由 source_ref URI 与 `key/offset/length` 构成的身份；
4. 相对 URI，仅在其相对 JSONL 目录的语义已冻结时使用。

生产数据优先写绝对路径。若使用相对 URI，必须相对最终 JSONL 所在目录，而不是相对来源
annotation 文件；传给 `ensure_valid()` 的 `base_dir` 和后续 CLI 的解析基准必须与此一致。

`split.canonical_asset_identities()` 会同时登记非空 `assets[].sha256`、仅 `verified: true` 的命名空间化
`assets[].source_identities[]`、URI 别名、完整 source_ref 身份及其 SHA：Windows/UNC 路径按 Windows
规则规范化，POSIX 和本地相对路径按 JSONL 位置解析，远程 URL 保留 query 但去掉 fragment。远程
`file://host/path` 保留并规范化 authority，不同 host 不会因为 path 相同而碰撞；只有空 authority 或
`localhost` 按本地路径处理。这使同一 release 的路径别名仍能连接；任一连通别名声明不同 SHA 会失败。
它**不会读取文件并自动计算内容哈希**。因此：

- 镜像副本、软链接或不同路径下的同一内容，应由 adapter 写入相同 `sha256`；
- 来源 ID 的 namespace 必须包含数据集和必要的 release，禁止不同来源共享裸数字命名空间；
- 带临时签名参数的 URL 不应作为长期唯一身份，优先使用内容哈希或来源 media ID；
- 不要删除用于区分资源的稳定 query，例如 `...?id=A` 与 `...?id=B`；
- URI 只做身份与定位，不把用户凭据、Cookie、token 写入数据。

`provisional_media_group_id(media_identities, fallback)` 对一条来源记录内去重、排序后的
媒体身份集合做 SHA-256；没有媒体时返回 `text:<fallback>`。它只是行内 provisional hint，
不能替代全数据集共享媒体审计。

### 2.3 全局 connected component 后处理

多资产样本会形成传递关系。例如 record A 使用图片 1 和 2，record B 使用图片 2 和 3，
record C 使用图片 3；三者必须属于同一切分连通分量，即使初始 `group_id` 不同。

`split_jsonl(..., audit_assets=True)` 分两遍处理：

1. 以输入 `group_id` 为节点；同一规范 asset identity 或同一非空 `episode.id` 出现在两个 group 时做 union；
2. 对每个传递闭包生成确定性 component ID，并以
   `SHA-256(seed + NUL + component_id)` 决定 train/val；
3. 被合并的 record 会重写为 `group_id="s1cc:<sha256>"`，并在
   `extensions["s1_udf.split"]` 保存 `source_group_id` 与 `component_group_id`。

这一步是错误分组的全局补救，也是最终泄漏保护；正常 adapter 仍应尽量从源头赋正确
`group_id`。episode 被拆成多条且没有共同 asset 时，只要各条保留相同 `episode.id`，split 也会
自动合并它们；如果连 episode ID 都丢失，后处理才无法恢复关系。只有另有同等级全局媒体审计
时才允许关闭 `audit_assets`；关闭它只禁用 asset 扫描，相同 episode ID 仍会 union。

验收时检查 split 摘要中的 `merged_components`、`merged_groups`、`rewritten_records`。
非零不一定是错误，但必须解释来源，并把最终 component `group_id` 当作发布身份。切分后对
train 和 val 再运行 `validate`；该命令会报告重复 record ID、同 group 跨 split、同 asset
跨 group 冲突和 episode group/split 泄漏。

## 3. 多进程与确定性

`convert-source` 和 `convert-target` 已使用统一 runner，不需要每个插件手写进程池。runner 将输入
按 `--batch-size` 分批，并只保留至多 `--max-pending-batches` 个待处理 batch；未显式设置时默认
为 `max(1, workers * 2)`。完成时间不会改变输出顺序，一条输入可以按 adapter 返回顺序展开为
多条输出。

adapter 仍必须满足以下确定性约束：

1. `iter_source(source, context)` 按稳定 shard 和逻辑键产出 `(source_id, value)`，不能依赖临时遍历顺序；
2. `convert()` 无共享副作用，不写最终文件、不更新全局计数器，也不按 worker/完成顺序生成 ID；
3. context、options 和输入 value 必须可 pickle，插件必须能在 worker 进程中重新 import；
4. 一对多结果按稳定子项键排序，所有待落盘集合显式排序；
5. `workers=1/2/N` 应生成字节一致的输出和 diagnostic 顺序。

source runner 逐条校验 adapter 产生的 record，并检测全局重复输出 ID；target runner 在投影前校验
S1-UDF record，并检测重复输入 record ID。主进程以确定顺序消费结果。`MemoryError`、
`KeyboardInterrupt` 和其他进程级中断会重新抛出，不会伪装成普通行级转换错误。

## 4. 各层映射

### 4.1 `assets`

每个 asset 必须含 `id`、`kind`，并至少提供非空 direct `uri` 或 external `source_ref`；两者可同时存在。
图片、视频、音频、深度、mask、点云、mesh、tensor、表格等应是独立 asset，不要把多个逻辑资源塞入
一个未说明的路径字符串。

- 图片填写 `media.width/height/channels`；视频另填 `duration/fps/frame_count/time_base`；
- 音频填写 `duration/sample_rate`；深度可记录 `depth_scale/invalid_value`；
- 容器内成员、切片或张量用 `source_ref`/`dataRef` 的 `format/key/offset/length`；
- 具身媒体用 `sensor_id`、`frame_id`、`time_span` 连接 episode；
- 大数组使用 tensor 的 `ref`，不要为追求单文件而膨胀每条 JSONL。

### 4.2 `entities`、`regions`、`relations` 和 `tracks`

`entity` 表示同一 record 内的语义对象或实例；类别放入 `labels`，来源类别 ID 放
`labels[].id` 或 `source_id`，实例属性放 `attributes`。文本中每次提及不应自动新建 entity。

`region` 必须指向 `asset_id`，可指向 `entity_id`。图像区域通常没有 `frame_index`；视频
区域应给 0-based `frame_index` 或明确 `time_span`。mask 文件使用
`geometry.type="mask_ref"` 和 `geometry.asset_ref`；RLE 使用 `size=[height,width]` 和
`counts`。

`relation.subject/object` 各是只含一个键的 node ref，可引用 asset、entity、region 或
track。`track` 可声明一个 `asset_id` 或多个 `asset_ids`，也可绑定 entity；observations 按帧和时间
非递减排序。每个 observation 必须给 `frame_index`、`timestamp` 或 `time_point`，并采用 region 模式
（`region_id`）或显式 state 模式（`state` 字符串）。region 模式下 region 必须属于同一 track asset，
帧号必须一致；state 模式也必须由 observation 或 track 解析出 asset。不要把 track ID 当作视频 group ID。

### 4.3 `conversations`

conversation 适合已经定义好 SFT 表达的监督。message role 可为 `system`、`user`、
`assistant`、`tool`、`environment`；content 必须是结构化 part 数组：

- `text`：普通文本；
- `asset`：图片/视频/音频插入位置，可带帧或时间范围；
- `entity`：grounding 的对象引用；
- `region`：grounding 的区域引用；
- `tool_call`、`tool_result`、`custom`：仅在目标投影有明确策略时使用。

媒体插入顺序必须来自来源语义，不要把所有 asset 无条件放到问题开头。多轮对话保持原始
轮次、role 和 part 顺序。候选回答或偏好放 `message.candidates`，不能伪装成多个 assistant
轮次。candidate-only message 是合法 S1-UDF preference/ranking 数据，但不是可隐式挑选的 SFT target；
投影到单响应格式时必须使用专用 preference adapter 或明确失败。

### 4.4 `annotations`

annotation 用于保存不依赖某个 SFT prompt 的原生监督：

| 来源 | 推荐表达 |
| --- | --- |
| caption | `type="caption/captioning/dense_caption"`、`target.asset_id`、非空 `text` 或 `content` |
| VQA | `type="qa/vqa/document_qa"`、`question`、非空 `answers`、可选 choices/canonical index |
| 分类 | 自定义稳定 `type`、`target(s)`、`label_ids` 或 `value` |
| 检测/分割 | entities + regions，annotation 用 `entity_ids/region_ids` 聚合任务监督 |
| tracking | regions + tracks，annotation 用 `track_ids` 指定评估集合 |
| 时序标签 | `time_span`，并明确 reference asset 和 unit |

QA 不要提前丢弃答案分布：保留每个 answer 的 `text/content`、`count/score/weight`，并仅在
来源定义 canonical answer 时写 `canonical_answer_index`。grounding 问题或答案应使用
结构化 entity/region part，使引用关系可校验。

同一 record 同时有 conversations 和 annotations 是允许的，但导出到具体目标时必须显式
决定哪个是训练目标，不能假设两者会自动合并。

### 4.5 `episode`

一个具身 demo/trajectory 通常使用 episode ID 作为 `group_id`。`episode.id` 保留稳定来源
ID；`task/environment/agent` 保存任务和场景信息；`coordinate_frames` 构成无环 parent 树；
`sensors` 指向 frame 并记录 intrinsics/extrinsics。

内嵌 `steps` 时必须至少有一步，index 严格递增，timestamp 非递减；每个 observation 至少使用
`asset_id`、`value`、`tensor` 或 `data_ref` 之一，action 至少有 `value`、`tensor`、`data_ref` 或
`language`。大型步骤表用 `steps_ref + step_count`，其中 `step_count > 0`；禁止用空 `steps` 代替外部
引用。terminal 只能出现在最后一步且必须同时是 last，`is_first/is_last/is_terminal` 必须和实际边界一致。

观测 asset、sensor、frame，action target/frame 都必须能在同一 record 内解析。若来源把
一个 episode 拆到多个容器，使用 provenance/dataRef 回链，但 group 仍以完整 episode 为单位。

## 5. 坐标、帧和时间

adapter 不得从字段名猜测坐标语义。每个 geometry 显式写：

```json
{
  "type": "bbox2d",
  "format": "xyxy",
  "coordinates": [x0, y0, x1, y1],
  "coordinate_space": {"type": "pixel", "image_size": [width, height]}
}
```

规则：

- bbox `format` 明确为 `xyxy`、`xywh` 或 `cxcywh`；不要依赖默认值；
- 每个 pixel 2D geometry 必须能从 `coordinate_space.image_size` 或来源 image asset 的
  `media.width/height` 得到可靠尺寸并验界；缺少尺寸必须失败，同时注明 crop/resize 前后是哪一空间；
- normalized 使用 `[0,1]`，percent 不可冒充 normalized；
- 3D/机器人坐标写 `frame_id`、unit、axes、origin、handedness 和旋转约定；
- 来源 1-based 帧号先规范为 0-based `frame_index`，原值放 metadata/provenance；
- `time_span` 总是写 `start/end/unit`，视频片段尽量写 `reference_asset_id`；
- PTS 使用 `unit="pts"` 和 `time_base=[num,den]`，不要未经说明转成浮点秒；
- 保存来源闭区间/半开区间约定，转换端点时加入可审计 metadata；
- 不允许 NaN 或 Infinity。

当前校验器会检查常见 2D 形状、RLE/mask_ref 必填字段、normalized/pixel 范围、bbox 正宽高、
asset 时长/帧数边界、track 顺序、frame parent 缺失/成环和 episode 顺序。tensor inline data
还会检查矩形嵌套、实际 shape、元素数量和 `-1` 推断维度；custom/复杂 3D 语义、dtype 值域及
外部 tensor 内容仍需要 adapter 的领域测试。

## 6. Provenance、raw 和扩展字段

每条 record 的 provenance 最少必须记录：

```json
{
  "provenance": {
    "dataset": "source-name",
    "dataset_version": "source-version",
    "source_split": "source-split",
    "source_record_id": "stable-source-id",
    "source_files": [{"uri": "/path/to/annotations.json", "format": "json"}],
    "revision": "source-revision",
    "license": "source-license",
    "conversion": {
      "tool": "source-to-s1-udf",
      "version": "1.0.0",
      "code_revision": "git-commit",
      "arguments": {"preserve_raw": true}
    },
    "raw_record": {}
  }
}
```

最小必填字段只有 `dataset`、`source_record_id` 和 `conversion.tool/version`。示例中的
`dataset_version`、`revision`、source split、license、source files 等只在来源确实提供或能够核验时填写，
不得为满足 schema 虚构版本或 revision。

`AdapterContext.preserve_raw` 决定是否内嵌原记录。小记录可深拷贝到 `raw_record`；大型记录、
二进制内容或重复 episode 步骤使用 `raw_ref`/`source_files`，写明容器、key、offset 和 checksum。
不要在 raw 中保留密钥、签名 token 或非有限数。

核心 schema 已能表达的语义必须映射到核心字段，不能只藏在 raw。来源特有且仍需查询的字段
放在 `extensions`，使用命名空间键，例如 `source_name.annotation`; 一般描述性附加值放
`metadata`。raw 是回溯证据，不是跳过建模的借口。

## 7. 错误、warning 和损失策略

每个 adapter 在代码和 manifest 中冻结一份字段消费表：来源字段 -> S1-UDF 字段 -> 目标投影
状态。状态至少区分 `preserved`、`normalized`、`raw-only`、`target-dropped`、`fatal`。

以下情况应抛异常或返回 `severity="error"` diagnostic；默认 `error` 策略会终止原子输出：

- 缺少稳定 source/media/episode identity；
- 媒体不存在、引用悬空、局部 ID 重复；
- 坐标格式、坐标空间、帧基或时间单位不明确；
- 非有限数、负宽高、越界且来源没有明确裁剪规则；
- 多答案、多个监督载体或不支持 geometry 会导致目标含义不唯一；
- 未知来源 schema/version 或必填字段类型改变。

优先把可审计事件写成 `AdapterDiagnostic(stage, code, severity, message, ...)`。severity 可为
`info`、`warning` 或 `error`；`AdapterResult.warnings`/`TargetAdapterResult.warnings` 仅为兼容入口，
runner 会将其转换成 `code="adapter_warning"` 的 warning diagnostic。使用稳定 code、source locator
和 record ID，不要让 diagnostic 顺序依赖 worker 完成顺序。被允许只保留在 raw 或目标格式丢弃
的字段必须进入转换报告和 manifest statistics/extensions，不能静默忽略。

`convert-source` 和 `convert-target` 的 `--error-policy` 行为一致：

- `error`：exception 或 `severity="error"` diagnostic 使转换失败；默认原子 writer 不提交部分输出；
- `skip`：跳过产生问题的完整 input item，而不是只丢其中一条派生输出；继续处理后状态为 `complete_with_skips`；
- 两种策略都不会吞掉 `MemoryError`、键盘中断或其他进程级中断。

结构化 report 包含 adapter spec、input/output/skipped 数量、severity/code 聚合、有限 examples、
fatal error 和并行参数。`--max-diagnostic-examples` 只限制样例，不限制计数。input、output、report
必须是不同路径；已有 output/report 默认拒绝，显式 `--overwrite` 同时控制二者。source 默认 writer
和 target 默认 JSONL writer 使用临时文件原子提交。

### ms-swift 投影前的显式决策

`record_to_ms_swift()` 的默认策略倾向失败而非猜测：

- conversations 与 annotations 同时存在时，显式选择 `annotation_policy`：`error`、
  `prefer-conversations` 或 `prefer-annotations`；
- QA 多答案且无 canonical index 时，显式选择 `answer_policy`：`error`、`first` 或
  `highest-count`；
- QA `choices` 按原数组顺序确定性渲染为 `Choices:` 和从 1 开始的编号；choice 的 `text` 与
  结构化 `content` 只能选一种，空 question、caption、answer 或 choice 必须失败；
- caption 需要 `metadata.sft_prompt` 或显式 `caption_prompt`；
- 只有 caption/QA annotation 有隐式 SFT 映射；tracking、segmentation、episode 等必须先用
  任务专用 renderer 生成 conversation；
- ms-swift grounding 仅能直接承载 image 上 pixel/normalized 的 point2d/bbox2d；polygon、
  mask/RLE、3D 不可伪装成 bbox；
- message candidates 是合法 S1-UDF preference 数据，但不能隐式选成单响应 SFT target；内置导出器
  会失败，并要求 conversation 至少有一个非空、可渲染的 assistant content；
- 只有 `source_ref`、没有 direct `uri` 的 asset 必须先 materialize，或交给目标专用 resolver；
- `--drop-ids` 只删除 record `id`，内置导出始终保留 canonical `group_id`；
- 导入 ms-swift 的 real bbox 时，用 Pillow 探测每张被引用本地图片的真实宽高；远程 URI、缺失/损坏
  图片或 Pillow 缺失均 fail closed。`bbox_type="norm1"` 不需要此探测；
- 部分工具/custom content、非 image/video/audio 插入和 asset 插入的 frame/time 信息可能不可表示。

导出前调用 `ms_swift_projection_warnings()`，将 warnings 写入报告。它会提示被忽略的监督、
多答案策略、未表示的 relations/tracks/episode 和媒体插入时间损失。warning 清单不是完整的
自动 loss proof；adapter 仍需维护字段消费表和目标级 golden tests。

## 8. Python source/target adapter skeleton

下面模板使用当前真实接口。`source_key()`、`map_source()` 和 `render_target()` 是必须替换并测试的
数据集/目标逻辑；模板本身不表示支持任何具体数据集。

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

from universal_dataset.adapter import (
    AdapterContext,
    AdapterResult,
    AdapterSpec,
    SourceAdapter,
    TargetAdapter,
    TargetAdapterResult,
    TargetContext,
    register_source_adapter,
    register_target_adapter,
)
from universal_dataset.io import iter_jsonl


def source_key(value: Mapping[str, Any]) -> str:
    """SOURCE-SPECIFIC: return an immutable logical key, never a row number."""
    raise NotImplementedError


def map_source(
    source_id: str, value: Mapping[str, Any], context: AdapterContext
) -> Dict[str, Any]:
    """SOURCE-SPECIFIC: build one complete S1-UDF record."""
    raise NotImplementedError


def render_target(
    record: Mapping[str, Any], context: TargetContext
) -> List[Dict[str, Any]]:
    """TARGET-SPECIFIC: render one S1-UDF record to one or more target values."""
    raise NotImplementedError


@register_source_adapter
class MySourceAdapter(SourceAdapter):
    spec = AdapterSpec(
        name="my-source",
        version="1.0.0",
        direction="source",
        udf_versions=("1.0.0",),
        capabilities=("image", "qa"),
        loss_contract=("unmapped source fields are retained only when preserve_raw is true",),
        description="Convert the audited my-source schema to S1-UDF.",
    )

    def iter_source(
        self, source: Path, context: AdapterContext
    ) -> Iterator[Tuple[str, Mapping[str, Any]]]:
        for _, value in iter_jsonl(source):
            source_id = source_key(value)
            if not source_id:
                raise ValueError("source_key() returned an empty identity")
            yield source_id, value

    def convert(
        self, source_id: str, value: Mapping[str, Any], context: AdapterContext
    ) -> AdapterResult:
        return AdapterResult(records=[map_source(source_id, value, context)])


@register_target_adapter
class MyTargetAdapter(TargetAdapter):
    spec = AdapterSpec(
        name="my-target",
        version="1.0.0",
        direction="target",
        udf_versions=("1.0.0",),
        capabilities=("image", "qa"),
        loss_contract=("only the selected QA supervision is rendered",),
        description="Render the supported S1-UDF subset to my target format.",
    )

    def convert(
        self, record: Mapping[str, Any], context: TargetContext
    ) -> TargetAdapterResult:
        return TargetAdapterResult(values=render_target(record, context))
```

`convert-source` 会校验每个 `AdapterResult.records` 元素并拒绝空结果或重复输出 record ID；
`convert-target` 会先校验每个输入 record，并拒绝空 `TargetAdapterResult.values` 和重复输入 ID。
adapter 可返回 `AdapterDiagnostic` 列表；不要在插件里重复实现 runner 已有的原子 JSONL 写入、
进程池或报告逻辑。

每个 adapter 注册时必须支持当前 S1-UDF `1.0.0`，否则立即失败。registry 以 direction/name 管理，
因此 source 和 target 可同名；当前 CLI 不执行 schema migration，也不允许注册同方向同名的多个
adapter 版本。`capabilities` 和 `loss_contract` 是机器可读声明与审计线索，不是自动无损证明。

插件注册后可检查：

```bash
python scripts/universal_dataset/cli.py list-adapters --plugin /path/to/plugin.py
python scripts/universal_dataset/cli.py inspect-adapter my-source \
  --direction source --plugin /path/to/plugin.py
```

`TargetAdapter.write_output()` 默认完整消费有序 stream 并原子写 JSONL。只有聚合 JSON、Parquet、
多文件或二进制目标才应覆盖它；自定义实现必须按顺序完整消费 iterable、返回消费数量，并自行
使用临时路径实现原子提交和失败回滚。runner 会检测提前返回、负数/非整数和计数不一致，但不能
撤销 writer 已经提交的外部副作用。

## 9. Manifest 生成

manifest 在最终 split 文件写完后生成。不要手工填写 count、checksum 或聚合统计，直接扫描实际
record 文件：

```bash
python scripts/universal_dataset/cli.py build-manifest train.jsonl val.jsonl \
  --record-split train.jsonl=train --record-split val.jsonl=val \
  --output manifest.json --dataset-name source-name --dataset-version source-version \
  --revision source-revision \
  --source-schema '{"name":"audited-schema","version":"v1"}' \
  --provenance '{"adapter":"my-source","code_revision":"git-commit"}'

python scripts/universal_dataset/cli.py validate-manifest manifest.json --check-files
```

`build-manifest` 默认逐 record 校验，自动生成 record file 的相对路径、format、split、count 和
SHA-256，汇总 `task_types`、`splits` 及 `statistics.records/groups/unique_assets/episodes`。生成的
grouping identity 明确包括：

```json
{
  "identity_fields": [
    "group_id",
    "assets[].sha256",
    "assets[].source_identities[]",
    "assets[].uri",
    "assets[].source_ref",
    "episode.id"
  ]
}
```

manifest 的必填审计字段为 `format`、`schema_version`、`dataset`、`record_files`、`task_types`、
`grouping` 和 `statistics`；每个 record file 必须有 path/format/count/SHA-256，statistics 必须完整包含
records/groups/unique_assets/episodes。`splits` 仅对真正未切分的数据集可省略；只要 record 或 file 声明
split，就必须与扫描汇总一致。

构建器要求所有 record file 合计至少一条记录，并拒绝混合已切分/未切分记录、asset 或 episode 的
group/split 冲突，以及 manifest 输出和任一 record 文件路径冲突。单个空 split 文件必须通过可重复的
`--record-split PATH=SPLIT` 显式登记；这样 `record_files[].count=0` 和 `splits.<name>=0` 都能被 manifest
及后续 `--check-files` 对账。显式 split 也会核对非空文件内每条记录的 `split`。写入前还会校验生成的
manifest schema。`--source-schema JSON` 可重复，`--provenance JSON` 必须是对象。只有上游已经完成
等价校验时才传 `--no-record-validation`，需要核对本地媒体存在性时传 `--check-assets`。

目标 schema 未统一覆盖的 `coordinate_conventions`、taxonomy、split 摘要或损失统计，可通过当前
manifest schema 已声明的字段和 extensions 补充；内容必须从审计结果生成，不能从 worker 计数
猜测。随后执行：

```bash
python scripts/universal_dataset/cli.py validate-manifest manifest.json --check-files
```

`--check-files` 会核对文件存在性、checksum、count、JSONL 内声明 split、episode 泄漏、`splits`
汇总，以及 records/groups/unique_assets/episodes 统计。它不会替代逐 record 的 `validate`。

## 10. 测试清单

发布 adapter 前至少覆盖：

- [ ] profile 覆盖所有 shard、来源 split 和可选字段分支；
- [ ] 插件代码经过审阅并固定 revision/hash，未加载来源数据或网页携带的可执行代码；
- [ ] `AdapterSpec` direction/version/capabilities/loss contract 正确，注册时通过 UDF 1.0.0 handshake；
- [ ] 同一输入转换两次，输出 JSONL SHA-256 完全一致；
- [ ] `workers=1/2/N` 输出字节一致，diagnostic 顺序一致，pending batch 上限可控；
- [ ] record ID 全局唯一，局部 asset/entity/region/track/annotation ID 唯一；
- [ ] 同一图片、完整视频或 episode 的所有派生监督拥有同一 group；
- [ ] 多资产桥接的传递连通分量不会跨 train/val；
- [ ] 同内容不同路径通过 SHA-256 合并，稳定不同 URL query 不被误合并；
- [ ] 所有 asset/entity/region/relation/track/conversation/episode 引用可解析；
- [ ] bbox 三种 format、normalized/pixel 边界、mask/RLE 和 3D frame 约定有测试；
- [ ] 视频 frame index、PTS/time_base、clip 时间边界和 track 顺序有测试；
- [ ] episode frame tree、sensor 引用、step 顺序、首尾标志、tensor inline shape 和 tensor ref 有测试；
- [ ] `preserve_raw=True/False` 行为明确，raw 能回溯且不含秘密；
- [ ] 每个来源字段被标记为 preserved/normalized/raw-only/target-dropped/fatal；
- [ ] exception 和 `severity="error"` diagnostic 在 error/skip policy 下行为正确；
- [ ] fatal 情况不会留下部分默认输出，结构化 report 的计数、code 和 examples 正确；
- [ ] input/output/report 路径冲突被拒绝，已有 output/report 只有 `--overwrite` 才替换；
- [ ] 自定义 target writer 完整消费有序 stream、返回准确数量并能原子提交/失败回滚；
- [ ] `validate --check-assets`、split 后双侧 validate 均通过；
- [ ] manifest 的 count、split、SHA-256 和 records/groups/assets/episodes 统计经 `--check-files` 验证；
- [ ] 若导出 ms-swift，VQA/grounding golden 样本 token 数量和 objects 引用一致；
- [ ] ms-swift 的 answer/annotation/caption policy 已冻结，projection warnings 已审阅；
- [ ] tracking、segmentation、episode 等无隐式投影的监督会明确失败或先经专用 renderer。

最小回归命令：

```bash
python -m compileall -q scripts/universal_dataset
python -m unittest discover -s scripts/universal_dataset/tests -v
python scripts/universal_dataset/cli.py validate output.jsonl --check-assets
python scripts/universal_dataset/cli.py validate-manifest manifest.json --check-files
```
