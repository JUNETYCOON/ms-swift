# S1-UDF 1.0.0 规范

状态：Stable  
格式名称：`s1-udf`  
Record schema：`schemas/record-1.0.0.schema.json`  
Manifest schema：`schemas/manifest-1.0.0.schema.json`

本文是 S1-UDF 1.0.0 的规范性入口。JSON Schema 定义可机器检查的字段与类型；
`validation.py`、`split.py` 和 `manifest.py` 实现本文要求的跨引用、空间、时间、分组和数据集级约束。
发生冲突时，版本匹配的 JSON Schema 与校验器是可执行依据，冲突应作为规范缺陷修复，不能由 adapter
自行选择较宽松解释。

文中的“必须”“禁止”“应”“可以”分别表示强制要求、强制禁止、推荐要求和可选能力。

## 1. 目标与边界

S1-UDF 是保真中转格式，不是单一模型的训练格式。它必须能够显式承载：

- image、video、audio、document、depth、mask、point cloud、mesh、tensor 和 table 等资源；
- caption、VQA、分类、检测、grounding、分割、关键点、关系、跟踪和自定义监督；
- 多轮多模态对话、候选回答、工具调用与结构化内容；
- 2D/3D 坐标、坐标系、时间轴、传感器、stream 和 embodied episode；
- 来源、转换版本、原始引用、允许损失与扩展字段；
- 可校验的文件清单、checksum、split 和统计。

S1-UDF 不统一不同机器人的动作含义，不把 polygon/mask/3D box 猜写成 2D bbox，也不保证任意目标格式
能无损表达全部字段。无法可靠映射的来源结构必须保留在 `extensions`/`provenance`，或使转换明确失败。

## 2. Record 结构

每条 record 必须是一个 JSON object，并包含：

| 字段 | 要求 |
| --- | --- |
| `format` | 固定为 `s1-udf` |
| `schema_version` | 固定为 `1.0.0` |
| `id` | 数据集内稳定且唯一的非空 record ID |
| `group_id` | 不可拆分来源单元的稳定分组提示 |
| `provenance` | 至少包含 dataset、source record ID 与 conversion tool/version 的来源证据 |

可选顶层字段为 `split`、`task_types`、`assets`、`entities`、`regions`、`relations`、`tracks`、
`conversations`、`annotations`、`clocks`、`streams`、`episode`、`metadata` 和
`extensions`。顶层不允许其他字段。来源专用结构必须进入命名空间化的 `extensions` 或 provenance。

同一 collection 内的 `id` 必须唯一。所有 `asset_id`、`entity_id`、`region_id`、`track_id`、
`frame_id`、`sensor_id` 和其他显式引用必须指向同一 record 中已声明的对象，除非字段明确是 external
`data_ref`/`raw_ref`。

## 3. 身份与无泄漏切分

`id` 标识 record；`group_id` 标识不可拆来源单元；asset identity 标识底层媒体。三者不能混用。
QA ID、caption ID、bbox ID、帧号或 Parquet 行号禁止作为媒体级 `group_id`。

来源 adapter 必须在展开同媒体的 QA/caption/标注前给出 provisional `group_id`：

- image 数据通常按原图；
- video 数据默认按完整 canonical video，而不是 QA 或任意 clip；
- embodied 数据按完整 episode/trajectory/demo；
- 多图记录连接其所有图片；
- 同 episode 的全部 step、视角和派生帧必须同组。

发布前，splitter 必须以 source group、asset identity 和 `episode.id` 建立传递连通分量，再对完整分量
做确定性 train/val 分配。同一分量禁止跨 split。

asset identity 必须登记所有已声明且可验证的别名：

1. asset 或 `source_ref` 声明的 64hex `sha256`；
2. 经 adapter 核验、带 release 命名空间且显式写有 `verified: true` 的 `source_identities`；
3. 规范化的 asset `uri`；
4. 由 `source_ref.uri` 与可选 `key/offset/length` 共同构成的外部引用身份。

S1-UDF 通用 splitter 只消费 record 中声明的 SHA、verified source identity、URI、完整 source_ref 身份和
`episode.id`，不替 adapter 读取任意媒体。
因此路径别名或跨数据集媒体副本要获得内容级保证，adapter/预审计必须计算 SHA-256。相同规范 URI 声明
不同 SHA 必须失败；有 SHA 和只有 URI 的记录必须通过别名图连接。

URI 规范化必须保留远程 `file://authority/path` 的 authority；不同 host 的相同 path 不是同一资产。仅
authority 为空或等于 `localhost` 的 file URI 按本地路径处理。远程 URI 保留 query、移除 fragment，不能
删除用于区分资源的稳定 query。

`val_ratio` 是对 canonical component 的稳定哈希概率，不保证小数据集得到精确条数。空 val 是合法结果，
但 manifest 必须显式记录对应文件的 `split=val` 和 `count=0`。

## 4. Assets、空间与时间

每个 asset 必须有 record 内唯一 `id`、`kind`，并至少有非空 `uri` 或 external `source_ref`。已知时应保存 `sha256`、
`size_bytes`、MIME、width/height、duration、FPS、frame count、sample rate、point count、sensor/frame ID
和 time span。相对 URI 以 record 文件所在目录为基准。

每个 region 必须引用 asset。geometry 必须显式声明 `type` 和 `coordinate_space`；支持 point/bbox、
polygon/polyline、RLE/mask、2D/3D keypoints、3D box/cuboid/pose、point set 与 custom geometry。
坐标格式、单位、轴约定、frame ID、图像尺寸和旋转约定在来源已知时必须填写。

- 每个 pixel 2D geometry 必须由 `coordinate_space.image_size` 或来源 image asset 的
  `media.width/height` 提供可靠尺寸并验界；无法确认尺寸时必须失败；
- normalized 坐标必须在 `[0,1]`；
- `xywh`/`cxcywh` 不能在未记录转换的情况下当作 `xyxy`；
- RLE 必须保留 `size` 和 `counts`；mask reference 必须引用 mask asset；
- 3D box 的维度顺序、坐标 frame 和旋转 convention 必须显式保存；
- frame index、timestamp、PTS、timebase 和 clock 不得混为同一单位。

track observation 必须给出 `frame_index`、`timestamp` 或 `time_point`，并采用两种模式之一：引用同 asset 上
的 `region_id`，或用 `state` 字符串明确保存状态（例如 absent/occluded）。无论哪种模式，都必须能从 observation、
region 或 track 解析到已声明 asset。多 asset track、动态变换和异步传感器应通过显式
stream/clock/frame 关系表达，禁止仅按文件名猜测同步。

## 5. 对话与监督

conversation message 的 `content` 是有序结构化 part。`asset` part 决定媒体插入位置；`text`、`entity`、
`region`、`document_anchor`、`tool_call`、`tool_result` 和 `custom` part 保持原顺序。消息必须含非空
content 或 candidates；candidate-only message 是合法的 S1-UDF preference/ranking 数据，但不能由通用 SFT
投影器隐式选择候选。训练 target 不能只是空字符串/空 text part。

annotation 的 `type` 是开放字符串；通用校验器为常见任务增加最低监督要求。QA/VQA/document QA 必须有
非空 question 和至少一个非空 answer，可保存完整 answers、choices、正确 choice 索引、canonical answer、
program 和 evidence。caption 必须有非空 text 或结构化 content。检测、分割、tracking 和 relation 必须引用
对应 region/track/relation，不得只写任务名称。

结构化监督应优先进入 annotation；只有明确的 SFT 话术才进入 conversation。来源同时提供原始监督和 SFT
对话时可以同时保留，但投影到单目标格式前必须显式选择，不能静默重复或丢弃。

## 6. Embodied episode

episode 是不可拆分切分单元。它可以声明 coordinate frames、sensors、streams、dynamic transforms、steps、
外部 `steps_ref` 和 feature specs。每个 step 可包含 observation、action、reward、discount、终止标记和语言。

内嵌 `steps` 必须至少有一项；外部步骤必须用 `steps_ref` 加大于零的 `step_count`，禁止用空数组代表未知或
外部轨迹。step index 必须严格递增，timestamp 必须非递减；terminal 只能出现在最后一步且必须同时是 last，
首尾/terminal 标记必须自洽。observation/action tensor 的
dtype、shape、单位、控制模式和 frame 必须保留。不同数据集的同名 `action` 禁止仅凭字段名合并成同一语义。
没有真实 action/state/step 证据的数据集不得因名称含 Robo/Robot 就伪造成 episode。

## 7. Provenance 与扩展

每条转换记录的 provenance 必须包含非空 `dataset`、稳定 `source_record_id` 以及
`conversion.tool/version`。`dataset_version`、`revision`、source split、license 和 source files 仅在来源已知或
能够核验时填写；不得为满足格式而捏造 version/revision。小型原始记录可以放 `provenance.raw_record`；大型来源
应使用 checksum 可核验的 `raw_ref`/`source_files`，避免复制。

`extensions` 键必须满足 schema 的命名规则，并应使用来源命名空间，例如 `coco.segmentation`。无法确认的
坐标尺度、时间单位、ID join 或字段语义必须标为 unresolved 并保留原值；禁止猜测后写入标准字段。

## 8. Manifest

manifest 必须包含格式/版本、dataset metadata、实际 `record_files`、`task_types`、`grouping` 和完整
`statistics.records/groups/unique_assets/episodes`。每个文件登记 path、format、count、SHA-256 和可选 split；
grouping 必须声明 `group_id` 语义以及 splitter 的全部 identity fields。manifest 的 count/checksum/statistics
必须通过扫描最终文件生成，禁止手工估算。`splits` 对真正未切分的数据集可省略；一旦 record 或 record file
声明 split，就必须与最终文件扫描结果对账。

构建器必须检查：record ID 唯一、group/asset/episode 不跨 split、同 URI 的 SHA 冲突、文件 split 声明、
records/groups/canonical assets/episodes 统计及输出路径冲突。所有 record file 合计必须非空；单个空 split
文件必须用 `--record-split PATH=SPLIT` 显式登记。

## 9. Adapter 与转换

source adapter 必须声明名称、版本、支持的 S1-UDF 版本、能力和损失契约，并产生稳定 source locator、ID、
group 和 provenance。target adapter 必须声明其不可表示字段和投影策略。未知来源 schema/version、类型漂移、
缺媒体或含义不唯一必须产生结构化 diagnostic，并按 `error|skip` policy 处理。

并行 runner 必须保持输入顺序、限制 pending 工作、支持一对多展开、验证每条输出、检测重复 ID，并以临时
文件原子提交。worker 禁止直接写最终输出或依赖完成顺序。插件是可执行代码，只能加载已审阅并固定 revision
或 hash 的文件。

一个来源被 catalog 描述，不等于已实现 adapter。宣称“支持转换”至少需要 source adapter、真实样本 golden
test、无泄漏 split、双侧 validation、manifest 对账、目标投影和 loss report。

## 10. ms-swift 投影

ms-swift 是目标投影，不反向限制 S1-UDF：

- 有序 image/video/audio asset part 分别渲染为 `<image>/<video>/<audio>`，token 与数组严格对齐；
- entity/2D point 或 bbox 可渲染为 `<ref-object>/<bbox>` 与 `objects`，ref 和 bbox 独立计数；
- pixel 和 norm1 bbox 不得混淆，多图 bbox 必须有正确 `image_id`；
- QA choices 按来源顺序渲染为 `Choices:` 和从 1 开始的编号；assistant 由明确答案策略选择；
- caption/QA 的空 target、歧义多答案和不支持 annotation 默认失败；
- conversation 必须有至少一个非空、可渲染的 assistant content；含 candidates 的 preference message
  不能被隐式选成 SFT response；
- 只有 external `source_ref`、没有 direct `uri` 的媒体必须先 materialize 或使用目标专用 resolver；
- 导入 `bbox_type=real`（包括缺省值）时必须用 Pillow 探测所有被引用本地图片的实际宽高；远程、缺失、
  损坏媒体或 Pillow 缺失必须失败，不能带未知 pixel 尺寸继续；
- 导出始终保留 canonical `group_id`；`--drop-ids` 只删除 record `id`；
- mask/RLE、polygon、tracking 时序、3D、point cloud、关系图、OCR reading order 和 episode 控制语义需要
  目标专用 renderer，禁止伪装成通用 bbox 或普通对话。

每次投影必须输出 conversion report，记录 policy、warnings、skips、errors 和允许损失。所有从同媒体/episode
派生的目标记录必须继承同一 canonical `group_id`。

## 11. 发布与版本

发布数据集的最低流水线为：

```text
profile -> source adapter -> canonical split -> validate train/val
        -> build manifest -> validate manifest --check-files
        -> target adapter/ms-swift export -> conversion report + golden test
```

1.0.x 只能做不改变有效实例集合和字段含义的兼容修复。新增可选能力但不破坏旧数据时升级 minor；删除字段、
改变坐标/身份/切分语义或使旧实例失效时必须升级 major，并提供显式 migration。schema version 不能由文件名、
adapter 默认值或“尽力兼容”猜测。

服务器实测来源和外部 CV/具身格式的映射证据见 `SOURCE_SCHEMA_CATALOG.md`；adapter 实现要求与 skeleton 见
`ADAPTER_GUIDE.md`。这两个文档是设计与实现指南，不扩大本规范声明的已实现 adapter 清单。
