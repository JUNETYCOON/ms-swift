# S1-UDF：Stage1 通用数据中转格式

S1-UDF（Stage1 Universal Dataset Format）是面向数据审计、格式转换和训练集生成的中转契约。当前版本为 `1.0.0`。它把图像、视频、音频、2D/3D 几何、关系、跟踪、对话、通用标注和具身轨迹放在同一套可引用的数据模型中，并用 `group_id` 约束训练集/验证集切分，避免同一底层媒体或 episode 的不同问答落入不同 split。

S1-UDF 本身不是某个模型的最终训练格式。仓库目前提供 JSONL 工具和 ms-swift 双向适配器；其他来源或目标格式需要编写显式 adapter。转换原则是：保留来源语义、明确坐标与时间单位、无法可靠映射时失败，不凭空猜测标签含义。

## 目录

```text
universal_dataset/
├── SPEC.md                         # S1-UDF 1.0.0 规范性入口
├── SOURCE_SCHEMA_CATALOG.md        # 服务器实测与外部 CV/具身 schema 证据
├── ADAPTER_GUIDE.md                # 来源/目标 adapter 实现与验收指南
├── schemas/
│   ├── record-1.0.0.schema.json    # 单条记录的 JSON Schema 2020-12
│   └── manifest-1.0.0.schema.json  # 数据集 manifest 的 JSON Schema
├── examples/
│   ├── representative.jsonl        # 五类代表性记录
│   └── manifest.json               # 示例 manifest
├── tests/test_universal_dataset.py
├── adapter.py                      # 双向 adapter 契约、版本声明、注册表与插件加载
├── builtin_adapters.py             # 内置 ms-swift source/target adapter
├── cli.py                          # profile/validate/split/manifest/convert CLI
├── conversion.py                   # 有界并行、确定顺序、诊断与转换报告
├── io.py                           # 流式 JSONL 与原子写入
├── manifest.py                     # 从 record 文件构建并校验 manifest
├── ms_swift.py                     # ms-swift 双向适配器
├── profile.py                      # 只读容器与嵌套字段审计
├── schema_registry.py              # 版本化 record/manifest schema 注册表
├── split.py                        # 共享 asset/episode 连通分量与确定性切分
└── validation.py                   # Schema、引用、坐标和时序校验
```

以下命令默认从 `stage1` 仓库根目录执行：

```bash
python scripts/universal_dataset/cli.py --help
python scripts/universal_dataset/cli.py schema-path --kind record
python scripts/universal_dataset/cli.py schema-path --kind manifest
```

profile、split、adapter 列举/检查，以及显式关闭 JSON Schema 的语义校验可只使用 Python 标准库。`validate`、`validate-manifest`、`build-manifest`、`import-ms-swift`、`export-ms-swift`、`convert-source` 和 `convert-target` 的默认校验路径要求安装 `jsonschema`；缺失时会失败，而不是静默降级。只有明确接受不做正式 schema 校验时才对支持该选项的命令传 `--no-json-schema`；`build-manifest` 始终校验生成的 manifest schema。

- `jsonschema`：执行 record 和 manifest 的 JSON Schema 2020-12 校验；默认执行路径需要它。
- `pyarrow`：读取 Parquet、Arrow 和 Feather 元数据。
- `numpy`：读取 `.npy`、`.npz` 的 dtype 和 shape。
- `h5py`：读取 HDF5 dataset 的 dtype 和 shape。
- `ijson`：流式采样超过内置大小阈值的顶层数组或对象映射 JSON。
- `google-crc32c`：加速 TFRecord frame CRC32C 校验；缺失时使用内置实现。
- `mcap`、`rosbags`：可选读取 MCAP/ROS bag 元数据；缺失时对应文件明确标为 skipped。

## Record 契约

每条记录是一个 JSON 对象，JSONL 文件每行一条。五个顶层字段始终必填：

```json
{
  "format": "s1-udf",
  "schema_version": "1.0.0",
  "id": "vqa:0001",
  "group_id": "image:coco-000000000001",
  "provenance": {
    "dataset": "VQAv2",
    "source_record_id": 1,
    "conversion": {"tool": "my-vqav2-adapter", "version": "1.0.0"}
  }
}
```

- `format` 必须等于 `s1-udf`。
- `schema_version` 必须等于 `1.0.0`。
- `id` 是记录级非空唯一标识；同一 JSONL 中不能重复。
- `group_id` 是来源给出的切分分组提示，不是问答 ID；`split` 会把共享媒体或相同 episode ID 连接起来，并在必要时重写为 canonical component ID。
- `provenance` 必须至少包含 `dataset`、`source_record_id` 以及 `conversion.tool/version`；来源没有可靠版本信息时，不得捏造 `dataset_version` 或 `revision`。
- `split` 和 `task_types` 可选；切分命令会根据 canonical component 重写输出记录的 `split`。
- 除上述标识与 provenance 外，语义校验要求 `assets`、`conversations`、`annotations`、`episode`、`entities`、`regions`、`tracks` 或 `streams` 至少有一个非空载荷。
- 顶层不接受未声明字段。通用附加信息放入 `metadata`，保留来源特有结构时使用 `extensions` 或 `provenance.raw_record/raw_ref`。

### 分层模型

| 层 | 字段 | 作用 |
| --- | --- | --- |
| 身份与任务 | `id`、`group_id`、`split`、`task_types` | 记录身份、无泄漏切分和任务标签 |
| 资源 | `assets` | 图像、视频、音频、深度、mask、点云、mesh、tensor、表格、归档等资源 |
| 语义 | `entities`、`regions`、`relations`、`tracks` | 对象、空间区域、图关系和跨帧轨迹 |
| 监督 | `conversations`、`annotations` | SFT 对话和非对话标注 |
| 具身 | `episode` | 坐标系、传感器、观测、动作、状态和时序步骤 |
| 审计 | `provenance` | 来源数据集、原始记录、版本、许可和转换程序 |
| 扩展 | `metadata`、`extensions` | 不改变核心契约的附加信息 |

### Assets

每个 asset 至少包含 `id`、`kind`，并用非空 `uri` 或 external `source_ref` 定位资源；两者也可以同时存在。`kind` 当前允许：

```text
image, video, audio, text, document, depth, mask, point_cloud,
mesh, tensor, table, archive, other
```

`uri` 可以是绝对路径、相对 JSONL 所在目录的路径或远程 URL；`source_ref` 用容器 URI 及可选的 `key/offset/length` 标识容器成员或切片。远程 `file://host/path` 的 authority 是身份的一部分，会保留并与其他 host 区分；只有空 authority 或 `localhost` 才按本地文件 URI 处理。可选的 `sha256` 必须是严格的 64 位十六进制字符串；`source_identities` 只接受已核验并显式写有 `verified: true` 的 `{namespace,value}` 来源媒体 ID，namespace 应包含数据集及必要的 release 范围。切分器同时登记 record 声明的 SHA、verified source identity、规范 URI、完整 source_ref 身份及其 SHA 为别名，从而连接跨路径记录；任一稳定别名连通的不同 SHA 会直接失败。切分器不会读取媒体自动计算哈希。跨数据集副本不能只靠裸来源 ID 连接，仍应计算内容 SHA-256。`sha256`、`size_bytes` 可用于身份和完整性审计；`media` 可记录 `mime_type`、宽高、时长、FPS、帧数、采样率、点数和编码。具身数据还可用 `sensor_id`、`frame_id` 和 `time_span` 将资源连接到传感器、坐标系和时间范围。

大型数组不必内嵌。`tensor` 支持 `dtype`、`shape` 以及二选一的 `data` 或 `ref`；`dataRef` 使用 `uri`，并可带容器格式、key、offset、length、校验和与压缩信息。

### Entities、Regions、Relations 与 Tracks

- `entities` 表示语义对象，标签可带原始 taxonomy、分数、同义词和来源 ID。
- `regions` 通过 `asset_id` 指向媒体，通过可选 `entity_id` 指向对象，并携带 `geometry`、`frame_index` 或 `time_span`。
- `relations` 的 subject/object 是只含一个键的 node reference，可指向 asset、entity、region 或 track。
- `tracks` 可绑定一个或多个 asset，也可绑定 entity；每个 observation 至少带 `frame_index`、`timestamp` 或 `time_point`，并选择 region 模式（`region_id`）或显式状态模式（`state` 字符串）。两种模式都必须能从 observation、region 或 track 解析到已声明 asset。

`geometry.type` 支持：

```text
point2d, bbox2d, polygon2d, polyline2d, rle, mask_ref, keypoints2d,
point3d, bbox3d, cuboid3d, pose3d, point_set, custom
```

每个 geometry 必须显式声明 `coordinate_space`。可用空间为 `pixel`、`normalized`、`camera`、`world`、`robot`、`map`、`voxel` 和 `custom`，并可附 `frame_id`、单位、轴约定及 `[width, height]`。2D bbox 推荐明确写 `format`：`xyxy`、`xywh` 或 `cxcywh`。

当前语义校验器会检查：

- `point2d`、`bbox2d`、`point3d` 的坐标长度，polygon/polyline/keypoints 的点数，以及 NaN/Infinity；
- RLE 是否有 `size=[height,width]` 和 `counts`，`mask_ref` 是否有且引用已知 `asset_ref`；
- normalized 几何是否位于 `[0, 1]`；每个 pixel 2D 几何是否能从 `coordinate_space.image_size` 或来源 image asset 的 `media.width/height` 得到可靠尺寸，并据此验界；
- bbox 的 `xyxy`、`xywh`、`cxcywh` 转换后是否为非负宽高；
- region/entity/asset/relation/track、几何坐标 frame、frame index 与 track observation 的一致性；
- track frame/timestamp 是否分别非递减；
- `time_span.end >= time_span.start`，并在目标 asset 有时长或帧数时检查其边界。

3D/custom geometry 的坐标语义仍应由来源 adapter 和 `coordinate_space`/`metadata` 明确记录，不能只依赖通用校验器推断。

### Conversations 与 Annotations

conversation 包含非空 `messages`。message role 由 schema 限定为 `system`、`user`、`assistant`、`tool` 或 `environment`；content 是结构化 part 数组：

- `text`：文本；
- `asset`：在对话中的媒体插入位置；
- `entity`：对象引用，对应 grounding 的 `<ref-object>`；
- `region`：区域引用，对应 grounding 的 `<bbox>`；
- `tool_call`、`tool_result`：结构化工具调用与结果；
- `custom`：目标格式尚无统一表达的内容。

message 必须有非空 `content` 或非空 `candidates`；因此仅有 candidates 的 preference/ranking 数据是合法 S1-UDF。它不是隐式 SFT target：内置 ms-swift 导出器拒绝任何含 candidates 的 message，并要求 conversation 至少有一个非空、可渲染的 assistant content。通用 `annotations` 用于 caption、QA、分类、检测、分割、跟踪或自定义监督；schema 不封闭 `annotation.type` 的枚举，但语义校验器目前对 `caption` 和 `qa` 增加了必填语义检查。QA 的 question 和 answer 都可以使用结构化 content，并可保存 `answer_mode`、`choices`、`correct_choice_indices`、`canonical_answer_index`、`question_type`、reasoning program 和 evidence；校验器检查 canonical answer 与 choice 索引边界。

### Embodied Episode

`episode` 用于具身、机器人和交互轨迹：

- `task`、`environment`、`agent` 保存任务、场景和载体描述；
- `coordinate_frames` 用 `parent_id` 和 `transform_to_parent` 表达坐标系树，变换可用平移、`quaternion_xyzw`、`quaternion_wxyz` 或矩阵；
- `sensors` 保存类型、坐标系、频率、内参和外参；
- `steps` 保存递增 index、timestamp、observations、actions、state、reward、discount、终止标记和语言指令；
- observation 可三选一引用 asset、保存普通 value 或保存 tensor；action 至少使用 value、tensor 或 language 之一；
- episode 可内嵌至少一个 `steps`，也可用 `steps_ref` 和大于零的 `step_count` 引用外部容器；禁止用空 `steps` 表示外部或未知轨迹。

当前校验器检查 frame/sensor ID 重复、frame parent 是否存在、坐标系 parent 环、asset/sensor/geometry/observation/action 的 frame 引用、step index 严格递增、timestamp 非递减、step 中的 asset/sensor/action target 引用、首尾/terminal 标记，以及内嵌 steps 与 `step_count` 一致性。tensor 的 inline data 必须是矩形嵌套数组，实际嵌套 shape、元素数量和最多一个 `-1` 推断维度都会与声明 shape 对账；dtype 值域、外部 tensor 内容和复杂领域语义仍需 adapter 补充检查。

### Provenance 与扩展

每条 record 的 provenance 必须至少写入：

```json
{
  "provenance": {
    "dataset": "VQAv2",
    "dataset_version": "v2",
    "source_split": "train",
    "source_record_id": 1,
    "conversion": {
      "tool": "my-vqav2-adapter",
      "version": "1.0.0"
    }
  }
}
```

`dataset_version`、`revision`、`source_split`、license 和来源文件等是可选审计字段，只在来源确实提供或能够核验时填写；最小必填集合不要求虚构版本或 revision。

小型且需要精确回溯的来源记录可放入 `raw_record`；大型原始内容建议使用 `raw_ref` 或 `source_files`，避免复制进每条 JSONL。`extensions` 的键只允许字母、数字、下划线、点和连字符；建议使用带命名空间的键，例如 `coco.segmentation`。

## Manifest 契约

manifest 描述整个数据集，而不是单条样本。必填字段是 `format`、`schema_version`、`dataset`、`record_files`、`task_types`、`grouping` 和 `statistics`。每个 `record_files[]` 必须登记 path、format、实际 count 和 SHA-256；`statistics` 必须完整登记 records、groups、unique_assets 和 episodes；`grouping.identity_fields` 必须覆盖 splitter 使用的全部身份字段。`splits` 只对真正未切分的数据集可省略；只要 record 或 file 声明了 split，它就必须与最终文件逐条扫描结果对账。taxonomy、坐标约定、来源 schema、provenance 和 extensions 可选。

示例见 `examples/manifest.json`。record 与 manifest 分别由 `validate` 和 `validate-manifest` 校验；`schema-path --kind record|manifest` 可打印对应 schema 路径。

## group_id 与无泄漏切分

来源 adapter 应在扩增问答、caption 或帧级标注之前给出稳定的 provisional `group_id`。推荐规则：

- 单图任务：同一原始 image ID 的所有 QA、caption、bbox、mask 共用一个 group；
- 视频任务：同一完整 video/clip 产生的 caption、QA、subtitle、point 和 track 共用一个 group；
- 具身任务：同一 episode/demo 的所有 step 和派生样本共用一个 group；
- 多媒体组合：共享任一底层资源的记录最终必须进入同一连通分量；
- ID 应包含数据集命名空间，避免不同来源的裸数字碰撞，例如 `image:coco:000000000001`。

不要使用 QA ID、caption ID、bbox ID 或逐帧 ID 作为 `group_id`。同一 release 内的路径别名可用已核验且带命名空间的 `assets[].source_identities[]` 连接；同一媒体在不同数据集或未知别名下存在副本时，必须优先计算并填写 `assets[].sha256`，否则仅靠路径无法识别内容重复。

切分命令：

```bash
python scripts/universal_dataset/cli.py split input.jsonl --train-output train.jsonl --val-output val.jsonl --val-ratio 0.02 --seed 42
```

默认切分先以 source `group_id` 为节点，把共享 canonical asset identity 或相同非空 `episode.id` 的 group 做 union，再用并查集求传递连通分量。例如 A 与 B 共享图片 1、B 与 C 共享图片 2，或者两条无共享媒体的记录拥有相同 episode ID，它们都必须进入同一 component。单 group component 沿用原 ID；合并 component 使用排序后的成员 ID 计算 `s1cc:<sha256>`。随后以 `SHA-256(seed + NUL + canonical_component_id)` 生成稳定分配，因此：

- 同一个 canonical component 的全部记录一定进入同一 split；
- 输入顺序和进程环境不会改变结果；相同 seed、ratio 和 component ID 得到相同结果；
- `val_ratio` 是按 component 的哈希概率，不保证小数据集得到精确比例；
- 输出采用临时文件加原子替换，默认拒绝覆盖已有文件，需显式传 `--overwrite`。

发生合并时，输出记录的 `group_id` 会改写成 canonical component ID，同时在 `extensions["s1_udf.split"]` 保存 `source_group_id` 和 `component_group_id`；若输入已占用这个保留扩展键，命令会失败。摘要同时报告 source group、component、合并 component、被合并 group 和改写 record 的数量。

asset identity 使用严格的 64hex `sha256`、仅 `verified: true` 的命名空间化 `source_identities`、规范 `uri`，以及由 source_ref URI 与 `key/offset/length` 组成的外部引用身份；source_ref 自带的 SHA 也参与别名图。没有 SHA 时仍可使用经过核验的来源 ID、Windows/UNC/POSIX/相对路径或远程 URI。远程 URI 会小写 scheme/host、移除 fragment，但保留 query；远程 `file://host/path` 保留 host authority，不会与 `file://other-host/path` 或本地 `/path` 合并。`unique_assets` 统计别名合并后的连通身份数。仅在明确接受无法判断媒体泄漏时使用 `--no-asset-audit`：此时不扫描/连接 asset，`unique_assets=0` 且 `leakage_overlap=null`，含义是媒体 leakage 状态 unknown，不是已证明为 0；相同 `episode.id` 仍会被连接。

## CLI

### 1. 只读分析原始 schema

```bash
python scripts/universal_dataset/cli.py profile /path/to/source --sample-rows 100 --max-depth 8 --max-files 10000 --workers 8 --relative-paths --output profile.json
python scripts/universal_dataset/cli.py profile /path/to/source --max-files 30000 --workers 8 --relative-paths --summary-only --output profile-summary.json
```

`profile` 可对单文件或目录工作。它支持 JSON/JSONL 的嵌套路径、类型和数组长度统计；Parquet/Arrow/Feather 的 schema 和行数；CSV/TSV 表头；NumPy/HDF5 的 dtype/shape；TFRecord `Example`/`SequenceExample` 的 protobuf feature schema 与 CRC32C 校验；ROS 2 DB3 的表、topic 和消息统计；以及可选依赖支持下的 MCAP/ROS bag 元数据。ZIP/TAR 会在不落盘解压的前提下审计成员，并对有界数量和大小的内嵌 JSON/JSONL 做 schema profile。

目录模式会统计全部文件的后缀，但排除 `.cache`、AppleDouble/`__MACOSX` 和已知操作系统元数据；随后优先选择已知结构化容器进入逐文件 profile，`--max-files` 限制这部分文件。TFRecord 同时识别标准后缀和 `train.tfrecord-00000-of-00175` 一类分片名。`--workers` 使用有界 pending 的线程并行读取；`--relative-paths` 让 root/file path 可跨机器复现；输出的 `schema_families` 按 `kind + schema` 指纹聚合同构文件，并给出规范化 schema、文件数、已知 record 数和代表路径，`issues` 聚合所有 error/invalid/skipped 原因及有限路径样例。

默认报告保留每个被 profile 文件的详细结果，适合局部排查。全库审计应使用 `--summary-only`：它仍扫描并聚合所有选中文件，但不把重复的 per-file schema 写入结果，只保留自包含的 schema families、issue 聚合、selection buckets 和 `file_reports_omitted` 数量。该选项不是采样缩减；是否全量仍以 `truncated=false`、每个 bucket 的 `omitted=0` 及 issue 审计为准。

为减少路径、文本和标注值泄露，默认报告不包含字段值 examples；确实需要抽样值时显式传 `--include-examples`，每个标量路径最多记录 3 个有限长度样例。profile 是 schema 采样器，不是完整语义校验器：`sample_rows` 之外不参与 shape 统计，可选依赖缺失时对应文件状态为 `skipped`。JSONL 仍会完整解析以拒绝后续行中的非法 JSON 和 NaN/Infinity，但不要用 profile 的 `status=ok` 替代 `validate`。

### 2. 校验 S1-UDF JSONL

```bash
python scripts/universal_dataset/cli.py validate converted.jsonl --report validation.json
python scripts/universal_dataset/cli.py validate converted.jsonl --check-assets --max-errors 200
```

校验包括 JSON Schema、非空 record、ID 唯一性、跨引用、坐标与媒体边界、时间顺序、episode 约束、同一 group 跨 split 泄漏，以及同一 asset 跨 group 冲突。`--check-assets` 额外检查本地文件是否存在；远程 URL 不做网络访问。相对资源路径按输入 JSONL 所在目录解析，空 JSONL 视为错误。

JSON Schema 默认 fail-closed：未安装 `jsonschema` 也会返回 invalid。`--no-json-schema` 是显式的语义校验模式，只关闭正式 schema 检查，其余检查仍运行。报告默认打印到 stdout，也可写入 `--report`；合法返回码为 0，发现错误返回 1。错误达到 `--max-errors` 后停止扫描。

`validate` 会把共享 asset 对应多个 group 报为 `asset_group_conflict`，并报告 episode group/split 冲突；`split` 的默认行为则是把共享 asset 或 episode 的 group 合并成 canonical component。生产结果应在 split 之后再次 validate，并由 manifest 的跨文件检查完成最终对账。

### 3. 校验 Manifest

```bash
python scripts/universal_dataset/cli.py validate-manifest scripts/universal_dataset/examples/manifest.json --check-files --report manifest-validation.json
```

不带 `--check-files` 时检查 manifest schema 与重复的解析后文件路径。带该参数后还检查 record file 是否存在、SHA-256、声明 count、JSONL 每条记录的 split 与文件声明是否一致、episode group/split 泄漏、manifest `splits` 汇总，以及 `statistics.records/groups/unique_assets/episodes` 对账。JSON 和 JSONL 使用严格 JSON 解析；Parquet 行数检查需要 `pyarrow`。它不会替代逐 record 的 `validate`。该命令同样默认 fail-closed，也提供显式的 `--no-json-schema`。

### 4. 构建 Manifest

```bash
python scripts/universal_dataset/cli.py build-manifest train.jsonl val.jsonl \
  --record-split train.jsonl=train --record-split val.jsonl=val \
  --output manifest.json --dataset-name my-dataset --dataset-version v1
```

`build-manifest` 扫描实际 record 文件，自动生成每个文件的格式、split、count 和 SHA-256，并汇总 task、split、group、canonical asset 和 episode 统计。`--record-split PATH=SPLIT` 可重复使用：它会核对非空文件中每条记录的 split，也能为 0 条记录的文件显式登记 split，例如保留小数据集哈希切分产生的空 `val.jsonl` 和 `splits.val=0`。参数中的路径按当前工作目录解析，且必须也是位置参数中的 record file。

默认逐条校验非空 record；可用 `--check-assets` 增加本地资源存在性检查。`--source-schema JSON` 可重复传入，`--provenance JSON` 接受一个 JSON 对象。只有明确由其他步骤完成 record 校验时才使用 `--no-record-validation`。

整个数据集必须至少包含一条 record；单个空 split 文件只有通过 `--record-split` 明确身份时才会计入该 split 的 0 条统计。数据集不能混合已切分和未切分记录；共享 asset 或相同 `episode.id` 不能跨 group/split。输出路径不能与任一 record 文件冲突，生成结果会在返回前通过 manifest schema 校验。默认拒绝覆盖已有 manifest，显式传 `--overwrite` 才会原子替换。

### 5. 按共享媒体和 episode 连通分量切分

```bash
python scripts/universal_dataset/cli.py split converted.jsonl --train-output train.jsonl --val-output val.jsonl --val-ratio 0.02 --seed 42
```

命令输出 record、source group、canonical component、合并/改写、asset 和 episode 数量摘要，并在输出记录中写入 `split=train|val`。输入不能为空，train 与 val 路径必须不同，也不能覆盖输入文件。train/val 两个文件作为一对提交；覆盖时若提交中途失败会恢复原文件。

### 6. 导入 ms-swift

```bash
python scripts/universal_dataset/cli.py import-ms-swift swift.jsonl --output udf.jsonl --dataset-name my-dataset --workers 8 --chunksize 64
```

当前导入器接受：

- 非空 `messages`；
- `images`、`videos`、`audios` 数组；
- 文本中的 `<image>`、`<video>`、`<audio>`、`<ref-object>`、`<bbox>`；
- `objects.ref`、`objects.bbox`、`objects.bbox_type`（`real` 或 `norm1`）和多图 `objects.image_id`。

每种 token 的数量必须与对应数组中的值完全一致，多余 token 或未消费值都会报错。message role 只接受 `system/user/assistant/tool`，content 必须是字符串；媒体路径和 grounding 数值也会做类型、有限数与边界检查。长度为 2 的 bbox 项导入为 `point2d`，长度为 4 的项导入为 `bbox2d/xyxy`。`ref` 与 `bbox` 按各自 token 顺序独立消费，适配器不会推断二者之间的实体归属。只要存在 `objects.bbox` 且 `bbox_type="real"`（包括省略该字段时的默认值），导入器就用 Pillow 探测每张被引用本地图片的真实宽高并写入 asset media；远程 URI、文件缺失、格式损坏或 Pillow 缺失都会使导入失败，不会带着未知尺寸继续转换。`bbox_type="norm1"` 不需要读取图片尺寸。

输入若已有非空 `group_id` 就原样保留；否则对媒体路径集合计算稳定哈希生成 `media:<sha256>`，纯文本则使用 `text:<record_id>`。相对媒体路径按输入 ms-swift JSONL 的目录解析并写成绝对路径；Windows/UNC/POSIX 路径跨平台保留其语义，远程 URL 保留 query、移除 fragment。导入默认把完整来源记录保存在 `provenance.raw_record`；数据量较大且不需要内嵌回溯时可传 `--drop-raw`。

导入默认执行 JSON Schema；仅在明确接受语义校验时传 `--no-json-schema`。`--workers > 1` 使用多进程，`--chunksize` 控制 worker 批量调度，同时保持输入输出顺序。输出已存在时需传 `--overwrite`。

### 7. 导出 ms-swift

```bash
python scripts/universal_dataset/cli.py export-ms-swift udf.jsonl --output swift.jsonl --workers 8 --chunksize 64 --report export-report.json
```

conversation 中的结构化 part 映射为 ms-swift 文本 token 和 `images/videos/audios/objects`。相对 S1-UDF asset 路径按输入 JSONL 的目录解析为绝对路径；只有 `source_ref`、没有直接 `uri` 的 asset 必须先 materialize 或由目标专用 resolver 处理，内置导出器会明确失败。`entity` 输出 `<ref-object>`，`region` 输出 `<bbox>`；region 目前只支持 pixel/normalized 的 `point2d` 和 `bbox2d`。`xywh`、`cxcywh` 会转换为 xyxy；同一条样本混合 real/norm1 bbox 时统一转换为 norm1，此时 pixel box 对应 asset 必须有宽高。多图 grounding 必要时生成 `objects.image_id`。

默认额外保留 record `id`，并且始终保留 canonical `group_id`。`--drop-ids` 只移除 record `id`，绝不移除 `group_id`，以便所有下游 train/val 操作仍能按媒体或 episode 防泄漏。一个 S1-UDF record 可能因多个 conversation 或 annotation 输出多条 ms-swift record；保留 ID 时，多输出自动改成包含 record ID、类型、索引和局部 ID 的唯一值，所有输出继续共享原 `group_id`。

导出也默认执行 JSON Schema，并支持 `--no-json-schema`、`--workers` 和 `--chunksize`。多进程输出保持输入顺序。

### 8. 列举和检查 Adapter

```bash
python scripts/universal_dataset/cli.py list-adapters
python scripts/universal_dataset/cli.py list-adapters \
  --direction source --plugin /path/to/plugin.py
python scripts/universal_dataset/cli.py inspect-adapter my-source \
  --direction source --plugin /path/to/plugin.py
```

每个 adapter 都通过 `AdapterSpec` 声明名称、adapter 版本、方向、支持的 S1-UDF 版本、能力、允许损失和说明。内置 registry 当前提供 `ms-swift` source 和 target adapter；仓库没有为服务器上的各个原始数据集预先实现 dataset-specific adapter。`--plugin` 可重复传入 Python 文件或可导入模块，并在列举、检查或转换前显式加载其注册项。

### 9. 通用 Source/Target 转换

```bash
python scripts/universal_dataset/cli.py convert-source /path/to/source \
  --adapter my-source --plugin /path/to/plugin.py \
  --output unsplit.jsonl --dataset-name my-dataset \
  --workers 8 --batch-size 64 --max-pending-batches 16 \
  --error-policy error --report source-report.json

python scripts/universal_dataset/cli.py convert-target train.jsonl \
  --adapter my-target --plugin /path/to/plugin.py \
  --output target-output --workers 8 --batch-size 64 \
  --max-pending-batches 16 --error-policy error \
  --report target-report.json
```

`--option KEY=JSON` 是可重复的强类型 adapter 参数：value 必须是合法 JSON，key 必须非空且不能重复。runner 以 batch 为单位限制 pending 工作量，多进程结果仍按输入顺序输出，并支持一条输入生成多条输出。`--error-policy error` 在首个异常或 `severity="error"` diagnostic 处失败；`skip` 跳过完整 input item 并继续，报告状态为 `complete_with_skips`。`MemoryError`、键盘中断和其他进程级中断不会被降级为普通 row error。

结构化 report 包含输入/输出/跳过数量、adapter spec、severity/code 计数、有限 diagnostic examples、fatal error 和实际并行参数。input、output 和 report 必须是不同路径；output/report 已存在时默认拒绝，二者都由同一个 `--overwrite` 控制。默认 source JSONL writer 和默认 target JSONL writer 使用临时文件原子提交；自定义 target writer 的额外责任见下文和 `ADAPTER_GUIDE.md`。

## ms-swift 投影策略、报告与不支持字段

默认 policy 都采用保守行为，需要有损投影时必须显式选择：

- `--answer-policy error|first|highest-count`：QA 优先使用合法的 `canonical_answer_index`；单答案直接使用。多答案且无 canonical 时，默认 `error`，也可显式选第一项或 count 最大项（并列时取最早项）。
- `--annotation-policy error|prefer-conversations|prefer-annotations`：record 同时有 conversations 和 annotations 时默认报错；显式选择一侧后，报告记录 `ignored_annotations` 或 `ignored_conversations`。
- `--caption-prompt TEXT`：caption 优先使用 `annotation.metadata.sft_prompt`，否则使用该 CLI 参数；两者都没有时失败，不再注入固定英文 prompt。
- `--unsupported-policy error|skip`：默认任一不支持转换会中止并丢弃临时输出；`skip` 会跳过整条输入 record 并继续，最终状态为 `complete_with_skips`。它不绕过输入的 schema/语义校验错误。

QA annotation 含 `choices` 时，导出器按原数组顺序把候选项确定性追加到 user 内容：`Choices:` 后使用从 1 开始的编号。choice 可以是非空 `text` 或结构化 `content`，同时提供二者会因含义不唯一而失败。`correct_choice_indices` 仍保留在 S1-UDF 中用于校验和回溯；assistant target 由 `canonical_answer_index`/`--answer-policy` 选择，不会由导出器猜测候选索引的文本表达。空字符串或仅包含空 text part 的 question、caption、answer/choice 都会被拒绝。

导出摘要始终打印到 stdout，`--report` 可另存 JSON。报告包含输入/输出/跳过数量、policy、按类型聚合的 `projection_warnings`、有限错误样例和 fatal error；`--max-error-examples` 控制保存的错误样例上限。当前投影告警覆盖被忽略的 conversation/annotation、多答案选择、未表示的 relations/tracks/episode，以及 asset 插入位置的 time span/frame index 丢失。

仍需注意以下目标能力边界：

- 没有 conversation 时，仅 `caption` 和 `qa` annotation 有隐式 SFT 映射；分割、检测、tracking、episode 等 annotation 需目标专用 renderer，或由 `unsupported-policy` 决定失败/跳过。
- message `candidates` 是合法的 S1-UDF preference/ranking 表达，但无法隐式选择成单响应 SFT target；内置导出器会报错。conversation 还必须至少包含一个非空、可渲染的 assistant content。
- S1-UDF 的 `environment` role 不能导出到当前 ms-swift role 集；`tool_call` 需要目标专用工具训练 adapter；`tool_result` 的结构化值会编码成紧凑 JSON；`custom` content 不支持。
- 对话中只能插入 image、video、audio asset。depth、mask、point cloud、tensor 等 asset kind 不能直接作为 ms-swift 媒体 token。
- grounding 导出不支持 polygon、RLE、mask、3D geometry 等；应设计任务专用文本/掩码编码或选择能承载该监督的目标格式。

报告不是所有来源字段的自动可逆性证明。对目标任务有意义但未参与所选 conversation/annotation 的其他结构，仍应在数据集 adapter 中定义允许损失清单并增加领域检查。

## Adapter 平台与插件

`adapter.py` 对来源和目标方向使用对称的版本化契约：

- `SourceAdapter.iter_source(source, context)` 产生稳定的 `(source_id, value)`，`convert(source_id, value, context)` 返回一对多的 `AdapterResult`；
- `TargetAdapter.convert(record, context)` 返回一对多的 `TargetAdapterResult`，默认 `write_output()` 将有序值原子写成 JSONL；
- `AdapterSpec` 必须声明 `name/version/direction/udf_versions/capabilities/loss_contract/description`，注册时会拒绝方向错误或不支持当前 S1-UDF `1.0.0` 的 adapter；
- `AdapterRegistry` 按 direction 和 name 分开注册、列举及实例化；使用 `@register_source_adapter` 和 `@register_target_adapter`，旧的 `@register_adapter` 只是 source 方向兼容别名；
- `AdapterContext` 传递 dataset、source root、raw 策略、typed options 和 UDF 版本；`TargetContext` 传递输出路径、输入基准目录、typed options 和 UDF 版本；
- `AdapterDiagnostic` 用 `info|warning|error`、稳定 code 和 source/record 定位描述可审计问题，旧式 `warnings` 会归一化为 warning diagnostic；
- `stable_record_id` 和 `provisional_media_group_id` 只提供确定性 helper。最终切分仍会全局归并共享 asset 和相同 episode ID。

`capabilities` 与 `loss_contract` 是供发现、审计和评审使用的声明，不会自动证明转换无损，也不会自动生成字段映射。当前 registry 内置 `ms-swift` 双向 adapter；每个其他来源数据集仍需要真实 schema 对应的插件，不能因为能被 `profile` 读取就声称已经支持。当前 CLI 也不提供 schema migration 或在同名 adapter 的多个版本间自动选择。

插件通过可重复的 `--plugin /path/plugin.py` 或 `--plugin package.module` 显式导入。插件是可执行 Python 代码，不是普通配置：主进程会执行它，多进程 worker 也会再次加载它。只加载已审阅、固定 revision 或 hash 的可信插件，不要直接加载从数据集、网页或未知仓库下载的 Python 文件。完整的 source/target 骨架见 `ADAPTER_GUIDE.md`。

通用 runner 已负责有界 pending batch、多进程初始化、确定性顺序、一对多展开、逐 record 校验、重复 ID 检测、错误策略和结构化报告。adapter 的 `convert()` 必须无共享副作用；不要由 worker 写最终文件或依赖完成顺序。来源默认输出和目标默认 JSONL writer 都原子提交。

目标 adapter 可覆盖 `write_output(values, context, overwrite=False)` 以支持聚合 JSON、Parquet、多文件或二进制格式。自定义 writer 必须完整、按顺序消费惰性 iterable，返回实际消费数量，并自行实现临时输出、原子提交和失败回滚；runner 会检测 writer 提前返回、非法计数或计数不一致，但无法替它撤销已经自行提交的外部副作用。

实现 dataset-specific adapter 时仍应先审计 schema，在展开 QA/caption/帧之前确定媒体或 episode 身份，明确坐标与时间单位，填写 provenance 和允许损失表，并让未知 schema/version 或含义不唯一的字段失败。完整发布链路必须经过 canonical split、双侧 validate、manifest 构建/对账和目标 golden test。

### 常见任务映射

| 来源任务 | S1-UDF 主要字段 | ms-swift 前置处理 |
| --- | --- | --- |
| VQA | image asset + `annotations[type=qa/vqa/document_qa]` | 写非空 canonical answer，或显式选择 answer policy；choices 按原顺序编号渲染 |
| Caption | image/video asset + `annotations[type=caption/captioning/dense_caption]` | 提供非空 `text/content` 及 `metadata.sft_prompt` 或 `--caption-prompt` |
| Grounding/Detection | entities + regions + conversation/annotation | 将实体和区域放进 conversation 以生成 token |
| Tracking/Video point | video asset + per-frame regions + tracks | 需要任务专用 conversation renderer |
| Segmentation/Panoptic | image/mask assets + mask/RLE regions | 当前无隐式 ms-swift 映射 |
| Embodied/Robot | episode + assets/tensors + frames/sensors/steps | 当前无隐式 ms-swift 映射 |

## 示例与验证

`examples/representative.jsonl` 包含五条已用于测试的记录：

1. VQAv2 风格单图 QA；
2. Visual Genome 风格多对象 grounding 对话；
3. Molmo2 风格视频 point tracking；
4. COCO 风格 panoptic mask 引用；
5. 带坐标系、相机、tensor、动作和 reward 的具身 episode。

运行示例校验和测试：

```bash
python scripts/universal_dataset/cli.py validate scripts/universal_dataset/examples/representative.jsonl
python scripts/universal_dataset/cli.py validate-manifest scripts/universal_dataset/examples/manifest.json --check-files
python -m unittest discover -s scripts/universal_dataset/tests -v
python -m compileall -q scripts/universal_dataset
```

上述两个 CLI 校验命令默认需要 `jsonschema`；无该依赖且明确只做语义 smoke test 时，可分别追加 `--no-json-schema`。测试还覆盖双向 adapter registry/version handshake、插件加载、确定顺序与有界并行、diagnostic error/skip、writer 完整消费、路径冲突、共享 asset/episode component、tensor inline shape，以及 manifest 的 record/group/asset/episode 统计对账。

## 推荐生产流水线

```text
原始数据 -> profile -> convert-source（插件 + 转换报告）
        -> split（共享媒体/episode component）
        -> validate(train/val, --check-assets)
        -> build-manifest -> validate-manifest --check-files
        -> convert-target（或兼容的 export-ms-swift）+ 转换报告
```

在发布转换结果前，至少保存无 examples 的 profile 报告、source/target conversion report、validation 报告、split 参数与 component 摘要、manifest、插件 revision/hash 及未支持/允许损失清单。这样才能区分“来源本来没有该字段”“adapter 没有映射该字段”和“目标格式无法表达该字段”。
