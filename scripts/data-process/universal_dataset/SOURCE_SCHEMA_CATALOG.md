# S1-UDF 来源 Schema 实测目录

本文记录 S1-UDF `1.0.0` 的来源格式证据、分组规则和映射边界。它是 adapter 实现前的审计基线，不是“看到相似字段后自动猜测”的许可。

## 证据范围与口径

- **服务器实测根目录**：`/mnt/luojunkun/stage1/dataset`
- **审计日期**：2026-08-10（Asia/Shanghai）
- **边界**：通过 SSH 只读枚举文件、读取容器元数据、字段 schema 和有限样本；没有改写源文件，没有解压或执行归档内容，也没有尝试下载缺失媒体。
- **覆盖目录**：除 `.cache` 外共 20 个顶层数据目录，逐项见下文。
- **容器概况**：实测 926 个 Parquet 文件，归为 17 个 schema 族；COCO 另有 45 个 Arrow 文件，归为 3 个 schema 族。这个数字是物理文件数，不是独立样本数。
- **行数口径**：Parquet/Arrow 取容器元数据之和；顶层 JSON 数组取数组长度；newline JSON 按一行一对象计数。缓存副本、类别子集、testdev 子集和相同媒体上的多条 QA 都可能重复，不能把物理行数直接当作独立媒体数。
- **媒体盘点边界**：本轮结构审计没有发现可作为监督输入的 `audio` 字段，也没有建立全库 loose-video 完整清单。URL、归档成员和记录内 `video` 字段只表示引用存在，不表示媒体可访问。

文中标记含义：

- **服务器实测**：本轮在上述根目录实际读到。
- **官方参考**：来自数据集官方 schema/API 文档，本轮服务器不一定存在对应文件。
- **未确认**：当前只读审计没有足够证据；adapter 必须失败或保留原始值，不能补猜。

## 强制的分组与防泄漏规则

`group_id` 不是 QA ID、caption ID、目标框 ID、逐帧 ID 或 Parquet 行号。adapter 必须先把同一 episode/demo 等不可拆单元赋成相同 `group_id`；随后 splitter 以 source `group_id`、`assets[]` 的规范媒体身份和 `episode.id` 建立连通关系。任意共享媒体或 episode 产生的**传递连通分量**必须得到同一个 canonical `group_id`，之后才能做 train/val 哈希切分。`--no-asset-audit` 只关闭共享 asset 合并，不会关闭 episode 的不可分割约束。

媒体身份优先级如下：

1. 已验证的媒体字节 `sha256`；
2. `assets[].source_identities[]` 中带数据集/release 命名空间的稳定原始媒体 ID，并且其 ID 到媒体映射已核验；
3. 规范化绝对路径或不含签名参数的规范化 URL；
4. 无 ID 的嵌入 bytes 现场计算内容哈希；
5. 上述信息都没有时，停止转换并报告，不能退化为 QA/行级分组。

实现上优先级不表示丢弃低优先级别名：asset 的 SHA、source identity 和 URI 都进入身份连通图，以连接不同路径的记录。任一同名 source identity/规范 URI 对应不同 SHA 视为内容漂移或元数据冲突，必须失败；跨数据集副本仍以字节 SHA 为准，不能让两个来源的裸数字 ID 偶然碰撞。

需要额外执行以下合并：

- 一条多图记录连接其全部图片；记录 `{A,B}` 与 `{A,C}` 因共享 `A` 必须合并为同一分量。
- 同一完整视频的不同 clip、问题、字幕、point/track 标注应合并；仅在有官方保证“clip 互不重叠且不存在同源泄漏”时才允许 clip 级分组。
- 同一 episode/demo 的所有 step、相机视角、派生帧和语言标注应由 adapter 预先赋成同一 `group_id`；共享 asset 审计只提供额外的传递合并。
- 合并多个来源目录训练时，COCO/VG 等媒体副本必须按字节哈希跨目录连接；不能因为目录名不同就视为不同媒体。
- 当前通用 canonicalizer 保留 URL 的完整 query 并丢弃 fragment，因为 query 可能具有内容选择语义。adapter 只能在明确识别认证/签名参数后做预归一化，不得笼统删除 query；无论如何都要用媒体哈希检查 URL 重定向和内容漂移。

推荐 source `group_id` 为 `媒体类型:数据集命名空间:稳定原始ID`；多个 source group 因共享 asset 被合并时，实现用排序后的成员身份计算 SHA-256，写成 `s1cc:<sha256>`。当前切分摘要报告 record、source group、canonical component、合并 component、被合并 group、改写 record、独立 asset、train/val 数量和切分参数；最大分量、原 split 冲突及媒体副本统计若为发布要求，应由 adapter 审计另行生成，不能假设 split CLI 已提供。

## S1-UDF 映射速记

| 来源语义 | S1-UDF 目标 |
| --- | --- |
| 图片、视频、深度、mask、点云、表格、归档 | `assets[]`；媒体尺寸、FPS、时长写 `assets[].media` |
| 类别、对象、指代表达 | `entities[]` |
| 点、框、多边形、RLE、mask、3D box、pose | `regions[]`，且显式写 `geometry.type/format/coordinate_space` |
| 对象关系或 scene graph 边 | `relations[]` |
| 跨帧对象 | `tracks[]`，观测引用 `regions[]` |
| QA、caption、分类、分割等监督 | `annotations[]`；SFT 话术才写 `conversations[]` |
| 多轮对话和媒体插入顺序 | `conversations[].messages[].content[]` 的结构化 part |
| episode、观测、动作、状态、标定 | `episode`、`assets[].sensor_id/frame_id` |
| 不可无损统一的来源字段 | `extensions` 或 `provenance.raw_ref/raw_record`，不得静默丢弃 |

## 服务器 20 个目录实测

### 1. `COCO`

- **容器与行数（服务器实测）**：`COCO-MODELSCOPE` 有 train 39 个 Arrow、test 2 个 Arrow，分别为 117,218 和 5,000 行。`COCO-caption2014` 的逻辑数据为 train 16,000、validation 8,000 行；相同 Arrow 又存在于 cache 树，物理合计 48,000 caption 行，因此只能计 24,000 条逻辑记录。
- **精确核心字段**：ModelScope train 为 `images: list<struct<bytes,path>>`、`labels: list<int64>`；test 为 `images`。caption 为 `uniq_id`、`image_id`、`caption`、`image`。
- **媒体分组键**：优先嵌入图片 bytes 的 SHA-256；caption 的 `image_id` 可作数据集内 join key，但跨 `COCO-MODELSCOPE`、caption、VQAv2、LLaVA 时仍需内容哈希合并。
- **S1-UDF 映射**：图片到 `assets[kind=image]`；`labels` 到分类 `annotations`/`entities`；caption 到 `annotations[type=caption]`；`uniq_id` 仅作 `provenance.source_record_id`，不能作 `group_id`。
- **坐标、重复与风险**：这个本地快照未实测 COCO detection/keypoint/panoptic 原始 JSON。cache Arrow 是已确认副本；同图多 caption 也不是独立切分单元。检测等几何只能按后文“COCO 官方参考”另写 adapter，不能从此 Arrow schema 推断。

### 2. `Chartqa`

- **容器与行数（服务器实测）**：5 个 Parquet（train 三 shard、val/test 各一 shard），总计 32,719 行。
- **精确核心字段**：`image`、`query`、`label`（列表）、`human_or_machine`。
- **媒体分组键**：记录中没有稳定 media ID；必须对 `image.bytes` 计算 SHA-256。同一图的 human/machine 多条 QA 通过哈希归组。
- **S1-UDF 映射**：`image` 到 asset；`query` 与 `label[]` 到 `annotations[type=qa]`，多答案完整保留；`human_or_machine` 放 annotation metadata/provenance。
- **坐标、重复与风险**：无结构化坐标字段。按行切分会把同一 chart 的不同问题泄漏到两边；禁止用问题文本哈希替代图片哈希。

### 3. `GQA`

- **容器与行数（服务器实测）**：多 split、多 `all/balanced`、`images/instructions` Parquet。图片表共 192,695 行；无答案问题视图共 6,437,083 行；富问题视图共 17,577,023 行。它们包含重叠视图，不能相加为唯一问题数。
- **精确核心字段**：图片表 `id`、`image`。富问题 schema 包含 `id`、`imageId`、`question`、`answer`、`fullAnswer`、`isBalanced`、`groups`、`entailed`、`equivalent`、`types`、`annotations`（含文本 span 到 `objectId`）、`semantic` reasoning program；submission/challenge 等视图没有完整 answer 字段。
- **媒体分组键**：`imageId` join 图片表 `id`，并用图片 bytes 哈希审计。`all`、`balanced`、instruction 视图中引用同一 `imageId` 的记录全部同组。
- **S1-UDF 映射**：图片到 asset；问题/答案到 QA annotation；`semantic` 到 `reasoning_program`；`entailed/equivalent`、`groups/types` 和 span-object 链接保留到 `evidence`/`extensions`；对象引用只有在 scene graph 实体实际导入后才可生成 `entity` part。
- **坐标、重复与风险**：本地这些 Parquet 没有建立 object ID 到 bbox 的完整几何 join；不能凭 span 生成 bbox。`all/balanced` 和不同任务视图高度重叠，必须先按稳定问题 ID 去重，再按图片连通分量切分。

### 4. `Molmo2-VideoCapQA`

- **容器与行数（服务器实测）**：`CapQA-00000-of-00001.parquet` 950,579 行；`LongCapQA-00000-of-00001.parquet` 49,996 行；另有 `youtube_id_to_urls_mapping.json`。
- **精确核心字段**：CapQA 为 `video_id`、`Question`、`Answer`、`Category`、`NegativeAnswers`；LongCapQA 以 `video_id` 和嵌套 `qa_list` 组织。映射 JSON 提供视频到 GCP/YouTube URL 的关系。
- **媒体分组键**：以完整 `video_id` 为最低分组粒度；解析到实际媒体后增加内容哈希。CapQA 与 LongCapQA 的相同 video 必须连接。
- **S1-UDF 映射**：视频引用到 `assets[kind=video]`；QA 到 annotations；`NegativeAnswers` 保留为候选/负例 metadata，不能混作多个正确答案；`Category` 保留任务分类。
- **坐标、重复与风险**：无几何字段。URL 存在失效、访问受限或同 ID 多 URL 风险；本轮没有确认所有 URL 可访问，也没有确认 loose video。媒体不可用时仍可保存 source URI，但不能把“有 URL”报告成“媒体已校验”。

### 5. `Molmo2-VideoPoint`

- **容器与行数（服务器实测）**：主 `data/train-00000-of-00001.parquet` 为 658,340 行；8 个类别目录各有 Parquet，它们是主表的分类视图/重复子集，并额外带索引列 `__index_level_0__`，不可再次累加。另有 URL mapping JSON。`generated_videos/` 下有 `sora2-1128.tar.gz`、`sora2.tar.gz`、`videos.tar.gz` 三个视频支持归档；它们是媒体库存，不是另一套 point 监督 schema，成员与 `video_id` 的对应关系仍需 adapter 校验。
- **精确核心字段**：主表为 `video_id`、`question`、`label`、`count`、`two_fps_timestamps`、`points`（每时刻为对象列表，点含 `x/y`）、`raw_frames`、`raw_timestamps`、`annotator_unsure`、`category`、`video_duration`、`video_source`、`clip_start`、`clip_end`。审计样本中四个逐帧数组按位对应，adapter 必须逐行检查等长。
- **媒体分组键**：完整 `video_id`；`clip_start/clip_end` 只描述监督时间范围，不能默认作为 split group。若同视频 ID 映射到多个字节对象，按二部图继续合并/报冲突。
- **S1-UDF 映射**：视频到 asset；时间戳写 region/annotation 的 `time_span` 或 track observation timestamp；每个点写 `regions[geometry.type=point2d]`，连续对象可写 tracks；`question/label/category/count/annotator_unsure` 保留在 annotation/metadata，`video_duration` 写 asset media，`clip_start/clip_end` 写监督时间窗。
- **坐标、重复与风险**：`x/y` 实测使用 0..100 百分坐标，不是 `norm1`。只有显式除以 100 后才可标 `normalized`；原值和换算策略需进 provenance。类别 Parquet 是重复视图；逐帧数组错位、null/不确定点、时间单位与 clip 边界都要在 adapter 中校验。

### 6. `Molmo2-VideoSubtitleQA`

- **容器与行数（服务器实测）**：3 个 `SubtitleQA` Parquet shard，共 468,502 行；另有 `youtube_id_to_urls_mapping.json`。
- **精确核心字段**：`video_id`、`AlignmentType`、`Question`、`Answer`、`Category`、`subtitle`（元素含 `start/end/text`）、`NegativeAnswers`。
- **媒体分组键**：完整 `video_id`，同视频的所有字幕片段和 QA 同组；与其他 Molmo2 数据合并时，只有经 URL/内容映射确认是同一底层视频才能跨目录连接。
- **S1-UDF 映射**：视频 asset；subtitle 作为带 `time_span` 的 text annotation/evidence；问答和负候选分别保留；`AlignmentType/Category` 写 metadata。
- **坐标、重复与风险**：没有空间坐标。`start/end` 的单位和是否相对 clip 起点必须读取来源说明并抽样核验，本轮不能仅凭数值宣布；先检查 `end >= start` 及是否超出媒体 duration。URL 可达性未确认。

### 7. `Molmo2-VideoTrack`

- **容器与行数（服务器实测）**：APTv2、animaltrack、bdd100k、bft、dancetrack、mose、mosev2、mot2020、personpath22、sav、seadrones、soccernet、sportsmot、teamtrack、uavdt、vipseg 等来源的 point-track Parquet，共 29,704 行；多数目录另有 `point_tracks.json`，部分保留 ZIP/meta 文件。
- **精确核心字段**：`id`、`video`、`clip`、`video_dataset`、`video_source`、`exp`、`obj_id`、`mask_id`、`points`（元素含 `object_id` 和逐帧 `points: [[x,y] | null]`）、`segments`、`start_frame`、`end_frame`、`w`、`h`、`n_frames`、`fps`。实测 JSON 视图缺少 `clip`，不能假设它与 Parquet 等价。
- **媒体分组键**：最少用 `(video_dataset, video)`，而不是行级 `id` 或 `clip`；同完整视频的所有对象和片段同组。跨来源同名 video 不能合并，除非内容哈希一致。
- **S1-UDF 映射**：视频到 asset；`exp` 到实体的 surface/对话文本；每帧非 null 点到 pixel `point2d` region；`object_id` 连接 entity/track，`mask_id` 保留来源链接；`segments/start_frame/end_frame/fps/n_frames` 写 track/asset 时间元数据。
- **坐标、重复与风险**：点为基于 `w/h` 的像素坐标，null 是缺失/不可见槽，必须保留时序位置。JSON 缺 `clip`、MOSE/MOSEv2 meta 与归档未完全内联验证；重叠 clip 会泄漏，因此默认完整视频分组。逐帧点数须与 `n_frames`/segment 定义校验。

### 8. `VQAv2`

- **容器与行数（服务器实测）**：Parquet 总计 1,200,753 行：train 443,757、validation 211,202、testdev 107,394、test 438,400。
- **精确核心字段**：`question_type`、`multiple_choice_answer`、`answer_type`、`question`、`answers`（10 项）、`image_id`、`question_id`、嵌入 `image` bytes；测试视图的监督字段可能为空/不可用。
- **媒体分组键**：`image_id` 加图片 bytes 哈希；`testdev` 是 test 的子集语义，跨视图相同媒体必须连接。`question_id` 只用于记录身份。
- **S1-UDF 映射**：图片 asset；问题到 QA annotation；10 个答案逐项保存 text、置信/计数信息，`multiple_choice_answer` 可作 canonical answer，但不能删除原始共识；question/answer type 写 metadata。
- **坐标、重复与风险**：无 bbox。单图多问题是预期重复，按行切分必然泄漏；testdev/test 物理行不可当独立样本相加。导出单一 assistant 答案前必须记录答案聚合策略。

### 9. `VisualGenome`

- **容器与行数（服务器实测）**：12 个完整 ZIP。两个媒体归档 `images.zip`、`images2.zip` 分别含 64,346 和 43,903 个 JPG，共 108,249 张；其余 10 个 annotation ZIP 为 `attributes`、`image_data`、`objects_v1_2`、`qa_to_region_mapping`、`question_answers`、`region_descriptions`、`region_graphs`、`relationships_v1_2`、`scene_graphs`、`synsets`。旁边的 `images2.zip.1` 无有效 ZIP 中央目录，必须排除，不能计作第 13 个归档或图片来源。
- **精确核心字段**：已通过 ZIP 流式只读确认 10 个 annotation JSON 的根结构和首条结构：除 `qa_to_region_mapping.json` 外均为顶层数组，覆盖 `image_id` 与图像尺寸/URL/COCO/Flickr ID，对象和属性的 `object_id/names/synsets/x/y/w/h/attributes`，QA 的 `qa_id/question/answer/q_objects/a_objects`，region 的 `region_id/phrase/x/y/width/height`，region graph 的 objects/relationships/synset 文本跨度，关系的 subject/object/predicate/relationship_id，scene graph 及 `synset_name/synset_definition`。`qa_to_region_mapping.json` 已全量解析为 1,006,968 个唯一十进制字符串 key 到唯一整数 value 的映射，value 范围为 1..2,759,876；其根类型、全部键值类型、唯一性和数值范围均为 verified。
- **媒体分组键**：官方 `image_id` join 各 annotation 归档并关联对应 JPG；转换时以图片哈希复核 `images` 与 `images2`，还要与 GQA/COCO 等潜在副本连接。
- **S1-UDF 映射**：对象到 entities，`x/y/w/h` 框与 phrase region 到 regions，关系到 relations，QA/region description 到 annotations，synset 保留 taxonomy；所有跨文件 ID 必须先验证存在。
- **坐标、重复与风险**：首条样本显示 `x/y/w/h`，但全量坐标范围、退化框和全部 annotation 跨归档引用完整性尚未审计。首图 91 个 QA 均能在 mapping 找到 key，但只有 88 个目标能 join 到同图 region；缺失例为 `986809 -> 2084`、`986871 -> 3235`、`986905 -> 3785`。因此 mapping 文件自身的结构统计是 verified，**全量 QA-region 引用完整性仍为 unresolved**；adapter 必须同时校验 QA 和 region 的 `image_id`，不能把跨图或缺失 region 静默连接。不得直接解压覆盖现有目录；需在临时只读工作区做 ZIP 完整性和成员路径检查，防止路径穿越，并报告重复 image/region/object ID。

### 10. `ai2d`

- **容器与行数（服务器实测）**：`ai2d-all.zip` 加 loose 提取树；有效图片/annotation 对 4,903 个，question 文件关联 4,563 张图。`__MACOSX` 下的 `._*` 是资源叉伪文件，不计数据。
- **精确核心字段**：diagram annotation 含 `arrowHeads`、`arrows`、`blobs`、`containers`、`relationships`、`text`，形状为 rectangle/polygon 等；question 文件含 `imageName` 与 `questions` 映射，每项为 `abcLabel`、`answerTexts`、`correctAnswer`、`questionId`。
- **媒体分组键**：`imageName` 对应 diagram 图片；同图的全部结构标注和问题同组，并用图片哈希检查重名/副本。
- **S1-UDF 映射**：图片 asset；diagram 元素到 entities/regions，箭头和容器关系到 relations；问题到多项选择 QA annotation，保存 choices 和正确索引。
- **坐标、重复与风险**：来源几何为图像像素空间，rectangle/polygon 必须保留原格式和顶点顺序并按图片宽高验界。忽略 `__MACOSX` 与 `._*`；ZIP 与 loose 树可能是同一内容，不能重复导入。

### 11. `llava-instruct`

- **容器与行数（服务器实测）**：六个顶层 JSON 数组精确长度分别为：`complex_reasoning_77k.json` 76,643、`conversation_58k.json` 56,681、`detail_23k.json` 23,240、`llava_instruct_150k.json` 157,712、`llava_instruct_80k.json` 80,000、`llava_v1_5_mix665k.json` 665,298。另有 79,995 个 JPG。内含 OCR-VQA Parquet 共 207,549 行。`datasets/coco2017/` 还库存 `train2017.zip`、`val2017.zip` 两个图片媒体 ZIP 和 `annotations_trainval2017.zip` 一个支持 annotation ZIP；三者是 COCO 媒体/辅助归档，不是上述 LLaVA 对话 JSON 或 OCR-VQA Parquet 的额外监督行。
- **精确核心字段**：LLaVA 数组为 `id`、`image`、`conversations`，turn 为 `from`、`value`。OCR-VQA 为 `image`、`image_id`、`questions`、`answers`、`ocr_tokens`、`ocr_info`、`title`、`authorName`、`genre`、`image_width`、`image_height`、`image_url`、`set_name`；`ocr_info` 含 `word` 和由 `width/height/top_left_x/top_left_y` 构成的 normalized bbox。
- **媒体分组键**：LLaVA 用 `image` 路径解析后的内容哈希；OCR-VQA 用 `image_id` 加图片 bytes 哈希。六个 JSON 的命名集合可能互为拼接/子集，不能把 1,059,574 行宣称为唯一对话；必须按 record ID 和媒体图做重复审计。
- **S1-UDF 映射**：LLaVA image 到 asset，`from=human/gpt` 显式映射 user/assistant，按原 `value` 中媒体 token 位置构造 content parts；OCR word/bbox 到 entities/regions，questions/answers 到 QA annotations。
- **坐标、重复与风险**：OCR bbox 已标 normalized，仍需检查范围、格式和旋转语义；不得再次按 width/height 除法。LLaVA 对话文件高度重叠且 COCO 图片也可能与其他顶层目录重复，合并训练时须跨目录做内容哈希连通分量。

### 12. `pixmo-cap`

- **容器与行数（服务器实测）**：4 个 train Parquet，共 717,042 行。
- **精确核心字段**：`image_url`、`caption`、`transcripts`。`transcripts` 实测为文本，不是音频资源。
- **媒体分组键**：规范化 `image_url`；下载/缓存到媒体后应改用图片 SHA-256，并把多个 URL 指向同字节的记录连接。
- **S1-UDF 映射**：URL 到 image asset；caption 到 caption annotation；transcripts 作为来源文本/证据或额外 annotation，不能生成 `audio` asset。
- **坐标、重复与风险**：无几何字段。URL 可能失效、重定向或内容漂移；没有字节哈希时只能把 URL 身份标为较弱证据。同图多 caption/transcript 不得跨 split。

### 13. `pixmo-points`

- **容器与行数（服务器实测）**：2 个 train Parquet，共 2,376,222 行。
- **精确核心字段**：`image_url`、`image_sha256`、`label`、`collection_method`、`points`、`count`。
- **媒体分组键**：首选 `image_sha256`，URL 仅作定位；同一 SHA 的所有 label/points 同组。
- **S1-UDF 映射**：图片 asset 写 `sha256`；label 到 entity；每点到 point2d region；`collection_method/count` 写 annotation metadata，并验证 count 与点数量语义。
- **坐标、重复与风险**：points 使用 0..100 百分坐标，审计发现少量轻微负值。不得静默 clamp，也不得直接标 `norm1`；先记录异常，再按明确策略除以 100。重复 SHA 是同图多监督，不是可独立切分样本。

### 14. `robo2vlm`

- **容器与行数（服务器实测）**：train/test Parquet 的唯一逻辑行 684,710；物理扫描为 810,797 行，因为嵌套 `data/data` 中有 126,087 行副本。
- **精确核心字段**：`id`、`question`、`choices`、`correct_answer`、嵌入 `image` bytes。`choices` 实测是 Python list 的字符串表示，而不是 Parquet list。
- **媒体分组键**：图片 bytes SHA-256；`id` 是问答记录 ID，不能用作媒体组。
- **S1-UDF 映射**：图片 asset；question/choices/correct answer 到 multiple-choice QA annotation；原 ID 到 provenance。
- **坐标、重复与风险**：无几何字段。`choices` 只能用受限的 literal parser 并校验结果为字符串列表，禁止 `eval`；解析失败须报告。导入前按内容/记录 ID 去掉嵌套副本，再按图片分组。

### 15. `robovqa`

- **容器与行数（服务器实测）**：6 个顶层 JSON 数组。五个 reasoning 文件分别为 184,003、184,003、184,003、184,003、183,999 行，reasoning 合计 920,011；`robovqa_understanding.json` 为 218,504 行；物理总计 1,138,515 行。`clips/` 另有 100 个 `clips_part_000.tar.gz` 至 `clips_part_099.tar.gz` 视频媒体归档；它们是 JSON 中视频引用的候选媒体库存，不是 100 个监督 shard，也不能在未核验成员映射前增加监督样本数。
- **精确核心字段**：顶层 `video`、`conversations`、`metadata`；turn 为 `role/content`；metadata 含 `width`、`height`、`num_frames`、`num_bytes`、`location`、`framerate`、`task_metadata`，后者含 `uid`、`split`、`task`、`video`、`question`、`answer`、`video_id` 等。
- **媒体分组键**：优先 `task_metadata.video_id`，用 `metadata.location`/实际视频 bytes 复核；同 video 的 understanding/reasoning 和所有问题同组。`uid` 只作记录 ID。
- **S1-UDF 映射**：视频 asset 的宽高、帧数、字节数、FPS 写 media；对话按 role/content 写 conversations；任务元数据保留 provenance/annotation。只有存在真实 step/action/state 证据时才映射 episode，不能仅因名称含 Robo 就臆造机器人轨迹。
- **坐标、重复与风险**：未实测结构化 bbox。`location` 指向性与媒体可达性需要转换时检查；六个数组可能共享相同视频，必须全局连接。原 metadata 的 `split` 若与新 group split 冲突，应报告而不是按行保留。

### 16. `robovqa-2unuse`

- **容器与行数（服务器实测）**：`json/train|val/data-*.json` 实际是 newline JSON，共 221,912 行；精确字段为 `uid`、`text`、`video`。TFRecord 数据实测为 `tf.train.SequenceExample`：两处目录有 94 + 20 = 114 个物理 data shard，首条 record header/CRC 均通过；其中 19 个是重复 shard，按 shard identity 合并后为预期 `175` 个编号中的 95 个唯一 shard，仍缺 80 个。cache 中的 `.lock`、`.metadata` 和 incomplete 文件不是数据 shard，必须排除。
- **SequenceExample schema（服务器抽样实测）**：context 的 `unique_id`、`video_filename` 均为单值 `bytes_list[1]`。抽样 record 的 feature_lists 中，`images` 为 16 step、每 step 一个 JPEG `bytes_list[1]`，`timestamps` 为 16 step、每 step 一个 `int64_list[1]`；`raw_texts`、`texts` 各为 1 step 的 `bytes_list[1]`，`texts_start`、`texts_end` 各为 1 step 的 `int64_list[1]`，抽样值为 15。字段类型是实测证据，但 16/1 的基数尚未全量证明为固定契约。两个 shard 已完整扫描到精确 record 边界，分别为 1,362 和 1,369 条；这不能外推为全库 example 数。
- **错放容器（服务器实测）**：`data/*.parquet` 的 schema 与 TextVQA 相同，而不是 RoboVQA；共有 train/test/validation 27 shard，对应 45,336 行语料，禁止按目录名导入为 RoboVQA。
- **媒体分组键**：raw JSON 按 `video` 指向的底层视频或内容哈希归组；SequenceExample 按 context `video_filename` 连接同一底层视频，并用嵌入 JPEG/实际视频 bytes 复核，`unique_id`/`uid` 只作记录 ID。错放的 TextVQA Parquet 按 `image_id`/图片哈希归组。
- **S1-UDF 映射**：raw `text` 只有在解析出明确 role/任务协议后才能进入 conversation，否则先作为 text annotation/raw record；video 到 asset。SequenceExample 的 JPEG steps 按顺序写图像 asset/observation 引用，`timestamps` 和 text start/end 保留为序列 evidence；当前没有 action/state 证据，不得仅因数据集名把它映射成机器人控制 episode。
- **坐标、重复与风险**：目录名已标 `2unuse`，且存在错放数据、重复和缺失 TFRecord，默认排除训练。95 个唯一 shard 中除上述两个外尚未完成逐 record CRC/边界全扫，80 个 shard 缺失，`timestamps` 与 text start/end 的单位和基准也未确认；因此剩余全量 CRC、总 example 数、时序语义和与 raw JSON 的全量 join 均为 **unresolved**。

### 17. `spatialvlm`

- **容器与行数（服务器实测）**：Parquet 共 28,039 行。
- **精确核心字段**：`messages`（元素含 `role` 和结构化 `content` parts），part 使用 `type`、`index`、`text`；`images` 为列表，元素含 `bytes/path`。
- **媒体分组键**：一条记录引用的全部 image bytes 构成边；对每张图片计算 SHA-256，并对共享任一图片的记录取连通分量，而不是对整组路径做一次独立哈希。
- **S1-UDF 映射**：images 到多个 image assets；按 part 原顺序将 image `index` 转成 asset content part，text 转 text part；role 显式映射 conversations。
- **坐标、重复与风险**：未实测独立 geometry 列。必须检查每个 image index 的基数、范围和 token 顺序；多图记录最容易出现 `{A,B}`/`{A,C}` 泄漏，单记录媒体集合哈希不足以防止。

### 18. `textvqa`

- **容器与行数（服务器实测）**：Parquet 共 45,336 行：train 34,602、validation 5,000、test 5,734。
- **精确核心字段**：`image_id`、`question_id`、`question`、`question_tokens`、`image`、`image_width`、`image_height`、`flickr_original_url`、`flickr_300k_url`、`answers`（10 项）、`image_classes`、`set_name`、`ocr_tokens`。
- **媒体分组键**：`image_id` 加图片 bytes SHA-256；question ID 仅是记录身份。URL 只作 provenance/备用定位。
- **S1-UDF 映射**：图片 asset；问题与 10 答案到 QA annotation；OCR token 保留为 OCR annotation/metadata，只有来源提供对应 geometry 时才能生成 region。
- **坐标、重复与风险**：当前 schema 未确认 OCR token 到 bbox 的结构化对应，不能虚构 OCR 框。同图多问题全部同组；答案导出单 target 前必须保留共识信息。

### 19. `textvqa-2unuse`

- **容器与行数（服务器实测）**：与 `textvqa` 的 Parquet schema、split 行数完全一致，共 45,336 行；本轮确认它是副本。
- **精确核心字段**：同 `textvqa`：`image_id/question_id/question/question_tokens/image/image_width/image_height/flickr_original_url/flickr_300k_url/answers/image_classes/set_name/ocr_tokens`。
- **媒体分组键**：同 `textvqa`，并需跨两个顶层目录按图片和记录身份去重。
- **S1-UDF 映射**：与 `textvqa` 相同；provenance 应指向选定的唯一物理来源。
- **坐标、重复与风险**：默认排除该 `2unuse` 副本，不能把两个目录拼接后宣称样本翻倍。OCR bbox 仍未确认。

### 20. `vlm-r1`

- **主表与媒体引用（服务器实测）**：`sft_related/mllm_rec_json.json` 是顶层数组，共 321,308 行，含 28,157 个不同图像引用，同一图像最多关联 77 行；因此必须先按图像身份分组，不能按 SFT 行切分。主数组含 `messages`、`images`；assistant 文本内含 Markdown code fence 包裹的 JSON 并出现 `bbox_2d`。`images` 中的旧 `/data/shz/...` 路径在当前服务器失效。
- **六个 ZIP 库存（服务器实测）**：`gui_multi-image.zip` 含 3,594 个 PNG，没有配套 JSON/JSONL 监督；`lisa_test.zip` 含 779 个 JPG 和 779 个同 stem JSON，配对完整；`rec_jsons_internvl.zip` 与 `rec_jsons_processed.zip` 各含 16 个 JSON 和 4 个 JSONL；`refgta.zip` 含 502 个 PNG；`train2014.zip` 含 82,783 个 JPG。GUI、RefGTA 和 train2014 的成员计数是媒体库存，不等于新增监督记录数。
- **LISA schema（服务器抽样实测）**：JSON root 为 `is_sentence: bool`、`text: list[str]`、`shapes: list`；shape 含 `flags: object`、`group_id: null`、`group_ids: list`、`image_name: str`、`label: str`、`labels: list`、`points: list`、`shape_type: str`。已抽查 10 个 JSON、44 个 shape，`shape_type` 均为 `polygon`；779 对同 stem 只证明成员配对，不能把 10 文件的字段/类型分布外推到全部 JSON。
- **REC JSON/JSONL schema（服务器抽样实测）**：两个 REC ZIP 的成员首条记录共归为 12 个 schema family。普通数组记录包含 `image/problem/normal_caption`，InternVL 的 `solution` 为 0..1000 整数列表；processed 记录增加 `normalized_solution`（0..1000 整数列表），其 `solution` 为原图 pixel float 列表。train 变体还含 `bbox_list/area/category_id/id/image_id/original_id/iscrowd/width/height`。JSONL 为 `image: str`、`conversations: list[{from: str, value: str | list[number]}]`，抽样中 human value 为字符串、gpt value 为四坐标列表。
- **媒体分组键**：必须先把旧路径映射到实际图片并校验 bytes，再对 28,157 个主表图像引用及 ZIP 内引用统一计算 SHA-256；同图最多 77 行必须同组，多图记录继续按媒体二部图取传递连通分量。仅对失效路径字符串做哈希会制造错误分组。
- **S1-UDF 映射**：原 messages 可先无损进入 conversations；LISA polygon 写 polygon region，但须先结合配对 JPG 尺寸验证坐标空间，不能仅凭 `points` 字段宣布为 pixel。InternVL `solution` 和 processed `normalized_solution` 的 0..1000 整数是模型刻度，**不是 `norm1`**，只能保留自定义坐标空间或显式除以 1000 后再标 normalized；processed `solution` 是原图 pixel float，但仍须确认列表长度/坐标顺序并以对应 width/height 验界后才能写 region。JSONL 四坐标只有在确认顺序和目标图像后才能写 bbox。所有解析前后文本与坐标来源都保留 provenance。
- **坐标、缺失与未决边界**：主 `mllm_rec_json.json` 的 `bbox_2d` 尺度尚未全量确认，不能从 REC ZIP 的坐标规则反推。服务器未找到主表所需的 Flickr 媒体；GUI ZIP 没有监督文件，RefGTA/train2014 也只验证了媒体成员库存。REC 的 12 个 family 来自每成员首条记录，LISA shape 只抽查 10 文件；全量可选字段/类型漂移、JSONL 全行语义、Flickr 媒体恢复、GUI/RefGTA 与外部监督的 join、主表路径到 ZIP 成员的全量映射均为 **unresolved**。任何 parse failure、缺媒体或无监督媒体都必须进入 loss/unsupported report，不能直接用于训练。

## 主流视觉/视频格式覆盖（官方参考，不是服务器实测）

下表用于设计未来 adapter。除上节明确列出的本地快照外，这些字段均来自官方规范；落地时仍要对实际下载版本重新 profile。

| 官方格式 | 核心 schema 与 S1-UDF 映射 | 分组、坐标与保真要求 |
| --- | --- | --- |
| COCO detection/keypoints/caption/panoptic | `images{id,file_name,width,height}`、`annotations{image_id,category_id,bbox,segmentation,area,iscrowd}`、`categories`；keypoints 为 `(x,y,v)` 与 skeleton；caption 一图多 annotation；panoptic PNG 的 RGB 编码 segment ID 配 `segments_info`。映射为 assets/entities/regions/annotations。 | `group=image_id` 并用 bytes hash 复核。bbox 是 pixel `xywh`；segmentation 是 polygon 或 RLE；`v` 为 0/1/2。panoptic mask 不能仅当普通 RGB 图。 |
| LVIS | COCO 风格 image/annotation/category；额外保留 image 的 `neg_category_ids/not_exhaustive_category_ids`，category 的 `synonyms/synset/def/frequency`。映射为 assets/entities/regions/annotations。 | 以图片分组并用媒体 hash 与 COCO 等复用图片的来源联合去重；bbox 为 pixel `xywh`，segmentation 为 polygon/RLE。负类别和非穷尽标志影响 loss，不能丢到普通 COCO 类别表之外。 |
| Open Images V7 | 多 CSV：image metadata/license、class description、normalized bbox，及 occluded/truncated/groupOf/depiction/inside、segmentation mask、关系、rotation、extreme clicks。映射 assets/entities/regions/relations/annotations。 | 以 `ImageID` 分组并校验媒体 hash。bbox 为 normalized `XMin,XMax,YMin,YMax`；rotation 会改变显示坐标。extreme-click 坐标中的 `-1` 是未标注 sentinel，不能生成 point region。 |
| Visual Genome | image、objects、attributes、relationships、region descriptions、QA、synsets 和 phrase/region 映射形成多文件 ID 图。 | 以 `image_id` 分组；官方 bbox 通常为 pixel `x,y,w,h`，落地必须以下载版本复核。object/region/relation 跨文件引用失败时硬报错。 |
| VQAv2 | questions 通过 `question_id/image_id` 连接 annotations；带 annotation 的 train/val 每题有 10 个 annotator answer，含 `answer_confidence`、question/answer type 和 multiple-choice answer；test 只有 questions，没有 GT annotation。 | 以 `image_id` 分组；train/val 的 10 个答案完整存入 `answers[]`，test 不得伪造答案。不能先展开答案再随机切分。 |
| TextVQA | question 通过 `question_id/image_id` 连接图片；train/val 可含 10 个 answers、OCR tokens 和 OCR metadata（如 token bbox/confidence 及 rotation/yaw/roll/pitch）。映射 QA 到 annotations/conversations，OCR 到 entities/regions/extensions。 | 以 `image_id`/媒体 hash 分组。answers 与 OCR token 顺序完整保留；test 或具体 release 缺失的监督字段不得补造，OCR 坐标口径以该 release schema 为准。 |
| TextOCR v0.1 | 根对象含 `info/imgs/anns/img2Anns`；annotation 含 `id,image_id,bbox,points,utf8_string,area`，其中 `points` 是任意 polygon 真值。映射图片到 assets、文本到 entities、polygon/bbox 到 regions。 | 以 `image_id`/媒体 hash 分组。polygon 不得降成 bbox；官方 v0.1 未定义 reading order 或文档层级，adapter 不得从数组顺序臆造。 |
| DocVQA family | release/task 分别包括 Single Page DocVQA、Multi-page DocVQA、InfographicVQA 等；共同核心是 page/document 媒体、question/answers，OCR、层级、evidence page/span 仅在具体任务发布时保留。 | 单页任务至少按 page/media 分组，多页任务按完整 document 分组；不能把不同 release 的可选字段当成通用必填项。页号和已有 evidence 在 ms-swift 投影之外仍须保留。 |
| Flickr30k Entities | sentence token/phrase 通过 phrase ID 连接一个或多个 bbox；部分 phrase 明确没有 box。映射 entity、多个 region 和 grounded caption。 | 以 image ID 分组。phrase-to-box 是一对多，no-box 是合法状态；不能伪造空框或强行一一配对。 |
| Ego4D | canonical video、clip、秒级时间、frame/PTS/timebase；NLQ/VQ 等 annotation 含 query、temporal window，视觉 query 还可含 response track。 | 默认按 canonical video 分组；至少保留 clip 秒、frame、PTS、timebase 的换算链。不要把相邻/重叠 clip 随机分开。 |
| MOTChallenge | GT 常见 9 列为 `frame,id,bb_left,bb_top,bb_width,bb_height,mark,class,visibility`；tracker submission 为 10 列 `frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z`。帧/视频映射 assets，bbox 映射 pixel bbox2d regions，target ID 映射 tracks。 | 按完整 sequence 分组；frame 与 track ID 通常 1-based。转换时显式改为 0-based frame 或保留来源基准，且按 benchmark/version 区分 GT 与 submission，不能混用列语义。 |
| YouTube-VIS 2019/2021/2022 | 各 release 的 video 记录含 frame file list、width/height；instance annotation 对每帧存 bbox/segmentation 数组，目标缺席帧为 `null`。映射 assets/regions/tracks/annotations，并记录 release。 | 按完整 video 分组。保持数组长度与 frame 数一致，`null` 代表缺席；不可压掉空槽导致时序位移，不同年份的类别表和 split 不得无版本混合。 |
| NYU Depth V2 | RGB、depth、filled/inpainted depth、labels/instances、相机参数/同步信息。labeled `.mat` 的 `rawDepths` 已投影到 RGB image plane，单位为米但未填洞；raw sequence 另含需按标定对齐的原始流。 | 以 scene/sequence 分组。`rawDepths` 与 filled depth 必须是不同 asset/channel，并保留 invalid mask、投影状态、坐标系和标定 provenance；不得对已投影的 labeled depth 重复投影。 |
| KITTI | object detection label 含 class、truncation、occlusion、alpha、2D bbox、3D dimensions `h,w,l`、location `x,y,z`、`rotation_y`，预测可追加 `score`；raw/tracking/odometry release 另有 sequence、相机/LiDAR calibration、pose 和时间。 | Object Detection 默认按独立 frame/media 分组，只有可靠恢复 raw drive 映射后才合并；tracking/raw/odometry 按完整 sequence/drive。2D 是 pixel，3D box 在 camera frame；维度顺序和 camera/LiDAR 变换必须显式保留。 |
| Pascal VOC | 每图 XML `annotation` 含 folder/filename/size 和 `object{name,pose,truncated,difficult,bndbox{xmin,ymin,xmax,ymax}}`；segmentation release 另有 class/object mask PNG。映射 assets/entities/regions/annotations。 | 以图片 ID/hash 分组；bbox 为 pixel 边界但索引/闭区间惯例须按使用的 devkit 版本记录。mask 调色板索引是标签，不得当普通 RGB。 |
| YOLO family | 每图标签文本由 class 与 normalized center `x,y,w,h` 组成；segmentation/pose/OBB 等变体使用 polygon、keypoint 或 oriented-box 列，类别表和 split 通常来自 dataset YAML。 | 以对应图片路径/hash 分组；空/缺失标签按具体版本区分负样本与缺失标注。必须记录方言/版本并据此解析列数，不能以检测五列规则猜测所有 YOLO 标签。 |
| Cityscapes | 图像、fine/coarse semantic/instance label PNG、polygon JSON，以及可选 disparity、camera、vehicle/sequence 数据按 city/sequence/frame 命名。映射 assets/entities/regions/annotations 与 camera metadata。 | 单帧任务按图片分组；使用 sequence/video 上下文的任务按完整 sequence 分组。label ID、train ID、instance 编码和 ignore label 均保留，disparity 不得冒充 metric depth。 |
| ICDAR/RRC OCR family | 不同年份/task 的文本文件或 JSON 保存 image/document、四边形或 polygon、transcription，部分任务以 `###` 等标记 ignore/don't-care。映射 assets/entities/regions/annotations。 | 以原图片或完整 document/page 组分组；严格绑定 challenge/year/task 的列定义、编码和坐标顺序。ignore 标记不是普通文字标签，polygon 不得无损失降成 bbox。 |
| ScanNet | `.sens` 多帧 RGB/depth/pose/intrinsics，mesh、vertex segmentation、aggregation/object instance 与 2D projection。 | 按 scan/scene 分组。保留 camera-to-world pose、depth scale、mesh vertex/segment/instance 引用；2D/3D entity ID 要通过 projection 显式连接。 |
| nuScenes | token 关系表：scene/sample/sample_data/ego_pose/calibrated_sensor/instance/sample_annotation 等；多相机、LiDAR、雷达和 map。 | 按 scene 分组。3D box 位于 global frame，size 为 `w,l,h`，rotation 为 quaternion `wxyz`；时间同步依赖 token 和 timestamp，禁止仅按文件名 join。 |
| SemanticKITTI | `.bin` 点云配 `.label` uint32；低 16 位 semantic label，高 16 位 instance ID，另有 sequence pose/calibration。 | 按 sequence 分组。使用位运算拆 label；保留原 uint32 asset/ref、point index 对齐及 moving/static remap，不能把 instance 高位当类别。 |
| Waymo Open Dataset | Frame proto 含 context、timestamp、车辆 pose、多相机图像、LiDAR range image/point、calibration 与 labels。3D label box 含 center、`length,width,height,heading`。 | 按 segment/context 分组。box 位于 vehicle frame；range image 到点云依赖 beam calibration和 pose。不同 camera/LiDAR 帧通过 timestamp/pose 连接，不能按单传感器帧随机切分。 |

官方参考入口：

- COCO：<https://cocodataset.org/#format-data>、<https://github.com/cocodataset/cocoapi>
- LVIS：<https://www.lvisdataset.org/>、<https://github.com/lvis-dataset/lvis-api>
- Open Images：<https://storage.googleapis.com/openimages/web/download_v7.html>
- Visual Genome：<https://homes.cs.washington.edu/~ranjay/visualgenome/api.html>
- VQAv2：<https://visualqa.org/download.html>、<https://visualqa.org/evaluation.html>
- TextVQA：<https://textvqa.org/dataset/>
- TextOCR：<https://textvqa.org/textocr/>
- DocVQA：<https://www.docvqa.org/>、<https://rrc.cvc.uab.es/?ch=17>
- Flickr30k Entities：<https://github.com/BryanPlummer/flickr30k_entities>
- Ego4D：<https://ego4d-data.org/docs/data/annotations-schemas/>
- MOT：<https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official>
- YouTube-VIS：<https://youtube-vos.org/dataset/vis/>、<https://github.com/youtubevos/MaskTrackRCNN>
- NYU Depth V2：<https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html>
- KITTI：<https://www.cvlibs.net/datasets/kitti/>
- Pascal VOC：<http://host.robots.ox.ac.uk/pascal/VOC/>
- YOLO 格式：<https://docs.ultralytics.com/datasets/detect/>
- Cityscapes：<https://www.cityscapes-dataset.com/dataset-overview/>
- ICDAR/RRC：<https://rrc.cvc.uab.es/>
- ScanNet：<https://github.com/ScanNet/ScanNet>
- nuScenes：<https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md>
- SemanticKITTI：<https://www.semantic-kitti.org/dataset.html>、<https://github.com/PRBonn/semantic-kitti-api>
- Waymo：<https://github.com/waymo-research/waymo-open-dataset/tree/master/src/waymo_open_dataset>

## 具身与机器人格式覆盖（官方参考，不是服务器实测）

这些格式的共同切分原子是完整 episode/trajectory/demo，而不是 step 或单帧。动作空间、状态维度和坐标系在数据集间不统一，S1-UDF 负责显式记录差异，不负责猜测统一控制语义。

| 官方格式 | 核心 schema 与 S1-UDF 映射 | 分组与风险 |
| --- | --- | --- |
| RLDS | `episode` 包含 `steps` Dataset；`is_first/is_last` 是核心 step 字段，常见字段另有 `observation/action/reward/discount/is_terminal` 和 episode metadata/`invalid`。映射到 `episode.steps`、assets 与 tensors。 | `group=episode`。保留 episode invalid 状态并校验 flags、step 顺序和 feature spec；`is_last`/`is_terminal` 步的 action、reward、discount 可能无效，adapter 不得直接把它们构造成普通 transition。 |
| Open X-Embodiment | 基于 RLDS 聚合多机器人数据，每个 constituent dataset 的 observation/action/state/language key、维度、单位和控制空间不同。 | 先保存来源 dataset/episode，再按各自 feature spec 映射。禁止仅按列名 `action` 将异构动作拼成同一 tensor；需要 dataset-specific adapter/normalizer。 |
| LeRobot Dataset v3 | Parquet 保存低维时序列，MP4 保存视觉媒体；metadata 描述 episodes、tasks、features、stats 与 chunk/file 索引，一个数据文件可含多个 episode。 | 从 `meta/episodes` 的 offset/length 恢复边界；`group_id` 使用 repo/dataset/version/episode 的 namespaced identity，不能裸用 `episode_index`。frame/timestamp/video frame 严格 join，不能按 Parquet row 或文件切分。 |
| CALVIN | episode 帧包含 static/gripper RGB、depth、robot state、scene state、action；`ep_start_end_ids` 定义 episode 边界，语言 annotation 的 `info.indx` 连接 instruction 与帧区间。 | 以这些官方边界恢复完整 episode/trajectory；跨边界或重叠语言区间按共享帧连通。RGB/depth/state/action 的帧索引必须对齐，并明确 absolute/relative action 与相机标定版本。 |
| ManiSkill demos | HDF5 按 trajectory group 保存 actions、observations 或 environment states；sidecar JSON 保存 env/task、episode metadata、control mode 等。 | `group=trajectory`。HDF5 key 随 observation mode 变化；state-only 与 sensor observations 不可互相伪造，action/control mode 必须一起保留。 |
| Habitat episode | episode 记录 `episode_id`、`scene_id`、`start_position/start_rotation`、goals，任务扩展可有 shortest path/info。 | 以 scene 内 episode 分组；场景 asset 与所有派生 observation 相连。位置/旋转所在坐标系、四元数顺序和 simulator 版本必须显式记录。 |
| RLBench | demo/observation 含多相机 RGB/depth/mask，以及 joint、gripper、末端状态和 task/variation 描述。 | `group=task/variation/episode`。多相机在每 step 全部连接；depth/mask 编码、相机内外参和 action mode 需随数据一起保存。 |
| BridgeData V2 | 发布物同时存在 raw trajectory、NumPy/custom TFRecord 和 TFDS/RLDS 表达；均可包含 observation images/state、actions、自然语言 instruction，但容器字段和 episode flags 不完全相同。 | 按具体 release 的原始 trajectory/episode 边界分组并记录表示版本。相机视角、状态/action 维度及 normalization 不能跨表示猜测；相邻帧或语言片段绝不能独立随机切分。 |
| ROS1 bag / rosbag2 / MCAP | ROS1 bag 或 rosbag2 SQLite3/MCAP 按 topic、message type/schema、record/log timestamp 保存序列化消息；rosbag2 另有 `metadata.yaml`，传感器标定和 `tf/tf_static` 形成坐标图。 | 以原 recording/session 分组；topics 映射 sensors/streams，消息映射 episode steps/assets/tensors。必须保留原始包、type definition、时钟域、QoS/serialization、TF 图和时间戳，不能仅抽帧后丢失同步关系。 |

官方参考入口：

- RLDS：<https://github.com/google-research/rlds>
- Open X-Embodiment：<https://github.com/google-deepmind/open_x_embodiment>
- LeRobot Dataset v3：<https://huggingface.co/docs/lerobot/lerobot-dataset-v3>
- CALVIN：<https://github.com/mees/calvin/blob/main/dataset/README.md>
- ManiSkill demos：<https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html>
- Habitat-Lab：<https://github.com/facebookresearch/habitat-lab>、<https://aihabitat.org/docs/habitat-lab/habitat.core.dataset.Episode.html>
- RLBench：<https://github.com/stepjam/RLBench>
- BridgeData V2：<https://github.com/rail-berkeley/bridge_data_v2>
- ROS bag/rosbag2/MCAP：<https://wiki.ros.org/Bags>、<https://github.com/ros2/rosbag2>、<https://mcap.dev/spec>

## ms-swift 投影规则

S1-UDF 是保真中转层；ms-swift 是训练投影。投影前保留 S1-UDF，不能让 ms-swift 的较窄 schema 反向决定源数据被删除。

### 媒体与对话

- S1-UDF 的结构化 asset content part 按出现顺序渲染为 `<image>`、`<video>`、`<audio>`；对应 `images/videos/audios` 数组必须与同类 token 数量和顺序完全一致。
- ms-swift 资源路径优先使用已验证的绝对路径；远程 URL、失效路径和签名链接先做可达性/脱敏审计。
- 多图记录的 `images` 数组只是单条训练样本的媒体列表，不替代全库 connected-component 分组。
- LLaVA/SpatialVLM 等来源的媒体 token 必须按原消息位置转成 content part；不能统一前置 token 而改变语义。

### VQA 与 caption

- VQA 通常投影为 user 的媒体 token + question、assistant 的一个训练目标；但 VQAv2/TextVQA 的 10 人答案、GQA program、negative answers 和候选项先完整保存在 S1-UDF。
- 从多答案选一个 assistant target、按频次合并答案或展开多条 SFT，都是有损策略；必须显式配置并输出 loss report。若展开，所有派生记录继承同一 `group_id`。
- 多选 VQA 的 `choices` 按来源数组顺序确定性渲染为 `Choices:` 与从 1 开始的编号；assistant target 仍由明确的 canonical answer/answer policy 决定。不得静默丢弃候选项，也不得把 `correct_choice_indices` 猜写成另一种答案格式。
- caption 可投影为 user 指令加 assistant caption；同图多 caption 可以展开，但不得跨 split。

### Grounding、point、bbox 与多图

- `<ref-object>` 与 `<bbox>` 是两条独立占位符序列：`objects.ref` 长度必须等于 `<ref-object>` 数，`objects.bbox` 长度必须等于 `<bbox>` 数；不得臆造二者一一对应。
- `objects.bbox` 每项长度 2 表示 point，每项长度 4 表示 2D box。S1-UDF 中必须先分别建 `point2d`/`bbox2d` region，导出时再按 ms-swift 契约渲染。
- `bbox_type=real` 只用于真实像素坐标；`bbox_type=norm1` 只用于已归一化到 `[0,1]` 的坐标。PixMo/Molmo 的 0..100 坐标必须显式除以 100；轻微负值先报异常，不能静默 clamp。
- pixel `xywh/cxcywh` 导出前可确定性转换为 `xyxy`，但必须有对应图片 width/height 并保留原格式 provenance。混合 real/norm1 时只允许显式统一策略。
- 多图 grounding 的 `objects.image_id` 与 bbox 一一对齐，索引从 0 开始且必须小于 `images` 长度；没有可靠 region-to-asset 引用时停止导出。
- `vlm-r1` 的 `bbox_2d` 坐标尺度尚未确认，确认前不得标成 real 或 norm1，也不得直接用于训练。

### 不能直接通用投影的任务

- segmentation/RLE/mask、tracking 的整条时序、3D box/pose/point cloud、关系图、OCR reading order、episode/action/state/calibration 没有一个无损的通用 ms-swift 表达。
- 这些任务必须使用目标模型专用 prompt/serializer，把监督显式渲染进 conversation；否则导出器应硬失败或将记录列入 unsupported report，不能只导出对话并静默丢掉 annotations/tracks/episode。
- 任何目标专用渲染产生的新文本都应记录 serializer 名称、版本、参数和源 region/track/step ID，以便回溯。

## Adapter 验收清单

每个来源 adapter 合并前至少满足：

1. 固定并打印实际 source schema、文件清单、物理行数、逻辑去重数和抽样版本；schema 漂移时 fail closed。
2. 输出 record ID 唯一性、媒体 ID 到 bytes 的一对多/多对一冲突、坏路径和缺媒体统计。
3. 在**全量记录**上构建媒体二部图连通分量，再生成 `group_id` 和 split；禁止 worker 分片内各自分组后直接合并。
4. 校验所有 entity/region/track/asset/episode 引用、逐帧数组长度、时间单调性、bbox/point 范围和 coordinate-space 元数据。
5. 去重策略区分物理副本、同媒体不同监督和真正重复监督；不能把同图不同 QA 当重复删除。
6. 转换后运行 S1-UDF JSON Schema 与语义校验；抽样反向回溯到原 record，验证文字、媒体顺序、答案、坐标和时序。
7. 导出 ms-swift 时生成 loss/unsupported report；至少统计多答案聚合、未导出的 annotation/track/episode、坐标换算、失效路径和解析失败。
8. train/val 最终审计必须证明没有相同 `group_id`、媒体 SHA-256、稳定媒体 ID 或 connected component 跨 split。
