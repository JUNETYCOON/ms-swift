#!/usr/bin/env python3
"""Build the source-only VLM capability analysis delivery."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from source_task_taxonomy import (
    SIMPLE_TASK_LABELS,
    SIMPLE_TASK_ORDER,
    TASK_FAMILY_COLORS,
    TASK_FAMILY_ORDER,
    classify_task,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUB = ROOT / "sub-dataset"
ARTIFACTS = ROOT / "artifacts"
SOURCE_ROOT = "/mnt/luojunkun/stage1/dataset"
ANALYSIS_DATE = "2026-08-06"

DATASET_ORDER = [
    "COCO",
    "VQAv2",
    "VisualGenome",
    "GQA",
    "TextVQA",
    "ChartQA",
    "AI2D",
    "LLaVA-Instruct",
    "VLM-R1",
    "Robo2VLM",
    "RoboVQA",
    "SpatialVLM",
    "PixMo-Cap",
    "PixMo-Points",
    "Molmo2-VideoCapQA",
    "Molmo2-VideoPoint",
    "Molmo2-VideoSubtitleQA",
    "Molmo2-VideoTrack",
]

CAPABILITY_COLUMNS = [
    ("general_perception", "通用感知"),
    ("caption", "描述"),
    ("vqa_logic", "VQA/逻辑"),
    ("ocr_document", "OCR/文本"),
    ("diagram_chart", "图表/图解"),
    ("grounding_pointing", "定位/指点"),
    ("spatial_2d3d", "2D/3D 空间"),
    ("counting", "计数"),
    ("temporal_action", "时序/动作"),
    ("tracking", "跟踪"),
    ("affordance_state", "可供性/状态"),
    ("success_future", "成功/未来"),
    ("planning", "规划"),
    ("navigation_memory", "导航/记忆"),
]

SOURCE_DETAILS = {
    "COCO": ("Arrow", "Arrow image bytes embedded", "source media SHA256", "Arrow shard + row"),
    "VQAv2": ("Parquet + Arrow", "joined instruction and embedded-image tables", "COCO image_id + media SHA256", "instruction shard/row + joined image row"),
    "VisualGenome": ("ZIP + JSON + JPEG", "ZIP archive members", "VisualGenome image_id / COCO id / Flickr id / SHA256", "archive + member"),
    "GQA": ("Parquet + Arrow", "joined instruction and image tables", "GQA/VisualGenome image_id", "instruction shard/row + joined image row"),
    "TextVQA": ("Parquet", "joined instruction and embedded-image tables", "TextVQA image_id", "instruction shard/row + joined image row"),
    "ChartQA": ("Parquet", "embedded image bytes", "image SHA256", "shard + row"),
    "AI2D": ("JSON + image files", "local source image files", "image filename + question_id + SHA256", "relative path + question JSON"),
    "LLaVA-Instruct": ("JSON + source image trees", "local source image path resolved by prefix", "upstream family + image path + SHA256", "JSON source row + resolved image path"),
    "VLM-R1": ("JSON + ZIP/JPEG", "COCO archive member", "COCO image_id + SHA256", "source JSON row + archive/member"),
    "Robo2VLM": ("Parquet", "embedded robot-scene image bytes", "scene lineage (_qN removed) + SHA256", "shard + row"),
    "RoboVQA": ("JSON + TAR/MP4", "video archive members", "video_id + uid + source-video SHA256", "JSON row + TAR archive/member"),
    "SpatialVLM": ("Parquet + remote image URL", "remote media snapshot archived for sample", "source URL/media identity + SHA256", "shard + row + source URL"),
    "PixMo-Cap": ("Parquet + remote image URL", "remote URL; no complete local media snapshot", "URL identity + source SHA256 when available", "shard + row + source URL"),
    "PixMo-Points": ("Parquet + remote image URL", "remote URL; sampled media downloaded read-only", "image_sha256 + URL identity", "shard + row + source URL"),
    "Molmo2-VideoCapQA": ("Parquet", "video ID/URL mapping; source package has no video bytes", "video_id", "shard + row"),
    "Molmo2-VideoPoint": ("Parquet + TAR/MP4", "generated videos packaged; other source families external", "video_id + source-video SHA256", "shard/row + archive/member"),
    "Molmo2-VideoSubtitleQA": ("Parquet", "video ID/URL mapping; source package has no video bytes", "video_id", "shard + row"),
    "Molmo2-VideoTrack": ("Parquet + partial video archives", "media availability differs across 12 upstream sources", "upstream source + clip/video lineage", "shard + row + upstream clip id"),
}

SOURCE_SPLITS = {
    "COCO": "train, test",
    "VQAv2": "train, validation, test, testdev",
    "VisualGenome": "no physical split field observed",
    "GQA": "train, val, test, testdev, challenge, submission; balanced is a duplicate view",
    "TextVQA": "train, validation, test",
    "ChartQA": "train, val, test",
    "AI2D": "no physical split field observed",
    "LLaVA-Instruct": "no physical split field observed; upstream family is retained",
    "VLM-R1": "no physical split field observed",
    "Robo2VLM": "train, test",
    "RoboVQA": "train only",
    "SpatialVLM": "train, test",
    "PixMo-Cap": "train",
    "PixMo-Points": "train",
    "Molmo2-VideoCapQA": "no physical split field observed",
    "Molmo2-VideoPoint": "train canonical table; category paths are duplicate views",
    "Molmo2-VideoSubtitleQA": "no physical split field observed",
    "Molmo2-VideoTrack": "no physical split field observed; upstream source retained",
}

QUALITY_RISK_SCORES = {
    "COCO": (0, 1, 1, 1, 2),
    "VQAv2": (0, 0, 1, 1, 2),
    "VisualGenome": (1, 2, 2, 2, 2),
    "GQA": (0, 1, 3, 2, 2),
    "TextVQA": (0, 0, 1, 2, 2),
    "ChartQA": (0, 0, 1, 2, 2),
    "AI2D": (0, 1, 0, 1, 3),
    "LLaVA-Instruct": (1, 1, 3, 3, 3),
    "VLM-R1": (0, 2, 3, 3, 2),
    "Robo2VLM": (0, 1, 3, 3, 2),
    "RoboVQA": (0, 1, 2, 3, 3),
    "SpatialVLM": (0, 1, 1, 3, 3),
    "PixMo-Cap": (2, 1, 1, 2, 3),
    "PixMo-Points": (2, 2, 3, 2, 3),
    "Molmo2-VideoCapQA": (3, 1, 2, 2, 3),
    "Molmo2-VideoPoint": (2, 2, 2, 2, 3),
    "Molmo2-VideoSubtitleQA": (3, 2, 2, 2, 3),
    "Molmo2-VideoTrack": (3, 2, 2, 2, 3),
}

CAPABILITY_LEVELS = [
    ("E0", "通用感知与语言基础", ("general_perception", "caption", "vqa_logic", "ocr_document", "diagram_chart", "counting")),
    ("E1", "定位、指点与空间关系", ("grounding_pointing", "spatial_2d3d", "tracking")),
    ("E2", "动作、时序与状态变化", ("temporal_action",)),
    ("E3", "可供性、状态与行动结果", ("affordance_state",)),
    ("E4", "成功判断与下一步动作", ("success_future",)),
    ("E5", "子目标与长程规划", ("planning",)),
    ("E6", "导航、记忆、恢复与安全", ("navigation_memory",)),
]

META: dict[str, dict[str, Any]] = {
    "COCO": {
        "source_dir": "COCO/COCO-MODELSCOPE",
        "modality": "image",
        "simple_task": "multi-label classification",
        "domain": "通用图像、对象类别感知",
        "input": "Arrow 内嵌单图字节",
        "output": "train 为类别 ID 列表；test 无 labels 字段",
        "judgment": "源包是对象多标签分类，而非同名数据集常见的 caption、检测或分割版本。图像字节完整，但能力窄，且 train 内有重复媒体、train/test 有 1 个 SHA 交集。",
        "rating": "B",
        "license": "图像权利随原始 COCO/Flickr 条目变化；当前源包未提供统一商用授权，待核验",
        "embodied": "none",
        "embodied_evidence": "无 agent、目标、动作或决策监督",
        "capabilities": {"general_perception": 3},
    },
    "VQAv2": {
        "source_dir": "VQAv2",
        "modality": "image+text",
        "simple_task": "open VQA",
        "domain": "通用图文问答、对象属性、计数、常识",
        "input": "COCO 图像与自然语言问题",
        "output": "train/validation 多人答案与聚合答案；test/testdev 无答案",
        "judgment": "标准通用 VQA 监督，答案覆盖对象、属性、数量和常识；同图多问且 testdev 是 test 的子视图，必须按 COCO image_id 分组并排除无答案评测行。",
        "rating": "B",
        "license": "VQA 标注与 COCO 图像条款需分别核验",
        "embodied": "none",
        "embodied_evidence": "问题不提供 agent 状态或行动决策契约",
        "capabilities": {"general_perception": 2, "vqa_logic": 3, "spatial_2d3d": 1, "counting": 2},
    },
    "VisualGenome": {
        "source_dir": "VisualGenome",
        "modality": "image+text+box",
        "simple_task": "VQA / description / grounding",
        "domain": "通用场景图、区域描述、关系与视觉定位",
        "input": "JPEG、问题或区域坐标",
        "output": "短答案、region phrase、x/y/width/height",
        "judgment": "区域与关系监督密集，适合作为 grounding 和场景图底座；但媒体与主表相差 172 张、5,788 个 region 越界，且 51,208 张图带 COCO lineage，评测污染风险高。",
        "rating": "C",
        "license": "Flickr 图像与 Visual Genome 标注条款混合，需逐项核验",
        "embodied": "prerequisite",
        "embodied_evidence": "提供实体与空间关系，不含 agent 决策",
        "capabilities": {"general_perception": 3, "caption": 2, "vqa_logic": 2, "grounding_pointing": 3, "spatial_2d3d": 2, "counting": 1},
    },
    "GQA": {
        "source_dir": "GQA",
        "modality": "image+text",
        "simple_task": "compositional VQA",
        "domain": "场景图多步关系、逻辑组合、属性与计数",
        "input": "Visual Genome 图像与组合问题",
        "output": "短答案及 structural/semantic 类型",
        "judgment": "逻辑与关系覆盖强，但 balanced 是 all 的重复子视图，多个 evaluation 视图共享媒体，且 629 万余行没有答案。训练口径必须限定 canonical all 中有答案的 train/val。",
        "rating": "C",
        "license": "源 README/上游 Visual Genome 条款，待统一核验",
        "embodied": "prerequisite",
        "embodied_evidence": "关系推理可迁移，但无 agent、目标和行动状态",
        "capabilities": {"general_perception": 2, "vqa_logic": 3, "spatial_2d3d": 3, "counting": 2},
    },
    "TextVQA": {
        "source_dir": "textvqa",
        "modality": "image+text+OCR tokens",
        "simple_task": "OCR VQA",
        "domain": "场景文字识别与文字问答",
        "input": "图像、问题与 OCR tokens",
        "output": "10 个答案标注",
        "judgment": "场景文字监督直接、schema 稳定；同图多问，且 LLaVA 完整复用了其 train 媒体，训练与 benchmark 切分必须按 image_id 统一保护。",
        "rating": "B",
        "license": "当前源包未给出统一授权文本，需核验 TextVQA 与图像来源条款",
        "embodied": "none",
        "embodied_evidence": "无行动决策监督",
        "capabilities": {"general_perception": 1, "vqa_logic": 2, "ocr_document": 3},
    },
    "ChartQA": {
        "source_dir": "Chartqa",
        "modality": "image+text",
        "simple_task": "chart VQA",
        "domain": "图表读取、数值比较与数据可视化推理",
        "input": "图表图像与问题",
        "output": "文本或数值答案",
        "judgment": "图表问答标签完整，但全量 SHA 审计发现 train/val/test 间 16 个交集事件；应按图像 SHA 重建隔离组后再用于训练。",
        "rating": "C",
        "license": "当前源包授权信息不足，待核验",
        "embodied": "none",
        "embodied_evidence": "无具身场景或 agent 决策",
        "capabilities": {"vqa_logic": 3, "diagram_chart": 3, "counting": 2},
    },
    "AI2D": {
        "source_dir": "ai2d/ai2d",
        "modality": "image+text+diagram annotation",
        "simple_task": "multiple-choice diagram QA",
        "domain": "科学图解、教育图表、图形关系",
        "input": "科学示意图与四选一问题",
        "output": "correctAnswer 下标；另有图形元素与关系标注",
        "judgment": "科学图解结构丰富且答案索引有效；但 843/4,903 张图没有任何 QA，另有 2 个空选项，源 license 明确限制用途，需先做授权确认和监督分流。",
        "rating": "B",
        "license": "源 license.txt 含非商业/再分发限制，商用前必须法务核验",
        "embodied": "none",
        "embodied_evidence": "科学图解不等同于具身决策",
        "capabilities": {"vqa_logic": 2, "diagram_chart": 3, "spatial_2d3d": 1},
    },
    "LLaVA-Instruct": {
        "source_dir": "llava-instruct",
        "modality": "image+text / text-only",
        "simple_task": "multimodal instruction / description / VQA",
        "domain": "通用图文指令混合",
        "input": "可选图像、当前用户消息与历史对话",
        "output": "assistant 自由文本",
        "judgment": "覆盖面广但不是独立媒体源：GQA、TextVQA、Visual Genome 子集为 100% 来源复用，COCO 子集也与 VQAv2 高重叠；40,688 条纯文本和大量重复 id 需单独分流。",
        "rating": "C",
        "license": "多来源混合许可，必须按上游来源分别核验",
        "embodied": "none",
        "embodied_evidence": "通用 instruction 不自动构成具身监督",
        "capabilities": {"general_perception": 3, "caption": 3, "vqa_logic": 3, "ocr_document": 2, "diagram_chart": 1, "grounding_pointing": 1, "spatial_2d3d": 2, "counting": 1},
    },
    "VLM-R1": {
        "source_dir": "vlm-r1",
        "modality": "image+text+box",
        "simple_task": "grounding",
        "domain": "通用图像指代表达定位",
        "input": "COCO train2014 图像与 referring expression",
        "output": "JSON fence 中的 bbox_2d 与 label",
        "judgment": "grounding 标签集中且坐标结构统一，但 321,308 条仅覆盖 28,157 张 COCO 图，全部 image_id 也出现在 VQAv2；80 条 label 双引号未转义导致非法 JSON。",
        "rating": "C",
        "license": "COCO 图像条款与派生标注授权需分别核验",
        "embodied": "prerequisite",
        "embodied_evidence": "提供实体定位，不提供 agent 行动决策",
        "capabilities": {"general_perception": 1, "grounding_pointing": 3, "spatial_2d3d": 2},
    },
    "Robo2VLM": {
        "source_dir": "robo2vlm",
        "modality": "robot image+text",
        "simple_task": "multiple-choice embodied QA",
        "domain": "机器人空间、轨迹、目标状态、成功与下一步运动",
        "input": "机器人场景图与多选问题",
        "output": "选项下标",
        "judgment": "当前最完整的图像级具身监督，覆盖 3D 对应、深度、轨迹、成功和目标状态；但 684,710 行只有 19,331 个唯一问题文本，且 5,239 个 test scene 全部出现在 train。",
        "rating": "C",
        "license": "当前源包未确认统一授权，待核验",
        "embodied": "direct",
        "embodied_evidence": "问题显式包含 robot、任务目标、运动方向、障碍或成功状态",
        "capabilities": {"general_perception": 2, "vqa_logic": 3, "grounding_pointing": 1, "spatial_2d3d": 3, "temporal_action": 2, "tracking": 2, "affordance_state": 2, "success_future": 3, "planning": 1},
    },
    "RoboVQA": {
        "source_dir": "robovqa",
        "modality": "robot video+text",
        "simple_task": "description / embodied VQA / planning",
        "domain": "机器人操作、可供性、成功判断、未来预测和子任务规划",
        "input": "短机器人视频、目标、历史步骤与问题",
        "output": "描述、yes/no、下一动作或未来 5 个子任务；reasoning 含 think/answer 标签",
        "judgment": "具身任务语义最强，218,504 个视频覆盖从描述到五步规划；但 reasoning 有 143,382 条重复任务签名，全部为 train，且视频级 task_metadata 被多文件重复携带。",
        "rating": "C",
        "license": "内部/混合源授权未在当前源包确认，待核验",
        "embodied": "direct",
        "embodied_evidence": "显式 agent、目标、状态、动作可行性、成功和规划监督",
        "capabilities": {"general_perception": 2, "caption": 2, "vqa_logic": 2, "spatial_2d3d": 1, "temporal_action": 3, "affordance_state": 3, "success_future": 3, "planning": 3},
    },
    "SpatialVLM": {
        "source_dir": "spatialvlm",
        "modality": "image+multi-turn text",
        "simple_task": "spatial VQA / grounding",
        "domain": "方向、距离、深度、尺寸与相对尺度",
        "input": "单图与多轮空间问题",
        "output": "短答案、解释或归一化 bbox",
        "judgment": "空间先验覆盖直接，但不是 agent 决策数据；全量 SHA 审计发现 130 个 test 图也在 train，且源中多问复用同图，官方 split 不可直接沿用。",
        "rating": "C",
        "license": "混合来源，当前源包未给出统一授权，待核验",
        "embodied": "prerequisite",
        "embodied_evidence": "空间估计是具身前置能力，但缺少行动与目标",
        "capabilities": {"general_perception": 2, "vqa_logic": 3, "grounding_pointing": 1, "spatial_2d3d": 3, "counting": 1},
    },
    "PixMo-Cap": {
        "source_dir": "pixmo-cap",
        "modality": "remote image URL+text",
        "simple_task": "long description",
        "domain": "开放域高密度长图像描述",
        "input": "远程图片 URL",
        "output": "长 caption 与人工 transcript",
        "judgment": "长描述规模大、语言丰富；媒体不在源包内，确定性候选中为得到 200 张有效图经历 303 次失败，训练前必须冻结媒体快照和授权清单。",
        "rating": "C",
        "license": "开放网页图片来源混合，URL 可访问不代表可训练或可商用",
        "embodied": "none",
        "embodied_evidence": "描述不含结构化 agent 决策",
        "capabilities": {"general_perception": 3, "caption": 3, "ocr_document": 1},
    },
    "PixMo-Points": {
        "source_dir": "pixmo-points",
        "modality": "remote image URL+point",
        "simple_task": "pointing / counting",
        "domain": "开放域对象指点与数量监督",
        "input": "远程图片、label 与任务类型",
        "output": "点坐标或 count",
        "judgment": "指点和计数覆盖大，但 237 万行只对应 22.8 万媒体；800 个候选中 132 个媒体失败，含 59 个源 SHA 不匹配，点总量只能做样本估计。",
        "rating": "C",
        "license": "开放网页图片来源混合，需逐 URL 来源核验",
        "embodied": "prerequisite",
        "embodied_evidence": "对象指点可支撑动作绑定，但没有 agent 状态或动作序列",
        "capabilities": {"general_perception": 2, "grounding_pointing": 3, "spatial_2d3d": 2, "counting": 3},
    },
    "Molmo2-VideoCapQA": {
        "source_dir": "Molmo2-VideoCapQA",
        "modality": "video ID+text (media unavailable)",
        "simple_task": "video multiple-choice QA",
        "domain": "视频对象、动作、时序、空间、因果与计数",
        "input": "video_id、问题与候选答案",
        "output": "答案及负选项",
        "judgment": "标注覆盖丰富且有 120 万 QA，但源包没有视频字节，无法验证视觉依赖、解码质量或媒体授权；在媒体 join 完成前不能进入训练池。",
        "rating": "Q",
        "license": "媒体与标注授权均待逐来源核验",
        "embodied": "prerequisite",
        "embodied_evidence": "有动作和时序，但没有稳定 agent 决策契约",
        "capabilities": {"general_perception": 2, "vqa_logic": 3, "spatial_2d3d": 2, "counting": 2, "temporal_action": 3},
    },
    "Molmo2-VideoPoint": {
        "source_dir": "Molmo2-VideoPoint",
        "modality": "video+temporal point",
        "simple_task": "video pointing",
        "domain": "视频对象/动作时空指点与指代表达",
        "input": "视频、问题、时间戳与 label",
        "output": "跨帧 points/count",
        "judgment": "时空指点监督稠密，但分类 Parquet 是 canonical 主表的重复视图；96,319 条 annotator_unsure，且本地只打包了 15,204 个 generated video，其他媒体仍依赖外部来源。",
        "rating": "C",
        "license": "generated、YouTube、MammalNet 多来源混合，需分别核验",
        "embodied": "prerequisite",
        "embodied_evidence": "提供动作对象时空绑定，不提供 agent 决策",
        "capabilities": {"general_perception": 2, "grounding_pointing": 3, "spatial_2d3d": 2, "temporal_action": 3, "tracking": 2},
    },
    "Molmo2-VideoSubtitleQA": {
        "source_dir": "Molmo2-VideoSubtitleQA",
        "modality": "video ID+subtitle+text (media unavailable)",
        "simple_task": "subtitle-grounded video QA",
        "domain": "字幕-视觉对齐、事件顺序、因果与解释定位",
        "input": "video_id、字幕、问题与负选项",
        "output": "文本答案",
        "judgment": "字幕与视觉推理标签规模可观，但源包没有视频字节且 alignment type 存在大小写、多语言和近义类别漂移；媒体恢复与标签归一前应隔离。",
        "rating": "Q",
        "license": "视频与字幕多来源授权待核验",
        "embodied": "prerequisite",
        "embodied_evidence": "时序和因果可迁移，但不是机器人行动监督",
        "capabilities": {"general_perception": 1, "vqa_logic": 3, "ocr_document": 2, "spatial_2d3d": 1, "temporal_action": 3},
    },
    "Molmo2-VideoTrack": {
        "source_dir": "Molmo2-VideoTrack",
        "modality": "video lineage+track point (partially unavailable)",
        "simple_task": "tracking",
        "domain": "跨帧对象持续性、指代表达与轨迹",
        "input": "video/clip、对象表达与帧范围",
        "output": "跨帧 point/segment 轨迹",
        "judgment": "跟踪 supervision 清晰，但 12 个上游视频源的打包完整性不一致，当前采样无法完成统一媒体 join；6 条表达为空，训练前必须恢复媒体和按原视频 lineage 分组。",
        "rating": "Q",
        "license": "多跟踪数据集混合许可，必须逐上游核验",
        "embodied": "prerequisite",
        "embodied_evidence": "对象持续性是具身前置能力，不含动作决策",
        "capabilities": {"grounding_pointing": 2, "temporal_action": 3, "tracking": 3},
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def compact(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "<max-depth>"
    if isinstance(value, dict):
        return {str(key): compact(item, depth + 1) for key, item in list(value.items())[:40]}
    if isinstance(value, list):
        rows = [compact(item, depth + 1) for item in value[:16]]
        if len(value) > 16:
            rows.append({"omitted_items": len(value) - 16})
        return rows
    if isinstance(value, str):
        value = re.sub(r"/data/[^/]+/dataset/", "/data/<redacted>/dataset/", value)
        return value if len(value) <= 1200 else value[:1200] + "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def fmt(value: Any, digits: int = 1) -> str:
    if value is None or value == "":
        return "未验证"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def pct(numerator: int | float, denominator: int | float) -> str:
    return f"{numerator / denominator * 100:.2f}%" if denominator else "n/a"


def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def short(value: Any, length: int = 260) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value if len(value) <= length else value[:length] + "..."


def artifact_data() -> dict[str, Any]:
    legacy = load_json(ARTIFACTS / "legacy-tabular-full-statistics.json")
    return {
        "legacy": {row["dataset"]: row for row in legacy["datasets"]},
        "ai2d": load_json(ARTIFACTS / "source-json-statistics-ai2d.json"),
        "visual": load_json(ARTIFACTS / "source-json-statistics-visualgenome.json"),
        "vlm": load_json(ARTIFACTS / "source-json-statistics-vlm_r1.json"),
        "llava": load_json(ARTIFACTS / "source-json-statistics-llava_instruct.json"),
        "robovqa": load_json(ARTIFACTS / "source-json-statistics-robovqa.json"),
        "overlap": load_json(ARTIFACTS / "source-lineage-overlap-statistics.json"),
        "split": load_json(ARTIFACTS / "source-split-media-statistics.json"),
        "robo2caps": load_json(ARTIFACTS / "source-robo2vlm-capabilities.json"),
    }


def full_stats(dataset: str) -> dict[str, Any]:
    path = SUB / dataset / "full-statistics.json"
    return load_json(path) if path.exists() else {}


def summary(dataset: str) -> dict[str, Any]:
    return load_json(SUB / dataset / "sampling-summary.json")


def build_metrics(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    legacy = data["legacy"]
    ai = data["ai2d"]
    vg = data["visual"]
    llava = data["llava"]
    vlm = data["vlm"]
    rvqa = data["robovqa"]
    split = data["split"]["details"]
    robo2 = legacy["Robo2VLM"]
    metrics: dict[str, dict[str, Any]] = {
        "COCO": {
            "source_records": 122218,
            "unique_media": 122204,
            "unique_scene": None,
            "supervision_units": split["COCO"]["label_annotation_entries"],
            "effective_units": 117205,
            "token_estimate": None,
            "evidence": "full Arrow scan",
        },
        "VQAv2": {
            "source_records": legacy["VQAv2"]["canonical_records"],
            "unique_media": legacy["VQAv2"]["unique_media_ids"],
            "unique_scene": None,
            "supervision_units": legacy["VQAv2"]["canonical_records"],
            "effective_units": legacy["VQAv2"]["canonical_records"] - legacy["VQAv2"]["answer_type_counts"]["<missing>"],
            "token_estimate": legacy["VQAv2"]["supervised_token_estimate_chars_div_4"],
            "evidence": "full Parquet projection",
        },
        "VisualGenome": {
            "source_records": vg["canonical_image_records"],
            "unique_media": 108249,
            "unique_scene": None,
            "supervision_units": vg["canonical_qa_records"] + vg["canonical_region_records"],
            "effective_units": vg["canonical_qa_records"] + vg["canonical_region_records"] - vg["out_of_bounds_regions_against_metadata"] - vg["empty_region_phrases"],
            "token_estimate": vg["qa_token_estimate_chars_div_4"] + vg["region_token_estimate_chars_div_4"],
            "evidence": "full ZIP-member JSON scan",
        },
        "GQA": {
            "source_records": legacy["GQA"]["canonical_records"],
            "unique_media": legacy["GQA"]["unique_media_ids_image_table_union"],
            "unique_scene": None,
            "supervision_units": legacy["GQA"]["answered_records"],
            "effective_units": legacy["GQA"]["records_by_split"]["train"] + legacy["GQA"]["records_by_split"]["val"],
            "token_estimate": legacy["GQA"]["supervised_token_estimate_chars_div_4"],
            "evidence": "full canonical Parquet projection",
        },
        "TextVQA": {
            "source_records": legacy["TextVQA"]["canonical_records"],
            "unique_media": legacy["TextVQA"]["unique_media_ids"],
            "unique_scene": None,
            "supervision_units": legacy["TextVQA"]["supervision_units"],
            "effective_units": legacy["TextVQA"]["supervision_units"],
            "token_estimate": legacy["TextVQA"]["supervised_token_estimate_chars_div_4"],
            "evidence": "full Parquet projection",
        },
        "ChartQA": {
            "source_records": legacy["ChartQA"]["canonical_records"],
            "unique_media": legacy["ChartQA"]["unique_media_sha256"],
            "unique_scene": None,
            "supervision_units": legacy["ChartQA"]["supervision_units"],
            "effective_units": None,
            "token_estimate": legacy["ChartQA"]["supervised_token_estimate_chars_div_4"],
            "evidence": "full Parquet and SHA scan",
        },
        "AI2D": {
            "source_records": ai["canonical_image_records"],
            "unique_media": ai["canonical_image_records"],
            "unique_scene": None,
            "supervision_units": ai["canonical_question_records"],
            "effective_units": ai["canonical_question_records"],
            "token_estimate": ai["supervised_token_estimate_chars_div_4"],
            "evidence": "full source JSON-file scan",
        },
        "LLaVA-Instruct": {
            "source_records": llava["canonical_records"],
            "unique_media": llava["unique_media_paths"],
            "unique_scene": None,
            "supervision_units": llava["canonical_records"],
            "effective_units": llava["canonical_records"] - llava["text_only_records"],
            "token_estimate": llava["supervised_token_estimate_assistant_chars_div_4"],
            "evidence": "full source JSON streaming scan",
        },
        "VLM-R1": {
            "source_records": vlm["canonical_records"],
            "unique_media": vlm["unique_media_filenames"],
            "unique_scene": None,
            "supervision_units": vlm["canonical_records"],
            "effective_units": vlm["canonical_records"] - vlm["assistant_parse_failures"],
            "token_estimate": vlm["supervised_token_estimate_assistant_chars_div_4"],
            "evidence": "full source JSON streaming scan",
        },
        "Robo2VLM": {
            "source_records": robo2["canonical_records"],
            "unique_media": robo2["unique_scene_lineages_suffix_q_removed"],
            "unique_scene": robo2["unique_scene_lineages_suffix_q_removed"],
            "supervision_units": robo2["supervision_units"],
            "effective_units": None,
            "token_estimate": robo2["supervised_token_estimate_chars_div_4"],
            "evidence": "full canonical Parquet projection and rule classification",
        },
        "RoboVQA": {
            "source_records": rvqa["canonical_source_records"],
            "unique_media": rvqa["unique_video_ids_in_json"],
            "unique_scene": rvqa["unique_uids"],
            "supervision_units": rvqa["canonical_source_records"],
            "effective_units": rvqa["unique_matched_conversation_task_signatures"] + rvqa["conversation_kind_counts"]["understanding_description"],
            "token_estimate": round(rvqa["conversation_assistant_characters"] / 4),
            "evidence": "full six-file source JSON streaming scan",
        },
    }
    for dataset in DATASET_ORDER[11:]:
        stats = full_stats(dataset)
        if dataset == "SpatialVLM":
            metrics[dataset] = {
                "source_records": stats["source_records"], "unique_media": stats["unique_media_identities"], "unique_scene": None,
                "supervision_units": stats["qa_pairs"], "effective_units": None,
                "token_estimate": stats["supervised_token_estimate"], "evidence": "full source Parquet scan",
            }
        elif dataset == "PixMo-Cap":
            metrics[dataset] = {
                "source_records": stats["source_records"], "unique_media": stats["unique_media_identities"], "unique_scene": None,
                "supervision_units": stats["source_records"], "effective_units": None,
                "token_estimate": stats["supervised_token_estimate"], "evidence": "full metadata; media availability sampled",
            }
        elif dataset == "PixMo-Points":
            metrics[dataset] = {
                "source_records": stats["source_records"], "unique_media": stats["unique_media_identities"], "unique_scene": None,
                "supervision_units": stats["source_records"], "effective_units": None,
                "token_estimate": stats["supervised_token_estimate"], "evidence": "full metadata; points/media quality sampled",
            }
        elif dataset == "Molmo2-VideoCapQA":
            metrics[dataset] = {
                "source_records": stats["canonical_source_records"], "unique_media": stats["unique_video_ids"], "unique_scene": None,
                "supervision_units": stats["supervision_units"], "effective_units": None,
                "token_estimate": stats["supervised_token_estimate_chars_div_4"], "evidence": "full annotation metadata; media unavailable",
            }
        elif dataset == "Molmo2-VideoPoint":
            metrics[dataset] = {
                "source_records": stats["canonical_source_records"], "unique_media": stats["unique_video_ids"], "unique_scene": None,
                "supervision_units": stats["supervision_units"], "effective_units": None,
                "token_estimate": stats["supervised_token_estimate_chars_div_4"], "evidence": "full canonical metadata; partial local media",
            }
        elif dataset == "Molmo2-VideoSubtitleQA":
            metrics[dataset] = {
                "source_records": stats["canonical_source_records"], "unique_media": stats["unique_video_ids"], "unique_scene": None,
                "supervision_units": stats["supervision_units"], "effective_units": None,
                "token_estimate": stats["supervised_token_estimate_chars_div_4"], "evidence": "full annotation metadata; media unavailable",
            }
        elif dataset == "Molmo2-VideoTrack":
            metrics[dataset] = {
                "source_records": stats["canonical_source_records"], "unique_media": stats["unique_video_ids"], "unique_scene": stats["unique_clip_ids"],
                "supervision_units": stats["supervision_units"], "effective_units": None,
                "token_estimate": stats["supervised_token_estimate_chars_div_4"], "evidence": "full annotation metadata; media join incomplete",
            }
    for dataset, values in metrics.items():
        values["annotations_per_media"] = (
            values["supervision_units"] / values["unique_media"] if values.get("unique_media") else None
        )
        values["tokens_per_unit"] = (
            values["token_estimate"] / values["supervision_units"]
            if values.get("token_estimate") and values.get("supervision_units") else None
        )
    effective_types = {
        "COCO": "unique train media SHA (test schema has no labels)",
        "VQAv2": "answer-bearing QA rows",
        "VisualGenome": "QA + structurally valid in-bounds region rows",
        "GQA": "canonical answer-bearing train+val rows",
        "TextVQA": "QA rows",
        "ChartQA": "rows pending media and annotation quality review",
        "AI2D": "question rows (diagram-only images excluded from this count)",
        "LLaVA-Instruct": "visual conversation rows pending source quality review",
        "VLM-R1": "parseable grounding rows pending media and annotation review",
        "Robo2VLM": "rows pending scene-level quality review",
        "RoboVQA": "unique matched reasoning task signatures + one understanding record per video",
        "SpatialVLM": "rows pending media and annotation quality review",
        "PixMo-Cap": "pending complete media snapshot",
        "PixMo-Points": "pending complete media snapshot and SHA audit",
        "Molmo2-VideoCapQA": "pending media join",
        "Molmo2-VideoPoint": "pending per-source media join and unsure filtering",
        "Molmo2-VideoSubtitleQA": "pending media join",
        "Molmo2-VideoTrack": "pending media join",
    }
    for dataset, values in metrics.items():
        values["effective_unit_type"] = effective_types[dataset]
    return metrics


def task_row(dataset: str, task: str, count: int, denominator: int, basis: str, capability: str, embodied: str, unit: str = "records") -> dict[str, Any]:
    return {
        "dataset": dataset, "task": task, "count": count, "count_unit": unit,
        "denominator": denominator, "percentage": count / denominator if denominator else 0,
        "classification_basis": basis, "fine_capability": capability, "embodied_level": embodied,
    }


def build_task_rows(data: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    legacy = data["legacy"]
    coco_labels = data["split"]["details"]["COCO"]["label_annotation_entries"]
    rows.append(task_row("COCO", "object multi-label classification", coco_labels, coco_labels, "physical labels field", "对象类别识别", "none", "label annotations"))
    for key, value in legacy["VQAv2"]["answer_type_counts"].items():
        label = {"other": "open answer", "yes/no": "binary VQA", "number": "counting/numeric", "<missing>": "evaluation row without answer"}.get(key, key)
        rows.append(task_row("VQAv2", label, value, metrics["VQAv2"]["source_records"], "source answer_type", "通用 VQA", "none"))
    vg = data["visual"]
    rows.extend([
        task_row("VisualGenome", "open VQA", vg["canonical_qa_records"], vg["canonical_qa_records"] + vg["canonical_region_records"], "physical QA rows", "场景图问答", "prerequisite"),
        task_row("VisualGenome", "region description / grounding", vg["canonical_region_records"], vg["canonical_qa_records"] + vg["canonical_region_records"], "physical region rows", "区域定位与描述", "prerequisite"),
    ])
    gqa = legacy["GQA"]
    for key, value in gqa["structural_type_counts"].items():
        rows.append(task_row("GQA", key, value, gqa["answered_records"], "source types.structural", "关系与组合逻辑", "prerequisite"))
    rows.append(task_row("TextVQA", "OCR / scene text VQA", metrics["TextVQA"]["supervision_units"], metrics["TextVQA"]["supervision_units"], "dataset physical schema", "场景文字理解", "none"))
    rows.append(task_row("ChartQA", "chart VQA", metrics["ChartQA"]["supervision_units"], metrics["ChartQA"]["supervision_units"], "dataset physical schema", "图表数值推理", "none"))
    rows.append(task_row("AI2D", "diagram multiple-choice QA", metrics["AI2D"]["supervision_units"], metrics["AI2D"]["supervision_units"], "questions JSON", "科学图解推理", "none"))
    llava_map = {"coco": "COCO-derived instruction", "gqa": "GQA-derived instruction", "ocr_vqa": "OCR-VQA-derived instruction", "textvqa": "TextVQA-derived instruction", "vg": "VisualGenome-derived instruction", "<text-only>": "text-only instruction"}
    for key, value in data["llava"]["records_by_media_prefix"].items():
        rows.append(task_row("LLaVA-Instruct", llava_map.get(key, key), value, data["llava"]["canonical_records"], "source image prefix", "通用多模态指令", "none"))
    rows.append(task_row("VLM-R1", "referring-expression grounding", metrics["VLM-R1"]["source_records"], metrics["VLM-R1"]["source_records"], "assistant bbox schema", "指代表达定位", "prerequisite"))
    robo2 = data["robo2caps"]
    for key, value in robo2["primary_capability_counts"].items():
        rows.append(task_row("Robo2VLM", key, value, robo2["canonical_records"], "full deterministic source-question rules", key, "direct"))
    rvqa = data["robovqa"]
    for key, value in rvqa["conversation_task_counts"].items():
        rows.append(task_row("RoboVQA", key, value, rvqa["canonical_source_records"], "prompt to source task_metadata exact prefix match", key, "direct"))
    spatial = full_stats("SpatialVLM")
    for key, value in spatial["capability_counts"].items():
        rows.append(task_row("SpatialVLM", key, value, spatial["qa_pairs"], "deterministic multi-label source-text rules", key, "prerequisite"))
    for dataset in ["PixMo-Cap", "PixMo-Points", "Molmo2-VideoCapQA", "Molmo2-VideoPoint", "Molmo2-VideoSubtitleQA", "Molmo2-VideoTrack"]:
        stats = full_stats(dataset)
        denominator = metrics[dataset]["supervision_units"]
        for key, value in stats.get("task_counts", {}).items():
            rows.append(task_row(dataset, key, value, denominator, "physical field / source schema", key, META[dataset]["embodied"]))
    return rows


def simple_task_group(dataset: str, task: str) -> str:
    value = task.lower()
    if "without answer" in value or "text-only" in value:
        return "holdout_or_text_only"
    if dataset == "COCO" or "classification" in value:
        return "classification"
    if dataset == "Molmo2-VideoTrack" or "track" in value:
        return "tracking"
    if dataset == "Molmo2-VideoPoint" or "point" in value:
        return "pointing"
    if dataset == "VLM-R1" or "ground" in value or "region" in value:
        return "grounding"
    if dataset == "PixMo-Cap" or "caption" in value or "description" in value or "transcript" in value:
        return "description"
    if dataset in {"LLaVA-Instruct"}:
        return "multimodal_instruction"
    if dataset in {"Robo2VLM", "RoboVQA"}:
        if any(token in value for token in ("plan", "subtask", "five step", "5 step")):
            return "planning"
        return "embodied_reasoning"
    if dataset in {"Molmo2-VideoCapQA", "Molmo2-VideoSubtitleQA"}:
        return "video_qa"
    if dataset in {"TextVQA", "ChartQA", "AI2D"} or any(token in value for token in ("ocr", "chart", "diagram")):
        return "specialized_vqa"
    if "count" in value or "numeric" in value:
        return "counting"
    return "open_or_compositional_vqa"


def task_to_embodied_level(dataset: str, task: str, group: str) -> str:
    value = task.lower()
    if any(token in value for token in ("navigation", "memory", "recovery", "safety")):
        return "E6"
    if group == "planning" or any(token in value for token in ("long horizon", "subgoal", "subtask")):
        return "E5"
    if any(token in value for token in ("next action", "future", "forecast")):
        return "E4"
    if any(token in value for token in ("afford", "success", "failure", "causal", "state")) and dataset in {"Robo2VLM", "RoboVQA"}:
        return "E3"
    if dataset in {"Robo2VLM", "RoboVQA", "Molmo2-VideoCapQA", "Molmo2-VideoSubtitleQA", "Molmo2-VideoTrack"} and any(
        token in value for token in ("action", "temporal", "trajectory", "track", "sequence", "motion")
    ):
        return "E2"
    if group in {"grounding", "pointing", "tracking"} or any(token in value for token in ("spatial", "depth", "distance", "location")):
        return "E1"
    if dataset in {"Robo2VLM", "RoboVQA"} and group == "embodied_reasoning":
        return "E3"
    return "E0"


def enrich_task_rows(rows: list[dict[str, Any]], metrics: dict[str, dict[str, Any]]) -> None:
    totals = Counter()
    for row in rows:
        row["task_group"] = simple_task_group(row["dataset"], row["task"])
        row["embodied_capability_level"] = task_to_embodied_level(row["dataset"], row["task"], row["task_group"])
        taxonomy = classify_task(row["dataset"], row["task"])
        row["source_classification_basis"] = row["classification_basis"]
        row.update({
            "taxonomy_version": taxonomy["taxonomy_version"],
            "simple_task_category": taxonomy["simple_task_category"],
            "simple_task_label": taxonomy["simple_task_label"],
            "task_family": taxonomy["task_family"],
            "fine_task": taxonomy["fine_task"],
            "task_description": taxonomy["task_description"],
            "input": taxonomy["input"],
            "output": taxonomy["output"],
            "domain": taxonomy["domain"],
            "taxonomy_classification_basis": taxonomy["classification_basis"],
        })
        totals[row["dataset"]] += int(row["count"])
    for row in rows:
        dataset = row["dataset"]
        token_total = metrics[dataset].get("token_estimate")
        row["estimated_supervised_tokens"] = (
            round(token_total * row["count"] / totals[dataset]) if token_total and totals[dataset] else None
        )
        row["token_allocation_method"] = "dataset source-token estimate allocated in proportion to task counts; multi-label rows normalized within dataset"
        row["source_path"] = f"{SOURCE_ROOT}/{META[dataset]['source_dir']}"
        row["evidence_date"] = ANALYSIS_DATE


def source_accounting_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rules: dict[str, tuple[int, int, str]] = {
        "COCO": (117205, 5013, "unique labeled train media retained; test rows and duplicate train media excluded"),
        "VQAv2": (654959, 545794, "answer-bearing rows retained; answerless evaluation rows excluded"),
        "VisualGenome": (108077, 0, "image metadata groups retained; invalid region annotations are rejected separately"),
        "GQA": (16317209, 6463195, "answer-bearing canonical train/val retained; evaluation/no-answer rows excluded"),
        "TextVQA": (45336, 0, "all source QA rows structurally usable; cross-source lineage must still be protected"),
        "ChartQA": (0, 0, "all rows unresolved until cross-split SHA groups are reassigned"),
        "AI2D": (4060, 843, "images with one or more QA retained for VQA; diagram-only images split out"),
        "LLaVA-Instruct": (0, 40688, "text-only rows excluded; visual rows unresolved until upstream-family dedup and benchmark protection"),
        "VLM-R1": (0, 80, "invalid assistant JSON rejected; remaining rows unresolved because every media id overlaps VQAv2"),
        "Robo2VLM": (0, 0, "all canonical rows unresolved until scene-lineage split is rebuilt"),
        "RoboVQA": (995133, 143382, "duplicate reasoning task signatures removed; understanding rows retained once per video"),
        "SpatialVLM": (0, 0, "all rows unresolved until train/test SHA groups are rebuilt"),
        "PixMo-Cap": (0, 0, "complete immutable media snapshot and license lineage unavailable"),
        "PixMo-Points": (0, 0, "complete media snapshot and full SHA validation unavailable"),
        "Molmo2-VideoCapQA": (0, 0, "source package has annotations but no video bytes"),
        "Molmo2-VideoPoint": (0, 96319, "annotator_unsure rows rejected; remaining rows await per-source media closure"),
        "Molmo2-VideoSubtitleQA": (0, 0, "source package has annotations but no video bytes"),
        "Molmo2-VideoTrack": (0, 6, "empty expressions rejected; remaining rows await media join by upstream lineage"),
    }
    rows = []
    for dataset in DATASET_ORDER:
        total = int(metrics[dataset]["source_records"])
        effective, rejected, reason = rules[dataset]
        unresolved = total - effective - rejected
        if unresolved < 0:
            raise ValueError(f"negative unresolved accounting for {dataset}")
        rows.append({
            "dataset": dataset,
            "canonical_source_records": total,
            "confirmed_effective_records": effective,
            "known_rejected_records": rejected,
            "unresolved_records": unresolved,
            "accounting_check": effective + rejected + unresolved,
            "status": "resolved" if unresolved == 0 else "requires remediation",
            "decision_basis": reason,
            "source_path": f"{SOURCE_ROOT}/{META[dataset]['source_dir']}",
            "evidence_scope": metrics[dataset]["evidence"],
            "evidence_date": ANALYSIS_DATE,
        })
    return rows


def quality_risk_rows() -> list[dict[str, Any]]:
    labels = ("media_completeness_risk", "schema_annotation_risk", "duplication_concentration_risk", "split_leakage_risk", "license_commercial_risk")
    rows = []
    for dataset in DATASET_ORDER:
        scores = QUALITY_RISK_SCORES[dataset]
        row = {"dataset": dataset, **dict(zip(labels, scores))}
        row.update({
            "overall_rating": META[dataset]["rating"],
            "rubric": "0=low, 1=moderate, 2=high, 3=critical/quarantine; derived from quality-audit.csv and observed source availability",
            "source_path": f"{SOURCE_ROOT}/{META[dataset]['source_dir']}",
            "evidence_date": ANALYSIS_DATE,
        })
        rows.append(row)
    return rows


def capability_gap_rows() -> list[dict[str, Any]]:
    quality_weights = {"A": 1.0, "B": 0.8, "C": 0.5, "Q": 0.0}
    target = 9.0
    rows = []
    for level, label, keys in CAPABILITY_LEVELS:
        contributors = []
        quarantined = []
        points = 0.0
        for dataset in DATASET_ORDER:
            strength = max((META[dataset]["capabilities"].get(key, 0) for key in keys), default=0)
            if not strength:
                continue
            weight = quality_weights[META[dataset]["rating"]]
            if weight:
                points += strength * weight
                contributors.append(dataset)
            else:
                quarantined.append(dataset)
        capped = min(target, points)
        rows.append({
            "embodied_level": level,
            "capability": label,
            "quality_adjusted_source_points": round(points, 3),
            "target_points": target,
            "coverage_percent": round(capped / target * 100, 3),
            "gap_points": round(max(0.0, target - points), 3),
            "qualified_contributors": "; ".join(contributors) or "none",
            "quarantined_contributors": "; ".join(quarantined) or "none",
            "formula": "max 0-3 evidence per dataset x quality weight (A=1,B=.8,C=.5,Q=0); target=3 independent core-strength sources=9 points",
            "evidence_date": ANALYSIS_DATE,
        })
    return rows


def embodied_distribution_rows(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality_weights = {"A": 1.0, "B": 0.8, "C": 0.5, "Q": 0.0}
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"raw": 0, "adjusted": 0.0, "datasets": set(), "direct": set()})
    for row in task_rows:
        dataset = row["dataset"]
        if META[dataset]["embodied"] == "none":
            continue
        level = row["embodied_capability_level"]
        grouped[level]["raw"] += int(row["count"])
        grouped[level]["adjusted"] += int(row["count"]) * quality_weights[META[dataset]["rating"]]
        grouped[level]["datasets"].add(dataset)
        if META[dataset]["embodied"] == "direct":
            grouped[level]["direct"].add(dataset)
    rows = []
    names = {level: label for level, label, _ in CAPABILITY_LEVELS}
    for level, _, _ in CAPABILITY_LEVELS:
        values = grouped[level]
        rows.append({
            "embodied_level": level,
            "capability": names[level],
            "raw_multilabel_supervision_evidence": values["raw"],
            "quality_adjusted_evidence": round(values["adjusted"]),
            "contributing_source_count": len(values["datasets"]),
            "direct_embodied_source_count": len(values["direct"]),
            "contributing_datasets": "; ".join(sorted(values["datasets"])) or "none",
            "evidence_note": "task rows are multi-label and may exceed canonical rows; Q sources receive zero quality weight",
            "evidence_date": ANALYSIS_DATE,
        })
    return rows


def quality_rows(data: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues: dict[str, list[tuple[str, Any, Any, str, str]]] = {
        "COCO": [("train 内重复媒体行", 13, 117218, "full SHA", "按 SHA 去重"), ("train/test SHA 交集", 1, None, "full SHA", "同一 leakage group 只保留一个 split")],
        "VQAv2": [("无答案 evaluation 行", 545794, 1200753, "full answer_type", "训练池排除 test/testdev"), ("testdev 与 test 媒体交集", 36018, 36807, "full image_id", "视为 test 子视图，不独立切分")],
        "VisualGenome": [("region 越界", 5788, 5408689, "full coordinate check", "修复或拒绝"), ("媒体与 metadata 数量差", 172, 108249, "archive vs main table", "隔离无 join 媒体"), ("无效 images2.zip.1", 1, 1, "ZIP open", "永久排除")],
        "GQA": [("balanced 物理重复行", 1233702, None, "Parquet footer", "仅使用 all canonical view"), ("无答案行", 22780404 - 16489383, 22780404, "full null projection", "只保留 answer-bearing train/val")],
        "TextVQA": [("LLaVA 完整复用 train image_id", 21953, 21953, "full lineage intersection", "联合切分并保护 benchmark 媒体")],
        "ChartQA": [("跨 split 图像 SHA 交集事件", 16, None, "full SHA", "按 SHA 重分组")],
        "AI2D": [("无任何 QA 的图片", 843, 4903, "full questions scan", "作为 diagram-only 分流或排除"), ("空 answer option", 2, 62004, "full option scan", "修复/拒绝")],
        "LLaVA-Instruct": [("纯文本记录", 40688, 665298, "full JSON", "与 VLM 视觉训练分流"), ("非唯一 source id 行", 665298 - 389722, 665298, "full id set", "不得以 id 直接切分"), ("上游媒体复用", 72140 + 21953 + 86417, None, "full lineage", "按上游 family 联合去重")],
        "VLM-R1": [("非法 assistant JSON", 80, 321308, "full parse", "转义 label 双引号或拒绝"), ("VQAv2 COCO image_id 重叠", 28157, 28157, "full lineage", "benchmark 媒体保护"), ("每图平均监督密度", 321308, 28157, "full image filenames", "每图每 epoch 上限 4")],
        "Robo2VLM": [("test scene 全部出现在 train", 5239, 5239, "full lineage", "按 scene 重建 split"), ("nested duplicate rows", 126087, None, "Parquet footer", "排除 data/data"), ("唯一问题模板过少", 19331, 684710, "full question text", "模板降权与场景均衡")],
        "RoboVQA": [("reasoning 重复任务签名", 143382, 920011, "full matched signature", "任务签名去重"), ("仅 train split", 6089063, 6089063, "source task metadata", "按 video/uid 建验证集"), ("archive 视频无 JSON 记录", 219793 - 218504, 219793, "tar vs JSON IDs", "隔离无标注视频")],
        "SpatialVLM": [("test 图像 SHA 进入 train", 130, 2796, "full SHA", "按 SHA 重建 split"), ("同媒体多记录", 28039 - 27328, 28039, "full media identity", "每媒体限额")],
        "PixMo-Cap": [("达到 200 有效媒体前的 URL/解码失败", 303, 503, "deterministic candidate decisions", "冻结媒体快照并重新验证"), ("URL identity 重复", 717042 - 716551, 717042, "full URL set", "按 URL+SHA 去重")],
        "PixMo-Points": [("候选媒体失败", 132, 332, "deterministic candidate decisions", "冻结可访问媒体"), ("source SHA mismatch", 59, 800, "candidate media validation", "拒绝并追溯 URL"), ("每媒体平均记录密度", 2376222, 227963, "full media IDs", "每媒体限额 6")],
        "Molmo2-VideoCapQA": [("源包缺视频字节", 1000575, 1000575, "source layout", "完成 media join 前隔离")],
        "Molmo2-VideoPoint": [("annotator_unsure", 96319, 658340, "full source field", "降权或排除"), ("非 canonical 重复视图行", 658340, 1316680, "Parquet footer", "仅使用 train canonical 表"), ("仅本地打包 generated video", 15204, 275837, "full video_source and tar", "分来源验证媒体")],
        "Molmo2-VideoSubtitleQA": [("源包缺视频字节", 468502, 468502, "source layout", "完成 media join 前隔离"), ("alignment type 标签漂移", None, None, "full category counts", "归一大小写/语言/近义类别")],
        "Molmo2-VideoTrack": [("统一媒体 join 未完成", 29704, 29704, "source archive probe", "按 12 个上游分别恢复媒体"), ("空 referring expression", 6, 29704, "full field scan", "拒绝")],
    }
    rows = []
    for dataset in DATASET_ORDER:
        for issue, observed, denominator, evidence, action in issues[dataset]:
            rows.append({
                "dataset": dataset, "rating": META[dataset]["rating"], "issue": issue,
                "observed": observed, "denominator": denominator,
                "rate": observed / denominator if isinstance(observed, (int, float)) and denominator else None,
                "evidence": evidence, "required_action": action,
            })
    return rows


def sample_hash_sets() -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for dataset in DATASET_ORDER:
        values = set()
        for row in load_jsonl(SUB / dataset / "sampling-manifest.jsonl"):
            digest = row.get("sha256") or row.get("source_video_sha256")
            if digest:
                values.add(str(digest))
        output[dataset] = values
    return output


def overlap_outputs(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    edges = data["overlap"]["source_lineage_edges"]
    hashes = sample_hash_sets()
    edge_rows = []
    cell_notes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in edges:
        left = row["left"].split(":", 1)[0]
        right = row["right"].split(":", 1)[0]
        key = tuple(sorted((left, right)))
        note = f"full {row['namespace']}: {row['intersection']:,}"
        cell_notes[key].append(note)
        edge_rows.append(row)
    matrix = []
    for left in DATASET_ORDER:
        row: dict[str, Any] = {"dataset": left}
        for right in DATASET_ORDER:
            if left == right:
                row[right] = "self"
                continue
            key = tuple(sorted((left, right)))
            if cell_notes.get(key):
                row[right] = " | ".join(cell_notes[key])
            else:
                intersection = len(hashes[left] & hashes[right])
                row[right] = f"sample SHA: {intersection}/{min(len(hashes[left]), len(hashes[right]))}"
        matrix.append(row)
    split_rows = list(data["overlap"]["split_media_audit"]) + list(data["split"]["split_media_audit"])
    for row in split_rows:
        if row["dataset"] == "VQAv2" and {row["left_split"], row["right_split"]} == {"test", "testdev"}:
            row["interpretation"] = "expected evaluation subset view; never treat as independent media split"
        elif row["dataset"] == "GQA" and "submission" in {row["left_split"], row["right_split"]}:
            row["interpretation"] = "submission is an aggregate evaluation view; source media overlap is expected but train candidates must exclude it"
        elif row["intersection"]:
            row["interpretation"] = "leakage group crosses source split; rebuild split"
        else:
            row["interpretation"] = "isolated in observed source namespace"
    return matrix, edge_rows, split_rows


def inventory_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    probe_map: dict[str, dict[str, Any]] = {}
    name_map = {"Chartqa": "ChartQA", "ai2d": "AI2D", "llava-instruct": "LLaVA-Instruct", "vlm-r1": "VLM-R1", "robo2vlm": "Robo2VLM", "robovqa": "RoboVQA", "spatialvlm": "SpatialVLM", "textvqa": "TextVQA", "pixmo-cap": "PixMo-Cap", "pixmo-points": "PixMo-Points"}
    for path in ARTIFACTS.glob("*probe*.json"):
        obj = load_json(path)
        for item in obj.get("datasets", []):
            name = name_map.get(item.get("dataset"), item.get("dataset"))
            if name in DATASET_ORDER:
                probe_map[name] = item
    evidence = load_json(ROOT / "source-readonly-evidence.json")
    digest_map = {row["dataset"]: row.get("after_digest") for row in evidence.get("datasets", [])}
    rows = []
    for dataset in DATASET_ORDER:
        item = probe_map.get(dataset, {})
        files = item.get("files", [])
        source_path = f"{SOURCE_ROOT}/{META[dataset]['source_dir']}"
        source_formats, media_storage, identity_keys, locator = SOURCE_DETAILS[dataset]
        modality = META[dataset]["modality"]
        image_count: Any = metrics[dataset]["unique_media"] if "image" in modality and "video" not in modality else 0
        video_count: Any = metrics[dataset]["unique_media"] if "video" in modality else 0
        rows.append({
            "entry": dataset, "status": "included", "source_path": source_path,
            "source_version": "not declared in observed source package",
            "source_revision": "not declared; analysis snapshot is bound by source_population_digest",
            "source_splits": SOURCE_SPLITS[dataset],
            "source_formats": source_formats,
            "file_count": item.get("file_count"), "source_bytes_observed": sum(row.get("size", 0) for row in files) if files else None,
            "source_records": metrics[dataset]["source_records"], "unique_media": metrics[dataset]["unique_media"],
            "image_count": image_count, "video_count": video_count, "audio_count": 0,
            "unique_scene_episode_trajectory": metrics[dataset]["unique_scene"] if metrics[dataset]["unique_scene"] is not None else "not exposed in source",
            "supervision_units": metrics[dataset]["supervision_units"], "supervised_token_estimate": metrics[dataset]["token_estimate"],
            "media_storage": media_storage, "media_and_lineage_identity": identity_keys, "reproducible_locator": locator,
            "source_population_digest": digest_map.get(dataset),
            "download_or_generation_time": "not declared in source; observed snapshot audited 2026-08-05/06",
            "modality": modality, "simple_task": META[dataset]["simple_task"],
            "license": META[dataset]["license"], "exclusion_reason": "",
            "evidence_scope": metrics[dataset]["evidence"],
            "evidence_time": ANALYSIS_DATE,
        })
    rows.extend([
        {"entry": ".cache", "status": "excluded", "source_path": f"{SOURCE_ROOT}/.cache", "file_count": None, "source_bytes_observed": None, "source_records": None, "unique_media": None, "modality": "cache", "simple_task": "none", "license": "n/a", "exclusion_reason": "运行缓存，不是源数据集边界", "evidence_time": ANALYSIS_DATE},
        {"entry": ".gitattributes", "status": "excluded", "source_path": f"{SOURCE_ROOT}/.gitattributes", "file_count": 1, "source_bytes_observed": 2307, "source_records": None, "unique_media": None, "modality": "repository metadata", "simple_task": "none", "license": "n/a", "exclusion_reason": "根目录仓库元数据", "evidence_time": ANALYSIS_DATE},
        {"entry": "robovqa-2unuse", "status": "excluded", "source_path": f"{SOURCE_ROOT}/robovqa-2unuse", "file_count": None, "source_bytes_observed": None, "source_records": None, "unique_media": None, "modality": "unknown/unused", "simple_task": "none", "license": "未核验", "exclusion_reason": "现场名称明确标记 2unuse；未授权纳入 canonical RoboVQA", "evidence_time": ANALYSIS_DATE},
        {"entry": "textvqa-2unuse", "status": "excluded", "source_path": f"{SOURCE_ROOT}/textvqa-2unuse", "file_count": None, "source_bytes_observed": None, "source_records": None, "unique_media": None, "modality": "unknown/unused", "simple_task": "none", "license": "未核验", "exclusion_reason": "现场名称明确标记 2unuse；未授权纳入 canonical TextVQA", "evidence_time": ANALYSIS_DATE},
    ])
    return rows


def training_reference_rows() -> list[dict[str, Any]]:
    references = [
        (1, "Liu et al.", 2023, "Visual Instruction Tuning", "2304.08485",
         "视觉-语言特征对齐后进行视觉指令微调的分阶段范式。",
         "Stage 1；通用能力保持；混合回放前的基础对齐"),
        (2, "McKinzie et al.", 2024, "MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training", "2403.09611",
         "消融表明 image-caption、interleaved image-text 与 text-only 的谨慎混合对多模态预训练至关重要。",
         "Stage 1；Stage 5；通用数据混合与回放"),
        (3, "Deitke et al.", 2024, "Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models", "2409.17146",
         "分别使用高质量长描述、自由式问答和 2D pointing 数据，强调数据质量与训练流水线。",
         "Stage 1；Stage 2；caption、VQA 与 pointing 监督"),
        (4, "Chen et al.", 2024, "SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities", "2401.12168",
         "使用大规模定性与定量 3D spatial VQA 增强空间推理，并验证其机器人下游价值。",
         "Stage 2；空间与三维几何前置训练"),
        (5, "Li et al.", 2024, "LLaVA-OneVision: Easy Visual Task Transfer", "2408.03326",
         "统一单图、多图与视频任务，并报告由图像任务向视频理解的跨场景迁移。",
         "Stage 3；图像能力向视频时序任务迁移"),
        (6, "Driess et al.", 2023, "PaLM-E: An Embodied Multimodal Language Model", "2303.03378",
         "联合训练机器人规划、VQA、caption 与互联网语言视觉任务，观察到正迁移并保留通用语言能力。",
         "通用-具身联合训练；Stage 4；Stage 5 回放"),
        (7, "Brohan et al.", 2023, "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", "2307.15818",
         "将机器人轨迹与互联网 VQA 等任务共同微调，并把动作编码为与文本一致的 token。",
         "Stage 4；具身推理优先；通用与机器人数据共同训练"),
        (8, "Open X-Embodiment Collaboration et al.", 2023, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", "2310.08864",
         "标准化多机器人、多任务轨迹并展示跨 embodiment 的正迁移。",
         "Stage 4；新增真实 trajectory；跨来源 episode 配比"),
        (9, "Kim et al.", 2024, "OpenVLA: An Open-Source Vision-Language-Action Model", "2406.09246",
         "在 970k 多样真实机器人示范上训练 VLA，并验证新任务与多任务环境微调。",
         "Stage 4；真实状态-动作监督；具身推理优先"),
        (10, "Zawalski et al.", 2024, "Robotic Control via Embodied Chain-of-Thought Reasoning", "2407.08693",
         "在动作预测前显式监督计划、子任务、运动、目标框和末端执行器位置等具身推理步骤。",
         "Stage 4；子目标、规划与 grounded reasoning"),
    ]
    return [
        {
            "citation_id": citation_id,
            "authors": authors,
            "year": year,
            "title": title,
            "arxiv_id": arxiv_id,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "supported_principle": supported_principle,
            "used_in": used_in,
            "citation_scope": "支持训练设计原则，不提供本报告的固定数据占比",
            "metadata_source": "arXiv official API",
            "verified_date": ANALYSIS_DATE,
        }
        for citation_id, authors, year, title, arxiv_id, supported_principle, used_in in references
    ]


def training_rows() -> list[dict[str, Any]]:
    recipe_weights = {
        "通用能力保持": [38, 22, 14, 10, 16],
        "通用-具身均衡": [25, 25, 18, 20, 12],
        "具身推理优先": [15, 20, 20, 30, 15],
    }
    stages = [
        ("Stage 1", "通用感知、caption、VQA、OCR、diagram", "COCO; VQAv2; TextVQA; ChartQA; AI2D; PixMo-Cap; filtered LLaVA", "450k-900k images", "1.2m-2.4m", "350m-900m", "caption 1; VQA 3/media", "同一 COCO/VG/TextVQA family 合并后采样", "RealWorldQA; OCRBench v2; ChartQA; AI2D", "1;2;3"),
        ("Stage 2", "grounding、pointing、2D/3D 空间、深度、跟踪", "VisualGenome; VLM-R1; PixMo-Points; SpatialVLM; Robo2VLM geometry; Molmo2-VideoPoint; VideoTrack after media join", "250k-500k media", "900k-1.6m", "180m-450m", "bbox 4/image; point 6/image; spatial QA 4/media", "VideoTrack 恢复媒体后再进入；各来源设置单媒体监督上限", "RoboSpatial; grounding val; point/track held-out", "3;4"),
        ("Stage 3", "视频时序、状态变化、动作与未来预测", "RoboVQA past/future; Robo2VLM trajectory; VideoCapQA/SubtitleQA after media join; VideoPoint", "180k-320k videos", "550k-950k", "250m-520m", "6 tasks/video; 4 QA/video/source", "按 original video/uid 分组，Molmo 三源共享 video_id 联合限额", "Video-MME; EgoPlan temporal subset; held-out robot videos", "5;6"),
        ("Stage 4", "affordance、成功判断、下一步动作、子目标与规划", "RoboVQA affordance/success/planning; Robo2VLM obstacle/goal/success; new OXE/Bridge/DROID data", "160k-300k episodes", "450k-900k", "220m-500m", "1 description + 6 unique tasks/video; 8 QA/scene", "当前源缺 E6；新增真实 trajectory 前不做过长阶段", "EgoPlan-Bench; OpenEQA; task-success held-out", "6;7;8;9;10"),
        ("Stage 5", "混合回放与抗遗忘", "clean general pool + all qualified embodied pools", "350k-650k mixed media", "800k-1.4m", "300m-650m", "任一媒体总上限 8", "至少 25% 通用感知回放；所有来源通过媒体与格式准入", "全部 benchmark 回归 + 通用/具身双门槛", "2;6;7"),
    ]
    rows = []
    evidence_basis = [
        "E0 general capability replay; B/C/Q gates from quality-risk-matrix.csv",
        "E1 grounding/spatial gap; media completeness gates from source-accounting.csv",
        "E2 temporal/action gap; only media-closed video sources admitted",
        "E3-E6 affordance/planning gap; current E6 coverage is zero and requires new trajectory sources",
        "anti-forgetting replay; all stages filtered by media completeness and A/B/C/Q status",
    ]
    for recipe, weights in recipe_weights.items():
        for index, stage in enumerate(stages):
            rows.append({
                "recipe": recipe, "stage": stage[0], "total_mix_percent": weights[index],
                "capability_target": stage[1], "datasets": stage[2], "unique_media_target": stage[3],
                "effective_supervision_target": stage[4], "token_target": stage[5],
                "per_media_cap": stage[6], "entry_and_quota_rule": stage[7], "validation_benchmarks": stage[8],
                "reference_ids": stage[9],
                "evidence_basis": evidence_basis[index],
                "recipe_status": "candidate recommendation only; no conversion or training executed",
                "evidence_date": ANALYSIS_DATE,
            })
    return rows


def recommended_rows() -> list[dict[str, Any]]:
    rows = [
        ("Open X-Embodiment", "robot video/state/action", "跨机器人真实操作", "trajectory policy learning", "E2-E6", "约 1M trajectories，待版本核验", "yes", "各子集许可不同", "公开集合/需逐源下载", "medium", "benchmark lineage 需排查", "high", "P0"),
        ("BridgeData V2", "multi-view robot trajectory", "厨房与桌面操作", "language-conditioned manipulation", "E2-E5", "约 60k trajectories，待核验", "yes", "待核验", "公开", "medium", "low-medium", "medium", "P0"),
        ("DROID", "robot video+proprioception+action", "多场景真实机器人操作", "state/action/trajectory", "E2-E6", "约 76k trajectories，待核验", "yes", "待核验", "公开申请/下载", "low", "low", "high", "P0"),
        ("RT-1", "robot trajectory", "办公/厨房移动操作", "language-conditioned action", "E3-E6", "约 130k episodes，待核验", "yes", "待核验", "原始数据可得性待确认", "medium", "medium", "high", "P1"),
        ("BC-Z", "robot trajectory+language", "桌面物体操作", "zero-shot task generalization", "E3-E5", "待核验", "yes", "待核验", "公开状态待确认", "medium", "low", "medium", "P1"),
        ("Language Table", "top-down video+action", "桌面推物", "language/action grounding", "E2-E5", "数十万 episodes，待核验", "yes", "待核验", "公开", "low", "low", "medium", "P1"),
        ("RoboCasa", "simulation video/state/action", "家庭长程操作", "planning/trajectory/success", "E3-E6", "规模随生成配置，待核验", "yes", "代码/资产许可需分开", "可生成", "low", "low", "high", "P0"),
        ("ManiSkill", "simulation RGB-D/state/action", "多类机器人操作", "control/trajectory/planning", "E2-E6", "任务与 demonstration 版本待核验", "yes", "待核验", "公开", "low", "low", "medium", "P1"),
        ("Ego4D", "egocentric video+audio+text", "人类第一视角日常活动", "action/forecasting/memory", "E2-E6", "数千小时，具体版本待核验", "partial", "研究许可协议", "申请", "medium", "high for Ego benchmarks", "high", "P1"),
        ("Ego-Exo4D", "ego+exo multi-view video", "技能执行与跨视角", "3D/action/trajectory correspondence", "E1-E5", "千小时级，待核验", "partial", "研究许可协议", "申请", "medium", "medium-high", "high", "P1"),
        ("EPIC-KITCHENS-100", "egocentric video+action labels", "厨房操作", "action/state/temporal", "E2-E4", "100 hours", "partial", "非商业/研究条款待核验", "申请", "medium", "high", "medium", "P2"),
        ("ALFRED", "simulation trajectory+language", "室内导航与操作", "subgoal/long-horizon planning", "E3-E6", "约 25k instructions，待核验", "yes", "待核验", "公开", "low", "medium", "medium", "P0"),
        ("TEACh", "dialogue+simulation trajectory", "交互式室内任务", "dialogue planning/recovery", "E4-E6", "待核验", "yes", "待核验", "公开", "low", "medium", "high", "P1"),
        ("ScanQA", "3D scan+QA", "真实室内 3D 场景", "3D spatial QA", "E1/E6 prerequisite", "约 41k QA，待核验", "no", "依赖 ScanNet 条款", "申请", "medium", "high for ScanQA", "medium", "P1"),
        ("OpenEQA", "embodied environment QA", "真实/仿真室内", "memory/exploration/open QA", "E5-E6", "benchmark 规模，待核验", "partial", "待核验", "公开 benchmark", "high", "very high", "low for training; high for eval", "EVAL"),
        ("EgoPlan-Bench / RoboSpatial", "video/image benchmark", "具身规划与机器人空间", "planning/spatial evaluation", "E1-E6", "benchmark，待核验", "partial", "待核验", "公开 benchmark", "high", "very high", "not for training", "EVAL"),
    ]
    columns = ["dataset", "modality", "scene", "task", "target_capability", "scale", "has_action_trajectory_state", "license", "availability", "overlap_risk", "benchmark_contamination_risk", "preprocessing_cost", "priority"]
    output = [dict(zip(columns, row)) for row in rows]
    for row in output:
        row["evidence_status"] = "external recommendation; not part of current server source statistics"
        row["evidence_date"] = ANALYSIS_DATE
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def representative_records(dataset: str, data: dict[str, Any]) -> list[Any]:
    preferred: list[Any] = []
    if dataset == "AI2D": preferred = data["ai2d"].get("representative_source_records", [])
    elif dataset == "VisualGenome":
        groups = data["visual"].get("representative_source_records", {})
        preferred = [item for rows in groups.values() for item in rows[:1]]
    elif dataset == "LLaVA-Instruct": preferred = data["llava"].get("representative_source_records", [])
    elif dataset == "VLM-R1": preferred = data["vlm"].get("representative_source_records", [])
    elif dataset == "RoboVQA":
        preferred = [rows[0] for rows in data["robovqa"].get("representative_source_records_by_file", {}).values() if rows]
    elif dataset == "Robo2VLM":
        preferred = [rows[0] for rows in data["robo2caps"].get("representative_source_records_by_primary_capability", {}).values() if rows]
    else:
        preferred = full_stats(dataset).get("representative_records", [])
    records = [compact(item.get("record", item) if isinstance(item, dict) else item) for item in preferred[:3]]
    manifests = load_jsonl(SUB / dataset / "sampling-manifest.jsonl")
    for row in manifests:
        if len(records) >= 3: break
        relative = row.get("record_archive_path")
        path = SUB / dataset / relative if relative else None
        if path and path.exists():
            try: records.append(compact(load_json(path)))
            except (OSError, json.JSONDecodeError): pass
    if len(records) < 3:
        record_dir = SUB / dataset / "records"
        if record_dir.exists():
            for path in sorted(record_dir.rglob("*.json")):
                if len(records) >= 3: break
                try: records.append(compact(load_json(path)))
                except (OSError, json.JSONDecodeError): pass
    if len(records) < 3:
        records.extend(compact(row) for row in manifests[: 3 - len(records)])
    if len(records) < 3:
        raise RuntimeError(f"{dataset} has fewer than three representative source records")
    return records[:3]


def build_schemas(data: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> None:
    target = ROOT / "source-schemas"
    target.mkdir(exist_ok=True)
    for dataset in DATASET_ORDER:
        path = SUB / dataset / "source-schema.json"
        observed = load_json(path)
        if observed.get("_generated_by") == "build_source_analysis_report.py":
            observed = observed["observed_schema"]
        payload = {
            "dataset": dataset,
            "source_root": f"{SOURCE_ROOT}/{META[dataset]['source_dir']}",
            "source_only": True,
            "observed_schema": observed,
            "input_contract": META[dataset]["input"],
            "output_contract": META[dataset]["output"],
            "schema_evidence": metrics[dataset]["evidence"],
            "representative_source_records": representative_records(dataset, data),
            "coordinate_or_source_note": META[dataset]["embodied_evidence"],
            "_generated_by": "build_source_analysis_report.py",
            "_generated_at": ANALYSIS_DATE,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")
        (target / f"{dataset}.json").write_text(text, encoding="utf-8")


STYLE = """
:root{--ink:#1d262d;--muted:#63717a;--line:#d7dde0;--paper:#fff;--wash:#f3f5f4;--teal:#08786f;--teal-soft:#dff2ef;--blue:#28658c;--blue-soft:#e5f0f7;--amber:#99600a;--amber-soft:#fff3d8;--red:#a33d36;--red-soft:#f9e7e5;--green:#247247;--shadow:0 1px 2px rgba(25,35,40,.08)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;letter-spacing:0}a{color:var(--blue);text-decoration-thickness:1px;text-underline-offset:2px}main{width:min(1240px,calc(100% - 32px));margin:auto;padding:28px 0 64px;min-width:0}header{padding:24px 0 25px;border-bottom:1px solid var(--line)}section[id],h2,h3,h4{scroll-margin-top:58px}.eyebrow{color:var(--teal);font-weight:750;font-size:13px}h1{font-size:46px;line-height:1.12;margin:8px 0 13px;letter-spacing:0}h2{font-size:25px;line-height:1.28;margin:40px 0 14px;letter-spacing:0}h3{font-size:18px;margin:20px 0 9px;letter-spacing:0}p{max-width:92ch}.lede{font-size:18px;max-width:86ch}.nav{display:flex;gap:14px;flex-wrap:wrap;padding:12px 0;border-bottom:1px solid var(--line);position:sticky;top:0;background:rgba(243,245,244,.97);z-index:5}.nav a{font-size:13px}.facts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:22px 0}.fact{background:#fff;padding:17px;min-width:0}.fact b{display:block;font-size:24px;color:var(--teal);overflow-wrap:anywhere}.fact span{color:var(--muted);font-size:13px}.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.band{background:#fff;border-block:1px solid var(--line);padding:20px;margin:20px 0}.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.flow span{background:var(--teal-soft);padding:7px 10px;border-radius:4px}.flow b{color:var(--muted)}.note,.warning,.danger{padding:12px 15px;background:#fff;border-left:4px solid var(--blue)}.warning{border-color:var(--amber);background:var(--amber-soft)}.danger{border-color:var(--red);background:var(--red-soft)}.table-wrap{overflow-x:auto;min-width:0;border:1px solid var(--line);background:#fff}table{width:100%;border-collapse:collapse;background:#fff;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:9px 10px;overflow-wrap:anywhere}th{background:#edf1f1;position:sticky;top:0;z-index:1}.rating{display:inline-block;min-width:28px;text-align:center;padding:1px 5px;border:1px solid currentColor;font-weight:800}.rating-A{color:var(--green)}.rating-B{color:var(--blue)}.rating-C{color:var(--amber)}.rating-Q{color:var(--red)}.bar-list{display:grid;gap:9px}.bar-row{display:grid;grid-template-columns:minmax(150px,240px) minmax(160px,1fr) 105px;gap:10px;align-items:center}.bar-track{height:13px;background:#e3e8e8;position:relative}.bar-fill{height:100%;background:var(--teal)}.bar-fill.blue{background:var(--blue)}.bar-fill.amber{background:var(--amber)}.chart-block{margin:18px 0 28px}.chart-note{font-size:12px;color:var(--muted)}.legend{display:flex;flex-wrap:wrap;gap:8px 14px;margin:8px 0 12px;font-size:12px}.legend span{display:inline-flex;align-items:center;gap:5px}.swatch{width:12px;height:12px;display:inline-block}.stack-chart{display:grid;gap:7px}.stack-row{display:grid;grid-template-columns:minmax(130px,210px) minmax(240px,1fr) 80px;gap:10px;align-items:center}.stack-track{height:18px;background:#e2e7e7;display:flex;overflow:hidden}.stack-seg{height:100%;min-width:0}.paired-row{display:grid;grid-template-columns:minmax(130px,210px) minmax(220px,1fr) 120px;gap:10px;align-items:center}.paired-track{display:grid;gap:3px}.paired-track i{height:7px;display:block}.risk{min-width:760px}.wide-table{min-width:1760px}.risk td:not(:first-child),.overlap td:not(:first-child){text-align:center}.r0{background:#edf5ef}.r1{background:#fff3d8}.r2{background:#f8ddd8}.r3{background:#a33d36;color:#fff;font-weight:700}.ov-self{background:#e8ecec;color:#758087}.ov-full{background:#08786f;color:#fff}.ov-sample-hit{background:#fff3d8}.ov-sample-zero{background:#f6f7f7;color:#8b9599}.overlap{min-width:1700px}.overlap th,.overlap td{font-size:10px;padding:5px;max-width:105px}.provenance{border-left:3px solid var(--teal);padding:9px 12px;background:#fff;font-size:12px;color:var(--muted)}.heat{min-width:1060px}.heat td:not(:first-child){text-align:center;padding:7px}.s0{background:#f6f7f7;color:#9aa3a7}.s1{background:#e5f0f7;color:#185b52}.s2{background:#cae7df;color:#185b52}.s3{background:#08786f;color:white;font-weight:700}.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.sample,.record-sample{background:#fff;border:1px solid var(--line);box-shadow:var(--shadow);min-width:0}.sample>a{display:block;aspect-ratio:4/3;background:#e7ebeb;overflow:hidden}.sample img{width:100%;height:100%;object-fit:contain;display:block}.sample-body,.record-body{padding:12px;min-width:0}.sample-head{display:flex;justify-content:space-between;gap:8px}.sample-head b{color:var(--teal)}.sample p,.record-sample p{font-size:12px;margin:7px 0;overflow-wrap:anywhere}.sample dl{display:grid;grid-template-columns:48px minmax(0,1fr);gap:3px 7px;margin:0;font-size:11px}.sample dt{color:var(--muted)}.sample dd{margin:0;overflow-wrap:anywhere}.record-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.record-sample{padding:0}.record-sample pre{max-height:220px;overflow:auto}.schema{max-height:500px;overflow:auto;background:#20282d;color:#edf3f3;padding:14px;font:12px/1.55 Consolas,monospace;white-space:pre-wrap;word-break:break-word}.mini{font-size:12px;color:var(--muted)}footer{margin-top:42px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}code{font-family:Consolas,monospace;overflow-wrap:anywhere}@media(max-width:900px){h1{font-size:38px}.facts,.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}.three,.record-grid{grid-template-columns:minmax(0,1fr)}.two{grid-template-columns:minmax(0,1fr)}.bar-row,.stack-row,.paired-row{grid-template-columns:130px minmax(100px,1fr) 80px}}@media(max-width:540px){main{width:calc(100% - 20px);padding-top:16px}h1{font-size:31px}.facts,.gallery{grid-template-columns:minmax(0,1fr)}.bar-row,.stack-row,.paired-row{grid-template-columns:95px minmax(80px,1fr) 58px;font-size:11px}.nav{position:static}.sample p{min-height:0}}
"""

STYLE += """
.s1{color:#285d7d}
.wide-table th:nth-child(1),.wide-table td:nth-child(1){min-width:110px}
.wide-table th:nth-child(2),.wide-table td:nth-child(2){min-width:88px}
.wide-table th:nth-child(3),.wide-table td:nth-child(3){min-width:72px}
.wide-table th:nth-child(4),.wide-table td:nth-child(4){min-width:180px}
.wide-table th:nth-child(5),.wide-table td:nth-child(5){min-width:300px}
.task-ledger{min-width:2300px}
h4{font-size:15px;margin:17px 0 8px;letter-spacing:0}
.basis-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin:16px 0 34px}
.basis-grid>div{border-top:3px solid var(--teal);padding-top:10px;min-width:0}
.basis-grid b{color:var(--teal)}
.basis-grid p{font-size:13px;margin:5px 0;color:var(--muted)}
.recipe-plan{background:#fff;border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:4px;padding:20px;margin:22px 0 30px;min-width:0}
.recipe-balanced{border-top-color:var(--teal)}
.recipe-embodied{border-top-color:var(--red)}
.recipe-head{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(360px,.95fr);gap:28px;align-items:center}
.recipe-head h3{font-size:22px;margin:4px 0 7px}
.recipe-head p{margin:0;color:var(--muted)}
.recipe-kicker{font-size:12px;font-weight:750;color:var(--blue)}
.recipe-balanced .recipe-kicker{color:var(--teal)}
.recipe-embodied .recipe-kicker{color:var(--red)}
.mix-track{height:20px;background:#e3e8e8;display:flex;overflow:hidden}
.mix-track i{display:block;height:100%}
.mix-legend{display:flex;flex-wrap:wrap;gap:6px 12px;margin-top:9px;font-size:11px;color:var(--muted)}
.mix-legend span{display:inline-flex;align-items:center;gap:5px}
.mix-legend i{display:inline-block;width:10px;height:10px}
.recipe-analysis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:20px 0}
.recipe-analysis>div{border-left:3px solid var(--line);padding-left:12px;min-width:0}
.recipe-analysis p{font-size:13px;margin:0;color:var(--muted)}
.citations{display:inline-flex;gap:3px;margin-left:4px;white-space:nowrap}.citations a{font-size:11px;font-weight:750}
.reference-boundary{border-left:4px solid var(--amber);background:var(--amber-soft);padding:12px 15px;max-width:none}
.reference-list{margin:12px 0 24px;padding-left:34px;border-top:1px solid var(--line)}
.reference-list li{padding:14px 8px 14px 4px;border-bottom:1px solid var(--line);scroll-margin-top:58px}
.reference-title{font-weight:750}.reference-meta{font-size:12px;color:var(--muted);margin-top:3px}
.reference-list p{font-size:13px;margin:6px 0;max-width:110ch}
@media(max-width:900px){.basis-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.recipe-head{grid-template-columns:minmax(0,1fr)}.recipe-analysis{grid-template-columns:minmax(0,1fr)}}
@media(max-width:540px){.basis-grid{grid-template-columns:minmax(0,1fr)}.recipe-plan{padding:14px}.recipe-head h3{font-size:19px}.mix-legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}}
"""


def table_html(rows: Iterable[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    rows = list(rows)
    head = "".join(f"<th>{e(label)}</th>" for _, label in columns)
    body = "".join("<tr>" + "".join(f"<td>{e(row.get(key, ''))}</td>" for key, _ in columns) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def dataset_inventory_table_html(
    rows: Iterable[dict[str, Any]],
    columns: list[tuple[str, str]],
) -> str:
    rows = list(rows)
    head = "".join(f"<th>{e(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if key == "dataset":
                cells.append(f'<td><a href="{quote(str(value))}.html">{e(value)}</a></td>')
            else:
                cells.append(f"<td>{e(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def wide_table_html(rows: Iterable[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    return table_html(rows, columns).replace("<table>", '<table class="wide-table">', 1)


def rating_html(value: str) -> str:
    return f'<span class="rating rating-{e(value)}">{e(value)}</span>'


def sample_io(row: dict[str, Any]) -> tuple[str, str, str]:
    input_value = next((row.get(key) for key in ["question", "user_prompt", "first_user", "input_preview", "first_question", "logical_image_path"] if row.get(key)), "未在 manifest 展开")
    output_value = next((row.get(key) for key in ["answer_preview", "assistant_raw", "first_assistant", "output_preview", "first_answer_text", "label_names", "parsed_output"] if row.get(key) not in (None, "", [])), "源输出缺失或不适用")
    source = row.get("source_uri") or row.get("source_image_path") or row.get("old_image_path") or row.get("source_member") or row.get("member") or f"{row.get('shard','')}#row={row.get('shard_row',row.get('source_row',''))}"
    return short(input_value), short(output_value), short(source, 180)


def gallery_html(dataset: str) -> tuple[str, int, int]:
    manifests = load_jsonl(SUB / dataset / "sampling-manifest.jsonl")
    cards = []
    broken = 0
    for row in manifests:
        media = row.get("media_archive_path") or row.get("preview_archive_path")
        if not media or not (SUB / dataset / media).exists():
            broken += 1
            continue
        input_value, output_value, source = sample_io(row)
        record = row.get("record_archive_path") or row.get("qa_archive_path")
        media_url = f"sub-dataset/{quote(dataset)}/{quote(str(media), safe='/')}"
        detail_url = f"sample-details/{quote(dataset)}/{quote(str(row['sample_id']))}.html"
        record_link = f'<a href="sub-dataset/{quote(dataset)}/{quote(str(record), safe="/")}">record</a>' if record and (SUB / dataset / record).exists() else "record n/a"
        digest = row.get("sha256") or row.get("source_video_sha256") or row.get("preview_sha256") or "n/a"
        cards.append(f'''<article class="sample" id="sample-{quote(str(row['sample_id']))}"><a href="{detail_url}"><img loading="lazy" src="{media_url}" alt="{e(dataset)} {e(row.get('sample_id','sample'))} source sample"></a><div class="sample-body"><div class="sample-head"><b><a href="{detail_url}">{e(row.get('sample_id','sample'))}</a></b><span>{e(row.get('split', row.get('task_splits','')))}</span></div><p><strong>输入：</strong>{e(input_value)}</p><p><strong>输出：</strong>{e(output_value)}</p><dl><dt>源</dt><dd>{e(source)}</dd><dt>hash</dt><dd>{e(str(digest)[:20])}...</dd><dt>查看</dt><dd><a href="{detail_url}">完整 QA / 标注</a> · <a href="{media_url}">媒体</a> · {record_link}</dd></dl></div></article>''')
    if manifests:
        return "".join(cards), len(manifests), broken
    record_manifests = load_jsonl(SUB / dataset / "record-sampling-manifest.jsonl")
    for row in record_manifests:
        relative = row.get("record_archive_path")
        path = SUB / dataset / str(relative or "")
        try:
            record = compact(load_json(path)) if relative and path.is_file() else {"error": "archived record unavailable"}
        except (OSError, json.JSONDecodeError):
            record = {"error": "unreadable archived record"}
            broken += 1
        text = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
        sample_id = str(row.get("sample_id", f"{dataset}-{int(row.get('draw_order', 0)) + 1:03d}"))
        detail_url = f"sample-details/{quote(dataset)}/{quote(sample_id)}.html"
        source_record_url = f"sub-dataset/{quote(dataset)}/{quote(str(relative), safe='/')}" if relative else ""
        source_link = f'<a href="{source_record_url}">source record</a>' if source_record_url else "source record n/a"
        cards.append(f'''<article class="record-sample" id="sample-{quote(sample_id)}"><div class="record-body"><div class="sample-head"><b><a href="{detail_url}">{e(sample_id)}</a></b>{source_link}</div><p><strong>输入：</strong>{e(row.get('input_preview',''))}</p><p><strong>输出：</strong>{e(row.get('output_preview',''))}</p><p><a href="{detail_url}">查看完整 QA / 标注与原始 JSON</a></p><pre>{e(text[:2600])}</pre></div></article>''')
    return "".join(cards), 0, 0


def capability_table(dataset: str) -> str:
    values = META[dataset]["capabilities"]
    rows = [{"capability": label, "score": values.get(key, 0), "meaning": ["无直接证据", "少量/间接", "显著覆盖", "核心能力"][values.get(key, 0)]} for key, label in CAPABILITY_COLUMNS if values.get(key, 0)]
    return table_html(rows, [("capability", "能力"), ("score", "覆盖 0-3"), ("meaning", "证据强度")])


def legacy_dataset_page(dataset: str, data: dict[str, Any], metrics: dict[str, dict[str, Any]], qrows: list[dict[str, Any]], edge_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> str:
    meta = META[dataset]
    metric = metrics[dataset]
    summ = summary(dataset)
    schema = load_json(ROOT / "source-schemas" / f"{dataset}.json")
    gallery, accepted, broken = gallery_html(dataset)
    expected = summ.get("selected_count", 0)
    if accepted != expected or broken:
        raise RuntimeError(f"{dataset}: gallery accepted={accepted}, expected={expected}, broken={broken}")
    task_rows = [row for row in build_task_rows(data, metrics) if row["dataset"] == dataset]
    local_quality = [row for row in qrows if row["dataset"] == dataset]
    local_edges = [row for row in edge_rows if row["left"].split(":", 1)[0] == dataset or row["right"].split(":", 1)[0] == dataset]
    local_splits = [row for row in split_rows if row["dataset"] == dataset and row["intersection"]]
    task_table = table_html(
        [{**row, "percentage": f"{row['percentage']*100:.2f}%"} for row in task_rows],
        [("task", "任务"), ("count", "数量"), ("count_unit", "单位"), ("percentage", "占比"), ("classification_basis", "分类依据")],
    )
    quality_table = table_html(
        [{**row, "rate": f"{row['rate']*100:.2f}%" if row["rate"] is not None else "n/a"} for row in local_quality],
        [("issue", "问题"), ("observed", "观测"), ("denominator", "分母"), ("rate", "比例"), ("evidence", "证据"), ("required_action", "处理")],
    )
    overlap_table = table_html(local_edges, [("left", "左源"), ("right", "右源"), ("intersection", "交集"), ("namespace", "命名空间"), ("relationship", "关系")]) if local_edges else "<p>没有可共享的全量 ID 命名空间；仅保留确定性样本 SHA 审计。</p>"
    split_table = table_html(local_splits, [("left_split", "split A"), ("right_split", "split B"), ("intersection", "媒体交集"), ("namespace", "证据"), ("interpretation", "解释")]) if local_splits else "<p>当前源中没有可验证的跨 split 交集，或源本身没有 split 字段。</p>"
    source_kind = "视觉样本" if expected else "record evidence"
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(dataset)} 原始源数据分析</title><style>{STYLE}</style></head><body><main><header><div class="eyebrow">VLM SOURCE CAPABILITY / {e(meta['embodied'].upper())}</div><h1>{e(dataset)}</h1><p class="lede"><strong>总判断：</strong>{e(meta['judgment'])}</p></header><nav class="nav"><a href="index.html">总览</a><a href="#contract">输入输出</a><a href="#population">规模质量</a><a href="#capability">能力</a><a href="#samples">证据样本</a></nav><section class="facts"><div class="fact"><b>{fmt(metric['source_records'])}</b><span>canonical 源记录</span></div><div class="fact"><b>{fmt(metric['unique_media'])}</b><span>唯一媒体/lineage</span></div><div class="fact"><b>{fmt(metric['supervision_units'])}</b><span>监督单元</span></div><div class="fact"><b>{rating_html(meta['rating'])}</b><span>源训练可用评级</span></div></section><section id="contract"><h2>一、输入、输出与源存储</h2><div class="band"><div class="flow"><span>{e(meta['input'])}</span><b>→</b><span>{e(meta['output'])}</span></div></div><div class="two"><div><h3>任务与领域</h3><p><strong>{e(meta['simple_task'])}</strong><br>{e(meta['domain'])}</p><p class="mini">源路径：<code>{e(SOURCE_ROOT + '/' + meta['source_dir'])}</code></p></div><div><h3>具身判定</h3><p><strong>{e(meta['embodied'])}</strong>：{e(meta['embodied_evidence'])}</p><p class="mini">许可：{e(meta['license'])}</p></div></div><p class="note"><strong>证据边界：</strong>统计、schema、样本与质量结论只来自原始源文件。媒体不可取得时明确标为 incomplete，不用其他数据补齐。</p></section><section id="population"><h2>二、全量口径、质量与泄露</h2><div class="facts"><div class="fact"><b>{fmt(metric['effective_units'])}</b><span>源证据可确认候选量：{e(metric['effective_unit_type'])}</span></div><div class="fact"><b>{fmt(metric['token_estimate'])}</b><span>监督 token 估计</span></div><div class="fact"><b>{fmt(metric['annotations_per_media'],2)}</b><span>每媒体监督密度</span></div><div class="fact"><b>{fmt(summ.get('seed'))}</b><span>确定性样本 seed</span></div></div><p>全量证据：{e(metric['evidence'])}。抽样单位为 <code>{e(summ.get('sampling_unit'))}</code>，候选 {fmt(summ.get('candidate_count'))}，accepted {fmt(expected)}，record evidence {fmt(summ.get('record_sample_count',0))}。</p>{quality_table}<h3>源 split 隔离</h3>{split_table}<h3>跨源 lineage</h3>{overlap_table}</section><section id="capability"><h2>三、任务与能力分布</h2>{task_table}<h3>能力强度矩阵</h3>{capability_table(dataset)}</section><section id="schema"><h2>四、原始 source schema</h2><p><a href="source-schemas/{quote(dataset)}.json">打开完整 schema 证据</a>。下方包含实际字段、结构变体与至少 3 条脱敏真实记录。</p><pre class="schema">{e(schema_text)}</pre></section><section id="samples"><h2>五、完整确定性 {e(source_kind)}</h2><p>seed={fmt(summ.get('seed'))}，候选按固定排列决策。图片为源字节归档；视频缩略图明确标为派生预览。incomplete 数据集只显示源记录。</p><p><a href="sample-details/{quote(dataset)}/index.html">打开 200 条完整 QA / 标注抽检目录</a></p><div class="{'gallery' if expected else 'record-grid'}">{gallery}</div></section><footer>分析时间 {ANALYSIS_DATE}。源访问只读；最终 before/after 指纹见 <a href="source-readonly-evidence.json">source-readonly-evidence.json</a>。</footer></main></body></html>'''


def bar_rows(metrics: dict[str, dict[str, Any]], field: str, color: str = "") -> str:
    values = [metrics[name].get(field) or 0 for name in DATASET_ORDER]
    maximum = max(math.log10(value + 1) for value in values) or 1
    rows = []
    for name, value in zip(DATASET_ORDER, values):
        width = math.log10(value + 1) / maximum * 100
        rows.append(f'<div class="bar-row"><span>{e(name)}</span><div class="bar-track"><div class="bar-fill {color}" style="width:{width:.2f}%"></div></div><b>{fmt(value)}</b></div>')
    return "".join(rows)


def heatmap() -> str:
    head = "".join(f"<th>{e(label)}</th>" for _, label in CAPABILITY_COLUMNS)
    body = []
    for dataset in DATASET_ORDER:
        cells = "".join(f'<td class="s{META[dataset]["capabilities"].get(key,0)}">{META[dataset]["capabilities"].get(key,0)}</td>' for key, _ in CAPABILITY_COLUMNS)
        body.append(f"<tr><th><a href=\"{quote(dataset)}.html\">{e(dataset)}</a></th>{cells}</tr>")
    return f'<div class="table-wrap"><table class="heat"><thead><tr><th>数据集</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


TASK_GROUP_ORDER = [
    "classification", "description", "open_or_compositional_vqa", "specialized_vqa",
    "counting", "grounding", "pointing", "tracking", "video_qa",
    "multimodal_instruction", "embodied_reasoning", "planning", "holdout_or_text_only",
]
TASK_GROUP_LABELS = {
    "classification": "分类", "description": "描述", "open_or_compositional_vqa": "通用/组合 VQA",
    "specialized_vqa": "OCR/图表/图解 VQA", "counting": "计数", "grounding": "Grounding",
    "pointing": "Pointing", "tracking": "Tracking", "video_qa": "Video QA",
    "multimodal_instruction": "图文指令", "embodied_reasoning": "具身推理",
    "planning": "规划", "holdout_or_text_only": "评测无答案/纯文本",
}
TASK_COLORS = {
    "classification": "#247247", "description": "#28658c", "open_or_compositional_vqa": "#08786f",
    "specialized_vqa": "#6d8f3c", "counting": "#b7831d", "grounding": "#4e78a0",
    "pointing": "#45a18f", "tracking": "#8b6f9e", "video_qa": "#bd6d41",
    "multimodal_instruction": "#64717b", "embodied_reasoning": "#a33d36",
    "planning": "#7b3f38", "holdout_or_text_only": "#b8c0c3",
}


def legend_html(order: list[str], labels: dict[str, str], colors: dict[str, str]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i class="swatch" style="background:{colors[key]}"></i>{e(labels[key])}</span>' for key in order
    ) + "</div>"


def task_stack_chart(task_rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in task_rows:
        grouped[row["dataset"]][row["task_family"]] += int(row["count"])
    body = []
    for dataset in DATASET_ORDER:
        total = sum(grouped[dataset].values()) or 1
        segments = []
        for family in TASK_FAMILY_ORDER:
            count = grouped[dataset][family]
            if not count:
                continue
            width = count / total * 100
            title = f"{family}: {count:,} ({width:.2f}%)"
            segments.append(f'<i class="stack-seg" title="{e(title)}" style="width:{width:.4f}%;background:{TASK_FAMILY_COLORS[family]}"></i>')
        body.append(f'<div class="stack-row"><span>{e(dataset)}</span><div class="stack-track">{"".join(segments)}</div><b>{fmt(total)}</b></div>')
    labels = {family: family for family in TASK_FAMILY_ORDER}
    return legend_html(TASK_FAMILY_ORDER, labels, TASK_FAMILY_COLORS) + '<div class="stack-chart">' + "".join(body) + "</div>"


def sampled_task_family_stack_chart(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row["dataset"]][row["task_family"]] += int(row["annotation_unit_count"])
    body = []
    for dataset in DATASET_ORDER:
        total = sum(grouped[dataset].values()) or 1
        segments = []
        for family in TASK_FAMILY_ORDER:
            count = grouped[dataset][family]
            if not count:
                continue
            width = count / total * 100
            title = f"{family}: {count:,} ({width:.2f}%)"
            segments.append(
                f'<i class="stack-seg" title="{e(title)}" '
                f'style="width:{width:.4f}%;background:{TASK_FAMILY_COLORS[family]}"></i>'
            )
        body.append(
            f'<div class="stack-row"><span>{e(dataset)}</span><div class="stack-track">'
            f'{"".join(segments)}</div><b>n={fmt(total)}</b></div>'
        )
    labels = {family: family for family in TASK_FAMILY_ORDER}
    return legend_html(TASK_FAMILY_ORDER, labels, TASK_FAMILY_COLORS) + '<div class="stack-chart">' + "".join(body) + "</div>"


def task_family_portfolio_chart(rows: list[dict[str, str]], *, balanced: bool) -> str:
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_dataset[row["dataset"]][row["task_family"]] += int(row["annotation_unit_count"])

    values: dict[str, float] = defaultdict(float)
    raw_counts: Counter[str] = Counter()
    for dataset in DATASET_ORDER:
        total = sum(by_dataset[dataset].values()) or 1
        for family, count in by_dataset[dataset].items():
            raw_counts[family] += count
            if balanced:
                values[family] += count / total / len(DATASET_ORDER) * 100
    if not balanced:
        total = sum(raw_counts.values()) or 1
        values = {family: count / total * 100 for family, count in raw_counts.items()}

    ordered = sorted(TASK_FAMILY_ORDER, key=lambda family: (-values.get(family, 0), TASK_FAMILY_ORDER.index(family)))
    maximum = max(values.values(), default=1) or 1
    bars = []
    for family in ordered:
        percentage = values.get(family, 0)
        if percentage <= 0:
            continue
        width = percentage / maximum * 100
        suffix = f"{percentage:.1f}%"
        bars.append(
            f'<div class="bar-row"><span>{e(family)}</span><div class="bar-track">'
            f'<div class="bar-fill" style="width:{width:.3f}%;background:{TASK_FAMILY_COLORS[family]}"></div>'
            f'</div><b>{e(suffix)}</b></div>'
        )
    return '<div class="bar-list">' + "".join(bars) + "</div>"


def simple_task_portfolio_chart(rows: list[dict[str, str]]) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[row["simple_task_category"]] += int(row["annotation_unit_count"])
    colors = {
        "classification": "#247247", "description": "#28658c", "vqa": "#08786f",
        "grounding": "#4e78a0", "pointing": "#45a18f", "tracking": "#8b6f9e",
        "instruction": "#64717b", "planning": "#a33d36",
    }
    total = sum(counts.values()) or 1
    segments = []
    for task in SIMPLE_TASK_ORDER:
        count = counts[task]
        if not count:
            continue
        width = count / total * 100
        title = f"{SIMPLE_TASK_LABELS[task]}: {count:,} ({width:.2f}%)"
        segments.append(
            f'<i class="stack-seg" title="{e(title)}" '
            f'style="width:{width:.4f}%;background:{colors[task]}"></i>'
        )
    return (
        legend_html(SIMPLE_TASK_ORDER, SIMPLE_TASK_LABELS, colors)
        + f'<div class="stack-row"><span>抽样监督单元</span><div class="stack-track">'
        + "".join(segments)
        + f'</div><b>n={fmt(total)}</b></div>'
    )


def sampled_task_ledger_html(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        percentage = float(row["within_dataset_percentage"]) * 100
        body.append(
            f'<tr><td><a href="{quote(row["dataset"])}.html">{e(row["dataset"])}</a></td>'
            f'<td>{e(row["simple_task_label"])}</td><td>{e(row["task_family"])}</td>'
            f'<td>{e(row["fine_task"])}</td><td>{e(row["task_description"])}</td>'
            f'<td>{e(row["input"])}</td><td>{e(row["output"])}</td><td>{e(row["domain"])}</td>'
            f'<td><code>{e(row["source_task_label"])}</code></td>'
            f'<td>{int(row["annotation_unit_count"]):,}</td><td>{percentage:.2f}%</td>'
            f'<td>{e(row["example_sample_id"])}</td></tr>'
        )
    head = "".join(
        f"<th>{label}</th>" for label in [
            "数据集", "简单类别", "任务族", "细分任务", "任务详细描述", "输入", "输出",
            "领域", "源任务标签", "抽样标注单元", "数据集内占比", "样本 ID",
        ]
    )
    return f'<div class="table-wrap"><table class="wide-table task-ledger"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def full_task_ledger_html(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            f'<tr><td><a href="{quote(row["dataset"])}.html">{e(row["dataset"])}</a></td>'
            f'<td>{e(row["simple_task_label"])}</td><td>{e(row["task_family"])}</td>'
            f'<td>{e(row["fine_task"])}</td><td>{e(row["task_description"])}</td>'
            f'<td>{e(row["input"])}</td><td>{e(row["output"])}</td><td>{e(row["domain"])}</td>'
            f'<td><code>{e(row["task"])}</code></td><td>{int(row["count"]):,} {e(row["count_unit"])}</td>'
            f'<td>{float(row["percentage"])*100:.2f}%</td><td>{e(row["source_classification_basis"])}</td></tr>'
        )
    head = "".join(
        f"<th>{label}</th>" for label in [
            "数据集", "简单类别", "任务族", "细分任务", "任务详细描述", "输入", "输出",
            "领域", "源任务标签", "全量计数", "源分母占比", "源分类依据",
        ]
    )
    return f'<div class="table-wrap"><table class="wide-table task-ledger"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def paired_record_media_chart(metrics: dict[str, dict[str, Any]]) -> str:
    maximum = max(
        math.log10(max(metrics[name]["source_records"], metrics[name]["unique_media"]) + 1)
        for name in DATASET_ORDER
    )
    rows = []
    for dataset in DATASET_ORDER:
        records = metrics[dataset]["source_records"]
        media = metrics[dataset]["unique_media"]
        record_width = math.log10(records + 1) / maximum * 100
        media_width = math.log10(media + 1) / maximum * 100
        ratio = records / media if media else 0
        rows.append(
            f'<div class="paired-row"><span>{e(dataset)}</span><div class="paired-track">'
            f'<i title="canonical records {records:,}" style="width:{record_width:.3f}%;background:var(--teal)"></i>'
            f'<i title="unique media {media:,}" style="width:{media_width:.3f}%;background:var(--blue)"></i>'
            f'</div><b>{ratio:.1f}x</b></div>'
        )
    legend = '<div class="legend"><span><i class="swatch" style="background:var(--teal)"></i>canonical records</span><span><i class="swatch" style="background:var(--blue)"></i>unique media/lineage</span></div>'
    return legend + '<div class="stack-chart">' + "".join(rows) + "</div>"


def aggregate_length_chart(length_rows: list[dict[str, str]]) -> str:
    order = ["0", "1-16", "17-32", "33-64", "65-128", "129-256", "257-512", "513+"]
    colors = dict(zip(order, ["#d7dde0", "#c7dfd9", "#8fc6b9", "#4ba290", "#08786f", "#4e78a0", "#8b6f9e", "#a33d36"]))
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in length_rows:
        grouped[row["side"]][row["length_bin_characters"]] += int(row["sample_count"])
    labels = {key: key + " 字符" for key in order}
    body = []
    for side, side_label in (("input", "输入文本"), ("output", "输出文本")):
        total = sum(grouped[side].values()) or 1
        segments = []
        for key in order:
            count = grouped[side][key]
            width = count / total * 100
            segments.append(f'<i class="stack-seg" title="{e(labels[key])}: {count:,} ({width:.2f}%)" style="width:{width:.4f}%;background:{colors[key]}"></i>')
        body.append(f'<div class="stack-row"><span>{side_label}</span><div class="stack-track">{"".join(segments)}</div><b>n={fmt(total)}</b></div>')
    return legend_html(order, labels, colors) + '<div class="stack-chart">' + "".join(body) + "</div>"


def annotation_density_chart(rows: list[dict[str, str]]) -> str:
    order = ["0", "1", "2-4", "5-8", "9-16", "17-32", "33-64", "65+"]
    colors = dict(zip(order, ["#d7dde0", "#dff2ef", "#a9d7cc", "#63b4a1", "#08786f", "#4e78a0", "#8b6f9e", "#a33d36"]))
    labels = {key: key + " 监督" for key in order}
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row["dataset"]][row["annotation_count_bin"]] += int(row["sample_count"])
    body = []
    for dataset in DATASET_ORDER:
        total = sum(grouped[dataset].values()) or 1
        segments = []
        for key in order:
            count = grouped[dataset][key]
            width = count / total * 100
            segments.append(f'<i class="stack-seg" title="{e(labels[key])}: {count:,} ({width:.2f}%)" style="width:{width:.4f}%;background:{colors[key]}"></i>')
        body.append(f'<div class="stack-row"><span>{e(dataset)}</span><div class="stack-track">{"".join(segments)}</div><b>n={fmt(total)}</b></div>')
    return legend_html(order, labels, colors) + '<div class="stack-chart">' + "".join(body) + "</div>"


def quality_risk_matrix_html(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("media_completeness_risk", "媒体"), ("schema_annotation_risk", "标注/schema"),
        ("duplication_concentration_risk", "重复/集中"),
        ("license_commercial_risk", "许可/商用"),
    ]
    head = "".join(f"<th>{e(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = "".join(f'<td class="r{row[key]}">{row[key]}</td>' for key, _ in columns)
        body.append(f'<tr><th><a href="{quote(row["dataset"])}.html">{e(row["dataset"])}</a></th>{cells}<td>{rating_html(row["overall_rating"])}</td></tr>')
    return f'<div class="table-wrap"><table class="risk"><thead><tr><th>数据集</th>{head}<th>总评</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def overlap_matrix_html(matrix: list[dict[str, Any]]) -> str:
    head = "".join(f"<th>{e(name)}</th>" for name in DATASET_ORDER)
    body = []
    for row in matrix:
        cells = []
        for dataset in DATASET_ORDER:
            value = str(row[dataset])
            if value == "self":
                css, label = "ov-self", "self"
            elif value.startswith("full "):
                css = "ov-full"
                matches = re.findall(r":\s*([0-9,]+)", value)
                label = "/".join(matches) if matches else "full"
            else:
                match = re.search(r"sample SHA:\s*(\d+)/(\d+)", value)
                hit = int(match.group(1)) if match else 0
                css = "ov-sample-hit" if hit else "ov-sample-zero"
                label = match.group(1) if match else "n/a"
            cells.append(f'<td class="{css}" title="{e(value)}">{e(label)}</td>')
        body.append(f'<tr><th>{e(row["dataset"])}</th>{"".join(cells)}</tr>')
    return f'<div class="table-wrap"><table class="overlap"><thead><tr><th>left / right</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def capability_gap_chart(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        current = float(row["coverage_percent"])
        gap = 100 - current
        body.append(
            f'<div class="stack-row"><span>{e(row["embodied_level"] + " " + row["capability"])}</span>'
            f'<div class="stack-track"><i class="stack-seg" title="current {current:.1f}%" style="width:{current:.3f}%;background:var(--teal)"></i>'
            f'<i class="stack-seg" title="gap {gap:.1f}%" style="width:{gap:.3f}%;background:#f1cf91"></i></div><b>{current:.1f}%</b></div>'
        )
    return '<div class="legend"><span><i class="swatch" style="background:var(--teal)"></i>当前质量加权覆盖</span><span><i class="swatch" style="background:#f1cf91"></i>距 9 分目标缺口</span></div><div class="stack-chart">' + "".join(body) + "</div>"


def embodied_evidence_chart(rows: list[dict[str, Any]]) -> str:
    maximum = max(math.log10(float(row["quality_adjusted_evidence"]) + 1) for row in rows) or 1
    body = []
    for row in rows:
        value = float(row["quality_adjusted_evidence"])
        width = math.log10(value + 1) / maximum * 100
        body.append(
            f'<div class="bar-row"><span>{e(row["embodied_level"] + " " + row["capability"])}</span>'
            f'<div class="bar-track"><div class="bar-fill blue" style="width:{width:.3f}%"></div></div>'
            f'<b>{fmt(value)}</b></div>'
        )
    return '<div class="bar-list">' + "".join(body) + "</div>"


def dataset_page(
    dataset: str,
    data: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    task_rows: list[dict[str, Any]],
    qrows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    inventory_map: dict[str, dict[str, Any]],
    accounting_map: dict[str, dict[str, Any]],
    sample_evidence: dict[str, Any],
) -> str:
    meta = META[dataset]
    metric = metrics[dataset]
    summ = summary(dataset)
    schema = load_json(ROOT / "source-schemas" / f"{dataset}.json")
    expected = int(summ.get("selected_count", 0))

    local_tasks = [row for row in task_rows if row["dataset"] == dataset]
    local_sample_tasks = [row for row in sample_evidence["task_distribution"] if row["dataset"] == dataset]
    local_quality = [row for row in qrows if row["dataset"] == dataset]
    task_table = wide_table_html(
        [{**row, "percentage": f"{row['percentage']*100:.2f}%"} for row in local_tasks],
        [
            ("simple_task_label", "简单类别"), ("task_family", "任务族"),
            ("fine_task", "细分任务"), ("task_description", "任务详细描述"),
            ("input", "输入"), ("output", "输出"), ("task", "源任务标签"),
            ("count", "数量"), ("count_unit", "单位"), ("percentage", "源分母占比"),
            ("source_classification_basis", "源统计依据"),
        ],
    )
    sample_task_table = sampled_task_ledger_html(local_sample_tasks)
    sample_task_chart = task_family_portfolio_chart(local_sample_tasks, balanced=False)
    quality_table = table_html(
        [{**row, "rate": f"{row['rate']*100:.2f}%" if row["rate"] is not None else "not_evaluable"} for row in local_quality],
        [("issue", "问题"), ("observed", "观测"), ("denominator", "分母"),
         ("rate", "比例"), ("evidence", "证据"), ("required_action", "处理")],
    )
    inventory = inventory_map[dataset]
    accounting = accounting_map[dataset]
    source_table = table_html([{
        "version": inventory["source_version"],
        "splits": inventory["source_splits"],
        "formats": inventory["source_formats"],
        "files": inventory["file_count"],
        "bytes": fmt(inventory["source_bytes_observed"]),
        "storage": inventory["media_storage"],
        "identity": inventory["media_and_lineage_identity"].replace("lineage", "source group"),
        "locator": inventory["reproducible_locator"],
        "digest": inventory["source_population_digest"],
    }], [
        ("version", "版本"), ("splits", "源 split"), ("formats", "格式"), ("files", "文件数"),
        ("bytes", "已观察字节"), ("storage", "媒体存储"), ("identity", "媒体/来源 ID"),
        ("locator", "可复现定位"), ("digest", "源总体指纹"),
    ])
    accounting_table = table_html([accounting], [
        ("canonical_source_records", "canonical 记录"), ("confirmed_effective_records", "确认有效"),
        ("known_rejected_records", "确认拒绝"), ("unresolved_records", "待处理/不可确认"),
        ("accounting_check", "对账"), ("status", "状态"),
    ])

    media = sample_evidence["media"][dataset]
    media_table = table_html([media], [
        ("sample_total", "样本 n"), ("decoded_or_inspected_media", "已检查媒体"),
        ("media_status", "媒体状态"), ("format_counts", "格式"),
        ("width_median", "宽 p50"), ("width_p90", "宽 p90"),
        ("height_median", "高 p50"), ("height_p90", "高 p90"),
        ("duration_seconds_median", "时长 p50(s)"), ("duration_seconds_p90", "时长 p90(s)"),
        ("fps_median", "fps p50"), ("frames_median", "帧数 p50"),
    ])
    lengths = sample_evidence["lengths"][dataset]
    length_table = table_html(lengths, [
        ("side", "文本侧"), ("sample_total", "n"), ("mean_characters", "字符均值"),
        ("median_characters", "p50"), ("p90_characters", "p90"), ("max_characters", "最大"),
    ])
    text_quality = sample_evidence["text_quality"][dataset]
    density = sample_evidence["density"][dataset]
    sample_quality_table = table_html([{
        "empty_input": text_quality["empty_input_count"],
        "empty_output": text_quality["empty_output_count"],
        "unique_input": f"{float(text_quality['unique_input_rate'])*100:.1f}%",
        "unique_output": f"{float(text_quality['unique_output_rate'])*100:.1f}%",
        "entropy": text_quality["output_entropy_bits"],
        "density_p50": density["median_annotations"],
        "density_p90": density["p90_annotations"],
        "density_max": density["max_annotations"],
        "measure": density["measure_notes"],
    }], [
        ("empty_input", "空输入"), ("empty_output", "空输出"),
        ("unique_input", "唯一输入率"), ("unique_output", "唯一输出率"),
        ("entropy", "输出熵 bits"), ("density_p50", "监督/媒体 p50"),
        ("density_p90", "监督/媒体 p90"), ("density_max", "监督/媒体 max"),
        ("measure", "密度口径"),
    ])

    local_strata = [row for row in sample_evidence["strata"] if row["dataset"] == dataset]
    local_strata.sort(key=lambda row: (row["dimension"], -int(row["sample_count"]), row["value"]))
    strata_table = table_html(
        [
            {
                **row,
                "value": row["value"].replace("lineage", "source_group"),
                "proportion": f"{float(row['proportion'])*100:.2f}%",
            }
            for row in local_strata[:24]
        ],
        [("dimension", "分层维度"), ("value", "源值"), ("sample_count", "样本数"),
         ("dimension_total", "维度总计"), ("proportion", "比例")],
    )
    manifest_name = "sampling-manifest.jsonl" if expected else "record-sampling-manifest.jsonl"
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)
    sample_section = sample_evidence["sample_sections"][dataset]

    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(dataset)} 原始源数据分析</title><style>{STYLE}{sample_evidence["sample_style"]}</style></head><body><main data-dataset="{e(dataset)}">
<header><div class="eyebrow">SOURCE DATA TASK AUDIT</div><h1>{e(dataset)}</h1><p class="lede"><strong>数据定位：</strong>{e(meta['domain'])}</p></header>
<nav class="nav"><a href="index.html">总览</a><a href="#contract">输入输出</a><a href="#tasks">任务分布</a><a href="#population">规模质量</a><a href="#schema">Schema</a><a href="#samples">抽样媒体与 QA</a></nav>
<section class="facts"><div class="fact"><b>{fmt(metric['source_records'])}</b><span>canonical 源记录</span></div><div class="fact"><b>{fmt(metric['unique_media'])}</b><span>唯一媒体</span></div><div class="fact"><b>{fmt(metric['supervision_units'])}</b><span>监督单元</span></div><div class="fact"><b>{rating_html(meta['rating'])}</b><span>源训练可用评级</span></div></section>
<section id="contract"><h2>一、输入、输出与源存储</h2><div class="band"><div class="flow"><span>{e(meta['input'])}</span><b>→</b><span>{e(meta['output'])}</span></div></div><div class="two"><div><h3>任务与领域</h3><p><strong>{e(meta['simple_task'])}</strong><br>{e(meta['domain'])}</p><p class="mini">源路径：<code>{e(inventory['source_path'])}</code></p></div><div><h3>具身判定</h3><p><strong>{e(meta['embodied'])}</strong>：{e(meta['embodied_evidence'])}</p><p class="mini">许可：{e(meta['license'])}</p></div></div>{source_table}<p class="note"><strong>证据边界：</strong>统计、schema、样本与质量结论只来自原始源文件。媒体不可取得时明确标为 incomplete，不用其他数据补齐。</p></section>
<section id="population"><h2>二、全量口径与质量</h2><div class="facts"><div class="fact"><b>{fmt(metric['effective_units'])}</b><span>源证据候选量：{e(metric['effective_unit_type'])}</span></div><div class="fact"><b>{fmt(metric['token_estimate'])}</b><span>监督 token 估计</span></div><div class="fact"><b>{fmt(metric['annotations_per_media'],2)}</b><span>全量监督/媒体均值</span></div><div class="fact"><b>{fmt(summ.get('seed'))}</b><span>确定性样本 seed</span></div></div><h3>canonical 记录对账</h3>{accounting_table}<p class="provenance">全量证据：{e(metric['evidence'])}；源路径 <code>{e(inventory['source_path'])}</code>；口径时间 {ANALYSIS_DATE}。确认有效 + 确认拒绝 + 待处理/不可确认必须等于 canonical 源记录。</p>{quality_table}<h3>确定性样本媒体与文本质量</h3>{media_table}{length_table}{sample_quality_table}<p class="chart-note">以上为归档源样本估计，n={fmt(expected or int(summ.get('record_sample_count', 0)))}；媒体缺失时显示 not_evaluable，不外推为全量媒体质量。</p></section>
<section id="tasks"><h2>三、数据本身的任务分类与分布</h2><h3>200 条抽样中实际展开的 QA / 标注任务族</h3><p class="chart-note">Deterministic source sample · 每条展开的 QA/标注算一个单元；同一媒体可以贡献多个任务。柱长按本数据集抽样标注单元占比。</p>{sample_task_chart}<h3>抽样任务明细</h3>{sample_task_table}<p><a href="sample-task-annotations.csv">逐标注任务底表</a> · <a href="sample-task-distribution.csv">聚合任务分布</a></p><h3>服务器全量源任务统计</h3><p class="chart-note">Full source evidence · 优先使用源 task/category/type 字段；缺字段时使用版本化文本规则。计数单位见表，不能把 label、QA、point、track 直接混加为训练样本数。</p>{task_table}<h3>能力强度矩阵（辅助视图）</h3>{capability_table(dataset)}<h3>样本 split / 源字段 / 媒体覆盖审计</h3>{strata_table}<p class="chart-note">视觉归档遵循完整总体固定 seed 无放回排列；本表是源字段覆盖审计，不代替上方统一任务 taxonomy。超过 24 个 strata 时完整结果见 sampling-strata.csv。</p></section>
<section id="schema"><h2>四、原始 source schema</h2><p><a href="source-schemas/{quote(dataset)}.json">完整 schema</a> · <a href="sub-dataset/{quote(dataset)}/source-layout.txt">源布局</a> · <a href="sub-dataset/{quote(dataset)}/sampling-summary.json">抽样摘要</a> · <a href="sub-dataset/{quote(dataset)}/{manifest_name}">manifest</a>。schema 含实际字段、结构变体与至少 3 条脱敏真实记录。</p><pre class="schema">{e(schema_text)}</pre></section>
<section id="samples"><h2>五、200 条抽样媒体与 QA</h2><p class="chart-note">Deterministic source sample · 按固定 draw order 展示本数据集全部 200 条抽样记录。图片和视频来自原始源归档；无可验证媒体字节时保留完整 QA 并明确标记媒体不可用。</p>{sample_section}</section>
<footer>分析时间 {ANALYSIS_DATE}。源访问只读；最终 before/after 指纹见 <a href="source-readonly-evidence.json">source-readonly-evidence.json</a>。</footer></main></body></html>'''


def legacy_index_page(data: dict[str, Any], metrics: dict[str, dict[str, Any]], qrows: list[dict[str, Any]], matrix: list[dict[str, Any]], edge_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]], recipes: list[dict[str, Any]], recommendations: list[dict[str, Any]]) -> str:
    total_records = sum(row["source_records"] for row in metrics.values())
    direct = sum(META[name]["embodied"] == "direct" for name in DATASET_ORDER)
    quality_summary = Counter(META[name]["rating"] for name in DATASET_ORDER)
    dataset_rows = []
    for dataset in DATASET_ORDER:
        m = metrics[dataset]
        dataset_rows.append({"dataset": dataset, "task": META[dataset]["simple_task"], "domain": META[dataset]["domain"], "records": fmt(m["source_records"]), "media": fmt(m["unique_media"]), "density": fmt(m["annotations_per_media"],2), "embodied": META[dataset]["embodied"], "rating": META[dataset]["rating"]})
    dataset_table = table_html(dataset_rows, [("dataset", "数据集"), ("task", "Task"), ("domain", "领域"), ("records", "源记录"), ("media", "唯一媒体"), ("density", "监督/媒体"), ("embodied", "具身"), ("rating", "评级")])
    embodied_scores = [("E0 通用感知",95,"多源充分"),("E1 空间/定位",90,"多源充分但有泄露"),("E2 时序/动作",62,"集中在视频源"),("E3 可供性/状态",48,"主要依赖 RoboVQA/Robo2VLM"),("E4 成功/下一步",51,"来源集中"),("E5 子目标/长程规划",28,"文本计划多，真实轨迹少"),("E6 导航/记忆/安全",4,"近乎空白")]
    embodied_bars = "".join(f'<div class="bar-row"><span>{e(name)}</span><div class="bar-track"><div class="bar-fill {"amber" if value<50 else "blue"}" style="width:{value}%"></div></div><b>{value}/100</b></div>' for name,value,_ in embodied_scores)
    split_bad = [row for row in split_rows if row["intersection"]]
    split_table = table_html(split_bad, [("dataset", "数据集"), ("left_split", "split A"), ("right_split", "split B"), ("intersection", "交集"), ("namespace", "证据"), ("interpretation", "解释")])
    edge_table = table_html(edge_rows, [("left", "左源"), ("right", "右源"), ("intersection", "交集"), ("left_overlap_rate", "左侧比例"), ("namespace", "共享命名空间"), ("relationship", "关系")])
    recipe_table = table_html(recipes, [("recipe", "方案"), ("stage", "阶段"), ("total_mix_percent", "总量 %"), ("capability_target", "能力"), ("datasets", "数据"), ("per_media_cap", "媒体限额"), ("entry_and_quota_rule", "进入规则")])
    rec_table = table_html(recommendations, [("dataset", "推荐数据"), ("modality", "模态"), ("scene", "场景"), ("target_capability", "目标能力"), ("scale", "规模"), ("license", "许可"), ("benchmark_contamination_risk", "污染风险"), ("priority", "优先级")])
    quality_dataset_rows = [{"dataset": name, "rating": META[name]["rating"], "summary": META[name]["judgment"]} for name in DATASET_ORDER]
    quality_table = table_html(quality_dataset_rows, [("dataset", "数据集"), ("rating", "评级"), ("summary", "结论")])
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VLM 原始源数据能力与具身训练分析</title><style>{STYLE}</style></head><body><main><header><div class="eyebrow">SOURCE-ONLY / FULL-POPULATION + DETERMINISTIC EVIDENCE</div><h1>VLM 原始源数据能力分析</h1><p class="lede">18 类源数据的规模、schema、任务能力、质量、lineage 泄露与训练配比。目标是保持通用视觉能力，同时把有效监督逐步推向可供性、状态、未来动作与长程具身规划。</p></header><nav class="nav"><a href="#inventory">总表</a><a href="#distribution">分布</a><a href="#embodied">具身</a><a href="#leakage">泄露</a><a href="#quality">质量</a><a href="#recipes">训练</a><a href="#recommend">补充数据</a><a href="methodology.md">方法</a></nav><section class="facts"><div class="fact"><b>18</b><span>纳入数据集</span></div><div class="fact"><b>{fmt(total_records)}</b><span>canonical 源记录（不可直接相加训练）</span></div><div class="fact"><b>{direct}</b><span>直接具身监督数据集</span></div><div class="fact"><b>{quality_summary.get('Q',0)}</b><span>媒体未闭环、需隔离</span></div></section><section><h2>核心判断</h2><div class="three"><div class="note"><h3>能力结构</h3><p>通用 VQA、描述、grounding 和空间能力规模充足；直接具身监督只集中在 Robo2VLM 与 RoboVQA，E6 导航、环境记忆和安全约束几乎空白。</p></div><div class="warning"><h3>规模错觉</h3><p>问答行数显著放大视觉多样性：Robo2VLM 平均 44.9 条/scene，VLM-R1 平均 11.4 条/图，PixMo-Points 平均 10.4 条/图。配比必须以媒体、监督和 token 三轴同时限额。</p></div><div class="danger"><h3>泄露与不可用</h3><p>Robo2VLM test scene 100% 进入 train；LLaVA 完整复用多个 benchmark 源；3 个 Molmo2 视频源缺少可验证媒体。沿用源行级 split 会高估泛化。</p></div></div></section><section id="inventory"><h2>一、数据集总表</h2>{dataset_table}<p class="mini">每个名称链接到独立报告：{' · '.join(f'<a href="{quote(name)}.html">{e(name)}</a>' for name in DATASET_ORDER)}</p></section><section id="distribution"><h2>二、源规模与媒体多样性</h2><div class="two"><div><h3>canonical 源记录（log10）</h3><div class="bar-list">{bar_rows(metrics,'source_records')}</div></div><div><h3>唯一媒体/lineage（log10）</h3><div class="bar-list">{bar_rows(metrics,'unique_media','blue')}</div></div></div><h3>数据集 × 细粒度能力热力图</h3><p>0=无直接证据，1=少量/间接，2=显著，3=核心。评分基于实际 schema 与任务分布，不按数据集名称推断。</p>{heatmap()}</section><section id="embodied"><h2>三、具身能力覆盖与缺口</h2><div class="two"><div><div class="bar-list">{embodied_bars}</div></div><div><h3>为什么重点仍不足</h3><p>RoboVQA 提供 affordance、success、future 与 1/5-step planning；Robo2VLM 提供几何、轨迹、障碍和目标状态。但二者高度模板化、缺少独立验证 split，也没有连续低层 action/state 轨迹、主动探索、环境记忆与失败恢复闭环。</p><p class="warning"><strong>结论：</strong>继续增加通用 QA 行数不会自动提高具身推理。下一批数据应优先增加真实 episode、状态-动作对、失败恢复和导航记忆。</p></div></div></section><section id="leakage"><h2>四、跨源重叠与 split 泄露</h2><h3>全量共享 lineage</h3>{edge_table}<h3>源 split 媒体交集</h3>{split_table}<p><a href="source-overlap-matrix.csv">下载 18×18 overlap matrix</a> · <a href="split-leakage-matrix.csv">下载 split audit</a></p></section><section id="quality"><h2>五、质量与训练可用性</h2><p>A 可直接候选；B 可用但需限额；C 必须清洗/重切分；Q 在媒体或 lineage 闭环前隔离。</p>{quality_table}<p><a href="quality-audit.csv">查看逐问题、分母、比例和处理动作</a></p></section><section id="recipes"><h2>六、三套分阶段训练方案</h2><p>比例是总有效监督预算的目标占比，而不是从物理行数直接抽样。每阶段同时约束 unique media、有效监督、token 和单媒体上限。</p>{recipe_table}<p><a href="training-recipes.csv">打开含目标规模、token 范围和 benchmark 的完整配方</a></p></section><section id="recommend"><h2>七、推荐补充数据</h2><p>优先级由 E3-E6 缺口决定。当前报告未下载这些外部数据；规模、许可和可获取性中未现场核验的字段明确标为待核验。benchmark 类只用于评估。</p>{rec_table}<p><a href="recommended-datasets.csv">打开完整推荐表</a></p></section><footer>生成时间 {ANALYSIS_DATE}；唯一服务器源根：<code>{SOURCE_ROOT}</code>。所有源数据只读证据见 <a href="source-readonly-evidence.json">source-readonly-evidence.json</a>。</footer></main></body></html>'''


def citation_links(reference_ids: Iterable[int]) -> str:
    return '<span class="citations">' + "".join(
        f'<a href="#ref-{int(reference_id)}">[{int(reference_id)}]</a>'
        for reference_id in reference_ids
    ) + "</span>"


def training_references_html(references: list[dict[str, Any]]) -> str:
    items = []
    for reference in references:
        citation_id = int(reference["citation_id"])
        items.append(
            f'<li id="ref-{citation_id}"><div class="reference-title">'
            f'<a href="{e(reference["url"])}">{e(reference["title"])}</a></div>'
            f'<div class="reference-meta">{e(reference["authors"])} · {e(reference["year"])} · '
            f'arXiv:{e(reference["arxiv_id"])}</div>'
            f'<p><strong>本方案采用的证据：</strong>{e(reference["supported_principle"])}</p>'
            f'<p><strong>对应位置：</strong>{e(reference["used_in"])}</p></li>'
        )
    return (
        '<h3>参考文献与方案映射</h3>'
        '<p class="reference-boundary"><strong>引用边界：</strong>下列论文支持分阶段训练、数据混合、空间/视频迁移、'
        '通用与机器人任务联合训练及轨迹/动作/规划监督等设计原则；三套方案的具体百分比、媒体上限和 token 范围'
        '由本报告的数据分布与能力缺口推导，不是论文复刻参数。</p>'
        f'<ol class="reference-list">{"".join(items)}</ol>'
        f'<p class="chart-note">论文元数据与摘要通过 arXiv official API 核验，日期 {ANALYSIS_DATE}。'
        '<a href="training-references.csv">下载参考文献与证据映射 CSV</a>。</p>'
    )


def training_recipe_analysis_html(
    recipes: list[dict[str, Any]],
    sample_task_rows: list[dict[str, str]],
    gap_rows: list[dict[str, Any]],
    sample_record_count: int,
) -> str:
    total_units = sum(int(row["annotation_unit_count"]) for row in sample_task_rows)
    simple_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for row in sample_task_rows:
        count = int(row["annotation_unit_count"])
        simple_counts[row["simple_task_category"]] += count
        family_counts[row["task_family"]] += count

    def share(count: int) -> float:
        return count / total_units * 100 if total_units else 0.0

    vqa_share = share(simple_counts["vqa"])
    planning_share = share(simple_counts["planning"])
    spatial_localization_share = share(
        family_counts["空间与三维几何"] + family_counts["视觉定位与指点"]
    )
    supervision_density = total_units / sample_record_count if sample_record_count else 0.0
    e6_coverage = next(
        (float(row["coverage_percent"]) for row in gap_rows if row["embodied_level"] == "E6"),
        0.0,
    )

    basis_items = [
        ("监督结构", f"VQA 占 {vqa_share:.1f}%，planning 仅 {planning_share:.1f}%；通用问答充分，但行动规划监督偏少。", [2, 6, 7]),
        ("空间前置", f"空间/三维与定位/指点合计 {spatial_localization_share:.1f}%，应独立安排 grounding 与空间阶段，而不是混入通用 VQA。", [3, 4]),
        ("媒体密度", f"{total_units:,} 个监督单元来自 {sample_record_count:,} 条抽样记录，平均 {supervision_density:.2f} 单元/记录；训练需设置单媒体上限。", [2]),
        ("具身缺口", f"E6 导航、记忆、恢复与安全覆盖为 {e6_coverage:.1f}%；仅重复现有 QA 不能补齐，需要引入轨迹与环境交互数据。", [7, 8, 9, 10]),
    ]
    basis_html = "".join(
        f'<div><b>{e(title)}</b><p>{e(description)} {citation_links(reference_ids)}</p></div>'
        for title, description, reference_ids in basis_items
    )

    plan_meta = [
        {
            "name": "通用能力保持",
            "class": "general",
            "label": "稳健保持",
            "summary": "优先控制通用视觉、OCR、图表与描述能力回退，再逐步加入空间和具身监督。",
            "basis": f"抽样监督中 VQA 已占 {vqa_share:.1f}%，但它也是基座能力的主要回放来源；Stage 1 与 Stage 5 合计 54%，用于稳定输出格式和通用能力。",
            "scenario": "适合首次继续训练、通用 benchmark 已出现退化，或新增具身数据尚未完成大规模轨迹补充时。",
            "tradeoff": "遗忘风险最低，训练最稳；Stage 4 仅 10%，可供性、下一步动作和长程规划提升速度最慢。",
            "references": [1, 2, 6],
        },
        {
            "name": "通用-具身均衡",
            "class": "balanced",
            "label": "默认推荐",
            "summary": "在保留通用能力的同时，提高空间、视频动作、状态判断与任务规划的有效占比。",
            "basis": f"空间与定位相关任务已有 {spatial_localization_share:.1f}% 的抽样覆盖，而 planning 仅 {planning_share:.1f}%；因此 Stage 2 提到 25%，Stage 4 提到 20%，并保留 37% 通用与混合回放。",
            "scenario": "适合只有一轮主要训练预算、既要维持综合 VLM benchmark，又要明显改善具身推理的默认实验。",
            "tradeoff": "能力覆盖最均衡，便于定位每阶段收益；训练周期较长，需要阶段性 checkpoint 和通用/具身双门槛评估。",
            "references": [3, 4, 5, 6, 7],
        },
        {
            "name": "具身推理优先",
            "class": "embodied",
            "label": "目标强化",
            "summary": "把主要预算投入视频时序、动作结果、可供性、下一步预测与长程规划。",
            "basis": f"planning 仅占 {planning_share:.1f}%，E6 覆盖为 {e6_coverage:.1f}%；Stage 3 与 Stage 4 合计 50%，用于主动修正当前行动与规划监督不足。",
            "scenario": "适合基座通用能力已经稳定，并且已经补充真实 state-action trajectory、失败恢复或导航记忆数据的专项训练。",
            "tradeoff": "具身增益潜力最高，但通用能力遗忘和来源集中风险也最高；缺少新增轨迹时，不应仅靠重复 RoboVQA/Robo2VLM 放大到该比例。",
            "references": [7, 8, 9, 10],
        },
    ]
    stage_colors = {
        "Stage 1": "#247247",
        "Stage 2": "#45a18f",
        "Stage 3": "#bd6d41",
        "Stage 4": "#a33d36",
        "Stage 5": "#4e78a0",
    }
    plans = []
    for index, meta in enumerate(plan_meta, 1):
        stage_rows = sorted(
            (row for row in recipes if row["recipe"] == meta["name"]),
            key=lambda row: int(str(row["stage"]).split()[-1]),
        )
        mix_segments = "".join(
            f'<i title="{e(row["stage"])}: {int(row["total_mix_percent"])}%" '
            f'style="width:{int(row["total_mix_percent"])}%;background:{stage_colors[row["stage"]]}"></i>'
            for row in stage_rows
        )
        mix_legend = "".join(
            f'<span><i style="background:{stage_colors[row["stage"]]}"></i>'
            f'{e(row["stage"])} {int(row["total_mix_percent"])}%</span>'
            for row in stage_rows
        )
        display_stage_rows = [
            {**row, "reference_ids": " ".join(f'[{value}]' for value in str(row["reference_ids"]).split(";"))}
            for row in stage_rows
        ]
        stage_table = wide_table_html(display_stage_rows, [
            ("stage", "阶段"), ("total_mix_percent", "占比 %"),
            ("capability_target", "能力目标"), ("datasets", "候选数据"),
            ("unique_media_target", "唯一媒体范围"),
            ("effective_supervision_target", "有效监督范围"),
            ("token_target", "token 范围"), ("per_media_cap", "单媒体上限"),
            ("entry_and_quota_rule", "进入条件/配额"),
            ("validation_benchmarks", "阶段验证"),
            ("reference_ids", "参考文献"),
        ])
        plans.append(f'''<section class="recipe-plan recipe-{meta['class']}">
<div class="recipe-head"><div><div class="recipe-kicker">方案 {index} · {e(meta['label'])}</div><h3>{e(meta['name'])}</h3><p>{e(meta['summary'])} {citation_links(meta['references'])}</p></div><div><div class="mix-track">{mix_segments}</div><div class="mix-legend">{mix_legend}</div></div></div>
<div class="recipe-analysis"><div><h4>配比依据</h4><p>{e(meta['basis'])}</p></div><div><h4>适用场景</h4><p>{e(meta['scenario'])}</p></div><div><h4>收益与代价</h4><p>{e(meta['tradeoff'])}</p></div></div>
<h4>五阶段配置</h4>{stage_table}</section>''')

    return (
        '<h3>配比依据</h3><p class="chart-note">比例以总有效监督预算为分母，依据本报告的任务分布、媒体密度与 E0-E6 能力缺口，不按物理行数直接归一。</p>'
        f'<div class="basis-grid">{basis_html}</div>'
        + "".join(plans)
        + training_references_html(training_reference_rows())
    )


def index_page(
    metrics: dict[str, dict[str, Any]],
    task_rows: list[dict[str, Any]],
    qrows: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    recipes: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    accounting_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    embodied_rows: list[dict[str, Any]],
    sample_evidence: dict[str, Any],
) -> str:
    total_records = sum(row["source_records"] for row in metrics.values())
    total_media = sum(row["unique_media"] for row in metrics.values())
    direct = sum(META[name]["embodied"] == "direct" for name in DATASET_ORDER)
    q_count = sum(META[name]["rating"] == "Q" for name in DATASET_ORDER)
    unresolved = sum(row["unresolved_records"] for row in accounting_rows)
    sample_task_rows = sample_evidence["task_distribution"]
    sample_unit_count = sum(int(row["annotation_unit_count"]) for row in sample_task_rows)
    sample_page_count = int(sample_evidence["detail_summary"]["sample_records"])
    observed_family_count = len({row["task_family"] for row in sample_task_rows})
    observed_fine_task_count = len({(row["task_family"], row["fine_task"]) for row in sample_task_rows})

    accounting_map = {row["dataset"]: row for row in accounting_rows}
    dataset_rows = []
    for dataset in DATASET_ORDER:
        metric = metrics[dataset]
        dataset_rows.append({
            "dataset": dataset,
            "task": META[dataset]["simple_task"],
            "domain": META[dataset]["domain"],
            "records": fmt(metric["source_records"]),
            "media": fmt(metric["unique_media"]),
            "scene": fmt(metric["unique_scene"]),
            "supervision": fmt(metric["supervision_units"]),
            "tokens": fmt(metric["token_estimate"]),
            "effective": fmt(accounting_map[dataset]["confirmed_effective_records"]),
            "unresolved": fmt(accounting_map[dataset]["unresolved_records"]),
            "rating": META[dataset]["rating"],
        })
    dataset_table = dataset_inventory_table_html(dataset_rows, [
        ("dataset", "数据集"), ("task", "Task"), ("domain", "领域"),
        ("records", "canonical 记录"), ("media", "唯一媒体"), ("scene", "scene/episode"),
        ("supervision", "监督单元"), ("tokens", "token 估计"),
        ("effective", "确认有效记录"), ("unresolved", "待处理记录"), ("rating", "评级"),
    ])

    gap_table = table_html([
        {**row, "coverage_percent": f'{float(row["coverage_percent"]):.1f}%'}
        for row in gap_rows
    ], [
        ("embodied_level", "层级"), ("capability", "能力"),
        ("quality_adjusted_source_points", "当前点数"), ("target_points", "目标点数"),
        ("coverage_percent", "覆盖 %"), ("qualified_contributors", "可用来源"),
        ("quarantined_contributors", "隔离来源"),
    ])
    recipe_analysis = training_recipe_analysis_html(
        recipes, sample_task_rows, gap_rows, sample_page_count
    )
    rec_table = wide_table_html(recommendations, [
        ("dataset", "推荐数据"), ("modality", "模态"), ("scene", "场景"), ("task", "任务"),
        ("target_capability", "目标能力"), ("scale", "规模"),
        ("has_action_trajectory_state", "动作/轨迹/状态"), ("license", "许可"),
        ("availability", "可获取性"),
        ("preprocessing_cost", "预处理成本"), ("priority", "优先级"),
    ])

    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLM 原始源数据任务、分布与训练分析</title><style>{STYLE}</style></head><body><main>
<header><div class="eyebrow">SOURCE-ONLY / TASK-FIRST DATA AUDIT</div><h1>VLM 原始源数据任务分析</h1><p class="lede">先回答 18 类源数据实际包含哪些任务、输入、输出和领域，再分析规模及训练配比。唯一服务器证据根为 <code>{SOURCE_ROOT}</code>；不读取转换后数据。</p></header>
<nav class="nav"><a href="#inventory">总表</a><a href="#tasks">任务分布</a><a href="#distribution">全量规模</a><a href="#embodied">具身子集</a><a href="#recipes">训练</a><a href="#recommend">补充数据</a><a href="methodology.md">方法</a></nav>
<section class="facts"><div class="fact"><b>18</b><span>纳入源数据集</span></div><div class="fact"><b>{fmt(sample_page_count)}</b><span>固定 seed 随机抽检记录</span></div><div class="fact"><b>{fmt(sample_unit_count)}</b><span>已展开并分类的 QA / 标注单元</span></div><div class="fact"><b>{fmt(total_records)}</b><span>canonical 源记录求和；不可直接当训练量</span></div></section>
<section><h2>核心判断</h2><div class="three"><div class="note"><h3>任务组成</h3><p>源数据不只包含具身任务，还覆盖对象分类、通用与组合 VQA、OCR、图表/科学图解、长描述、指令跟随、grounding、pointing、空间几何、视频时序、动作交互和 tracking。下方按真实源标签与抽样任务统计逐项展开。</p></div><div class="warning"><h3>分布口径</h3><p>同一媒体可带多个任务，且不同源的计数单位包括 label、QA、point 和 track。报告以实际展开的 QA/标注单元统计总体任务组成，并通过逐数据集堆叠图展示各数据源内部的任务分布。</p></div><div class="danger"><h3>训练约束</h3><p>{q_count} 个 Molmo2 视频源当前缺少完整媒体，部分来源还存在格式不一致或高重复。训练配比必须同时约束唯一媒体、有效监督、token 和单媒体重复次数。</p></div></div><p class="provenance">时间 {ANALYSIS_DATE}；全量数字来自原始 JSON/Parquet/Arrow/ZIP/TAR 扫描，抽样任务来自 {fmt(sample_page_count)} 条固定 seed 源记录中实际展开的 {fmt(sample_unit_count)} 个 QA/标注单元。每个任务都保留来源数据集、源任务标签和分类依据。</p></section>
<section id="inventory"><h2>一、数据集总表</h2>{dataset_table}<p><a href="inventory.csv">完整 inventory（版本、split、格式、文件数/字节、媒体存储、identity、locator、指纹）</a> · <a href="source-accounting.csv">canonical 记录对账</a></p></section>
<section id="tasks"><h2>二、数据本身的任务分类与分布</h2><section class="facts"><div class="fact"><b>{fmt(sample_unit_count)}</b><span>抽样 QA / 标注单元</span></div><div class="fact"><b>{observed_family_count}</b><span>实际出现的统一任务族</span></div><div class="fact"><b>{observed_fine_task_count}</b><span>统一细分任务</span></div><div class="fact"><b>{len(task_rows)}</b><span>服务器全量源任务统计行</span></div></section><h3>简单任务形态分布</h3><p class="chart-note">Deterministic source sample · n={fmt(sample_unit_count)} QA/标注单元。classification、description、VQA、grounding、pointing、tracking、instruction、planning 是输出契约层的简单分类。</p>{simple_task_portfolio_chart(sample_task_rows)}<h3>标注单元加权任务族</h3><p class="chart-note">每个实际展开的 QA/标注权重相同，反映本次抽检统计中的监督单元组成；高密度媒体权重更大。</p>{task_family_portfolio_chart(sample_task_rows, balanced=False)}<h3>每个数据集的抽样任务族堆叠图</h3><p class="chart-note">每行按该数据集抽样中实际展开的 QA/标注单元归一。同一媒体中的多轮问答和多个任务分别计数。</p>{sampled_task_family_stack_chart(sample_task_rows)}<h3>抽样任务详细台账</h3><p>逐项列出任务详细描述、输入、输出、领域、源任务标签、数量和数据集内占比。</p>{sampled_task_ledger_html(sample_task_rows)}<p><a href="sample-task-annotations.csv">下载 {fmt(sample_unit_count)} 条逐标注分类</a> · <a href="sample-task-distribution.csv">下载聚合任务分布</a></p></section>
<section id="distribution"><h2>三、服务器全量源任务与规模</h2><h3>全量源任务族分布</h3><p class="chart-note">Full source evidence · 每数据集内部归一；优先使用源 task/category/type 字段，缺失时才使用版本化规则。底层计数单位可能是 label、QA、point 或 track，不能跨数据集直接相加。</p>{task_stack_chart(task_rows)}<h3>全量源任务详细台账</h3>{full_task_ledger_html(task_rows)}<h3>数据集源规模柱状图</h3><div class="two"><div><p class="chart-note">Full population · canonical source records · log10</p><div class="bar-list">{bar_rows(metrics,'source_records')}</div></div><div><p class="chart-note">Full population · unique media · log10</p><div class="bar-list">{bar_rows(metrics,'unique_media','blue')}</div></div></div><h3>记录数与唯一媒体数对比</h3><p class="chart-note">Full population · 双条均为 log10，右侧为 records/media。</p>{paired_record_media_chart(metrics)}<h3>数据集 × 能力强度热力图（辅助视图）</h3><p class="chart-note">Observed schema + source task evidence；0=无直接证据，1=少量/间接，2=显著，3=核心。该矩阵只用于能力覆盖概览，不替代上方任务计数。</p>{heatmap()}<h3>每媒体标注数量分布</h3><p class="chart-note">Deterministic archived source sample estimate · n=200/数据集；字段不能给出完整媒体监督数时明确退化为单行监督，具体口径见 sample-annotation-density-summary.csv。</p>{annotation_density_chart(sample_evidence['density_rows'])}<h3>输入输出长度分布</h3><p class="chart-note">Deterministic archived source sample estimate · input/output 各 n=3,600；按 manifest 的真实源输入输出预览字符数分箱。</p>{aggregate_length_chart(sample_evidence['length_rows'])}<p><a href="task-distribution.csv">全量任务底表</a> · <a href="sampling-strata.csv">源字段覆盖</a> · <a href="sample-length-distribution.csv">长度 bins</a> · <a href="sample-annotation-density.csv">监督密度 bins</a></p></section>
<section id="embodied"><h2>四、具身任务子集与目标缺口</h2><p>本节只是在完整任务 taxonomy 上切出具身相关监督，不代表其他数据任务被降级或忽略。</p><div class="two"><div><h3>具身相关监督证据分布</h3><p class="chart-note">Full task evidence · Q 级来源权重为 0；横轴为质量加权证据量 log10，多标签可超过 canonical 行数。</p>{embodied_evidence_chart(embodied_rows)}</div><div><h3>当前覆盖与目标差距</h3><p class="chart-note">每数据集取该层最高 0-3 能力强度 × 质量权重；目标为 3 个相互独立、核心强度来源，即 9 点。公式见 capability-gap.csv。</p>{capability_gap_chart(gap_rows)}</div></div>{gap_table}<p class="warning">当前 E3-E5 主要由 Robo2VLM/RoboVQA 两个 C 级且有格式、媒体或重复问题的来源贡献；E6 为 0。通用 QA 可以作为基础回放，但不能替代状态-动作轨迹、主动探索、记忆或恢复闭环。</p></section>
<section id="recipes"><h2>五、三套五阶段训练候选方案</h2><p>三套方案分别面向通用能力保持、通用与具身均衡、具身推理强化。比例按总有效监督预算分配，不执行数据转换或训练；每阶段同时约束唯一媒体、有效监督、token、单媒体上限和验证门槛。</p>{recipe_analysis}<p><a href="training-recipes.csv">完整训练配方 CSV</a> · <a href="sample-task-distribution.csv">抽样任务依据</a> · <a href="capability-gap.csv">E0-E6 缺口依据</a></p></section>
<section id="recommend"><h2>六、推荐补充数据</h2><p>E3-E6 是具身目标下的优先缺口，同时应按上方完整任务分布保持通用感知、文字、图表、描述、定位、空间和视频理解回放。下表是建议，不属于服务器源事实；未现场核验的规模、许可和可获取性均保留“待核验”。benchmark 类只用于评估。</p>{rec_table}<p><a href="recommended-datasets.csv">完整推荐表 CSV</a></p></section>
<footer>生成时间 {ANALYSIS_DATE}；唯一服务器源根：<code>{SOURCE_ROOT}</code>。源 before/after 指纹见 <a href="source-readonly-evidence.json">source-readonly-evidence.json</a>，方法与限制见 <a href="methodology.md">methodology.md</a>。</footer>
</main></body></html>'''


def legacy_methodology(metrics: dict[str, dict[str, Any]]) -> str:
    return f"""# VLM 原始源数据分析方法

## 范围

- 唯一源根：`{SOURCE_ROOT}`
- 纳入：{', '.join(DATASET_ORDER)}
- 现场排除：`.cache`、根 `.gitattributes`、`robovqa-2unuse`、`textvqa-2unuse`
- 输出时间：{ANALYSIS_DATE}
- 本报告只使用原始源目录、原始归档、原始 Parquet/Arrow/JSON/ZIP/TAR 和从中保存的确定性证据；派生训练格式不在证据范围。

## 统计口径

1. **源记录**：canonical 物理训练/标注行。GQA 排除 balanced 重复子视图；Robo2VLM 排除 `data/data` 重复；Molmo2-VideoPoint 排除分类重复视图。
2. **唯一媒体**：共享命名空间中的 image/video ID、源字节 SHA256、原视频 lineage 或 scene ID。问答行不能替代媒体数。
3. **监督单元**：QA、caption、bbox、point、track 或类别 annotation。多标签统计明确允许总和超过记录数。
4. **token 估计**：源文本字符数除以 4。它用于配比量级，不是某个 tokenizer 的精确结果。
5. **有效量**：只在可由源证据确认时给出；媒体未取得时留空，不用其他数据补齐。

## 全量与抽样

- JSON 顶层数组按流式 `JSONDecoder.raw_decode` 全量读取。
- Parquet 使用 footer 和列投影；Arrow stream 无法投影时顺序读取 record batch。
- ZIP/TAR 通过中央目录或顺序 member 表读取，不向源目录解压。
- 每数据集以固定 seed 对完整声明总体生成无放回候选；默认 400 个候选、200 个 accepted 视觉样本。媒体不足的数据集保存 200 条原始 record evidence 并标为 incomplete。
- PixMo-Points 的 point 总数只来自固定 800 候选估计，报告不把它冒充全量统计。

## 能力分类

- 物理 task/category/type 字段优先。
- RoboVQA 当前 conversation 任务通过 user prompt 与 `task_metadata.question_processed` 的最长精确前缀匹配，920,011 条 reasoning 全部可匹配。
- Robo2VLM 没有 task 字段，因此对 684,710 条 source question 使用版本固定的可复现正则规则；这是全量确定性估计，不是人工标注真值。
- 只有记录显式包含 agent、目标、动作、环境状态或决策时标为 direct embodied。空间、grounding、tracking 仅标为 prerequisite。

## 泄露审计

- COCO family、GQA、TextVQA、Visual Genome 与 Molmo2 video 使用全量共享 ID lineage 交集。
- COCO、ChartQA、SpatialVLM 使用源媒体字节 SHA256 检查 split。
- Robo2VLM 使用去除 `_qN` 的 scene lineage 检查 split。
- 没有共享 ID namespace 的跨源对只报告 200 个确定性样本 SHA 交集，不能据此证明全量无重叠。

## 评级

- A：可直接进入训练候选池。
- B：可用，但需限额、授权核验或轻度过滤。
- C：必须清洗、去重、重切分或补充媒体审计。
- Q：媒体/lineage 无法闭环，训练前隔离。

## 训练配比原则

- 同时约束 unique media/episode、有效监督、token、单媒体最大重复次数和单数据集上限。
- COCO/VQAv2/LLaVA/VLM-R1/VisualGenome 按 COCO lineage 联合分组。
- LLaVA 的 GQA、TextVQA、VG 子源与对应上游数据统一去重。
- Molmo2 三类视频数据按共享 video_id 统一限额。
- 具身阶段不以机器人画面描述替代行动监督；Stage 4 需要 agent 目标、状态、可供性、成功或规划字段。

## 主要限制

- 三个 record-only 视频数据集无法完成媒体解码和视觉依赖验证。
- 远程 URL 数据的可访问率来自固定候选决策，不代表永久可用性。
- 许可结论只按当前源包可见信息记录；“待核验”不是授权。
- 外部推荐数据未在本任务中下载，相关规模与许可按待核验处理。

## 可复现工具

- `tools/remote_collect_json_stats.py`
- `tools/remote_collect_overlap_stats.py`
- `tools/remote_collect_split_media_stats.py`
- `tools/remote_collect_robo2vlm_capabilities.py`
- `tools/build_source_analysis_report.py`

最终 before/after 指纹与访问方式记录在 `source-readonly-evidence.json`。
"""


def methodology(metrics: dict[str, dict[str, Any]]) -> str:
    return f"""# VLM 原始源数据分析方法

## 范围与证据边界

- 唯一服务器源根：`{SOURCE_ROOT}`
- 纳入数据集：{', '.join(DATASET_ORDER)}
- 现场排除：`.cache`、根 `.gitattributes`、`robovqa-2unuse`、`textvqa-2unuse`
- 分析快照时间：{ANALYSIS_DATE}
- 禁止证据：`/mnt/luojunkun/stage1/dataset_ms-swift`、转换后 JSONL、ms-swift schema、转换日志、转换 split、loader/template 测试。

所有总体、schema、样本、质量与配方输入均来自原始源目录或从该目录只读归档的证据。外部推荐数据仅是建议，不参与当前统计。

## 六种统计口径

1. **canonical 源记录**：逐数据集定义的逻辑源行。GQA 排除 balanced 重复视图；Robo2VLM 排除嵌套重复 Parquet；Molmo2-VideoPoint 排除 category 重复视图。
2. **唯一媒体**：源 image/video ID、源媒体 SHA256、scene 或原视频分组。不同 QA 行不能替代媒体数。
3. **唯一 scene/episode/trajectory**：只在源中存在稳定字段或可验证规则时给出；否则写 `not exposed`，不以媒体数伪造。
4. **监督单元**：QA、caption、bbox、point、track、conversation 或类别 annotation。它可多于 canonical 记录。
5. **监督 token 估计**：源监督文本字符数除以 4。它是统一配比量级，不是特定 tokenizer 精确值。
6. **确认有效/拒绝/不可确认记录**：`confirmed_effective + known_rejected + unresolved = canonical_source_records`。需要媒体闭环、格式修复或质量核验但尚未完成的记录归入 unresolved，不伪报有效。

`source-accounting.csv` 使用 canonical 记录对账；`dataset-capabilities.csv` 另保留监督单元和 token，两者禁止相加。

## 全量扫描与固定样本

- JSON 顶层数组以流式 decoder 全量读取；Parquet 读取 footer 与列投影；Arrow 顺序读取 batch；ZIP/TAR 使用 member table，不向源目录解压。
- 每个数据集先定义完整 sampling population，再使用保存的固定 seed 生成无放回排列。默认记录 400 个候选并接受 200 个唯一、可解码媒体；不足时沿同一排列扩展。
- 15 个媒体闭环数据集归档 3,000 个视觉样本。Molmo2-VideoCapQA、Molmo2-VideoSubtitleQA、Molmo2-VideoTrack 缺完整媒体，各归档 200 条 source record evidence，共 600 条。
- 视觉归档遵循 Skill 的全总体均匀候选契约，不追认成配额式分层抽样。`sampling-strata.csv` 对固定样本按 split、源任务/能力和媒体类型做覆盖审计；全量任务分布仍以源 metadata 扫描为准。
- `sample-length-distribution.csv`、`sample-annotation-density.csv`、`sample-media-summary.csv` 均是 deterministic sample estimate。媒体缺失时写 `not_evaluable`。

## 任务与能力分类

- 报告先做与具身目标无关的统一任务 taxonomy：简单任务形态、任务族、细分任务、任务详细描述、输入、输出和领域。具身 E0-E6 只是后续子集视图。
- 源 task/category/type/answer_type 优先；同一源记录允许多标签。统一规则版本写入 `taxonomy_version`，当前实现见 `tools/source_task_taxonomy.py`。
- 3,600 条固定抽样实际展开 8,810 个 QA/标注单元。不发布独立的 3,600 个详情页；每个数据集独立报告内嵌本集全部 200 条媒体与 QA，逐单元底表为 `sample-task-annotations.csv`，聚合表为 `sample-task-distribution.csv`。
- 抽样任务族分布按标注单元加权，每个 QA/标注权重相同。
- RoboVQA reasoning prompt 与 `task_metadata.question_processed` 做最长精确前缀匹配；920,011 条 reasoning 均有源任务匹配。
- Robo2VLM 缺少显式 task 字段，对 684,710 个 canonical 问题运行版本固定的规则；这是全量确定性 heuristic，不是人工真值。
- LLaVA-Instruct、SpatialVLM 和问题级 VQA 的抽样细分类使用问题文本规则；Molmo2 视频 QA 优先使用源 `Category`/`AlignmentType`。这些规则用于审计分布，不冒充人工标注真值。
- 只有显式 agent、目标、动作、环境状态或决策监督才标为 direct embodied。空间、grounding、pointing、tracking 仅是 prerequisite。
- 任务堆叠图在每数据集内对多标签 task counts 重新归一；token 按任务计数比例分配，字段中保留该派生方法。

## E0-E6 覆盖与差距

- E0：通用感知/语言；E1：定位与空间；E2：动作与时序；E3：可供性与结果；E4：下一步与成功；E5：长程规划；E6：导航、记忆、恢复和安全。
- 每数据集在某层取最高 0-3 证据强度，乘质量权重 `A=1.0, B=0.8, C=0.5, Q=0`。
- 目标设为三个相互独立、核心强度来源，即 9 点；coverage 为 `min(current_points, 9) / 9`。
- 该分数衡量“可用来源覆盖”，不是模型性能。贡献来源与隔离来源完整写入 `capability-gap.csv`。

## 质量与风险

- A：可直接候选；B：可用但需限额；C：需要清洗或媒体核验；Q：媒体与标注无法闭环，必须隔离。
- 风险矩阵四轴为媒体完整性、schema/标注、重复/集中、许可/商用；0=低、1=中、2=高、3=严重/隔离。
- 风险等级由 `quality-audit.csv` 的全量问题、样本媒体检查和源中可见的许可信息派生，不代表法律授权结论。
- 文本长度、输出熵、唯一率、媒体尺寸、视频时长/fps/frame 只在 manifest 有真实字段或媒体已检查时计算；缺失不补推。

## 图表口径

`index.html` 先展示简单任务形态、标注单元加权任务族、逐数据集任务堆叠和任务明细，再展示全量源任务、规模、能力、具身子集、输入输出长度、监督密度与质量。每张图标注 full population 或 deterministic sample estimate。

## 训练配比

- 三套方案均包含五个阶段，并同时限制 unique media/episode、有效监督、token、单媒体重复和数据集准入条件。
- 同一媒体的多问设置统一上限，避免高密度 QA 来源放大训练权重。
- Molmo2 视频源按共享 video_id 统一限额；Q 级和 unresolved 数据在媒体、格式与许可条件满足前不得进入候选池。
- Stage 4 不以机器人画面 caption 代替行动监督，必须包含 agent 目标、状态、可供性、成功、动作或规划字段。
- `training-references.csv` 记录 10 篇一手论文、arXiv 链接、支持原则和方案映射；`training-recipes.csv.reference_ids` 标明每个阶段使用的文献编号。
- 论文只支持分阶段、多任务混合、空间/视频迁移、通用-机器人联合训练和轨迹/规划监督等原则；本报告的具体比例、媒体上限与 token 范围来自当前数据分布和 E0-E6 缺口，不是论文原始配方。

## 复现与只读证明

主要命令：

```text
python tools/build_sample_distribution_artifacts.py
python tools/build_source_analysis_report.py
python tools/validate_source_delivery.py
node tools/browser_audit.mjs
```

远程 collector 均通过 SSH stdin 执行，只写服务器 `/tmp` 并从 stdout 返回；源目录不落脚本、缓存或预览。18 个数据集的 before/after 指纹见 `source-readonly-evidence.json`。
"""


def main() -> int:
    from build_sample_distribution_artifacts import main as build_sample_artifacts
    from build_sample_detail_pages import INLINE_SAMPLE_STYLE, build_sample_details

    build_sample_artifacts()
    data = artifact_data()
    metrics = build_metrics(data)
    tasks = build_task_rows(data, metrics)
    enrich_task_rows(tasks, metrics)
    qrows = quality_rows(data, metrics)
    leakage_markers = (
        "泄露", "重叠", "复用", "leakage", "overlap", "cross-split", "跨 split",
        "train/test", "test scene", "仅 train split", "benchmark 媒体", "lineage intersection",
        "交集", "test 图像", "testdev", "重建 split",
    )
    qrows = [
        row for row in qrows
        if not any(
            marker in " ".join(str(row.get(key, "")) for key in ("issue", "evidence", "required_action")).lower()
            for marker in leakage_markers
        )
    ]
    for row in qrows:
        row["source_path"] = f"{SOURCE_ROOT}/{META[row['dataset']]['source_dir']}"
        row["evidence_date"] = ANALYSIS_DATE
    matrix, edge_rows, split_rows = overlap_outputs(data)
    for row in edge_rows:
        row["evidence_date"] = ANALYSIS_DATE
    for row in split_rows:
        row["evidence_date"] = ANALYSIS_DATE
    inventory = inventory_rows(metrics)
    recipes = training_rows()
    training_references = training_reference_rows()
    recommendations = recommended_rows()
    for row in recommendations:
        row.pop("overlap_risk", None)
        row.pop("benchmark_contamination_risk", None)
    accounting = source_accounting_rows(metrics)
    accounting_basis = {
        "TextVQA": "all source QA rows structurally usable",
        "ChartQA": "rows pending media and annotation quality review",
        "LLaVA-Instruct": "text-only rows excluded; visual rows pending source-family quality review",
        "VLM-R1": "invalid assistant JSON rejected; remaining rows pending media and annotation quality review",
        "Robo2VLM": "canonical rows pending scene-level media and annotation quality review",
        "PixMo-Cap": "complete immutable media snapshot and license provenance unavailable",
        "Molmo2-VideoTrack": "empty expressions rejected; remaining rows await media join by upstream source grouping",
    }
    for row in accounting:
        if row["dataset"] in accounting_basis:
            row["decision_basis"] = accounting_basis[row["dataset"]]
    risks = quality_risk_rows()
    for row in risks:
        row.pop("split_leakage_risk", None)
    gaps = capability_gap_rows()
    embodied = embodied_distribution_rows(tasks)

    length_rows = load_csv(ROOT / "sample-length-distribution.csv")
    length_summaries = load_csv(ROOT / "sample-length-summary.csv")
    density_rows = load_csv(ROOT / "sample-annotation-density.csv")
    density_summaries = load_csv(ROOT / "sample-annotation-density-summary.csv")
    media_summaries = load_csv(ROOT / "sample-media-summary.csv")
    text_quality_summaries = load_csv(ROOT / "sample-text-quality-summary.csv")
    strata_rows = load_csv(ROOT / "sampling-strata.csv")
    lengths_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in length_summaries:
        row["side"] = {"input": "输入", "output": "输出"}.get(row["side"], row["side"])
        lengths_by_dataset[row["dataset"]].append(row)
    sample_evidence = {
        "length_rows": length_rows,
        "density_rows": density_rows,
        "lengths": lengths_by_dataset,
        "density": {row["dataset"]: row for row in density_summaries},
        "media": {row["dataset"]: row for row in media_summaries},
        "text_quality": {row["dataset"]: row for row in text_quality_summaries},
        "strata": strata_rows,
    }

    build_schemas(data, metrics)
    detail_summary = build_sample_details(
        ROOT,
        SUB,
        DATASET_ORDER,
        STYLE,
        ANALYSIS_DATE,
        include_report_sections=True,
    )
    sample_evidence["sample_sections"] = detail_summary.pop("_report_sections")
    sample_evidence["sample_style"] = INLINE_SAMPLE_STYLE
    sample_evidence["task_annotations"] = load_csv(ROOT / "sample-task-annotations.csv")
    sample_evidence["task_distribution"] = load_csv(ROOT / "sample-task-distribution.csv")
    sample_evidence["detail_summary"] = detail_summary

    write_csv(ROOT / "inventory.csv", inventory)
    write_csv(ROOT / "source-accounting.csv", accounting)
    write_csv(ROOT / "quality-risk-matrix.csv", risks)
    write_csv(ROOT / "capability-gap.csv", gaps)
    write_csv(ROOT / "embodied-task-distribution.csv", embodied)
    capability_rows = []
    accounting_map = {row["dataset"]: row for row in accounting}
    for dataset in DATASET_ORDER:
        m = metrics[dataset]
        capability_rows.append({
            "dataset": dataset, "task_category": META[dataset]["simple_task"], "domain": META[dataset]["domain"],
            "modality": META[dataset]["modality"], "input": META[dataset]["input"], "output": META[dataset]["output"],
            "embodied_level": META[dataset]["embodied"], "embodied_evidence": META[dataset]["embodied_evidence"],
            "capability_codes": ";".join(key for key, value in META[dataset]["capabilities"].items() if value),
            "source_records": m["source_records"], "unique_media": m["unique_media"], "unique_scene_episode": m["unique_scene"],
            "supervision_units": m["supervision_units"], "effective_units": m["effective_units"], "token_estimate": m["token_estimate"],
            "effective_unit_type": m["effective_unit_type"], "annotations_per_media": m["annotations_per_media"],
            "confirmed_effective_source_records": accounting_map[dataset]["confirmed_effective_records"],
            "known_rejected_source_records": accounting_map[dataset]["known_rejected_records"],
            "unresolved_source_records": accounting_map[dataset]["unresolved_records"],
            "quality_rating": META[dataset]["rating"], "evidence_scope": m["evidence"],
            "source_path": f"{SOURCE_ROOT}/{META[dataset]['source_dir']}", "evidence_date": ANALYSIS_DATE,
        })
    write_csv(ROOT / "dataset-capabilities.csv", capability_rows)
    write_csv(ROOT / "task-distribution.csv", tasks)
    write_csv(ROOT / "quality-audit.csv", qrows)
    write_csv(ROOT / "training-recipes.csv", recipes)
    write_csv(ROOT / "training-references.csv", training_references)
    write_csv(ROOT / "recommended-datasets.csv", recommendations)
    (ROOT / "methodology.md").write_text(methodology(metrics), encoding="utf-8")
    scope = {
        "version": 1,
        "analysis_mode": "source-only",
        "ssh_host": "8.146.226.25",
        "server_allowlist": [SOURCE_ROOT],
        "server_denylist": [
            "/mnt/luojunkun/stage1/dataset_ms-swift",
            "converted JSONL and conversion manifests",
            "converted train/validation splits",
            "ms-swift loader/template outputs",
        ],
        "local_output_root": str(ROOT),
        "analysis_date": ANALYSIS_DATE,
    }
    (ROOT / "scope-declaration.json").write_text(
        json.dumps(scope, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    inventory_map = {row["entry"]: row for row in inventory if row["status"] == "included"}
    for dataset in DATASET_ORDER:
        page = dataset_page(
            dataset, data, metrics, tasks, qrows, edge_rows, split_rows,
            inventory_map, accounting_map, sample_evidence,
        )
        (ROOT / f"{dataset}.html").write_text(page, encoding="utf-8")
    index = index_page(
        metrics, tasks, qrows, matrix, edge_rows, split_rows, recipes, recommendations,
        accounting, risks, gaps, embodied, sample_evidence,
    )
    (ROOT / "index.html").write_text(index, encoding="utf-8")

    generated = [
        "index.html", "inventory.csv", "dataset-capabilities.csv", "task-distribution.csv",
        "quality-audit.csv", "training-recipes.csv", "training-references.csv", "recommended-datasets.csv", "methodology.md",
        "source-accounting.csv", "quality-risk-matrix.csv", "capability-gap.csv",
        "embodied-task-distribution.csv", "sampling-strata.csv", "sample-length-distribution.csv",
        "sample-length-summary.csv", "sample-annotation-density.csv",
        "sample-annotation-density-summary.csv", "sample-media-summary.csv",
        "sample-text-quality-summary.csv", "scope-declaration.json",
        "sample-task-annotations.csv", "sample-task-distribution.csv",
        *[f"{name}.html" for name in DATASET_ORDER],
    ]
    digest = hashlib.sha256()
    for name in sorted(generated):
        digest.update(name.encode("utf-8") + b"\0" + (ROOT / name).read_bytes())
    build = {
        "generated_at": ANALYSIS_DATE,
        "source_root": SOURCE_ROOT,
        "source_only": True,
        "dataset_reports": len(DATASET_ORDER),
        "sample_records": detail_summary["sample_records"],
        "embedded_sample_records": detail_summary["embedded_sample_records"],
        "qa_annotation_units": detail_summary["qa_annotation_units"],
        "generated_files": generated,
        "delivery_digest": digest.hexdigest(),
    }
    (ARTIFACTS / "report-build-summary.json").write_text(json.dumps(build, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(build, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
