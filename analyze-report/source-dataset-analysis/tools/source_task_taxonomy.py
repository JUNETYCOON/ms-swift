#!/usr/bin/env python3
"""Versioned, source-evidence task taxonomy shared by report builders."""

from __future__ import annotations

import re
from typing import Any


TAXONOMY_VERSION = "source-task-taxonomy-v1.0"

SIMPLE_TASK_ORDER = [
    "classification", "description", "vqa", "grounding",
    "pointing", "tracking", "instruction", "planning",
]
SIMPLE_TASK_LABELS = {
    "classification": "classification",
    "description": "description / caption",
    "vqa": "VQA",
    "grounding": "grounding",
    "pointing": "pointing",
    "tracking": "tracking",
    "instruction": "multimodal instruction",
    "planning": "planning / action prediction",
}

TASK_FAMILY_ORDER = [
    "视觉识别与属性感知",
    "通用视觉问答",
    "关系、逻辑与常识推理",
    "数量与数值推理",
    "OCR 与文字理解",
    "图表与科学图解推理",
    "图像或视频描述",
    "多模态指令跟随",
    "视觉定位与指点",
    "空间与三维几何",
    "视频时序与事件理解",
    "对象交互与动作理解",
    "机器人状态、可供性与结果",
    "行动预测与任务规划",
    "跨帧对象跟踪",
    "无答案评测或纯文本",
]

TASK_FAMILY_COLORS = {
    "视觉识别与属性感知": "#247247",
    "通用视觉问答": "#08786f",
    "关系、逻辑与常识推理": "#4e78a0",
    "数量与数值推理": "#b7831d",
    "OCR 与文字理解": "#a33d36",
    "图表与科学图解推理": "#6d8f3c",
    "图像或视频描述": "#28658c",
    "多模态指令跟随": "#64717b",
    "视觉定位与指点": "#45a18f",
    "空间与三维几何": "#5f6fb2",
    "视频时序与事件理解": "#bd6d41",
    "对象交互与动作理解": "#9a5b45",
    "机器人状态、可供性与结果": "#9c4545",
    "行动预测与任务规划": "#7b3f38",
    "跨帧对象跟踪": "#8b6f9e",
    "无答案评测或纯文本": "#a8b0b4",
}


DATASET_PROFILES: dict[str, dict[str, str]] = {
    "COCO": {
        "domain": "通用自然图像与对象类别",
        "input": "单张自然图像",
        "output": "多个对象类别 ID 或类别名",
    },
    "VQAv2": {
        "domain": "通用自然图像问答、属性、数量与常识",
        "input": "COCO 图像与自然语言问题",
        "output": "开放短答案、是非答案或数值；评测 split 可无答案",
    },
    "VisualGenome": {
        "domain": "通用场景图、区域描述、关系与视觉定位",
        "input": "自然图像、问题或区域坐标",
        "output": "短答案，或区域短语与边界框",
    },
    "GQA": {
        "domain": "场景图关系、组合逻辑、属性与比较",
        "input": "Visual Genome 图像与组合问题",
        "output": "短答案及 structural/semantic 类型",
    },
    "TextVQA": {
        "domain": "场景文字识别与文字问答",
        "input": "自然图像、问题与可选 OCR token",
        "output": "多标注短文本答案",
    },
    "ChartQA": {
        "domain": "图表读取、数值比较与数据可视化推理",
        "input": "图表图像与问题",
        "output": "文本或数值答案",
    },
    "AI2D": {
        "domain": "科学图解、教育图表与图形关系",
        "input": "科学示意图与四选一问题",
        "output": "选项下标、答案文本及图解元素标注",
    },
    "LLaVA-Instruct": {
        "domain": "通用图文描述、问答、推理与多轮指令",
        "input": "可选图像、用户指令与对话历史",
        "output": "assistant 自由文本",
    },
    "VLM-R1": {
        "domain": "通用图像指代表达定位",
        "input": "图像与 referring expression",
        "output": "目标标签与二维边界框",
    },
    "Robo2VLM": {
        "domain": "机器人场景、几何、轨迹、目标状态与动作结果",
        "input": "机器人场景图与多选问题",
        "output": "选项下标或选项文本",
    },
    "RoboVQA": {
        "domain": "机器人操作、对象动作、可供性、成功判断与规划",
        "input": "机器人视频、目标、历史上下文与问题",
        "output": "描述、判断、未来动作或子任务计划",
    },
    "SpatialVLM": {
        "domain": "方向、距离、深度、尺寸、相对尺度与通用问答",
        "input": "单图与多轮自然语言问题",
        "output": "短答案、解释、度量值或空间关系",
    },
    "PixMo-Cap": {
        "domain": "开放域高密度长图像描述",
        "input": "开放域图像",
        "output": "长 caption 与人工 transcript",
    },
    "PixMo-Points": {
        "domain": "开放域对象指点与数量监督",
        "input": "图像、对象标签与采集任务类型",
        "output": "点坐标集合或对象数量",
    },
    "Molmo2-VideoCapQA": {
        "domain": "视频对象、动作、场景、时序、空间、因果、文字与数量",
        "input": "视频、问题与候选答案；当前源包部分媒体不可用",
        "output": "多选答案与负选项",
    },
    "Molmo2-VideoPoint": {
        "domain": "视频异常、对象或动作的时空指点",
        "input": "视频、指代表达与时间范围",
        "output": "跨帧点坐标、时间戳与计数",
    },
    "Molmo2-VideoSubtitleQA": {
        "domain": "字幕与视觉对齐、事件时序、因果、对象交互与文字理解",
        "input": "视频、时间字幕、问题与负选项；当前源包媒体不可用",
        "output": "文本答案",
    },
    "Molmo2-VideoTrack": {
        "domain": "跨帧对象持续性、指代表达与多目标轨迹",
        "input": "视频片段与对象指代表达",
        "output": "逐帧点、可见区间与对象轨迹",
    },
}


VIDEO_CATEGORY_ZH = {
    "action count": "动作计数",
    "action localization": "动作时间定位",
    "action reasoning": "动作推理",
    "action recognition": "动作识别",
    "action sequence": "动作先后顺序",
    "anomaly recognition": "异常识别",
    "anomaly reasoning": "异常推理",
    "camera movement": "镜头运动理解",
    "causal reasoning": "因果推理",
    "dialogue content": "对话内容理解",
    "event causality": "事件因果关系",
    "event detection": "事件检测",
    "event location": "事件空间或时间位置",
    "event sequence": "事件先后顺序",
    "event summary": "事件摘要",
    "facial recognition": "人脸识别",
    "gesture recognition": "手势识别",
    "human appearance": "人物外观",
    "human attributes": "人物属性",
    "human cognitive process": "人物意图或认知过程",
    "human emotion": "人物情绪",
    "human human interaction recognition": "人物间交互识别",
    "human human relationship": "人物关系",
    "human object interaction count": "人-物交互计数",
    "human object interaction recognition": "人-物交互识别",
    "human object relationship": "人物与对象关系",
    "human pose": "人体姿态",
    "movement direction": "运动方向",
    "narrative reasoning": "叙事推理",
    "object count": "对象计数",
    "object interaction": "对象交互",
    "object location": "对象位置",
    "object object relationship": "对象间关系",
    "object ordering": "对象顺序",
    "object presence": "对象存在判断",
    "object properties": "对象属性",
    "object reasoning": "对象推理",
    "object recognition": "对象识别",
    "object state": "对象状态",
    "planning": "行动规划",
    "process description": "过程描述",
    "property comparison": "属性比较",
    "quantity comparison": "数量比较",
    "scene description": "场景描述",
    "scene reasoning": "场景推理",
    "scene segmentation": "场景分段",
    "scene sequence": "场景先后顺序",
    "single object event count": "单对象事件计数",
    "single object event recognition": "单对象事件识别",
    "single object location change": "单对象位置变化",
    "single object presence change": "单对象出现或消失",
    "single object quantity change": "单对象数量变化",
    "single object speed": "单对象速度",
    "single object state change": "单对象状态变化",
    "social reasoning": "社会关系推理",
    "spatial perception": "空间感知",
    "spatial reasoning": "空间推理",
    "spatial relationship": "空间关系",
    "state comparison": "状态比较",
    "temporal perception": "时间感知",
    "temporal reasoning": "时序推理",
    "text count": "文字计数",
    "text location": "文字位置",
    "text recognition": "文字识别",
    "video editing effects": "视频编辑效果",
    "video topic": "视频主题",
}


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _annotation_value(annotation: Any, *keys: str) -> Any:
    if not isinstance(annotation, dict):
        return None
    for key in keys:
        if annotation.get(key) not in (None, ""):
            return annotation[key]
    return None


def _question_semantics(question: str) -> tuple[str, str, str]:
    value = question.lower().strip()
    if re.search(r"\b(how many|number of|count|quantity|total number)\b", value):
        return "数量与数值推理", "数量计算问答", "根据视觉内容计算对象、动作或文字数量。"
    if re.search(r"\b(read|text|word|letter|number written|title|author|brand name|sign say|written on)\b", value):
        return "OCR 与文字理解", "场景文字读取问答", "读取图像或视频中的可见文字并回答问题。"
    if re.search(r"\b(where|location|left|right|above|below|behind|front of|near|far|distance|closest|farthest)\b", value):
        return "空间与三维几何", "位置、方向或距离问答", "判断对象位置、方向、遮挡、深度或距离关系。"
    if re.search(r"\b(why|reason|cause|because|infer|likely|common with|relationship)\b", value):
        return "关系、逻辑与常识推理", "因果、关系或常识问答", "结合视觉证据与关系或常识进行推理。"
    if re.search(r"\b(compare|same|different|larger|smaller|more than|less than|taller|shorter)\b", value):
        return "关系、逻辑与常识推理", "属性或对象比较", "比较两个或多个对象的属性、状态或数量。"
    if re.search(r"\b(color|shape|material|wearing|look like|kind of|type of|attribute)\b", value):
        return "视觉识别与属性感知", "对象属性问答", "识别对象的颜色、形状、材质、类别或外观属性。"
    if re.search(r"\b(action|doing|happen|move|motion|interact|holding|carrying|use|play|working)\b", value):
        return "对象交互与动作理解", "动作与对象交互问答", "识别主体动作、对象交互或场景中的事件。"
    if re.match(r"^(is|are|do|does|did|can|could|has|have|was|were)\b", value):
        return "通用视觉问答", "是非判断问答", "对可见对象、属性或场景事实进行二元判断。"
    return "通用视觉问答", "开放式对象与场景问答", "根据视觉内容回答对象、属性、人物或场景问题。"


def _video_family(category: str, question: str) -> tuple[str, str, str]:
    value = category.lower().strip()
    label = VIDEO_CATEGORY_ZH.get(value, value or "未声明类别")
    if any(token in value for token in ("text", "dialogue")):
        family = "OCR 与文字理解"
        description = "结合画面文字、字幕或对话内容回答视频问题。"
    elif any(token in value for token in ("count", "quantity")):
        family = "数量与数值推理"
        description = "对视频中的对象、动作、事件或文字进行计数或数量比较。"
    elif any(token in value for token in ("location", "spatial", "direction")):
        family = "空间与三维几何"
        description = "判断视频中的对象位置、运动方向或空间关系。"
    elif any(token in value for token in ("sequence", "temporal", "localization", "change", "speed", "camera", "editing", "detection", "segmentation")):
        family = "视频时序与事件理解"
        description = "理解事件顺序、时间定位、状态变化、镜头或视频编辑过程。"
    elif any(token in value for token in ("causal", "reasoning", "relationship", "social", "narrative")):
        family = "关系、逻辑与常识推理"
        description = "结合跨帧视觉、字幕或上下文推断因果、关系和意图。"
    elif any(token in value for token in ("interaction", "action", "pose", "gesture", "process")):
        family = "对象交互与动作理解"
        description = "识别人、物体和动作之间的交互或操作过程。"
    elif any(token in value for token in ("recognition", "appearance", "attributes", "properties", "emotion", "state", "presence", "ordering")):
        family = "视觉识别与属性感知"
        description = "识别视频中的对象、人物、属性、状态或可见性。"
    elif any(token in value for token in ("summary", "description", "topic", "domain")):
        family = "图像或视频描述"
        description = "概括视频事件、主题、场景或过程。"
    elif "planning" in value:
        family = "行动预测与任务规划"
        description = "根据视频上下文预测或规划后续行动。"
    else:
        family, _, description = _question_semantics(question)
    return family, f"视频问答：{label}", description


def classify_task(
    dataset: str,
    source_task: Any,
    *,
    question: Any = "",
    answer: Any = None,
    instruction: Any = "",
    annotation: Any = None,
) -> dict[str, str]:
    """Classify one source task or sampled supervision unit deterministically."""

    profile = DATASET_PROFILES[dataset]
    task = _text(source_task).strip() or dataset
    task_lower = task.lower()
    question_text = _text(question).strip()
    combined = " ".join((task_lower, question_text.lower(), _text(instruction).lower()))
    answer_type = _text(_annotation_value(annotation, "answer_type")).lower()
    category = _text(_annotation_value(annotation, "Category", "category")).strip()

    simple = "vqa"
    family = "通用视觉问答"
    fine = "开放式视觉问答"
    description = "根据视觉输入回答自然语言问题。"
    output = profile["output"]
    basis = "数据集任务契约"

    if dataset == "COCO":
        simple, family, fine = "classification", "视觉识别与属性感知", "对象多标签分类"
        description = "判断一张自然图像中同时出现的多个对象类别。"
        basis = "源 labels 字段"
    elif dataset == "VQAv2":
        missing_answer = "without answer" in task_lower or (
            answer_type == "" and isinstance(answer, str) and "未提供" in answer
        )
        semantic_family, semantic_fine, semantic_description = _question_semantics(question_text)
        if missing_answer:
            family, fine = "无答案评测或纯文本", f"无答案评测输入：{semantic_fine}"
            description = "源评测行仅含图像与问题，不提供可用于监督训练的答案。"
            output = "无答案"
        elif "count" in task_lower or "numeric" in task_lower or answer_type == "number":
            family, fine, description = "数量与数值推理", "数量或数值 VQA", "根据图像计算数量或回答数值问题。"
        elif "binary" in task_lower or answer_type == "yes/no":
            family, fine, description = "通用视觉问答", "是非判断 VQA", "对图像中的事实进行 yes/no 判断。"
        elif task_lower in {"vqav2", "open answer"}:
            family, fine, description = semantic_family, semantic_fine, semantic_description
        else:
            family, fine, description = semantic_family, semantic_fine, semantic_description
        basis = "源 answer_type/question_type 与问题文本规则"
    elif dataset == "VisualGenome":
        if "region" in task_lower:
            simple, family, fine = "grounding", "视觉定位与指点", "区域描述与边界框定位"
            description = "用短语描述图像区域，并通过边界框绑定到对应实体。"
            output = "区域短语与 x/y/width/height"
            basis = "源 region phrase 与坐标字段"
        else:
            family, fine, description = _question_semantics(question_text)
            fine = "场景图问答：" + fine
            basis = "源 question_answers 记录与问题文本规则"
    elif dataset == "GQA":
        mapping = {
            "choose": ("二选一关系或属性选择", "在候选对象、关系或属性之间进行选择。"),
            "compare": ("对象与属性比较", "比较对象的属性、数量、位置或关系。"),
            "logical": ("多条件逻辑组合", "组合多个场景图条件执行与、或等逻辑推理。"),
            "query": ("对象、属性或关系查询", "查询场景图中的对象、属性或关系。"),
            "verify": ("对象、属性或关系验证", "验证场景图命题是否成立。"),
        }
        structural = task_lower if task_lower in mapping else _text(_annotation_value(annotation, "types.structural", "structural")).lower()
        if structural in mapping:
            fine, description = mapping[structural]
            family = "关系、逻辑与常识推理"
        elif isinstance(answer, str) and "未提供" in answer:
            family, fine, description, output = "无答案评测或纯文本", "GQA 无答案评测输入", "源 evaluation 行没有答案。", "无答案"
        else:
            family, fine, description = _question_semantics(question_text)
        basis = "源 types.structural/types.semantic 与问题文本规则"
    elif dataset == "TextVQA":
        family, fine, description = "OCR 与文字理解", "场景文字视觉问答", "读取自然场景中的可见文字并回答相关问题。"
        basis = "数据集 OCR-VQA 契约"
    elif dataset == "ChartQA":
        family, fine, description = "图表与科学图解推理", "图表读取与数值推理", "读取图表元素、趋势与数值并回答问题。"
        basis = "数据集图表 QA 契约"
    elif dataset == "AI2D":
        family, fine, description = "图表与科学图解推理", "科学图解多项选择问答", "理解科学示意图中的元素、过程与关系并选择答案。"
        output = "答案选项下标与文本"
        basis = "源 questions JSON"
    elif dataset == "LLaVA-Instruct":
        if "text-only" in task_lower:
            simple, family, fine = "instruction", "无答案评测或纯文本", "纯文本指令对话"
            description = "记录不含图像，不构成视觉监督。"
            basis = "源 image 字段缺失"
        elif re.search(r"\b(caption|describe|description|what is this (photo|image) about)\b", combined):
            simple, family, fine = "description", "图像或视频描述", "通用图像描述"
            description = "按用户要求生成一句或多段图像描述。"
            basis = "源 user instruction 文本规则"
        elif question_text:
            simple = "vqa"
            family, fine, description = _question_semantics(question_text)
            fine = "指令式" + fine
            basis = "源多轮 user/assistant 内容规则"
        else:
            simple, family, fine = "instruction", "多模态指令跟随", "通用多模态指令"
            description = "根据图像与自然语言指令生成自由文本。"
            basis = "源会话契约或上游 lineage"
    elif dataset == "VLM-R1":
        simple, family, fine = "grounding", "视觉定位与指点", "指代表达边界框定位"
        description = "根据 referring expression 定位目标并输出二维边界框。"
        output = "目标标签与 bbox_2d"
        basis = "assistant bbox schema"
    elif dataset == "Robo2VLM":
        rules = [
            ("cross_view_3d_correspondence", "grounding", "空间与三维几何", "跨视角三维对应", "判断两个相机视图中对应同一三维位置的点。"),
            ("depth_distance_estimation", "vqa", "空间与三维几何", "深度与距离估计", "比较机器人场景中点或对象的远近和深度。"),
            ("trajectory_language_alignment", "vqa", "对象交互与动作理解", "轨迹与语言指令对齐", "选择与可视轨迹最匹配的机器人语言指令。"),
            ("success_judgment", "vqa", "机器人状态、可供性与结果", "任务成功判断", "判断机器人是否完成指定任务。"),
            ("next_motion_prediction", "planning", "行动预测与任务规划", "下一步运动预测", "根据当前场景预测机器人下一步运动方向。"),
            ("obstacle_reachability", "vqa", "机器人状态、可供性与结果", "障碍与可达性判断", "判断目标是否可达或是否被障碍阻挡。"),
            ("goal_state_recognition", "vqa", "机器人状态、可供性与结果", "目标状态识别", "识别符合任务目标的最终对象配置。"),
            ("affordance_feasibility", "vqa", "机器人状态、可供性与结果", "动作可供性判断", "判断机器人能否执行抓取、放置、移动或开合动作。"),
            ("spatial_relation_localization", "grounding", "空间与三维几何", "机器人场景空间定位", "判断机器人场景中的相对位置或目标点。"),
            ("robot_action_object_understanding", "vqa", "对象交互与动作理解", "机器人动作与对象理解", "识别机器人、动作、工具、对象和任务指令之间的关系。"),
        ]
        selected = next((row for row in rules if row[0] in task_lower), None)
        if selected is None:
            patterns = {
                "cross_view_3d_correspondence": r"left image|right image|same 3d location|ext[12] camera",
                "depth_distance_estimation": r"farthest|closest|nearer|farther|depth|distance from the camera",
                "trajectory_language_alignment": r"trajectory|language instruction best describes|path shown",
                "success_judgment": r"successfully completed|task successful|achieved the task",
                "next_motion_prediction": r"move next|next move|direction .*robot.*move|arrow correctly shows",
                "obstacle_reachability": r"obstacle|blocking the robot|reachable|reach the",
                "goal_state_recognition": r"goal state|configuration shows|desired state|final state",
                "affordance_feasibility": r"is it possible|can the robot|able to (pick|place|move|reach|open|close)",
                "spatial_relation_localization": r"which (point|side|location)|to the left|to the right|above|below|relative position",
            }
            matched = next((name for name, pattern in patterns.items() if re.search(pattern, question_text, re.I)), "robot_action_object_understanding")
            selected = next(row for row in rules if row[0] == matched)
        _, simple, family, fine, description = selected
        output = "多选答案"
        basis = "源问题文本的版本化 Robo2VLM 规则"
    elif dataset == "RoboVQA":
        if "description.robot" in task_lower or "video description" in task_lower:
            simple, family, fine = "description", "图像或视频描述", "机器人对象、动作与过程描述"
            description = "描述视频中的对象、机器人动作及其状态变化。"
        elif "affordance" in task_lower:
            family, fine = "机器人状态、可供性与结果", "动作可供性判断"
            description = "判断或解释机器人动作是否可执行。"
        elif "success" in task_lower:
            family, fine = "机器人状态、可供性与结果", "任务成功或失败判断"
            description = "根据操作视频判断任务是否成功。"
        elif "future_prediction" in task_lower:
            simple, family, fine = "planning", "行动预测与任务规划", "未来动作预测"
            description = "预测视频上下文之后最可能发生的机器人动作。"
        elif "remaining5_planning" in task_lower:
            simple, family, fine = "planning", "行动预测与任务规划", "剩余五步子任务规划"
            description = "结合历史上下文生成剩余五个子任务。"
        elif "immediate_planning" in task_lower:
            simple, family, fine = "planning", "行动预测与任务规划", "带上下文的下一步规划"
            description = "结合过去操作上下文生成当前下一步子任务。"
        elif "task:planning" in task_lower:
            simple, family, fine = "planning", "行动预测与任务规划", "任务级规划"
            description = "根据目标生成机器人任务计划。"
        elif "past_description" in task_lower:
            simple, family, fine = "description", "对象交互与动作理解", "历史动作描述"
            description = "描述机器人过去已经执行的动作。"
        else:
            family, fine, description = "对象交互与动作理解", "机器人视频问答", "根据机器人操作视频回答动作与对象问题。"
        basis = "源 metadata.task_metadata.task"
    elif dataset == "SpatialVLM":
        explicit = {
            "spatial.depth_order": ("空间与三维几何", "前后与深度顺序", "判断对象在深度方向的前后关系。"),
            "spatial.directional_relation": ("空间与三维几何", "二维方向关系", "判断左、右、上、下等方向关系。"),
            "spatial.distance": ("空间与三维几何", "距离估计", "估计或比较对象之间的距离。"),
            "spatial.metric_size": ("空间与三维几何", "绝对尺寸估计", "估计对象的宽、高或真实尺度。"),
            "spatial.relative_scale": ("空间与三维几何", "相对尺度比较", "比较对象的大小、高度或尺度。"),
            "vqa.object_attribute": ("视觉识别与属性感知", "对象属性问答", "识别对象属性。"),
            "vqa.general": ("通用视觉问答", "通用图像问答", "回答非空间类图像问题。"),
        }
        selected = next((value for key, value in explicit.items() if key in task_lower), None)
        if selected:
            family, fine, description = selected
        else:
            family, fine, description = _question_semantics(question_text)
        basis = "源问题文本与全量 SpatialVLM 规则"
    elif dataset == "PixMo-Cap":
        simple, family, fine = "description", "图像或视频描述", "开放域长图像描述"
        description = "生成覆盖对象、关系、场景与细节的长描述。"
        basis = "源 caption/transcript 字段"
    elif dataset == "PixMo-Points":
        method = _text(_annotation_value(annotation, "collection_method")).lower()
        count_value = answer.get("count") if isinstance(answer, dict) else None
        if task_lower in {"counting", "vqa.counting"} or method == "counting" or count_value is not None:
            simple, family, fine = "vqa", "数量与数值推理", "指代对象计数"
            description = "根据对象标签统计图像中匹配对象的数量。"
            output = "对象数量"
        else:
            simple, family, fine = "pointing", "视觉定位与指点", "对象点坐标定位"
            description = "根据对象标签输出一个或多个目标点坐标。"
            output = "点坐标集合"
        basis = "源 collection_method、count 与 points 字段"
    elif dataset in {"Molmo2-VideoCapQA", "Molmo2-VideoSubtitleQA"}:
        family, fine, description = _video_family(category, question_text)
        if dataset == "Molmo2-VideoSubtitleQA":
            fine = "字幕对齐" + fine
            description += " 该记录同时提供带时间戳字幕。"
        output = "多选或文本答案"
        basis = "源 Category、AlignmentType 与问题文本规则"
    elif dataset == "Molmo2-VideoPoint":
        simple, family = "pointing", "视觉定位与指点"
        if "anomaly" in (category.lower() or task_lower):
            fine = "视频异常时空指点"
            description = "在生成视频的具体时间与画面位置标出视觉异常或失真。"
        else:
            fine = "视频对象或动作时空指点"
            description = "在视频帧中定位被指代对象或动作。"
        output = "跨帧点坐标、时间戳与计数"
        basis = "源 category、timestamps 与 points 字段"
    elif dataset == "Molmo2-VideoTrack":
        simple, family, fine = "tracking", "跨帧对象跟踪", "指代表达多目标视频跟踪"
        description = "根据对象表达输出一个或多个对象在连续帧中的点与可见区间。"
        output = "逐帧点、segment 与对象 ID"
        basis = "源 exp、points、segments 与 frame 字段"

    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "simple_task_category": simple,
        "simple_task_label": SIMPLE_TASK_LABELS[simple],
        "task_family": family,
        "fine_task": fine,
        "task_description": description,
        "input": profile["input"],
        "output": output,
        "domain": profile["domain"],
        "classification_basis": basis,
    }
