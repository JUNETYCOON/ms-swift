#!/usr/bin/env python3
"""Evaluate grounding predictions in ms-swift inference JSONL files.

The default input is produced by::

    swift infer --val_dataset val.jsonl --result_path predictions.jsonl

For causal language model inference, ms-swift keeps the original ``objects``
field, stores the generated text in ``response``, and stores the removed
assistant answer in ``labels``. This evaluator reads ground-truth boxes from
``objects.bbox`` and parses predicted boxes from Qwen-VL legacy tags, Qwen3-VL
JSON output, or common plain-text box formats.

Prediction parse failures are scored as zero rather than silently skipped.
Rows whose coordinate systems cannot be aligned are reported separately.
Point-only annotations are counted but excluded because IoU is undefined for
points.

输入是推理结果 JSONL，输出 IoU/Acc/F1 报告。
python /mnt/workspace/ms-swift/scripts/evaluate_grounding_iou.py \
  --input /mnt/luojunkun/stage1/eval_results/grounding_infer.jsonl \
  --prediction-space norm1000 \
  --thresholds 0.25 0.5 0.75 \
  --report /mnt/luojunkun/stage1/eval_results/iou_report.json \
  --details-output /mnt/luojunkun/stage1/eval_results/iou_errors.jsonl \
  --overwrite


  --input              推理结果 jsonl
--prediction-space   模型输出 bbox 的坐标格式,
                    real / norm1 / norm1000
--thresholds         IoU 阈值，例如 0.5
--task-filter auto   自动跳过“输入 bbox，让模型描述区域”的样本
--report             汇总指标 json
--details-output     错误样本明细 jsonl
--overwrite          允许覆盖已有报告


Qwen/Qwen3-VL grounding 输出，大概率用：
--prediction-space norm1000
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, TextIO


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LEGACY_BOX_RE = re.compile(r"<\|box_start\|>(.*?)<\|box_end\|>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PAIR_BOX_RE = re.compile(
    rf"\(\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*\)\s*,?\s*"
    rf"\(\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*\)"
)
BBOX_KEY_RE = re.compile(
    rf"(?:bbox_2d|bounding_box|bbox|box|start_box)\s*[\"']?\s*[:=]\s*"
    rf"[\[(]\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*,\s*"
    rf"({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*[\])]",
    re.IGNORECASE,
)
POINT_KEY_RE = re.compile(
    rf"(?:point_2d|click_point|point|start_box)\s*[\"']?\s*[:=]\s*"
    rf"[\[(]\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*[\])]",
    re.IGNORECASE,
)
TAGGED_BOX_RE = re.compile(r"<(?:bbox|box)>(.*?)</(?:bbox|box)>", re.DOTALL | re.IGNORECASE)

BOX_KEYS = ("bbox_2d", "bounding_box", "bbox", "box", "start_box")
POINT_KEYS = ("point_2d", "click_point", "point")
PREDICTION_FALLBACK_FIELDS = ("response", "prediction", "pred", "output")


@dataclass(frozen=True)
class Geometry:
    coordinates: tuple[float, ...]
    image_id: int = 0


@dataclass
class ParsedPrediction:
    boxes: list[Geometry] = field(default_factory=list)
    points: list[Geometry] = field(default_factory=list)
    parser: Optional[str] = None


@dataclass
class EvaluationOutcome:
    status: str
    gt_box_count: int = 0
    pred_box_count: int = 0
    gt_point_count: int = 0
    pred_point_count: int = 0
    best_gt_ious: list[float] = field(default_factory=list)
    matched_ious: list[float] = field(default_factory=list)
    parse_success: bool = False
    parser: Optional[str] = None
    error: Optional[str] = None
    invalid_pred_boxes: int = 0
    clipped_boxes: int = 0
    reordered_boxes: int = 0


@dataclass
class MetricAccumulator:
    thresholds: tuple[float, ...]
    rows: int = 0
    filtered_non_box_output_rows: int = 0
    rows_with_gt_boxes: int = 0
    rows_without_gt_boxes: int = 0
    rows_with_gt_points: int = 0
    parse_success_rows: int = 0
    parse_failure_rows: int = 0
    coordinate_error_rows: int = 0
    scored_rows: int = 0
    single_target_rows: int = 0
    gt_boxes_total: int = 0
    gt_boxes_scored: int = 0
    pred_boxes_scored: int = 0
    gt_points_total: int = 0
    pred_points_total: int = 0
    invalid_pred_boxes: int = 0
    clipped_boxes: int = 0
    reordered_boxes: int = 0
    sum_iou_over_gt: float = 0.0
    true_positives: dict[float, int] = field(default_factory=dict)
    complete_records: dict[float, int] = field(default_factory=dict)
    exact_set_records: dict[float, int] = field(default_factory=dict)
    single_target_hits: dict[float, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for threshold in self.thresholds:
            self.true_positives.setdefault(threshold, 0)
            self.complete_records.setdefault(threshold, 0)
            self.exact_set_records.setdefault(threshold, 0)
            self.single_target_hits.setdefault(threshold, 0)

    def update(self, outcome: EvaluationOutcome) -> None:
        self.rows += 1
        self.gt_points_total += outcome.gt_point_count
        self.pred_points_total += outcome.pred_point_count
        self.invalid_pred_boxes += outcome.invalid_pred_boxes
        self.clipped_boxes += outcome.clipped_boxes
        self.reordered_boxes += outcome.reordered_boxes

        if outcome.status == "filtered_non_box_output":
            self.filtered_non_box_output_rows += 1
            return

        if outcome.gt_point_count:
            self.rows_with_gt_points += 1
        if not outcome.gt_box_count:
            self.rows_without_gt_boxes += 1
            return

        self.rows_with_gt_boxes += 1
        self.gt_boxes_total += outcome.gt_box_count
        if outcome.parse_success:
            self.parse_success_rows += 1
        else:
            self.parse_failure_rows += 1

        if outcome.status == "coordinate_error":
            self.coordinate_error_rows += 1
            return
        if outcome.status not in {"scored", "parse_failure"}:
            return

        self.scored_rows += 1
        self.gt_boxes_scored += outcome.gt_box_count
        self.pred_boxes_scored += outcome.pred_box_count
        self.sum_iou_over_gt += sum(outcome.best_gt_ious)
        if outcome.gt_box_count == 1:
            self.single_target_rows += 1

        for threshold in self.thresholds:
            tp = sum(iou >= threshold for iou in outcome.matched_ious)
            self.true_positives[threshold] += tp
            if tp == outcome.gt_box_count:
                self.complete_records[threshold] += 1
            if tp == outcome.gt_box_count == outcome.pred_box_count:
                self.exact_set_records[threshold] += 1
            if outcome.gt_box_count == 1 and tp >= 1:
                self.single_target_hits[threshold] += 1

    def to_dict(self) -> dict[str, Any]:
        parse_rate = safe_div(self.parse_success_rows, self.rows_with_gt_boxes)
        result: dict[str, Any] = {
            "rows": self.rows,
            "filtered_non_box_output_rows": self.filtered_non_box_output_rows,
            "rows_with_gt_boxes": self.rows_with_gt_boxes,
            "rows_without_gt_boxes": self.rows_without_gt_boxes,
            "rows_with_gt_points": self.rows_with_gt_points,
            "scored_rows": self.scored_rows,
            "coordinate_error_rows": self.coordinate_error_rows,
            "parse_success_rows": self.parse_success_rows,
            "parse_failure_rows": self.parse_failure_rows,
            "parse_success_rate": parse_rate,
            "single_target_rows": self.single_target_rows,
            "gt_boxes_total": self.gt_boxes_total,
            "gt_boxes_scored": self.gt_boxes_scored,
            "pred_boxes_scored": self.pred_boxes_scored,
            "gt_points_total": self.gt_points_total,
            "pred_points_total": self.pred_points_total,
            "invalid_pred_boxes": self.invalid_pred_boxes,
            "clipped_boxes": self.clipped_boxes,
            "reordered_boxes": self.reordered_boxes,
            "mean_iou_per_gt_box": safe_div(self.sum_iou_over_gt, self.gt_boxes_scored),
            "thresholds": {},
        }
        for threshold in self.thresholds:
            tp = self.true_positives[threshold]
            fp = self.pred_boxes_scored - tp
            fn = self.gt_boxes_scored - tp
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            key = format_threshold(threshold)
            result["thresholds"][key] = {
                "true_positive_boxes": tp,
                "false_positive_boxes": fp,
                "false_negative_boxes": fn,
                "box_precision": precision,
                "box_recall": recall,
                "box_f1": safe_div(2 * precision * recall, precision + recall),
                "all_targets_accuracy": safe_div(self.complete_records[threshold], self.scored_rows),
                "exact_set_accuracy": safe_div(self.exact_set_records[threshold], self.scored_rows),
                "single_target_accuracy": safe_div(
                    self.single_target_hits[threshold], self.single_target_rows
                ),
            }
        return result


@dataclass
class RecordItem:
    line_number: int
    record: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class CoordinateError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate grounding boxes in an ms-swift inference JSONL file."
    )
    parser.add_argument("--input", type=Path, required=True, help="Prediction JSONL from swift infer.")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional line-aligned ground-truth JSONL. By default, ground truth is read from --input.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output JSON report. Defaults to <input_stem>_iou_report.json.",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=None,
        help="Optional per-record JSONL details file.",
    )
    parser.add_argument(
        "--details-mode",
        choices=("all", "errors"),
        default="errors",
        help="Write all records or only failures to --details-output.",
    )
    parser.add_argument("--prediction-field", default="response", help="Dotted prediction field path.")
    parser.add_argument("--label-field", default="labels", help="Dotted reference-answer field path.")
    parser.add_argument("--ground-truth-field", default="objects.bbox", help="Dotted ground-truth box field.")
    parser.add_argument(
        "--ground-truth-type-field",
        default="objects.bbox_type",
        help="Dotted bbox type field. ms-swift values are real or norm1.",
    )
    parser.add_argument("--images-field", default="images", help="Dotted image list field.")
    parser.add_argument(
        "--image-id-field", default="objects.image_id", help="Dotted ground-truth image-id field."
    )
    parser.add_argument("--id-field", default="id", help="Dotted identifier field used in details.")
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        help="Dotted fields for grouped metrics, for example: --group-by source category.",
    )
    parser.add_argument(
        "--task-filter",
        choices=("auto", "all"),
        default="auto",
        help=(
            "auto evaluates only rows whose reference answer expects bbox output; "
            "all evaluates every row containing objects.bbox"
        ),
    )
    parser.add_argument(
        "--prediction-space",
        choices=("norm1000", "norm1", "real", "auto"),
        default="norm1000",
        help="Prediction coordinate space. Qwen3-VL defaults to norm1000.",
    )
    parser.add_argument(
        "--ground-truth-space",
        choices=("objects", "norm1000", "norm1", "real", "auto"),
        default="objects",
        help="Ground-truth space. objects reads objects.bbox_type and defaults to real.",
    )
    parser.add_argument(
        "--box-format",
        choices=("xyxy", "xywh"),
        default="xyxy",
        help="Coordinate order for parsed boxes.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="IoU thresholds to report.",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.5,
        help="Threshold used to select failed rows for --details-mode errors.",
    )
    parser.add_argument(
        "--matching",
        choices=("optimal", "ordered"),
        default="optimal",
        help="One-to-one matching strategy for multi-box samples.",
    )
    parser.add_argument(
        "--prediction-limit",
        type=int,
        default=0,
        help="Keep only the first N predicted boxes. Use 1 for strict RefCOCO-style evaluation; 0 keeps all.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional base directory for relative image paths. Defaults to the ground-truth JSONL directory.",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Evaluate at most N paired rows.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print progress every N rows.")
    parser.add_argument("--strict", action="store_true", help="Stop on malformed rows or coordinate errors.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite report/details outputs.")
    args = parser.parse_args()

    thresholds = set(args.thresholds)
    thresholds.add(args.primary_threshold)
    if any(not 0 <= value <= 1 for value in thresholds):
        parser.error("IoU thresholds must be between 0 and 1.")
    if args.prediction_limit < 0:
        parser.error("--prediction-limit must be non-negative.")
    args.thresholds = tuple(sorted(thresholds))
    if args.report is None:
        args.report = args.input.with_name(f"{args.input.stem}_iou_report.json")
    return args


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def format_threshold(value: float) -> str:
    return f"{value:.2f}"


def get_nested(record: Any, dotted_path: str, default: Any = None) -> Any:
    current = record
    for part in dotted_path.split("."):
        if isinstance(current, str):
            current = parse_json_like(current)
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def parse_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_numeric_sequence(value: Any, lengths: set[int]) -> bool:
    return isinstance(value, (list, tuple)) and len(value) in lengths and all(is_number(v) for v in value)


def to_image_id(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def geometry_from_value(value: Any, image_id: int = 0) -> tuple[list[Geometry], list[Geometry]]:
    boxes: list[Geometry] = []
    points: list[Geometry] = []
    if isinstance(value, str):
        pair_matches = list(PAIR_BOX_RE.finditer(value))
        if pair_matches:
            boxes.extend(
                Geometry(tuple(float(number) for number in match.groups()), image_id)
                for match in pair_matches
            )
            return boxes, points
    value = parse_json_like(value)
    if is_numeric_sequence(value, {4}):
        boxes.append(Geometry(tuple(float(v) for v in value), image_id))
    elif is_numeric_sequence(value, {2}):
        points.append(Geometry(tuple(float(v) for v in value), image_id))
    elif isinstance(value, str):
        numbers = [float(v) for v in re.findall(NUMBER_RE, value)]
        if len(numbers) == 4:
            boxes.append(Geometry(tuple(numbers), image_id))
        elif len(numbers) == 2:
            points.append(Geometry(tuple(numbers), image_id))
    elif isinstance(value, (list, tuple)):
        for item in value:
            item_boxes, item_points = geometry_from_value(item, image_id)
            boxes.extend(item_boxes)
            points.extend(item_points)
    return boxes, points


def collect_structured_geometries(value: Any, inherited_image_id: int = 0) -> ParsedPrediction:
    result = ParsedPrediction(parser="structured")
    value = parse_json_like(value)
    if is_numeric_sequence(value, {2, 4}):
        boxes, points = geometry_from_value(value, inherited_image_id)
        result.boxes.extend(boxes)
        result.points.extend(points)
        return result
    if isinstance(value, (list, tuple)):
        for item in value:
            child = collect_structured_geometries(item, inherited_image_id)
            result.boxes.extend(child.boxes)
            result.points.extend(child.points)
        return result
    if not isinstance(value, dict):
        return result

    image_id = to_image_id(value.get("image_id", value.get("image_index", inherited_image_id)))
    recognized = False
    for key in BOX_KEYS:
        if key in value:
            boxes, points = geometry_from_value(value[key], image_id)
            result.boxes.extend(boxes)
            result.points.extend(points)
            recognized = True
    for key in POINT_KEYS:
        if key in value:
            boxes, points = geometry_from_value(value[key], image_id)
            result.boxes.extend(boxes)
            result.points.extend(points)
            recognized = True

    coordinate_keys = ("x1", "y1", "x2", "y2")
    if all(key in value and is_number(value[key]) for key in coordinate_keys):
        result.boxes.append(Geometry(tuple(float(value[key]) for key in coordinate_keys), image_id))
        recognized = True
    edge_keys = ("left", "top", "right", "bottom")
    if all(key in value and is_number(value[key]) for key in edge_keys):
        result.boxes.append(Geometry(tuple(float(value[key]) for key in edge_keys), image_id))
        recognized = True

    if not recognized:
        for child_value in value.values():
            if isinstance(child_value, (dict, list, tuple)):
                child = collect_structured_geometries(child_value, image_id)
                result.boxes.extend(child.boxes)
                result.points.extend(child.points)
    return result


def final_answer_text(text: str) -> str:
    answer_matches = ANSWER_RE.findall(text)
    if answer_matches:
        return answer_matches[-1].strip()
    lower = text.lower()
    marker = "</think>"
    if marker in lower:
        index = lower.rfind(marker)
        return text[index + len(marker):].strip()
    return text.strip()


def structured_candidates(text: str) -> Iterator[Any]:
    candidates: list[str] = [text.strip()]
    candidates.extend(match.strip() for match in FENCE_RE.findall(text))
    for opening, closing in (("[", "]"), ("{", "}")):
        start = text.find(opening)
        end = text.rfind(closing)
        if 0 <= start < end:
            candidates.append(text[start:end + 1].strip())

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        parsed = parse_json_like(candidate)
        if not isinstance(parsed, str):
            yield parsed


def deduplicate_geometries(geometries: Iterable[Geometry]) -> list[Geometry]:
    result: list[Geometry] = []
    seen: set[tuple[Any, ...]] = set()
    for geometry in geometries:
        key = (geometry.image_id, *(round(value, 8) for value in geometry.coordinates))
        if key not in seen:
            seen.add(key)
            result.append(geometry)
    return result


def parse_prediction(value: Any, prediction_limit: int = 0) -> ParsedPrediction:
    if isinstance(value, (dict, list, tuple)):
        result = collect_structured_geometries(value)
        result.boxes = deduplicate_geometries(result.boxes)
        result.points = deduplicate_geometries(result.points)
        if prediction_limit:
            result.boxes = result.boxes[:prediction_limit]
        return result
    if value is None:
        return ParsedPrediction()

    text = final_answer_text(str(value))
    boxes: list[Geometry] = []
    points: list[Geometry] = []

    legacy_segments = LEGACY_BOX_RE.findall(text)
    if legacy_segments:
        for segment in legacy_segments:
            segment_boxes, segment_points = geometry_from_value(segment)
            boxes.extend(segment_boxes)
            points.extend(segment_points)
        parser_name = "qwen_legacy"
    else:
        parser_name = None
        for candidate in structured_candidates(text):
            parsed = collect_structured_geometries(candidate)
            boxes.extend(parsed.boxes)
            points.extend(parsed.points)
        if boxes or points:
            parser_name = "structured"

    if not boxes and not points:
        for segment in TAGGED_BOX_RE.findall(text):
            segment_boxes, segment_points = geometry_from_value(segment)
            boxes.extend(segment_boxes)
            points.extend(segment_points)
        if boxes or points:
            parser_name = "tagged"

    if not boxes:
        for match in BBOX_KEY_RE.finditer(text):
            boxes.append(Geometry(tuple(float(value_) for value_ in match.groups())))
        if boxes:
            parser_name = "bbox_key"

    if not points:
        for match in POINT_KEY_RE.finditer(text):
            points.append(Geometry(tuple(float(value_) for value_ in match.groups())))
        if points and parser_name is None:
            parser_name = "point_key"

    if not boxes:
        for match in PAIR_BOX_RE.finditer(text):
            boxes.append(Geometry(tuple(float(value_) for value_ in match.groups())))
        if boxes:
            parser_name = "coordinate_pairs"

    boxes = deduplicate_geometries(boxes)
    points = deduplicate_geometries(points)
    if prediction_limit:
        boxes = boxes[:prediction_limit]
    return ParsedPrediction(boxes=boxes, points=points, parser=parser_name)


def parse_ground_truth(value: Any, image_ids: Any) -> ParsedPrediction:
    value = parse_json_like(value)
    image_ids = parse_json_like(image_ids)
    if isinstance(image_ids, (int, float, str)):
        image_ids = [to_image_id(image_ids)]
    elif not isinstance(image_ids, (list, tuple)):
        image_ids = []

    if is_numeric_sequence(value, {2, 4}):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return ParsedPrediction()

    boxes: list[Geometry] = []
    points: list[Geometry] = []
    for index, item in enumerate(value):
        image_id = to_image_id(image_ids[index]) if index < len(image_ids) else 0
        item_boxes, item_points = geometry_from_value(item, image_id)
        boxes.extend(item_boxes)
        points.extend(item_points)
    return ParsedPrediction(
        boxes=deduplicate_geometries(boxes),
        points=deduplicate_geometries(points),
        parser="ground_truth",
    )


def infer_prediction_value(record: dict[str, Any], preferred_field: str) -> Any:
    value = get_nested(record, preferred_field)
    if value is not None:
        return value
    for field_name in PREDICTION_FALLBACK_FIELDS:
        value = get_nested(record, field_name)
        if value is not None:
            return value
    return None


def reference_answer(record: dict[str, Any], label_field: str, allow_message_fallback: bool) -> Any:
    value = get_nested(record, label_field)
    if value is not None:
        return value
    if allow_message_fallback:
        messages = record.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    return message.get("content")
    return None


def reference_expects_box(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        parsed = collect_structured_geometries(value)
        return bool(parsed.boxes)
    text = str(value)
    markers = ("<bbox>", "<|box_start|>", "bbox_2d", "bounding_box")
    if any(marker in text for marker in markers):
        return True
    return bool(BBOX_KEY_RE.search(text) or PAIR_BOX_RE.search(text))


def resolve_gt_space(record: dict[str, Any], args: argparse.Namespace) -> str:
    if args.ground_truth_space != "objects":
        return args.ground_truth_space
    bbox_type_value = get_nested(record, args.ground_truth_type_field, "real")
    bbox_type = "real" if bbox_type_value is None else str(bbox_type_value).lower()
    aliases = {
        "real": "real",
        "absolute": "real",
        "pixel": "real",
        "pixels": "real",
        "norm1": "norm1",
        "normalized": "norm1",
        "norm1000": "norm1000",
    }
    if bbox_type not in aliases:
        raise CoordinateError(f"Unsupported ground-truth bbox type: {bbox_type!r}")
    return aliases[bbox_type]


def extract_embedded_size(record: dict[str, Any], image_id: int) -> Optional[tuple[float, float]]:
    for field_name in ("images_size", "image_sizes", "image_size"):
        value = get_nested(record, field_name)
        value = parse_json_like(value)
        if value is None:
            continue
        if isinstance(value, dict):
            width = value.get("width", value.get("w"))
            height = value.get("height", value.get("h"))
            if is_number(width) and is_number(height):
                return float(width), float(height)
        if is_numeric_sequence(value, {2}) and image_id == 0:
            return float(value[0]), float(value[1])
        if isinstance(value, (list, tuple)) and image_id < len(value):
            item = value[image_id]
            if is_numeric_sequence(item, {2}):
                return float(item[0]), float(item[1])
            if isinstance(item, dict):
                width = item.get("width", item.get("w"))
                height = item.get("height", item.get("h"))
                if is_number(width) and is_number(height):
                    return float(width), float(height)
    if image_id == 0:
        width = record.get("image_width", record.get("width"))
        height = record.get("image_height", record.get("height"))
        if is_number(width) and is_number(height):
            return float(width), float(height)
    return None


def extract_image_path(record: dict[str, Any], images_field: str, image_id: int) -> Optional[str]:
    images = parse_json_like(get_nested(record, images_field))
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, (list, tuple)) or image_id >= len(images):
        return None
    image = images[image_id]
    if isinstance(image, dict):
        for key in ("path", "image", "url"):
            if isinstance(image.get(key), str):
                return image[key]
        return None
    if isinstance(image, (list, tuple)) and image:
        image = image[0]
    return str(image) if isinstance(image, (str, Path)) else None


@lru_cache(maxsize=100000)
def read_image_size(path: str) -> tuple[float, float]:
    try:
        from PIL import Image
    except ImportError as error:
        raise CoordinateError("Pillow is required to read image dimensions: pip install Pillow") from error
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as error:
        raise CoordinateError(f"Cannot read image size from {path!r}: {error}") from error
    if width <= 0 or height <= 0:
        raise CoordinateError(f"Invalid image size for {path!r}: {(width, height)}")
    return float(width), float(height)


def get_image_size(
    record: dict[str, Any],
    image_id: int,
    images_field: str,
    image_root: Path,
) -> tuple[float, float]:
    embedded = extract_embedded_size(record, image_id)
    if embedded is not None:
        return embedded
    image_path = extract_image_path(record, images_field, image_id)
    if not image_path:
        raise CoordinateError(f"No image path or size for image_id={image_id}")
    if image_path.startswith(("http://", "https://", "data:")):
        raise CoordinateError(f"Remote/base64 image has no embedded size: {image_path[:120]!r}")
    if image_path.startswith("file://"):
        image_path = image_path[7:]
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = image_root / path
    return read_image_size(str(path.resolve()))


def infer_coordinate_space(geometry: Geometry, image_size: Optional[tuple[float, float]]) -> str:
    values = geometry.coordinates
    if max(abs(value) for value in values) <= 1.5:
        return "norm1"
    if image_size is not None and len(values) == 4:
        width, height = image_size
        x_values = (values[0], values[2])
        y_values = (values[1], values[3])
        if max(x_values) <= width * 1.05 and max(y_values) <= height * 1.05:
            return "real"
    return "norm1000"


def normalize_geometry(
    geometry: Geometry,
    coordinate_space: str,
    record: dict[str, Any],
    args: argparse.Namespace,
    image_root: Path,
) -> tuple[Optional[Geometry], bool, bool]:
    coordinates = list(geometry.coordinates)
    image_size: Optional[tuple[float, float]] = None
    if coordinate_space in {"real", "auto"}:
        try:
            image_size = get_image_size(record, geometry.image_id, args.images_field, image_root)
        except CoordinateError:
            if coordinate_space == "real":
                raise
    if coordinate_space == "auto":
        coordinate_space = infer_coordinate_space(geometry, image_size)
    if coordinate_space == "real":
        if image_size is None:
            image_size = get_image_size(record, geometry.image_id, args.images_field, image_root)
        width, height = image_size
        if len(coordinates) == 4:
            coordinates = [
                coordinates[0] / width,
                coordinates[1] / height,
                coordinates[2] / width,
                coordinates[3] / height,
            ]
        else:
            coordinates = [coordinates[0] / width, coordinates[1] / height]
    elif coordinate_space == "norm1000":
        coordinates = [value / 1000.0 for value in coordinates]
    elif coordinate_space != "norm1":
        raise CoordinateError(f"Unsupported coordinate space: {coordinate_space}")

    if len(coordinates) == 4 and args.box_format == "xywh":
        coordinates[2] += coordinates[0]
        coordinates[3] += coordinates[1]

    reordered = False
    if len(coordinates) == 4:
        x1, y1, x2, y2 = coordinates
        reordered = x2 < x1 or y2 < y1
        coordinates = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    unclipped = list(coordinates)
    coordinates = [min(1.0, max(0.0, value)) for value in coordinates]
    clipped = any(abs(before - after) > 1e-12 for before, after in zip(unclipped, coordinates))
    if not all(math.isfinite(value) for value in coordinates):
        return None, clipped, reordered
    if len(coordinates) == 4 and (coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]):
        return None, clipped, reordered
    return Geometry(tuple(coordinates), geometry.image_id), clipped, reordered


def box_iou(first: Geometry, second: Geometry) -> float:
    if first.image_id != second.image_id:
        return 0.0
    ax1, ay1, ax2, ay2 = first.coordinates
    bx1, by1, bx2, by2 = second.coordinates
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def hungarian_minimize(cost_matrix: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    if not cost_matrix or not cost_matrix[0]:
        return []
    original_rows = len(cost_matrix)
    original_columns = len(cost_matrix[0])
    transposed = original_rows > original_columns
    if transposed:
        matrix = [
            [float(cost_matrix[row][column]) for row in range(original_rows)]
            for column in range(original_columns)
        ]
    else:
        matrix = [[float(value) for value in row] for row in cost_matrix]

    row_count = len(matrix)
    column_count = len(matrix[0])
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        column0 = 0
        minimum = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1][column - 1] - row_potential[row0] - column_potential[column]
                if current < minimum[column]:
                    minimum[column] = current
                    previous_column[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(column_count + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = previous_column[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment: list[tuple[int, int]] = []
    for column in range(1, column_count + 1):
        row = matched_row[column]
        if not row:
            continue
        pair = (row - 1, column - 1)
        assignment.append((pair[1], pair[0]) if transposed else pair)
    return assignment


def match_boxes(
    predictions: Sequence[Geometry], ground_truth: Sequence[Geometry], matching: str
) -> list[tuple[int, int, float]]:
    if matching == "ordered":
        return [
            (index, index, box_iou(predictions[index], ground_truth[index]))
            for index in range(min(len(predictions), len(ground_truth)))
        ]
    cost_matrix = [
        [1.0 - box_iou(prediction, target) for target in ground_truth]
        for prediction in predictions
    ]
    return [
        (prediction_index, gt_index, box_iou(predictions[prediction_index], ground_truth[gt_index]))
        for prediction_index, gt_index in hungarian_minimize(cost_matrix)
    ]


def normalize_boxes(
    geometries: Sequence[Geometry],
    coordinate_space: str,
    record: dict[str, Any],
    args: argparse.Namespace,
    image_root: Path,
    is_ground_truth: bool,
) -> tuple[list[Geometry], int, int, int]:
    normalized: list[Geometry] = []
    invalid = 0
    clipped = 0
    reordered = 0
    for geometry in geometries:
        item, was_clipped, was_reordered = normalize_geometry(
            geometry, coordinate_space, record, args, image_root
        )
        clipped += int(was_clipped)
        reordered += int(was_reordered)
        if item is None:
            if is_ground_truth:
                raise CoordinateError(f"Invalid ground-truth box: {geometry.coordinates}")
            invalid += 1
            continue
        normalized.append(item)
    return normalized, invalid, clipped, reordered


def evaluate_record(
    prediction_record: dict[str, Any],
    ground_truth_record: dict[str, Any],
    args: argparse.Namespace,
    image_root: Path,
) -> tuple[EvaluationOutcome, dict[str, Any]]:
    prediction_value = infer_prediction_value(prediction_record, args.prediction_field)
    parsed_prediction = parse_prediction(prediction_value, args.prediction_limit)
    gt_value = get_nested(ground_truth_record, args.ground_truth_field)
    image_ids = get_nested(ground_truth_record, args.image_id_field)
    parsed_gt = parse_ground_truth(gt_value, image_ids)

    label_value = reference_answer(
        ground_truth_record,
        args.label_field,
        allow_message_fallback=args.ground_truth is not None,
    )
    expects_box = reference_expects_box(label_value)

    outcome = EvaluationOutcome(
        status="no_gt_box",
        gt_box_count=len(parsed_gt.boxes),
        pred_box_count=len(parsed_prediction.boxes),
        gt_point_count=len(parsed_gt.points),
        pred_point_count=len(parsed_prediction.points),
        parse_success=bool(parsed_prediction.boxes),
        parser=parsed_prediction.parser,
    )
    detail: dict[str, Any] = {
        "prediction": prediction_value,
        "parser": parsed_prediction.parser,
        "pred_boxes_raw": [list(item.coordinates) for item in parsed_prediction.boxes],
        "gt_boxes_raw": [list(item.coordinates) for item in parsed_gt.boxes],
        "pred_points_raw": [list(item.coordinates) for item in parsed_prediction.points],
        "gt_points_raw": [list(item.coordinates) for item in parsed_gt.points],
        "reference_expects_box": expects_box,
    }
    if args.task_filter == "auto" and expects_box is False:
        outcome.status = "filtered_non_box_output"
        detail["status"] = outcome.status
        return outcome, detail
    if not parsed_gt.boxes:
        return outcome, detail

    if not parsed_prediction.boxes:
        outcome.status = "parse_failure"
        outcome.best_gt_ious = [0.0] * len(parsed_gt.boxes)
        outcome.matched_ious = []
        detail.update({"status": outcome.status, "best_gt_ious": outcome.best_gt_ious})
        return outcome, detail

    try:
        gt_space = resolve_gt_space(ground_truth_record, args)
        normalized_gt, _, gt_clipped, gt_reordered = normalize_boxes(
            parsed_gt.boxes, gt_space, ground_truth_record, args, image_root, is_ground_truth=True
        )
        normalized_pred, invalid_pred, pred_clipped, pred_reordered = normalize_boxes(
            parsed_prediction.boxes,
            args.prediction_space,
            ground_truth_record,
            args,
            image_root,
            is_ground_truth=False,
        )
    except CoordinateError as error:
        outcome.status = "coordinate_error"
        outcome.error = str(error)
        detail.update({"status": outcome.status, "error": outcome.error})
        if args.strict:
            raise
        return outcome, detail

    outcome.invalid_pred_boxes = invalid_pred
    outcome.clipped_boxes = gt_clipped + pred_clipped
    outcome.reordered_boxes = gt_reordered + pred_reordered
    # Invalid predicted boxes still count as false positives.
    outcome.pred_box_count = len(normalized_pred) + invalid_pred
    if not normalized_pred:
        outcome.status = "parse_failure"
        outcome.parse_success = False
        outcome.best_gt_ious = [0.0] * len(normalized_gt)
        detail.update({"status": outcome.status, "best_gt_ious": outcome.best_gt_ious})
        return outcome, detail

    matches = match_boxes(normalized_pred, normalized_gt, args.matching)
    best_gt_ious = [0.0] * len(normalized_gt)
    for _, gt_index, iou in matches:
        best_gt_ious[gt_index] = iou
    outcome.status = "scored"
    outcome.best_gt_ious = best_gt_ious
    outcome.matched_ious = [item[2] for item in matches]
    detail.update(
        {
            "status": outcome.status,
            "pred_boxes_norm1": [list(item.coordinates) for item in normalized_pred],
            "gt_boxes_norm1": [list(item.coordinates) for item in normalized_gt],
            "matches": [
                {"prediction_index": pred_index, "ground_truth_index": gt_index, "iou": iou}
                for pred_index, gt_index, iou in matches
            ],
            "best_gt_ious": best_gt_ious,
            "mean_iou": safe_div(sum(best_gt_ious), len(best_gt_ious)),
        }
    )
    return outcome, detail


def iter_jsonl(path: Path, strict: bool) -> Iterator[RecordItem]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
                if not isinstance(record, dict):
                    raise ValueError("JSONL row is not an object")
                yield RecordItem(line_number=line_number, record=record)
            except (json.JSONDecodeError, ValueError) as error:
                if strict:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                yield RecordItem(line_number=line_number, error=str(error))


def output_path_guard(path: Optional[Path], overwrite: bool) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def detail_is_error(outcome: EvaluationOutcome, primary_threshold: float) -> bool:
    if outcome.status != "scored":
        return True
    tp = sum(iou >= primary_threshold for iou in outcome.matched_ious)
    return tp != outcome.gt_box_count or outcome.pred_box_count != outcome.gt_box_count


def group_name(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {args.input}")
    if args.ground_truth is not None and not args.ground_truth.is_file():
        raise FileNotFoundError(f"Ground-truth file does not exist: {args.ground_truth}")
    output_path_guard(args.report, args.overwrite)
    output_path_guard(args.details_output, args.overwrite)

    image_root = args.image_root
    if image_root is None:
        image_root = (args.ground_truth or args.input).resolve().parent
    else:
        image_root = image_root.resolve()

    prediction_items = iter_jsonl(args.input, args.strict)
    if args.ground_truth is None:
        paired_items: Iterable[tuple[Optional[RecordItem], Optional[RecordItem]]] = (
            (item, item) for item in prediction_items
        )
    else:
        gt_items = iter_jsonl(args.ground_truth, args.strict)
        paired_items = zip_longest(prediction_items, gt_items)

    global_metrics = MetricAccumulator(args.thresholds)
    grouped_metrics: dict[str, dict[str, MetricAccumulator]] = {
        field_name: {} for field_name in args.group_by
    }
    parser_counts: dict[str, int] = {}
    invalid_prediction_rows = 0
    invalid_ground_truth_rows = 0
    length_mismatch = False
    processed = 0
    error_examples: list[dict[str, Any]] = []
    details_handle: Optional[TextIO] = None
    if args.details_output is not None:
        details_handle = args.details_output.open("w", encoding="utf-8", newline="\n")

    try:
        for prediction_item, gt_item in paired_items:
            if args.max_records is not None and processed >= args.max_records:
                break
            processed += 1
            if prediction_item is None or gt_item is None:
                length_mismatch = True
                if args.strict:
                    raise ValueError("Prediction and ground-truth JSONL files have different row counts.")
                break
            if prediction_item.error:
                invalid_prediction_rows += 1
                if len(error_examples) < 20:
                    error_examples.append(
                        {"line": prediction_item.line_number, "type": "invalid_prediction_json", "error": prediction_item.error}
                    )
                continue
            if gt_item.error:
                invalid_ground_truth_rows += 1
                if len(error_examples) < 20:
                    error_examples.append(
                        {"line": gt_item.line_number, "type": "invalid_ground_truth_json", "error": gt_item.error}
                    )
                continue

            assert prediction_item.record is not None
            assert gt_item.record is not None
            try:
                outcome, detail = evaluate_record(
                    prediction_item.record, gt_item.record, args, image_root
                )
            except Exception as error:
                if args.strict:
                    raise
                outcome = EvaluationOutcome(status="coordinate_error", error=str(error))
                detail = {"status": outcome.status, "error": outcome.error}

            global_metrics.update(outcome)
            if outcome.parser:
                parser_counts[outcome.parser] = parser_counts.get(outcome.parser, 0) + 1

            for field_name in args.group_by:
                value = group_name(get_nested(gt_item.record, field_name))
                accumulator = grouped_metrics[field_name].setdefault(
                    value, MetricAccumulator(args.thresholds)
                )
                accumulator.update(outcome)

            record_id = get_nested(gt_item.record, args.id_field)
            detail.update(
                {
                    "line": prediction_item.line_number,
                    "id": record_id,
                    "images": get_nested(gt_item.record, args.images_field),
                    "gt_box_count": outcome.gt_box_count,
                    "pred_box_count": outcome.pred_box_count,
                }
            )
            for field_name in args.group_by:
                detail[field_name] = get_nested(gt_item.record, field_name)

            if outcome.error and len(error_examples) < 20:
                error_examples.append(
                    {
                        "line": prediction_item.line_number,
                        "id": record_id,
                        "type": outcome.status,
                        "error": outcome.error,
                    }
                )
            if details_handle is not None and (
                args.details_mode == "all" or detail_is_error(outcome, args.primary_threshold)
            ):
                details_handle.write(json.dumps(detail, ensure_ascii=False) + "\n")

            if args.progress_every > 0 and processed % args.progress_every == 0:
                metrics = global_metrics.to_dict()
                print(
                    "[progress] "
                    f"rows={processed:,} scored={metrics['scored_rows']:,} "
                    f"parse_success={metrics['parse_success_rows']:,} "
                    f"coordinate_errors={metrics['coordinate_error_rows']:,} "
                    f"mIoU={metrics['mean_iou_per_gt_box']:.4f}",
                    flush=True,
                )
    finally:
        if details_handle is not None:
            details_handle.close()

    metrics_dict = global_metrics.to_dict()
    report = {
        "input": str(args.input.resolve()),
        "ground_truth": str(args.ground_truth.resolve()) if args.ground_truth else None,
        "image_root": str(image_root),
        "arguments": serialize_args(args),
        "input_validation": {
            "processed_pairs": processed,
            "invalid_prediction_rows": invalid_prediction_rows,
            "invalid_ground_truth_rows": invalid_ground_truth_rows,
            "length_mismatch": length_mismatch,
            "error_examples": error_examples,
        },
        "parsers": dict(sorted(parser_counts.items())),
        "metrics": metrics_dict,
        "groups": {
            field_name: {
                value: accumulator.to_dict()
                for value, accumulator in sorted(values.items())
            }
            for field_name, values in grouped_metrics.items()
        },
    }
    with args.report.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    threshold_key = format_threshold(args.primary_threshold)
    threshold_metrics = metrics_dict["thresholds"][threshold_key]
    print(
        "[summary] "
        f"rows={metrics_dict['rows']:,} gt_box_rows={metrics_dict['rows_with_gt_boxes']:,} "
        f"scored={metrics_dict['scored_rows']:,} parse_rate={metrics_dict['parse_success_rate']:.4f} "
        f"coordinate_errors={metrics_dict['coordinate_error_rows']:,}"
    )
    print(
        f"[iou@{threshold_key}] "
        f"mIoU={metrics_dict['mean_iou_per_gt_box']:.4f} "
        f"single_target_accuracy={threshold_metrics['single_target_accuracy']:.4f} "
        f"precision={threshold_metrics['box_precision']:.4f} "
        f"recall={threshold_metrics['box_recall']:.4f} "
        f"f1={threshold_metrics['box_f1']:.4f} "
        f"exact_set_accuracy={threshold_metrics['exact_set_accuracy']:.4f}"
    )
    print(f"[report] {args.report}")
    if args.details_output is not None:
        print(f"[details] {args.details_output}")

    if not metrics_dict["rows_with_gt_boxes"]:
        print("[error] No ground-truth boxes were found.", file=sys.stderr)
        return 2
    if not metrics_dict["scored_rows"]:
        print("[error] No rows could be scored.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
