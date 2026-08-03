import ast
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from .registry import normalize_name
except ImportError:
    from registry import normalize_name


ID_KEYS = ('id', 'sample_id', 'image_id', 'question_id', 'uid')
PRED_BOX_KEYS = ('pred_box', 'prediction_box', 'pred_bbox', 'prediction_bbox', 'bbox_pred', 'box', 'bbox')
GT_BOX_KEYS = ('gt_box', 'ground_truth_box', 'target_box', 'answer_box', 'gt_bbox', 'bbox_gt', 'box', 'bbox')
BOX_FIELD_SETS = (
    ('x1', 'y1', 'x2', 'y2'),
    ('xmin', 'ymin', 'xmax', 'ymax'),
    ('left', 'top', 'right', 'bottom'),
)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _normalize_box(box_a)
    bx1, by1, bx2, by2 = _normalize_box(box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0.0 else inter_area / union


def score_grounding(
        prediction_file: Path,
        annotation_file: Path,
        model: str,
        dataset: str,
        output_file: Path,
        iou_threshold: float = 0.5) -> Dict[str, float]:
    predictions = _load_items(prediction_file)
    annotations = _load_items(annotation_file)
    annotation_by_id = {_item_id(item): item for item in annotations}
    ious: List[float] = []

    for prediction in predictions:
        item_id = _item_id(prediction)
        if item_id not in annotation_by_id:
            continue
        pred_box = _extract_box(prediction, PRED_BOX_KEYS)
        gt_box = _extract_box(annotation_by_id[item_id], GT_BOX_KEYS)
        if pred_box is None or gt_box is None:
            continue
        ious.append(box_iou(pred_box, gt_box))

    if not ious:
        raise ValueError('No matched grounding boxes found. Check ids and bbox fields in prediction/annotation files.')

    metrics = {
        'mean_iou': sum(ious) / len(ious),
        'acc_iou_0_5': sum(1 for value in ious if value >= iou_threshold) / len(ious),
        'num_samples': float(len(ious)),
    }
    _write_result(output_file, model, dataset, metrics)
    return metrics


def _load_items(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        with path.open('r', encoding='utf-8-sig') as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == '.csv':
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    if suffix == '.json':
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ('data', 'items', 'annotations', 'predictions', 'results'):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            return [
                dict({'id': key}, **value) if isinstance(value, dict) else {
                    'id': key,
                    'value': value
                } for key, value in data.items()
            ]
    raise ValueError(f'Unsupported grounding file format: {path}')


def _item_id(item: Mapping[str, Any]) -> str:
    lowered = {normalize_name(str(key)): value for key, value in item.items()}
    for key in ID_KEYS:
        value = lowered.get(normalize_name(key))
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f'Missing id field in item: {item}')


def _extract_box(item: Mapping[str, Any], box_keys: Iterable[str]) -> Optional[Tuple[float, float, float, float]]:
    lowered = {normalize_name(str(key)): value for key, value in item.items()}
    for key in box_keys:
        value = lowered.get(normalize_name(key))
        box = _parse_box_value(value)
        if box is not None:
            return box
    for field_set in BOX_FIELD_SETS:
        values = [lowered.get(normalize_name(field)) for field in field_set]
        if all(value is not None for value in values):
            return tuple(float(value) for value in values)
    return None


def _parse_box_value(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None or value == '':
        return None
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                value = [part.strip() for part in text.replace(';', ',').split(',')]
    if isinstance(value, dict):
        for field_set in BOX_FIELD_SETS:
            normalized = {normalize_name(str(key)): val for key, val in value.items()}
            values = [normalized.get(normalize_name(field)) for field in field_set]
            if all(val is not None for val in values):
                return tuple(float(val) for val in values)
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return tuple(float(item) for item in value[:4])
    return None


def _normalize_box(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _write_result(output_file: Path, model: str, dataset: str, metrics: Mapping[str, float]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'benchmark', *metrics.keys()])
        writer.writeheader()
        row = {'model': model, 'benchmark': dataset}
        row.update(metrics)
        writer.writerow(row)
