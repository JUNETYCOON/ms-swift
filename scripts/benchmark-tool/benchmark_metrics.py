import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from .registry import canonical_benchmark_name, metric_spec_for, normalize_name
except ImportError:
    from registry import canonical_benchmark_name, metric_spec_for, normalize_name

try:
    from .custom_eval import vqa_scores
except ImportError:
    from custom_eval import vqa_scores


PRED_KEYS = ('response', 'prediction', 'pred', 'predict', 'output', 'generated_text')
LABEL_KEYS = ('labels', 'label', 'answer', 'answers', 'target', 'ground_truth')
NLG_METRICS = ('rouge_1', 'rouge_2', 'rouge_l', 'bleu_4')
TYPE_AWARE_VQA_METRICS = (
    'answer_accuracy', 'semantic_similarity', 'yes_no_accuracy', 'freeform_accuracy', 'freeform_similarity')


def score_custom_file(dataset: str, model: str, input_file: Path, output_file: Path,
                      metrics: Sequence[str]) -> Dict[str, float]:
    dataset = canonical_benchmark_name(dataset)
    items = _load_items(input_file)
    if not items:
        raise ValueError(f'No records found in {input_file}.')
    scores = _score_custom_dataset(items, metrics)
    _write_result(output_file, model, dataset, scores)
    return scores


def _score_custom_dataset(items: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    normalized_metrics = [normalize_name(metric) for metric in metrics]
    if 'acc' in normalized_metrics:
        preds, labels = _extract_pred_label_pairs(items)
        scores['acc'] = sum(pred == label for pred, label in zip(preds, labels)) / len(preds)
    if any(metric in {'rouge', 'nlg'} or metric in NLG_METRICS for metric in normalized_metrics):
        preds, labels = _extract_pred_label_pairs(items)
        nlg_scores = _compute_swift_rouge_bleu(preds, labels)
        if any(metric in {'rouge', 'nlg'} for metric in normalized_metrics):
            scores.update(nlg_scores)
        else:
            for metric in normalized_metrics:
                if metric in nlg_scores:
                    scores[metric] = nlg_scores[metric]
    requested_type_aware = [metric for metric in normalized_metrics if metric in TYPE_AWARE_VQA_METRICS]
    if requested_type_aware:
        preds, labels = _extract_pred_label_pairs(items)
        type_aware_scores = _compute_type_aware_vqa(preds, labels)
        scores.update({metric: type_aware_scores[metric] for metric in requested_type_aware})

    unsupported = [
        metric for metric in normalized_metrics
        if metric not in scores and metric not in {'rouge', 'nlg'} and metric not in NLG_METRICS
        and metric not in TYPE_AWARE_VQA_METRICS
    ]
    if unsupported:
        raise ValueError(
            f'Unsupported custom dataset metric(s): {", ".join(unsupported)}. '
            'Official benchmarks must be evaluated with `vlm-eval run`.')
    return scores


def _extract_pred_label_pairs(items: Sequence[Mapping[str, Any]]) -> tuple:
    preds, labels = [], []
    for item in items:
        pred = _extract_text(item, PRED_KEYS)
        label = _extract_text(item, LABEL_KEYS)
        if pred is None or label is None:
            continue
        preds.append(pred)
        labels.append(label)
    if not preds:
        raise ValueError(
            'Cannot compute ms-swift custom metric: no response/labels pairs found. '
            f'Prediction fields: {PRED_KEYS}; label fields: {LABEL_KEYS}.')
    return preds, labels


def _compute_type_aware_vqa(preds: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    per_sample = [vqa_scores(prediction, [label]) for prediction, label in zip(preds, labels)]
    yes_no = [score for score in per_sample if score['answer_type'] == 'yes_no']
    freeform = [score for score in per_sample if score['answer_type'] == 'freeform']

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        'answer_accuracy': mean([score['answer_accuracy'] for score in per_sample]),
        'semantic_similarity': mean([score['semantic_similarity'] for score in per_sample]),
        'yes_no_accuracy': mean([score['answer_accuracy'] for score in yes_no]),
        'freeform_accuracy': mean([score['answer_accuracy'] for score in freeform]),
        'freeform_similarity': mean([score['semantic_similarity'] for score in freeform]),
    }


def _extract_text(item: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    normalized = {normalize_name(str(key)): value for key, value in item.items()}
    for key in keys:
        value = normalized.get(normalize_name(key))
        if value is None:
            continue
        if isinstance(value, list):
            value = value[0] if len(value) == 1 else ' '.join(str(part) for part in value)
        return str(value)
    return None


def _compute_swift_rouge_bleu(preds: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    try:
        from swift.metrics import compute_rouge_bleu
    except ImportError as e:
        raise RuntimeError('ms-swift rouge metric requires `swift.metrics.compute_rouge_bleu`.') from e
    raw_scores = compute_rouge_bleu(list(preds), list(labels))
    return {normalize_name(key): value for key, value in raw_scores.items()}


def _load_items(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        with path.open('r', encoding='utf-8-sig') as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == '.csv':
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    if suffix == '.json':
        with path.open('r', encoding='utf-8-sig') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ('data', 'items', 'results', 'records'):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            return [data]
    raise ValueError(f'Unsupported custom score file format: {path}.')


def _write_result(output_file: Path, model: str, dataset: str, metrics: Mapping[str, float]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'benchmark', *metrics.keys()])
        writer.writeheader()
        row = {'model': model, 'benchmark': dataset}
        row.update(metrics)
        writer.writerow(row)
