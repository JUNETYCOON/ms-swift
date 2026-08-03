import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from .records import EvaluationRecord
    from .registry import canonical_metric_name, normalize_name
except ImportError:
    from records import EvaluationRecord
    from registry import canonical_metric_name, normalize_name


MODEL_KEYS = ('model', 'model_name', 'policy', 'agent')
BENCHMARK_KEYS = ('benchmark', 'bench', 'suite', 'dataset', 'env')
METRIC_CONTAINER_KEYS = ('metrics', 'metric', 'results', 'result', 'scores', 'score')


def _to_float(value: Any, percentage: bool = False) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        has_percent = text.endswith('%')
        if has_percent:
            text = text[:-1].strip()
        try:
            number = float(text)
        except ValueError:
            return None
        if has_percent:
            number /= 100.0
    else:
        return None
    if percentage and 1.0 < number <= 100.0:
        number /= 100.0
    return number


def _find_field(data: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    lowered = {normalize_name(str(key)): value for key, value in data.items()}
    for key in keys:
        value = lowered.get(normalize_name(key))
        if value is not None:
            return value
    return None


def _infer_from_path(path: Path, candidates: Sequence[str]) -> Optional[str]:
    normalized_candidates = {normalize_name(candidate): candidate for candidate in candidates}
    for part in path.parts:
        normalized_part = normalize_name(part)
        stem_part = normalize_name(Path(part).stem)
        if normalized_part in normalized_candidates:
            return normalized_candidates[normalized_part]
        if stem_part in normalized_candidates:
            return normalized_candidates[stem_part]
    return None


def _extract_metrics(data: Dict[str, Any], benchmarks: Sequence[str],
                     percentage_metrics: Iterable[str]) -> Dict[str, float]:
    metric_data: Dict[str, Any] = {}
    for key in METRIC_CONTAINER_KEYS:
        value = _find_field(data, (key, ))
        if isinstance(value, dict):
            metric_data.update(value)
    metric_data.update(data)

    percentage_set = {normalize_name(metric) for metric in percentage_metrics}
    metrics: Dict[str, float] = {}
    for raw_key, raw_value in metric_data.items():
        metric_name = canonical_metric_name(str(raw_key), benchmarks)
        if metric_name is None:
            continue
        value = _to_float(raw_value, percentage=normalize_name(metric_name) in percentage_set)
        if value is not None:
            metrics[metric_name] = value
    return metrics


def _looks_like_record(data: Dict[str, Any], benchmarks: Sequence[str]) -> bool:
    if _find_field(data, MODEL_KEYS) is not None or _find_field(data, BENCHMARK_KEYS) is not None:
        return True
    if any(isinstance(_find_field(data, (key, )), dict) for key in METRIC_CONTAINER_KEYS):
        return True
    return any(canonical_metric_name(str(key), benchmarks) is not None for key in data)


def _extract_json_records(
        data: Any,
        path: Path,
        models: Sequence[str],
        benchmarks: Sequence[str],
        percentage_metrics: Iterable[str],
        model_context: Optional[str] = None,
        benchmark_context: Optional[str] = None) -> Iterator[EvaluationRecord]:
    if isinstance(data, list):
        for item in data:
            yield from _extract_json_records(
                item,
                path,
                models,
                benchmarks,
                percentage_metrics,
                model_context=model_context,
                benchmark_context=benchmark_context)
        return

    if not isinstance(data, dict):
        return

    if _looks_like_record(data, benchmarks):
        model = _find_field(data, MODEL_KEYS) or model_context or _infer_from_path(path, models)
        benchmark = _find_field(data, BENCHMARK_KEYS) or benchmark_context or _infer_from_path(path, benchmarks)
        if model is not None and benchmark is not None:
            metrics = _extract_metrics(data, benchmarks, percentage_metrics)
            if metrics:
                yield EvaluationRecord(model=str(model), benchmark=str(benchmark), metrics=metrics)

    for key, value in data.items():
        if not isinstance(value, (dict, list)):
            continue
        normalized_key = normalize_name(str(key))
        next_model = model_context
        next_benchmark = benchmark_context
        for model in models:
            if normalized_key == normalize_name(model):
                next_model = model
                break
        for benchmark in benchmarks:
            if normalized_key == normalize_name(benchmark):
                next_benchmark = benchmark
                break
        yield from _extract_json_records(
            value,
            path,
            models,
            benchmarks,
            percentage_metrics,
            model_context=next_model,
            benchmark_context=next_benchmark)


def _load_json(path: Path, models: Sequence[str], benchmarks: Sequence[str],
               percentage_metrics: Iterable[str]) -> List[EvaluationRecord]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return list(_extract_json_records(data, path, models, benchmarks, percentage_metrics))


def _load_jsonl(path: Path, models: Sequence[str], benchmarks: Sequence[str],
                percentage_metrics: Iterable[str]) -> List[EvaluationRecord]:
    records: List[EvaluationRecord] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.extend(_extract_json_records(json.loads(line), path, models, benchmarks, percentage_metrics))
    return records


def _load_csv(path: Path, models: Sequence[str], benchmarks: Sequence[str],
              percentage_metrics: Iterable[str]) -> List[EvaluationRecord]:
    records: List[EvaluationRecord] = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = _find_field(row, MODEL_KEYS) or _infer_from_path(path, models)
            benchmark = _find_field(row, BENCHMARK_KEYS) or _infer_from_path(path, benchmarks)
            if model is None or benchmark is None:
                continue
            metrics = _extract_metrics(row, benchmarks, percentage_metrics)
            if metrics:
                records.append(EvaluationRecord(model=str(model), benchmark=str(benchmark), metrics=metrics))
    return records


def discover_result_files(results_dir: Optional[Path]) -> List[Path]:
    if results_dir is None or not results_dir.exists():
        return []
    suffixes = {'.json', '.jsonl', '.csv'}
    return sorted(path for path in results_dir.rglob('*') if path.is_file() and path.suffix.lower() in suffixes)


def load_records(paths: Sequence[Path], models: Sequence[str], benchmarks: Sequence[str],
                 percentage_metrics: Iterable[str]) -> List[EvaluationRecord]:
    records: List[EvaluationRecord] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == '.json':
            records.extend(_load_json(path, models, benchmarks, percentage_metrics))
        elif suffix == '.jsonl':
            records.extend(_load_jsonl(path, models, benchmarks, percentage_metrics))
        elif suffix == '.csv':
            records.extend(_load_csv(path, models, benchmarks, percentage_metrics))
    return records


def aggregate_records(records: Sequence[EvaluationRecord], models: Sequence[str],
                      benchmarks: Sequence[str]) -> List[EvaluationRecord]:
    model_set = {normalize_name(model): model for model in models}
    benchmark_set = {normalize_name(benchmark): benchmark for benchmark in benchmarks}
    buckets: Dict[Tuple[str, str, str], List[float]] = {}

    for record in records:
        model = model_set.get(normalize_name(record.model))
        benchmark = benchmark_set.get(normalize_name(record.benchmark))
        if model is None or benchmark is None:
            continue
        for metric, value in record.metrics.items():
            buckets.setdefault((model, benchmark, normalize_name(metric)), []).append(value)

    by_record: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (model, benchmark, metric), values in buckets.items():
        by_record.setdefault((model, benchmark), {})[metric] = sum(values) / len(values)

    return [
        EvaluationRecord(model=model, benchmark=benchmark, metrics=metrics)
        for (model, benchmark), metrics in sorted(by_record.items())
    ]
