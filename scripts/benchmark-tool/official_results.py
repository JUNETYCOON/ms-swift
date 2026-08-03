import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from .records import EvaluationRecord
    from .registry import metric_spec_for, normalize_name
except ImportError:
    from records import EvaluationRecord
    from registry import metric_spec_for, normalize_name


def _number(value: Any) -> Tuple[Optional[float], bool]:
    if isinstance(value, bool):
        return float(value), False
    if isinstance(value, (int, float)):
        number = float(value)
        return (number, False) if math.isfinite(number) else (None, False)
    if not isinstance(value, str):
        return None, False
    text = value.strip()
    if not text:
        return None, False
    has_percent = text.endswith('%')
    if has_percent:
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None, False
    if not math.isfinite(number):
        return None, False
    return number, has_percent


def _normalize_value(value: Any, benchmark: str, metric: str, scale: Optional[float] = None) -> float:
    number, has_percent = _number(value)
    if number is None:
        raise ValueError(f'Official result for `{benchmark}/{metric}` is not numeric: {value!r}.')
    if scale is not None:
        return number * scale
    if has_percent:
        return number / 100.0
    spec = metric_spec_for(benchmark, metric)
    if spec and spec.percentage and 1.0 < number <= 100.0:
        return number / 100.0
    return number


def _json_pointer(data: Any, pointer: str) -> Any:
    if pointer == '':
        return data
    if not pointer.startswith('/'):
        current = data
        for part in pointer.split('.'):
            if not isinstance(current, Mapping) or part not in current:
                raise KeyError(pointer)
            current = current[part]
        return current

    current = data
    for raw_part in pointer[1:].split('/'):
        part = raw_part.replace('~1', '/').replace('~0', '~')
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            current = current[int(part)]
        elif isinstance(current, Mapping):
            current = current[part]
        else:
            raise KeyError(pointer)
    return current


def _load_json(path: Path, result_format: str) -> Any:
    if result_format == 'json':
        with path.open('r', encoding='utf-8-sig') as f:
            return json.load(f)
    records = []
    with path.open('r', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f'Official JSONL result is empty: {path}.')
    return records[-1]


def _select_csv_row(rows: Sequence[Mapping[str, str]], config: Mapping[str, Any]) -> Mapping[str, str]:
    if not rows:
        raise ValueError('Official CSV result contains no rows.')
    row_filter = config.get('row_filter', {})
    if row_filter:
        rows = [
            row for row in rows
            if all(str(row.get(str(key), '')) == str(expected) for key, expected in row_filter.items())
        ]
        if not rows:
            raise ValueError(f'No official CSV row matched row_filter={row_filter!r}.')
    row_index = int(config.get('row_index', -1))
    try:
        return rows[row_index]
    except IndexError as e:
        raise ValueError(f'Official CSV row_index {row_index} is out of range.') from e


def _selector_details(selector: Any, location_key: str) -> Tuple[str, Optional[float]]:
    if isinstance(selector, str):
        return selector, None
    if not isinstance(selector, Mapping):
        raise ValueError(f'Metric selector must be a string or object, got {selector!r}.')
    location = selector.get(location_key) or selector.get('path')
    if not isinstance(location, str) or not location:
        raise ValueError(f'Metric selector requires `{location_key}`: {selector!r}.')
    scale = selector.get('scale')
    return location, float(scale) if scale is not None else None


def _canonical_config_metrics(benchmark: str, selectors: Mapping[str, Any]) -> Dict[str, Any]:
    canonical: Dict[str, Any] = {}
    for raw_metric, selector in selectors.items():
        spec = metric_spec_for(benchmark, raw_metric)
        if spec is None:
            raise ValueError(f'Metric `{raw_metric}` is not registered for benchmark `{benchmark}`.')
        canonical[spec.name] = selector
    return canonical


def _extract_selected_json(data: Any, benchmark: str, selectors: Mapping[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for metric, selector in _canonical_config_metrics(benchmark, selectors).items():
        pointer, scale = _selector_details(selector, 'path')
        try:
            value = _json_pointer(data, pointer)
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f'Cannot find official metric `{metric}` at JSON path `{pointer}`.') from e
        metrics[metric] = _normalize_value(value, benchmark, metric, scale)
    return metrics


def _extract_selected_csv(row: Mapping[str, str], benchmark: str,
                          selectors: Mapping[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for metric, selector in _canonical_config_metrics(benchmark, selectors).items():
        column, scale = _selector_details(selector, 'column')
        if column not in row:
            raise ValueError(f'Cannot find official metric `{metric}` in CSV column `{column}`.')
        metrics[metric] = _normalize_value(row[column], benchmark, metric, scale)
    return metrics


def _extract_selected_text(text: str, benchmark: str, selectors: Mapping[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for metric, selector in _canonical_config_metrics(benchmark, selectors).items():
        pattern, scale = _selector_details(selector, 'regex')
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is None or match.lastindex is None:
            raise ValueError(f'Cannot extract official metric `{metric}` with regex `{pattern}`.')
        metrics[metric] = _normalize_value(match.group(1), benchmark, metric, scale)
    return metrics


def _walk_metric_values(data: Any, benchmark: str, found: Dict[str, List[Any]]) -> None:
    if isinstance(data, Mapping):
        for key, value in data.items():
            spec = metric_spec_for(benchmark, str(key))
            number, _ = _number(value)
            if spec is not None and number is not None:
                found.setdefault(spec.name, []).append(value)
            if isinstance(value, (Mapping, list, tuple)):
                _walk_metric_values(value, benchmark, found)
    elif isinstance(data, (list, tuple)):
        for value in data:
            _walk_metric_values(value, benchmark, found)


def _extract_automatic(data: Any, benchmark: str) -> Dict[str, float]:
    found: Dict[str, List[Any]] = {}
    _walk_metric_values(data, benchmark, found)
    metrics: Dict[str, float] = {}
    for metric, values in found.items():
        normalized = [_normalize_value(value, benchmark, metric) for value in values]
        distinct = []
        for value in normalized:
            if not any(math.isclose(value, existing, rel_tol=1e-9, abs_tol=1e-12) for existing in distinct):
                distinct.append(value)
        if len(distinct) > 1:
            raise ValueError(
                f'Official result contains multiple values for `{benchmark}/{metric}`: {distinct}. '
                'Add an explicit metric selector to the runner config.')
        metrics[metric] = distinct[0]
    return metrics


def extract_official_metrics(path: Path, benchmark: str,
                             config: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    config = dict(config or {})
    result_format = str(config.get('format') or path.suffix.lstrip('.')).lower()
    selectors = config.get('metrics')
    if selectors is not None and not isinstance(selectors, Mapping):
        raise ValueError('Runner result `metrics` must be an object mapping metric names to selectors.')

    if result_format in {'json', 'jsonl'}:
        data = _load_json(path, result_format)
        metrics = _extract_selected_json(data, benchmark, selectors) if selectors else _extract_automatic(
            data, benchmark)
    elif result_format == 'csv':
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            row = _select_csv_row(list(csv.DictReader(f)), config)
        metrics = _extract_selected_csv(row, benchmark, selectors) if selectors else _extract_automatic(
            row, benchmark)
    elif result_format in {'text', 'txt', 'log'}:
        if not selectors:
            raise ValueError('Text official results require regex metric selectors in the runner config.')
        metrics = _extract_selected_text(path.read_text(encoding='utf-8', errors='replace'), benchmark, selectors)
    else:
        raise ValueError(f'Unsupported official result format `{result_format}` for {path}.')

    if not metrics:
        raise ValueError(
            f'No registered official metrics were found in {path} for benchmark `{benchmark}`. '
            'Add metric selectors to the runner config.')
    return metrics


def write_score_csv(record: EvaluationRecord, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'benchmark', *record.metrics.keys()])
        writer.writeheader()
        row = {'model': record.model, 'benchmark': record.benchmark}
        row.update(record.metrics)
        writer.writerow(row)
