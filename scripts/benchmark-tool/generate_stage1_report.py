#!/usr/bin/env python3
"""Generate a standalone CSV and HTML report for the stage-1 evaluation queue."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_STAGE_ROOT = Path('/mnt/luojunkun/stage1')
MODEL_ORDER = {'baseline': 0, 'ours': 1}
OFFICIAL_MODELS = ('baseline', 'ours')
CUSTOM_PRIMARY = {'vqa': 'vqa_accuracy', 'description': 'chrf', 'grounding': 'mean_iou', 'grounded': 'chrf'}
BENCHMARK_DISPLAY = {
    'video_mme': 'Video-MME',
    'ocrbench_v2': 'OCRBench v2',
    'robospatial': 'RoboSpatial',
    'egoplan': 'EgoPlan-Bench',
    'openeqa': 'OpenEQA',
    'flickr30k': 'Flickr30k',
}
BENCHMARK_TASKS = {
    'video_mme': ('VQA', 'Video multiple-choice question answering'),
    'ocrbench_v2': ('VQA', 'Image OCR and text-centric question answering'),
    'robospatial': ('VQA + grounding', 'Spatial binary QA and point grounding'),
    'egoplan': ('VQA', 'Egocentric video planning multiple choice'),
    'openeqa': ('VQA', 'Embodied open-ended video question answering'),
    'flickr30k': ('DESCRIPTION', 'Image caption generation for the downloaded split'),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-root', type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument('--output-dir', type=Path)
    return parser


def _numeric_metrics(
        values: Mapping[str, Any], excluded: Iterable[str] = (), prefix: str = '') -> Dict[str, float]:
    excluded = set(excluded)
    metrics = {}
    for key, value in values.items():
        if key in excluded or isinstance(value, bool):
            continue
        name = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, Mapping):
            metrics.update(_numeric_metrics(value, excluded, name))
            continue
        if not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value):
            metrics[name] = value
    return metrics


def _custom_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for score_file in sorted(root.glob('*/*/scores.json')):
        report = json.loads(score_file.read_text(encoding='utf-8'))
        model = score_file.parent.name
        dataset = score_file.parent.parent.name
        for score in report.get('scores', []):
            task = str(score.get('task') or 'unknown')
            metrics = _numeric_metrics(score, {'num_samples', 'failed_samples'})
            primary = CUSTOM_PRIMARY.get(task)
            rows.append({
                'group': 'self_val',
                'dataset': dataset,
                'task': task,
                'model': model,
                'status': report.get('status', 'unknown'),
                'metric_scope': 'local_metric',
                'num_samples': score.get('num_samples', report.get('processed_samples', '')),
                'failed_samples': score.get('failed_samples', ''),
                'primary_metric': primary or '',
                'primary_value': metrics.get(primary, '') if primary else '',
                'metrics': metrics,
                'result_file': str(score_file),
            })
    return rows


def _official_primary(metrics: Mapping[str, float]) -> Tuple[str, Any]:
    priorities = (
        'accuracy', 'English Overall Score', 'Chinese Overall Score', 'CIDEr', 'chrf', 'vqa_accuracy',
        'token_f1', 'rouge_l', 'bleu_4', 'parse_success_rate')
    for metric in priorities:
        if metric in metrics:
            return metric, metrics[metric]
    return next(iter(metrics.items()), ('', ''))


def _official_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for dataset in BENCHMARK_DISPLAY:
        for model in OFFICIAL_MODELS:
            result_file = root / model / dataset / 'official_result.json'
            if not result_file.is_file():
                continue
            report = json.loads(result_file.read_text(encoding='utf-8'))
            raw_metrics = report.get('metrics', {})
            metrics = _numeric_metrics(raw_metrics, {'num_samples', 'num_correct'})
            primary, value = _official_primary(metrics)
            rows.append({
                'group': 'benchmark',
                'dataset': report.get('benchmark', dataset),
                'task': 'benchmark',
                'model': model,
                'status': 'complete',
                'metric_scope': raw_metrics.get('metric_scope', 'unknown'),
                'num_samples': raw_metrics.get('num_samples', ''),
                'failed_samples': '',
                'primary_metric': primary,
                'primary_value': value,
                'metrics': metrics,
                'result_file': str(result_file),
            })
    return rows


def collect(stage_root: Path) -> List[Dict[str, Any]]:
    custom_root = stage_root / 'benchmark-eval-result' / 'self-valdataset'
    official_root = stage_root / 'benchmark-eval-result' / 'stage1-autoeval' / 'official-local'
    rows = _custom_rows(custom_root) + _official_rows(official_root)
    return sorted(rows, key=lambda row: (
        row['group'], row['dataset'], row['task'], MODEL_ORDER.get(row['model'], 99), row['model']))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        'group', 'dataset', 'task', 'model', 'status', 'metric_scope', 'num_samples', 'failed_samples',
        'primary_metric', 'primary_value', 'metrics_json', 'result_file')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key, '') for key in columns}
            output['metrics_json'] = json.dumps(row['metrics'], ensure_ascii=False, sort_keys=True)
            writer.writerow(output)


def _format_value(value: Any) -> str:
    if value == '' or value is None:
        return '-'
    if isinstance(value, (int, float)):
        return f'{value:.6f}'.rstrip('0').rstrip('.')
    return str(value)


def _comparison_rows(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[Any, ...]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    scopes: Dict[Tuple[str, str, str, str], str] = {}
    for row in rows:
        key = (row['group'], row['dataset'], row['task'], row['primary_metric'])
        if not row['primary_metric']:
            continue
        grouped.setdefault(key, {})[row['model']] = row['primary_value']
        scopes[key] = row['metric_scope']
    output = []
    for key, values in sorted(grouped.items()):
        baseline = values.get('baseline', '')
        ours = values.get('ours', '')
        delta = ours - baseline if isinstance(ours, (int, float)) and isinstance(baseline, (int, float)) else ''
        output.append((*key, scopes[key], baseline, ours, delta))
    return output


def _benchmark_metric_rows(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[Any, ...]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    scopes: Dict[Tuple[str, str], str] = {}
    for row in rows:
        if row['group'] != 'benchmark':
            continue
        for metric, value in row['metrics'].items():
            key = (str(row['dataset']), str(metric))
            grouped.setdefault(key, {})[str(row['model'])] = value
            scopes[key] = str(row['metric_scope'])
    output = []
    for key, values in sorted(grouped.items()):
        baseline = values.get('baseline', '')
        ours = values.get('ours', '')
        delta = ours - baseline if isinstance(ours, (int, float)) and isinstance(baseline, (int, float)) else ''
        output.append((*key, scopes[key], baseline, ours, delta))
    return output


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    comparisons = [row for row in _comparison_rows(rows) if row[0] == 'benchmark']
    lines = [
        '# Stage 1 VLM Benchmark Comparison',
        '',
        '## Task Types',
        '',
        '| Benchmark | Type | Task |',
        '| --- | --- | --- |',
    ]
    for dataset, (task_type, task) in BENCHMARK_TASKS.items():
        lines.append(f'| {BENCHMARK_DISPLAY[dataset]} | {task_type} | {task} |')
    lines.extend([
        '',
        '## Primary Metrics',
        '',
        '| Benchmark | Metric | Scope | Baseline | Ours | Delta |',
        '| --- | --- | --- | ---: | ---: | ---: |',
    ])
    for _group, dataset, _task, metric, scope, baseline, ours, delta in comparisons:
        display = BENCHMARK_DISPLAY.get(str(dataset), str(dataset))
        values = (display, metric, scope, _format_value(baseline), _format_value(ours), _format_value(delta))
        escaped = [str(value).replace('|', '\\|').replace('\n', ' ') for value in values]
        lines.append('| ' + ' | '.join(escaped) + ' |')
    if not comparisons:
        lines.append('| - | - | - | - | - | - |')
    lines.extend([
        '',
        '## All Scalar Metrics',
        '',
        '| Benchmark | Metric | Scope | Baseline | Ours | Delta |',
        '| --- | --- | --- | ---: | ---: | ---: |',
    ])
    metric_rows = _benchmark_metric_rows(rows)
    for dataset, metric, scope, baseline, ours, delta in metric_rows:
        display = BENCHMARK_DISPLAY.get(str(dataset), str(dataset))
        values = (display, metric, scope, _format_value(baseline), _format_value(ours), _format_value(delta))
        escaped = [str(value).replace('|', '\\|').replace('\n', ' ') for value in values]
        lines.append('| ' + ' | '.join(escaped) + ' |')
    if not metric_rows:
        lines.append('| - | - | - | - | - | - |')
    lines.extend([
        '',
        '## Metric Notes',
        '',
        '- `official_vlmevalkit` and `official_exact_choice*` use benchmark-compatible deterministic scoring.',
        '- `standard_caption_generation_not_retrieval` reports VLMEvalKit BLEU, ROUGE-L, and CIDEr '
        'after multiplying raw values by 100; CIDEr is not bounded at 100, and these are not retrieval Recall@K.',
        '- `diagnostic_reference_text_no_llm_judge` is an OpenEQA lexical diagnostic, not the official LLM-Match score.',
        '- Video-MME is evaluated without subtitles unless the result scope states otherwise.',
        '',
    ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def _write_benchmark_csvs(directory: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    benchmark_rows = [row for row in rows if row['group'] == 'benchmark']
    for dataset in sorted({str(row['dataset']) for row in benchmark_rows}):
        _write_csv(directory / f'{dataset}.csv', [row for row in benchmark_rows if row['dataset'] == dataset])


def _html_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], classes: Optional[Sequence[str]] = None) -> str:
    classes = classes or [''] * len(headers)
    head = ''.join(f'<th>{html.escape(str(value))}</th>' for value in headers)
    body = []
    for row in rows:
        cells = ''.join(
            f'<td class="{html.escape(classes[index])}">{html.escape(_format_value(value))}</td>'
            for index, value in enumerate(row))
        body.append(f'<tr>{cells}</tr>')
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _write_html(path: Path, rows: Sequence[Mapping[str, Any]], csv_path: Path) -> None:
    comparisons = _comparison_rows(rows)
    complete = sum(row['status'] == 'complete' for row in rows)
    diagnostic = sum('diagnostic' in str(row['metric_scope']) for row in rows)
    comparison_table = _html_table(
        ('Group', 'Dataset', 'Task', 'Metric', 'Scope', 'Baseline', 'Ours', 'Delta'), comparisons,
        ('', '', '', '', 'scope', 'number', 'number', 'number'))
    detail_rows = [
        (row['group'], row['dataset'], row['model'], row['task'], row['status'], row['metric_scope'],
         row['num_samples'], row['failed_samples'], row['primary_metric'], row['primary_value'],
         json.dumps(row['metrics'], ensure_ascii=False, sort_keys=True))
        for row in rows
    ]
    detail_table = _html_table(
        ('Group', 'Dataset', 'Model', 'Task', 'Status', 'Scope', 'N', 'Failed', 'Primary', 'Value', 'All metrics'),
        detail_rows, ('', '', '', '', 'status', 'scope', 'number', 'number', '', 'number', 'metrics'))
    document = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage 1 VLM Evaluation</title>
<style>
:root {{ color-scheme: light; --ink:#17202a; --muted:#59636e; --line:#d8dde3; --band:#f4f6f8; --accent:#0d6b4f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#fff; font:14px/1.45 Arial, sans-serif; letter-spacing:0; }}
header {{ padding:24px 28px 18px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 6px; font-size:24px; letter-spacing:0; }}
p {{ margin:0; color:var(--muted); }}
main {{ padding:22px 28px 40px; }}
.summary {{ display:flex; gap:28px; flex-wrap:wrap; margin-bottom:24px; }}
.stat strong {{ display:block; font-size:22px; color:var(--accent); }}
h2 {{ margin:28px 0 10px; font-size:17px; letter-spacing:0; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
th {{ position:sticky; top:0; background:#e9eef1; text-align:left; font-size:12px; padding:9px 10px; border-bottom:1px solid #aeb7bf; }}
td {{ padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
tbody tr:nth-child(even) {{ background:var(--band); }}
.number {{ text-align:right; font-variant-numeric:tabular-nums; }}
.scope {{ color:var(--muted); }}
.status {{ font-weight:600; }}
.metrics {{ white-space:normal; min-width:320px; font-family:ui-monospace, SFMono-Regular, Consolas, monospace; font-size:12px; }}
a {{ color:var(--accent); }}
</style>
</head>
<body>
<header><h1>Stage 1 VLM Evaluation</h1><p>Baseline 与 ours 的自建验证集及本地 benchmark 汇总</p></header>
<main>
<div class="summary">
  <div class="stat"><strong>{len(rows)}</strong><span>result rows</span></div>
  <div class="stat"><strong>{complete}</strong><span>complete</span></div>
  <div class="stat"><strong>{diagnostic}</strong><span>diagnostic rows</span></div>
</div>
<p>Scope 为 diagnostic 的指标不是 benchmark 官方主指标。完整机器可读结果见 <a href="{html.escape(csv_path.name)}">{html.escape(csv_path.name)}</a>。</p>
<h2>Baseline vs Ours</h2><div class="table-wrap">{comparison_table}</div>
<h2>Detailed Metrics</h2><div class="table-wrap">{detail_table}</div>
</main>
</body>
</html>
'''
    path.write_text(document, encoding='utf-8')


def generate(stage_root: Path, output_dir: Path) -> Tuple[Path, Path, Path, List[Dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(stage_root)
    csv_path = output_dir / 'all-results.csv'
    html_path = output_dir / 'results.html'
    deliverable_root = stage_root / 'benchmark-eval-result'
    markdown_path = deliverable_root / 'benchmark-comparison.md'
    benchmark_csv = deliverable_root / 'benchmark-summary.csv'
    _write_csv(csv_path, rows)
    _write_csv(benchmark_csv, [row for row in rows if row['group'] == 'benchmark'])
    _write_benchmark_csvs(deliverable_root / 'benchmark-csv', rows)
    _write_markdown(markdown_path, rows)
    _write_html(html_path, rows, csv_path)
    return csv_path, html_path, markdown_path, rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    stage_root = args.stage_root.resolve()
    output_dir = args.output_dir or stage_root / 'benchmark-eval-result' / 'stage1-autoeval'
    csv_path, html_path, markdown_path, rows = generate(stage_root, output_dir.resolve())
    print(f'Wrote {len(rows)} result rows: {csv_path}')
    print(f'Wrote HTML report: {html_path}')
    print(f'Wrote Markdown comparison: {markdown_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
