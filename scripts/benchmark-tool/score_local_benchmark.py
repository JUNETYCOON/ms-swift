#!/usr/bin/env python3
"""Score locally adapted benchmark predictions with explicit metric provenance."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import tempfile
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_SCORER_PYTHON = Path('/mnt/workspace/benchmark-eval-venv/bin/python')
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from custom_eval import description_scores, final_answer_text, vqa_scores


BENCHMARKS = ('flickr30k', 'ocrbench_v2', 'realworldqa', 'egoplan', 'openeqa', 'robospatial', 'video_mme')
CHOICE_RE = re.compile(r'(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)')


def _ensure_scoring_environment(argv: Optional[Sequence[str]]) -> None:
    scorer_python = Path(os.environ.get('BENCHMARK_SCORER_PYTHON', str(DEFAULT_SCORER_PYTHON)))
    if os.environ.get('BENCHMARK_SCORER_BOOTSTRAPPED') == '1' or not scorer_python.is_file():
        return
    if Path(sys.executable).resolve() == scorer_python.resolve():
        return

    env = dict(os.environ)
    env['BENCHMARK_SCORER_BOOTSTRAPPED'] = '1'
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    os.execve(
        str(scorer_python),
        [str(scorer_python), str(Path(__file__).resolve()), *cli_args],
        env,
    )


def _install_lightweight_vlmeval_namespace() -> None:
    if 'vlmeval.dataset' in sys.modules:
        return
    try:
        package_root = Path(importlib.metadata.distribution('ms-vlmeval').locate_file('vlmeval')).resolve()
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError('Scoring requires the ms-vlmeval package.') from error
    if not package_root.is_dir():
        raise RuntimeError(f'Cannot locate the installed VLMEvalKit package at {package_root}.')

    def namespace(name: str, path: Path) -> types.ModuleType:
        module = types.ModuleType(name)
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        module.__file__ = str(path / '__init__.py')
        module.__package__ = name
        module.__path__ = [str(path)]
        module.__spec__ = spec
        return module

    root_module = namespace('vlmeval', package_root)
    dataset_module = namespace('vlmeval.dataset', package_root / 'dataset')
    root_module.dataset = dataset_module
    sys.modules['vlmeval'] = root_module
    sys.modules['vlmeval.dataset'] = dataset_module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=BENCHMARKS)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output-file', required=True, type=Path)
    return parser


def _records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_number}: prediction row must be a JSON object.')
            yield row


def _references(row: Mapping[str, Any]) -> List[str]:
    values = row.get('labels', row.get('answers'))
    if not isinstance(values, list) or not values:
        raise ValueError(f'Prediction {row.get("sample_id", row.get("id"))!r} has no references.')
    return [str(value) for value in values]


def _description_metric(rows: Sequence[Mapping[str, Any]], scope: str) -> Dict[str, Any]:
    totals = Counter()
    for row in rows:
        scores = description_scores(str(row.get('response', '')), _references(row))
        totals.update(scores)
    count = len(rows)
    return {
        'metric_scope': scope,
        'num_samples': count,
        **{name: value / count if count else 0.0 for name, value in totals.items()},
    }


def _caption_metric(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _install_lightweight_vlmeval_namespace()
    try:
        from vlmeval.dataset.image_caption import COCO_Caption_Scorer
    except ImportError as error:
        raise RuntimeError('Caption scoring requires VLMEvalKit and pycocoevalcap.') from error

    predictions = {
        str(index): [final_answer_text(str(row.get('response', '')))]
        for index, row in enumerate(rows)
    }
    references = {
        str(index): _references(row)
        for index, row in enumerate(rows)
    }
    raw_scores = COCO_Caption_Scorer(predictions, references).compute_scores()
    bleu_scores = raw_scores.pop('Bleu')
    return {
        'metric_scope': 'standard_caption_generation_not_retrieval',
        'metric_unit': 'vlmevalkit_x100',
        'num_samples': len(rows),
        **{f'BLEU_{order}': value for order, value in enumerate(bleu_scores, 1)},
        **raw_scores,
    }


def _vqa_metric(rows: Sequence[Mapping[str, Any]], scope: str) -> Dict[str, Any]:
    totals = Counter()
    for row in rows:
        totals.update(vqa_scores(str(row.get('response', '')), _references(row)))
    count = len(rows)
    return {
        'metric_scope': scope,
        'num_samples': count,
        **{name: value / count if count else 0.0 for name, value in totals.items()},
    }


def _choice(value: Any) -> str:
    text = final_answer_text(str(value)).strip().upper()
    if text in {'A', 'B', 'C', 'D'}:
        return text
    match = CHOICE_RE.search(text)
    return match.group(1) if match else ''


def _choice_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    correct = 0
    parsed = 0
    for row in rows:
        prediction = _choice(row.get('response', ''))
        answer = _choice(_references(row)[0])
        parsed += bool(prediction)
        correct += prediction == answer and bool(prediction)
    count = len(rows)
    return {
        'num_samples': count,
        'num_correct': correct,
        'accuracy': correct / count if count else 0.0,
        'parse_success_rate': parsed / count if count else 0.0,
    }


def _choice_metric(
        rows: Sequence[Mapping[str, Any]], scope: str = 'official_exact_choice',
        group_fields: Sequence[str] = ()) -> Dict[str, Any]:
    metrics = {'metric_scope': scope, **_choice_counts(rows)}
    for field in group_fields:
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field) or 'unknown')].append(row)
        metrics[f'{field}_metrics'] = {
            name: _choice_counts(group_rows)
            for name, group_rows in sorted(groups.items())
        }
    return metrics


def _robospatial_metric(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _install_lightweight_vlmeval_namespace()
    try:
        from PIL import Image
        from vlmeval.dataset.robospatialbench import RoboSpatialBench
    except ImportError as error:
        raise RuntimeError('RoboSpatial scoring requires Pillow and the bundled VLMEvalKit.') from error

    category_total = Counter()
    category_correct = Counter()
    correct_count = 0
    parsed_count = 0
    for row in rows:
        image_path = Path(row['images'][0])
        with Image.open(image_path) as image:
            width, height = image.size
        correct, _is_binary, _parsed, parsable = RoboSpatialBench.evaluate_answer(
            _references(row)[0], final_answer_text(str(row.get('response', ''))), width, height)
        category = str(row.get('category') or 'unknown')
        category_total[category] += 1
        category_correct[category] += bool(correct)
        correct_count += bool(correct)
        parsed_count += bool(parsable)
    count = len(rows)
    return {
        'metric_scope': 'official_vlmevalkit',
        'num_samples': count,
        'num_correct': correct_count,
        'accuracy': correct_count / count if count else 0.0,
        'parse_success_rate': parsed_count / count if count else 0.0,
        'category_accuracy': {
            category: category_correct[category] / total if total else 0.0
            for category, total in sorted(category_total.items())
        },
    }


def _ocr_references(row: Mapping[str, Any]) -> List[Any]:
    answers: List[Any] = _references(row)
    if row.get('ocr_type') != 'chart parsing en':
        return answers

    parsed_answers = []
    for answer in answers:
        parsed = json.loads(answer) if isinstance(answer, str) else answer
        if not isinstance(parsed, dict):
            raise ValueError('OCRBench chart parsing reference must decode to a JSON object.')
        parsed_answers.append(parsed)
    return parsed_answers


def _ocr_bbox(row: Mapping[str, Any]) -> Any:
    bbox = row.get('ocr_bbox')
    return row.get('ocr_bbox_list') if bbox is None else bbox


def _install_fast_ocrbench_edit_distance() -> None:
    """Replace OCRBench's quadratic Python loops with an exact native implementation."""
    try:
        from Levenshtein import distance
        from vlmeval.dataset.utils.Ocrbench_v2 import page_ocr_metric, vqa_metric
    except ImportError:
        return
    vqa_metric.levenshtein_distance = distance
    page_ocr_metric.nltk.edit_distance = distance


def _ocrbench_metric(rows: Sequence[Mapping[str, Any]], output_file: Path) -> Dict[str, Any]:
    _install_lightweight_vlmeval_namespace()
    try:
        import pandas as pd
        from vlmeval.dataset.image_vqa import OCRBench_v2
    except ImportError as error:
        raise RuntimeError('OCRBench_v2 scoring requires pandas and the bundled VLMEvalKit.') from error

    _install_fast_ocrbench_edit_distance()
    normalized = []
    for index, row in enumerate(rows):
        bbox = _ocr_bbox(row)
        normalized.append({
            'index': index,
            'prediction': final_answer_text(str(row.get('response', ''))),
            'answer': repr(_ocr_references(row)),
            'category': str(row['ocr_type']),
            'question': str(row['messages'][0]['content']).replace('<image>', '').strip(),
            'eval': str(row.get('ocr_eval')) if row.get('ocr_eval') not in {None, 'None'} else 'without eval',
            'bbox': repr(bbox) if bbox is not None else 'without bbox',
            'content': repr(row['ocr_content']) if row.get('ocr_content') is not None else 'without content',
        })
    detail_file = output_file.with_name(f'{output_file.stem}-details.xlsx')
    with tempfile.TemporaryDirectory(prefix='ocrbench-v2-score-') as temp_dir:
        scoring_file = Path(temp_dir) / detail_file.name
        pd.DataFrame(normalized).to_excel(scoring_file, index=False)
        scores = OCRBench_v2.evaluate(str(scoring_file))
        detail_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scoring_file, detail_file)
    return {
        'metric_scope': 'official_vlmevalkit',
        'num_samples': len(rows),
        'details_file': str(detail_file),
        **scores,
    }


def score(benchmark: str, rows: Sequence[Mapping[str, Any]], output_file: Path) -> Dict[str, Any]:
    if not rows:
        raise ValueError('Prediction file contains no records.')
    if benchmark == 'flickr30k':
        metrics = _caption_metric(rows)
    elif benchmark == 'openeqa':
        metrics = _description_metric(rows, 'diagnostic_reference_text_no_llm_judge')
        metrics['official_llm_judge_status'] = 'not_scored_no_judge_configuration'
    elif benchmark == 'egoplan':
        metrics = _choice_metric(rows)
    elif benchmark == 'video_mme':
        metrics = _choice_metric(
            rows,
            scope='official_exact_choice_without_subtitles',
            group_fields=('duration', 'domain', 'sub_category', 'task_type'),
        )
    elif benchmark == 'robospatial':
        metrics = _robospatial_metric(rows)
    elif benchmark == 'ocrbench_v2':
        metrics = _ocrbench_metric(rows, output_file)
    elif benchmark == 'realworldqa':
        metrics = _vqa_metric(rows, 'reference_answer_accuracy')
    else:
        raise ValueError(f'Unsupported benchmark: {benchmark}.')
    return {'benchmark': benchmark, 'metrics': _json_safe(metrics)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, 'item'):
        return _json_safe(value.item())
    return str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_scoring_environment(argv)
    args = _parser().parse_args(argv)
    rows = list(_records(args.predictions.resolve()))
    report = score(args.benchmark, rows, args.output_file.resolve())
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
