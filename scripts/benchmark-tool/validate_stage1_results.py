#!/usr/bin/env python3
"""Validate that every stage-1 benchmark result is complete and correctly scoped."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_RESULT_ROOT = Path(
    '/mnt/luojunkun/stage1/benchmark-eval-result/stage1-autoeval/official-local')
EXPECTED: Mapping[str, Tuple[int, str]] = {
    'video_mme': (2700, 'official_exact_choice_without_subtitles'),
    'ocrbench_v2': (10000, 'official_vlmevalkit'),
    'robospatial': (350, 'official_vlmevalkit'),
    'egoplan': (3343, 'official_exact_choice'),
    'openeqa': (1636, 'diagnostic_reference_text_no_llm_judge'),
    'flickr30k': (31783, 'standard_caption_generation_not_retrieval'),
}
MODELS = ('baseline', 'ours')
EXPECTED_WEIGHTS: Mapping[str, Path] = {
    'baseline': Path('/mnt/luojunkun/stage1/model'),
    'ours': Path('/mnt/luojunkun/stage1/sft-model/qwen3-vl-instruct4b/v30-20260728-003837/checkpoint-400'),
}
EXPECTED_TASKS: Mapping[str, str] = {
    'video_mme': 'vqa',
    'ocrbench_v2': 'vqa',
    'robospatial': 'vqa',
    'egoplan': 'vqa',
    'openeqa': 'description',
    'flickr30k': 'description',
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, default=DEFAULT_RESULT_ROOT)
    return parser


def _load_json(path: Path, errors: List[str]) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        errors.append(f'missing file: {path}')
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f'invalid JSON: {path}: {error}')
        return None
    if not isinstance(value, dict):
        errors.append(f'expected JSON object: {path}')
        return None
    return value


def _jsonl_count(path: Path, errors: List[str]) -> Optional[int]:
    if not path.is_file():
        errors.append(f'missing file: {path}')
        return None
    try:
        with path.open('r', encoding='utf-8') as stream:
            return sum(1 for line in stream if line.strip())
    except OSError as error:
        errors.append(f'cannot read JSONL: {path}: {error}')
        return None


def validate_model_result(
        result_root: Path, benchmark: str, model: str, expected_count: int, expected_scope: str,
        expected_weights: Optional[Path] = None, expected_task: Optional[str] = None) -> List[str]:
    output_dir = result_root / model / benchmark
    errors: List[str] = []
    scores = _load_json(output_dir / 'scores.json', errors)
    official = _load_json(output_dir / 'official_result.json', errors)
    prediction_count = _jsonl_count(output_dir / 'predictions.jsonl', errors)

    if scores is not None:
        if scores.get('status') != 'complete':
            errors.append(f'{benchmark}/{model}: scores status is {scores.get("status")!r}')
        if expected_weights is not None:
            actual_weights = Path(str(scores.get('model_weights', '')))
            if actual_weights != expected_weights:
                errors.append(
                    f'{benchmark}/{model}: model_weights={str(actual_weights)!r}, '
                    f'expected {str(expected_weights)!r}')
        if scores.get('model') != model:
            errors.append(f'{benchmark}/{model}: model label={scores.get("model")!r}')
        if scores.get('model_type') != 'qwen3_vl':
            errors.append(f'{benchmark}/{model}: model_type={scores.get("model_type")!r}')
        if int(scores.get('processed_samples', -1)) != expected_count:
            errors.append(
                f'{benchmark}/{model}: processed_samples={scores.get("processed_samples")!r}, '
                f'expected {expected_count}')
        task_scores = scores.get('scores')
        if not isinstance(task_scores, list) or not task_scores:
            errors.append(f'{benchmark}/{model}: scores list is missing or empty')
        else:
            failed = sum(int(item.get('failed_samples', 0)) for item in task_scores)
            scored = sum(int(item.get('num_samples', 0)) for item in task_scores)
            tasks = {str(item.get('task')) for item in task_scores}
            if failed:
                errors.append(f'{benchmark}/{model}: failed_samples={failed}')
            if scored != expected_count:
                errors.append(f'{benchmark}/{model}: scored samples={scored}, expected {expected_count}')
            if expected_task is not None and tasks != {expected_task}:
                errors.append(f'{benchmark}/{model}: tasks={sorted(tasks)!r}, expected {[expected_task]!r}')

    if prediction_count is not None and prediction_count != expected_count:
        errors.append(f'{benchmark}/{model}: predictions={prediction_count}, expected {expected_count}')

    if official is not None:
        if official.get('benchmark') != benchmark:
            errors.append(f'{benchmark}/{model}: official benchmark={official.get("benchmark")!r}')
        metrics = official.get('metrics')
        if not isinstance(metrics, dict):
            errors.append(f'{benchmark}/{model}: official metrics are missing')
        else:
            if int(metrics.get('num_samples', -1)) != expected_count:
                errors.append(
                    f'{benchmark}/{model}: official num_samples={metrics.get("num_samples")!r}, '
                    f'expected {expected_count}')
            if metrics.get('metric_scope') != expected_scope:
                errors.append(
                    f'{benchmark}/{model}: metric_scope={metrics.get("metric_scope")!r}, '
                    f'expected {expected_scope!r}')
    return errors


def validate(result_root: Path) -> List[str]:
    errors = []
    for benchmark, (expected_count, expected_scope) in EXPECTED.items():
        for model in MODELS:
            errors.extend(validate_model_result(
                result_root, benchmark, model, expected_count, expected_scope,
                expected_weights=EXPECTED_WEIGHTS[model], expected_task=EXPECTED_TASKS[benchmark]))
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result_root = args.result_root.resolve()
    errors = validate(result_root)
    if errors:
        print(f'[validate-stage1] FAILED with {len(errors)} issue(s):')
        for error in errors:
            print(f'  - {error}')
        return 1
    print(f'[validate-stage1] complete: {len(EXPECTED) * len(MODELS)} result sets validated')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
