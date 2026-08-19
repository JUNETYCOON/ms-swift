#!/usr/bin/env python3
"""Recompute custom-eval metrics from saved predictions without loading a model."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from custom_eval import (FREEFORM_SIMILARITY_THRESHOLD, PREDICTION_COLUMNS, ScoreAccumulator, _raw_user_text,
                         _reference_answers, _sample_id, _write_scores_csv, description_scores,
                         grounded_description_scores, grounding_scores, vqa_scores)


OUTPUT_NAMES = ('predictions.jsonl', 'results.csv', 'scores.csv', 'scores.json')


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-file', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--task', required=True, choices=('vqa', 'description', 'grounding', 'grounded'))
    parser.add_argument('--dataset-dir', type=Path, help='Base directory for relative media paths.')
    parser.add_argument('--model-weights', default='')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--prediction-space', choices=('norm1000', 'norm1', 'real'), default='norm1000')
    parser.add_argument('--progress-every', type=int, default=1000)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def _records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{line_number}: prediction row must be an object.')
            yield record


def _references(record: Mapping[str, Any]) -> List[str]:
    labels = record.get('labels')
    if isinstance(labels, list) and labels:
        return [str(value) for value in labels]
    references = _reference_answers(record)
    if not references:
        raise ValueError(f'Prediction {record.get("sample_id")!r} has no reference answer.')
    return references


def _score_row(record: Mapping[str, Any], line_number: int, task: str, model_name: str, dataset_name: str,
               dataset_dir: Path, prediction_space: str, iou_threshold: float,
               accumulator: ScoreAccumulator) -> Dict[str, Any]:
    prediction = str(record.get('response', record.get('prediction', '')))
    references = _references(record)
    error = str(record.get('eval_error') or record.get('error') or '')
    row: Dict[str, Any] = {
        'model': model_name,
        'dataset': dataset_name,
        'sample_id': str(record.get('sample_id') or _sample_id(record, line_number)),
        'line_number': line_number,
        'task': task,
        'question': str(record.get('question') or _raw_user_text(record)),
        'reference': json.dumps(references, ensure_ascii=False),
        'prediction': prediction,
        'error': error,
    }
    if task == 'vqa':
        scores = vqa_scores(
            prediction,
            references,
            record.get('question') or _raw_user_text(record),
        )
        row.update(scores)
        accumulator.add_vqa(scores, failed=bool(error))
    elif task == 'description':
        scores = description_scores(prediction, references)
        row.update(scores)
        accumulator.add_description(scores, failed=bool(error))
    elif task == 'grounding':
        scores = grounding_scores(record, prediction, prediction_space, iou_threshold, dataset_dir)
        row.update({
            'pred_boxes': json.dumps(scores['pred_boxes']),
            'gt_boxes': json.dumps(scores['gt_boxes']),
            'mean_iou': scores['mean_iou'],
            'iou_accuracy': scores['iou_accuracy'],
            'parse_success': int(scores['parse_success']),
        })
        accumulator.add_grounding(scores, failed=bool(error))
    else:
        description = grounded_description_scores(prediction, references)
        grounding = grounding_scores(record, prediction, prediction_space, iou_threshold, dataset_dir)
        row.update(description)
        row.update({
            'pred_boxes': json.dumps(grounding['pred_boxes']),
            'gt_boxes': json.dumps(grounding['gt_boxes']),
            'mean_iou': grounding['mean_iou'],
            'iou_accuracy': grounding['iou_accuracy'],
            'parse_success': int(grounding['parse_success']),
        })
        accumulator.add_grounded(description, grounding, failed=bool(error))
    return row


def rescore(predictions_file: Path, output_dir: Path, model_name: str, dataset_name: str, task: str,
            dataset_dir: Optional[Path] = None, model_weights: str = '', iou_threshold: float = 0.5,
            prediction_space: str = 'norm1000', progress_every: int = 1000,
            overwrite: bool = False) -> List[Dict[str, Any]]:
    predictions_file = predictions_file.resolve()
    output_dir = output_dir.resolve()
    dataset_dir = (dataset_dir or predictions_file.parent).resolve()
    if progress_every <= 0:
        raise ValueError('--progress-every must be greater than zero.')
    targets = {name: output_dir / name for name in OUTPUT_NAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Output files already exist: {", ".join(str(path) for path in existing)}')

    accumulator = ScoreAccumulator(iou_threshold)
    count = 0
    with tempfile.TemporaryDirectory(prefix='benchmark-rescore-', dir=tempfile.gettempdir()) as temp_dir:
        temporary = {name: Path(temp_dir) / name for name in OUTPUT_NAMES}
        with temporary['predictions.jsonl'].open('w', encoding='utf-8', newline='\n') as prediction_stream, \
                temporary['results.csv'].open('w', encoding='utf-8', newline='') as result_stream:
            writer = csv.DictWriter(result_stream, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            for line_number, record in enumerate(_records(predictions_file)):
                try:
                    row = _score_row(record, line_number, task, model_name, dataset_name, dataset_dir,
                                     prediction_space, iou_threshold, accumulator)
                except Exception as error:
                    raise ValueError(
                        f'Cannot rescore prediction line {line_number + 1} '
                        f'(sample_id={record.get("sample_id")!r}): {error}') from error
                prediction_record = dict(record)
                prediction_record['task'] = task
                prediction_stream.write(json.dumps(prediction_record, ensure_ascii=False) + '\n')
                writer.writerow(row)
                count += 1
                if count % progress_every == 0:
                    print(f'[rescore-custom] processed={count}', flush=True)
        if count == 0:
            raise ValueError(f'No prediction records found in {predictions_file}.')

        summary_rows = accumulator.summary_rows(model_name, dataset_name)
        _write_scores_csv(temporary['scores.csv'], summary_rows)
        report = {
            'status': 'complete',
            'processed_samples': count,
            'model': model_name,
            'model_weights': model_weights,
            'val_dataset': str(predictions_file),
            'iou_threshold': iou_threshold,
            'prediction_space': prediction_space,
            'freeform_similarity_threshold': FREEFORM_SIMILARITY_THRESHOLD,
            'rescore_source': str(predictions_file),
            'outputs': {name.rsplit('.', 1)[0]: str(path) for name, path in targets.items()},
            'scores': summary_rows,
        }
        temporary['scores.json'].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

        output_dir.mkdir(parents=True, exist_ok=True)
        for name, source in temporary.items():
            staged = targets[name].with_name(f'{targets[name].name}.rescore.tmp')
            shutil.copyfile(source, staged)
            staged.replace(targets[name])
    print(f'[rescore-custom] wrote {count} samples to {output_dir}', flush=True)
    return summary_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    rows = rescore(
        args.predictions_file, args.output_dir, args.model_name, args.dataset_name, args.task,
        dataset_dir=args.dataset_dir, model_weights=args.model_weights, iou_threshold=args.iou_threshold,
        prediction_space=args.prediction_space, progress_every=args.progress_every, overwrite=args.overwrite)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
