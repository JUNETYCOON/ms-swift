#!/usr/bin/env python3
"""Clean and convert COCO-MODELSCOPE Arrow data to ms-swift JSONL.

The source contains embedded image bytes. Train rows have COCO category IDs,
while the official test rows are unlabeled. The converter therefore writes
train/validation SFT files and a separate user-only inference file for test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_INPUT_DIR = Path('/mnt/luojunkun/stage1/dataset/COCO/COCO-MODELSCOPE')
DEFAULT_OUTPUT_DIR = Path('/mnt/luojunkun/stage1/dataset_ms-swift/coco')
CONVERTER_ID = 'coco-multilabel-ms-swift-v1'
MARKER_NAME = '.coco_multilabel_converter.json'
OUTPUT_NAMES = {
    'train': 'coco_train_sft_msswift.jsonl',
    'val': 'coco_val_sft_msswift.jsonl',
    'test': 'coco_test_inference_msswift.jsonl',
}
FORMAT_SUFFIXES = {
    'JPEG': '.jpg',
    'PNG': '.png',
    'WEBP': '.webp',
    'BMP': '.bmp',
    'GIF': '.gif',
    'TIFF': '.tiff',
}
DEFAULT_PROMPT = (
    'Perform multi-label object classification for this image. Select every applicable category from: {labels}. '
    'Return only the selected category names as a comma-separated list.'
)


class RowRejected(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ShardTask:
    source_path: str
    source_relative_path: str
    source_split: str
    shard_ordinal: int
    row_limit: int | None
    label_count: int
    output_dir: str
    fragment_dir: str


@dataclass(frozen=True)
class ShardResult:
    source_path: str
    source_relative_path: str
    source_split: str
    shard_ordinal: int
    rows_read: int
    indexed_rows: int
    rejected_rows: int
    normalized_duplicate_labels: int
    index_fragment: str
    rejected_fragment: str
    schema: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        '--max-rows-per-split',
        type=int,
        default=None,
        help='Process only the first N source rows of each split. Intended for smoke tests.',
    )
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--relative-paths', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--keep-temp', action='store_true')
    return parser.parse_args(argv)


def import_pyarrow() -> Any:
    try:
        import pyarrow as pa
    except ImportError as error:
        raise SystemExit('pyarrow is required: python -m pip install pyarrow') from error
    return pa


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.val_ratio < 1:
        raise SystemExit('--val-ratio must be between 0 and 1')
    if args.num_workers <= 0:
        raise SystemExit('--num-workers must be greater than zero')
    if args.max_rows_per_split is not None and args.max_rows_per_split <= 0:
        raise SystemExit('--max-rows-per-split must be greater than zero')
    if '{labels}' not in args.prompt:
        raise SystemExit('--prompt must contain {labels}')
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise SystemExit('--output-dir must not be inside --input-dir')


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f'Output directory is not empty; use --overwrite: {output_dir}')
        marker_path = output_dir / MARKER_NAME
        if not marker_path.is_file():
            raise RuntimeError(f'Refusing to overwrite an unowned directory without {MARKER_NAME}: {output_dir}')
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        if marker.get('converter') != CONVERTER_ID:
            raise RuntimeError(f'Refusing to overwrite output owned by another converter: {output_dir}')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / MARKER_NAME,
        {'converter': CONVERTER_ID, 'status': 'in_progress', 'started_at': utc_now()},
    )


def source_snapshot(input_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted((item for item in input_dir.rglob('*') if item.is_file()), key=lambda item: item.as_posix()):
        stat = path.stat()
        files.append({
            'path': path.relative_to(input_dir).as_posix(),
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        })
    payload = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {
        'root': str(input_dir),
        'file_count': len(files),
        'total_bytes': sum(item['size'] for item in files),
        'digest': hashlib.sha256(payload).hexdigest(),
        'files': files,
    }


def load_labels(input_dir: Path) -> list[str]:
    path = input_dir / 'labels.txt'
    if not path.is_file():
        raise FileNotFoundError(f'labels.txt not found: {path}')
    labels = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(labels) != 80:
        raise ValueError(f'Expected 80 COCO labels, found {len(labels)}')
    if len(set(labels)) != len(labels):
        raise ValueError('labels.txt contains duplicate category names')
    return labels


def discover_shards(input_dir: Path) -> dict[str, list[Path]]:
    shards = {}
    for split in ('train', 'test'):
        split_dir = input_dir / split
        paths = sorted(split_dir.glob('*.arrow'))
        if not paths:
            raise FileNotFoundError(f'No Arrow shards found: {split_dir}')
        shards[split] = paths
    return shards


def arrow_row_count(path: Path) -> int:
    pa = import_pyarrow()
    with pa.memory_map(str(path), 'r') as source:
        return sum(batch.num_rows for batch in pa.ipc.open_stream(source))


def build_tasks(
    input_dir: Path,
    output_dir: Path,
    shards: dict[str, list[Path]],
    label_count: int,
    max_rows_per_split: int | None,
) -> tuple[list[ShardTask], dict[str, int]]:
    tasks = []
    declared_rows = {'train': 0, 'test': 0}
    fragment_dir = output_dir / 'audit' / 'fragments'
    fragment_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'test'):
        remaining = max_rows_per_split
        for ordinal, path in enumerate(shards[split]):
            row_count = arrow_row_count(path)
            declared_rows[split] += row_count
            if remaining == 0:
                continue
            row_limit = None if remaining is None else min(row_count, remaining)
            tasks.append(
                ShardTask(
                    source_path=str(path),
                    source_relative_path=path.relative_to(input_dir).as_posix(),
                    source_split=split,
                    shard_ordinal=ordinal,
                    row_limit=row_limit,
                    label_count=label_count,
                    output_dir=str(output_dir),
                    fragment_dir=str(fragment_dir),
                ))
            if remaining is not None:
                remaining -= row_limit
    return tasks, declared_rows


def normalize_labels(value: Any, label_count: int) -> tuple[list[int], bool]:
    if not isinstance(value, list):
        raise RowRejected('invalid_labels_type', f'Expected list[int], found {type(value).__name__}')
    if not value:
        raise RowRejected('empty_labels', 'Training row has no positive category')
    labels = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise RowRejected('invalid_label_id', f'Expected integer label ID, found {item!r}')
        if not 0 <= item < label_count:
            raise RowRejected('out_of_range_label_id', f'Label ID {item} is outside [0, {label_count})')
        labels.append(item)
    normalized = sorted(set(labels))
    return normalized, len(normalized) != len(labels)


def decode_image(image_bytes: bytes) -> tuple[str, int, int, str]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise RuntimeError('Pillow is required: python -m pip install Pillow') from error
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            image_format = (image.format or '').upper()
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise RowRejected('image_decode_error', str(error)) from error
    if width <= 0 or height <= 0:
        raise RowRejected('invalid_image_dimensions', f'Invalid image size: {width}x{height}')
    suffix = FORMAT_SUFFIXES.get(image_format)
    if suffix is None:
        raise RowRejected('unsupported_image_format', f'Unsupported Pillow image format: {image_format!r}')
    return suffix, width, height, mode


def extract_image_bytes(row: dict[str, Any]) -> bytes:
    images = row.get('images')
    if not isinstance(images, list) or len(images) != 1:
        size = len(images) if isinstance(images, list) else None
        raise RowRejected('invalid_images_field', f'Expected exactly one image, found {size!r}')
    image = images[0]
    if not isinstance(image, dict):
        raise RowRejected('invalid_image_entry', f'Expected image struct, found {type(image).__name__}')
    value = image.get('bytes')
    if not isinstance(value, (bytes, bytearray, memoryview)) or not value:
        raise RowRejected('missing_image_bytes', 'Embedded image bytes are empty or missing')
    return bytes(value)


def store_image(output_dir: Path, digest: str, suffix: str, image_bytes: bytes) -> str:
    relative_path = Path('images') / digest[:2] / f'{digest}{suffix}'
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != len(image_bytes):
            raise RuntimeError(f'Hash-addressed image has conflicting size: {destination}')
        return relative_path.as_posix()
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    try:
        temporary.write_bytes(image_bytes)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return relative_path.as_posix()


def _write_fragment_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')


def process_shard(task: ShardTask) -> ShardResult:
    pa = import_pyarrow()
    source_path = Path(task.source_path)
    output_dir = Path(task.output_dir)
    fragment_dir = Path(task.fragment_dir)
    stem = f'{task.source_split}-{task.shard_ordinal:05d}'
    index_path = fragment_dir / f'{stem}.index.jsonl'
    rejected_path = fragment_dir / f'{stem}.rejected.jsonl'
    index_tmp = index_path.with_suffix(index_path.suffix + f'.{os.getpid()}.tmp')
    rejected_tmp = rejected_path.with_suffix(rejected_path.suffix + f'.{os.getpid()}.tmp')
    rows_read = indexed_rows = rejected_rows = normalized_duplicate_labels = 0
    with pa.memory_map(str(source_path), 'r') as source:
        reader = pa.ipc.open_stream(source)
        schema = reader.schema
        required = {'images'} | ({'labels'} if task.source_split == 'train' else set())
        missing = required - set(schema.names)
        if missing:
            raise RuntimeError(f'{source_path}: missing required Arrow fields: {sorted(missing)}')
        with index_tmp.open('w', encoding='utf-8', newline='') as index_handle, rejected_tmp.open(
                'w', encoding='utf-8', newline='') as rejected_handle:
            stop = False
            shard_row = 0
            for batch in reader:
                images_column = batch.column(batch.schema.get_field_index('images'))
                labels_index = batch.schema.get_field_index('labels')
                labels_column = batch.column(labels_index) if labels_index >= 0 else None
                for batch_row in range(batch.num_rows):
                    if task.row_limit is not None and rows_read >= task.row_limit:
                        stop = True
                        break
                    row = {'images': images_column[batch_row].as_py()}
                    if labels_column is not None:
                        row['labels'] = labels_column[batch_row].as_py()
                    source_id = f'coco:{task.source_split}:{task.source_relative_path}#row={shard_row}'
                    base = {
                        'source_id': source_id,
                        'source_split': task.source_split,
                        'source_path': task.source_relative_path,
                        'shard_ordinal': task.shard_ordinal,
                        'source_row': shard_row,
                    }
                    rows_read += 1
                    shard_row += 1
                    try:
                        if not isinstance(row, dict):
                            raise RowRejected('invalid_row_type', f'Expected object, found {type(row).__name__}')
                        image_bytes = extract_image_bytes(row)
                        if task.source_split == 'train':
                            labels, normalized = normalize_labels(row.get('labels'), task.label_count)
                            normalized_duplicate_labels += int(normalized)
                        else:
                            if row.get('labels') is not None:
                                raise RowRejected('unexpected_test_labels', 'Official test is expected to be unlabeled')
                            labels = None
                        suffix, width, height, mode = decode_image(image_bytes)
                        digest = hashlib.sha256(image_bytes).hexdigest()
                        image_path = store_image(output_dir, digest, suffix, image_bytes)
                        _write_fragment_line(
                            index_handle,
                            {
                                **base,
                                'media_sha256': digest,
                                'image_path': image_path,
                                'image_bytes': len(image_bytes),
                                'image_width': width,
                                'image_height': height,
                                'image_mode': mode,
                                'label_ids': labels,
                            },
                        )
                        indexed_rows += 1
                    except RowRejected as error:
                        _write_fragment_line(rejected_handle, {**base, 'reason': error.reason, 'detail': error.detail})
                        rejected_rows += 1
                if stop:
                    break
    os.replace(index_tmp, index_path)
    os.replace(rejected_tmp, rejected_path)
    return ShardResult(
        source_path=task.source_path,
        source_relative_path=task.source_relative_path,
        source_split=task.source_split,
        shard_ordinal=task.shard_ordinal,
        rows_read=rows_read,
        indexed_rows=indexed_rows,
        rejected_rows=rejected_rows,
        normalized_duplicate_labels=normalized_duplicate_labels,
        index_fragment=str(index_path),
        rejected_fragment=str(rejected_path),
        schema=str(schema),
    )


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'Invalid JSON at {path}:{line_number}: {error}') from error


def source_order(record: dict[str, Any]) -> tuple[int, int, int]:
    return (0 if record['source_split'] == 'train' else 1, record['shard_ordinal'], record['source_row'])


def stable_split(media_sha256: str, val_ratio: float, seed: int) -> str:
    key = f'{CONVERTER_ID}:{seed}:{media_sha256}'.encode('ascii')
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
    return 'val' if value < val_ratio else 'train'


def rejection_from_record(record: dict[str, Any], reason: str, detail: str) -> dict[str, Any]:
    return {
        'source_id': record['source_id'],
        'source_split': record['source_split'],
        'source_path': record['source_path'],
        'shard_ordinal': record['shard_ordinal'],
        'source_row': record['source_row'],
        'media_sha256': record.get('media_sha256'),
        'reason': reason,
        'detail': detail,
    }


def clean_and_assign(
    indexed: dict[str, list[dict[str, Any]]],
    worker_rejections: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    retained = {'train': [], 'val': [], 'test': []}
    rejections = list(worker_rejections)
    media_rows = []
    groups = {
        split: defaultdict(list)
        for split in ('train', 'test')
    }
    for split in ('train', 'test'):
        for record in sorted(indexed[split], key=source_order):
            groups[split][record['media_sha256']].append(record)

    test_hashes = set(groups['test'])
    for digest, records in groups['test'].items():
        canonical = records[0]
        retained['test'].append(canonical)
        for index, record in enumerate(records):
            if index == 0:
                assigned_split, status, reason = 'test', 'retained', ''
            else:
                assigned_split, status, reason = 'quarantine', 'rejected', 'duplicate_test_media'
                rejections.append(
                    rejection_from_record(record, reason, f'Canonical source: {canonical["source_id"]}'))
            media_rows.append(make_media_row(record, assigned_split, status, reason))

    for digest, records in groups['train'].items():
        if digest in test_hashes:
            test_source = groups['test'][digest][0]['source_id']
            for record in records:
                reason = 'train_test_media_overlap'
                rejections.append(rejection_from_record(record, reason, f'Protected test source: {test_source}'))
                media_rows.append(make_media_row(record, 'quarantine', 'rejected', reason))
            continue
        label_sets = {tuple(record['label_ids']) for record in records}
        if len(label_sets) != 1:
            detail = f'Conflicting label sets: {sorted(label_sets)}'
            for record in records:
                reason = 'conflicting_duplicate_labels'
                rejections.append(rejection_from_record(record, reason, detail))
                media_rows.append(make_media_row(record, 'quarantine', 'rejected', reason))
            continue
        canonical = records[0]
        assigned = stable_split(digest, val_ratio, seed)
        retained[assigned].append(canonical)
        for index, record in enumerate(records):
            if index == 0:
                assigned_split, status, reason = assigned, 'retained', ''
            else:
                assigned_split, status, reason = 'quarantine', 'rejected', 'duplicate_train_media'
                rejections.append(
                    rejection_from_record(record, reason, f'Canonical source: {canonical["source_id"]}'))
            media_rows.append(make_media_row(record, assigned_split, status, reason))

    for record in worker_rejections:
        media_rows.append(make_media_row(record, 'quarantine', 'rejected', record['reason']))
    for split in retained:
        retained[split].sort(key=source_order)
    rejections.sort(key=source_order)
    media_rows.sort(key=source_order)
    return retained, rejections, media_rows


def make_media_row(record: dict[str, Any], assigned_split: str, status: str, reason: str) -> dict[str, Any]:
    digest = record.get('media_sha256', '')
    labels = record.get('label_ids')
    return {
        'source_split': record['source_split'],
        'source_path': record['source_path'],
        'source_row': record['source_row'],
        'source_id': record['source_id'],
        'media_sha256': digest,
        'lineage_root': f'coco:sha256:{digest}' if digest else '',
        'label_ids': json.dumps(labels, separators=(',', ':')) if labels is not None else '',
        'assigned_split': assigned_split,
        'status': status,
        'reason': reason,
        'image_path': record.get('image_path', ''),
        'shard_ordinal': record['shard_ordinal'],
    }


def image_reference(output_dir: Path, image_path: str, relative_paths: bool) -> str:
    return image_path if relative_paths else str((output_dir / image_path).resolve())


def make_sft_record(
    record: dict[str, Any],
    labels: list[str],
    prompt: str,
    output_dir: Path,
    relative_paths: bool,
) -> dict[str, Any]:
    answer = ', '.join(labels[index] for index in record['label_ids'])
    return {
        'messages': [
            {'role': 'user', 'content': f'<image>\n{prompt}'},
            {'role': 'assistant', 'content': answer},
        ],
        'images': [image_reference(output_dir, record['image_path'], relative_paths)],
    }


def make_test_record(
    record: dict[str, Any], prompt: str, output_dir: Path, relative_paths: bool
) -> dict[str, Any]:
    return {
        'messages': [{'role': 'user', 'content': f'<image>\n{prompt}'}],
        'images': [image_reference(output_dir, record['image_path'], relative_paths)],
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    count = 0
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            count += 1
    os.replace(temporary, path)
    return count


def write_media_groups(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        'source_split', 'source_path', 'source_row', 'source_id', 'media_sha256', 'lineage_root', 'label_ids',
        'assigned_split', 'status', 'reason', 'image_path'
    ]
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_delivery(
    output_dir: Path,
    output_paths: dict[str, Path],
    retained: dict[str, list[dict[str, Any]]],
    labels: list[str],
) -> dict[str, Any]:
    allowed_labels = set(labels)
    observed_paths = {}
    counts = {}
    for split, path in output_paths.items():
        count = 0
        for line_number, row in enumerate(read_jsonl(path), start=1):
            messages = row.get('messages')
            images = row.get('images')
            if not isinstance(messages, list) or not isinstance(images, list) or len(images) != 1:
                raise AssertionError(f'{path}:{line_number}: invalid messages/images structure')
            expected_roles = ['user'] if split == 'test' else ['user', 'assistant']
            if [message.get('role') for message in messages] != expected_roles:
                raise AssertionError(f'{path}:{line_number}: invalid message roles')
            if messages[0].get('content', '').count('<image>') != 1:
                raise AssertionError(f'{path}:{line_number}: image placeholder mismatch')
            if split != 'test':
                answer_labels = [value.strip() for value in messages[1].get('content', '').split(',')]
                if not answer_labels or not set(answer_labels) <= allowed_labels:
                    raise AssertionError(f'{path}:{line_number}: invalid assistant labels')
            image_path = Path(images[0])
            if not image_path.is_absolute():
                image_path = output_dir / image_path
            if not image_path.is_file():
                raise AssertionError(f'{path}:{line_number}: missing image: {image_path}')
            if images[0] in observed_paths:
                raise AssertionError(
                    f'Media occurs in multiple outputs: {images[0]} ({observed_paths[images[0]]}, {split})')
            observed_paths[images[0]] = split
            count += 1
        if count != len(retained[split]):
            raise AssertionError(f'{path}: expected {len(retained[split])} rows, found {count}')
        counts[split] = count
    media_sets = {split: {record['media_sha256'] for record in rows} for split, rows in retained.items()}
    has_overlap = (
        media_sets['train'] & media_sets['val'] or media_sets['train'] & media_sets['test']
        or media_sets['val'] & media_sets['test'])
    if has_overlap:
        raise AssertionError('Output splits share a media SHA-256')
    return {
        'status': 'passed',
        'output_rows': counts,
        'unique_output_media': len(observed_paths),
        'message_roles_valid': True,
        'media_placeholders_valid': True,
        'output_media_exists': True,
        'image_decode_validation': 'passed during source indexing',
        'split_media_sha256_disjoint': True,
    }


def remove_unreferenced_images(
    output_dir: Path,
    indexed: dict[str, list[dict[str, Any]]],
    retained: dict[str, list[dict[str, Any]]],
) -> int:
    all_images = {record['image_path'] for rows in indexed.values() for record in rows}
    retained_images = {record['image_path'] for rows in retained.values() for record in rows}
    removed = 0
    for relative_path in sorted(all_images - retained_images):
        path = output_dir / relative_path
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def convert(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)
    before = source_snapshot(input_dir)
    labels = load_labels(input_dir)
    shards = discover_shards(input_dir)
    tasks, source_population_rows = build_tasks(
        input_dir, output_dir, shards, len(labels), args.max_rows_per_split)
    print(f'[scan] {len(tasks)} Arrow shard tasks with {args.num_workers} worker(s)')
    if args.num_workers == 1:
        results = [process_shard(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(executor.map(process_shard, tasks))
    for result in results:
        print(
            f'[indexed] {result.source_relative_path}: read={result.rows_read}, '
            f'valid={result.indexed_rows}, rejected={result.rejected_rows}')

    indexed = {'train': [], 'test': []}
    worker_rejections = []
    for result in results:
        indexed[result.source_split].extend(read_jsonl(Path(result.index_fragment)))
        worker_rejections.extend(read_jsonl(Path(result.rejected_fragment)))
    retained, rejections, media_rows = clean_and_assign(
        indexed, worker_rejections, args.val_ratio, args.seed)
    removed_orphan_images = remove_unreferenced_images(output_dir, indexed, retained)

    labels_text = ', '.join(labels)
    prompt = args.prompt.format(labels=labels_text)
    output_paths = {split: output_dir / name for split, name in OUTPUT_NAMES.items()}
    write_jsonl(
        output_paths['train'],
        (make_sft_record(row, labels, prompt, output_dir, args.relative_paths) for row in retained['train']),
    )
    write_jsonl(
        output_paths['val'],
        (make_sft_record(row, labels, prompt, output_dir, args.relative_paths) for row in retained['val']),
    )
    write_jsonl(
        output_paths['test'],
        (make_test_record(row, prompt, output_dir, args.relative_paths) for row in retained['test']),
    )
    write_jsonl(output_dir / 'rejected.jsonl', rejections)
    write_media_groups(output_dir / 'media_groups.tsv', media_rows)

    processed_by_split = {
        split: sum(result.rows_read for result in results if result.source_split == split)
        for split in ('train', 'test')
    }
    rejected_by_split = Counter(row['source_split'] for row in rejections)
    retained_by_source = {
        'train': len(retained['train']) + len(retained['val']),
        'test': len(retained['test']),
    }
    for split in ('train', 'test'):
        if processed_by_split[split] != retained_by_source[split] + rejected_by_split[split]:
            raise AssertionError(
                f'{split} accounting mismatch: read={processed_by_split[split]}, '
                f'retained={retained_by_source[split]}, rejected={rejected_by_split[split]}')

    validation = validate_delivery(output_dir, output_paths, retained, labels)
    after = source_snapshot(input_dir)
    source_unchanged = before == after
    if not source_unchanged:
        raise AssertionError('Source snapshot changed during conversion')
    readonly_evidence = {'status': 'passed', 'unchanged': True, 'before': before, 'after': after}
    atomic_write_json(output_dir / 'source-readonly-evidence.json', readonly_evidence)
    schemas = {
        split: sorted({result.schema for result in results if result.source_split == split})
        for split in ('train', 'test')
    }
    schema_report = {
        'version': 1,
        'dataset': 'COCO-MODELSCOPE',
        'source_root': str(input_dir),
        'source_format': 'Hugging Face Arrow IPC stream',
        'task': 'multi-label object classification',
        'input_contract': 'exactly one embedded, decodable image',
        'output_contract': {
            'train': 'non-empty list[int] mapped through labels.txt',
            'test': 'unlabeled; emitted only for inference',
        },
        'labels': labels,
        'schemas': schemas,
        'source_shards': {split: len(paths) for split, paths in shards.items()},
        'source_population_rows': source_population_rows,
        'processed_rows': processed_by_split,
        'limited_run': args.max_rows_per_split is not None,
    }
    atomic_write_json(output_dir / 'schema_report.json', schema_report)

    reason_counts = Counter(row['reason'] for row in rejections)
    report = {
        'version': 1,
        'converter': CONVERTER_ID,
        'generated_at': utc_now(),
        'source_root': str(input_dir),
        'output_root': str(output_dir),
        'parameters': {
            'val_ratio': args.val_ratio,
            'seed': args.seed,
            'num_workers': args.num_workers,
            'max_rows_per_split': args.max_rows_per_split,
            'relative_paths': args.relative_paths,
        },
        'source_population_rows': source_population_rows,
        'processed_rows': processed_by_split,
        'indexed_rows': {
            split: len(rows)
            for split, rows in indexed.items()
        },
        'output_rows': validation['output_rows'],
        'rejected_rows': dict(sorted(reason_counts.items())),
        'normalized_duplicate_label_rows': sum(result.normalized_duplicate_labels for result in results),
        'removed_unreferenced_images': removed_orphan_images,
        'accounting': {
            split: {
                'read': processed_by_split[split],
                'retained': retained_by_source[split],
                'rejected': rejected_by_split[split],
                'balanced': True,
            }
            for split in ('train', 'test')
        },
        'split_policy': {
            'group': 'SHA-256 of original embedded image bytes',
            'official_test_protected': True,
            'train_val_assignment': f'SHA-256({CONVERTER_ID}:seed:media_sha256), val_ratio={args.val_ratio}',
        },
        'validation': validation,
        'source_readonly': {'status': 'passed', 'digest': before['digest']},
        'limitations': [
            'Official test rows have no labels and are not valid SFT targets.',
            'Category supervision is image-level presence only; no captions, boxes, masks, or counts are fabricated.',
            'An ms-swift loader/template smoke test requires the target ms-swift/model environment '
            'and is external to this converter.',
        ],
    }
    atomic_write_json(output_dir / 'conversion_report.json', report)
    atomic_write_json(output_dir / 'validation.json', validation)
    atomic_write_json(
        output_dir / MARKER_NAME,
        {'converter': CONVERTER_ID, 'status': 'complete', 'completed_at': utc_now()},
    )
    if not args.keep_temp:
        shutil.rmtree(output_dir / 'audit' / 'fragments')
        audit_dir = output_dir / 'audit'
        if audit_dir.exists() and not any(audit_dir.iterdir()):
            audit_dir.rmdir()
    print(json.dumps(report['output_rows'], ensure_ascii=False, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> None:
    convert(parse_args(argv))


if __name__ == '__main__':
    main()
