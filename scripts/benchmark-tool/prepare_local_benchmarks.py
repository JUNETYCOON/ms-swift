#!/usr/bin/env python3
"""Prepare locally downloaded benchmark data for eval-custom."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set


DEFAULT_SOURCE_ROOT = Path('/mnt/luojunkun/stage1/benchmark-stage1')
DEFAULT_OUTPUT_ROOT = Path('/mnt/luojunkun/stage1/benchmark-eval-result/stage1-autoeval/prepared')
BENCHMARKS = ('flickr30k', 'ocrbench_v2', 'realworldqa', 'egoplan', 'openeqa', 'robospatial', 'video_mme')


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=BENCHMARKS)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip-media-extraction', action='store_true')
    return parser


def _parquet_rows(files: Sequence[Path], columns: Optional[Sequence[str]] = None) -> Iterator[Dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError('Preparing parquet benchmarks requires pyarrow.') from error
    for path in files:
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()


def _limited(rows: Iterable[Dict[str, Any]], limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            return
        yield row


def _safe_name(value: Any) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._') or 'sample'


def _write_bytes(path: Path, content: bytes) -> None:
    if path.is_file() and path.stat().st_size == len(content):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_stream(path: Path, source: Any, expected_size: int) -> None:
    if path.is_file() and path.stat().st_size == expected_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    temporary.replace(path)


def _extract_parquet_image(image: Mapping[str, Any], image_dir: Path, name: Any) -> Path:
    source_name = str(image.get('path') or '')
    suffix = Path(source_name).suffix.lower()
    if suffix not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}:
        suffix = '.jpg'
    target = image_dir / f'{_safe_name(name)}{suffix}'
    content = image.get('bytes')
    if not isinstance(content, bytes):
        raise ValueError(f'Parquet image {_safe_name(name)!r} does not contain encoded bytes.')
    _write_bytes(target, content)
    return target.resolve()


def _messages(prompt: str, answer: str, media_token: str) -> List[Dict[str, str]]:
    return [
        {'role': 'user', 'content': f'{media_token}\n{prompt}'},
        {'role': 'assistant', 'content': answer},
    ]


def _prepare_flickr30k(source: Path, output: Path, limit: Optional[int], extract_media: bool) -> Iterable[Dict[str, Any]]:
    files = sorted((source / 'flickr30k' / 'data').glob('test-*.parquet'))
    image_dir = output / 'images'
    for row in _limited(_parquet_rows(files), limit):
        captions = [str(value) for value in row['caption']]
        image_path = _extract_parquet_image(row['image'], image_dir, row['img_id']) if extract_media else (
            image_dir / f"{_safe_name(row['img_id'])}.jpg")
        yield {
            'id': str(row['img_id']),
            'benchmark': 'flickr30k',
            'filename': row['filename'],
            'messages': _messages('Describe the image in one concise sentence.', captions[0], '<image>'),
            'answers': captions,
            'images': [str(image_path)],
        }


def _prepare_ocrbench(source: Path, output: Path, limit: Optional[int], extract_media: bool) -> Iterable[Dict[str, Any]]:
    files = sorted((source / 'ocrbench' / 'data').glob('test-*.parquet'))
    image_dir = output / 'images'
    for row in _limited(_parquet_rows(files), limit):
        answers = [str(value) for value in row['answers']]
        image_path = _extract_parquet_image(row['image'], image_dir, row['id']) if extract_media else (
            image_dir / f"{_safe_name(row['id'])}.jpg")
        yield {
            'id': str(row['id']),
            'benchmark': 'ocrbench_v2',
            'messages': _messages(str(row['question']), answers[0], '<image>'),
            'answers': answers,
            'images': [str(image_path)],
            'ocr_dataset_name': row['dataset_name'],
            'ocr_type': row['type'],
            'ocr_eval': row['eval'],
            'ocr_bbox': row['bbox'],
            'ocr_bbox_list': row['bbox_list'],
            'ocr_content': row['content'],
        }


def _prepare_realworldqa(source: Path, output: Path, limit: Optional[int], extract_media: bool) -> Iterable[Dict[str, Any]]:
    files = sorted((source / 'realworldqa' / 'data').glob('test-*.parquet'))
    image_dir = output / 'images'
    for index, row in enumerate(_limited(_parquet_rows(files), limit)):
        image_path = _extract_parquet_image(row['image'], image_dir, index) if extract_media else image_dir / f'{index}.jpg'
        yield {
            'id': str(index),
            'benchmark': 'realworldqa',
            'messages': _messages(str(row['question']), str(row['answer']), '<image>'),
            'answers': [str(row['answer'])],
            'images': [str(image_path)],
            'original_image_path': row['image_path'],
        }


def _extract_zip_archives(
        archives: Sequence[Path], target_dir: Path, wanted_names: Optional[Set[str]] = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    remaining = set(wanted_names) if wanted_names is not None else None
    for archive in archives:
        if remaining is not None and not remaining:
            break
        print(f'[prepare] extracting {archive.name}', flush=True)
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                if member.is_dir():
                    continue
                member_name = PurePosixPath(member.filename).name
                if not member_name or (remaining is not None and member_name not in remaining):
                    continue
                target = target_dir / member_name
                if target.is_file() and target.stat().st_size == member.file_size:
                    if remaining is not None:
                        remaining.discard(member_name)
                    continue
                with stream.open(member) as source:
                    _write_stream(target, source, member.file_size)
                if remaining is not None:
                    remaining.discard(member_name)
    if remaining:
        preview = ', '.join(sorted(remaining)[:5])
        raise FileNotFoundError(f'Could not find {len(remaining)} requested files in zip archives: {preview}')


def _prepare_egoplan(source: Path, output: Path, limit: Optional[int], extract_media: bool) -> Iterable[Dict[str, Any]]:
    benchmark_root = source / 'egoplan'
    video_dir = output / 'videos'
    parquet = benchmark_root / 'egoplan' / 'validation-00000-of-00001.parquet'
    rows = list(_limited(_parquet_rows([parquet]), limit))
    if extract_media:
        wanted = {f'{row["rearranged_id"]}.mp4' for row in rows}
        _extract_zip_archives(sorted(benchmark_root.glob('videos_chunked_*.zip')), video_dir, wanted)
    for row in rows:
        options = '\n'.join(f'{letter}. {row[f"choice_{letter.lower()}"]}' for letter in 'ABCD')
        prompt = (
            f'Task goal: {row["task_goal"]}\n{row["question"]}\n{options}\n'
            'Respond with only the letter A, B, C, or D.'
        )
        answer = str(row['golden_choice_idx'])
        yield {
            'id': str(row['sample_id']),
            'benchmark': 'egoplan',
            'messages': _messages(prompt, answer, '<video>'),
            'answers': [answer],
            'videos': [str((video_dir / f'{row["rearranged_id"]}.mp4').resolve())],
            'video_id': row['video_id'],
            'video_source': row['video_source'],
            'is_valid': row['is_valid'],
        }


def _extract_tar_archives(
        archives: Sequence[Path], target_dir: Path, wanted_names: Optional[Set[str]] = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    remaining = set(wanted_names) if wanted_names is not None else None
    for archive in archives:
        if remaining is not None and not remaining:
            break
        print(f'[prepare] extracting {archive.name}', flush=True)
        with tarfile.open(archive, mode='r:gz') as stream:
            for member in stream:
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if len(parts) != 2 or parts[0] not in {'hm3d-v0', 'scannet-v0'}:
                    raise ValueError(f'Unsafe or unexpected OpenEQA tar member: {member.name!r}.')
                member_name = '/'.join(parts)
                if remaining is not None and member_name not in remaining:
                    continue
                target = target_dir.joinpath(*parts)
                if target.is_file() and target.stat().st_size == member.size:
                    if remaining is not None:
                        remaining.discard(member_name)
                    continue
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError(f'Cannot read tar member: {member.name!r}.')
                with source:
                    _write_stream(target, source, member.size)
                if remaining is not None:
                    remaining.discard(member_name)
    if remaining:
        preview = ', '.join(sorted(remaining)[:5])
        raise FileNotFoundError(f'Could not find {len(remaining)} requested files in tar archives: {preview}')


def _prepare_openeqa(source: Path, output: Path, limit: Optional[int], extract_media: bool) -> Iterable[Dict[str, Any]]:
    benchmark_root = source / 'openeqa'
    video_dir = output / 'videos'
    parquet = benchmark_root / 'v0' / 'test-00000-of-00001.parquet'
    rows = list(_limited(_parquet_rows([parquet]), limit))
    if extract_media:
        wanted = {f'{row["episode_history"]}.mp4' for row in rows}
        archives = [benchmark_root / 'hm3d-v0.tar.gz', benchmark_root / 'scannet-v0.tar.gz']
        _extract_tar_archives(archives, video_dir, wanted)
    for row in rows:
        answers = [str(row['answer'])]
        answers.extend(str(value) for value in (row.get('extra_answers') or []))
        yield {
            'id': str(row['question_id']),
            'benchmark': 'openeqa',
            'messages': _messages(str(row['question']), answers[0], '<video>'),
            'answers': answers,
            'videos': [str((video_dir / f'{row["episode_history"]}.mp4').resolve())],
            'category': row['category'],
            'episode_history': row['episode_history'],
        }


def _prepare_robospatial(source: Path, output: Path, limit: Optional[int], _extract_media: bool) -> Iterable[Dict[str, Any]]:
    benchmark_root = source / 'robospatial'
    records = json.loads((benchmark_root / 'annotations.json').read_text(encoding='utf-8'))
    for index, row in enumerate(_limited(iter(records), limit)):
        prompt = str(row['question'])
        if row['category'] == 'context':
            prompt = prompt.split('Your answer should')[0].strip()
            prompt += (
                '\n\nOutput one point coordinate in JSON format.\n'
                'For example:\n[{"point_2d": [x, y], "label": "point_1"}]'
            )
        yield {
            'id': str(index),
            'benchmark': 'robospatial',
            'messages': _messages(prompt, str(row['answer']), '<image>'),
            'answers': [str(row['answer'])],
            'images': [str((benchmark_root / row['img']).resolve())],
            'category': row['category'],
            'depth_image': str((benchmark_root / row['depth_image']).resolve()),
            'mask': str((benchmark_root / row['mask']).resolve()) if row.get('mask') else None,
        }


def _prepare_video_mme(source: Path, output: Path, limit: Optional[int], extract_media: bool) -> Iterable[Dict[str, Any]]:
    benchmark_root = source / 'video-mme'
    video_dir = output / 'videos'
    parquet = benchmark_root / 'videomme' / 'test-00000-of-00001.parquet'
    rows = list(_limited(_parquet_rows([parquet]), limit))
    if extract_media:
        wanted = {f'{row["videoID"]}.mp4' for row in rows}
        _extract_zip_archives(sorted(benchmark_root.glob('videos_chunked_*.zip')), video_dir, wanted)
    for row in rows:
        options = '\n'.join(str(value) for value in row['options'])
        prompt = (
            'Select the best answer to the following multiple-choice question based on the video.\n'
            f'{row["question"]}\n{options}\nRespond with only the letter A, B, C, or D.'
        )
        answer = str(row['answer'])
        yield {
            'id': str(row['question_id']),
            'benchmark': 'video_mme',
            'messages': _messages(prompt, answer, '<video>'),
            'answers': [answer],
            'videos': [str((video_dir / f'{row["videoID"]}.mp4').resolve())],
            'video_id': row['videoID'],
            'duration': row['duration'],
            'domain': row['domain'],
            'sub_category': row['sub_category'],
            'task_type': row['task_type'],
        }


PREPARERS = {
    'flickr30k': _prepare_flickr30k,
    'ocrbench_v2': _prepare_ocrbench,
    'realworldqa': _prepare_realworldqa,
    'egoplan': _prepare_egoplan,
    'openeqa': _prepare_openeqa,
    'robospatial': _prepare_robospatial,
    'video_mme': _prepare_video_mme,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be greater than zero.')
    output_dir = (args.output_root / args.benchmark).resolve()
    output_file = output_dir / 'eval.jsonl'
    if output_file.exists() and not args.overwrite:
        print(f'[prepare] already exists: {output_file}')
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_file = output_dir / 'eval.jsonl.tmp'
    rows = PREPARERS[args.benchmark](
        args.source_root.resolve(), output_dir, args.limit, not args.skip_media_extraction)
    count = 0
    with temp_file.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    temp_file.replace(output_file)
    print(f'[prepare] wrote {count} samples: {output_file}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
