#!/usr/bin/env python3
"""Collect read-only source snapshots and canonical JSONL counts on stage1."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path


SOURCE_ROOT = Path('/mnt/luojunkun/stage1/dataset')
CONVERTED_ROOT = Path('/mnt/luojunkun/stage1/dataset_ms-swift')

DATASETS = {
    'COCO': {
        'source': SOURCE_ROOT / 'COCO' / 'COCO-MODELSCOPE',
    },
    'VQAv2': {
        'source': SOURCE_ROOT / 'VQAv2',
        'source_train': [CONVERTED_ROOT / 'VQAv2' / 'vqav2_train_sft_msswift.jsonl'],
        'train': [CONVERTED_ROOT / 'VQAv2' / 'vqav2_train_sft_msswift_global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'VQAv2' / 'vqav2_validation_sft_msswift.jsonl'],
    },
    'VisualGenome': {
        'source': SOURCE_ROOT / 'VisualGenome',
        'source_train': [
            CONVERTED_ROOT / 'visualgenome' / 'visualgenome_qa_train.jsonl',
            CONVERTED_ROOT / 'visualgenome' / 'visualgenome_regions_train.jsonl',
        ],
        'train': [
            CONVERTED_ROOT / 'visualgenome' / 'visualgenome_qa_global_train.jsonl',
            CONVERTED_ROOT / 'visualgenome' / 'visualgenome_regions_global_train.jsonl',
        ],
        'eval': [
            CONVERTED_ROOT / 'visualgenome' / 'visualgenome_qa_val.jsonl',
            CONVERTED_ROOT / 'visualgenome' / 'visualgenome_regions_val.jsonl',
        ],
    },
    'GQA': {
        'source': SOURCE_ROOT / 'GQA',
        'source_train': [CONVERTED_ROOT / 'gqa' / 'gqa_train_balanced_sft_msswift.jsonl'],
        'train': [CONVERTED_ROOT / 'gqa' / 'gqa_train_balanced_sft_msswift_global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'gqa' / 'gqa_val_balanced_sft_msswift.jsonl'],
    },
    'TextVQA': {
        'source': SOURCE_ROOT / 'textvqa',
        'source_train': [CONVERTED_ROOT / 'textvqa' / 'textvqa_train_sft_msswift.jsonl'],
        'train': [CONVERTED_ROOT / 'textvqa' / 'textvqa_train_sft_msswift_global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'textvqa' / 'textvqa_validation_sft_msswift.jsonl'],
    },
    'ChartQA': {
        'source': SOURCE_ROOT / 'Chartqa',
        'source_train': [CONVERTED_ROOT / 'chartqa' / 'chartqa_train_sft_msswift.jsonl'],
        'train': [CONVERTED_ROOT / 'chartqa' / 'chartqa_train_sft_msswift_global_train.jsonl'],
        'eval': [
            CONVERTED_ROOT / 'chartqa' / 'chartqa_val_sft_msswift.jsonl',
            CONVERTED_ROOT / 'chartqa' / 'chartqa_test_sft_msswift.jsonl',
        ],
    },
    'AI2D': {
        'source': SOURCE_ROOT / 'ai2d' / 'ai2d',
        'source_train': [CONVERTED_ROOT / 'ai2d' / 'ai2d_pretrain_msswift_train.jsonl'],
        'train': [CONVERTED_ROOT / 'ai2d' / 'ai2d_pretrain_msswift_global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'ai2d' / 'ai2d_pretrain_msswift_eval.jsonl'],
    },
    'LLaVA-Instruct': {
        'source': SOURCE_ROOT / 'llava-instruct',
        'source_train': [
            CONVERTED_ROOT / 'llava-instruct' / 'llava_v1_5_mix665k_sft_msswift_train.jsonl'
        ],
        'train': [
            CONVERTED_ROOT / 'llava-instruct' / 'llava_v1_5_mix665k_sft_msswift_global_train.jsonl'
        ],
        'eval': [CONVERTED_ROOT / 'llava-instruct' / 'llava_v1_5_mix665k_sft_msswift_val.jsonl'],
    },
    'VLM-R1': {
        'source': SOURCE_ROOT / 'vlm-r1',
        'source_train': [
            CONVERTED_ROOT / 'vlm-r1' / 'vlm_r1_sft_grounding_msswift_train.jsonl'
        ],
        'train': [
            CONVERTED_ROOT / 'vlm-r1' / 'vlm_r1_sft_grounding_msswift_global_train.jsonl'
        ],
        'eval': [CONVERTED_ROOT / 'vlm-r1' / 'vlm_r1_sft_grounding_msswift_eval.jsonl'],
    },
    'Robo2VLM': {
        'source': SOURCE_ROOT / 'robo2vlm',
        'source_train': [CONVERTED_ROOT / 'robo2vlm' / 'robo2vlm_sft_train.jsonl'],
        'train': [CONVERTED_ROOT / 'robo2vlm' / 'robo2vlm_sft_global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'robo2vlm' / 'robo2vlm_sft_eval.jsonl'],
    },
    'RoboVQA': {
        'source': SOURCE_ROOT / 'robovqa',
        'source_train': [CONVERTED_ROOT / 'robovqa' / 'robovqa_train_sft_train.jsonl'],
        'train': [CONVERTED_ROOT / 'robovqa' / 'robovqa_train_sft_global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'robovqa' / 'robovqa_train_sft_eval.jsonl'],
    },
    'SpatialVLM': {
        'source': SOURCE_ROOT / 'spatialvlm',
        'source_train': [CONVERTED_ROOT / 'spatialvlm' / 'train.jsonl'],
        'train': [CONVERTED_ROOT / 'spatialvlm' / 'global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'spatialvlm' / 'val.jsonl'],
    },
    'PixMo-Cap': {
        'source': SOURCE_ROOT / 'pixmo-cap',
        'source_train': [CONVERTED_ROOT / 'pixmo-cap' / 'train.jsonl'],
        'train': [CONVERTED_ROOT / 'pixmo-cap' / 'global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'pixmo-cap' / 'val.jsonl'],
    },
    'PixMo-Points': {
        'source': SOURCE_ROOT / 'pixmo-points',
        'source_train': [CONVERTED_ROOT / 'pixmo-points' / 'train.jsonl'],
        'train': [CONVERTED_ROOT / 'pixmo-points' / 'global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'pixmo-points' / 'val.jsonl'],
    },
    'Molmo2-VideoCapQA': {
        'source': SOURCE_ROOT / 'Molmo2-VideoCapQA',
        'source_train': [CONVERTED_ROOT / 'Molmo2-VideoCapQA' / 'train.jsonl'],
        'train': [CONVERTED_ROOT / 'Molmo2-VideoCapQA' / 'global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'Molmo2-VideoCapQA' / 'val.jsonl'],
    },
    'Molmo2-VideoPoint': {
        'source': SOURCE_ROOT / 'Molmo2-VideoPoint',
        'source_train': [CONVERTED_ROOT / 'Molmo2-VideoPoint' / 'train.jsonl'],
        'train': [CONVERTED_ROOT / 'Molmo2-VideoPoint' / 'global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'Molmo2-VideoPoint' / 'val.jsonl'],
    },
    'Molmo2-VideoSubtitleQA': {
        'source': SOURCE_ROOT / 'Molmo2-VideoSubtitleQA',
        'source_train': [CONVERTED_ROOT / 'Molmo2-VideoSubtitleQA' / 'train.jsonl'],
        'train': [CONVERTED_ROOT / 'Molmo2-VideoSubtitleQA' / 'global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'Molmo2-VideoSubtitleQA' / 'val.jsonl'],
    },
    'Molmo2-VideoTrack': {
        'source': SOURCE_ROOT / 'Molmo2-VideoTrack',
        'source_train': [CONVERTED_ROOT / 'Molmo2-VideoTrack' / 'train.jsonl'],
        'train': [CONVERTED_ROOT / 'Molmo2-VideoTrack' / 'global_train.jsonl'],
        'eval': [CONVERTED_ROOT / 'Molmo2-VideoTrack' / 'val.jsonl'],
    },
}


def tree_snapshot(root: Path) -> dict:
    entries = []
    if root.is_dir():
        for current_root, directory_names, file_names in os.walk(root):
            directory_names.sort()
            current = Path(current_root)
            for file_name in sorted(file_names):
                path = current / file_name
                if path.is_symlink():
                    entries.append((path.relative_to(root).as_posix(), 'symlink', os.readlink(path)))
                    continue
                stat = path.stat()
                entries.append((path.relative_to(root).as_posix(), 'file', stat.st_size, stat.st_mtime_ns))
    canonical = '\n'.join(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) for entry in entries)
    return {
        'path': str(root),
        'exists': root.is_dir(),
        'file_count': sum(entry[1] == 'file' for entry in entries),
        'total_bytes': sum(entry[2] for entry in entries if entry[1] == 'file'),
        'fingerprint': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
    }


def count_lines(path: Path) -> dict:
    result = {'path': str(path), 'exists': path.is_file(), 'lines': None, 'size_bytes': None, 'mtime_ns': None}
    if not result['exists']:
        return result
    stat = path.stat()
    if os.environ.get('METADATA_ONLY') == '1':
        result.update({'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        return result
    lines = 0
    last_byte = b''
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            lines += chunk.count(b'\n')
            last_byte = chunk[-1:]
    if stat.st_size and last_byte != b'\n':
        lines += 1
    result.update({'lines': lines, 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return result


def main() -> None:
    selected_names = os.environ.get('DATASET_NAMES')
    selected = set(selected_names.split(',')) if selected_names else set(DATASETS)
    unknown = selected - set(DATASETS)
    if unknown:
        raise SystemExit(f'Unknown datasets: {sorted(unknown)}')
    result = {
        'version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(),
        'source_root': str(SOURCE_ROOT),
        'converted_root': str(CONVERTED_ROOT),
        'datasets': {},
    }
    for name, config in DATASETS.items():
        if name not in selected:
            continue
        row = {'source_snapshot': tree_snapshot(config['source'])}
        for field in ('source_train', 'train', 'eval'):
            row[field] = [count_lines(path) for path in config.get(field, [])]
        result['datasets'][name] = row
    print(json.dumps(result, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
