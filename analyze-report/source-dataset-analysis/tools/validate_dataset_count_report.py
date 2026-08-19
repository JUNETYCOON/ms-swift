#!/usr/bin/env python3
"""Validate dataset count evidence and the standalone HTML report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def file_metadata(files: list[dict]) -> list[dict]:
    keys = ('path', 'exists', 'size_bytes', 'mtime_ns')
    return [{key: item.get(key) for key in keys} for item in files]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, nargs='+', required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    before = load_json(args.before)
    after_datasets = {}
    for path in args.after:
        after_datasets.update(load_json(path)['datasets'])
    comparisons = []
    for name, old in before['datasets'].items():
        new = after_datasets.get(name)
        source_unchanged = new is not None and old['source_snapshot'] == new['source_snapshot']
        inputs_unchanged = new is not None and all(
            file_metadata(old[field]) == file_metadata(new[field]) for field in ('source_train', 'train', 'eval'))
        comparisons.append({
            'dataset': name,
            'source_unchanged': source_unchanged,
            'count_inputs_unchanged': inputs_unchanged,
            'unchanged': source_unchanged and inputs_unchanged,
            'before_source_fingerprint': old['source_snapshot']['fingerprint'],
            'after_source_fingerprint': None if new is None else new['source_snapshot']['fingerprint'],
        })
    evidence = {
        'version': 1,
        'source_host': before['host'],
        'source_root': before['source_root'],
        'converted_inputs_root': before['converted_root'],
        'before_generated_at': before['generated_at'],
        'after_generated_at': [load_json(path)['generated_at'] for path in args.after],
        'all_unchanged': all(item['unchanged'] for item in comparisons),
        'datasets': comparisons,
        'scope': '18 canonical source directories and configured source_train/train/eval JSONL inputs',
    }
    evidence_path = args.report_dir / 'source-readonly-evidence.json'
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    summary = load_json(args.report_dir / 'dataset-counts.json')
    report = (args.report_dir / 'dataset-counts.html').read_text(encoding='utf-8')
    checks = {
        'source_and_count_inputs_unchanged': evidence['all_unchanged'],
        'dataset_count_is_18': summary['dataset_count'] == 18,
        'table_row_count_is_18': report.count('<tr data-search=') == 18,
        'utf8_meta': '<meta charset="utf-8">' in report,
        'viewport_meta': 'name="viewport"' in report,
        'has_search_control': 'id="search"' in report,
        'has_status_filter': 'id="status"' in report,
        'no_external_resources': not re.search(r'(?:src|href)=["\']https?://', report, re.IGNORECASE),
        'global_train_denominator_is_17': summary['global_train_configured_datasets'] == 17,
        'global_train_available_is_zero': summary['global_train_available_datasets'] == 0,
    }
    validation = {
        'version': 1,
        'status': 'passed' if all(checks.values()) else 'failed',
        'checks': checks,
        'dataset_count': summary['dataset_count'],
        'current_available_sum': summary['current_available_sum'],
    }
    (args.report_dir / 'validation.json').write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(validation, ensure_ascii=False))
    if validation['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
