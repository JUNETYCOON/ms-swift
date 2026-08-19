#!/usr/bin/env python3
"""Build a small deterministic box-grounding eval set from a stage1 manifest.

This is meant for checkpoint-level W&B curves. It keeps only records that have
4-coordinate ground-truth boxes, because point annotations do not have IoU.
Each output row receives lightweight provenance fields:

  dataset_name, source_jsonl, source_line_no
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def stable_score(dataset_name: str, source_path: Path, line_no: int, record: dict[str, Any], seed: str) -> int:
    key_parts = [
        seed,
        dataset_name,
        str(source_path),
        str(line_no),
        str(record.get("id") or record.get("sample_id") or ""),
        json.dumps(record.get("images") or record.get("videos") or [], ensure_ascii=False, sort_keys=True),
    ]
    return int(hashlib.blake2b("|".join(key_parts).encode("utf-8"), digest_size=16).hexdigest(), 16)


def has_iou_box(record: dict[str, Any]) -> bool:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return False
    boxes = objects.get("bbox")
    if not isinstance(boxes, list) or not boxes:
        return False
    return any(isinstance(box, list) and len(box) == 4 for box in boxes)


def normalize_dataset_names(raw: list[str] | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip() for item in raw if item.strip()}


def manifest_eval_path(config: dict[str, Any]) -> str | None:
    for key in ("eval", "val", "validation", "source_eval"):
        value = config.get(key)
        if value:
            return str(value)
    return None


def build_eval_set(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    enabled_names = normalize_dataset_names(args.datasets)
    selected: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    scanned: dict[str, int] = {}
    eligible: dict[str, int] = {}

    for dataset_name, config in manifest.get("datasets", {}).items():
        if enabled_names is not None and dataset_name not in enabled_names:
            continue
        if not config.get("enabled", True):
            continue
        eval_path = manifest_eval_path(config)
        if not eval_path:
            continue
        source = Path(eval_path)
        if not source.is_file():
            continue
        scanned[dataset_name] = 0
        eligible[dataset_name] = 0
        with source.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                scanned[dataset_name] += 1
                record = json.loads(line)
                if not isinstance(record, dict) or not has_iou_box(record):
                    continue
                eligible[dataset_name] += 1
                record.setdefault("dataset_name", dataset_name)
                record.setdefault("source_jsonl", str(source))
                record.setdefault("source_line_no", line_no)
                score = stable_score(dataset_name, source, line_no, record, args.seed)
                bucket = selected[dataset_name]
                bucket.append((score, record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written_by_dataset: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for dataset_name in sorted(selected):
            rows = [record for _, record in sorted(selected[dataset_name], key=lambda item: item[0])]
            if args.max_per_dataset > 0:
                rows = rows[: args.max_per_dataset]
            written_by_dataset[dataset_name] = len(rows)
            for record in rows:
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "manifest": str(args.manifest),
        "output": str(args.output),
        "seed": args.seed,
        "max_per_dataset": args.max_per_dataset,
        "datasets_requested": sorted(enabled_names) if enabled_names else None,
        "scanned_rows": scanned,
        "eligible_iou_box_rows": eligible,
        "written_rows": sum(written_by_dataset.values()),
        "written_by_dataset": written_by_dataset,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a deterministic box-grounding IoU eval jsonl.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--seed", default="20260813")
    parser.add_argument("--max-per-dataset", type=int, default=64)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to include. Defaults to all enabled eval entrypoints with 4-coordinate boxes.",
    )
    return parser.parse_args()


def main() -> int:
    report = build_eval_set(parse_args())
    if report["written_rows"] <= 0:
        print("[error] no IoU-eligible box grounding rows were written")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
