#!/usr/bin/env python3
"""Split an already-clean RoboVQA JSONL by complete video identity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from grouped_jsonl_split import record_group, split_jsonl


DATA_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/robovqa")


def video_group(record: dict[str, Any], path: Path) -> str:
    return record_group(record, "videos", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DATA_DIR / "robovqa_train_sft.jsonl")
    parser.add_argument("--train-output", type=Path, default=DATA_DIR / "robovqa_train_sft_train.jsonl")
    parser.add_argument("--eval-output", type=Path, default=DATA_DIR / "robovqa_train_sft_eval.jsonl")
    parser.add_argument("--report-output", type=Path, default=DATA_DIR / "robovqa_grouped_split_report.json")
    parser.add_argument("--reserved-eval-json", type=Path, nargs="*", default=[])
    parser.add_argument("--reserve-only", action="store_true")
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = split_jsonl(
        input_paths=[args.input_json],
        train_output=args.train_output,
        eval_output=args.eval_output,
        report_output=args.report_output,
        group_resolver=video_group,
        group_key_description="canonical videos path / episode ID",
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        reserved_eval_paths=args.reserved_eval_json,
        reserve_only=args.reserve_only,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
        report_extra={"reasoning_cleanup_required_before_split": True},
    )
    print(f"[done] train={report['rows']['train']:,} eval={report['rows']['eval']:,}")
    print(f"[ok] video_overlap={report['groups']['train_eval_overlap']}")


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
