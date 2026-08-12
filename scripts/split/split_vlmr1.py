#!/usr/bin/env python3
"""Split VLM-R1 by image content hash, keeping all QA for an image together."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from grouped_jsonl_split import media_hash_group_resolver, record_values, split_jsonl


DATA_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/vlm-r1")


def task_kind(record: dict[str, Any]) -> str:
    return record_values(record, "__kind__", Path("."))[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DATA_DIR / "vlm_r1_sft_grounding_msswift.jsonl")
    parser.add_argument("--train-output", type=Path, default=DATA_DIR / "vlm_r1_sft_grounding_msswift_train.jsonl")
    parser.add_argument("--eval-output", type=Path, default=DATA_DIR / "vlm_r1_sft_grounding_msswift_eval.jsonl")
    parser.add_argument("--report-output", type=Path, default=DATA_DIR / "vlm_r1_grouped_split_report.json")
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
        group_resolver=media_hash_group_resolver("images", "sha256"),
        group_key_description="SHA-256 of image bytes",
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        reserved_eval_paths=args.reserved_eval_json,
        reserve_only=args.reserve_only,
        stratum_resolver=task_kind,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
        report_extra={
            "hash_algorithm": "sha256",
            "cross_dataset_coco_overlap_requires_global_filter": True,
        },
    )
    print(f"[done] train={report['rows']['train']:,} eval={report['rows']['eval']:,}")
    print(f"[ok] image_hash_overlap={report['groups']['train_eval_overlap']}")


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
