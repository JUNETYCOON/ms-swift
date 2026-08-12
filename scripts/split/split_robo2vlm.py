#!/usr/bin/env python3
"""Merge Robo2VLM source splits and split complete episode/frame lineages."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from grouped_jsonl_split import split_jsonl


DATA_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/robo2vlm")
QUESTION_SUFFIX = re.compile(r"_q\d+$", re.IGNORECASE)


def episode_group(record: dict[str, Any], _path: Path) -> str:
    sample_id = str(record.get("id") or "").strip()
    if not sample_id:
        raise ValueError("Robo2VLM row has no id")
    lineage = QUESTION_SUFFIX.sub("", sample_id)
    if lineage == sample_id:
        raise ValueError(f"Robo2VLM id has no _qN suffix: {sample_id!r}")
    return f"robo2vlm:{lineage.casefold()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        nargs="+",
        default=[DATA_DIR / "robo2vlm_train.jsonl", DATA_DIR / "robo2vlm_test.jsonl"],
    )
    parser.add_argument("--train-output", type=Path, default=DATA_DIR / "robo2vlm_sft_train.jsonl")
    parser.add_argument("--eval-output", type=Path, default=DATA_DIR / "robo2vlm_sft_eval.jsonl")
    parser.add_argument("--report-output", type=Path, default=DATA_DIR / "robo2vlm_grouped_split_report.json")
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
        input_paths=args.input_json,
        train_output=args.train_output,
        eval_output=args.eval_output,
        report_output=args.report_output,
        group_resolver=episode_group,
        group_key_description="record.id with trailing _qN removed (underlying episode/frame lineage)",
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        reserved_eval_paths=args.reserved_eval_json,
        reserve_only=args.reserve_only,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
        report_extra={
            "task_type": "multiple-choice VQA",
            "caption_label_forbidden": True,
            "source_test_is_not_an_independent_eval": True,
        },
    )
    print(f"[done] train={report['rows']['train']:,} eval={report['rows']['eval']:,}")
    print(f"[ok] episode_overlap={report['groups']['train_eval_overlap']}")


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
