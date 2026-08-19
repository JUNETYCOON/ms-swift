#!/usr/bin/env python3
"""Split AI2D by image content hash and preserve an existing strict eval set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from grouped_jsonl_split import media_hash_group_resolver, split_jsonl


DATA_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/ai2d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DATA_DIR / "ai2d_pretrain_msswift.jsonl")
    parser.add_argument("--train-output", type=Path, default=DATA_DIR / "ai2d_pretrain_msswift_train.jsonl")
    parser.add_argument("--eval-output", type=Path, default=DATA_DIR / "ai2d_pretrain_msswift_eval.jsonl")
    parser.add_argument("--report-output", type=Path, default=DATA_DIR / "ai2d_split_report.json")
    parser.add_argument("--reserved-eval-json", type=Path, nargs="*", default=[])
    parser.add_argument("--reserve-only", action="store_true")
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10_000)
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
        progress_every=args.progress_every,
        overwrite=args.overwrite,
        report_extra={
            "training_entrypoint": "strict _train only",
            "source_plus_eval_forbidden": True,
            "hash_algorithm": "sha256",
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
