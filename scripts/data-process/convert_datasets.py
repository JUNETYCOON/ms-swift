#!/usr/bin/env python3
"""Unified entry point for data-process scripts.

Examples:
  python convert_datasets.py list
  python convert_datasets.py run gqa --help
  python convert_datasets.py run gqa --gqa-root /data/gqa --output out.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


sys.path.insert(0, str(Path(__file__).resolve().parent))
from processors import PROCESSORS, ProcessingError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered processors.")
    list_parser.add_argument("--json", action="store_true", help="Print JSON output.")

    run_parser = subparsers.add_parser(
        "run", help="Run one processor with its native CLI arguments."
    )
    run_parser.add_argument("dataset", choices=sorted(PROCESSORS))
    run_parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Working directory for the underlying script.",
    )
    run_parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    run_parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _main(argv: Sequence[str]) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv))

    if args.command == "list":
        rows = []
        for name in sorted(PROCESSORS):
            processor = PROCESSORS[name]
            rows.append(
                {
                    "name": processor.name,
                    "description": processor.description,
                    "script": processor.script,
                    "task_types": list(processor.task_types),
                    "bbox_type": processor.bbox_type,
                }
            )
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            print("{:<28} {:<52} {}".format("NAME", "DESCRIPTION", "SCRIPT"))
            for row in rows:
                print(
                    "{:<28} {:<52} {}".format(
                        row["name"], row["description"], row["script"]
                    )
                )
        return 0

    if args.command == "run":
        processor = PROCESSORS[args.dataset]
        try:
            summary = processor.run(args.args, args.workdir)
        except ProcessingError as exc:
            print("ERROR: {}".format(exc), file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parser.error("unknown command: {}".format(args.command))
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))

