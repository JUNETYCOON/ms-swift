#!/usr/bin/env python3
"""Run the stage1 entrypoint validator and archive machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = ["python3", str(args.validator), "--manifest", str(args.manifest)]
    completed = subprocess.run(command, capture_output=True, text=True)
    train_entries = []
    eval_entries = []
    for line in completed.stdout.splitlines():
        if line.startswith("[train]"):
            train_entries.append(line)
        elif line.startswith("[eval]"):
            eval_entries.append(line)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "train_entries": train_entries,
                "eval_entries": eval_entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "returncode": completed.returncode,
                "train_entries": len(train_entries),
                "eval_entries": len(eval_entries),
            },
            ensure_ascii=False,
        )
    )
    return 0 if completed.returncode == 0 else completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
