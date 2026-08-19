#!/usr/bin/env python3
"""Write small JSON placeholders for report links before final validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    values = {
        "validation.json": {"status": "pending"},
        "browser-validation.json": {
            "status": "not_run",
            "reason": "Playwright browser check was not run in this pass.",
        },
        "skill-validator-result.json": {
            "status": "not_run",
            "reason": "Repository validator was used for this converted ms-swift audit.",
        },
    }
    for name, value in values.items():
        (args.root / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"written": sorted(values)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
