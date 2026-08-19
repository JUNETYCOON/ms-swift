#!/usr/bin/env python3
"""Run source-only JSON statistics over SSH and atomically store artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPORT_ROOT = HERE.parent
REMOTE_HOST = "8.146.226.25"
COLLECTOR = HERE / "remote_collect_json_stats.py"
DATASETS = ("LLaVA-Instruct", "VLM-R1", "VisualGenome", "AI2D", "RoboVQA")
TIMEOUTS = {
    "LLaVA-Instruct": 1800,
    "VLM-R1": 1200,
    "VisualGenome": 1800,
    "AI2D": 900,
    "RoboVQA": 7200,
}


def artifact_path(dataset: str) -> Path:
    slug = dataset.lower().replace("-", "_")
    return REPORT_ROOT / "artifacts" / f"source-json-statistics-{slug}.json"


def run(dataset: str) -> None:
    script = COLLECTOR.read_bytes()
    destination = artifact_path(dataset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix="source-json-statistics-",
            suffix=".json",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            process = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    REMOTE_HOST,
                    "env",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "python3",
                    "-",
                    dataset,
                ],
                input=script,
                stdout=stream,
                stderr=subprocess.PIPE,
                check=False,
                timeout=TIMEOUTS[dataset],
            )
        if process.returncode:
            raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
        assert temporary is not None
        with temporary.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        if result.get("dataset") != dataset or not result.get("source_only_contract"):
            raise RuntimeError(f"unexpected collector result for {dataset}")
        temporary.replace(destination)
        temporary = None
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        if stderr:
            print(stderr)
        print(f"wrote {destination}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=DATASETS)
    args = parser.parse_args()
    for dataset in args.datasets:
        print(f"collecting {dataset}", flush=True)
        run(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
