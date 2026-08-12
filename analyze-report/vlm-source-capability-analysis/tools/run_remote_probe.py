#!/usr/bin/env python3
"""Run an approved read-only JSON probe over SSH and save atomically."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPORT_ROOT = HERE.parent
REMOTE_HOST = "8.146.226.25"
PROBES = {
    "video-archives": (
        "remote_probe_video_archives.py",
        REPORT_ROOT / "artifacts" / "video-archive-probe.json",
    ),
    "legacy-tabular-stats": (
        "remote_collect_tabular_stats.py",
        REPORT_ROOT / "artifacts" / "legacy-tabular-full-statistics.json",
    ),
}


def run(name: str) -> None:
    script_name, output = PROBES[name]
    script = (HERE / script_name).read_bytes()
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
        ],
        input=script,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=900,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
    payload = json.loads(process.stdout.decode("utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False, newline="\n"
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=tuple(PROBES))
    args = parser.parse_args()
    run(args.probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
