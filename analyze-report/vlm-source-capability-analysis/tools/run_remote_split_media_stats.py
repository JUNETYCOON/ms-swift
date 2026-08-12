#!/usr/bin/env python3
"""Run source split-media audit over SSH and atomically save the result."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
DESTINATION = HERE.parent / "artifacts" / "source-split-media-statistics.json"


def main() -> int:
    script = (HERE / "remote_collect_split_media_stats.py").read_bytes()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix="source-split-media-",
            suffix=".json",
            dir=DESTINATION.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            process = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "8.146.226.25",
                    "env",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "python3",
                    "-",
                ],
                input=script,
                stdout=stream,
                stderr=subprocess.PIPE,
                check=False,
                timeout=3600,
            )
        if process.returncode:
            raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
        assert temporary is not None
        result = json.loads(temporary.read_text(encoding="utf-8"))
        if not result.get("source_only_contract"):
            raise RuntimeError("collector did not assert source-only contract")
        temporary.replace(DESTINATION)
        temporary = None
        print(f"wrote {DESTINATION}")
        return 0
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
