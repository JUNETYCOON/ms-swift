#!/usr/bin/env python3
"""Collect PixMo-Points source metadata without fetching external media."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from run_remote_collector import safe_extract


HERE = Path(__file__).resolve().parent
REPORT_ROOT = HERE.parent
TARGET = REPORT_ROOT / "sub-dataset" / "PixMo-Points"
REMOTE_HOST = "8.146.226.25"


def main() -> int:
    script = (HERE / "remote_collect_images.py").read_bytes()
    archive: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="pixmo-points-metadata-", suffix=".tar", delete=False
        ) as output:
            archive = Path(output.name)
            process = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    REMOTE_HOST,
                    "env",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "PIXMO_METADATA_ONLY=1",
                    "python3",
                    "-",
                    "PixMo-Points",
                ],
                input=script,
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
                timeout=2400,
            )
        if process.returncode:
            raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
        assert archive is not None
        safe_extract(archive, TARGET)
        print(TARGET)
        return 0
    finally:
        if archive is not None:
            archive.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
