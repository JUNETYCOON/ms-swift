#!/usr/bin/env python3
"""Run the VideoPoint media collector and merge its local sampling summary."""

from __future__ import annotations

import json
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
REPORT_ROOT = HERE.parent
TARGET = REPORT_ROOT / "sub-dataset" / "Molmo2-VideoPoint"
REMOTE_HOST = "8.146.226.25"


def safe_extract(archive: Path, target: Path) -> None:
    with tarfile.open(archive, "r:") as stream:
        members = stream.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe tar member: {member.name}")
        target.mkdir(parents=True, exist_ok=True)
        stream.extractall(target, members=members, filter="data")


def merge_summary() -> None:
    summary_path = TARGET / "sampling-summary.json"
    media_path = TARGET / "media-sampling-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    media = json.loads(media_path.read_text(encoding="utf-8"))
    annotation_population = summary.get("population_total")
    full_statistics = summary.get("full_statistics")
    summary.update(media)
    summary["annotation_population_total"] = annotation_population
    summary["full_statistics"] = full_statistics
    summary["record_sample_count"] = 200
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    script = (HERE / "remote_collect_videopoint_media.py").read_bytes()
    archive: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="videopoint-media-", suffix=".tar", delete=False
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
                    "python3",
                    "-",
                ],
                input=script,
                stdout=output,
                stderr=subprocess.PIPE,
                check=False,
                timeout=7200,
            )
        if process.returncode:
            raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
        assert archive is not None
        safe_extract(archive, TARGET)
        merge_summary()
        print(TARGET)
        return 0
    finally:
        if archive is not None:
            archive.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
