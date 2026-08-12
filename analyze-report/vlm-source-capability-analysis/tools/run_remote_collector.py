#!/usr/bin/env python3
"""Run a source collector over SSH and safely extract its tar stream locally."""

from __future__ import annotations

import argparse
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
REPORT_ROOT = HERE.parent
REMOTE_HOST = "8.146.226.25"
REMOTE_TIMEOUT_SECONDS = 2100
COLLECTORS = {
    "PixMo-Cap": "remote_collect_images.py",
    "PixMo-Points": "remote_collect_images.py",
    "SpatialVLM": "remote_collect_images.py",
    "Molmo2-VideoCapQA": "remote_collect_molmo_annotations.py",
    "Molmo2-VideoPoint": "remote_collect_molmo_annotations.py",
    "Molmo2-VideoSubtitleQA": "remote_collect_molmo_annotations.py",
    "Molmo2-VideoTrack": "remote_collect_molmo_annotations.py",
}


def safe_extract(archive: Path, target: Path) -> None:
    with tarfile.open(archive, "r:") as tar:
        members = tar.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe tar member: {member.name}")
        target.mkdir(parents=True, exist_ok=True)
        tar.extractall(target, members=members, filter="data")


def run(dataset: str) -> None:
    script = (HERE / COLLECTORS[dataset]).read_bytes()
    target = REPORT_ROOT / "sub-dataset" / dataset
    archive: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="source-collector-", suffix=".tar", delete=False
        ) as stream:
            archive = Path(stream.name)
            try:
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
                    timeout=REMOTE_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"remote collector timed out after {REMOTE_TIMEOUT_SECONDS}s: {dataset}"
                ) from exc
        if process.returncode:
            raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
        assert archive is not None
        safe_extract(archive, target)
    finally:
        if archive is not None:
            archive.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=tuple(COLLECTORS))
    args = parser.parse_args()
    for dataset in args.datasets:
        print(f"collecting {dataset}", flush=True)
        run(dataset)
        print(f"collected {dataset}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
