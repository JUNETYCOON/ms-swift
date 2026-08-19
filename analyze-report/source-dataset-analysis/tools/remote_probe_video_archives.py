#!/usr/bin/env python3
"""Read-only bounded probe for source video archive member naming."""

from __future__ import annotations

import json
import importlib.util
import shutil
import tarfile
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/luojunkun/stage1/dataset")


def first_tar_members(path: Path, limit: int = 40) -> dict:
    members = []
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            members.append({"name": member.name, "size": member.size})
            if len(members) >= limit:
                break
    return {
        "path": str(path),
        "probe_scope": f"first {limit} regular members from sequential gzip stream",
        "sample_members": members,
    }


def zip_central_directory(path: Path, limit: int = 40) -> dict:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        suffix_counts = Counter(Path(item.filename).suffix.lower() for item in members)
        samples = [
            {
                "name": item.filename,
                "size": item.file_size,
                "compressed_size": item.compress_size,
                "crc32": f"{item.CRC:08x}",
            }
            for item in members[:limit]
        ]
    return {
        "path": str(path),
        "probe_scope": "full ZIP central directory",
        "member_count": len(members),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "sample_members": samples,
    }


def main() -> int:
    point_root = ROOT / "Molmo2-VideoPoint" / "generated_videos"
    track_root = ROOT / "Molmo2-VideoTrack"
    payload = {
        "source_root": str(ROOT),
        "runtime": {
            "python_modules": {
                name: importlib.util.find_spec(name) is not None
                for name in ("av", "cv2", "decord", "imageio", "imageio_ffmpeg")
            },
            "executables": {
                name: shutil.which(name) for name in ("ffmpeg", "ffprobe")
            },
        },
        "video_point": [
            first_tar_members(path) for path in sorted(point_root.glob("*.tar.gz"))
        ],
        "video_track": [
            zip_central_directory(
                track_root / "data" / "mose" / "vedio" / "MOSE_release.zip"
            ),
            zip_central_directory(
                track_root
                / "data"
                / "mosev2"
                / "vedio"
                / "sample_submission_mosev2_valid.zip"
            ),
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
