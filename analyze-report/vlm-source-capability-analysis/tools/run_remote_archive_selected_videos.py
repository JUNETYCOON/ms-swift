#!/usr/bin/env python3
"""Archive the 200 selected RoboVQA and VideoPoint source videos locally."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUB = ROOT / "sub-dataset"
ARTIFACTS = ROOT / "artifacts"
REMOTE_HOST = "8.146.226.25"
DATASETS = ["RoboVQA", "Molmo2-VideoPoint"]
SOURCE_ROOTS = {
    "RoboVQA": PurePosixPath("/mnt/luojunkun/stage1/dataset/robovqa"),
    "Molmo2-VideoPoint": PurePosixPath("/mnt/luojunkun/stage1/dataset/Molmo2-VideoPoint"),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, target: Path) -> None:
    target_root = target.resolve()
    with tarfile.open(archive, "r:") as stream:
        members = stream.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe tar member: {member.name}")
            destination = target.joinpath(*relative.parts).resolve()
            if target_root != destination and target_root not in destination.parents:
                raise RuntimeError(f"tar member escapes target: {member.name}")
        stream.extractall(target, members=members, filter="data")


def build_selections() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    selections = []
    manifests = {}
    for dataset in DATASETS:
        manifest_path = SUB / dataset / "sampling-manifest.jsonl"
        rows = load_jsonl(manifest_path)
        if len(rows) != 200:
            raise RuntimeError(f"{dataset}: expected 200 selected videos, got {len(rows)}")
        manifests[dataset] = rows
        for row in rows:
            source_archive = row.get("source_archive") if dataset == "Molmo2-VideoPoint" else row.get("archive")
            source_member = row.get("source_member") if dataset == "Molmo2-VideoPoint" else row.get("member")
            if not source_archive or not source_member:
                raise RuntimeError(f"{dataset}/{row.get('sample_id')}: source archive/member unavailable")
            relative_video = f"source-video/{row['sample_id']}.mp4"
            selections.append({
                "dataset": dataset,
                "sample_id": row["sample_id"],
                "source_archive": str(SOURCE_ROOTS[dataset] / PurePosixPath(str(source_archive))),
                "source_member": str(source_member),
                "expected_sha256": row["source_video_sha256"],
                "expected_bytes": int(row["source_video_byte_length"]),
                "output_path": f"sub-dataset/{dataset}/{relative_video}",
            })
    return selections, manifests


def local_archive_complete(manifests: dict[str, list[dict[str, Any]]]) -> bool:
    for dataset, rows in manifests.items():
        for row in rows:
            relative = row.get("source_video_archive_path")
            if not relative:
                return False
            path = SUB / dataset / str(relative)
            if not path.is_file() or path.stat().st_size != int(row["source_video_byte_length"]):
                return False
            if sha256(path) != row["source_video_sha256"]:
                return False
    return True


def update_manifests(manifests: dict[str, list[dict[str, Any]]]) -> None:
    for dataset, rows in manifests.items():
        archived_bytes = 0
        for row in rows:
            relative = f"source-video/{row['sample_id']}.mp4"
            path = SUB / dataset / relative
            if not path.is_file():
                raise RuntimeError(f"missing extracted source video: {path}")
            if path.stat().st_size != int(row["source_video_byte_length"]) or sha256(path) != row["source_video_sha256"]:
                raise RuntimeError(f"source video identity mismatch after extraction: {path}")
            row["source_video_archive_path"] = relative
            row["source_video_archived"] = True
            archived_bytes += path.stat().st_size
        atomic_jsonl(SUB / dataset / "sampling-manifest.jsonl", rows)
        summary_path = SUB / dataset / "sampling-summary.json"
        summary = load_json(summary_path)
        summary["source_video_bytes_archived"] = True
        summary["source_video_archive_count"] = len(rows)
        summary["source_video_archive_bytes"] = archived_bytes
        summary["preview_only"] = False
        atomic_json(summary_path, summary)


def main() -> int:
    selections, manifests = build_selections()
    if local_archive_complete(manifests):
        print("all 400 selected source videos already archived and verified")
        return 0

    remote_script = (HERE / "remote_archive_selected_videos.py").read_bytes()
    future_import = b"from __future__ import annotations\n"
    if future_import not in remote_script:
        raise RuntimeError("remote collector future import marker unavailable")
    injected = remote_script.replace(
        future_import,
        future_import + ("SELECTIONS = " + repr(selections) + "\n").encode("utf-8"),
        1,
    )
    archive: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="selected-source-videos-", suffix=".tar", delete=False) as output:
            archive = Path(output.name)
            process = subprocess.run(
                [
                    "ssh", "-o", "BatchMode=yes", REMOTE_HOST,
                    "env", "PYTHONDONTWRITEBYTECODE=1", "python3", "-",
                ],
                input=injected,
                stdout=output,
                stderr=None,
                check=False,
                timeout=14_400,
            )
        if process.returncode:
            raise RuntimeError(f"remote selected-video collector failed with exit code {process.returncode}")
        assert archive is not None
        safe_extract(archive, ROOT)
        summary = load_json(ARTIFACTS / "selected-video-archive-summary.json")
        if summary["archived_count"] != 400 or summary["failure_count"] or not summary["source_archives_unchanged"]:
            raise RuntimeError("remote selected-video archive summary failed validation")
        update_manifests(manifests)
        print(json.dumps({
            "status": "archived",
            "videos": summary["archived_count"],
            "bytes": summary["archived_bytes"],
            "source_archives_unchanged": summary["source_archives_unchanged"],
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        if archive is not None:
            archive.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
