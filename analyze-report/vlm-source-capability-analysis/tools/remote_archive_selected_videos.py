#!/usr/bin/env python3
"""Stream selected source videos from read-only TAR archives to stdout."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


ALLOWED_ROOTS = [
    Path("/mnt/luojunkun/stage1/dataset/Molmo2-VideoPoint"),
    Path("/mnt/luojunkun/stage1/dataset/robovqa"),
]


try:
    SELECTIONS
except NameError as exc:  # pragma: no cover - injected by the local runner
    raise RuntimeError("SELECTIONS must be injected by the local runner") from exc


def normalized_member(name: str) -> str:
    path = PurePosixPath(name)
    return PurePosixPath(*[part for part in path.parts if part not in {".", ""}]).as_posix()


def validate_archive(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or not any(path == root or root in path.parents for root in ALLOWED_ROOTS):
        raise RuntimeError(f"source archive outside allowlist: {path}")
    if not path.is_file() or not path.name.endswith(".tar.gz"):
        raise RuntimeError(f"source archive unavailable or unsupported: {path}")
    return path


def validate_member(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe source member: {value}")
    return value


def validate_output(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".mp4":
        raise RuntimeError(f"unsafe output member: {value}")
    return path.as_posix()


def add_bytes(output: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    output.addfile(info, io.BytesIO(payload))


def archive_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main() -> int:
    selections = []
    for raw in SELECTIONS:
        item = dict(raw)
        item["source_archive"] = str(validate_archive(item["source_archive"]))
        item["source_member"] = validate_member(str(item["source_member"]))
        item["output_path"] = validate_output(str(item["output_path"]))
        selections.append(item)
    if len(selections) != 400:
        raise RuntimeError(f"expected 400 selected videos, got {len(selections)}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selections:
        grouped[item["source_archive"]].append(item)
    before = {name: archive_stat(Path(name)) for name in grouped}
    results = []
    failures = []

    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as output:
        for archive_name in sorted(grouped):
            archive_path = Path(archive_name)
            selected = grouped[archive_name]
            pending = {normalized_member(item["source_member"]): item for item in selected}
            if len(pending) != len(selected):
                raise RuntimeError(f"duplicate normalized members in {archive_name}")
            print(
                f"scanning {archive_path.name}: {len(selected)} selected member(s)",
                file=sys.stderr,
                flush=True,
            )
            with tarfile.open(archive_path, mode="r|gz") as source:
                for member in source:
                    if not member.isfile():
                        continue
                    key = normalized_member(member.name)
                    item = pending.get(key)
                    if item is None:
                        continue
                    stream = source.extractfile(member)
                    if stream is None:
                        failures.append({**item, "reason": "member_stream_unavailable"})
                        pending.pop(key, None)
                        continue
                    payload = stream.read()
                    digest = hashlib.sha256(payload).hexdigest()
                    expected_digest = str(item["expected_sha256"])
                    expected_bytes = int(item["expected_bytes"])
                    if digest != expected_digest or len(payload) != expected_bytes:
                        failures.append({
                            **item,
                            "reason": "source_video_identity_mismatch",
                            "actual_sha256": digest,
                            "actual_bytes": len(payload),
                        })
                    else:
                        add_bytes(output, item["output_path"], payload)
                        results.append({
                            "dataset": item["dataset"],
                            "sample_id": item["sample_id"],
                            "source_archive": item["source_archive"],
                            "source_member": item["source_member"],
                            "output_path": item["output_path"],
                            "sha256": digest,
                            "bytes": len(payload),
                        })
                    pending.pop(key, None)
                    if not pending:
                        break
            for item in pending.values():
                failures.append({**item, "reason": "source_member_not_found"})

        after = {name: archive_stat(Path(name)) for name in grouped}
        summary = {
            "version": 1,
            "source_only": True,
            "selection_count": len(selections),
            "archived_count": len(results),
            "archived_bytes": sum(item["bytes"] for item in results),
            "failure_count": len(failures),
            "failures": failures,
            "source_archive_stats_before": before,
            "source_archive_stats_after": after,
            "source_archives_unchanged": before == after,
            "results": sorted(results, key=lambda item: (item["dataset"], item["sample_id"])),
        }
        add_bytes(
            output,
            "artifacts/selected-video-archive-summary.json",
            (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    if failures or before != after:
        print(json.dumps({"failures": failures, "source_archives_unchanged": before == after}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
