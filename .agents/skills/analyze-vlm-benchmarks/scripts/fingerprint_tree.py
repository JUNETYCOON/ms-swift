#!/usr/bin/env python3
"""Snapshot or compare a read-only result tree without writing inside it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outside(root: Path, output: Path) -> None:
    try:
        output.resolve().relative_to(root.resolve())
    except ValueError:
        return
    raise SystemExit(f"output must stay outside source root: {output}")


def snapshot(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise SystemExit(f"source root is not a directory: {root}")
    outside(root, output)
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append({"path": relative, "kind": "symlink", "target": os.readlink(path)})
            continue
        if not path.is_file():
            continue
        stat = path.stat()
        entry = {
            "path": relative,
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if args.content_hash:
            entry["sha256"] = sha256_file(path)
        entries.append(entry)
    canonical = "\n".join(json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in entries)
    result = {
        "version": 1,
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "algorithm": "sha256 over sorted canonical path/kind/size/mtime_ns" + (
            "/content_sha256" if args.content_hash else ""
        ),
        "entry_count": len(entries),
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("label", "entry_count", "digest")}, ensure_ascii=False))
    return 0


def compare(args: argparse.Namespace) -> int:
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    before_entries = {entry["path"]: entry for entry in before.get("entries", [])}
    after_entries = {entry["path"]: entry for entry in after.get("entries", [])}
    added = sorted(after_entries.keys() - before_entries.keys())
    removed = sorted(before_entries.keys() - after_entries.keys())
    changed = sorted(
        path for path in before_entries.keys() & after_entries.keys()
        if before_entries[path] != after_entries[path]
    )
    unchanged = before.get("digest") == after.get("digest") and not (added or removed or changed)
    result = {
        "version": 1,
        "source_results_unchanged": unchanged,
        "before": {key: before.get(key) for key in ("label", "root", "entry_count", "digest", "algorithm")},
        "after": {key: after.get(key) for key in ("label", "root", "entry_count", "digest", "algorithm")},
        "added": added,
        "removed": removed,
        "changed": changed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_results_unchanged": unchanged, "added": len(added), "removed": len(removed), "changed": len(changed)}))
    return 0 if unchanged else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("root", type=Path)
    snapshot_parser.add_argument("output", type=Path)
    snapshot_parser.add_argument("--label", default="source")
    snapshot_parser.add_argument("--content-hash", action="store_true")
    snapshot_parser.set_defaults(function=snapshot)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(function=compare)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
