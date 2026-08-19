#!/usr/bin/env python3
"""Read-only source probe for dataset files on the SSH host.

The script writes JSON to stdout and never writes below the source root. It is
designed to be streamed to ``python3 -`` over SSH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SOURCE_ROOT = Path("/mnt/luojunkun/stage1/dataset")
MAX_TEXT = 2_000
MAX_COLLECTION = 12


def retry(operation, attempts: int = 10):
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except (FileNotFoundError, OSError) as exc:
            error = exc
            time.sleep(min(0.25 * (attempt + 1), 2.0))
    assert error is not None
    raise error


def clean_value(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "<max-depth>"
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "<truncated>"
    if isinstance(value, dict):
        return {str(key): clean_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [clean_value(item, depth + 1) for item in value[:MAX_COLLECTION]]
        if len(value) > MAX_COLLECTION:
            items.append({"truncated_items": len(value) - MAX_COLLECTION})
        return items
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def split_hint(relative: str) -> str:
    lower = relative.lower()
    for split in ("validation", "testdev", "challenge", "submission", "train", "test", "val"):
        if split in lower:
            return split
    return "unspecified"


def list_files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in {".cache", "__pycache__"})
        current_path = Path(current)
        for name in sorted(files):
            path = current_path / name
            stat = retry(path.stat)
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "suffix": path.suffix.lower(),
                }
            )
    return rows


def fingerprint(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(files, key=lambda item: item["relative_path"]):
        digest.update(
            f"{row['relative_path']}\t{row['size']}\t{row['mtime_ns']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def inspect_parquet(root: Path, relative: str, include_sample: bool) -> dict[str, Any]:
    path = root / relative
    parquet = retry(lambda: pq.ParquetFile(path))
    schema = parquet.schema_arrow
    sample_rows: list[Any] = []
    if include_sample and parquet.metadata.num_rows:
        table = retry(lambda: parquet.read_row_group(0))
        sample_rows = [clean_value(row) for row in table.slice(0, 3).to_pylist()]
    result = {
        "relative_path": relative,
        "split_hint": split_hint(relative),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "created_by": parquet.metadata.created_by,
        "schema": str(schema),
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
        "sample_rows": sample_rows,
    }
    parquet.close()
    return result


def read_text_metadata(root: Path, files: list[dict[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    accepted = {"readme.md", "dataset_infos.json", "dataset_info.json", ".gitattributes"}
    for row in files:
        relative = row["relative_path"]
        path = root / relative
        if path.name.lower() not in accepted or row["size"] > 2_000_000:
            continue
        try:
            output[relative] = retry(lambda: path.read_text(encoding="utf-8"))[:50_000]
        except UnicodeDecodeError:
            output[relative] = "<non-UTF-8 metadata>"
    return output


def inspect_dataset(name: str, sample_count: int, workers: int) -> dict[str, Any]:
    root = SOURCE_ROOT / name
    if not root.is_dir():
        return {"dataset": name, "source_root": str(root), "status": "missing"}
    files = list_files(root)
    parquet_files = [row for row in files if row["suffix"] == ".parquet"]
    tasks = [
        (root, row["relative_path"], index < sample_count)
        for index, row in enumerate(parquet_files)
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        parquet_reports = list(executor.map(lambda args: inspect_parquet(*args), tasks))
    suffix_counts = Counter(row["suffix"] or "<none>" for row in files)
    return {
        "dataset": name,
        "source_root": str(root),
        "status": "ok",
        "file_count": len(files),
        "total_file_bytes": sum(row["size"] for row in files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "files": files,
        "fingerprint_algorithm": "sha256 over sorted relative path, size, and mtime_ns",
        "fingerprint": fingerprint(files),
        "parquet_file_count": len(parquet_reports),
        "parquet_row_total_all_files": sum(row["rows"] for row in parquet_reports),
        "parquet": parquet_reports,
        "source_metadata": read_text_metadata(root, files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    payload = {
        "version": 1,
        "source_root": str(SOURCE_ROOT),
        "python": sys.version,
        "datasets": [
            inspect_dataset(name, sample_count=args.sample_count, workers=args.workers)
            for name in args.datasets
        ],
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
