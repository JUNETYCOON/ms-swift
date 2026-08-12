"""Read-only schema profiler for common dataset container formats."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tarfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional


TYPE_NAMES = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}


def _reject_nonfinite(value: str) -> None:
    raise ValueError("non-finite JSON number {!r} is not allowed".format(value))


class ShapeProfiler:
    def __init__(self, max_depth: int = 8, max_examples: int = 0):
        self.max_depth = max_depth
        self.max_examples = max_examples
        self.types: DefaultDict[str, Counter] = defaultdict(Counter)
        self.examples: DefaultDict[str, List[Any]] = defaultdict(list)
        self.array_lengths: DefaultDict[str, List[int]] = defaultdict(list)

    def add(self, value: Any, path: str = "$", depth: int = 0) -> None:
        value_type = TYPE_NAMES.get(type(value), type(value).__name__)
        self.types[path][value_type] += 1
        if self.max_examples > 0 and value_type not in ("object", "array") and len(self.examples[path]) < self.max_examples:
            example = value
            if isinstance(example, str) and len(example) > 160:
                example = example[:157] + "..."
            if example not in self.examples[path]:
                self.examples[path].append(example)
        if depth >= self.max_depth:
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                self.add(child, "{}.{}".format(path, key), depth + 1)
        elif isinstance(value, list):
            self.array_lengths[path].append(len(value))
            for child in value[:64]:
                self.add(child, path + "[]", depth + 1)

    def result(self) -> Dict[str, Any]:
        fields = {}
        for path in sorted(self.types):
            entry: Dict[str, Any] = {
                "types": dict(sorted(self.types[path].items())),
                "observations": sum(self.types[path].values()),
            }
            if self.examples[path]:
                entry["examples"] = self.examples[path]
            lengths = self.array_lengths[path]
            if lengths:
                entry["array_length"] = {"min": min(lengths), "max": max(lengths)}
            fields[path] = entry
        return {"fields": fields}


def _profile_jsonl(path: Path, sample_rows: int, max_depth: int, max_examples: int) -> Dict[str, Any]:
    profiler = ShapeProfiler(max_depth=max_depth, max_examples=max_examples)
    total = 0
    sampled = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            total += 1
            try:
                value = json.loads(line, parse_constant=_reject_nonfinite)
            except (json.JSONDecodeError, ValueError) as error:
                return {
                    "kind": "jsonl",
                    "status": "invalid",
                    "error": "line {}: {}".format(line_number, error),
                    "total_records_seen": total,
                }
            if sampled < sample_rows:
                profiler.add(value)
                sampled += 1
    return {
        "kind": "jsonl",
        "status": "ok",
        "total_records": total,
        "sampled_records": sampled,
        "schema": profiler.result(),
    }


def _profile_json(
    path: Path, sample_rows: int, max_depth: int, max_json_bytes: int, max_examples: int
) -> Dict[str, Any]:
    if path.stat().st_size > max_json_bytes:
        try:
            import ijson
        except ImportError:
            return {
                "kind": "json",
                "status": "skipped",
                "error": "file exceeds max_json_bytes and optional ijson is not installed",
            }
        profiler = ShapeProfiler(max_depth=max_depth, max_examples=max_examples)
        sampled = 0
        with path.open("rb") as stream:
            first_event = next(ijson.parse(stream), None)
        if first_event is None:
            return {"kind": "json", "status": "invalid", "error": "empty JSON file"}
        root_event = first_event[1]
        if root_event == "start_array":
            iterator_name = "array"
            with path.open("rb") as stream:
                iterator = ijson.items(stream, "item")
                for value in iterator:
                    profiler.add(value)
                    sampled += 1
                    if sampled >= sample_rows:
                        break
        elif root_event == "start_map":
            iterator_name = "object_map"
            with path.open("rb") as stream:
                iterator = ijson.kvitems(stream, "")
                for _, value in iterator:
                    profiler.add(value, "$.*")
                    sampled += 1
                    if sampled >= sample_rows:
                        break
        else:
            return {
                "kind": "json",
                "status": "skipped",
                "error": "large scalar JSON roots are not profiled",
            }
        return {
            "kind": "json",
            "status": "ok",
            "root_kind": iterator_name,
            "sampled_records": sampled,
            "total_records": None,
            "schema": profiler.result(),
        }
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    profiler = ShapeProfiler(max_depth=max_depth, max_examples=max_examples)
    if isinstance(value, list):
        for item in value[:sample_rows]:
            profiler.add(item)
        total = len(value)
        sampled = min(total, sample_rows)
    else:
        profiler.add(value)
        total = 1
        sampled = 1
    return {
        "kind": "json",
        "status": "ok",
        "total_records": total,
        "sampled_records": sampled,
        "schema": profiler.result(),
    }


def _arrow_field(field: Any) -> Dict[str, Any]:
    value: Dict[str, Any] = {"name": field.name, "type": str(field.type), "nullable": bool(field.nullable)}
    if getattr(field.type, "num_fields", 0):
        value["children"] = [_arrow_field(field.type.field(index)) for index in range(field.type.num_fields)]
    return value


def _profile_parquet(path: Path) -> Dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return {"kind": "parquet", "status": "skipped", "error": "optional pyarrow is not installed"}
    value = parquet.ParquetFile(path)
    arrow_schema = value.schema_arrow
    return {
        "kind": "parquet",
        "status": "ok",
        "total_records": value.metadata.num_rows,
        "row_groups": value.metadata.num_row_groups,
        "schema": {
            "fields": [_arrow_field(field) for field in arrow_schema],
            "metadata": {
                str(key, "utf-8", "replace"): str(item, "utf-8", "replace")
                for key, item in (arrow_schema.metadata or {}).items()
            },
        },
    }


def _profile_arrow(path: Path) -> Dict[str, Any]:
    try:
        import pyarrow as pa
    except ImportError:
        return {"kind": "arrow", "status": "skipped", "error": "optional pyarrow is not installed"}
    with path.open("rb") as stream:
        try:
            reader = pa.ipc.open_file(stream)
            total = sum(reader.get_batch(index).num_rows for index in range(reader.num_record_batches))
            batches = reader.num_record_batches
        except pa.ArrowInvalid:
            stream.seek(0)
            reader = pa.ipc.open_stream(stream)
            total = 0
            batches = 0
            for batch in reader:
                total += batch.num_rows
                batches += 1
        schema = reader.schema
    return {
        "kind": "arrow",
        "status": "ok",
        "total_records": total,
        "record_batches": batches,
        "schema": {"fields": [_arrow_field(field) for field in schema]},
    }


def _profile_csv(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
        header = next(reader, [])
    return {"kind": "csv", "status": "ok", "schema": {"columns": header}}


def _profile_numpy(path: Path) -> Dict[str, Any]:
    try:
        import numpy as np
    except ImportError:
        return {"kind": "numpy", "status": "skipped", "error": "optional numpy is not installed"}
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if hasattr(value, "files"):
        arrays = {key: {"dtype": str(value[key].dtype), "shape": list(value[key].shape)} for key in value.files}
        value.close()
    else:
        arrays = {"array": {"dtype": str(value.dtype), "shape": list(value.shape)}}
    return {"kind": "numpy", "status": "ok", "schema": {"arrays": arrays}}


def _profile_hdf5(path: Path) -> Dict[str, Any]:
    try:
        import h5py
    except ImportError:
        return {"kind": "hdf5", "status": "skipped", "error": "optional h5py is not installed"}
    datasets: Dict[str, Any] = {}
    with h5py.File(path, "r") as value:
        def visit(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset):
                datasets[name] = {"dtype": str(item.dtype), "shape": list(item.shape)}

        value.visititems(visit)
    return {"kind": "hdf5", "status": "ok", "schema": {"datasets": datasets}}


def _profile_archive(path: Path, max_members: int) -> Dict[str, Any]:
    suffixes: Counter = Counter()
    members = 0
    truncated = False
    if zipfile.is_zipfile(path):
        kind = "zip"
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                members += 1
                suffixes[Path(item.filename).suffix.lower() or "<none>"] += 1
                if members >= max_members:
                    truncated = len(archive.infolist()) > members
                    break
    else:
        kind = "tar"
        with tarfile.open(path, "r:*") as archive:
            for item in archive:
                if not item.isfile():
                    continue
                members += 1
                suffixes[Path(item.name).suffix.lower() or "<none>"] += 1
                if members >= max_members:
                    truncated = True
                    break
    return {
        "kind": kind,
        "status": "ok",
        "members_scanned": members,
        "truncated": truncated,
        "member_suffixes": dict(sorted(suffixes.items())),
    }


def profile_file(
    path: Path,
    sample_rows: int = 100,
    max_depth: int = 8,
    max_json_bytes: int = 256 * 1024 * 1024,
    max_archive_members: int = 100000,
    include_examples: bool = False,
) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    result: Dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}
    lower = path.name.lower()
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            result.update(_profile_jsonl(path, sample_rows, max_depth, 3 if include_examples else 0))
        elif suffix == ".json":
            result.update(
                _profile_json(path, sample_rows, max_depth, max_json_bytes, 3 if include_examples else 0)
            )
        elif suffix == ".parquet":
            result.update(_profile_parquet(path))
        elif suffix in (".arrow", ".feather"):
            result.update(_profile_arrow(path))
        elif suffix in (".csv", ".tsv"):
            result.update(_profile_csv(path))
        elif suffix in (".npy", ".npz"):
            result.update(_profile_numpy(path))
        elif suffix in (".h5", ".hdf5"):
            result.update(_profile_hdf5(path))
        elif suffix == ".zip" or lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            result.update(_profile_archive(path, max_archive_members))
        else:
            result.update({"kind": suffix.lstrip(".") or "unknown", "status": "inventory_only"})
    except Exception as error:
        result.update({"status": "error", "error": "{}: {}".format(type(error).__name__, error)})
    return result


def profile_path(
    path: Path,
    sample_rows: int = 100,
    max_depth: int = 8,
    max_files: int = 10000,
    include_examples: bool = False,
    relative_paths: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_file():
        report = profile_file(
            path, sample_rows=sample_rows, max_depth=max_depth, include_examples=include_examples
        )
        if relative_paths:
            report["path"] = path.name
        return {"root": path.name if relative_paths else str(path), "files": [report]}
    if not path.is_dir():
        raise FileNotFoundError(path)
    structured_suffixes = {
        ".json", ".jsonl", ".parquet", ".arrow", ".feather", ".csv", ".tsv",
        ".npy", ".npz", ".h5", ".hdf5", ".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz",
    }
    total_files = 0
    structured_files = 0
    suffix_counts: Counter = Counter()
    selected: List[Path] = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        total_files += 1
        suffix = item.suffix.lower() or "<none>"
        suffix_counts[suffix] += 1
        lower = item.name.lower()
        is_structured = suffix in structured_suffixes or lower.endswith(
            (".tar.gz", ".tar.bz2", ".tar.xz")
        )
        if is_structured:
            structured_files += 1
            if len(selected) < max_files:
                selected.append(item)
    selected.sort()
    truncated = structured_files > max_files
    def run(item: Path) -> Dict[str, Any]:
        result = profile_file(
            item,
            sample_rows=sample_rows,
            max_depth=max_depth,
            include_examples=include_examples,
        )
        if relative_paths:
            result["path"] = item.relative_to(path).as_posix()
        return result
    if workers <= 1:
        file_reports = [run(item) for item in selected]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            file_reports = list(executor.map(run, selected))
    families: Dict[str, Dict[str, Any]] = {}
    for report in file_reports:
        signature_payload = {"kind": report.get("kind"), "schema": report.get("schema")}
        fingerprint = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        family = families.setdefault(
            fingerprint,
            {"fingerprint": fingerprint, "kind": report.get("kind"), "files": 0, "records": 0, "examples": []},
        )
        family["files"] += 1
        if isinstance(report.get("total_records"), int):
            family["records"] += report["total_records"]
        if len(family["examples"]) < 5:
            family["examples"].append(report["path"])
    return {
        "root": "." if relative_paths else str(path),
        "total_files": total_files,
        "structured_files": structured_files,
        "profiled_files": len(selected),
        "truncated": truncated,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "schema_families": sorted(families.values(), key=lambda value: (str(value["kind"]), value["fingerprint"])),
        "files": file_reports,
    }
