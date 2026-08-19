#!/usr/bin/env python3
"""Shared deterministic JSONL splitting with group-level leakage checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TextIO
from urllib.parse import urlsplit, urlunsplit


GroupResolver = Callable[[dict[str, Any], Path], str]
StratumResolver = Callable[[dict[str, Any]], str]
MEDIA_KEYS = {"images", "videos", "audios"}
URL_SCHEMES = {"http", "https"}


def load_record(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
    return value


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in URL_SCHEMES or not parsed.netloc:
        raise ValueError(f"invalid media URL: {value!r}")
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    default_port = (parsed.scheme.lower(), port) in {("http", 80), ("https", 443)}
    netloc = hostname if not port or default_port else f"{hostname}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", "", ""))


def _flatten_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten_values(item))
        return result
    return [value]


def record_values(
    record: Mapping[str, Any], key: str, jsonl_path: Path
) -> list[str]:
    if key == "__kind__":
        objects = record.get("objects")
        return ["grounding" if isinstance(objects, dict) and objects.get("bbox") else "qa"]
    values: list[str] = []
    for item in _flatten_values(record.get(key)):
        text = str(item).strip()
        if not text:
            continue
        if key in MEDIA_KEYS:
            if urlsplit(text).scheme.lower() in URL_SCHEMES:
                text = _canonical_url(text)
            else:
                path = Path(text).expanduser()
                if not path.is_absolute():
                    path = jsonl_path.parent / path
                text = os.path.normcase(os.path.abspath(os.path.normpath(str(path))))
        values.append(text)
    return sorted(set(values))


def record_group(
    record: Mapping[str, Any],
    key: str,
    jsonl_path: Path,
    group_pattern: str | re.Pattern[str] | None = None,
) -> str:
    values = record_values(record, key, jsonl_path)
    if group_pattern is not None:
        pattern = re.compile(group_pattern) if isinstance(group_pattern, str) else group_pattern
        transformed: list[str] = []
        for value in values:
            match = pattern.search(value)
            if not match:
                raise ValueError(f"value does not match grouping pattern {pattern.pattern!r}: {value!r}")
            transformed.append(match.group(1) if match.lastindex else match.group(0))
        values = sorted(set(transformed))
    if not values:
        raise ValueError(f"record has no non-empty {key!r} values")
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def stable_is_eval(group: str, eval_ratio: float, seed: int) -> bool:
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be in the range (0, 1)")
    threshold = int(eval_ratio * (1 << 64))
    digest = hashlib.blake2b(f"{seed}\0{group}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") < threshold


class FileDigestCache:
    def __init__(self, algorithm: str = "sha256") -> None:
        hashlib.new(algorithm)
        self.algorithm = algorithm
        self._values: dict[tuple[str, int, int], str] = {}

    def digest(self, path_text: str) -> str:
        if urlsplit(path_text).scheme.lower() in URL_SCHEMES:
            raise ValueError("content hashes require local media files")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"media file does not exist: {path}")
        stat = path.stat()
        key = (os.path.normcase(str(path.resolve())), stat.st_size, stat.st_mtime_ns)
        cached = self._values.get(key)
        if cached is not None:
            return cached
        hasher = hashlib.new(self.algorithm)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
        result = hasher.hexdigest()
        self._values[key] = result
        return result


def media_hash_group_resolver(
    key: str = "images", algorithm: str = "sha256"
) -> GroupResolver:
    cache = FileDigestCache(algorithm)

    def resolve(record: dict[str, Any], jsonl_path: Path) -> str:
        values = record_values(record, key, jsonl_path)
        if not values:
            raise ValueError(f"record has no non-empty {key!r} values")
        digests = sorted({f"{algorithm}:{cache.digest(value)}" for value in values})
        return json.dumps(digests, separators=(",", ":"))

    return resolve


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _open_temporary(path: Path) -> TextIO:
    return _temporary_sibling(path).open("w", encoding="utf-8", newline="\n")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_groups(paths: Iterable[Path], resolver: GroupResolver) -> tuple[set[str], int]:
    groups: set[str] = set()
    rows = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                rows += 1
                record = load_record(line, path, line_number)
                try:
                    groups.add(resolver(record, path))
                except (FileNotFoundError, OSError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: cannot derive group: {error}") from error
    return groups, rows


def split_jsonl(
    *,
    input_paths: Sequence[Path],
    train_output: Path,
    eval_output: Path,
    report_output: Path,
    group_resolver: GroupResolver,
    group_key_description: str,
    eval_ratio: float = 0.1,
    seed: int = 42,
    reserved_eval_paths: Sequence[Path] = (),
    reserve_only: bool = False,
    stratum_resolver: StratumResolver | None = None,
    progress_every: int = 50_000,
    overwrite: bool = False,
    report_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = [path.expanduser().resolve() for path in input_paths]
    reserved = [path.expanduser().resolve() for path in reserved_eval_paths]
    train_output = train_output.expanduser().resolve()
    eval_output = eval_output.expanduser().resolve()
    report_output = report_output.expanduser().resolve()
    if not inputs:
        raise ValueError("at least one input JSONL is required")
    for path in [*inputs, *reserved]:
        if not path.is_file():
            raise FileNotFoundError(f"input JSONL does not exist: {path}")
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be in the range (0, 1)")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    outputs = (train_output, eval_output, report_output)
    if len(set(outputs)) != len(outputs):
        raise ValueError("train, eval, and report outputs must be different")
    if any(output in inputs for output in outputs):
        raise ValueError("outputs must not overwrite a primary input JSONL")
    for output in outputs:
        if output.exists() and not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")

    forced_eval_groups, reserved_rows = _load_groups(reserved, group_resolver)
    if reserve_only and not forced_eval_groups:
        raise ValueError("reserve_only requires at least one reserved eval group")

    train_stream = _open_temporary(train_output)
    eval_stream = _open_temporary(eval_output)
    report_temporary = _temporary_sibling(report_output)
    stats: Counter[str] = Counter()
    split_groups: dict[str, str] = {}
    group_first_source: dict[str, int] = {}
    source_stats: dict[str, Counter[str]] = {str(path): Counter() for path in inputs}
    stratum_stats: dict[str, Counter[str]] = defaultdict(Counter)
    cross_source_groups: set[str] = set()
    try:
        for source_index, path in enumerate(inputs):
            with path.open("r", encoding="utf-8-sig") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    stats["input_rows"] += 1
                    source_stats[str(path)]["input_rows"] += 1
                    record = load_record(line, path, line_number)
                    try:
                        group = group_resolver(record, path)
                    except (FileNotFoundError, OSError, ValueError) as error:
                        raise ValueError(f"{path}:{line_number}: cannot derive group: {error}") from error
                    first_source = group_first_source.setdefault(group, source_index)
                    if first_source != source_index:
                        cross_source_groups.add(group)
                    if group in forced_eval_groups:
                        split = "eval"
                        stats["forced_eval_rows"] += 1
                    elif reserve_only:
                        split = "train"
                    else:
                        split = "eval" if stable_is_eval(group, eval_ratio, seed) else "train"
                    previous = split_groups.setdefault(group, split)
                    if previous != split:
                        raise AssertionError(f"group received conflicting split assignments: {group}")
                    output_line = line.rstrip("\r\n") + "\n"
                    (eval_stream if split == "eval" else train_stream).write(output_line)
                    stats[f"{split}_rows"] += 1
                    source_stats[str(path)][f"{split}_rows"] += 1
                    if stratum_resolver is not None:
                        stratum = str(stratum_resolver(record) or "unknown")
                        stratum_stats[stratum][split] += 1
                    if progress_every and stats["input_rows"] % progress_every == 0:
                        print(
                            f"[split] rows={stats['input_rows']:,} train={stats['train_rows']:,} "
                            f"eval={stats['eval_rows']:,} groups={len(split_groups):,}",
                            flush=True,
                        )

        missing_reserved = forced_eval_groups - set(split_groups)
        if missing_reserved:
            raise ValueError(
                f"{len(missing_reserved):,} reserved eval groups were absent from primary inputs"
            )
        if not stats["train_rows"] or not stats["eval_rows"]:
            raise ValueError("split must contain at least one train and one eval row")
        train_groups = {group for group, split in split_groups.items() if split == "train"}
        eval_groups = {group for group, split in split_groups.items() if split == "eval"}
        overlap = train_groups & eval_groups
        if overlap:
            raise AssertionError(f"group leakage detected: {len(overlap):,} groups")
        if reserve_only and eval_groups != forced_eval_groups:
            raise AssertionError("preserved eval group set changed")

        train_temporary = Path(train_stream.name)
        eval_temporary = Path(eval_stream.name)
        train_stream.close()
        eval_stream.close()
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "input_jsonl": [str(path) for path in inputs],
            "train_jsonl": str(train_output),
            "eval_jsonl": str(eval_output),
            "group_key": group_key_description,
            "seed": seed,
            "requested_eval_ratio": eval_ratio,
            "actual_eval_ratio": stats["eval_rows"] / stats["input_rows"],
            "reserved_eval": {
                "paths": [str(path) for path in reserved],
                "rows": reserved_rows,
                "groups": len(forced_eval_groups),
                "reserve_only": reserve_only,
                "missing_groups": 0,
            },
            "rows": {
                "input": stats["input_rows"],
                "train": stats["train_rows"],
                "eval": stats["eval_rows"],
                "forced_eval": stats["forced_eval_rows"],
            },
            "groups": {
                "total": len(split_groups),
                "train": len(train_groups),
                "eval": len(eval_groups),
                "train_eval_overlap": 0,
                "present_in_multiple_source_files": len(cross_source_groups),
            },
            "source_rows": {
                path: dict(sorted(counts.items())) for path, counts in source_stats.items()
            },
            "strata": {
                key: dict(sorted(counts.items())) for key, counts in sorted(stratum_stats.items())
            },
            "output_sha256": {
                "train": _file_sha256(train_temporary),
                "eval": _file_sha256(eval_temporary),
            },
        }
        if report_extra:
            report["dataset_policy"] = dict(report_extra)
        with report_temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(train_temporary, train_output)
        os.replace(eval_temporary, eval_output)
        os.replace(report_temporary, report_output)
        return report
    except BaseException:
        for stream in (train_stream, eval_stream):
            if not stream.closed:
                stream.close()
        for path in (Path(train_stream.name), Path(eval_stream.name), report_temporary):
            if path.exists():
                path.unlink()
        raise
