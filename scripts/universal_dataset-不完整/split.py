"""Deterministic group-level splitting with shared-asset leakage checks."""

from __future__ import annotations

import hashlib
import ntpath
import json
import os
import posixpath
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, TextIO, Tuple
from urllib.parse import urlsplit, urlunsplit

from .io import iter_jsonl, json_line


WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class SplitSummary:
    total_records: int
    train_records: int
    val_records: int
    total_groups: int
    train_groups: int
    val_groups: int
    total_components: int
    train_components: int
    val_components: int
    merged_components: int
    merged_groups: int
    rewritten_records: int
    unique_assets: int
    leakage_overlap: Optional[int]
    seed: int
    val_ratio: float

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0

    def find(self, value: str) -> str:
        self.add(value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        left_rank = self.rank[left_root]
        right_rank = self.rank[right_root]
        if left_rank < right_rank:
            left_root, right_root = right_root, left_root
            left_rank, right_rank = right_rank, left_rank
        self.parent[right_root] = left_root
        if left_rank == right_rank:
            self.rank[left_root] += 1


def deterministic_split(group_id: str, seed: int, val_ratio: float) -> str:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")
    digest = hashlib.sha256("{}\0{}".format(seed, group_id).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if value < val_ratio else "train"


def canonical_asset_identity(asset: Mapping[str, Any], input_path: Path) -> str:
    sha256 = asset.get("sha256")
    if isinstance(sha256, str) and sha256:
        if re.fullmatch(r"[A-Fa-f0-9]{64}", sha256) is None:
            raise ValueError("Asset {!r} has an invalid sha256 value".format(asset.get("id")))
        return "sha256:" + sha256.lower()
    uri = asset.get("uri")
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("Asset {!r} has neither sha256 nor uri".format(asset.get("id")))
    uri = uri.strip()
    if WINDOWS_DRIVE_PATH.match(uri):
        return "windows-path:" + ntpath.normcase(ntpath.normpath(uri))
    if uri.startswith("\\\\"):
        return "windows-path:" + ntpath.normcase(ntpath.normpath(uri))
    parsed = urlsplit(uri)
    if parsed.scheme and parsed.scheme.lower() != "file":
        canonical = urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, "")
        )
        return "uri:" + canonical
    local_value = parsed.path if parsed.scheme.lower() == "file" else uri
    if re.match(r"^/[A-Za-z]:/", local_value):
        return "windows-path:" + ntpath.normcase(ntpath.normpath(local_value[1:]))
    if local_value.startswith("/"):
        return "posix-path:" + posixpath.normpath(local_value)
    path = Path(local_value).expanduser()
    if not path.is_absolute():
        path = input_path.parent / path
    return "path:" + os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _temporary_stream(path: Path) -> Tuple[Path, TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent), text=True)
    return Path(name), os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")


def _commit_pair(
    train_temporary: Path,
    train_output: Path,
    val_temporary: Path,
    val_output: Path,
    overwrite: bool,
) -> None:
    backups: Dict[Path, Path] = {}
    committed: Set[Path] = set()
    try:
        for output in (train_output, val_output):
            if output.exists() and not overwrite:
                raise FileExistsError("Output appeared during split: {}".format(output))
            if output.exists():
                backup, stream = _temporary_stream(output)
                stream.close()
                backup.unlink()
                os.replace(str(output), str(backup))
                backups[output] = backup
        for temporary, output in ((train_temporary, train_output), (val_temporary, val_output)):
            os.replace(str(temporary), str(output))
            committed.add(output)
    except BaseException:
        for output in committed:
            try:
                output.unlink()
            except FileNotFoundError:
                pass
        for output, backup in backups.items():
            if backup.exists():
                os.replace(str(backup), str(output))
        raise
    else:
        for backup in backups.values():
            try:
                backup.unlink()
            except FileNotFoundError:
                pass


def split_jsonl(
    input_path: Path,
    train_output: Path,
    val_output: Path,
    val_ratio: float = 0.02,
    seed: int = 42,
    overwrite: bool = False,
    audit_assets: bool = True,
) -> SplitSummary:
    input_path = input_path.expanduser().resolve()
    train_output = train_output.expanduser().resolve()
    val_output = val_output.expanduser().resolve()
    if train_output == val_output:
        raise ValueError("train_output and val_output must differ")
    if input_path in (train_output, val_output):
        raise ValueError("split outputs must not overwrite the input")
    for path in (train_output, val_output):
        if path.exists() and not overwrite:
            raise FileExistsError("Output already exists: {}".format(path))

    groups = _DisjointSet()
    asset_groups: Dict[str, str] = {}
    total_records = 0
    for line_number, record in iter_jsonl(input_path):
        group_id = record.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("{}:{} has no non-empty group_id".format(input_path, line_number))
        groups.add(group_id)
        if audit_assets:
            for asset in record.get("assets") or []:
                if not isinstance(asset, Mapping):
                    continue
                identity = canonical_asset_identity(asset, input_path)
                previous = asset_groups.setdefault(identity, group_id)
                if previous != group_id:
                    groups.union(previous, group_id)
        total_records += 1
    if total_records == 0:
        raise ValueError("Input JSONL is empty: {}".format(input_path))

    component_members: Dict[str, List[str]] = {}
    for group_id in groups.parent:
        component_members.setdefault(groups.find(group_id), []).append(group_id)

    group_components: Dict[str, str] = {}
    merged_components = 0
    merged_groups = 0
    source_group_ids = set(groups.parent)
    for members in component_members.values():
        members.sort()
        if len(members) == 1:
            component_id = members[0]
        else:
            merged_components += 1
            merged_groups += len(members)
            digest = hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()
            component_id = "s1cc:" + digest
            if component_id in source_group_ids and component_id not in members:
                raise ValueError("Generated component group_id collides with source group_id {!r}".format(component_id))
        for group_id in members:
            group_components[group_id] = component_id

    component_splits = {
        component_id: deterministic_split(component_id, seed, val_ratio)
        for component_id in set(group_components.values())
    }

    train_temporary, train_stream = _temporary_stream(train_output)
    val_temporary, val_stream = _temporary_stream(val_output)
    train_records = 0
    val_records = 0
    rewritten_records = 0
    try:
        for _, record in iter_jsonl(input_path):
            source_group_id = record["group_id"]
            component_id = group_components[source_group_id]
            split = component_splits[component_id]
            output_record = dict(record)
            if component_id != source_group_id:
                extensions = output_record.get("extensions")
                if extensions is None:
                    extensions = {}
                elif not isinstance(extensions, Mapping):
                    raise ValueError("Record {!r} extensions must be an object".format(record.get("id")))
                else:
                    extensions = dict(extensions)
                extension_key = "s1_udf.split"
                if extension_key in extensions:
                    raise ValueError(
                        "Record {!r} already defines reserved extension {!r}".format(record.get("id"), extension_key)
                    )
                extensions[extension_key] = {
                    "source_group_id": source_group_id,
                    "component_group_id": component_id,
                }
                output_record["extensions"] = extensions
                output_record["group_id"] = component_id
                rewritten_records += 1
            output_record["split"] = split
            if split == "val":
                val_stream.write(json_line(output_record))
                val_records += 1
            else:
                train_stream.write(json_line(output_record))
                train_records += 1
        for stream in (train_stream, val_stream):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        _commit_pair(train_temporary, train_output, val_temporary, val_output, overwrite)
    except BaseException:
        for stream in (train_stream, val_stream):
            if not stream.closed:
                stream.close()
        for path in (train_temporary, val_temporary):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    train_groups = sum(component_splits[group_components[group_id]] == "train" for group_id in groups.parent)
    val_groups = len(groups.parent) - train_groups
    train_components = sum(split == "train" for split in component_splits.values())
    val_components = len(component_splits) - train_components
    return SplitSummary(
        total_records=total_records,
        train_records=train_records,
        val_records=val_records,
        total_groups=len(groups.parent),
        train_groups=train_groups,
        val_groups=val_groups,
        total_components=len(component_splits),
        train_components=train_components,
        val_components=val_components,
        merged_components=merged_components,
        merged_groups=merged_groups,
        rewritten_records=rewritten_records,
        unique_assets=len(asset_groups),
        leakage_overlap=0 if audit_assets else None,
        seed=seed,
        val_ratio=val_ratio,
    )
