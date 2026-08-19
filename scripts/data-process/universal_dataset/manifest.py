"""Build reproducible S1-UDF manifests from validated record files."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from .adapter import CURRENT_UDF_VERSION
from .io import iter_jsonl, parse_json_strict
from .split import AssetIdentityRegistry
from .validation import FORMAT_NAME, _file_sha256, ensure_valid, validate_manifest


class ManifestBuildError(ValueError):
    pass


def infer_record_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".parquet":
        return "parquet"
    raise ManifestBuildError("Unsupported record file suffix: {}".format(path))


def _iter_json_records(path: Path) -> Iterator[Tuple[int, Mapping[str, Any]]]:
    size = path.stat().st_size
    if size <= 256 * 1024 * 1024:
        value = parse_json_strict(path.read_text(encoding="utf-8"))
        values = value if isinstance(value, list) else [value]
        for index, record in enumerate(values, 1):
            if not isinstance(record, Mapping):
                raise ManifestBuildError("JSON record {} in {} is not an object".format(index, path))
            yield index, record
        return
    try:
        import ijson
    except ImportError as error:
        raise ManifestBuildError(
            "ijson is required to build a manifest for JSON files over 256 MiB: {}".format(path)
        ) from error
    with path.open("rb") as stream:
        first_event = next(ijson.parse(stream), None)
    if first_event is None or first_event[1] != "start_array":
        raise ManifestBuildError("Large S1-UDF JSON must have an array root: {}".format(path))
    with path.open("rb") as stream:
        for index, record in enumerate(ijson.items(stream, "item"), 1):
            if not isinstance(record, Mapping):
                raise ManifestBuildError("JSON record {} in {} is not an object".format(index, path))
            yield index, record


def iter_record_file(path: Path, file_format: str) -> Iterator[Tuple[int, Mapping[str, Any]]]:
    if file_format == "jsonl":
        yield from iter_jsonl(path)
        return
    if file_format == "json":
        yield from _iter_json_records(path)
        return
    if file_format == "parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ManifestBuildError("pyarrow is required to inspect {}".format(path)) from error
        row_number = 0
        value = parquet.ParquetFile(path)
        for batch in value.iter_batches(batch_size=1024):
            for record in batch.to_pylist():
                row_number += 1
                if not isinstance(record, Mapping):
                    raise ManifestBuildError(
                        "Parquet row {} in {} is not an object".format(row_number, path)
                    )
                yield row_number, record
        return
    raise ManifestBuildError("Unsupported record format {!r}".format(file_format))


def _relative_path(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(str(path), str(base_dir))).as_posix()


def build_manifest(
    record_files: Sequence[Path],
    manifest_path: Path,
    dataset_name: str,
    dataset_version: Optional[str] = None,
    description: Optional[str] = None,
    homepage: Optional[str] = None,
    license_name: Optional[str] = None,
    revision: Optional[str] = None,
    source_schemas: Sequence[Mapping[str, Any]] = (),
    provenance: Optional[Mapping[str, Any]] = None,
    record_splits: Optional[Mapping[Path, str]] = None,
    validate_records: bool = True,
    check_assets: bool = False,
) -> Dict[str, Any]:
    """Scan record files and derive counts, checksums, tasks, splits, and leakage identities."""

    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ManifestBuildError("dataset_name must be non-empty")
    if not record_files:
        raise ManifestBuildError("At least one record file is required")
    manifest_path = manifest_path.expanduser().resolve()
    base_dir = manifest_path.parent
    paths = [path.expanduser().resolve() for path in record_files]
    if len(set(paths)) != len(paths):
        raise ManifestBuildError("record_files contains duplicate resolved paths")
    if manifest_path in paths:
        raise ManifestBuildError("Manifest output must differ from every record file")
    explicit_splits: Dict[Path, str] = {}
    for raw_path, raw_split in (record_splits or {}).items():
        split_path = Path(raw_path).expanduser().resolve()
        split_name = str(raw_split).strip()
        if split_path not in paths:
            raise ManifestBuildError(
                "record_splits path is not present in record_files: {}".format(split_path)
            )
        if not split_name:
            raise ManifestBuildError("record_splits values must be non-empty")
        previous = explicit_splits.setdefault(split_path, split_name)
        if previous != split_name:
            raise ManifestBuildError(
                "record file {} is assigned conflicting splits".format(split_path)
            )

    seen_ids: Set[str] = set()
    group_splits: Dict[str, str] = {}
    asset_groups: Dict[str, str] = {}
    asset_splits: Dict[str, str] = {}
    asset_identities = AssetIdentityRegistry()
    episode_groups: Dict[str, str] = {}
    episode_splits: Dict[str, str] = {}
    groups: Set[str] = set()
    tasks: Set[str] = set()
    splits: Counter = Counter()
    entries: List[Dict[str, Any]] = []
    total_records = 0
    unsplit_records = 0

    for path in paths:
        if not path.is_file():
            raise ManifestBuildError("Record file does not exist: {}".format(path))
        file_format = infer_record_format(path)
        file_count = 0
        file_splits: Set[str] = set()
        file_has_unsplit = False
        explicit_split = explicit_splits.get(path)
        if explicit_split is not None:
            splits.setdefault(explicit_split, 0)
        for locator, record in iter_record_file(path, file_format):
            file_count += 1
            total_records += 1
            if validate_records:
                try:
                    ensure_valid(
                        record,
                        check_json_schema=True,
                        check_assets=check_assets,
                        base_dir=path.parent,
                    )
                except Exception as error:
                    raise ManifestBuildError(
                        "Invalid record {}:{}: {}".format(path, locator, error)
                    ) from error
            if record.get("format") != FORMAT_NAME or record.get("schema_version") != CURRENT_UDF_VERSION:
                raise ManifestBuildError(
                    "Record {}:{} is not {} {}".format(
                        path, locator, FORMAT_NAME, CURRENT_UDF_VERSION
                    )
                )
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ManifestBuildError("Record {}:{} has no stable id".format(path, locator))
            if identifier in seen_ids:
                raise ManifestBuildError("Duplicate record id {!r}".format(identifier))
            seen_ids.add(identifier)
            group_id = record.get("group_id")
            if not isinstance(group_id, str) or not group_id:
                raise ManifestBuildError("Record {!r} has no stable group_id".format(identifier))
            groups.add(group_id)
            split = record.get("split")
            if explicit_split is not None and split != explicit_split:
                raise ManifestBuildError(
                    "Record {}:{} has split {!r}, expected {!r} from record_splits".format(
                        path, locator, split, explicit_split
                    )
                )
            if isinstance(split, str) and split:
                splits[split] += 1
                file_splits.add(split)
                previous = group_splits.setdefault(group_id, split)
                if previous != split:
                    raise ManifestBuildError(
                        "group {!r} appears in splits {!r} and {!r}".format(group_id, previous, split)
                    )
            else:
                unsplit_records += 1
                file_has_unsplit = True
            for task in record.get("task_types") or []:
                if isinstance(task, str) and task:
                    tasks.add(task)
            for asset in record.get("assets") or []:
                if not isinstance(asset, Mapping):
                    continue
                try:
                    identities = asset_identities.add(asset, path)
                except ValueError as error:
                    raise ManifestBuildError(str(error)) from error
                for identity in identities:
                    previous_group = asset_groups.setdefault(identity, group_id)
                    if previous_group != group_id:
                        raise ManifestBuildError(
                            "asset {} belongs to groups {!r} and {!r}; run canonical split first".format(
                                identity, previous_group, group_id
                            )
                        )
                    if isinstance(split, str) and split:
                        previous_split = asset_splits.setdefault(identity, split)
                        if previous_split != split:
                            raise ManifestBuildError(
                                "asset {} appears in splits {!r} and {!r}".format(
                                    identity, previous_split, split
                                )
                            )
            episode = record.get("episode")
            if isinstance(episode, Mapping) and isinstance(episode.get("id"), str):
                episode_id = episode["id"]
                previous_group = episode_groups.setdefault(episode_id, group_id)
                if previous_group != group_id:
                    raise ManifestBuildError(
                        "episode {!r} belongs to groups {!r} and {!r}".format(
                            episode_id, previous_group, group_id
                        )
                    )
                if isinstance(split, str) and split:
                    previous_split = episode_splits.setdefault(episode_id, split)
                    if previous_split != split:
                        raise ManifestBuildError(
                            "episode {!r} appears in splits {!r} and {!r}".format(
                                episode_id, previous_split, split
                            )
                        )
        entry: Dict[str, Any] = {
            "path": _relative_path(path, base_dir),
            "format": file_format,
            "count": file_count,
            "sha256": _file_sha256(path),
        }
        if explicit_split is not None:
            entry["split"] = explicit_split
        elif len(file_splits) == 1 and not file_has_unsplit:
            entry["split"] = next(iter(file_splits))
        entries.append(entry)

    if total_records == 0:
        raise ManifestBuildError("Dataset contains no records")
    if splits and unsplit_records:
        raise ManifestBuildError(
            "Dataset mixes {} split records with {} records lacking split".format(
                sum(splits.values()), unsplit_records
            )
        )

    dataset: Dict[str, Any] = {"name": dataset_name.strip()}
    optional_dataset = {
        "version": dataset_version,
        "description": description,
        "homepage": homepage,
        "license": license_name,
        "revision": revision,
    }
    dataset.update({key: value for key, value in optional_dataset.items() if value is not None})
    manifest: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "schema_version": CURRENT_UDF_VERSION,
        "dataset": dataset,
        "record_files": entries,
        "task_types": sorted(tasks),
        "grouping": {
            "field": "group_id",
            "semantics": (
                "Complete source groups and episodes are indivisible; shared canonical assets "
                "must already have one canonical group_id before publication."
            ),
            "identity_fields": [
                "group_id",
                "assets[].sha256",
                "assets[].source_identities[]",
                "assets[].uri",
                "assets[].source_ref",
                "episode.id",
            ],
        },
        "statistics": {
            "records": total_records,
            "groups": len(groups),
            "unique_assets": len(asset_identities),
            "episodes": len(episode_groups),
        },
    }
    if splits:
        manifest["splits"] = dict(sorted(splits.items()))
    if source_schemas:
        manifest["source_schemas"] = [dict(item) for item in source_schemas]
    if provenance is not None:
        manifest["provenance"] = dict(provenance)
    schema_issues = validate_manifest(
        manifest,
        manifest_path=manifest_path,
        check_json_schema=True,
        check_files=False,
    )
    if schema_issues:
        raise ManifestBuildError(
            "Generated manifest violates the S1-UDF schema: {}".format(
                "; ".join(str(issue) for issue in schema_issues[:8])
            )
        )
    return manifest
