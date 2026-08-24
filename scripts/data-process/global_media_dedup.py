#!/usr/bin/env python3
"""Materialize decontaminated SFT train entrypoints from a manifest.

All enabled eval media are reserved first. Training datasets are then processed
in explicit priority order. Rows that touch the same dataset's eval media are
always excluded. Cross-dataset eval overlap can be exempted for related families
such as GQA and Visual Genome, so shared images stay in each dataset's train
split. Cross-dataset train media can either be assigned to the highest-priority
dataset or retained in every dataset. Repeated QA rows on the same media inside
one dataset remain intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import uuid
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO
from urllib.parse import urlsplit


SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = SCRIPT_DIR / "split"
if str(SPLIT_DIR) not in sys.path:
    sys.path.insert(0, str(SPLIT_DIR))

from grouped_jsonl_split import load_record, record_values


DEFAULT_MANIFEST = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/curated_dataset_entrypoints.json"
)
COCO_STEM_RE = re.compile(
    r"(?:COCO_(?:train|val|test)\d{4}_)?0*([0-9]+)$", re.IGNORECASE
)
ROBO2VLM_QUESTION_RE = re.compile(r"_q\d+$", re.IGNORECASE)
EXPLICIT_LINEAGE_FIELDS = (
    "episode_id",
    "episode",
    "trajectory_id",
    "trajectory",
    "video_id",
    "clip_id",
    "frame_id",
)
MEDIA_PLACEHOLDER_RE = re.compile(r"<\s*(?:image|video|audio)\s*>", re.IGNORECASE)
MEDIA_CONTENT_TYPES = {
    "audio",
    "audio_url",
    "image",
    "image_url",
    "video",
    "video_url",
}
ASSISTANT_ROLES = {"assistant", "gpt", "model"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    identity_namespace: str
    use_video_stem_identity: bool
    allow_text_only: bool
    source_train: Path
    clean_train: Path
    eval_paths: tuple[Path, ...]
    prerequisite_report: Path | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--exclusions-output", type=Path)
    parser.add_argument("--cache-db", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--max-report-examples", type=int, default=100)
    parser.add_argument("--hash-workers", type=int, default=1)
    parser.add_argument("--hash-prefetch-rows", type=int, default=2_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def canonical_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _nested_value(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"report has no required field {dotted_path!r}")
        current = current[part]
    return current


def load_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], list[DatasetSpec], list[str]]:
    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), dict):
        raise ValueError("manifest must contain a datasets object")
    policy = manifest.get("global_dedup")
    if not isinstance(policy, dict):
        raise ValueError("manifest must contain a global_dedup object")
    priority = policy.get("training_priority")
    if not isinstance(priority, list) or not all(isinstance(item, str) for item in priority):
        raise ValueError("global_dedup.training_priority must be a string list")

    base = manifest_path.parent
    enabled = {
        name
        for name, config in manifest["datasets"].items()
        if isinstance(config, dict) and config.get("enabled", True)
    }
    if len(priority) != len(set(priority)):
        raise ValueError("global_dedup.training_priority contains duplicate datasets")
    if set(priority) != enabled:
        raise ValueError(
            "training_priority must contain every enabled dataset exactly once; "
            f"missing={sorted(enabled - set(priority))} extra={sorted(set(priority) - enabled)}"
        )

    specs: list[DatasetSpec] = []
    output_paths: set[Path] = set()
    input_paths: set[Path] = set()
    for name in priority:
        config = manifest["datasets"][name]
        if not isinstance(config, dict):
            raise ValueError(f"dataset {name!r} must be an object")
        source_value = config.get("source_train") or config.get("split_train")
        clean_value = config.get("train")
        eval_value = config.get("eval")
        if not source_value or not clean_value or not eval_value:
            raise ValueError(
                f"enabled dataset {name!r} requires source_train, train, and eval"
            )
        eval_values = eval_value if isinstance(eval_value, list) else [eval_value]
        source_train = canonical_path(source_value, base)
        clean_train = canonical_path(clean_value, base)
        eval_paths = tuple(canonical_path(value, base) for value in eval_values)
        if not source_train.is_file():
            raise FileNotFoundError(f"{name}: source_train does not exist: {source_train}")
        for path in eval_paths:
            if not path.is_file():
                raise FileNotFoundError(f"{name}: eval does not exist: {path}")
        if clean_train == source_train or clean_train in eval_paths:
            raise ValueError(f"{name}: clean train output must differ from all inputs")
        if clean_train in output_paths:
            raise ValueError(f"duplicate clean train output: {clean_train}")
        if "dedup_owner" in config:
            raise ValueError(
                f"{name}: dedup_owner is not allowed; every manifest dataset must own "
                "its retained media independently"
            )
        identity_namespace = str(config.get("identity_namespace") or name).strip().casefold()
        if not identity_namespace or not re.fullmatch(r"[a-z0-9._-]+", identity_namespace):
            raise ValueError(
                f"{name}: identity_namespace must match [a-z0-9._-]+, "
                f"got {identity_namespace!r}"
            )
        allow_text_only = config.get("allow_text_only", False)
        if not isinstance(allow_text_only, bool):
            raise ValueError(f"{name}: allow_text_only must be a boolean")
        use_video_stem_identity = config.get("use_video_stem_identity", True)
        if not isinstance(use_video_stem_identity, bool):
            raise ValueError(f"{name}: use_video_stem_identity must be a boolean")
        prerequisite_value = config.get("prerequisite_report")
        prerequisite_requirements = config.get("prerequisite_values")
        prerequisite_report: Path | None = None
        if prerequisite_value is not None or prerequisite_requirements is not None:
            if not prerequisite_value or not isinstance(prerequisite_requirements, dict):
                raise ValueError(
                    f"{name}: prerequisite_report and prerequisite_values must be set together"
                )
            prerequisite_report = canonical_path(prerequisite_value, base)
            if not prerequisite_report.is_file():
                raise FileNotFoundError(
                    f"{name}: prerequisite report does not exist: {prerequisite_report}"
                )
            with prerequisite_report.open("r", encoding="utf-8") as stream:
                prerequisite = json.load(stream)
            for dotted_path, expected in prerequisite_requirements.items():
                if not isinstance(dotted_path, str) or not dotted_path:
                    raise ValueError(f"{name}: prerequisite field names must be non-empty strings")
                actual = _nested_value(prerequisite, dotted_path)
                if actual != expected:
                    raise ValueError(
                        f"{name}: prerequisite {dotted_path!r} is {actual!r}, "
                        f"expected {expected!r}"
                    )
            prerequisite_mtime = prerequisite_report.stat().st_mtime_ns
            for path in (source_train, *eval_paths):
                if path.stat().st_mtime_ns > prerequisite_mtime:
                    raise ValueError(
                        f"{name}: {path} is newer than prerequisite report; rerun conversion"
                    )
        specs.append(
            DatasetSpec(
                name,
                identity_namespace,
                use_video_stem_identity,
                allow_text_only,
                source_train,
                clean_train,
                eval_paths,
                prerequisite_report,
            )
        )
        input_paths.update((source_train, *eval_paths))
        output_paths.add(clean_train)
    overlap = output_paths & input_paths
    if overlap:
        raise ValueError(f"clean train outputs overlap input files: {sorted(map(str, overlap))}")
    return manifest, specs, priority


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _normal_text(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _normalized_local_path(path_text: str | Path) -> str:
    path = Path(path_text).expanduser()
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _file_fingerprint(path: Path, logical_path: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return {
        "path": str((logical_path or path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hasher.hexdigest(),
    }


class IdentityStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS file_hash_cache (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS eval_identity (
                identity TEXT NOT NULL,
                dataset TEXT NOT NULL,
                PRIMARY KEY(identity, dataset)
            );
            CREATE INDEX IF NOT EXISTS eval_identity_lookup ON eval_identity(identity);
            CREATE TABLE IF NOT EXISTS train_owner (
                identity TEXT PRIMARY KEY,
                dataset TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS verify_owner (
                identity TEXT PRIMARY KEY,
                dataset TEXT NOT NULL
            );
            """
        )
        self.connection.execute("DELETE FROM eval_identity")
        self.connection.execute("DELETE FROM train_owner")
        self.connection.execute("DELETE FROM verify_owner")
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def lookup(self, table: str, identities: Iterable[str]) -> dict[str, set[str]]:
        values = sorted(set(identities))
        if not values:
            return {}
        if table not in {"eval_identity", "train_owner", "verify_owner"}:
            raise ValueError(f"unsupported identity table: {table}")
        placeholders = ",".join("?" for _ in values)
        rows = self.connection.execute(
            f"SELECT identity, dataset FROM {table} WHERE identity IN ({placeholders})",
            values,
        )
        result: dict[str, set[str]] = {}
        for identity, dataset in rows:
            result.setdefault(identity, set()).add(dataset)
        return result

    def add_eval(self, identities: Iterable[str], dataset: str) -> int:
        added = 0
        for identity in set(identities):
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO eval_identity(identity, dataset) VALUES (?, ?)",
                (identity, dataset),
            )
            added += cursor.rowcount
        return added

    def add_owner(self, table: str, identities: Iterable[str], dataset: str) -> None:
        if table not in {"train_owner", "verify_owner"}:
            raise ValueError(f"unsupported owner table: {table}")
        self.connection.executemany(
            f"INSERT OR IGNORE INTO {table}(identity, dataset) VALUES (?, ?)",
            ((identity, dataset) for identity in set(identities)),
        )


class FileHasher:
    def __init__(
        self,
        store: IdentityStore,
        memory_limit: int = 100_000,
        hash_workers: int = 1,
    ) -> None:
        self.store = store
        self.memory_limit = memory_limit
        self.memory: OrderedDict[str, tuple[int, int, str]] = OrderedDict()
        self.stats: Counter[str] = Counter()
        self.executor = (
            ThreadPoolExecutor(max_workers=hash_workers, thread_name_prefix="media-hash")
            if hash_workers > 1
            else None
        )

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _inspect_file(
        item: tuple[str, tuple[int, int, str] | None],
    ) -> tuple[str, int, int, str, bool]:
        resolved, database_value = item
        path = Path(resolved)
        try:
            file_stat = path.stat()
        except FileNotFoundError:
            raise FileNotFoundError(f"media file does not exist: {path}") from None
        if not stat.S_ISREG(file_stat.st_mode):
            raise FileNotFoundError(f"media file does not exist: {path}")
        if database_value and database_value[:2] == (
            file_stat.st_size,
            file_stat.st_mtime_ns,
        ):
            return (
                resolved,
                file_stat.st_size,
                file_stat.st_mtime_ns,
                database_value[2],
                False,
            )
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
        return (
            resolved,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            hasher.hexdigest(),
            True,
        )

    def _database_values(
        self, resolved_paths: list[str]
    ) -> dict[str, tuple[int, int, str]]:
        result: dict[str, tuple[int, int, str]] = {}
        for start in range(0, len(resolved_paths), 500):
            batch = resolved_paths[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self.store.connection.execute(
                f"SELECT path, size, mtime_ns, sha256 FROM file_hash_cache "
                f"WHERE path IN ({placeholders})",
                batch,
            )
            for path, size, mtime_ns, digest in rows:
                result[str(path)] = (int(size), int(mtime_ns), str(digest))
        return result

    def _remember(self, resolved: str, size: int, mtime_ns: int, digest: str) -> None:
        self.memory[resolved] = (size, mtime_ns, digest)
        self.memory.move_to_end(resolved)
        if len(self.memory) > self.memory_limit:
            self.memory.popitem(last=False)

    def prefetch(self, path_values: Iterable[str]) -> None:
        if self.executor is None:
            return
        resolved_paths = sorted(
            {
                _normalized_local_path(path_text)
                for path_text in path_values
                if urlsplit(path_text).scheme.lower() not in {"http", "https"}
                and _normalized_local_path(path_text) not in self.memory
            }
        )
        if not resolved_paths:
            return
        database_values = self._database_values(resolved_paths)
        items = [(path, database_values.get(path)) for path in resolved_paths]
        for resolved, size, mtime_ns, digest, was_hashed in self.executor.map(
            self._inspect_file, items
        ):
            if was_hashed:
                self.stats["files_hashed"] += 1
                self.stats["bytes_hashed"] += size
                self.store.connection.execute(
                    """
                    INSERT INTO file_hash_cache(path, size, mtime_ns, sha256)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        size=excluded.size, mtime_ns=excluded.mtime_ns, sha256=excluded.sha256
                    """,
                    (resolved, size, mtime_ns, digest),
                )
            else:
                self.stats["database_hits"] += 1
            self._remember(resolved, size, mtime_ns, digest)

    def sha256(self, path_text: str) -> str:
        resolved = _normalized_local_path(path_text)
        cached = self.memory.get(resolved)
        if cached:
            self.memory.move_to_end(resolved)
            self.stats["memory_hits"] += 1
            return cached[2]

        path = Path(resolved)
        try:
            file_stat = path.stat()
        except FileNotFoundError:
            raise FileNotFoundError(f"media file does not exist: {path}")
        if not stat.S_ISREG(file_stat.st_mode):
            raise FileNotFoundError(f"media file does not exist: {path}")
        row = self.store.connection.execute(
            "SELECT size, mtime_ns, sha256 FROM file_hash_cache WHERE path = ?",
            (resolved,),
        ).fetchone()
        if row and (row[0], row[1]) == (file_stat.st_size, file_stat.st_mtime_ns):
            digest = str(row[2])
            self.stats["database_hits"] += 1
        else:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            self.stats["files_hashed"] += 1
            self.stats["bytes_hashed"] += file_stat.st_size
            self.store.connection.execute(
                """
                INSERT INTO file_hash_cache(path, size, mtime_ns, sha256)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size=excluded.size, mtime_ns=excluded.mtime_ns, sha256=excluded.sha256
                """,
                (resolved, file_stat.st_size, file_stat.st_mtime_ns, digest),
            )
        self._remember(resolved, file_stat.st_size, file_stat.st_mtime_ns, digest)
        return digest


class IdentityResolver:
    def __init__(self, hasher: FileHasher) -> None:
        self.hasher = hasher

    @staticmethod
    def _coco_id(path_text: str) -> str | None:
        path = Path(path_text)
        match = COCO_STEM_RE.fullmatch(path.stem)
        normalized = path_text.replace("\\", "/").casefold()
        if match and (path.stem.casefold().startswith("coco_") or "coco" in normalized):
            return str(int(match.group(1)))
        return None

    def _image_identities(self, path_text: str) -> set[str]:
        if urlsplit(path_text).scheme.lower() in {"http", "https"}:
            return {f"image:url:{path_text}"}
        identities = {
            f"image:path:{_normalized_local_path(path_text)}",
            f"image:sha256:{self.hasher.sha256(path_text)}",
        }
        if coco_id := self._coco_id(path_text):
            identities.add(f"image:coco:{coco_id}")
        return identities

    @staticmethod
    def _video_identities(
        path_text: str,
        identity_namespace: str,
        use_video_stem_identity: bool,
    ) -> set[str]:
        if urlsplit(path_text).scheme.lower() in {"http", "https"}:
            stem = Path(urlsplit(path_text).path).stem.casefold()
            result = {f"video:url:{path_text}"}
        else:
            path = Path(path_text)
            stem = path.stem.casefold()
            result = {f"video:path:{_normalized_local_path(path)}"}
        if stem and use_video_stem_identity:
            result.add(f"video:id:{identity_namespace}:{stem}")
        return result

    @classmethod
    def _contains_media_placeholder(cls, value: Any) -> bool:
        if isinstance(value, str):
            return MEDIA_PLACEHOLDER_RE.search(value) is not None
        if isinstance(value, Mapping):
            content_type = str(value.get("type") or "").strip().casefold()
            if content_type in MEDIA_CONTENT_TYPES:
                return True
            return any(cls._contains_media_placeholder(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(cls._contains_media_placeholder(item) for item in value)
        return False

    @classmethod
    def _contains_media_content_block(cls, value: Any) -> bool:
        if isinstance(value, Mapping):
            content_type = str(value.get("type") or "").strip().casefold()
            if content_type in MEDIA_CONTENT_TYPES:
                return True
            return any(cls._contains_media_content_block(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(cls._contains_media_content_block(item) for item in value)
        return False

    @classmethod
    def _messages_require_media(cls, messages: list[Any]) -> bool:
        for message in messages:
            if cls._contains_media_content_block(message):
                return True
            if not isinstance(message, Mapping):
                if cls._contains_media_placeholder(message):
                    return True
                continue
            role = str(message.get("role") or "").strip().casefold()
            if role not in ASSISTANT_ROLES and cls._contains_media_placeholder(
                message.get("content")
            ):
                return True
        return False

    @classmethod
    def _text_identity(cls, record: Mapping[str, Any]) -> str | None:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            return None
        if cls._messages_require_media(messages):
            return None
        payload = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"text:sha256:{hashlib.sha256(payload).hexdigest()}"

    def resolve(
        self,
        record: dict[str, Any],
        jsonl_path: Path,
        dataset: str,
        identity_namespace: str,
        use_video_stem_identity: bool,
        allow_text_only: bool = False,
    ) -> set[str]:
        identities: set[str] = set()
        for image in record_values(record, "images", jsonl_path):
            identities.update(self._image_identities(image))
        for video in record_values(record, "videos", jsonl_path):
            identities.update(
                self._video_identities(
                    video,
                    identity_namespace,
                    use_video_stem_identity,
                )
            )
        for field in EXPLICIT_LINEAGE_FIELDS:
            for value in record_values(record, field, jsonl_path):
                normalized = _normal_text(value)
                if normalized:
                    identities.add(
                        f"lineage:{identity_namespace}:{field}:{normalized}"
                    )
                    identities.add(
                        f"lineage:{identity_namespace}:value:{normalized}"
                    )
        if dataset.casefold().replace("_", "-") == "robo2vlm":
            sample_id = str(record.get("id") or "").strip()
            lineage = ROBO2VLM_QUESTION_RE.sub("", sample_id)
            if not sample_id or lineage == sample_id:
                raise ValueError(f"Robo2VLM row has invalid episode/question id: {sample_id!r}")
            identities.add(
                f"lineage:{identity_namespace}:robo2vlm:{lineage.casefold()}"
            )
        if not identities and allow_text_only:
            if text_identity := self._text_identity(record):
                identities.add(text_identity)
        return identities


def _iter_records(path: Path) -> Iterable[tuple[int, str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            yield line_number, line.rstrip("\r\n") + "\n", load_record(line, path, line_number)


def _iter_records_with_prefetch(
    path: Path,
    hasher: FileHasher,
    hash_workers: int,
    prefetch_rows: int,
) -> Iterable[tuple[int, str, dict[str, Any]]]:
    if hash_workers <= 1:
        yield from _iter_records(path)
        return
    batch: list[tuple[int, str, dict[str, Any]]] = []
    for item in _iter_records(path):
        batch.append(item)
        if len(batch) < prefetch_rows:
            continue
        hasher.prefetch(
            image
            for _line_number, _line, record in batch
            for image in record_values(record, "images", path)
        )
        yield from batch
        batch = []
    if batch:
        hasher.prefetch(
            image
            for _line_number, _line, record in batch
            for image in record_values(record, "images", path)
        )
        yield from batch


def _record_id(record: Mapping[str, Any], line_number: int) -> str:
    for key in ("id", "sample_id", "question_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"line:{line_number}"


def _identity_kind(identity: str) -> str:
    parts = identity.split(":", 2)
    return ":".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _conflict_payload(conflicts: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    return [
        {"identity": identity, "datasets": sorted(datasets)}
        for identity, datasets in sorted(conflicts.items())
    ]


def parse_eval_overlap_exempt_groups(
    value: Any, enabled: Sequence[str]
) -> tuple[frozenset[str], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(
            "global_dedup.eval_overlap_exempt_groups must be a list of dataset name lists"
        )
    enabled_names = set(enabled)
    groups: list[frozenset[str]] = []
    for index, group in enumerate(value):
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError(
                f"eval_overlap_exempt_groups[{index}] must contain at least two dataset names"
            )
        names: list[str] = []
        for item in group:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"eval_overlap_exempt_groups[{index}] must contain non-empty dataset names"
                )
            names.append(item.strip())
        if len(names) != len(set(names)):
            raise ValueError(f"eval_overlap_exempt_groups[{index}] contains duplicate datasets")
        unknown = sorted(set(names) - enabled_names)
        if unknown:
            raise ValueError(
                f"eval_overlap_exempt_groups[{index}] contains disabled or unknown "
                f"datasets: {unknown}"
            )
        groups.append(frozenset(names))
    return tuple(groups)


def eval_owner_is_exempt(
    dataset: str,
    owner: str,
    groups: Sequence[frozenset[str]],
) -> bool:
    if dataset == owner:
        return False
    return any(dataset in group and owner in group for group in groups)


def partition_eval_conflicts(
    conflicts: Mapping[str, set[str]],
    dataset: str,
    groups: Sequence[frozenset[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    blocking: dict[str, set[str]] = {}
    exempted: dict[str, set[str]] = {}
    for identity, owners in conflicts.items():
        blocked_owners = {
            owner for owner in owners if not eval_owner_is_exempt(dataset, owner, groups)
        }
        exempt_owners = set(owners) - blocked_owners
        if blocked_owners:
            blocking[identity] = blocked_owners
        if exempt_owners:
            exempted[identity] = exempt_owners
    return blocking, exempted


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.progress_every < 0 or args.max_report_examples < 0:
        raise ValueError("progress and example limits must be non-negative")
    if args.hash_workers < 1 or args.hash_prefetch_rows < 1:
        raise ValueError("hash workers and prefetch rows must be positive")
    manifest, specs, priority = load_configuration(args)
    manifest_path = args.manifest.expanduser().resolve()
    manifest_fingerprint = _file_fingerprint(manifest_path)
    policy = manifest["global_dedup"]
    require_media_identity = bool(policy.get("require_media_identity", True))
    deduplicate_cross_dataset_train = policy.get(
        "deduplicate_cross_dataset_train", True
    )
    if not isinstance(deduplicate_cross_dataset_train, bool):
        raise ValueError(
            "global_dedup.deduplicate_cross_dataset_train must be a boolean"
        )
    eval_overlap_exempt_groups = parse_eval_overlap_exempt_groups(
        policy.get("eval_overlap_exempt_groups"),
        priority,
    )
    base = manifest_path.parent
    report_output = canonical_path(
        args.report_output or policy.get("report", "global_media_dedup_report.json"), base
    )
    exclusions_output = canonical_path(
        args.exclusions_output or policy.get("exclusions", "global_media_dedup_exclusions.jsonl"), base
    )
    cache_db = canonical_path(
        args.cache_db or policy.get("cache_db", ".global_media_identity_cache.sqlite"), base
    )
    final_outputs = [report_output, exclusions_output]
    if not args.audit_only:
        final_outputs.extend(spec.clean_train for spec in specs)
    if len(final_outputs) != len(set(final_outputs)):
        raise ValueError("report, exclusions, and clean train output paths must be unique")
    for path in final_outputs:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {path}")

    store = IdentityStore(cache_db)
    hasher = FileHasher(store, hash_workers=args.hash_workers)
    resolver = IdentityResolver(hasher)
    global_stats: Counter[str] = Counter()
    dataset_reports: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    temporary_outputs: dict[str, Path] = {}
    streams: dict[str, TextIO] = {}
    exclusions_temporary = _temporary_sibling(exclusions_output)
    report_temporary = _temporary_sibling(report_output)
    exclusions_stream = exclusions_temporary.open("w", encoding="utf-8", newline="\n")
    try:
        for spec in specs:
            counts: Counter[str] = Counter()
            identity_kinds: Counter[str] = Counter()
            for eval_path in spec.eval_paths:
                for line_number, _line, record in _iter_records_with_prefetch(
                    eval_path,
                    hasher,
                    args.hash_workers,
                    args.hash_prefetch_rows,
                ):
                    counts["eval_rows"] += 1
                    try:
                        identities = resolver.resolve(
                            record,
                            eval_path,
                            spec.name,
                            spec.identity_namespace,
                            spec.use_video_stem_identity,
                            spec.allow_text_only,
                        )
                    except (FileNotFoundError, OSError, ValueError) as error:
                        raise ValueError(f"{eval_path}:{line_number}: {error}") from error
                    if not identities:
                        counts["eval_rows_without_identity"] += 1
                        if require_media_identity:
                            raise ValueError(
                                f"{eval_path}:{line_number}: no image, video, or lineage "
                                "identity could be derived"
                            )
                    if any(identity.startswith("text:sha256:") for identity in identities):
                        counts["eval_text_only_rows"] += 1
                    existing = store.lookup("eval_identity", identities)
                    other_datasets = {
                        owner for owners in existing.values() for owner in owners if owner != spec.name
                    }
                    if other_datasets:
                        counts["eval_rows_overlapping_other_eval"] += 1
                    counts["eval_unique_identities_added"] += store.add_eval(identities, spec.name)
                    for identity in identities:
                        identity_kinds[_identity_kind(identity)] += 1
                    if args.progress_every and counts["eval_rows"] % args.progress_every == 0:
                        print(f"[eval-index] {spec.name} rows={counts['eval_rows']:,}", flush=True)
                    if counts["eval_rows"] % 10_000 == 0:
                        store.connection.commit()
            dataset_reports[spec.name] = {
                "identity_namespace": spec.identity_namespace,
                "use_video_stem_identity": spec.use_video_stem_identity,
                "allow_text_only": spec.allow_text_only,
                "prerequisite_report": (
                    str(spec.prerequisite_report) if spec.prerequisite_report else None
                ),
                "source_train": str(spec.source_train),
                "train": str(spec.clean_train),
                "eval": [str(path) for path in spec.eval_paths],
                "counts": counts,
                "identity_observations": identity_kinds,
            }
            store.connection.commit()

        for spec in specs:
            report = dataset_reports[spec.name]
            counts = report["counts"]
            identity_kinds = report["identity_observations"]
            if not args.audit_only:
                temporary = _temporary_sibling(spec.clean_train)
                temporary_outputs[spec.name] = temporary
                streams[spec.name] = temporary.open("w", encoding="utf-8", newline="\n")
            for line_number, line, record in _iter_records_with_prefetch(
                spec.source_train,
                hasher,
                args.hash_workers,
                args.hash_prefetch_rows,
            ):
                counts["source_train_rows"] += 1
                global_stats["source_train_rows"] += 1
                try:
                    identities = resolver.resolve(
                        record,
                        spec.source_train,
                        spec.name,
                        spec.identity_namespace,
                        spec.use_video_stem_identity,
                        spec.allow_text_only,
                    )
                except (FileNotFoundError, OSError, ValueError) as error:
                    raise ValueError(f"{spec.source_train}:{line_number}: {error}") from error
                if not identities:
                    counts["train_rows_without_identity"] += 1
                    if require_media_identity:
                        raise ValueError(
                            f"{spec.source_train}:{line_number}: no image, video, or lineage "
                            "identity could be derived"
                        )
                if any(identity.startswith("text:sha256:") for identity in identities):
                    counts["train_text_only_rows"] += 1
                eval_conflicts, exempt_eval_conflicts = partition_eval_conflicts(
                    store.lookup("eval_identity", identities),
                    spec.name,
                    eval_overlap_exempt_groups,
                )
                owner_conflicts = {
                    identity: owners
                    for identity, owners in store.lookup("train_owner", identities).items()
                    if owners != {spec.name}
                }
                if owner_conflicts:
                    counts["cross_dataset_train_overlap_rows"] += 1
                    global_stats["cross_dataset_train_overlap_rows"] += 1
                reason = ""
                conflicts: Mapping[str, set[str]] = {}
                if eval_conflicts:
                    reason = "eval_media_overlap"
                    conflicts = eval_conflicts
                elif owner_conflicts and deduplicate_cross_dataset_train:
                    reason = "higher_priority_train_overlap"
                    conflicts = owner_conflicts
                if reason:
                    counts[f"excluded_{reason}_rows"] += 1
                    global_stats["excluded_rows"] += 1
                    entry = {
                        "dataset": spec.name,
                        "source": str(spec.source_train),
                        "line_number": line_number,
                        "record_id": _record_id(record, line_number),
                        "reason": reason,
                        "conflicts": _conflict_payload(conflicts),
                    }
                    exclusions_stream.write(
                        json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    if len(examples) < args.max_report_examples:
                        examples.append(entry)
                else:
                    counts["retained_train_rows"] += 1
                    global_stats["retained_train_rows"] += 1
                    if owner_conflicts:
                        counts["retained_cross_dataset_train_overlap_rows"] += 1
                        global_stats["retained_cross_dataset_train_overlap_rows"] += 1
                    if exempt_eval_conflicts:
                        counts["retained_exempt_eval_overlap_rows"] += 1
                        global_stats["retained_exempt_eval_overlap_rows"] += 1
                    if not args.audit_only:
                        streams[spec.name].write(line)
                    store.add_owner("train_owner", identities, spec.name)
                for identity in identities:
                    identity_kinds[_identity_kind(identity)] += 1
                if args.progress_every and counts["source_train_rows"] % args.progress_every == 0:
                    print(
                        f"[train-filter] {spec.name} rows={counts['source_train_rows']:,} "
                        f"keep={counts['retained_train_rows']:,} "
                        f"drop={counts['source_train_rows'] - counts['retained_train_rows']:,}",
                        flush=True,
                    )
                if counts["source_train_rows"] % 10_000 == 0:
                    store.connection.commit()
            store.connection.commit()

        exclusions_stream.close()
        for stream in streams.values():
            stream.close()

        verification = {
            "performed": not args.audit_only,
            "rows": 0,
            "train_eval_overlap_rows": 0,
            "exempt_cross_dataset_train_eval_overlap_rows": 0,
            "cross_dataset_train_overlap_rows": 0,
            "status": "not_run" if args.audit_only else "complete",
        }
        if not args.audit_only:
            for spec in specs:
                temporary = temporary_outputs[spec.name]
                for line_number, _line, record in _iter_records_with_prefetch(
                    temporary,
                    hasher,
                    args.hash_workers,
                    args.hash_prefetch_rows,
                ):
                    verification["rows"] += 1
                    identities = resolver.resolve(
                        record,
                        temporary,
                        spec.name,
                        spec.identity_namespace,
                        spec.use_video_stem_identity,
                        spec.allow_text_only,
                    )
                    eval_conflicts, exempt_eval_conflicts = partition_eval_conflicts(
                        store.lookup("eval_identity", identities),
                        spec.name,
                        eval_overlap_exempt_groups,
                    )
                    if eval_conflicts:
                        verification["train_eval_overlap_rows"] += 1
                    elif exempt_eval_conflicts:
                        verification["exempt_cross_dataset_train_eval_overlap_rows"] += 1
                    owner_conflicts = {
                        identity: owners
                        for identity, owners in store.lookup("verify_owner", identities).items()
                        if owners != {spec.name}
                    }
                    if owner_conflicts:
                        verification["cross_dataset_train_overlap_rows"] += 1
                    else:
                        store.add_owner("verify_owner", identities, spec.name)
                    if verification["rows"] % 10_000 == 0:
                        store.connection.commit()
            if verification["train_eval_overlap_rows"] or (
                deduplicate_cross_dataset_train
                and verification["cross_dataset_train_overlap_rows"]
            ):
                verification["status"] = "failed"
                raise AssertionError(f"post-filter verification failed: {verification}")

        fingerprint_cache: dict[Path, dict[str, Any]] = {}

        def input_fingerprint(path: Path) -> dict[str, Any]:
            resolved = path.resolve()
            value = fingerprint_cache.get(resolved)
            if value is None:
                value = _file_fingerprint(resolved)
                fingerprint_cache[resolved] = value
            return value

        specs_by_name = {spec.name: spec for spec in specs}
        serializable_reports: dict[str, Any] = {}
        for name, report in dataset_reports.items():
            counts: Counter[str] = report["counts"]
            identity_kinds: Counter[str] = report["identity_observations"]
            spec = specs_by_name[name]
            fingerprints = {
                "source_train": input_fingerprint(spec.source_train),
                "eval": [input_fingerprint(path) for path in spec.eval_paths],
                "train": None,
                "prerequisite_report": (
                    input_fingerprint(spec.prerequisite_report)
                    if spec.prerequisite_report
                    else None
                ),
            }
            if not args.audit_only:
                fingerprints["train"] = _file_fingerprint(
                    temporary_outputs[name], logical_path=spec.clean_train
                )
            serializable_reports[name] = {
                "source_train": report["source_train"],
                "identity_namespace": report["identity_namespace"],
                "use_video_stem_identity": report["use_video_stem_identity"],
                "allow_text_only": report["allow_text_only"],
                "prerequisite_report": report["prerequisite_report"],
                "train": report["train"],
                "eval": report["eval"],
                "fingerprints": fingerprints,
                "counts": dict(sorted(counts.items())),
                "identity_observations": dict(sorted(identity_kinds.items())),
            }
        report_payload = {
            "schema_version": 2,
            "status": "audit_only" if args.audit_only else "complete",
            "manifest": str(manifest_path),
            "manifest_fingerprint": manifest_fingerprint,
            "training_priority": priority,
            "policy": {
                "eval_is_reserved_globally": True,
                "eval_overlap_exempt_groups": [
                    sorted(group) for group in eval_overlap_exempt_groups
                ],
                "require_media_identity": require_media_identity,
                "deduplicate_cross_dataset_train": deduplicate_cross_dataset_train,
                "cross_dataset_train_owner": (
                    "first dataset in training_priority"
                    if deduplicate_cross_dataset_train
                    else "retained by every source dataset"
                ),
                "within_dataset_repeated_media_rows": "retained",
                "cross_dataset_repeated_media_rows": (
                    "excluded regardless of source family"
                    if deduplicate_cross_dataset_train
                    else "retained; only non-exempt train-vs-eval media is excluded"
                ),
                "cross_dataset_eval_overlap": (
                    "exempt families keep shared images in each dataset train split"
                    if eval_overlap_exempt_groups
                    else "any eval media blocks every training dataset"
                ),
                "media_file_mutability": (
                    "media files must remain unchanged for the duration of one run; "
                    "cross-run cache reuse validates size and mtime"
                ),
                "lineage_identity_namespace": (
                    "dataset-specific by default; shared only through explicit identity_namespace"
                ),
                "image_identities": ["SHA-256", "COCO image ID", "canonical path or URL"],
                "video_identities": ["canonical path or URL", "video filename ID", "explicit episode/video IDs"],
                "robo2vlm_identity": "record.id without trailing _qN",
                "text_only_identity": (
                    "opt-in canonical messages SHA-256; input media placeholders and media "
                    "content blocks without media fail"
                ),
            },
            "datasets": serializable_reports,
            "totals": dict(sorted(global_stats.items())),
            "verification": verification,
            "hash_cache": dict(sorted(hasher.stats.items())),
            "hash_concurrency": {
                "workers": args.hash_workers,
                "prefetch_rows": args.hash_prefetch_rows,
                "sqlite_writes": "main thread only",
            },
            "exclusion_log": str(exclusions_output),
            "exclusion_examples": examples,
        }
        with report_temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report_payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        if not args.audit_only:
            for spec in specs:
                os.replace(temporary_outputs[spec.name], spec.clean_train)
        os.replace(exclusions_temporary, exclusions_output)
        os.replace(report_temporary, report_output)
        return report_payload
    except BaseException:
        if not exclusions_stream.closed:
            exclusions_stream.close()
        for stream in streams.values():
            if not stream.closed:
                stream.close()
        for path in [*temporary_outputs.values(), exclusions_temporary, report_temporary]:
            if path.exists():
                path.unlink()
        raise
    finally:
        hasher.close()
        store.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    totals = report["totals"]
    print(
        f"[done] source={totals.get('source_train_rows', 0):,} "
        f"retained={totals.get('retained_train_rows', 0):,} "
        f"excluded={totals.get('excluded_rows', 0):,}"
    )
    print(f"[verify] {report['verification']['status']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except (FileExistsError, FileNotFoundError, OSError, ValueError, AssertionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
