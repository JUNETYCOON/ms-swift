"""Read-only schema profiler for common dataset container formats."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import re
import sqlite3
import struct
import tarfile
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

try:
    import google_crc32c
except ImportError:  # The table-driven fallback keeps profiling dependency-optional.
    google_crc32c = None


TYPE_NAMES = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}


STRUCTURED_SUFFIXES = {
    ".json", ".jsonl", ".parquet", ".arrow", ".feather", ".csv", ".tsv",
    ".npy", ".npz", ".h5", ".hdf5", ".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz",
    ".tfrecord", ".tfrecords", ".mcap", ".bag", ".db3",
}

TFRECORD_SHARD_PATTERN = re.compile(
    r"\.(?:tfrecord|tfrecords)(?:[-._]\d+(?:-of-\d+)?)?$", re.IGNORECASE
)
WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")

PRIORITY_CONTAINER_FORMATS = {"parquet", "arrow", "feather"}

MACOS_METADATA_DIRECTORIES = {
    "__macosx", ".appledouble", ".documentrevisions-v100", ".fseventsd", ".spotlight-v100", ".trashes",
}

KNOWN_METADATA_FILENAMES = {".ds_store", "desktop.ini", "ehthumbs.db", "icon\r", "thumbs.db"}
CACHE_DIRECTORIES = {".cache"}

# These are ordinary semantic fields found at the root of common dataset files.
# Seeing one is strong evidence that the root is a record/document, not an ID map.
SEMANTIC_ROOT_KEYS = {
    "annotations", "assets", "audios", "categories", "config", "conversations", "data", "dataset",
    "entities", "episode", "examples", "features", "images", "info", "licenses", "manifest",
    "messages", "metadata", "objects", "records", "relations", "schema", "splits", "test", "tracks",
    "train", "val", "validation", "version", "videos",
}

ID_KEY_PATTERN = re.compile(
    r"(?:^\d+$|^[0-9a-f]{16,}$|^[0-9a-f]{8}-[0-9a-f-]{27,}$|(?:^|[_-])\d+(?:$|[_-]))",
    re.IGNORECASE,
)

MAX_MAPPING_KEYS = 64
SCHEMA_FINGERPRINT_OMITTED_KEYS = {"array_length", "examples", "observations"}
MAX_TFRECORD_RECORD_BYTES = 256 * 1024 * 1024
MAX_TFRECORD_SAMPLE_BYTES = 64 * 1024 * 1024
TFRECORD_READ_CHUNK_BYTES = 1024 * 1024
MAX_ARCHIVE_SCHEMA_MEMBERS = 64
MAX_ARCHIVE_MEMBER_SAMPLE_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_TOTAL_SAMPLE_BYTES = 64 * 1024 * 1024
MAX_DB3_TABLES = 64
MAX_DB3_COLUMNS = 128
MAX_DB3_TOPICS = 1000


def _reject_nonfinite(value: str) -> None:
    raise ValueError("non-finite JSON number {!r} is not allowed".format(value))


def _metadata_exclusion_reason(relative_path: Path) -> Optional[str]:
    directory_parts = {part.casefold() for part in relative_path.parts[:-1]}
    if directory_parts & CACHE_DIRECTORIES:
        return "cache_directory"
    if directory_parts & MACOS_METADATA_DIRECTORIES:
        return "macos_metadata_directory"
    name = relative_path.name
    if name.startswith("._"):
        return "appledouble_sidecar"
    if name.casefold() in KNOWN_METADATA_FILENAMES:
        return "os_metadata_file"
    return None


def _normalize_exclude_patterns(values: Sequence[str]) -> List[str]:
    patterns: List[str] = []
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("exclude patterns must be non-empty strings")
        pattern = raw.strip().replace("\\", "/")
        if pattern.startswith("/") or WINDOWS_DRIVE_PATH.match(pattern):
            raise ValueError("exclude patterns must be relative to the profile root: {!r}".format(raw))
        if ".." in Path(pattern).parts:
            raise ValueError("exclude patterns must not escape the profile root: {!r}".format(raw))
        if pattern not in patterns:
            patterns.append(pattern)
    return patterns


def _matching_exclude_pattern(relative_path: Path, patterns: Sequence[str]) -> Optional[str]:
    value = relative_path.as_posix()
    for pattern in patterns:
        if fnmatch.fnmatchcase(value, pattern):
            return pattern
    return None


def _container_format(path: Path) -> str:
    lower = path.name.casefold()
    if TFRECORD_SHARD_PATTERN.search(lower):
        return "tfrecord"
    if lower.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        return "tar"
    return path.suffix.casefold().lstrip(".") or "<none>"


def _is_structured_file(path: Path) -> bool:
    lower = path.name.casefold()
    return (
        bool(TFRECORD_SHARD_PATTERN.search(lower))
        or path.suffix.casefold() in STRUCTURED_SUFFIXES
        or lower.endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
    )


def _shape_signature(value: Any) -> Any:
    if isinstance(value, Mapping):
        return (
            "object",
            tuple(
                sorted(
                    (str(key), TYPE_NAMES.get(type(child), type(child).__name__))
                    for key, child in _bounded_mapping_items(value)
                )
            ),
        )
    if isinstance(value, list):
        return (
            "array",
            tuple(sorted({TYPE_NAMES.get(type(item), type(item).__name__) for item in value[:16]})),
        )
    return TYPE_NAMES.get(type(value), type(value).__name__)


def _looks_like_dynamic_key(key: str) -> bool:
    if ID_KEY_PATTERN.search(key):
        return True
    if "/" in key or "\\" in key:
        return True
    if Path(key).suffix.casefold() in {
        ".avi", ".bmp", ".gif", ".jpeg", ".jpg", ".mkv", ".mov", ".mp4", ".png", ".webm",
    }:
        return True
    return len(key) >= 4 and any(character.isdigit() for character in key)


def _looks_like_strong_dynamic_key(key: str) -> bool:
    if ID_KEY_PATTERN.search(key) or "/" in key or "\\" in key:
        return True
    return Path(key).suffix.casefold() in {
        ".avi", ".bmp", ".gif", ".jpeg", ".jpg", ".mkv", ".mov", ".mp4", ".png", ".webm",
    }


def _looks_like_object_map(items: List[Any]) -> bool:
    """Conservatively distinguish a root ID->record map from a document object."""

    if not items:
        return False
    keys = [str(key) for key, _ in items]
    if {key.casefold() for key in keys} & SEMANTIC_ROOT_KEYS:
        return False
    if len(items) == 1:
        key, value = items[0]
        return _looks_like_strong_dynamic_key(str(key)) and isinstance(value, (Mapping, list))

    dynamic_keys = sum(_looks_like_dynamic_key(key) for key in keys)
    if dynamic_keys * 5 >= len(keys) * 4:
        return True

    values = [value for _, value in items]
    value_kinds = Counter(TYPE_NAMES.get(type(value), type(value).__name__) for value in values)
    dominant_kind, dominant_kind_count = value_kinds.most_common(1)[0]
    if dominant_kind_count * 5 < len(values) * 4:
        return False

    signatures = Counter(_shape_signature(value) for value in values)
    dominant_shape_count = signatures.most_common(1)[0][1]

    # Repeated object schemas are the common ID-map case, including alphabetic
    # IDs. A shared child field also tolerates optional record fields.
    if dominant_kind == "object":
        object_values = [value for value in values if isinstance(value, Mapping)]
        common_fields = (
            {str(key) for key, _ in _bounded_mapping_items(object_values[0])}
            if object_values else set()
        )
        for value in object_values[1:]:
            common_fields.intersection_update(str(key) for key, _ in _bounded_mapping_items(value))
        if dynamic_keys * 2 >= len(keys) or (len(items) >= 3 and common_fields):
            return True
        if len(items) >= 3 and dominant_shape_count * 5 >= len(values) * 4:
            return True
        if len(items) >= 4:
            return True
    if dynamic_keys * 2 >= len(keys):
        return True
    return len(items) >= 4 and dominant_shape_count * 5 >= len(values) * 4


def _bounded_mapping_items(value: Mapping[Any, Any]) -> List[Any]:
    """Return a deterministic, breadth-bounded view of a mapping."""

    items = list(islice(value.items(), MAX_MAPPING_KEYS))
    return sorted(items, key=lambda item: (str(item[0]), type(item[0]).__name__))


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
        if (
            self.max_examples > 0
            and value_type not in ("object", "array")
            and len(self.examples[path]) < self.max_examples
        ):
            example = value
            if isinstance(example, str) and len(example) > 160:
                example = example[:157] + "..."
            if example not in self.examples[path]:
                self.examples[path].append(example)
        if depth >= self.max_depth:
            return
        if isinstance(value, Mapping):
            items = _bounded_mapping_items(value)
            if _looks_like_object_map(items):
                for _, child in items:
                    self.add(child, path + ".*", depth + 1)
            else:
                for key, child in items:
                    child_path = (
                        path + ".*"
                        if _looks_like_strong_dynamic_key(str(key))
                        else "{}.{}".format(path, key)
                    )
                    self.add(child, child_path, depth + 1)
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


def _canonical_schema_structure(value: Any, parent_key: Optional[str] = None) -> Any:
    """Remove sample-dependent values before computing a schema-family hash."""

    if isinstance(value, Mapping):
        if parent_key == "types":
            return sorted(str(key) for key, _ in _bounded_mapping_items(value))
        return {
            str(key): _canonical_schema_structure(child, str(key))
            for key, child in _bounded_mapping_items(value)
            if str(key) not in SCHEMA_FINGERPRINT_OMITTED_KEYS
        }
    if isinstance(value, list):
        return [_canonical_schema_structure(child) for child in value]
    return value


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


def _profile_json_value(value: Any, sample_rows: int, max_depth: int, max_examples: int) -> Dict[str, Any]:
    profiler = ShapeProfiler(max_depth=max_depth, max_examples=max_examples)
    if isinstance(value, list):
        for item in value[:sample_rows]:
            profiler.add(item)
        total = len(value)
        sampled = min(total, sample_rows)
        root_kind = "array"
    elif isinstance(value, Mapping):
        items = _bounded_mapping_items(value)
        if _looks_like_object_map(items):
            root_kind = "object_map"
            sampled = min(len(items), sample_rows)
            for _, item in items[:sampled]:
                profiler.add(item, "$.*")
            total = len(value)
        else:
            root_kind = "object"
            profiler.add(value)
            total = 1
            sampled = 1
    else:
        profiler.add(value)
        total = 1
        sampled = 1
        root_kind = "scalar"
    return {
        "kind": "json",
        "status": "ok",
        "root_kind": root_kind,
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
        with path.open("rb") as stream:
            first_event = next(ijson.parse(stream), None)
        if first_event is None:
            return {"kind": "json", "status": "invalid", "error": "empty JSON file"}
        root_event = first_event[1]
        if root_event == "start_array":
            sampled = 0
            with path.open("rb") as stream:
                iterator = ijson.items(stream, "item")
                for value in iterator:
                    profiler.add(value)
                    sampled += 1
                    if sampled >= sample_rows:
                        break
            root_kind = "array"
            total_records = None
        elif root_event == "start_map":
            probe_items = []
            probe_limit = min(max(sample_rows, 8), MAX_MAPPING_KEYS)
            with path.open("rb") as stream:
                iterator = ijson.kvitems(stream, "")
                for key, value in iterator:
                    probe_items.append((key, value))
                    if len(probe_items) >= probe_limit:
                        break
            if _looks_like_object_map(probe_items):
                root_kind = "object_map"
                sampled = min(len(probe_items), sample_rows)
                for _, value in probe_items[:sampled]:
                    profiler.add(value, "$.*")
                total_records = None
            else:
                root_kind = "object"
                profiler.add(dict(probe_items))
                sampled = 1
                total_records = 1
        else:
            return {
                "kind": "json",
                "status": "skipped",
                "root_kind": "scalar",
                "error": "large scalar JSON roots are not profiled",
            }
        return {
            "kind": "json",
            "status": "ok",
            "root_kind": root_kind,
            "sampled_records": sampled,
            "total_records": total_records,
            "schema": profiler.result(),
        }
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    return _profile_json_value(value, sample_rows, max_depth, max_examples)


def _make_crc32c_table() -> Any:
    polynomial = 0x82F63B78
    table = []
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = (crc >> 1) ^ polynomial if crc & 1 else crc >> 1
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)


CRC32C_TABLE = _make_crc32c_table()


def _crc32c(data: Any, crc: int = 0) -> int:
    if google_crc32c is not None:
        return int(google_crc32c.extend(crc, data))
    crc ^= 0xFFFFFFFF
    for byte in memoryview(data).cast("B"):
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(data: Any) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | ((crc << 17) & 0xFFFFFFFF)) + 0xA282EAD8) & 0xFFFFFFFF


def _decode_varint(data: Any, offset: int) -> Any:
    view = memoryview(data).cast("B")
    value = 0
    shift = 0
    while offset < len(view) and shift < 70:
        byte = view[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    if offset >= len(view):
        raise ValueError("truncated protobuf varint")
    raise ValueError("protobuf varint exceeds 10 bytes")


def _parse_wire_fields(data: Any) -> Dict[int, List[Any]]:
    view = memoryview(data).cast("B")
    fields: DefaultDict[int, List[Any]] = defaultdict(list)
    offset = 0
    while offset < len(view):
        tag, offset = _decode_varint(view, offset)
        field_number = tag >> 3
        wire_type = tag & 7
        if field_number == 0:
            raise ValueError("protobuf field number 0 is invalid")
        if wire_type == 0:
            value, offset = _decode_varint(view, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(view):
                raise ValueError("truncated protobuf fixed64 field")
            value = view[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _decode_varint(view, offset)
            end = offset + length
            if end > len(view):
                raise ValueError("truncated protobuf length-delimited field")
            value = view[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(view):
                raise ValueError("truncated protobuf fixed32 field")
            value = view[offset:end]
            offset = end
        else:
            raise ValueError("unsupported protobuf wire type {}".format(wire_type))
        fields[field_number].append((wire_type, value))
    return dict(fields)


def _length_delimited_values(fields: Mapping[int, List[Any]], field_number: int) -> List[Any]:
    values = []
    for wire_type, value in fields.get(field_number, []):
        if wire_type != 2:
            raise ValueError("protobuf field {} has unexpected wire type {}".format(field_number, wire_type))
        values.append(value)
    return values


def _parse_bytes_list(data: Any) -> int:
    fields = _parse_wire_fields(data)
    return len(_length_delimited_values(fields, 1))


def _parse_float_list(data: Any) -> int:
    fields = _parse_wire_fields(data)
    count = 0
    for wire_type, value in fields.get(1, []):
        if wire_type == 2:
            if len(value) % 4:
                raise ValueError("packed FloatList payload is not aligned to 4 bytes")
            count += len(value) // 4
        elif wire_type == 5:
            count += 1
        else:
            raise ValueError("FloatList.value has unexpected wire type {}".format(wire_type))
    return count


def _parse_int64_list(data: Any) -> int:
    fields = _parse_wire_fields(data)
    count = 0
    for wire_type, value in fields.get(1, []):
        if wire_type == 0:
            count += 1
        elif wire_type == 2:
            offset = 0
            while offset < len(value):
                _, offset = _decode_varint(value, offset)
                count += 1
        else:
            raise ValueError("Int64List.value has unexpected wire type {}".format(wire_type))
    return count


def _parse_feature(data: Any) -> Any:
    fields = _parse_wire_fields(data)
    present = []
    for field_number, kind, parser in (
        (1, "bytes", _parse_bytes_list),
        (2, "float", _parse_float_list),
        (3, "int64", _parse_int64_list),
    ):
        payloads = _length_delimited_values(fields, field_number)
        if payloads:
            present.append((kind, parser(payloads[-1])))
    if len(present) > 1:
        raise ValueError("Feature contains more than one oneof value")
    return present[0] if present else ("unset", 0)


def _parse_feature_map_entry(data: Any) -> Any:
    fields = _parse_wire_fields(data)
    keys = _length_delimited_values(fields, 1)
    values = _length_delimited_values(fields, 2)
    if not keys or not values:
        raise ValueError("feature map entry is missing key or value")
    key = bytes(keys[-1]).decode("utf-8")
    return key, _parse_feature(values[-1])


def _parse_features(data: Any) -> Dict[str, Any]:
    fields = _parse_wire_fields(data)
    result = {}
    for entry in _length_delimited_values(fields, 1):
        key, value = _parse_feature_map_entry(entry)
        result[key] = value
    return result


def _parse_feature_list(data: Any) -> List[Any]:
    fields = _parse_wire_fields(data)
    return [_parse_feature(value) for value in _length_delimited_values(fields, 1)]


def _parse_feature_list_map_entry(data: Any) -> Any:
    fields = _parse_wire_fields(data)
    keys = _length_delimited_values(fields, 1)
    values = _length_delimited_values(fields, 2)
    if not keys or not values:
        raise ValueError("feature-list map entry is missing key or value")
    key = bytes(keys[-1]).decode("utf-8")
    return key, _parse_feature_list(values[-1])


def _parse_feature_lists(data: Any) -> Dict[str, List[Any]]:
    fields = _parse_wire_fields(data)
    result = {}
    for entry in _length_delimited_values(fields, 1):
        key, value = _parse_feature_list_map_entry(entry)
        result[key] = value
    return result


def _parse_tf_example_record(data: Any) -> Any:
    fields = _parse_wire_fields(data)
    if not fields:
        # Empty Example and SequenceExample messages have identical wire
        # encodings; use Example as the deterministic default.
        return "Example", {"features": {}}
    features_payloads = _length_delimited_values(fields, 1)
    feature_lists_payloads = _length_delimited_values(fields, 2)
    if feature_lists_payloads:
        context = _parse_features(features_payloads[-1]) if features_payloads else {}
        return "SequenceExample", {
            "context": context,
            "feature_lists": _parse_feature_lists(feature_lists_payloads[-1]),
        }
    if features_payloads:
        return "Example", {"features": _parse_features(features_payloads[-1])}
    raise ValueError("record is neither a serialized Example nor SequenceExample")


def _new_feature_statistics() -> Dict[str, Any]:
    return {"types": Counter(), "observations": 0, "lengths": []}


def _add_feature_statistics(statistics: Dict[str, Any], feature: Any) -> None:
    kind, length = feature
    statistics["types"][kind] += 1
    statistics["observations"] += 1
    statistics["lengths"].append(length)


def _render_feature_statistics(
    statistics: Dict[str, Any], array_lengths: Optional[List[int]] = None
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "types": dict(sorted(statistics["types"].items())),
        "observations": statistics["observations"],
    }
    lengths = statistics["lengths"] if array_lengths is None else array_lengths
    if lengths:
        result["array_length"] = {"min": min(lengths), "max": max(lengths)}
    return result


class _TFRecordSchemaProfiler:
    def __init__(self) -> None:
        self.record_types: Counter = Counter()
        self.example_features: DefaultDict[str, Dict[str, Any]] = defaultdict(_new_feature_statistics)
        self.context_features: DefaultDict[str, Dict[str, Any]] = defaultdict(_new_feature_statistics)
        self.feature_lists: DefaultDict[str, Dict[str, Any]] = defaultdict(_new_feature_statistics)
        self.feature_list_lengths: DefaultDict[str, List[int]] = defaultdict(list)

    def add(self, record_type: str, value: Mapping[str, Any]) -> None:
        self.record_types[record_type] += 1
        if record_type == "Example":
            for name, feature in _bounded_mapping_items(value["features"]):
                _add_feature_statistics(self.example_features[str(name)], feature)
            return
        for name, feature in _bounded_mapping_items(value["context"]):
            _add_feature_statistics(self.context_features[str(name)], feature)
        for name, features in _bounded_mapping_items(value["feature_lists"]):
            name = str(name)
            self.feature_list_lengths[name].append(len(features))
            for feature in features:
                _add_feature_statistics(self.feature_lists[name], feature)

    @staticmethod
    def _render_mapping(value: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
        return {
            key: _render_feature_statistics(value[key])
            for key in sorted(value)[:MAX_MAPPING_KEYS]
        }

    def result(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"record_types": sorted(self.record_types)}
        if self.example_features:
            schema["example"] = {"features": self._render_mapping(self.example_features)}
        if self.context_features or self.feature_lists:
            rendered_lists = {}
            for key in sorted(self.feature_lists)[:MAX_MAPPING_KEYS]:
                rendered_lists[key] = _render_feature_statistics(
                    self.feature_lists[key], self.feature_list_lengths[key]
                )
            schema["sequence_example"] = {
                "context": self._render_mapping(self.context_features),
                "feature_lists": rendered_lists,
            }
        return schema


def _read_tfrecord_payload(stream: Any, length: int, capture: bool) -> Any:
    remaining = length
    crc = 0
    payload = bytearray() if capture else None
    while remaining:
        chunk = stream.read(min(remaining, TFRECORD_READ_CHUNK_BYTES))
        if not chunk:
            raise EOFError("truncated TFRecord payload")
        crc = _crc32c(chunk, crc)
        if payload is not None:
            payload.extend(chunk)
        remaining -= len(chunk)
    return (bytes(payload) if payload is not None else None), crc


def _profile_tfrecord(path: Path, sample_rows: int) -> Dict[str, Any]:
    profiler = _TFRecordSchemaProfiler()
    total = 0
    sampled = 0
    sampled_bytes = 0
    with path.open("rb") as stream:
        while True:
            frame_offset = stream.tell()
            header = stream.read(12)
            if not header:
                break
            if len(header) != 12:
                return {
                    "kind": "tfrecord",
                    "status": "invalid",
                    "error": "record {} at byte {} has a truncated frame header".format(total + 1, frame_offset),
                    "total_records_seen": total,
                }
            length_bytes = header[:8]
            length = struct.unpack("<Q", length_bytes)[0]
            expected_length_crc = struct.unpack("<I", header[8:])[0]
            if expected_length_crc != _masked_crc32c(length_bytes):
                return {
                    "kind": "tfrecord",
                    "status": "invalid",
                    "error": "record {} at byte {} has an invalid length CRC32C".format(total + 1, frame_offset),
                    "total_records_seen": total,
                }
            if length > MAX_TFRECORD_RECORD_BYTES:
                return {
                    "kind": "tfrecord",
                    "status": "invalid",
                    "error": "record {} length {} exceeds the {} byte safety limit".format(
                        total + 1, length, MAX_TFRECORD_RECORD_BYTES
                    ),
                    "total_records_seen": total,
                }
            capture = sampled < sample_rows and sampled_bytes + length <= MAX_TFRECORD_SAMPLE_BYTES
            try:
                payload, data_crc = _read_tfrecord_payload(stream, length, capture)
            except EOFError as error:
                return {
                    "kind": "tfrecord",
                    "status": "invalid",
                    "error": "record {} at byte {}: {}".format(total + 1, frame_offset, error),
                    "total_records_seen": total,
                }
            trailer = stream.read(4)
            if len(trailer) != 4:
                return {
                    "kind": "tfrecord",
                    "status": "invalid",
                    "error": "record {} at byte {} has a truncated data CRC32C".format(total + 1, frame_offset),
                    "total_records_seen": total,
                }
            expected_data_crc = struct.unpack("<I", trailer)[0]
            masked_data_crc = (((data_crc >> 15) | ((data_crc << 17) & 0xFFFFFFFF)) + 0xA282EAD8) & 0xFFFFFFFF
            if expected_data_crc != masked_data_crc:
                return {
                    "kind": "tfrecord",
                    "status": "invalid",
                    "error": "record {} at byte {} has an invalid data CRC32C".format(total + 1, frame_offset),
                    "total_records_seen": total,
                }
            total += 1
            if payload is not None:
                try:
                    record_type, value = _parse_tf_example_record(payload)
                except (UnicodeDecodeError, ValueError) as error:
                    return {
                        "kind": "tfrecord",
                        "status": "invalid",
                        "error": "record {} protobuf: {}".format(total, error),
                        "total_records_seen": total,
                    }
                profiler.add(record_type, value)
                sampled += 1
                sampled_bytes += length
    return {
        "kind": "tfrecord",
        "status": "ok",
        "total_records": total,
        "sampled_records": sampled,
        "sampled_payload_bytes": sampled_bytes,
        "sample_truncated": sampled < total,
        "record_type_counts": dict(sorted(profiler.record_types.items())),
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


def _profile_embedded_json(
    stream: Any,
    declared_size: int,
    sample_rows: int,
    max_depth: int,
    max_examples: int,
    byte_limit: int,
) -> Dict[str, Any]:
    if declared_size > byte_limit:
        return {
            "kind": "json",
            "status": "skipped",
            "error": "archive member exceeds the {} byte JSON sampling limit".format(byte_limit),
            "_bytes_read": 0,
        }
    payload = stream.read(byte_limit + 1)
    bytes_read = min(len(payload), byte_limit)
    if len(payload) > byte_limit:
        return {
            "kind": "json",
            "status": "skipped",
            "error": "archive member exceeds the {} byte JSON sampling limit".format(byte_limit),
            "_bytes_read": bytes_read,
        }
    try:
        value = json.loads(payload.decode("utf-8-sig"), parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        return {"kind": "json", "status": "invalid", "error": str(error), "_bytes_read": bytes_read}
    result = _profile_json_value(value, sample_rows, max_depth, max_examples)
    result["_bytes_read"] = bytes_read
    return result


def _profile_embedded_jsonl(
    stream: Any,
    sample_rows: int,
    max_depth: int,
    max_examples: int,
    byte_limit: int,
) -> Dict[str, Any]:
    profiler = ShapeProfiler(max_depth=max_depth, max_examples=max_examples)
    sampled = 0
    bytes_read = 0
    truncated = False
    line_number = 0
    while sampled < sample_rows and bytes_read < byte_limit:
        remaining = byte_limit - bytes_read
        raw_line = stream.readline(remaining + 1)
        if not raw_line:
            break
        line_number += 1
        bytes_read += len(raw_line)
        if bytes_read > byte_limit:
            bytes_read = byte_limit
            truncated = True
            break
        if not raw_line.strip():
            continue
        try:
            encoding = "utf-8-sig" if line_number == 1 else "utf-8"
            value = json.loads(raw_line.decode(encoding), parse_constant=_reject_nonfinite)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            return {
                "kind": "jsonl",
                "status": "invalid",
                "error": "line {}: {}".format(line_number, error),
                "sampled_records": sampled,
                "_bytes_read": bytes_read,
            }
        profiler.add(value)
        sampled += 1
    if sampled >= sample_rows:
        truncated = True
    return {
        "kind": "jsonl",
        "status": "ok",
        "sampled_records": sampled,
        "sample_truncated": truncated,
        "schema": profiler.result(),
        "_bytes_read": bytes_read,
    }


def _profile_archive(
    path: Path,
    max_members: int,
    sample_rows: int = 100,
    max_depth: int = 8,
    max_examples: int = 0,
) -> Dict[str, Any]:
    suffixes: Counter = Counter()
    members = 0
    truncated = False
    embedded_reports: List[Dict[str, Any]] = []
    embedded_bytes = 0

    def profile_member(name: str, declared_size: int, stream: Any) -> None:
        nonlocal embedded_bytes
        if len(embedded_reports) >= MAX_ARCHIVE_SCHEMA_MEMBERS:
            return
        if _metadata_exclusion_reason(Path(name)) is not None:
            return
        suffix = Path(name).suffix.casefold()
        if suffix not in (".json", ".jsonl"):
            return
        remaining_budget = MAX_ARCHIVE_TOTAL_SAMPLE_BYTES - embedded_bytes
        if remaining_budget <= 0:
            return
        byte_limit = min(MAX_ARCHIVE_MEMBER_SAMPLE_BYTES, remaining_budget)
        try:
            if suffix == ".json":
                report = _profile_embedded_json(
                    stream, declared_size, sample_rows, max_depth, max_examples, byte_limit
                )
            else:
                report = _profile_embedded_jsonl(
                    stream, sample_rows, max_depth, max_examples, byte_limit
                )
        except Exception as error:
            report = {
                "kind": suffix.lstrip("."),
                "status": "error",
                "error": "{}: {}".format(type(error).__name__, error),
                "_bytes_read": 0,
            }
        embedded_bytes += int(report.pop("_bytes_read", 0))
        report["path"] = name
        embedded_reports.append(report)

    if zipfile.is_zipfile(path):
        kind = "zip"
        with zipfile.ZipFile(path) as archive:
            files = [item for item in archive.infolist() if not item.is_dir()]
            truncated = len(files) > max_members
            for item in files[:max_members]:
                members += 1
                suffixes[Path(item.filename).suffix.lower() or "<none>"] += 1
                if Path(item.filename).suffix.casefold() in (".json", ".jsonl"):
                    try:
                        with archive.open(item, "r") as stream:
                            profile_member(item.filename, item.file_size, stream)
                    except Exception as error:
                        if len(embedded_reports) < MAX_ARCHIVE_SCHEMA_MEMBERS:
                            embedded_reports.append(
                                {
                                    "path": item.filename,
                                    "kind": Path(item.filename).suffix.casefold().lstrip("."),
                                    "status": "error",
                                    "error": "{}: {}".format(type(error).__name__, error),
                                }
                            )
    else:
        kind = "tar"
        with tarfile.open(path, "r:*") as archive:
            for item in archive:
                if not item.isfile():
                    continue
                if members >= max_members:
                    truncated = True
                    break
                members += 1
                suffixes[Path(item.name).suffix.lower() or "<none>"] += 1
                if Path(item.name).suffix.casefold() in (".json", ".jsonl"):
                    stream = archive.extractfile(item)
                    if stream is not None:
                        with stream:
                            profile_member(item.name, item.size, stream)
    embedded_summary = _summarize_file_reports(embedded_reports)
    archive_schema = {
        "member_suffixes": sorted(suffixes),
        "embedded_schema_families": [
            {
                key: value
                for key, value in family.items()
                if key not in ("files", "records", "examples")
            }
            for family in embedded_summary["schema_families"]
        ],
    }
    return {
        "kind": kind,
        "status": "ok",
        "schema": archive_schema,
        "members_scanned": members,
        "truncated": truncated,
        "member_suffixes": dict(sorted(suffixes.items())),
        "embedded_profiled_members": len(embedded_reports),
        "embedded_sampled_bytes": embedded_bytes,
        "embedded_status_counts": embedded_summary["status_counts"],
        "embedded_schema_families": embedded_summary["schema_families"],
    }


def _profile_db3(path: Path) -> Dict[str, Any]:
    uri = path.as_uri() + "?mode=ro&immutable=1"
    tables = []
    topics = []
    tables_truncated = False
    topics_truncated = False
    topics_total = 0
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT ?",
            (MAX_DB3_TABLES + 1,),
        ).fetchall()
        tables_truncated = len(rows) > MAX_DB3_TABLES
        table_names = [str(row[0]) for row in rows[:MAX_DB3_TABLES]]
        table_columns: Dict[str, List[str]] = {}
        for table_name in table_names:
            quoted = table_name.replace('"', '""')
            column_rows = connection.execute('PRAGMA table_info("{}")'.format(quoted)).fetchall()
            table_columns[table_name] = [str(row[1]) for row in column_rows]
            tables.append(
                {
                    "name": table_name,
                    "columns": [
                        {
                            "name": str(row[1]),
                            "type": str(row[2] or ""),
                            "not_null": bool(row[3]),
                            "primary_key": bool(row[5]),
                        }
                        for row in column_rows[:MAX_DB3_COLUMNS]
                    ],
                    "columns_truncated": len(column_rows) > MAX_DB3_COLUMNS,
                }
            )
        if "topics" in table_columns:
            available = set(table_columns["topics"])
            selected_columns = [
                name for name in ("id", "name", "type", "serialization_format") if name in available
            ]
            topics_total = int(connection.execute("SELECT COUNT(*) FROM topics").fetchone()[0])
            topics_truncated = topics_total > MAX_DB3_TOPICS
            if selected_columns:
                projection = ", ".join('"{}"'.format(name.replace('"', '""')) for name in selected_columns)
                order_column = "id" if "id" in available else selected_columns[0]
                topic_rows = connection.execute(
                    'SELECT {} FROM topics ORDER BY "{}" LIMIT ?'.format(
                        projection, order_column.replace('"', '""')
                    ),
                    (MAX_DB3_TOPICS,),
                ).fetchall()
                for row in topic_rows:
                    topics.append(
                        {
                            name: (value if isinstance(value, (int, float, str, type(None))) else str(value))
                            for name, value in zip(selected_columns, row)
                        }
                    )
    return {
        "kind": "db3",
        "status": "ok",
        "schema": {"tables": tables},
        "tables_truncated": tables_truncated,
        "topics": topics,
        "topics_total": topics_total,
        "topics_truncated": topics_truncated,
    }


def _profile_mcap(path: Path) -> Dict[str, Any]:
    try:
        from mcap.reader import make_reader
    except ImportError:
        return {"kind": "mcap", "status": "skipped", "error": "optional mcap is not installed"}
    with path.open("rb") as stream:
        summary = make_reader(stream).get_summary()
    if summary is None:
        return {
            "kind": "mcap",
            "status": "skipped",
            "error": "MCAP summary is absent; payload scanning is disabled",
        }
    schema_values = list((getattr(summary, "schemas", {}) or {}).values())
    channel_values = list((getattr(summary, "channels", {}) or {}).values())
    message_schemas = sorted(
        {
            (str(getattr(value, "name", "")), str(getattr(value, "encoding", "")))
            for value in schema_values
        }
    )[:MAX_MAPPING_KEYS]
    channels = [
        {
            "topic": str(getattr(value, "topic", "")),
            "message_encoding": str(getattr(value, "message_encoding", "")),
            "schema_id": int(getattr(value, "schema_id", 0)),
        }
        for value in sorted(
            channel_values,
            key=lambda value: (str(getattr(value, "topic", "")), int(getattr(value, "id", 0))),
        )[:MAX_MAPPING_KEYS]
    ]
    statistics = getattr(summary, "statistics", None)
    total_records = int(getattr(statistics, "message_count", 0)) if statistics is not None else None
    return {
        "kind": "mcap",
        "status": "ok",
        "total_records": total_records,
        "schema": {
            "message_schemas": [
                {"name": name, "encoding": encoding} for name, encoding in message_schemas
            ],
            "message_encodings": sorted({channel["message_encoding"] for channel in channels}),
        },
        "channels": channels,
        "channels_truncated": len(channel_values) > MAX_MAPPING_KEYS,
    }


def _profile_bag(path: Path) -> Dict[str, Any]:
    try:
        from rosbags.rosbag1 import Reader
    except ImportError:
        return {"kind": "bag", "status": "skipped", "error": "optional rosbags is not installed"}
    with Reader(path) as reader:
        values = list(reader.connections)
        connections = [
            {
                "topic": str(getattr(value, "topic", "")),
                "message_type": str(getattr(value, "msgtype", "")),
                "message_count": int(getattr(value, "msgcount", 0)),
            }
            for value in sorted(
                values,
                key=lambda value: (str(getattr(value, "topic", "")), str(getattr(value, "msgtype", ""))),
            )[:MAX_MAPPING_KEYS]
        ]
        total_records = sum(int(getattr(value, "msgcount", 0)) for value in values)
    return {
        "kind": "bag",
        "status": "ok",
        "total_records": total_records,
        "schema": {"message_types": sorted({item["message_type"] for item in connections})},
        "connections": connections,
        "connections_truncated": len(values) > MAX_MAPPING_KEYS,
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
    result: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "kind": _container_format(path),
    }
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
        elif TFRECORD_SHARD_PATTERN.search(lower):
            result.update(_profile_tfrecord(path, sample_rows))
        elif suffix == ".db3":
            result.update(_profile_db3(path))
        elif suffix == ".mcap":
            result.update(_profile_mcap(path))
        elif suffix == ".bag":
            result.update(_profile_bag(path))
        elif suffix == ".zip" or lower.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            result.update(
                _profile_archive(
                    path,
                    max_archive_members,
                    sample_rows,
                    max_depth,
                    3 if include_examples else 0,
                )
            )
        else:
            result.update({"kind": suffix.lstrip(".") or "unknown", "status": "inventory_only"})
    except Exception as error:
        result.update({"status": "error", "error": "{}: {}".format(type(error).__name__, error)})
    return result


def _candidate_sort_key(root: Path, item: Path) -> str:
    return item.relative_to(root).as_posix()


def _candidate_bucket_key(root: Path, item: Path) -> Any:
    relative = item.relative_to(root)
    dataset = relative.parts[0] if len(relative.parts) > 1 else "."
    return dataset, _container_format(item)


def _ordered_bucket_keys(buckets: Mapping[Any, List[Path]]) -> List[Any]:
    by_dataset: DefaultDict[str, List[Any]] = defaultdict(list)
    for key in buckets:
        by_dataset[key[0]].append(key)
    for keys in by_dataset.values():
        keys.sort(key=lambda key: (key[1] not in PRIORITY_CONTAINER_FORMATS, key[1]))

    ordered = []
    datasets = sorted(by_dataset)
    max_formats = max((len(by_dataset[dataset]) for dataset in datasets), default=0)
    for format_index in range(max_formats):
        for dataset in datasets:
            keys = by_dataset[dataset]
            if format_index < len(keys):
                ordered.append(keys[format_index])
    return ordered


def _select_profile_candidates(root: Path, candidates: List[Path], max_files: int) -> List[Path]:
    candidates = sorted(candidates, key=lambda item: _candidate_sort_key(root, item))
    if len(candidates) <= max_files:
        return candidates

    buckets: DefaultDict[Any, List[Path]] = defaultdict(list)
    for item in candidates:
        buckets[_candidate_bucket_key(root, item)].append(item)

    bucket_keys = _ordered_bucket_keys(buckets)
    offsets = {key: 0 for key in bucket_keys}
    selected: List[Path] = []

    # First preserve one representative from every top-level-dataset/format
    # bucket whenever the limit permits it.
    for key in bucket_keys:
        if len(selected) >= max_files:
            break
        selected.append(buckets[key][0])
        offsets[key] = 1

    def fill_round_robin(keys: List[Any]) -> None:
        while len(selected) < max_files:
            added = False
            for key in keys:
                offset = offsets[key]
                if offset >= len(buckets[key]):
                    continue
                selected.append(buckets[key][offset])
                offsets[key] = offset + 1
                added = True
                if len(selected) >= max_files:
                    return
            if not added:
                return

    # Columnar metadata files are cheap and authoritative schema sources, so
    # preserve as many of them as possible before filling from other formats.
    priority_keys = [key for key in bucket_keys if key[1] in PRIORITY_CONTAINER_FORMATS]
    fill_round_robin(priority_keys)
    fill_round_robin(bucket_keys)
    return sorted(selected, key=lambda item: _candidate_sort_key(root, item))


def _selection_bucket_summary(root: Path, candidates: List[Path], selected: List[Path]) -> List[Dict[str, Any]]:
    candidate_counts = Counter(_candidate_bucket_key(root, item) for item in candidates)
    selected_counts = Counter(_candidate_bucket_key(root, item) for item in selected)
    return [
        {
            "dataset": dataset,
            "format": container_format,
            "candidates": candidate_counts[(dataset, container_format)],
            "selected": selected_counts[(dataset, container_format)],
            "omitted": candidate_counts[(dataset, container_format)]
            - selected_counts[(dataset, container_format)],
        }
        for dataset, container_format in sorted(candidate_counts)
    ]


def _summarize_file_reports(file_reports: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts: Counter = Counter()
    issue_counts: Counter = Counter()
    issue_examples: DefaultDict[Any, List[str]] = defaultdict(list)
    families: Dict[str, Dict[str, Any]] = {}
    for report in file_reports:
        status = str(report.get("status", "unknown"))
        status_counts[status] += 1
        if status not in ("ok", "inventory_only"):
            message = str(report.get("error") or "unspecified profiler issue")
            issue_key = (status, message)
            issue_counts[issue_key] += 1
            path = report.get("path")
            if isinstance(path, str) and len(issue_examples[issue_key]) < 5:
                issue_examples[issue_key].append(path)
        if report.get("status") != "ok" or "schema" not in report or report.get("schema") is None:
            continue
        canonical_schema = _canonical_schema_structure(report["schema"])
        signature_payload = {
            "kind": report.get("kind"),
            "root_kind": report.get("root_kind"),
            "schema": canonical_schema,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                signature_payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        family_value = {
            "fingerprint": fingerprint,
            "kind": report.get("kind"),
            "files": 0,
            "records": 0,
            "examples": [],
            "schema": canonical_schema,
        }
        if report.get("root_kind") is not None:
            family_value["root_kind"] = report["root_kind"]
        family = families.setdefault(fingerprint, family_value)
        family["files"] += 1
        if isinstance(report.get("total_records"), int):
            family["records"] += report["total_records"]
        if len(family["examples"]) < 5:
            family["examples"].append(report["path"])
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "issues": [
            {
                "status": status,
                "message": message,
                "count": count,
                "examples": issue_examples[(status, message)],
            }
            for (status, message), count in sorted(
                issue_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
            )
        ],
        "schema_families": sorted(
            families.values(),
            key=lambda value: (str(value["kind"]), str(value.get("root_kind")), value["fingerprint"]),
        ),
    }


def _bounded_parallel_map(
    worker: Callable[[Path], Dict[str, Any]],
    items: Iterable[Path],
    workers: int,
) -> Iterator[Dict[str, Any]]:
    if workers <= 1:
        for item in items:
            yield worker(item)
        return
    pending_limit = max(workers * 2, 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for item in items:
            pending.append(executor.submit(worker, item))
            if len(pending) >= pending_limit:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def profile_path(
    path: Path,
    sample_rows: int = 100,
    max_depth: int = 8,
    max_files: int = 10000,
    include_examples: bool = False,
    relative_paths: bool = False,
    workers: int = 1,
    summary_only: bool = False,
    exclude_patterns: Sequence[str] = (),
) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    normalized_excludes = _normalize_exclude_patterns(exclude_patterns)
    if path.is_file():
        if normalized_excludes:
            raise ValueError("exclude patterns are only supported when profiling a directory")
        report = profile_file(
            path, sample_rows=sample_rows, max_depth=max_depth, include_examples=include_examples
        )
        if relative_paths:
            report["path"] = path.name
        summary = _summarize_file_reports([report])
        result = {
            "root": path.name if relative_paths else str(path),
            "status_counts": summary["status_counts"],
            "issues": summary["issues"],
            "schema_families": summary["schema_families"],
        }
        if summary_only:
            result["file_reports_omitted"] = 1
        else:
            result["files"] = [report]
        return result
    if not path.is_dir():
        raise FileNotFoundError(path)
    total_files = 0
    excluded_files = 0
    excluded_reasons: Counter = Counter()
    user_exclusion_counts: Counter = Counter()
    user_exclusion_examples: DefaultDict[str, List[str]] = defaultdict(list)
    structured_files = 0
    suffix_counts: Counter = Counter()
    candidates: List[Path] = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        total_files += 1
        relative_path = item.relative_to(path)
        exclusion_pattern = _matching_exclude_pattern(relative_path, normalized_excludes)
        if exclusion_pattern is not None:
            excluded_files += 1
            excluded_reasons["user_pattern"] += 1
            user_exclusion_counts[exclusion_pattern] += 1
            if len(user_exclusion_examples[exclusion_pattern]) < 10:
                user_exclusion_examples[exclusion_pattern].append(relative_path.as_posix())
            continue
        exclusion_reason = _metadata_exclusion_reason(relative_path)
        if exclusion_reason is not None:
            excluded_files += 1
            excluded_reasons[exclusion_reason] += 1
            continue
        suffix = item.suffix.lower() or "<none>"
        suffix_counts[suffix] += 1
        if _is_structured_file(item):
            structured_files += 1
            candidates.append(item)
    selected = _select_profile_candidates(path, candidates, max_files)
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
    reports = _bounded_parallel_map(run, selected, workers)
    if summary_only:
        summary = _summarize_file_reports(reports)
        file_reports = None
    else:
        file_reports = list(reports)
        summary = _summarize_file_reports(file_reports)
    result = {
        "root": "." if relative_paths else str(path),
        "total_files": total_files,
        "eligible_files": total_files - excluded_files,
        "excluded_files": excluded_files,
        "excluded_reasons": dict(sorted(excluded_reasons.items())),
        "user_exclusions": [
            {
                "pattern": pattern,
                "files": user_exclusion_counts[pattern],
                "examples": user_exclusion_examples[pattern],
            }
            for pattern in normalized_excludes
        ],
        "structured_files": structured_files,
        "profiled_files": len(selected),
        "truncated": truncated,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "selection_buckets": _selection_bucket_summary(path, candidates, selected),
        "status_counts": summary["status_counts"],
        "issues": summary["issues"],
        "schema_families": summary["schema_families"],
    }
    if summary_only:
        result["file_reports_omitted"] = len(selected)
    else:
        result["files"] = file_reports
    return result
