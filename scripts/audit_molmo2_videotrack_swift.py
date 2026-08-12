#!/usr/bin/env python3
"""Audit a converted Molmo2-VideoTrack ms-swift dataset end to end.

The audit joins source parquet annotations, the materialized-media index,
converted JSONL, and media_groups.tsv.  It also probes and decodes every
indexed MP4.  Source files are read only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DEFAULT_SOURCE_ROOT = Path("/mnt/luojunkun/stage1/dataset/Molmo2-VideoTrack")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack"
)
REQUIRED_SOURCE_COLUMNS = {
    "id",
    "video_dataset",
    "video",
    "clip",
    "exp",
    "points",
    "segments",
    "start_frame",
    "end_frame",
    "n_frames",
    "w",
    "h",
    "fps",
}
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SOURCE_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")


@dataclass(frozen=True)
class WindowDeclaration:
    path: str
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass(frozen=True)
class IndexEntry:
    key: str
    dataset: str
    clip: str
    source_video_id: str
    lineage_key: str
    width: int | None
    height: int | None
    fps: float | None
    windows: tuple[WindowDeclaration, ...]


@dataclass(frozen=True)
class SourceAnnotation:
    source: str
    row: int
    dataset: str
    video: str
    clip: str
    start_frame: int
    end_frame: int
    n_frames: int
    width: int
    height: int
    fps: float

    @property
    def key(self) -> str:
        return f"{self.dataset}::{self.clip}"

    @property
    def source_video_id(self) -> str:
        return f"{self.dataset}::{self.video}"

    @property
    def locator(self) -> str:
        return f"{self.source}#row={self.row}"


@dataclass(frozen=True)
class MediaExpectation:
    path: str
    frame_count: int
    width: int | None
    height: int | None
    fps: float | None


class Failures:
    """Collect de-duplicated hard failures in deterministic order."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}

    def add(self, code: str, detail: str, **context: Any) -> None:
        item: dict[str, Any] = {"code": code, "detail": detail}
        item.update({key: value for key, value in context.items() if value is not None})
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self._items[key] = item

    def sorted_items(self) -> list[dict[str, Any]]:
        return [self._items[key] for key in sorted(self._items)]

    def __bool__(self) -> bool:
        return bool(self._items)


class TrackCleaningError(ValueError):

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def positive_int(value: Any, field: str) -> int:
    result = strict_int(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def parse_namespaced(value: Any, field: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    namespace, separator, identity = value.partition("::")
    if not separator or not namespace or not identity or "::" in identity:
        raise ValueError(f"{field} must have the form namespace::identity")
    return namespace, identity


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return " ".join(str(value).strip().split())


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return ensure_list(value.tolist())
    return [value]


def raw_xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        raw_x, raw_y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        raw_x, raw_y = value
    else:
        return None
    try:
        if isinstance(raw_x, bool) or isinstance(raw_y, bool):
            return None
        x, y = float(raw_x), float(raw_y)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def clean_source_track_row(
    row: Mapping[str, Any], source: str, row_number: int
) -> SourceAnnotation:
    dataset = clean_text(row.get("video_dataset"))
    video = clean_text(row.get("video"))
    clip = clean_text(row.get("clip"))
    if not dataset or not video or not clip:
        raise TrackCleaningError(
            "invalid_media_identity", "video_dataset, video, and clip must be non-empty"
        )
    try:
        start = strict_int(row.get("start_frame"), "start_frame")
        end = strict_int(row.get("end_frame"), "end_frame")
        n_frames = strict_int(row.get("n_frames"), "n_frames")
    except ValueError as exc:
        raise TrackCleaningError("invalid_frame_bounds", str(exc)) from exc
    if start < 0 or end < start or n_frames != end - start + 1:
        raise TrackCleaningError(
            "invalid_frame_bounds", "frame bounds and n_frames are inconsistent"
        )
    try:
        width = positive_int(row.get("w"), "w")
        height = positive_int(row.get("h"), "h")
    except ValueError as exc:
        raise TrackCleaningError("invalid_dimensions", str(exc)) from exc
    try:
        fps = positive_float(row.get("fps"), "fps")
    except ValueError as exc:
        raise TrackCleaningError("invalid_fps", str(exc)) from exc

    points_by_object: dict[str, list[Any]] = {}
    for track in ensure_list(row.get("points")):
        if not isinstance(track, Mapping):
            raise TrackCleaningError("point_track_not_object", "point track is not an object")
        object_id = clean_text(track.get("object_id"))
        if not object_id:
            raise TrackCleaningError("empty_point_object_id", "point object_id is empty")
        if object_id in points_by_object:
            raise TrackCleaningError(
                "duplicate_point_object_id", f"point object_id is duplicated: {object_id}"
            )
        values = ensure_list(track.get("points"))
        if len(values) != n_frames:
            raise TrackCleaningError(
                "point_count_mismatch", "point count does not match n_frames"
            )
        for point in values:
            if point is None:
                continue
            xy = raw_xy(point)
            if xy is None:
                raise TrackCleaningError(
                    "invalid_visible_point", "visible point is not a finite [x, y] pair"
                )
            x, y = xy
            if x < 0 or y < 0 or x > width or y > height:
                raise TrackCleaningError(
                    "point_out_of_bounds", "visible point lies outside source dimensions"
                )
        points_by_object[object_id] = values
    if not points_by_object:
        raise TrackCleaningError("no_point_tracks", "annotation has no point tracks")

    segment_object_ids: set[str] = set()
    for segment_track in ensure_list(row.get("segments")):
        if not isinstance(segment_track, Mapping):
            raise TrackCleaningError(
                "segment_track_not_object", "segment track is not an object"
            )
        object_id = clean_text(segment_track.get("object_id"))
        if not object_id:
            raise TrackCleaningError(
                "empty_segment_object_id", "segment object_id is empty"
            )
        if object_id in segment_object_ids:
            raise TrackCleaningError(
                "duplicate_segment_object_id",
                f"segment object_id is duplicated: {object_id}",
            )
        segment_object_ids.add(object_id)
        points = points_by_object.get(object_id)
        if points is None:
            raise TrackCleaningError(
                "segment_without_point_track",
                "segment object has no matching point track",
            )
        visible_from_segments: set[int] = set()
        for segment in ensure_list(segment_track.get("segments")):
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                raise TrackCleaningError(
                    "invalid_visibility_segment",
                    "visibility segment must contain start and end",
                )
            try:
                segment_start = strict_int(segment[0], "segment.start")
                segment_end = strict_int(segment[1], "segment.end")
            except ValueError as exc:
                raise TrackCleaningError("invalid_visibility_segment", str(exc)) from exc
            if segment_start < 0 or segment_end < segment_start or segment_end >= n_frames:
                raise TrackCleaningError(
                    "visibility_segment_out_of_range",
                    "visibility segment is out of range",
                )
            segment_frames = set(range(segment_start, segment_end + 1))
            if visible_from_segments.intersection(segment_frames):
                raise TrackCleaningError(
                    "overlapping_visibility_segments", "visibility segments overlap"
                )
            visible_from_segments.update(segment_frames)
        visible_from_points = {
            index for index, point in enumerate(points) if point is not None
        }
        if visible_from_segments != visible_from_points:
            raise TrackCleaningError(
                "visibility_points_mismatch",
                "visibility segments do not match non-null points",
            )
    if segment_object_ids != set(points_by_object):
        raise TrackCleaningError(
            "point_segment_ids_mismatch", "point and segment object IDs differ"
        )
    if not clean_text(row.get("exp")):
        raise TrackCleaningError("empty_expression", "tracking expression is empty")
    return SourceAnnotation(
        source,
        row_number,
        dataset,
        video,
        clip,
        start,
        end,
        n_frames,
        width,
        height,
        fps,
    )


def normalize_mp4_path(value: Any, identity: str, failures: Failures) -> str | None:
    if not isinstance(value, str) or not value.strip():
        failures.add("invalid_media_path", "media path must be a non-empty string", identity=identity)
        return None
    path = Path(value.strip())
    if not path.is_absolute():
        failures.add("invalid_media_path", "media path is not absolute", identity=identity, path=str(path))
        return None
    if path.suffix.casefold() != ".mp4":
        failures.add("invalid_media_path", "media path does not end in .mp4", identity=identity, path=str(path))
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        failures.add("missing_media", "media path is not an existing file", identity=identity, path=str(resolved))
    return str(resolved)


def load_index(path: Path, failures: Failures) -> dict[str, IndexEntry]:
    if not path.is_file():
        failures.add("missing_index", "video-track index does not exist", path=str(path))
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except Exception as exc:
        failures.add("invalid_index_json", str(exc), path=str(path))
        return {}
    if not isinstance(raw, dict):
        failures.add("invalid_index_schema", "top-level index must be an object", path=str(path))
        return {}

    entries: dict[str, IndexEntry] = {}
    for raw_key in sorted(raw, key=lambda item: str(item)):
        value = raw[raw_key]
        key = str(raw_key)
        try:
            dataset, clip = parse_namespaced(key, "index key")
        except ValueError as exc:
            failures.add("invalid_index_key", str(exc), key=key)
            continue
        if not isinstance(value, dict):
            failures.add("invalid_index_entry", "index entry must be an object", key=key)
            continue
        try:
            source_dataset, source_video = parse_namespaced(
                value.get("source_video_id"), "source_video_id"
            )
            lineage_namespace, lineage_video = parse_namespaced(
                value.get("lineage_key"), "lineage_key"
            )
        except ValueError as exc:
            failures.add("invalid_index_lineage", str(exc), key=key)
            continue
        if source_dataset != dataset:
            failures.add(
                "index_dataset_mismatch",
                "index key dataset differs from source_video_id dataset",
                key=key,
                source_video_id=value.get("source_video_id"),
            )
        if lineage_video != source_video:
            failures.add(
                "index_lineage_mismatch",
                "lineage_key identity differs from source_video_id identity",
                key=key,
                source_video_id=value.get("source_video_id"),
                lineage_key=value.get("lineage_key"),
            )

        metadata: dict[str, int | float | None] = {}
        for field in ("width", "height"):
            if field not in value or value[field] is None:
                metadata[field] = None
            else:
                try:
                    metadata[field] = positive_int(value[field], field)
                except ValueError as exc:
                    failures.add("invalid_index_metadata", str(exc), key=key, field=field)
                    metadata[field] = None
        if "fps" not in value or value["fps"] is None:
            metadata["fps"] = None
        else:
            try:
                metadata["fps"] = positive_float(value["fps"], "fps")
            except ValueError as exc:
                failures.add("invalid_index_metadata", str(exc), key=key, field="fps")
                metadata["fps"] = None

        raw_windows = value.get("windows")
        if not isinstance(raw_windows, list) or not raw_windows:
            failures.add("invalid_index_windows", "windows must be a non-empty list", key=key)
            continue
        windows: list[WindowDeclaration] = []
        seen_paths: set[str] = set()
        for ordinal, raw_window in enumerate(raw_windows):
            identity = f"{key}#window={ordinal}"
            if not isinstance(raw_window, dict):
                failures.add("invalid_index_window", "window must be an object", identity=identity)
                continue
            if raw_window.get("mode") != "cropped":
                failures.add("invalid_index_window", "window mode must be cropped", identity=identity)
            media_path = normalize_mp4_path(raw_window.get("path"), identity, failures)
            try:
                start = strict_int(raw_window.get("source_start_frame"), "source_start_frame")
                end = strict_int(raw_window.get("source_end_frame"), "source_end_frame")
                if start < 0 or end < start:
                    raise ValueError("window frame bounds must form a non-negative closed interval")
            except ValueError as exc:
                failures.add("invalid_index_window", str(exc), identity=identity)
                continue
            if end - start + 1 > 128:
                failures.add(
                    "index_window_too_long",
                    "cropped VideoTrack windows must contain at most 128 frames",
                    identity=identity,
                    frame_count=end - start + 1,
                )
            if media_path is None:
                continue
            if media_path in seen_paths:
                failures.add("duplicate_window_path", "one index entry reuses an MP4 path", key=key, path=media_path)
            seen_paths.add(media_path)
            windows.append(WindowDeclaration(media_path, start, end))

        ordered = sorted(windows, key=lambda item: (item.start_frame, item.end_frame, item.path))
        if windows != ordered:
            failures.add("noncanonical_window_order", "windows are not sorted by source frame bounds", key=key)
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_frame != previous.end_frame + 1:
                failures.add(
                    "noncontinuous_windows",
                    "adjacent windows do not form a continuous non-overlapping partition",
                    key=key,
                    previous_end=previous.end_frame,
                    current_start=current.start_frame,
                )
        if not ordered:
            failures.add("invalid_index_windows", "entry has no usable windows", key=key)
            continue
        entries[key] = IndexEntry(
            key=key,
            dataset=dataset,
            clip=clip,
            source_video_id=f"{source_dataset}::{source_video}",
            lineage_key=f"{lineage_namespace}::{lineage_video}",
            width=metadata["width"] if isinstance(metadata["width"], int) else None,
            height=metadata["height"] if isinstance(metadata["height"], int) else None,
            fps=float(metadata["fps"]) if metadata["fps"] is not None else None,
            windows=tuple(ordered),
        )
    if not entries:
        failures.add("empty_index", "index contains no usable entries", path=str(path))
    return entries


def normalize_video_track_sources(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    selected: list[str] = []
    for raw_value in values:
        value = raw_value.strip().casefold()
        if not SOURCE_NAME_PATTERN.fullmatch(value):
            raise ValueError(f"invalid VideoTrack source name: {raw_value!r}")
        if value not in selected:
            selected.append(value)
    return selected


def discover_source_parquets(
    source_root: Path,
    video_track_sources: Sequence[str] | None,
    failures: Failures,
) -> list[Path]:
    sources = sorted(source_root.rglob("*.parquet"), key=lambda item: str(item))
    if not video_track_sources:
        return sources

    data_root = source_root / "data"
    sources_by_group: dict[str, list[Path]] = {}
    for source in sources:
        try:
            relative = source.relative_to(data_root)
        except ValueError:
            continue
        if len(relative.parts) < 2:
            continue
        sources_by_group.setdefault(relative.parts[0].casefold(), []).append(source)

    selected_groups = set(video_track_sources)
    missing = sorted(selected_groups - sources_by_group.keys())
    if missing:
        failures.add(
            "unknown_source_selection",
            "requested VideoTrack source groups are unavailable",
            missing=missing,
            available=sorted(sources_by_group),
        )
    return sorted(
        source
        for group, group_sources in sources_by_group.items()
        if group in selected_groups
        for source in group_sources
    )


def load_source_annotations(
    source_root: Path,
    failures: Failures,
    video_track_sources: Sequence[str] | None = None,
) -> tuple[
    list[SourceAnnotation],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not source_root.is_dir():
        failures.add("missing_source_root", "source root is not a directory", path=str(source_root))
        return [], [], []
    sources = discover_source_parquets(source_root, video_track_sources, failures)
    if not sources:
        failures.add("missing_source_parquet", "no parquet files were found", path=str(source_root))
        return [], [], []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        failures.add("missing_dependency", f"pyarrow is required: {exc}")
        return [], [], []

    annotations: list[SourceAnnotation] = []
    cleaning_rejections: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    columns = sorted(REQUIRED_SOURCE_COLUMNS)
    for source in sources:
        file_report: dict[str, Any] = {
            "path": str(source.resolve()),
            "rows": 0,
            "retained_rows": 0,
            "expected_rejected_rows": 0,
        }
        try:
            parquet = pq.ParquetFile(source)
            missing = sorted(REQUIRED_SOURCE_COLUMNS - set(parquet.schema_arrow.names))
            if missing:
                failures.add(
                    "source_schema_missing_columns",
                    "source parquet is missing required columns",
                    path=str(source.resolve()),
                    missing=missing,
                )
                file_report["missing_columns"] = missing
                file_reports.append(file_report)
                continue
            row_number = 0
            for batch in parquet.iter_batches(columns=columns, batch_size=8192):
                for row in batch.to_pylist():
                    locator = f"{source.resolve()}#row={row_number}"
                    file_report["rows"] += 1
                    try:
                        annotation = clean_source_track_row(
                            row, str(source.resolve()), row_number
                        )
                    except TrackCleaningError as exc:
                        rejection = {
                            "source": str(source.resolve()),
                            "row": row_number,
                            "locator": locator,
                            "dataset": clean_text(row.get("video_dataset")),
                            "video": clean_text(row.get("video")),
                            "clip": clean_text(row.get("clip")),
                            "id": clean_text(row.get("id")),
                            "reason": exc.reason,
                            "detail": str(exc),
                        }
                        cleaning_rejections.append(rejection)
                        file_report["expected_rejected_rows"] += 1
                    else:
                        annotations.append(annotation)
                        file_report["retained_rows"] += 1
                    row_number += 1
        except Exception as exc:
            failures.add("source_parquet_read_failed", str(exc), path=str(source.resolve()))
            file_report["error"] = str(exc)
        file_reports.append(file_report)
    return annotations, cleaning_rejections, file_reports


def validate_output_record(
    record: Any, split: str, line_number: int, failures: Failures
) -> str | None:
    identity = f"{split}.jsonl:{line_number}"
    if not isinstance(record, dict):
        failures.add("invalid_output_schema", "record must be an object", identity=identity)
        return None
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        failures.add("invalid_output_schema", "messages must be a non-empty list", identity=identity)
        return None
    for ordinal, message in enumerate(messages):
        if not isinstance(message, dict):
            failures.add("invalid_output_schema", "message must be an object", identity=identity, message=ordinal)
            return None
        if message.get("role") not in ALLOWED_ROLES:
            failures.add("invalid_output_schema", "message role is invalid", identity=identity, message=ordinal)
            return None
        if not isinstance(message.get("content"), str):
            failures.add("invalid_output_schema", "message content must be a string", identity=identity, message=ordinal)
            return None
    if messages[-1].get("role") != "assistant":
        failures.add("invalid_output_schema", "conversation must end with assistant", identity=identity)
    elif not messages[-1]["content"].strip():
        failures.add("invalid_output_schema", "final assistant content must be non-empty", identity=identity)
    video_tokens = sum(message["content"].count("<video>") for message in messages)
    videos = record.get("videos")
    if not isinstance(videos, list) or len(videos) != 1:
        failures.add("invalid_output_schema", "VideoTrack record must contain exactly one videos entry", identity=identity)
        return None
    if video_tokens != 1:
        failures.add("placeholder_mismatch", "record must contain exactly one <video> token", identity=identity, tokens=video_tokens)
    for token, plural in (("<image>", "images"), ("<audio>", "audios")):
        count = sum(message["content"].count(token) for message in messages)
        media = record.get(plural, [])
        if count or media:
            failures.add("unexpected_media_type", f"VideoTrack record must not contain {plural}", identity=identity)
    if record.get("objects"):
        failures.add("unexpected_objects", "VideoTrack textual tracking record must not contain objects", identity=identity)
    return normalize_mp4_path(videos[0], identity, failures)


def _percentile(sorted_values: Sequence[int], quantile: float) -> int | float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    value = sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (
        position - lower
    )
    return round(value, 3)


def distribution(values: Sequence[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1] if ordered else None,
    }


def assistant_content_metrics(record: Mapping[str, Any]) -> tuple[int, int] | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    contents = [
        message["content"]
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
    ]
    if not contents:
        return None
    content = "\n".join(contents)
    return len(content), sum(bool(line.strip()) for line in content.splitlines())


def load_split(
    output_dir: Path, split: str, failures: Failures
) -> tuple[
    dict[str, Any],
    Counter[str],
    dict[str, set[str]],
    dict[str, list[int]],
]:
    path = output_dir / f"{split}.jsonl"
    report: dict[str, Any] = {
        "path": str(path.resolve()),
        "records": 0,
        "valid_records": 0,
        "unique_media": 0,
    }
    media_counter: Counter[str] = Counter()
    media_splits: dict[str, set[str]] = defaultdict(set)
    assistant_values: dict[str, list[int]] = {"characters": [], "nonempty_lines": []}
    if not path.is_file():
        failures.add("missing_split_file", "split JSONL does not exist", split=split, path=str(path.resolve()))
        report["assistant_content"] = {
            "characters": distribution([]),
            "nonempty_lines": distribution([]),
        }
        return report, media_counter, media_splits, assistant_values
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                report["records"] += 1
                if not line.strip():
                    failures.add("invalid_jsonl", "blank JSONL line", split=split, line=line_number)
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    failures.add("invalid_jsonl", str(exc), split=split, line=line_number)
                    continue
                media_path = validate_output_record(record, split, line_number, failures)
                if media_path is not None:
                    report["valid_records"] += 1
                    media_counter[media_path] += 1
                    media_splits[media_path].add(split)
                    metrics = assistant_content_metrics(record)
                    if metrics is not None:
                        characters, nonempty_lines = metrics
                        assistant_values["characters"].append(characters)
                        assistant_values["nonempty_lines"].append(nonempty_lines)
    except Exception as exc:
        failures.add("split_read_failed", str(exc), split=split, path=str(path.resolve()))
    report["unique_media"] = len(media_counter)
    report["assistant_content"] = {
        field: distribution(values) for field, values in assistant_values.items()
    }
    return report, media_counter, media_splits, assistant_values


def load_media_groups(
    output_dir: Path, failures: Failures
) -> tuple[dict[str, str], dict[str, Any]]:
    path = output_dir / "media_groups.tsv"
    report: dict[str, Any] = {"path": str(path.resolve()), "groups": 0, "train": 0, "val": 0}
    if not path.is_file():
        failures.add("missing_media_groups", "media_groups.tsv does not exist", path=str(path.resolve()))
        return {}, report
    assignments: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        failures.add("media_groups_read_failed", str(exc), path=str(path.resolve()))
        return {}, report
    if not lines or lines[0] != "split\tmedia_key":
        failures.add("invalid_media_groups_header", "expected exact header: split\\tmedia_key", path=str(path.resolve()))
        data_lines = lines[1:] if lines else []
    else:
        data_lines = lines[1:]
    for line_number, line in enumerate(data_lines, 2):
        parts = line.split("\t")
        if len(parts) != 2 or parts[0] not in {"train", "val"} or not parts[1]:
            failures.add("invalid_media_group", "invalid media group row", line=line_number, value=line)
            continue
        split, media_key = parts
        if media_key in assignments:
            failures.add(
                "duplicate_media_group",
                "media key appears more than once",
                media_key=media_key,
                first_split=assignments[media_key],
                second_split=split,
            )
        else:
            assignments[media_key] = split
    counts = Counter(assignments.values())
    report.update({"groups": len(assignments), "train": counts["train"], "val": counts["val"]})
    return assignments, report


def media_key_for_lineage(lineage_key: str) -> str:
    namespace, identity = parse_namespaced(lineage_key, "lineage_key")
    return f"video:track:{namespace}:{identity}"


def build_media_expectations(
    entries: Mapping[str, IndexEntry], failures: Failures
) -> tuple[dict[str, MediaExpectation], dict[str, set[str]]]:
    raw: dict[str, dict[str, set[Any]]] = {}
    path_to_lineages: dict[str, set[str]] = defaultdict(set)
    for entry in entries.values():
        for window in entry.windows:
            values = raw.setdefault(
                window.path,
                {"frame_count": set(), "width": set(), "height": set(), "fps": set()},
            )
            values["frame_count"].add(window.frame_count)
            if entry.width is not None:
                values["width"].add(entry.width)
            if entry.height is not None:
                values["height"].add(entry.height)
            if entry.fps is not None:
                values["fps"].add(entry.fps)
            path_to_lineages[window.path].add(entry.lineage_key)
    expectations: dict[str, MediaExpectation] = {}
    for path in sorted(raw):
        values = raw[path]
        for field in ("frame_count", "width", "height", "fps"):
            if len(values[field]) > 1:
                failures.add(
                    "conflicting_media_declaration",
                    "one MP4 has conflicting index metadata",
                    path=path,
                    field=field,
                    values=sorted(values[field]),
                )
        if len(path_to_lineages[path]) > 1:
            failures.add(
                "media_lineage_conflict",
                "one MP4 belongs to multiple lineage keys",
                path=path,
                lineages=sorted(path_to_lineages[path]),
            )
        frame_values = sorted(values["frame_count"])
        if not frame_values:
            continue
        expectations[path] = MediaExpectation(
            path,
            int(frame_values[0]),
            int(sorted(values["width"])[0]) if values["width"] else None,
            int(sorted(values["height"])[0]) if values["height"] else None,
            float(sorted(values["fps"])[0]) if values["fps"] else None,
        )
    return expectations, path_to_lineages


def _parse_rate(value: Any) -> float | None:
    if not isinstance(value, str) or not value or value in {"N/A", "0/0"}:
        return None
    try:
        result = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _parse_frame_count(stream: Mapping[str, Any]) -> int | None:
    for field in ("nb_read_frames", "nb_frames"):
        value = stream.get(field)
        if value in (None, "", "N/A"):
            continue
        try:
            result = int(value)
        except (TypeError, ValueError):
            continue
        if result > 0:
            return result
    return None


def _check_value(expected: Any, actual: Any, *, float_value: bool = False) -> dict[str, Any]:
    if expected is None:
        return {"status": "unavailable", "expected": None, "actual": actual}
    if actual is None:
        return {"status": "failed", "expected": expected, "actual": None}
    matched = (
        math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6)
        if float_value
        else expected == actual
    )
    return {"status": "matched" if matched else "mismatch", "expected": expected, "actual": actual}


def _decode_frame(ffmpeg: str, path: str, frame_index: int) -> tuple[bool, str | None]:
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        path,
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        "-an",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=120)
    except Exception as exc:
        return False, str(exc)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or b"ffmpeg failed")[-2000:]
        return False, detail.decode("utf-8", errors="replace").strip()
    if not result.stdout.startswith(PNG_SIGNATURE):
        return False, "ffmpeg produced no decodable PNG payload"
    return True, None


def audit_media(expectation: MediaExpectation, ffprobe: str, ffmpeg: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": expectation.path,
        "expected": {
            "frame_count": expectation.frame_count,
            "width": expectation.width,
            "height": expectation.height,
            "fps": expectation.fps,
        },
        "probe": {},
        "checks": {},
        "errors": [],
    }
    actual_frames: int | None = None
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames,nb_frames",
        "-of",
        "json",
        expectation.path,
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=120
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "ffprobe failed")[-2000:].strip())
        payload = json.loads(completed.stdout)
        streams = payload.get("streams")
        if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
            raise ValueError("ffprobe did not return exactly one selected video stream")
        stream = streams[0]
        actual_frames = _parse_frame_count(stream)
        width = stream.get("width") if isinstance(stream.get("width"), int) else None
        height = stream.get("height") if isinstance(stream.get("height"), int) else None
        fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
        result["probe"] = {
            "frame_count": actual_frames,
            "width": width,
            "height": height,
            "fps": fps,
        }
        result["checks"] = {
            "frame_count": _check_value(expectation.frame_count, actual_frames),
            "width": _check_value(expectation.width, width),
            "height": _check_value(expectation.height, height),
            "fps": _check_value(expectation.fps, fps, float_value=True),
        }
        for field, check in result["checks"].items():
            if check["status"] in {"failed", "mismatch"}:
                result["errors"].append(f"{field} {check['status']}: expected={check['expected']}, actual={check['actual']}")
    except Exception as exc:
        result["errors"].append(f"ffprobe failed: {exc}")
        result["checks"] = {
            "frame_count": _check_value(expectation.frame_count, None),
            "width": _check_value(expectation.width, None),
            "height": _check_value(expectation.height, None),
            "fps": _check_value(expectation.fps, None, float_value=True),
        }

    last_frame = max(0, (actual_frames or expectation.frame_count) - 1)
    first_ok, first_error = _decode_frame(ffmpeg, expectation.path, 0)
    last_ok, last_error = _decode_frame(ffmpeg, expectation.path, last_frame)
    result["checks"]["decode_first"] = {"status": "passed" if first_ok else "failed", "frame": 0}
    result["checks"]["decode_last"] = {"status": "passed" if last_ok else "failed", "frame": last_frame}
    if first_error:
        result["errors"].append(f"first-frame decode failed: {first_error}")
    if last_error:
        result["errors"].append(f"last-frame decode failed: {last_error}")
    result["status"] = "passed" if not result["errors"] else "failed"
    return result


def audit_media_parallel(
    expectations: Mapping[str, MediaExpectation],
    workers: int,
    ffprobe: str,
    ffmpeg: str,
    audit_one: Callable[[MediaExpectation, str, str], dict[str, Any]] = audit_media,
) -> list[dict[str, Any]]:
    ordered = [expectations[path] for path in sorted(expectations)]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(audit_one, expectation, ffprobe, ffmpeg): expectation.path
            for expectation in ordered
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"path": path, "status": "failed", "errors": [f"media audit crashed: {exc}"]}
            results.append(result)
    return sorted(results, key=lambda item: item["path"])


def reconcile_sources(
    annotations: Sequence[SourceAnnotation],
    cleaning_rejections: Sequence[Mapping[str, Any]],
    entries: Mapping[str, IndexEntry],
    failures: Failures,
) -> tuple[Counter[str], list[dict[str, Any]], dict[str, Any]]:
    expected: Counter[str] = Counter()
    indexed_annotations: list[dict[str, Any]] = []
    source_keys = {annotation.key for annotation in annotations}
    datasets: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons: Counter[str] = Counter()
    rejection_sources: dict[str, Counter[str]] = defaultdict(Counter)
    for rejection in cleaning_rejections:
        dataset = str(rejection.get("dataset") or "<missing>")
        source = str(rejection.get("source") or "<unknown>")
        reason = str(rejection.get("reason") or "unknown")
        clip = str(rejection.get("clip") or "")
        datasets[dataset]["source_rows"] += 1
        datasets[dataset]["expected_rejected_rows"] += 1
        rejection_reasons[reason] += 1
        rejection_sources[source][reason] += 1
        if dataset != "<missing>" and clip:
            key = f"{dataset}::{clip}"
            source_keys.add(key)
            if key in entries:
                datasets[dataset]["indexed_expected_rejected_rows"] += 1
    for annotation in annotations:
        datasets[annotation.dataset]["source_rows"] += 1
        datasets[annotation.dataset]["retained_rows"] += 1
        entry = entries.get(annotation.key)
        if entry is None:
            datasets[annotation.dataset]["unindexed_retained_rows"] += 1
            continue
        datasets[annotation.dataset]["indexed_retained_rows"] += 1
        index_valid = True
        if entry.source_video_id != annotation.source_video_id:
            index_valid = False
            failures.add(
                "annotation_index_lineage_mismatch",
                "index source_video_id does not match source annotation",
                locator=annotation.locator,
                expected=annotation.source_video_id,
                actual=entry.source_video_id,
            )
        if entry.width is not None and entry.width != annotation.width:
            index_valid = False
            failures.add("annotation_index_metadata_mismatch", "width differs", locator=annotation.locator, expected=annotation.width, actual=entry.width)
        if entry.height is not None and entry.height != annotation.height:
            index_valid = False
            failures.add("annotation_index_metadata_mismatch", "height differs", locator=annotation.locator, expected=annotation.height, actual=entry.height)
        if entry.fps is not None and not math.isclose(entry.fps, annotation.fps, rel_tol=1e-7, abs_tol=1e-7):
            index_valid = False
            failures.add("annotation_index_metadata_mismatch", "fps differs", locator=annotation.locator, expected=annotation.fps, actual=entry.fps)
        if (
            entry.windows[0].start_frame != annotation.start_frame
            or entry.windows[-1].end_frame != annotation.end_frame
        ):
            index_valid = False
            failures.add(
                "annotation_window_coverage_mismatch",
                "index windows do not exactly cover the annotation closed interval",
                locator=annotation.locator,
                annotation_bounds=[annotation.start_frame, annotation.end_frame],
                index_bounds=[entry.windows[0].start_frame, entry.windows[-1].end_frame],
            )
        if any(window.frame_count > 128 for window in entry.windows):
            index_valid = False
        if not index_valid:
            datasets[annotation.dataset]["index_validation_rejected_rows"] += 1
            continue
        for window in entry.windows:
            expected[window.path] += 1
        indexed_annotations.append(
            {
                "locator": annotation.locator,
                "source_video_id": annotation.source_video_id,
                "lineage_key": entry.lineage_key,
                "paths": [window.path for window in entry.windows],
            }
        )
    unmatched = sorted(set(entries) - source_keys)
    for key in unmatched:
        failures.add("index_entry_without_source", "index entry has no matching source annotation", key=key)
    report = {
        "source_rows": len(annotations) + len(cleaning_rejections),
        "retained_rows": len(annotations),
        "expected_rejected_rows": len(cleaning_rejections),
        "indexed_retained_rows": len(indexed_annotations),
        "unindexed_or_index_invalid_retained_rows": len(annotations)
        - len(indexed_annotations),
        "index_entries": len(entries),
        "index_entries_without_source": unmatched,
        "expected_records": sum(expected.values()),
        "cleaning_rejections": {
            "total": len(cleaning_rejections),
            "by_reason": dict(sorted(rejection_reasons.items())),
            "by_source": {
                source: {
                    "total": sum(counts.values()),
                    "by_reason": dict(sorted(counts.items())),
                }
                for source, counts in sorted(rejection_sources.items())
            },
            "examples": sorted(
                (dict(item) for item in cleaning_rejections),
                key=lambda item: (str(item.get("source")), int(item.get("row", -1))),
            )[:100],
            "examples_omitted": max(0, len(cleaning_rejections) - 100),
        },
        "by_dataset": {dataset: dict(sorted(counts.items())) for dataset, counts in sorted(datasets.items())},
    }
    return expected, indexed_annotations, report


def _counter_mismatches(expected: Counter[str], actual: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"path": path, "expected": expected[path], "actual": actual[path]}
        for path in sorted(set(expected) | set(actual))
        if expected[path] != actual[path]
    ]


def run_audit(
    source_root: Path,
    output_dir: Path,
    index_path: Path,
    media_workers: int,
    ffprobe: str = "ffprobe",
    ffmpeg: str = "ffmpeg",
    audit_one: Callable[[MediaExpectation, str, str], dict[str, Any]] = audit_media,
    video_track_sources: Sequence[str] | None = None,
) -> dict[str, Any]:
    failures = Failures()
    source_root = source_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    index_path = index_path.expanduser().resolve()
    video_track_sources = normalize_video_track_sources(video_track_sources)

    entries = load_index(index_path, failures)
    expectations, path_to_lineages = build_media_expectations(entries, failures)
    annotations, cleaning_rejections, source_files = load_source_annotations(
        source_root, failures, video_track_sources
    )
    expected_counter, indexed_annotations, source_report = reconcile_sources(
        annotations, cleaning_rejections, entries, failures
    )
    physical_source_rows = sum(int(item.get("rows", 0)) for item in source_files)
    source_report.update(
        {
            "source_rows": physical_source_rows,
            "retained_source_rows": len(annotations),
            "expected_rejected_source_rows": len(cleaning_rejections),
            "unreadable_or_unaccounted_source_rows": physical_source_rows
            - len(annotations)
            - len(cleaning_rejections),
        }
    )

    split_reports: dict[str, Any] = {}
    actual_counter: Counter[str] = Counter()
    path_splits: dict[str, set[str]] = defaultdict(set)
    total_assistant_values: dict[str, list[int]] = {
        "characters": [],
        "nonempty_lines": [],
    }
    for split in ("train", "val"):
        split_report, counter, split_paths, assistant_values = load_split(
            output_dir, split, failures
        )
        split_reports[split] = split_report
        actual_counter.update(counter)
        for path, assigned in split_paths.items():
            path_splits[path].update(assigned)
        for field, values in assistant_values.items():
            total_assistant_values[field].extend(values)
    assistant_content_report = {
        "unit": "per converted record; all assistant messages joined with one newline",
        "percentile_method": "linear interpolation at index (n-1)*q",
        "train": split_reports["train"]["assistant_content"],
        "val": split_reports["val"]["assistant_content"],
        "total": {
            field: distribution(values)
            for field, values in total_assistant_values.items()
        },
    }
    for path, splits in sorted(path_splits.items()):
        if len(splits) > 1:
            failures.add("media_cross_split", "one MP4 appears in train and val", path=path, splits=sorted(splits))
        if path not in expectations:
            failures.add("output_media_not_indexed", "converted record references an MP4 absent from index", path=path)

    mismatches = _counter_mismatches(expected_counter, actual_counter)
    if mismatches:
        failures.add(
            "source_accounting_mismatch",
            "expected and converted per-MP4 record counts differ",
            mismatched_media=len(mismatches),
        )
    source_report.update(
        {
            "source_files": source_files,
            "actual_records": sum(actual_counter.values()),
            "record_count_balanced": not mismatches,
            "media_count_mismatches": mismatches,
        }
    )

    lineage_splits: dict[str, set[str]] = defaultdict(set)
    source_video_splits: dict[str, set[str]] = defaultdict(set)
    source_video_lineages: dict[str, set[str]] = defaultdict(set)
    incomplete_annotations = 0
    for annotation in indexed_annotations:
        splits: set[str] = set()
        for path in annotation["paths"]:
            splits.update(path_splits.get(path, set()))
        source_video_lineages[annotation["source_video_id"]].add(annotation["lineage_key"])
        if not splits:
            incomplete_annotations += 1
        if len(splits) > 1:
            failures.add(
                "annotation_cross_split",
                "windows from one source annotation appear in multiple splits",
                locator=annotation["locator"],
                splits=sorted(splits),
            )
        lineage_splits[annotation["lineage_key"]].update(splits)
        source_video_splits[annotation["source_video_id"]].update(splits)
    for lineage, splits in sorted(lineage_splits.items()):
        if len(splits) > 1:
            failures.add("lineage_cross_split", "one canonical video lineage appears in multiple splits", lineage_key=lineage, splits=sorted(splits))
    for source_video_id, splits in sorted(source_video_splits.items()):
        if len(splits) > 1:
            failures.add("source_video_cross_split", "one dataset/video identity appears in multiple splits", source_video_id=source_video_id, splits=sorted(splits))
    for source_video_id, lineages in sorted(source_video_lineages.items()):
        if len(lineages) > 1:
            failures.add("source_video_multiple_lineages", "one dataset/video identity maps to multiple canonical lineages", source_video_id=source_video_id, lineages=sorted(lineages))

    media_groups, media_groups_report = load_media_groups(output_dir, failures)
    expected_group_keys = {
        media_key_for_lineage(annotation["lineage_key"])
        for annotation in indexed_annotations
    }
    missing_groups = sorted(expected_group_keys - set(media_groups))
    extra_groups = sorted(set(media_groups) - expected_group_keys)
    if missing_groups:
        failures.add("missing_media_group_assignments", "expected lineage keys are absent from media_groups.tsv", count=len(missing_groups))
    if extra_groups:
        failures.add("extra_media_group_assignments", "media_groups.tsv contains keys absent from the index", count=len(extra_groups))
    for lineage, splits in sorted(lineage_splits.items()):
        if len(splits) != 1:
            continue
        media_key = media_key_for_lineage(lineage)
        group_split = media_groups.get(media_key)
        actual_split = next(iter(splits))
        if group_split is not None and group_split != actual_split:
            failures.add("media_group_split_mismatch", "media_groups.tsv differs from converted JSONL", media_key=media_key, expected=actual_split, actual=group_split)
    media_groups_report.update(
        {
            "missing": missing_groups,
            "extra": extra_groups,
            "cross_split_leakage": 0
            if all(len(splits) <= 1 for splits in lineage_splits.values())
            else sum(len(splits) > 1 for splits in lineage_splits.values()),
        }
    )

    media_results = audit_media_parallel(
        expectations, media_workers, ffprobe, ffmpeg, audit_one=audit_one
    )
    for result in media_results:
        for detail in result.get("errors", []):
            failures.add("media_validation_failed", str(detail), path=result.get("path"))
    metadata_availability = {
        field: {
            "available": sum(getattr(entry, field) is not None for entry in entries.values()),
            "unavailable": sum(getattr(entry, field) is None for entry in entries.values()),
        }
        for field in ("width", "height", "fps")
    }
    media_passed = sum(result.get("status") == "passed" for result in media_results)

    hard_failures = failures.sorted_items()
    return {
        "format": "molmo2-videotrack-ms-swift-audit",
        "version": 1,
        "status": "failed" if hard_failures else "passed",
        "inputs": {
            "source_root": str(source_root),
            "output_dir": str(output_dir),
            "video_track_index": str(index_path),
            "media_workers": media_workers,
            "ffprobe": ffprobe,
            "ffmpeg": ffmpeg,
            "video_track_sources": video_track_sources,
        },
        "splits": split_reports,
        "assistant_content_distribution": assistant_content_report,
        "index": {
            "entries": len(entries),
            "window_references": sum(len(entry.windows) for entry in entries.values()),
            "unique_media": len(expectations),
            "metadata_availability": metadata_availability,
        },
        "source_accounting": source_report,
        "media_groups": media_groups_report,
        "lineage": {
            "canonical_lineages": len(lineage_splits),
            "source_videos": len(source_video_splits),
            "indexed_annotations": len(indexed_annotations),
            "annotations_without_output": incomplete_annotations,
            "cross_split_lineages": sum(len(splits) > 1 for splits in lineage_splits.values()),
            "cross_split_source_videos": sum(len(splits) > 1 for splits in source_video_splits.values()),
        },
        "media": {
            "workers": media_workers,
            "files": len(media_results),
            "passed": media_passed,
            "failed": len(media_results) - media_passed,
            "results": media_results,
        },
        "hard_failures": hard_failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Molmo2-VideoTrack source accounting, split isolation, and MP4 integrity."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-track-index", type=Path, default=None)
    parser.add_argument(
        "--video-track-sources",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help="Only audit these Molmo2-VideoTrack data/<source> Parquet groups.",
    )
    parser.add_argument("--media-workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.media_workers <= 0:
        parser.error("--media-workers must be positive")
    try:
        args.video_track_sources = normalize_video_track_sources(args.video_track_sources)
    except ValueError as exc:
        parser.error(str(exc))
    args.source_root = args.source_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.video_track_index = (
        args.video_track_index.expanduser().resolve()
        if args.video_track_index is not None
        else args.output_dir / "video_track_media.json"
    )
    args.report = args.report.expanduser().resolve() if args.report is not None else None
    return args


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_audit(
        args.source_root,
        args.output_dir,
        args.video_track_index,
        args.media_workers,
        args.ffprobe,
        args.ffmpeg,
        video_track_sources=args.video_track_sources,
    )
    if args.report is not None:
        write_report(args.report, report)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
