#!/usr/bin/env python3
"""Convert Molmo2, PixMo, and SpatialVLM parquet datasets to ms-swift JSONL.

The converter uses a deterministic media-level split. Every sample sharing the
same image or video key is assigned to the same train/val file, including media
shared by different source parquet shards. Annotation conversion is parallel at
the parquet row-group level and remains streaming at merge time.

Image grounding rows use ms-swift ``objects``. Video point/track rows do not:
ms-swift currently normalizes ``objects`` against ``images`` only and has no
frame/timestamp field for video grounding. Their targets therefore retain the
time/frame association as normalized textual coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import sys
import tarfile
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


DEFAULT_INPUT_ROOT = Path("/mnt/luojunkun/stage1/dataset")
DEFAULT_OUTPUT_ROOT = Path("/mnt/luojunkun/stage1/dataset_ms-swift")
DATASETS = (
    "Molmo2-VideoCapQA",
    "Molmo2-VideoPoint",
    "Molmo2-VideoSubtitleQA",
    "Molmo2-VideoTrack",
    "pixmo-cap",
    "pixmo-points",
    "spatialvlm",
)
DATASET_ALIASES = {name.lower(): name for name in DATASETS}
DATASET_ALIASES.update(
    {
        "capqa": "Molmo2-VideoCapQA",
        "videocapqa": "Molmo2-VideoCapQA",
        "videopoint": "Molmo2-VideoPoint",
        "subtitleqa": "Molmo2-VideoSubtitleQA",
        "videosubtitleqa": "Molmo2-VideoSubtitleQA",
        "videotrack": "Molmo2-VideoTrack",
        "pixmocap": "pixmo-cap",
        "pixmopoints": "pixmo-points",
        "spatial": "spatialvlm",
    }
)
URL_SCHEMES = {"http", "https"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
NORMALIZED_BOX_RE = re.compile(
    r"\[\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\]"
)


@dataclass(frozen=True)
class RuntimeConfig:
    input_root: str
    output_root: str
    val_ratio: float
    seed: int
    include_subtitles: bool
    max_rejected_examples: int
    allow_unverified_remote_media: bool = False
    max_track_window_frames: int = 128


@dataclass(frozen=True)
class RowGroupTask:
    dataset: str
    source_path: str
    row_group: int
    ordinal: int
    row_limit: int | None
    fragment_dir: str
    forced_split: str | None = None


@dataclass
class WorkerResult:
    ordinal: int
    source_path: str
    train_path: str
    val_path: str
    groups_path: str
    rejected_path: str
    counters: dict[str, int]


@dataclass(frozen=True)
class ConvertedSample:
    media_key: str
    record: dict[str, Any]


@dataclass(frozen=True)
class TrackMediaEntry:
    windows: tuple[dict[str, Any], ...]
    source_video_id: str | None = None
    lineage_key: str | None = None


_RUNTIME: RuntimeConfig | None = None
_VIDEO_URLS: dict[str, str] = {}
_GENERATED_VIDEOS: dict[str, str] = {}
_TRACK_VIDEOS: dict[str, TrackMediaEntry] = {}
_PIXMO_POINT_GROUPS: dict[str, str] = {}
_PIXMO_URL_GROUPS: dict[str, str] = {}
_SPATIALVLM_TEST_MEDIA_KEYS: set[str] = set()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert seven local Molmo2/PixMo/SpatialVLM datasets to ms-swift JSONL."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset names/aliases to convert, or 'all'.",
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 2) - 1)),
        help="Parquet row-group worker processes.",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=None,
        help="Per-dataset source-row limit for smoke tests. Expanded QA rows may exceed this value.",
    )
    subtitle_group = parser.add_mutually_exclusive_group()
    subtitle_group.add_argument(
        "--include-subtitles",
        dest="include_subtitles",
        action="store_true",
        help="Include SubtitleQA transcripts in the prompt (default; retained for explicitness).",
    )
    subtitle_group.add_argument(
        "--exclude-subtitles",
        dest="include_subtitles",
        action="store_false",
        help="Drop SubtitleQA transcripts and intentionally convert the task to video-only QA.",
    )
    parser.set_defaults(include_subtitles=True)
    parser.add_argument(
        "--extract-generated-videos",
        action="store_true",
        help="Extract VideoPoint generated-video tar archives into the output videos directory.",
    )
    parser.add_argument(
        "--video-track-index",
        type=Path,
        default=None,
        help="JSON mapping dataset::clip (or row id) to one or more declared pre-cropped windows.",
    )
    parser.add_argument(
        "--video-track-sources",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help="Only convert these Molmo2-VideoTrack data/<source> Parquet groups (case-insensitive).",
    )
    parser.add_argument(
        "--max-track-window-frames",
        type=int,
        default=128,
        help="Maximum frames in each pre-cropped VideoTrack window (default: 128).",
    )
    parser.add_argument(
        "--video-media-index",
        type=Path,
        default=None,
        help="JSON mapping Molmo video_id to a local file/direct URL; overrides bundled URL mappings.",
    )
    parser.add_argument(
        "--max-rejected-examples",
        type=int,
        default=1000,
        help="Maximum rejected rows retained per row-group fragment; aggregate counts are always kept.",
    )
    parser.add_argument(
        "--max-reject-ratio",
        type=float,
        default=0.25,
        help="Fail a dataset when rejected source rows exceed this ratio.",
    )
    parser.add_argument(
        "--allow-empty-dataset",
        action="store_true",
        help="Allow zero converted records. Intended only for missing-media diagnostics.",
    )
    parser.add_argument(
        "--allow-unverified-remote-media",
        action="store_true",
        help="Permit HTTP(S) media references without downloading/decoding them during conversion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep row-group fragments after a successful merge for debugging.",
    )
    return parser.parse_args(argv)


def normalize_dataset_selection(values: Sequence[str]) -> list[str]:
    if any(value.lower() == "all" for value in values):
        if len(values) != 1:
            raise SystemExit("Use --datasets all by itself")
        return list(DATASETS)
    selected: list[str] = []
    for value in values:
        dataset = DATASET_ALIASES.get(value.lower())
        if dataset is None:
            raise SystemExit(f"Unknown dataset {value!r}. Choices: {', '.join(DATASETS)}")
        if dataset not in selected:
            selected.append(dataset)
    return selected


def normalize_video_track_sources(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    selected: list[str] = []
    for raw_value in values:
        value = raw_value.strip().casefold()
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value):
            raise SystemExit(f"Invalid --video-track-sources value: {raw_value!r}")
        if value not in selected:
            selected.append(value)
    return selected


def validate_args(args: argparse.Namespace) -> None:
    args.datasets = normalize_dataset_selection(args.datasets)
    args.video_track_sources = normalize_video_track_sources(args.video_track_sources)
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.video_track_sources and "Molmo2-VideoTrack" not in args.datasets:
        raise SystemExit("--video-track-sources requires Molmo2-VideoTrack in --datasets")
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio must be between 0 and 1")
    if args.num_workers <= 0:
        raise SystemExit("--num-workers must be greater than zero")
    if args.max_source_rows is not None and args.max_source_rows <= 0:
        raise SystemExit("--max-source-rows must be greater than zero")
    if args.max_rejected_examples < 0:
        raise SystemExit("--max-rejected-examples must be non-negative")
    if args.max_track_window_frames <= 0:
        raise SystemExit("--max-track-window-frames must be greater than zero")
    if not 0 <= args.max_reject_ratio <= 1:
        raise SystemExit("--max-reject-ratio must be in [0, 1]")
    if args.video_track_index is not None:
        args.video_track_index = args.video_track_index.expanduser().resolve()
    if args.video_media_index is not None:
        args.video_media_index = args.video_media_index.expanduser().resolve()


def import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required: pip install pyarrow") from exc
    return pq


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


def finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in URL_SCHEMES or not parsed.netloc:
        raise ValueError(f"invalid media URL: {value!r}")
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = hostname
    default_port = (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    )
    if port and not default_port:
        netloc = f"{hostname}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", "", ""))


def split_for_media(media_key: str, seed: int, val_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}\0{media_key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if bucket < val_ratio else "train"


def is_url(value: str) -> bool:
    return urlsplit(value).scheme.lower() in URL_SCHEMES


def validate_media_reference(value: str) -> None:
    if is_url(value):
        canonical_url(value)
        if _RUNTIME is not None and not _RUNTIME.allow_unverified_remote_media:
            raise ValueError(
                "remote media was not materialized; pass --allow-unverified-remote-media "
                "to retain HTTP(S) references explicitly"
            )
        return
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"local media path is not absolute: {value}")
    if not path.is_file():
        raise ValueError(f"local media file does not exist: {value}")


def placeholder_count(record: Mapping[str, Any], token: str) -> int:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(str(message.get("content") or "").count(token) for message in messages if isinstance(message, dict))


def validate_record(record: dict[str, Any]) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"invalid message at index {index}")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"message content at index {index} must be a string")
        if message.get("role") == "user":
            content_without_tags = re.sub(
                r"<(?:image|video|audio|bbox|ref-object)>", "", message["content"]
            )
            if not content_without_tags.strip():
                raise ValueError(f"user message at index {index} has no instruction")
        elif message.get("role") == "assistant" and not message["content"].strip():
            raise ValueError(f"assistant message at index {index} is empty")
    if messages[0].get("role") == "system":
        messages_without_system = messages[1:]
    else:
        messages_without_system = messages
    if not messages_without_system or messages_without_system[-1].get("role") != "assistant":
        raise ValueError("conversation must end with an assistant message")

    for singular, plural in (("image", "images"), ("video", "videos"), ("audio", "audios")):
        media = record.get(plural) or []
        if not isinstance(media, list):
            raise ValueError(f"{plural} must be a list")
        if placeholder_count(record, f"<{singular}>") != len(media):
            raise ValueError(f"<{singular}> count does not match {plural}")
        for item in media:
            if not isinstance(item, str) or not item:
                raise ValueError(f"{plural} entries must be non-empty strings")
            validate_media_reference(item)

    objects = record.get("objects")
    if objects is None:
        if placeholder_count(record, "<bbox>") or placeholder_count(record, "<ref-object>"):
            raise ValueError("grounding placeholders require objects")
        return
    if not isinstance(objects, dict):
        raise ValueError("objects must be an object")
    refs = objects.get("ref") or []
    boxes = objects.get("bbox") or []
    if not isinstance(refs, list) or not isinstance(boxes, list):
        raise ValueError("objects.ref and objects.bbox must be lists")
    if placeholder_count(record, "<ref-object>") != len(refs):
        raise ValueError("<ref-object> count does not match objects.ref")
    if placeholder_count(record, "<bbox>") != len(boxes):
        raise ValueError("<bbox> count does not match objects.bbox")
    bbox_type = objects.get("bbox_type", "real")
    if bbox_type not in {"real", "norm1"}:
        raise ValueError("objects.bbox_type must be real or norm1")
    for box in boxes:
        if not isinstance(box, list) or len(box) not in {2, 4}:
            raise ValueError("each bbox must contain two or four coordinates")
        numeric = [finite_float(value, "bbox coordinate") for value in box]
        if bbox_type == "norm1" and any(value < 0 or value > 1 for value in numeric):
            raise ValueError("norm1 bbox coordinates must be in [0, 1]")
    if objects and not record.get("images"):
        raise ValueError("ms-swift objects grounding requires images")


def update_record_counters(record: Mapping[str, Any], counters: Counter[str]) -> None:
    counters["converted_records"] += 1
    for media_type in ("images", "videos", "audios"):
        media = record.get(media_type) or []
        if media:
            counters[f"records_with_{media_type}"] += 1
            counters[f"referenced_{media_type}"] += len(media)
            counters["remote_media_references"] += sum(
                1 for value in media if isinstance(value, str) and is_url(value)
            )
            counters["local_media_references"] += sum(
                1 for value in media if isinstance(value, str) and not is_url(value)
            )
    objects = record.get("objects") or {}
    if objects:
        counters["grounding_records"] += 1
        counters["grounding_refs"] += len(objects.get("ref") or [])
        counters["grounding_boxes"] += len(objects.get("bbox") or [])


def make_qa_record(media_type: str, media: str, question: str, answer: str) -> dict[str, Any]:
    question = clean_text(question)
    answer = clean_text(answer)
    plural = f"{media_type}s"
    return {
        "messages": [
            {"role": "user", "content": f"<{media_type}>\n{question}"},
            {"role": "assistant", "content": answer},
        ],
        plural: [media],
    }


def direct_video_url(value: Any) -> str:
    if isinstance(value, str):
        candidate = value
    elif isinstance(value, dict):
        candidate = clean_text(value.get("gcp_url") or value.get("url") or value.get("video_url"))
    else:
        candidate = ""
    if not candidate:
        return ""
    try:
        canonical_url(candidate)
    except ValueError:
        return ""
    return candidate


def load_video_url_mappings(input_root: Path, datasets: Sequence[str]) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for dataset in datasets:
        mapping_path = input_root / dataset / "youtube_id_to_urls_mapping.json"
        if not mapping_path.is_file():
            continue
        print(f"Loading video URL mapping: {mapping_path}", flush=True)
        with mapping_path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise SystemExit(f"Video URL mapping is not an object: {mapping_path}")
        for video_id, value in raw.items():
            url = direct_video_url(value)
            if url:
                mappings.setdefault(str(video_id), url)
    return mappings


def load_video_media_index(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"Video media index does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise SystemExit("--video-media-index must contain a JSON object")
    result: dict[str, str] = {}
    for video_id, value in raw.items():
        if isinstance(value, dict):
            value = value.get("path") or value.get("url")
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"Invalid video media index entry: {video_id!r}")
        media = value.strip()
        if not is_url(media):
            media_path = Path(media).expanduser()
            media_path = (
                (path.parent / media_path).resolve()
                if not media_path.is_absolute()
                else media_path.resolve()
            )
            media = str(media_path)
        validate_media_reference(media)
        result[str(video_id)] = media
    return result


def parse_track_lineage_key(lineage_key: str) -> tuple[str, str]:
    namespace, separator, source_video_id = lineage_key.partition("::")
    if not separator or not namespace or not source_video_id:
        raise ValueError("lineage_key must have the form namespace::source_video_id")
    return namespace, source_video_id


def load_track_video_index(path: Path | None) -> dict[str, TrackMediaEntry]:
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"VideoTrack media index does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise SystemExit("--video-track-index must contain a JSON object")
    result: dict[str, TrackMediaEntry] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "windows" in value:
            raw_windows = value["windows"]
            raw_source_video_id = value.get("source_video_id")
            raw_lineage_key = value.get("lineage_key")
            has_source_video_id = raw_source_video_id is not None
            has_lineage_key = raw_lineage_key is not None
            if has_source_video_id != has_lineage_key:
                raise SystemExit(
                    f"VideoTrack index entry {key!r} must declare source_video_id and lineage_key together"
                )
            if has_source_video_id:
                if not isinstance(raw_source_video_id, str) or not raw_source_video_id.strip():
                    raise SystemExit(f"VideoTrack index entry {key!r} has invalid source_video_id")
                if not isinstance(raw_lineage_key, str) or not raw_lineage_key.strip():
                    raise SystemExit(f"VideoTrack index entry {key!r} has invalid lineage_key")
                source_video_id = raw_source_video_id.strip()
                lineage_key = raw_lineage_key.strip()
                try:
                    _, source_video = parse_track_lineage_key(source_video_id)
                    _, lineage_video_id = parse_track_lineage_key(lineage_key)
                except ValueError as exc:
                    raise SystemExit(f"VideoTrack index entry {key!r} has invalid lineage metadata: {exc}") from exc
                if lineage_video_id != source_video:
                    raise SystemExit(
                        f"VideoTrack index entry {key!r} lineage_key video {lineage_video_id!r} "
                        f"does not match source_video_id video {source_video!r}"
                    )
            else:
                source_video_id = None
                lineage_key = None
        elif isinstance(value, list):
            raw_windows = value
            source_video_id = None
            lineage_key = None
        else:
            if isinstance(value, dict) and (
                "source_video_id" in value or "lineage_key" in value
            ):
                raise SystemExit(
                    f"VideoTrack index entry {key!r} lineage metadata requires a windows list"
                )
            raw_windows = [value]
            source_video_id = None
            lineage_key = None
        if not isinstance(raw_windows, list) or not raw_windows:
            raise SystemExit(f"VideoTrack index entry {key!r} must contain at least one window")
        windows: list[dict[str, Any]] = []
        seen_bounds: set[tuple[int, int]] = set()
        seen_media: set[str] = set()
        for window_index, window in enumerate(raw_windows):
            identity = f"{key!r} window {window_index}"
            if not isinstance(window, dict) or window.get("mode") != "cropped":
                raise SystemExit(f"VideoTrack index entry {identity} must have mode='cropped'")
            path_value = window.get("path")
            url_value = window.get("url")
            has_path = isinstance(path_value, str) and bool(path_value.strip())
            has_url = isinstance(url_value, str) and bool(url_value.strip())
            if has_path == has_url:
                raise SystemExit(f"VideoTrack index entry {identity} must have exactly one path or url")
            media_value = path_value if has_path else url_value
            assert isinstance(media_value, str)
            if window.get("source_start_frame") is None or window.get("source_end_frame") is None:
                raise SystemExit(
                    f"VideoTrack index entry {identity} must declare "
                    "source_start_frame/source_end_frame"
                )
            try:
                source_start = strict_int(window["source_start_frame"], "source_start_frame")
                source_end = strict_int(window["source_end_frame"], "source_end_frame")
            except ValueError as exc:
                raise SystemExit(f"VideoTrack index entry {identity} has invalid frame bounds") from exc
            if source_start < 0 or source_end < source_start:
                raise SystemExit(f"VideoTrack index entry {identity} has invalid frame bounds")
            bounds = (source_start, source_end)
            if bounds in seen_bounds:
                raise SystemExit(f"VideoTrack index entry {key!r} repeats frame bounds {bounds}")
            seen_bounds.add(bounds)
            media = media_value.strip()
            if not is_url(media):
                media_path = Path(media).expanduser()
                if not media_path.is_absolute():
                    media_path = (path.parent / media_path).resolve()
                else:
                    media_path = media_path.resolve()
                media = str(media_path)
            if media in seen_media:
                raise SystemExit(
                    f"VideoTrack index entry {key!r} reuses one cropped media file for multiple windows"
                )
            seen_media.add(media)
            try:
                validate_media_reference(media)
            except ValueError as exc:
                raise SystemExit(f"Invalid VideoTrack media index entry {identity}: {exc}") from exc
            windows.append(
                {
                    "media": media,
                    "source_start_frame": source_start,
                    "source_end_frame": source_end,
                }
            )
        result[str(key)] = TrackMediaEntry(
            windows=tuple(
                sorted(
                    windows,
                    key=lambda item: (
                        item["source_start_frame"],
                        item["source_end_frame"],
                        item["media"],
                    ),
                )
            ),
            source_video_id=source_video_id,
            lineage_key=lineage_key,
        )
    return result


def build_pixmo_point_groups(input_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Build SHA connected components, joining hashes observed at the same URL."""
    pq = import_pyarrow_parquet()
    parent: dict[str, str] = {}
    url_owner: dict[str, str] = {}

    def find(value: str) -> str:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    data_dir = input_root / "pixmo-points" / "data"
    sources = sorted(data_dir.glob("*.parquet"))
    for source in sources:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(columns=["image_url", "image_sha256"], batch_size=65536):
            urls = batch.column(0).to_pylist()
            hashes = batch.column(1).to_pylist()
            for raw_url, raw_hash in zip(urls, hashes):
                image_hash = clean_text(raw_hash).lower()
                if not re.fullmatch(r"[0-9a-f]{64}", image_hash):
                    continue
                parent.setdefault(image_hash, image_hash)
                try:
                    url = canonical_url(clean_text(raw_url))
                except ValueError:
                    continue
                owner = url_owner.setdefault(url, image_hash)
                union(owner, image_hash)
    groups = {image_hash: find(image_hash) for image_hash in parent}
    url_groups = {url: find(image_hash) for url, image_hash in url_owner.items()}
    print(
        f"Built PixMo Points media graph: {len(groups):,} hashes, "
        f"{len(set(groups.values())):,} connected components",
        flush=True,
    )
    return groups, url_groups


def safe_tar_destination(root: Path, member_name: str) -> Path:
    pure = PurePosixPath(member_name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe tar member: {member_name}")
    destination = (root / Path(*pure.parts)).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"tar member escapes extraction root: {member_name}") from exc
    return destination


def extract_video_archive(archive_path: str, destination_root: str) -> dict[str, int]:
    archive = Path(archive_path)
    root = Path(destination_root)
    counters: Counter[str] = Counter()
    with tarfile.open(archive, mode="r:*") as tar:
        for member in tar:
            if not member.isfile() or Path(member.name).suffix.lower() not in VIDEO_SUFFIXES:
                continue
            try:
                destination = safe_tar_destination(root, member.name)
            except ValueError:
                counters["unsafe_member"] += 1
                continue
            if destination.is_file() and destination.stat().st_size == member.size:
                counters["existing"] += 1
                continue
            source = tar.extractfile(member)
            if source is None:
                counters["unreadable"] += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            try:
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                os.replace(temporary, destination)
                counters["extracted"] += 1
            finally:
                temporary.unlink(missing_ok=True)
    return dict(counters)


def maybe_extract_generated_videos(args: argparse.Namespace) -> Path:
    destination = args.output_root / "Molmo2-VideoPoint" / "videos" / "generated"
    destination.mkdir(parents=True, exist_ok=True)
    if not args.extract_generated_videos:
        return destination
    archive_dir = args.input_root / "Molmo2-VideoPoint" / "generated_videos"
    archives = sorted(archive_dir.glob("*.tar*"))
    if not archives:
        print(f"WARNING: no generated video archives found below {archive_dir}", file=sys.stderr)
        return destination
    workers = min(args.num_workers, len(archives))
    print(f"Extracting {len(archives)} generated-video archives with {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_archive = {
            executor.submit(extract_video_archive, str(archive), str(destination)): archive for archive in archives
        }
        for future in as_completed(future_to_archive):
            archive = future_to_archive[future]
            counters = future.result()
            print(f"  {archive.name}: {json.dumps(counters, sort_keys=True)}", flush=True)
    return destination


def build_generated_video_index(root: Path) -> dict[str, str]:
    by_key: dict[str, str] = {}
    by_stem: dict[str, list[str]] = {}
    if not root.is_dir():
        return by_key
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        absolute = str(path.resolve())
        relative = path.relative_to(root)
        relative_no_suffix = relative.with_suffix("").as_posix()
        by_key.setdefault(relative_no_suffix, absolute)
        by_key.setdefault(relative.as_posix(), absolute)
        if len(relative.parts) >= 2:
            by_key.setdefault(f"{relative.parts[0]}/{path.stem}", absolute)
        by_stem.setdefault(path.stem, []).append(absolute)
    for stem, paths in by_stem.items():
        if len(paths) == 1:
            by_key.setdefault(stem, paths[0])
    return by_key


def generated_video_id_candidates(video_id: str) -> list[str]:
    normalized = video_id.strip().replace("\\", "/").lstrip("./")
    path = PurePosixPath(normalized)
    without_suffix = path.with_suffix("").as_posix() if path.suffix.lower() in VIDEO_SUFFIXES else normalized
    candidates = [normalized, without_suffix, path.name, path.stem]
    if len(path.parts) >= 2:
        candidates.append(f"{path.parts[0]}/{path.stem}")
    return list(dict.fromkeys(candidates))


def resolve_video_point_media(row: Mapping[str, Any]) -> tuple[str, str]:
    source = clean_text(row.get("video_source")).lower()
    video_id = clean_text(row.get("video_id"))
    if not source or not video_id:
        raise ValueError("video_source/video_id is empty")
    media_key = f"video:{source}:{video_id}"
    if source == "youtube":
        media = _VIDEO_URLS.get(video_id, "")
    elif source == "generated":
        media = _VIDEO_URLS.get(video_id, "") or next(
            (
                _GENERATED_VIDEOS[candidate]
                for candidate in generated_video_id_candidates(video_id)
                if candidate in _GENERATED_VIDEOS
            ),
            "",
        )
    else:
        media = _VIDEO_URLS.get(video_id, "")
    if not media:
        raise ValueError(f"unresolved {source} video: {video_id}")
    validate_media_reference(media)
    return media_key, media


def normalized_percent_point(value: Any) -> tuple[list[float], bool]:
    if isinstance(value, dict):
        raw_x, raw_y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        raw_x, raw_y = value
    else:
        raise ValueError("point must contain x and y")
    x = finite_float(raw_x, "point.x")
    y = finite_float(raw_y, "point.y")
    clamped_x = min(100.0, max(0.0, x))
    clamped_y = min(100.0, max(0.0, y))
    point = [round(clamped_x / 100.0, 6), round(clamped_y / 100.0, 6)]
    return point, (clamped_x != x or clamped_y != y)


def point_text(point: Sequence[float]) -> str:
    return f"[{int(round(point[0] * 1000))}, {int(round(point[1] * 1000))}]"


def convert_capqa_row(row: Mapping[str, Any], source_path: Path) -> list[ConvertedSample]:
    video_id = clean_text(row.get("video_id"))
    if not video_id:
        raise ValueError("video_id is empty")
    video = _VIDEO_URLS.get(video_id, "")
    if not video:
        raise ValueError(f"unresolved video URL: {video_id}")
    media_key = f"video:youtube:{video_id}"
    if source_path.name.startswith("LongCapQA-"):
        qa_rows = ensure_list(row.get("qa_list"))
    else:
        qa_rows = [row]
    result: list[ConvertedSample] = []
    for qa in qa_rows:
        if not isinstance(qa, Mapping):
            raise ValueError("qa_list item is not an object")
        record = make_qa_record("video", video, qa.get("Question"), qa.get("Answer"))
        result.append(ConvertedSample(media_key, record))
    if not result:
        raise ValueError("qa_list is empty")
    return result


def format_subtitles(items: Any) -> str:
    lines: list[str] = []
    for item in ensure_list(items):
        if not isinstance(item, Mapping):
            continue
        text = clean_text(item.get("text"))
        if not text:
            continue
        start = finite_float(item.get("start"), "subtitle.start")
        end = finite_float(item.get("end"), "subtitle.end")
        lines.append(f"[{start:.3f}-{end:.3f}] {text}")
    return "\n".join(lines)


def convert_subtitleqa_row(row: Mapping[str, Any]) -> list[ConvertedSample]:
    video_id = clean_text(row.get("video_id"))
    if not video_id:
        raise ValueError("video_id is empty")
    video = _VIDEO_URLS.get(video_id, "")
    if not video:
        raise ValueError(f"unresolved video URL: {video_id}")
    question = clean_text(row.get("Question"))
    if _RUNTIME is not None and _RUNTIME.include_subtitles:
        subtitles = format_subtitles(row.get("subtitle"))
        if subtitles:
            question = f"Subtitles:\n{subtitles}\n\nQuestion: {question}"
    record = make_qa_record("video", video, question, row.get("Answer"))
    return [ConvertedSample(f"video:youtube:{video_id}", record)]


def convert_video_point_row(row: Mapping[str, Any], counters: Counter[str]) -> list[ConvertedSample]:
    media_key, video = resolve_video_point_media(row)
    question = clean_text(row.get("question"))
    label = clean_text(row.get("label"))
    if not question or not label:
        raise ValueError("question/label is empty")
    timestamps = ensure_list(row.get("raw_timestamps"))
    frames = ensure_list(row.get("raw_frames"))
    point_groups = ensure_list(row.get("points"))
    if len(point_groups) != len(timestamps):
        raise ValueError("points and raw_timestamps lengths differ")
    if frames and len(frames) != len(point_groups):
        raise ValueError("raw_frames and points lengths differ")
    lines: list[str] = []
    total_points = 0
    for index, raw_group in enumerate(point_groups):
        rendered: list[str] = []
        for raw_point in ensure_list(raw_group):
            point, clamped = normalized_percent_point(raw_point)
            if clamped:
                counters["clamped_points"] += 1
            rendered.append(point_text(point))
        if not rendered:
            continue
        timestamp = finite_float(timestamps[index], "raw_timestamp")
        frame_suffix = ""
        if frames:
            frame_suffix = f", frame {int(frames[index])}"
        lines.append(f"At {timestamp:.3f} seconds{frame_suffix}: {', '.join(rendered)}")
        total_points += len(rendered)
    if total_points:
        answer = "\n".join(lines)
    else:
        answer = "No matching points were annotated."
    prompt = (
        f"<video>\n{question}\n"
        f"Locate {label} at the relevant times. Return [x, y] points on a 0-1000 coordinate grid."
    )
    record = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "videos": [video],
    }
    count = row.get("count")
    if count is not None and int(count) != total_points:
        counters["point_count_mismatch"] += 1
    return [ConvertedSample(media_key, record)]


def resolve_track_windows(row: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    dataset = clean_text(row.get("video_dataset"))
    video_id = clean_text(row.get("video"))
    clip_id = clean_text(row.get("clip"))
    row_id = clean_text(row.get("id"))
    if not dataset or not video_id:
        raise ValueError("video_dataset/video is empty")
    specific_candidates = tuple(
        dict.fromkeys(
            key
            for key in (
                f"{dataset}::{clip_id}" if clip_id else "",
                row_id,
            )
            if key
        )
    )
    matches = [
        (key, _TRACK_VIDEOS[key])
        for key in specific_candidates
        if key in _TRACK_VIDEOS
    ]
    if not matches:
        fallback_key = f"{dataset}::{video_id}"
        if fallback_key in _TRACK_VIDEOS:
            matches = [(fallback_key, _TRACK_VIDEOS[fallback_key])]
    if not matches:
        raise ValueError(f"unresolved cropped VideoTrack media: {dataset}::{clip_id or video_id}")
    if any(windows != matches[0][1] for _, windows in matches[1:]):
        raise ValueError(
            "conflicting VideoTrack media index aliases: "
            + ", ".join(key for key, _ in matches)
        )
    indexed_entry = matches[0][1]
    if (indexed_entry.source_video_id is None) != (indexed_entry.lineage_key is None):
        raise ValueError("VideoTrack media index lineage metadata is incomplete")
    if indexed_entry.source_video_id is not None:
        source_dataset, source_video = parse_track_lineage_key(indexed_entry.source_video_id)
        if source_dataset != dataset or source_video != video_id:
            raise ValueError(
                f"VideoTrack index source_video_id {indexed_entry.source_video_id!r} "
                f"does not match annotation {dataset!r}::{video_id!r}"
            )
        assert indexed_entry.lineage_key is not None
        _, lineage_video_id = parse_track_lineage_key(indexed_entry.lineage_key)
        if lineage_video_id != source_video:
            raise ValueError(
                f"VideoTrack index lineage_key video {lineage_video_id!r} "
                f"does not match annotation video {source_video!r}"
            )
        lineage_key = indexed_entry.lineage_key
    else:
        lineage_dataset = "mose-family" if dataset.casefold() in {"mose", "mosev2"} else dataset
        lineage_key = f"{lineage_dataset}::{video_id}"
    indexed_windows = indexed_entry.windows
    source_start = strict_int(row.get("start_frame"), "start_frame")
    source_end = strict_int(row.get("end_frame"), "end_frame")
    windows = [
        entry
        for entry in indexed_windows
        if entry["source_start_frame"] >= source_start and entry["source_end_frame"] <= source_end
    ]
    if not windows:
        raise ValueError("VideoTrack media index has no windows inside the annotation frame bounds")
    max_frames = _RUNTIME.max_track_window_frames if _RUNTIME is not None else 128
    expected_start = source_start
    for entry in windows:
        window_start = strict_int(entry["source_start_frame"], "source_start_frame")
        window_end = strict_int(entry["source_end_frame"], "source_end_frame")
        if window_start != expected_start:
            raise ValueError(
                "VideoTrack media windows must be non-overlapping and continuously cover the annotation: "
                f"expected source frame {expected_start}, got {window_start}"
            )
        window_frames = window_end - window_start + 1
        if window_frames > max_frames:
            raise ValueError(
                f"VideoTrack media window has {window_frames} frames; maximum is {max_frames}"
            )
        validate_media_reference(str(entry["media"]))
        expected_start = window_end + 1
    if expected_start != source_end + 1:
        raise ValueError(
            "VideoTrack media windows do not cover the annotation through source frame "
            f"{source_end}"
        )
    lineage_dataset, lineage_video_id = parse_track_lineage_key(lineage_key)
    return f"video:track:{lineage_dataset}:{lineage_video_id}", windows


def raw_xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = value
    else:
        return None
    try:
        return finite_float(x, "track.x"), finite_float(y, "track.y")
    except ValueError:
        return None


def validate_track_structure(row: Mapping[str, Any]) -> int:
    start_frame = strict_int(row.get("start_frame"), "start_frame")
    end_frame = strict_int(row.get("end_frame"), "end_frame")
    n_frames = strict_int(row.get("n_frames"), "n_frames")
    if end_frame < start_frame or n_frames != end_frame - start_frame + 1:
        raise ValueError("VideoTrack frame bounds and n_frames are inconsistent")
    width = finite_float(row.get("w"), "w")
    height = finite_float(row.get("h"), "h")
    if width <= 0 or height <= 0:
        raise ValueError("w and h must be positive")
    points_by_object: dict[str, list[Any]] = {}
    for track in ensure_list(row.get("points")):
        if not isinstance(track, Mapping):
            raise ValueError("VideoTrack point track is not an object")
        object_id = clean_text(track.get("object_id"))
        if not object_id:
            raise ValueError("VideoTrack object_id is empty")
        if object_id in points_by_object:
            raise ValueError(f"VideoTrack object_id is duplicated: {object_id}")
        values = ensure_list(track.get("points"))
        if len(values) != n_frames:
            raise ValueError("VideoTrack point count does not match n_frames")
        for point in values:
            if point is None:
                continue
            xy = raw_xy(point)
            if xy is None:
                raise ValueError("VideoTrack visible point is not a finite [x, y] pair")
            x, y = xy
            if x < 0 or y < 0 or x > width or y > height:
                raise ValueError("VideoTrack visible point lies outside the source dimensions")
        points_by_object[object_id] = values
    if not points_by_object:
        raise ValueError("VideoTrack has no point tracks")
    segment_object_ids: set[str] = set()
    for segment_track in ensure_list(row.get("segments")):
        if not isinstance(segment_track, Mapping):
            raise ValueError("VideoTrack segment track is not an object")
        object_id = clean_text(segment_track.get("object_id"))
        if not object_id:
            raise ValueError("VideoTrack segment object_id is empty")
        if object_id in segment_object_ids:
            raise ValueError(f"VideoTrack segment object_id is duplicated: {object_id}")
        segment_object_ids.add(object_id)
        points = points_by_object.get(object_id)
        if points is None:
            raise ValueError("VideoTrack segment object has no matching point track")
        visible_from_segments: set[int] = set()
        for segment in ensure_list(segment_track.get("segments")):
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                raise ValueError("VideoTrack visibility segment must contain start/end")
            segment_start = strict_int(segment[0], "segment.start")
            segment_end = strict_int(segment[1], "segment.end")
            if segment_start < 0 or segment_end < segment_start or segment_end >= n_frames:
                raise ValueError("VideoTrack visibility segment is out of range")
            segment_frames = set(range(segment_start, segment_end + 1))
            if visible_from_segments.intersection(segment_frames):
                raise ValueError("VideoTrack visibility segments overlap")
            visible_from_segments.update(segment_frames)
        visible_from_points = {index for index, point in enumerate(points) if point is not None}
        if visible_from_segments != visible_from_points:
            raise ValueError("VideoTrack visibility segments do not match non-null points")
    if segment_object_ids != set(points_by_object):
        raise ValueError("VideoTrack point and segment object IDs differ")
    return n_frames


def convert_video_track_row(row: Mapping[str, Any], counters: Counter[str]) -> list[ConvertedSample]:
    validate_track_structure(row)
    media_key, windows = resolve_track_windows(row)
    expression = clean_text(row.get("exp"))
    if not expression:
        raise ValueError("tracking expression is empty")
    width = finite_float(row.get("w"), "w")
    height = finite_float(row.get("h"), "h")
    fps = finite_float(row.get("fps"), "fps")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("w, h, and fps must be positive")
    annotation_start = strict_int(row.get("start_frame"), "start_frame")
    tracks = ensure_list(row.get("points"))
    samples: list[ConvertedSample] = []
    for window in windows:
        window_start = strict_int(window["source_start_frame"], "source_start_frame")
        window_end = strict_int(window["source_end_frame"], "source_end_frame")
        source_offset_start = window_start - annotation_start
        source_offset_end = window_end - annotation_start
        entries: list[tuple[int, str, str]] = []
        for track in tracks:
            if not isinstance(track, Mapping):
                continue
            object_id = clean_text(track.get("object_id")) or "object"
            points = ensure_list(track.get("points"))
            for source_offset in range(source_offset_start, source_offset_end + 1):
                raw_point = points[source_offset]
                if raw_point is None:
                    continue
                xy = raw_xy(raw_point)
                if xy is None:
                    counters["invalid_track_points"] += 1
                    continue
                x, y = xy
                if x < 0 or y < 0 or x > width or y > height:
                    counters["invisible_track_points"] += 1
                    continue
                frame = source_offset - source_offset_start
                timestamp = frame / fps
                point = [x / width, y / height]
                entries.append(
                    (
                        frame,
                        object_id,
                        f"Object {object_id}, frame {frame} ({timestamp:.3f}s): {point_text(point)}",
                    )
                )
        entries.sort(key=lambda item: (item[0], item[1]))
        answer = (
            "\n".join(line for _, _, line in entries)
            if entries
            else "No visible track points were annotated."
        )
        window_frames = window_end - window_start + 1
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"<video>\nTrack {expression} throughout this clip "
                        f"(frames 0 through {window_frames - 1}). "
                        "Return [x, y] points on a 0-1000 coordinate grid with their frame numbers."
                    ),
                },
                {"role": "assistant", "content": answer},
            ],
            "videos": [str(window["media"])],
        }
        samples.append(ConvertedSample(media_key, record))
    counters["track_windows"] += len(samples)
    return samples


def convert_pixmo_cap_row(row: Mapping[str, Any]) -> list[ConvertedSample]:
    image_url = clean_text(row.get("image_url"))
    caption = clean_text(row.get("caption"))
    if not image_url or not caption:
        raise ValueError("image_url/caption is empty")
    canonical = canonical_url(image_url)
    record = {
        "messages": [
            {"role": "user", "content": "<image>\nDescribe this image in detail."},
            {"role": "assistant", "content": caption},
        ],
        "images": [image_url],
    }
    shared_hash = _PIXMO_URL_GROUPS.get(canonical)
    media_key = f"image:sha256:{shared_hash}" if shared_hash else f"image:url:{canonical}"
    return [ConvertedSample(media_key, record)]


def convert_pixmo_points_row(row: Mapping[str, Any], counters: Counter[str]) -> list[ConvertedSample]:
    image_url = clean_text(row.get("image_url"))
    image_sha256 = clean_text(row.get("image_sha256")).lower()
    label = clean_text(row.get("label"))
    if not image_url or not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
        raise ValueError("image_url or image_sha256 is invalid")
    if not label:
        raise ValueError("label is empty")
    canonical_url(image_url)
    boxes: list[list[float]] = []
    for raw_point in ensure_list(row.get("points")):
        point, clamped = normalized_percent_point(raw_point)
        if clamped:
            counters["clamped_points"] += 1
        boxes.append(point)
    raw_count = row.get("count")
    if raw_count is not None and int(raw_count) != len(boxes):
        counters["point_count_mismatch"] += 1
    assistant = "".join("<bbox>" for _ in boxes) if boxes else "No matching object is visible."
    record = {
        "messages": [
            {"role": "user", "content": "<image>\nPoint to every <ref-object>."},
            {"role": "assistant", "content": assistant},
        ],
        "images": [image_url],
        "objects": {"ref": [label], "bbox": boxes, "bbox_type": "norm1"},
    }
    group_hash = _PIXMO_POINT_GROUPS.get(image_sha256, image_sha256)
    return [ConvertedSample(f"image:sha256:{group_hash}", record)]


def image_suffix(original_path: str, data: bytes) -> str:
    suffix = Path(original_path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return suffix
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def embedded_image_payload(image: Any) -> tuple[str, bytes, str]:
    if not isinstance(image, Mapping):
        raise ValueError("embedded image entry is not an object")
    data = image.get("bytes")
    if not isinstance(data, (bytes, bytearray, memoryview)) or not data:
        raise ValueError("embedded image bytes are empty")
    payload = bytes(data)
    digest = hashlib.sha256(payload).hexdigest()
    original_path = clean_text(image.get("path"))
    return digest, payload, original_path


def spatialvlm_media_key(images: Any) -> str:
    raw_images = ensure_list(images)
    if len(raw_images) != 1:
        raise ValueError(
            "SpatialVLM multi-image rows are not supported until bbox image_id mapping is available"
        )
    digests = [embedded_image_payload(image)[0] for image in raw_images]
    return "image:sha256:" + "+".join(digests)


def write_embedded_image(image: Any) -> tuple[str, str]:
    digest, payload, original_path = embedded_image_payload(image)
    if _RUNTIME is None:
        raise RuntimeError("worker runtime is not initialized")
    images_dir = Path(_RUNTIME.output_root) / "spatialvlm" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    destination = images_dir / f"{digest}{image_suffix(original_path, payload)}"
    if destination.is_file():
        if destination.stat().st_size != len(payload):
            raise ValueError(f"existing image size mismatch: {destination}")
        return digest, str(destination.resolve())
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
        try:
            os.replace(temporary, destination)
        except FileNotFoundError:
            if not destination.is_file():
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return digest, str(destination.resolve())


def replace_normalized_boxes(text: str, boxes: list[list[float]]) -> str:
    def replace(match: re.Match[str]) -> str:
        values = [float(match.group(index)) for index in range(1, 5)]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            return match.group(0)
        x1, x2 = sorted((values[0], values[2]))
        y1, y2 = sorted((values[1], values[3]))
        boxes.append([x1, y1, x2, y2])
        return "<bbox>"

    return NORMALIZED_BOX_RE.sub(replace, text)


def spatial_message_content(content: Any, boxes: list[list[float]]) -> str:
    if isinstance(content, str):
        return replace_normalized_boxes(content, boxes)
    if not isinstance(content, list):
        raise ValueError("SpatialVLM message content must be a string or list")
    parts: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            raise ValueError("SpatialVLM content item is not an object")
        item_type = clean_text(item.get("type")).lower()
        if item_type == "image":
            parts.append("<image>")
        elif item_type == "text":
            parts.append(clean_text(item.get("text")))
        else:
            raise ValueError(f"unsupported SpatialVLM content type: {item_type}")
    return replace_normalized_boxes("".join(parts), boxes)


def normalize_role(value: Any) -> str:
    role = clean_text(value).lower()
    aliases = {"human": "user", "gpt": "assistant", "bot": "assistant"}
    role = aliases.get(role, role)
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"unsupported message role: {role}")
    return role


def convert_spatialvlm_row(row: Mapping[str, Any]) -> list[ConvertedSample]:
    raw_images = ensure_list(row.get("images"))
    if len(raw_images) != 1:
        raise ValueError(
            "SpatialVLM multi-image rows are not supported until bbox image_id mapping is available"
        )
    image_hashes: list[str] = []
    image_paths: list[str] = []
    for image in raw_images:
        digest, path = write_embedded_image(image)
        image_hashes.append(digest)
        image_paths.append(path)
    if not image_paths:
        raise ValueError("SpatialVLM row has no images")

    media_key = "image:sha256:" + "+".join(image_hashes)
    raw_messages = list(ensure_list(row.get("messages")))
    system_content = ""
    if raw_messages and isinstance(raw_messages[0], Mapping):
        if normalize_role(raw_messages[0].get("role")) == "system":
            system_boxes: list[list[float]] = []
            system_content = spatial_message_content(raw_messages.pop(0).get("content"), system_boxes)
            if system_boxes:
                raise ValueError("system message must not contain grounding boxes")
    if not raw_messages or len(raw_messages) % 2:
        raise ValueError("SpatialVLM messages must contain complete user/assistant pairs")

    samples: list[ConvertedSample] = []
    for index in range(0, len(raw_messages), 2):
        raw_user, raw_assistant = raw_messages[index : index + 2]
        if not isinstance(raw_user, Mapping) or not isinstance(raw_assistant, Mapping):
            raise ValueError("message is not an object")
        user_role = normalize_role(raw_user.get("role"))
        assistant_role = normalize_role(raw_assistant.get("role"))
        if user_role != "user" or assistant_role != "assistant":
            raise ValueError("SpatialVLM messages must alternate user and assistant")
        boxes: list[list[float]] = []
        user_content = spatial_message_content(raw_user.get("content"), boxes)
        assistant_content = spatial_message_content(raw_assistant.get("content"), boxes)
        image_tokens = user_content.count("<image>")
        if image_tokens == 0:
            user_content = "<image>" * len(image_paths) + user_content
        elif image_tokens != len(image_paths):
            raise ValueError("SpatialVLM user image token count does not match embedded images")
        messages: list[dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.extend(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        )
        record: dict[str, Any] = {"messages": messages, "images": image_paths}
        if boxes:
            record["objects"] = {"ref": [], "bbox": boxes, "bbox_type": "norm1"}
        samples.append(ConvertedSample(media_key, record))
    return samples


def convert_source_row(
    dataset: str,
    row: Mapping[str, Any],
    source_path: Path,
    counters: Counter[str],
) -> list[ConvertedSample]:
    if dataset == "Molmo2-VideoCapQA":
        return convert_capqa_row(row, source_path)
    if dataset == "Molmo2-VideoPoint":
        return convert_video_point_row(row, counters)
    if dataset == "Molmo2-VideoSubtitleQA":
        return convert_subtitleqa_row(row)
    if dataset == "Molmo2-VideoTrack":
        return convert_video_track_row(row, counters)
    if dataset == "pixmo-cap":
        return convert_pixmo_cap_row(row)
    if dataset == "pixmo-points":
        return convert_pixmo_points_row(row, counters)
    if dataset == "spatialvlm":
        return convert_spatialvlm_row(row)
    raise ValueError(f"unsupported dataset: {dataset}")


def rejection_identity(dataset: str, row: Mapping[str, Any]) -> dict[str, str]:
    keys_by_dataset = {
        "Molmo2-VideoCapQA": ("video_id",),
        "Molmo2-VideoPoint": ("video_source", "video_id"),
        "Molmo2-VideoSubtitleQA": ("video_id",),
        "Molmo2-VideoTrack": ("video_dataset", "video", "id"),
        "pixmo-cap": ("image_url",),
        "pixmo-points": ("image_sha256", "image_url"),
        "spatialvlm": (),
    }
    return {key: clean_text(row.get(key))[:500] for key in keys_by_dataset[dataset] if row.get(key) is not None}


def iter_task_rows(parquet, task: RowGroupTask) -> Iterator[dict[str, Any]]:
    remaining = task.row_limit
    batch_sizes = {"spatialvlm": 32, "Molmo2-VideoTrack": 128}
    batch_size = batch_sizes.get(task.dataset, 8192)
    for batch in parquet.iter_batches(row_groups=[task.row_group], batch_size=batch_size):
        rows = batch.to_pylist()
        if remaining is not None:
            rows = rows[:remaining]
        yield from rows
        if remaining is not None:
            remaining -= len(rows)
            if remaining <= 0:
                break


def write_rejection(
    stream,
    counters: Counter[str],
    task: RowGroupTask,
    source_path: Path,
    row_offset: int,
    row: Mapping[str, Any],
    exc: Exception,
    derived_index: int | None = None,
) -> None:
    counters[f"rejected_{type(exc).__name__}"] += 1
    if counters["rejected_examples"] >= _RUNTIME.max_rejected_examples:
        return
    rejected: dict[str, Any] = {
        "dataset": task.dataset,
        "source": str(source_path),
        "row_group": task.row_group,
        "row_offset": row_offset,
        "identity": rejection_identity(task.dataset, row),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if derived_index is not None:
        rejected["derived_index"] = derived_index
    stream.write(json.dumps(rejected, ensure_ascii=False, separators=(",", ":")) + "\n")
    counters["rejected_examples"] += 1


def process_row_group(task: RowGroupTask) -> WorkerResult:
    if _RUNTIME is None:
        raise RuntimeError("worker runtime is not initialized")
    pq = import_pyarrow_parquet()
    source_path = Path(task.source_path)
    fragment_dir = Path(task.fragment_dir)
    fragment_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{task.ordinal:06d}"
    train_path = fragment_dir / f"{prefix}.train.jsonl"
    val_path = fragment_dir / f"{prefix}.val.jsonl"
    groups_path = fragment_dir / f"{prefix}.groups.tsv"
    rejected_path = fragment_dir / f"{prefix}.rejected.jsonl"
    counters: Counter[str] = Counter()
    local_groups: dict[str, str] = {}

    parquet = pq.ParquetFile(source_path)

    with (
        train_path.open("w", encoding="utf-8", newline="\n") as train_stream,
        val_path.open("w", encoding="utf-8", newline="\n") as val_stream,
        rejected_path.open("w", encoding="utf-8", newline="\n") as rejected_stream,
    ):
        for row_offset, row in enumerate(iter_task_rows(parquet, task)):
            counters["source_rows"] += 1
            row_had_rejection = False
            try:
                samples = convert_source_row(task.dataset, row, source_path, counters)
            except Exception as exc:
                row_had_rejection = True
                counters["rejected_source_conversion"] += 1
                write_rejection(
                    rejected_stream, counters, task, source_path, row_offset, row, exc
                )
                samples = []
            counters["derived_records"] += len(samples)
            if len(samples) > 1:
                counters["expanded_rows"] += len(samples) - 1
            for derived_index, sample in enumerate(samples):
                try:
                    if (
                        task.dataset == "spatialvlm"
                        and task.forced_split == "train"
                        and sample.media_key in _SPATIALVLM_TEST_MEDIA_KEYS
                    ):
                        counters["excluded_train_records_with_test_image_hash"] += 1
                        continue
                    split = task.forced_split or split_for_media(
                        sample.media_key, _RUNTIME.seed, _RUNTIME.val_ratio
                    )
                    if split not in {"train", "val"}:
                        raise AssertionError(f"invalid forced split: {split}")
                    validate_record(sample.record)
                    update_record_counters(sample.record, counters)
                    previous = local_groups.setdefault(sample.media_key, split)
                    if previous != split:
                        raise AssertionError("deterministic split returned conflicting assignments")
                    stream = val_stream if split == "val" else train_stream
                    stream.write(json.dumps(sample.record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counters[f"written_{split}"] += 1
                    if task.dataset == "spatialvlm" and task.forced_split:
                        official_source = "test" if task.forced_split == "val" else "train"
                        counters[
                            f"official_{official_source}_records_written_to_{split}"
                        ] += 1
                except Exception as exc:
                    row_had_rejection = True
                    counters["rejected_derived_records"] += 1
                    write_rejection(
                        rejected_stream,
                        counters,
                        task,
                        source_path,
                        row_offset,
                        row,
                        exc,
                        derived_index,
                    )
            if row_had_rejection:
                counters["rejected_rows"] += 1

    with groups_path.open("w", encoding="utf-8", newline="\n") as stream:
        for media_key, split in sorted(local_groups.items()):
            stream.write(f"{split}\t{media_key}\n")
    return WorkerResult(
        ordinal=task.ordinal,
        source_path=str(source_path),
        train_path=str(train_path),
        val_path=str(val_path),
        groups_path=str(groups_path),
        rejected_path=str(rejected_path),
        counters=dict(counters),
    )


def discover_sources(
    dataset: str,
    input_root: Path,
    video_track_sources: Sequence[str] | None = None,
) -> list[Path]:
    root = input_root / dataset
    if not root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {root}")
    if dataset == "Molmo2-VideoCapQA":
        sources = sorted((root / "data").glob("CapQA-*.parquet"))
        sources += sorted((root / "data").glob("LongCapQA-*.parquet"))
    elif dataset == "Molmo2-VideoPoint":
        authoritative = root / "data" / "train-00000-of-00001.parquet"
        sources = [authoritative] if authoritative.is_file() else []
    elif dataset in {"Molmo2-VideoSubtitleQA", "pixmo-cap", "pixmo-points", "spatialvlm"}:
        sources = sorted((root / "data").glob("*.parquet"))
    elif dataset == "Molmo2-VideoTrack":
        sources = sorted(root.rglob("*.parquet"))
        if video_track_sources:
            data_root = root / "data"
            sources_by_group: dict[str, list[Path]] = {}
            for source in sources:
                try:
                    relative = source.relative_to(data_root)
                except ValueError:
                    continue
                if len(relative.parts) < 2:
                    continue
                sources_by_group.setdefault(relative.parts[0].casefold(), []).append(source)
            missing = sorted(set(video_track_sources) - sources_by_group.keys())
            if missing:
                available = ", ".join(sorted(sources_by_group)) or "none"
                raise SystemExit(
                    f"Unknown or unavailable Molmo2-VideoTrack source(s): {', '.join(missing)}. "
                    f"Available sources: {available}"
                )
            selected_groups = set(video_track_sources)
            sources = sorted(
                source
                for group, group_sources in sources_by_group.items()
                if group in selected_groups
                for source in group_sources
            )
    else:
        sources = []
    if not sources:
        raise SystemExit(f"No authoritative parquet sources found for {dataset} below {root}")
    return sources


def spatialvlm_official_split(source: Path) -> str:
    name = source.name.casefold()
    if name.startswith("train-"):
        return "train"
    if name.startswith(("test-", "validation-", "val-")):
        return "val"
    raise ValueError(
        f"SpatialVLM parquet filename does not declare an official split: {source.name}"
    )


def build_spatialvlm_test_media_keys(sources: Sequence[Path]) -> tuple[set[str], int]:
    pq = import_pyarrow_parquet()
    media_keys: set[str] = set()
    source_rows = 0
    for source in sources:
        if spatialvlm_official_split(source) != "val":
            continue
        parquet = pq.ParquetFile(source)
        if "images" not in parquet.schema_arrow.names:
            raise ValueError(f"SpatialVLM test source has no images column: {source}")
        for batch in parquet.iter_batches(batch_size=128, columns=["images"]):
            for row in batch.to_pylist():
                source_rows += 1
                media_keys.add(spatialvlm_media_key(row.get("images")))
    if not source_rows:
        raise ValueError("SpatialVLM has no official test rows to reserve")
    return media_keys, source_rows


def required_columns(dataset: str, source: Path) -> set[str]:
    if dataset == "Molmo2-VideoCapQA":
        if source.name.startswith("LongCapQA-"):
            return {"video_id", "qa_list"}
        return {"video_id", "Question", "Answer"}
    if dataset == "Molmo2-VideoPoint":
        return {"video_id", "video_source", "question", "label", "points", "raw_timestamps"}
    if dataset == "Molmo2-VideoSubtitleQA":
        return {"video_id", "Question", "Answer", "subtitle"}
    if dataset == "Molmo2-VideoTrack":
        return {
            "id",
            "video",
            "clip",
            "video_dataset",
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
    if dataset == "pixmo-cap":
        return {"image_url", "caption"}
    if dataset == "pixmo-points":
        return {"image_url", "image_sha256", "points", "label"}
    if dataset == "spatialvlm":
        return {"messages", "images"}
    return set()


def build_tasks(
    dataset: str,
    sources: Sequence[Path],
    fragment_dir: Path,
    max_source_rows: int | None,
) -> tuple[list[RowGroupTask], int]:
    pq = import_pyarrow_parquet()
    tasks: list[RowGroupTask] = []
    total_source_rows = 0
    ordinal = 0
    if max_source_rows is None:
        source_budgets: list[int | None] = [None] * len(sources)
    else:
        base, extra = divmod(max_source_rows, len(sources))
        source_budgets = [max(1, base + (1 if index < extra else 0)) for index in range(len(sources))]
    for source, source_budget in zip(sources, source_budgets):
        remaining = source_budget
        parquet = pq.ParquetFile(source)
        columns = set(parquet.schema_arrow.names)
        missing = required_columns(dataset, source) - columns
        if missing:
            raise SystemExit(f"{source} is missing required columns: {sorted(missing)}")
        for row_group in range(parquet.metadata.num_row_groups):
            row_count = parquet.metadata.row_group(row_group).num_rows
            if remaining is not None and remaining <= 0:
                break
            row_limit = None if remaining is None else min(row_count, remaining)
            tasks.append(
                RowGroupTask(
                    dataset=dataset,
                    source_path=str(source),
                    row_group=row_group,
                    ordinal=ordinal,
                    row_limit=row_limit,
                    fragment_dir=str(fragment_dir),
                    forced_split=(
                        spatialvlm_official_split(source)
                        if dataset == "spatialvlm"
                        else None
                    ),
                )
            )
            total_source_rows += row_count if row_limit is None else row_limit
            ordinal += 1
            if remaining is not None:
                remaining -= row_limit
    if not tasks:
        raise SystemExit(f"No parquet row groups selected for {dataset}")
    return tasks, total_source_rows


def initialize_worker(
    runtime: RuntimeConfig,
    video_urls: dict[str, str],
    generated_videos: dict[str, str],
    track_videos: dict[str, TrackMediaEntry],
    pixmo_point_groups: dict[str, str],
    pixmo_url_groups: dict[str, str],
    spatialvlm_test_media_keys: set[str],
) -> None:
    global _RUNTIME, _VIDEO_URLS, _GENERATED_VIDEOS, _TRACK_VIDEOS
    global _PIXMO_POINT_GROUPS, _PIXMO_URL_GROUPS, _SPATIALVLM_TEST_MEDIA_KEYS
    _RUNTIME = runtime
    _VIDEO_URLS = video_urls
    _GENERATED_VIDEOS = generated_videos
    _TRACK_VIDEOS = track_videos
    _PIXMO_POINT_GROUPS = pixmo_point_groups
    _PIXMO_URL_GROUPS = pixmo_url_groups
    _SPATIALVLM_TEST_MEDIA_KEYS = spatialvlm_test_media_keys


def run_tasks(tasks: Sequence[RowGroupTask], workers: int) -> list[WorkerResult]:
    if _RUNTIME is None:
        raise RuntimeError("runtime is not initialized")
    if workers == 1:
        return [process_row_group(task) for task in tasks]
    kwargs: dict[str, Any] = {
        "max_workers": min(workers, len(tasks)),
        "initializer": initialize_worker,
        "initargs": (
            _RUNTIME,
            _VIDEO_URLS,
            _GENERATED_VIDEOS,
            _TRACK_VIDEOS,
            _PIXMO_POINT_GROUPS,
            _PIXMO_URL_GROUPS,
            _SPATIALVLM_TEST_MEDIA_KEYS,
        ),
    }
    if os.name == "posix":
        kwargs["mp_context"] = multiprocessing.get_context("fork")
    results: list[WorkerResult] = []
    with ProcessPoolExecutor(**kwargs) as executor:
        future_to_task = {executor.submit(process_row_group, task): task for task in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"worker failed for {task.source_path} row group {task.row_group}"
                ) from exc
            completed += 1
            if completed == len(tasks) or completed % max(1, len(tasks) // 20) == 0:
                print(f"  completed row groups: {completed}/{len(tasks)}", flush=True)
    return sorted(results, key=lambda result: result.ordinal)


def append_file(source_path: str, destination) -> None:
    with Path(source_path).open("r", encoding="utf-8") as source:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
        "rejected": output_dir / "rejected.jsonl",
        "media_groups": output_dir / "media_groups.tsv",
        "report": output_dir / "conversion_report.json",
    }


def check_output_preconditions(dataset: str, args: argparse.Namespace) -> None:
    output_dir = args.output_root / dataset
    for path in dataset_output_paths(output_dir).values():
        backup = path.with_name(f".{path.name}.backup")
        if backup.exists():
            raise SystemExit(f"Stale rollback backup requires manual inspection: {backup}")
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists; pass --overwrite to replace it: {path}")


def merge_results(
    dataset: str,
    sources: Sequence[Path],
    results: Sequence[WorkerResult],
    output_dir: Path,
    args: argparse.Namespace,
    selected_source_rows: int,
    spatialvlm_test_source_rows: int = 0,
) -> dict[str, Any]:
    final_paths = dataset_output_paths(output_dir)
    temporary_paths = {name: path.with_name(f".{path.name}.{os.getpid()}.tmp") for name, path in final_paths.items()}
    counters: Counter[str] = Counter()
    source_counters: dict[str, Counter[str]] = {}
    media_splits: dict[str, str] = {}
    leakage_examples: list[dict[str, str]] = []
    try:
        with (
            temporary_paths["train"].open("w", encoding="utf-8", newline="\n") as train_stream,
            temporary_paths["val"].open("w", encoding="utf-8", newline="\n") as val_stream,
            temporary_paths["rejected"].open("w", encoding="utf-8", newline="\n") as rejected_stream,
        ):
            for result in sorted(results, key=lambda item: item.ordinal):
                append_file(result.train_path, train_stream)
                append_file(result.val_path, val_stream)
                append_file(result.rejected_path, rejected_stream)
                counters.update(result.counters)
                source_counters.setdefault(result.source_path, Counter()).update(result.counters)
                with Path(result.groups_path).open("r", encoding="utf-8") as groups_stream:
                    for line in groups_stream:
                        split, media_key = line.rstrip("\n").split("\t", 1)
                        previous = media_splits.setdefault(media_key, split)
                        if previous != split:
                            leakage_examples.append(
                                {"media_key": media_key, "first_split": previous, "second_split": split}
                            )
                            if len(leakage_examples) >= 20:
                                break
                if leakage_examples:
                    break
        if leakage_examples:
            raise RuntimeError(f"media leakage detected: {leakage_examples[:3]}")
        converted_records = counters["written_train"] + counters["written_val"]
        rejected_rows = counters["rejected_rows"]
        reject_ratio = rejected_rows / counters["source_rows"] if counters["source_rows"] else 0.0
        if converted_records == 0 and not args.allow_empty_dataset:
            raise RuntimeError(
                f"{dataset} produced zero records; inspect rejected fragments or provide required media"
            )
        if reject_ratio > args.max_reject_ratio:
            raise RuntimeError(
                f"{dataset} reject ratio {reject_ratio:.3%} exceeds --max-reject-ratio "
                f"{args.max_reject_ratio:.3%}"
            )
        source_reports: dict[str, dict[str, Any]] = {}
        for source_path, source_counter in source_counters.items():
            source_rows = source_counter["source_rows"]
            source_reject_ratio = source_counter["rejected_rows"] / source_rows if source_rows else 0.0
            source_reports[source_path] = {
                "counters": dict(sorted(source_counter.items())),
                "reject_ratio": source_reject_ratio,
            }
            if source_reject_ratio > args.max_reject_ratio:
                raise RuntimeError(
                    f"{dataset} source {source_path} reject ratio {source_reject_ratio:.3%} exceeds "
                    f"--max-reject-ratio {args.max_reject_ratio:.3%}"
                )
        split_media_counts = Counter(media_splits.values())
        with temporary_paths["media_groups"].open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("split\tmedia_key\n")
            for media_key, split in sorted(media_splits.items()):
                stream.write(f"{split}\t{media_key}\n")
        spatialvlm_train_test_hash_overlap = sum(
            1
            for media_key, split in media_splits.items()
            if split == "train" and media_key in _SPATIALVLM_TEST_MEDIA_KEYS
        )
        if dataset == "spatialvlm":
            spatialvlm_leakage = {
                "official_test_records_written_to_train": counters[
                    "official_test_records_written_to_train"
                ],
                "official_train_records_written_to_val": counters[
                    "official_train_records_written_to_val"
                ],
                "post_filter_train_test_hash_overlap": spatialvlm_train_test_hash_overlap,
            }
            if any(spatialvlm_leakage.values()):
                raise RuntimeError(
                    f"SpatialVLM official split leakage detected: {spatialvlm_leakage}"
                )
        output_sha256 = {
            key: file_sha256(temporary_paths[key])
            for key in ("train", "val", "rejected", "media_groups")
        }
        report = {
            "dataset": dataset,
            "input_files": [str(path) for path in sources],
            "output_files": {key: str(value) for key, value in final_paths.items() if key != "report"},
            "output_sha256": output_sha256,
            "configuration": {
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "num_workers": args.num_workers,
                "max_source_rows": args.max_source_rows,
                "include_subtitles": args.include_subtitles,
                "max_track_window_frames": args.max_track_window_frames,
                "video_track_sources": (
                    args.video_track_sources if dataset == "Molmo2-VideoTrack" else None
                ),
                "allow_unverified_remote_media": args.allow_unverified_remote_media,
                "max_reject_ratio": args.max_reject_ratio,
                "spatialvlm_split_policy": (
                    "preserve_official_train_test_and_reserve_test_image_sha256"
                    if dataset == "spatialvlm"
                    else None
                ),
            },
            "selected_source_rows": selected_source_rows,
            "counters": dict(sorted(counters.items())),
            "source_reports": source_reports,
            "unique_media": {
                "total": len(media_splits),
                "train": split_media_counts["train"],
                "val": split_media_counts["val"],
                "cross_split_leakage": 0,
            },
            "spatialvlm_official_split_audit": (
                {
                    "official_test_source_rows": spatialvlm_test_source_rows,
                    "test_media_hashes_reserved": len(_SPATIALVLM_TEST_MEDIA_KEYS),
                    "train_records_excluded_for_test_hash": counters[
                        "excluded_train_records_with_test_image_hash"
                    ],
                    "official_test_records_written_to_train": counters[
                        "official_test_records_written_to_train"
                    ],
                    "official_test_records_written_to_val": counters[
                        "official_test_records_written_to_val"
                    ],
                    "official_train_records_written_to_train": counters[
                        "official_train_records_written_to_train"
                    ],
                    "official_train_records_written_to_val": counters[
                        "official_train_records_written_to_val"
                    ],
                    "post_filter_train_test_hash_overlap": spatialvlm_train_test_hash_overlap,
                }
                if dataset == "spatialvlm"
                else None
            ),
            "reject_ratio": reject_ratio,
            "media_readiness": (
                "unverified_remote_references"
                if counters["remote_media_references"]
                else (
                    "local_files_exist_not_decoded"
                    if counters["local_media_references"]
                    else "no_converted_media"
                )
            ),
        }
        with temporary_paths["report"].open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        backups: dict[Path, Path] = {}
        installed: list[Path] = []
        try:
            for destination in final_paths.values():
                if destination.exists():
                    backup = destination.with_name(f".{destination.name}.backup")
                    os.replace(destination, backup)
                    backups[destination] = backup
            for name, destination in final_paths.items():
                os.replace(temporary_paths[name], destination)
                installed.append(destination)
        except Exception:
            for destination in installed:
                destination.unlink(missing_ok=True)
            for destination, backup in backups.items():
                if backup.exists():
                    os.replace(backup, destination)
            raise
        else:
            for backup in backups.values():
                backup.unlink(missing_ok=True)
        return report
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)


def prepare_output_dir(dataset: str, output_root: Path) -> Path:
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    media_subdir = "images" if dataset in {"pixmo-cap", "pixmo-points", "spatialvlm"} else "videos"
    (output_dir / media_subdir).mkdir(parents=True, exist_ok=True)
    return output_dir


def convert_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    global _SPATIALVLM_TEST_MEDIA_KEYS
    sources = discover_sources(
        dataset,
        args.input_root,
        args.video_track_sources if dataset == "Molmo2-VideoTrack" else None,
    )
    if dataset == "spatialvlm":
        _SPATIALVLM_TEST_MEDIA_KEYS, test_source_rows = build_spatialvlm_test_media_keys(
            sources
        )
        print(
            f"[spatialvlm] reserved official test rows={test_source_rows:,} "
            f"unique_image_hashes={len(_SPATIALVLM_TEST_MEDIA_KEYS):,}",
            flush=True,
        )
    else:
        _SPATIALVLM_TEST_MEDIA_KEYS = set()
        test_source_rows = 0
    output_dir = prepare_output_dir(dataset, args.output_root)
    fragment_dir = Path(tempfile.mkdtemp(prefix=".conversion-fragments-", dir=output_dir))
    completed_successfully = False
    try:
        tasks, selected_source_rows = build_tasks(
            dataset, sources, fragment_dir, args.max_source_rows
        )
        print(
            f"[{dataset}] {len(sources)} parquet files, {len(tasks)} row groups, "
            f"{selected_source_rows:,} selected source rows",
            flush=True,
        )
        results = run_tasks(tasks, args.num_workers)
        report = merge_results(
            dataset,
            sources,
            results,
            output_dir,
            args,
            selected_source_rows,
            spatialvlm_test_source_rows=test_source_rows,
        )
        counters = report["counters"]
        print(
            f"[{dataset}] train={counters.get('written_train', 0):,} "
            f"val={counters.get('written_val', 0):,} "
            f"rejected={counters.get('rejected_rows', 0):,} "
            f"media={report['unique_media']['total']:,}",
            flush=True,
        )
        completed_successfully = True
        return report
    finally:
        if not args.keep_temp and completed_successfully:
            shutil.rmtree(fragment_dir, ignore_errors=True)
        elif not completed_successfully:
            print(f"Retained failed conversion fragments: {fragment_dir}", file=sys.stderr, flush=True)


def audit_global_media(output_root: Path, datasets: Sequence[str] | None = None) -> dict[str, int]:
    assignments: dict[str, str] = {}
    dataset_memberships = 0
    manifest_count = 0
    if datasets is None:
        paths = sorted(output_root.glob("*/media_groups.tsv"))
    else:
        paths = [output_root / dataset / "media_groups.tsv" for dataset in datasets]
    for path in paths:
        if not path.is_file():
            continue
        manifest_count += 1
        with path.open("r", encoding="utf-8") as stream:
            header = stream.readline().rstrip("\n")
            if header != "split\tmedia_key":
                raise RuntimeError(f"invalid media group manifest header: {path}")
            for line in stream:
                split, media_key = line.rstrip("\n").split("\t", 1)
                previous = assignments.setdefault(media_key, split)
                if previous != split:
                    raise RuntimeError(
                        f"cross-dataset media leakage for {media_key}: {previous} vs {split}"
                    )
                dataset_memberships += 1
    split_counts = Counter(assignments.values())
    return {
        "manifests": manifest_count,
        "unique_media": len(assignments),
        "dataset_media_memberships": dataset_memberships,
        "train_media": split_counts["train"],
        "val_media": split_counts["val"],
        "cross_dataset_leakage": 0,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "conversion_summary.json"
    summary_backup = summary_path.with_name(f".{summary_path.name}.backup")
    if summary_backup.exists():
        raise SystemExit(f"Stale rollback backup requires manual inspection: {summary_backup}")
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; pass --overwrite to replace it: {summary_path}")
    for dataset in args.datasets:
        check_output_preconditions(dataset, args)
    if (
        "Molmo2-VideoTrack" in args.datasets
        and args.video_track_index is None
        and not args.allow_empty_dataset
    ):
        raise SystemExit(
            "Molmo2-VideoTrack requires --video-track-index with pre-cropped media entries"
        )

    runtime = RuntimeConfig(
        input_root=str(args.input_root),
        output_root=str(args.output_root),
        val_ratio=args.val_ratio,
        seed=args.seed,
        include_subtitles=args.include_subtitles,
        max_rejected_examples=args.max_rejected_examples,
        allow_unverified_remote_media=args.allow_unverified_remote_media,
        max_track_window_frames=args.max_track_window_frames,
    )
    video_urls = load_video_url_mappings(args.input_root, args.datasets)
    video_urls.update(load_video_media_index(args.video_media_index))
    generated_videos: dict[str, str] = {}
    if "Molmo2-VideoPoint" in args.datasets:
        generated_root = maybe_extract_generated_videos(args)
        generated_videos = build_generated_video_index(generated_root)
        print(f"Indexed {len(generated_videos):,} generated video keys", flush=True)
    track_videos = load_track_video_index(args.video_track_index)
    if any(dataset in args.datasets for dataset in ("pixmo-cap", "pixmo-points")):
        pixmo_point_groups, pixmo_url_groups = build_pixmo_point_groups(args.input_root)
    else:
        pixmo_point_groups, pixmo_url_groups = {}, {}
    initialize_worker(
        runtime,
        video_urls,
        generated_videos,
        track_videos,
        pixmo_point_groups,
        pixmo_url_groups,
        set(),
    )

    summary: dict[str, Any] = {}
    for dataset in args.datasets:
        summary[dataset] = convert_dataset(dataset, args)
    summary_document = {
        "datasets": summary,
        "global_media_audit": audit_global_media(args.output_root, args.datasets),
    }
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(summary_document, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        if summary_path.exists():
            os.replace(summary_path, summary_backup)
        try:
            os.replace(temporary, summary_path)
        except Exception:
            if summary_backup.exists():
                os.replace(summary_backup, summary_path)
            raise
        summary_backup.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
