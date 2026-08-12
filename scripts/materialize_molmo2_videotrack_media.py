#!/usr/bin/env python3
"""Materialize audited Molmo2-VideoTrack frame archives as cropped MP4 windows.

The source archives are never modified.  Frame ranges in the generated index are
zero-based and inclusive, matching the Molmo2-VideoTrack parquet annotations.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import threading
import uuid
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Protocol, Sequence


DEFAULT_DATASET_ROOT = Path("/mnt/luojunkun/stage1/dataset/Molmo2-VideoTrack")
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack/videos"
)
DATASET_ORDER = ("dancetrack", "soccernet", "mose", "mosev2", "vipseg")
MOSE_INNER_MEMBER = "MOSE_release/train.tar.gz"

DANCE_RE = re.compile(r"^(train[12])/([^/]+)/img1/([0-9]{8})\.jpg$")
SOCCERNET_RE = re.compile(r"^train/([^/]+)/img1/([0-9]{6})\.jpg$")
MOSE_RE = re.compile(r"^train/JPEGImages/([^/]+)/([0-9]{5})\.jpg$")
VIPSEG_RE = re.compile(r"^VIPSeg/imgs/([^/]+)/([0-9]+)\.jpg$")
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MaterializationError(RuntimeError):
    """A fail-closed source, mapping, or media validation error."""


class FrameValidationError(MaterializationError):

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


class ScratchLimitExceeded(MaterializationError):
    pass


@dataclass(frozen=True)
class ClipSpec:
    dataset: str
    video: str
    clip: str
    start_frame: int
    end_frame: int
    n_frames: int
    fps: float
    width: int
    height: int

    @property
    def key(self) -> str:
        return f"{self.dataset}::{self.clip}"

    @property
    def source_video_id(self) -> str:
        return f"{self.dataset}::{self.video}"

    @property
    def lineage_key(self) -> str:
        family = "mose-family" if self.dataset in {"mose", "mosev2"} else self.dataset
        return f"{family}::{self.video}"


@dataclass(frozen=True, order=True)
class FrameWindow:
    start_frame: int
    end_frame: int

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class VideoPlan:
    dataset: str
    video: str
    fps: float
    width: int
    height: int
    clips: tuple[ClipSpec, ...]
    clip_windows: dict[str, tuple[FrameWindow, ...]]
    windows: tuple[FrameWindow, ...]


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    size: int
    mtime_ns: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "mtime_ns": self.mtime_ns}


@dataclass(frozen=True)
class SourceDefinition:
    dataset: str
    parquet_path: Path
    kind: str
    archive_paths: tuple[Path, ...]
    archive_labels: tuple[str, ...] = ()
    checksum_path: Path | None = None

    @property
    def files(self) -> tuple[Path, ...]:
        checksum_files = () if self.checksum_path is None else (self.checksum_path,)
        return (self.parquet_path, *self.archive_paths, *checksum_files)


@dataclass
class WindowState:
    state: str
    final_path: Path
    staged_path: Path | None = None
    reason: str | None = None
    detail: str | None = None
    uploaded_size_bytes: int | None = None
    uploaded_sha256: str | None = None


@dataclass
class VideoCache:
    plan: VideoPlan
    required_frames: set[int]
    directory: Path
    seen_frames: set[int] = field(default_factory=set)
    stored_frames: set[int] = field(default_factory=set)
    frame_errors: dict[int, tuple[str, str]] = field(default_factory=dict)
    bytes_used: int = 0
    submitted: bool = False


@dataclass
class VideoEncodingResult:
    video: str
    states: dict[FrameWindow, WindowState]


@dataclass
class DatasetOutcome:
    dataset: str
    entries: dict[str, dict[str, Any]]
    staged_files: list[tuple[Path, Path]]
    accounting: dict[str, Any]
    failures: list[dict[str, Any]]
    fatal: bool = False


@dataclass(frozen=True)
class VideoProbe:
    codec_name: str
    pixel_format: str
    width: int
    height: int
    fps: float
    n_frames: int


@dataclass(frozen=True)
class UploadReceipt:
    size_bytes: int
    sha256: str


class MediaTool(Protocol):

    def encode(
        self,
        frame_pattern: Path,
        start_frame: int,
        n_frames: int,
        fps: float,
        pixel_format: str,
        destination: Path,
    ) -> None:
        ...

    def probe(self, path: Path) -> VideoProbe:
        ...

    def decode_frame(self, path: Path, local_frame: int, destination: Path) -> None:
        ...


class FfmpegMediaTool:

    def __init__(self, ffmpeg: str, ffprobe: str, threads: int):
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.threads = threads

    def encode(
        self,
        frame_pattern: Path,
        start_frame: int,
        n_frames: int,
        fps: float,
        pixel_format: str,
        destination: Path,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.mp4"
        )
        command = [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-xerror",
            "-framerate",
            format_fps(fps),
            "-start_number",
            str(start_frame),
            "-i",
            str(frame_pattern),
            "-frames:v",
            str(n_frames),
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            str(self.threads),
            "-pix_fmt",
            pixel_format,
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "ffmpeg failed").strip()
                raise MaterializationError(f"ffmpeg exit={result.returncode}: {detail[-2000:]}")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise MaterializationError("ffmpeg did not create a non-empty MP4")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def probe(self, path: Path) -> VideoProbe:
        command = [
            self.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames,nb_frames",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "ffprobe failed").strip()
            raise MaterializationError(f"ffprobe exit={result.returncode}: {detail[-2000:]}")
        try:
            payload = json.loads(result.stdout)
            streams = payload["streams"]
            if not isinstance(streams, list) or len(streams) != 1:
                raise ValueError("expected exactly one selected video stream")
            stream = streams[0]
            raw_frames = stream.get("nb_read_frames")
            if raw_frames in {None, "N/A"}:
                raw_frames = stream.get("nb_frames")
            if raw_frames in {None, "N/A"}:
                raise ValueError("ffprobe did not return a frame count")
            raw_fps = stream.get("avg_frame_rate")
            fps = float(Fraction(str(raw_fps)))
            return VideoProbe(
                codec_name=str(stream["codec_name"]),
                pixel_format=str(stream["pix_fmt"]),
                width=int(stream["width"]),
                height=int(stream["height"]),
                fps=fps,
                n_frames=int(raw_frames),
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
            raise MaterializationError(f"invalid ffprobe JSON for {path}: {exc}") from exc

    def decode_frame(self, path: Path, local_frame: int, destination: Path) -> None:
        if local_frame < 0:
            raise ValueError("local_frame cannot be negative")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.png"
        )
        command = [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{local_frame})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(temporary),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "ffmpeg frame decode failed").strip()
                raise MaterializationError(
                    f"ffmpeg frame decode exit={result.returncode}: {detail[-2000:]}"
                )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise MaterializationError(
                    f"ffmpeg did not decode local frame {local_frame} from {path}"
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


class ScratchBudget:

    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError("scratch limit must be positive")
        self.limit = limit
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def reserve(self, size: int) -> None:
        if size < 0:
            raise ValueError("scratch reservation cannot be negative")
        with self._lock:
            if self.current + size > self.limit:
                raise ScratchLimitExceeded(
                    f"scratch byte cap exceeded: requested={size}, current={self.current}, limit={self.limit}"
                )
            self.current += size
            self.peak = max(self.peak, self.current)

    def release(self, size: int) -> None:
        with self._lock:
            self.current -= size
            if self.current < 0:
                raise RuntimeError("scratch accounting underflow")


class ConcatenatedBinaryStream(io.RawIOBase):
    """Read split files as one non-seekable stream without joining them on disk."""

    def __init__(self, paths: Sequence[Path]):
        super().__init__()
        self._paths = tuple(paths)
        self._index = 0
        self._stream: BinaryIO | None = None
        self._hashers = [hashlib.sha256() for _ in self._paths]
        self._bytes_read = [0 for _ in self._paths]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def _ensure_stream(self) -> bool:
        while self._stream is None and self._index < len(self._paths):
            self._stream = self._paths[self._index].open("rb")
        return self._stream is not None

    def readinto(self, buffer) -> int:
        view = memoryview(buffer)
        total = 0
        while total < len(view) and self._ensure_stream():
            assert self._stream is not None
            count = self._stream.readinto(view[total:])
            if count:
                self._hashers[self._index].update(view[total : total + count])
                self._bytes_read[self._index] += count
                total += count
                continue
            self._stream.close()
            self._stream = None
            self._index += 1
        return total

    @property
    def part_digests(self) -> tuple[str, ...]:
        return tuple(hasher.hexdigest() for hasher in self._hashers)

    @property
    def part_bytes_read(self) -> tuple[int, ...]:
        return tuple(self._bytes_read)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


def format_fps(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid fps: {value!r}")
    return format(value, ".12g")


def strict_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} cannot be boolean")
    result = int(value)
    if isinstance(value, float) and value != result:
        raise ValueError(f"{field_name} is not an integer")
    return result


def safe_component(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not SAFE_COMPONENT_RE.fullmatch(text) or text in {".", ".."}:
        raise ValueError(f"unsafe {field_name}: {text!r}")
    return text


def partition_window(start_frame: int, end_frame: int, max_frames: int = 128) -> tuple[FrameWindow, ...]:
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("invalid inclusive frame bounds")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    windows = []
    cursor = start_frame
    while cursor <= end_frame:
        window_end = min(end_frame, cursor + max_frames - 1)
        windows.append(FrameWindow(cursor, window_end))
        cursor = window_end + 1
    return tuple(windows)


def parse_dancetrack_member(name: str, expected_root: str | None = None) -> tuple[str, int] | None:
    match = DANCE_RE.fullmatch(name)
    if match is None:
        return None
    root, video, raw_frame = match.groups()
    if expected_root is not None and root != expected_root:
        raise MaterializationError(f"DanceTrack member root mismatch: expected {expected_root}, got {name}")
    frame_number = int(raw_frame)
    if frame_number <= 0:
        raise MaterializationError(f"DanceTrack frame number is not one-based: {name}")
    return safe_component(video, "DanceTrack video"), frame_number - 1


def parse_soccernet_member(name: str) -> tuple[str, int] | None:
    match = SOCCERNET_RE.fullmatch(name)
    if match is None:
        return None
    video, raw_frame = match.groups()
    frame_number = int(raw_frame)
    if frame_number <= 0:
        raise MaterializationError(f"SoccerNet frame number is not one-based: {name}")
    return safe_component(video, "SoccerNet video"), frame_number - 1


def parse_mose_member(name: str) -> tuple[str, int] | None:
    match = MOSE_RE.fullmatch(name)
    if match is None:
        return None
    video, raw_frame = match.groups()
    return safe_component(video, "MOSE video"), int(raw_frame)


def parse_vipseg_member(name: str) -> tuple[str, int] | None:
    match = VIPSEG_RE.fullmatch(name)
    if match is None:
        return None
    video, raw_frame = match.groups()
    return safe_component(video, "VIPSeg video"), int(raw_frame)


def snapshot_file(path: Path) -> FileSnapshot:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    if not resolved.is_file():
        raise FileNotFoundError(f"required source file does not exist: {resolved}")
    return FileSnapshot(str(resolved), stat.st_size, stat.st_mtime_ns)


def snapshots_equal(before: Sequence[FileSnapshot], after: Sequence[FileSnapshot]) -> bool:
    return tuple(before) == tuple(after)


def copy_stream_sequential(
    reader: BinaryIO,
    writer: BinaryIO,
    chunk_size: int = 8 * 1024 * 1024,
) -> UploadReceipt:
    """Copy a stream without seeking while computing the uploaded digest."""
    if chunk_size <= 0:
        raise ValueError("copy chunk size must be positive")
    digest = hashlib.sha256()
    size_bytes = 0
    while True:
        payload = reader.read(chunk_size)
        if not payload:
            break
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = writer.write(view[offset:])
            if written is None or written <= 0 or written > len(view) - offset:
                raise OSError(f"invalid sequential write result: {written!r}")
            offset += written
        digest.update(view)
        size_bytes += len(view)
    return UploadReceipt(size_bytes=size_bytes, sha256=digest.hexdigest())


def upload_local_media_to_staging(local_path: Path, staged_path: Path) -> UploadReceipt:
    """Upload a stable local file to OSSFS using one non-seekable sequential write."""
    local_path = local_path.resolve()
    staged_path = staged_path.resolve()
    if local_path == staged_path:
        raise MaterializationError("local media and staging paths must be different")
    before = snapshot_file(local_path)
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    created_staging = False
    try:
        with local_path.open("rb") as reader:
            with staged_path.open("xb", buffering=0) as writer:
                created_staging = True
                receipt = copy_stream_sequential(reader, writer)
        after = snapshot_file(local_path)
        if before != after:
            raise MaterializationError(f"local media changed during upload: {local_path}")
        if receipt.size_bytes != before.size:
            raise MaterializationError(
                f"short local media upload: copied={receipt.size_bytes}, expected={before.size}"
            )
        staged_size = staged_path.stat().st_size
        if staged_size != receipt.size_bytes:
            raise MaterializationError(
                f"staged media size mismatch: staged={staged_size}, copied={receipt.size_bytes}"
            )
        return receipt
    except Exception:
        if created_staging:
            staged_path.unlink(missing_ok=True)
        raise


def validate_probe(probe: VideoProbe, window: FrameWindow, plan: VideoPlan) -> None:
    if probe.codec_name != "h264":
        raise MaterializationError(f"expected H.264, got {probe.codec_name!r}")
    if (probe.width, probe.height) != (plan.width, plan.height):
        raise MaterializationError(
            f"encoded dimensions {probe.width}x{probe.height} do not match {plan.width}x{plan.height}"
        )
    expected_pixel_format = pixel_format_for(plan.width, plan.height)
    if probe.pixel_format != expected_pixel_format:
        raise MaterializationError(
            f"encoded pixel format {probe.pixel_format!r} does not match {expected_pixel_format!r}"
        )
    if probe.n_frames != window.n_frames:
        raise MaterializationError(
            f"encoded frame count {probe.n_frames} does not match {window.n_frames}"
        )
    if not math.isclose(probe.fps, plan.fps, rel_tol=1e-7, abs_tol=1e-7):
        raise MaterializationError(f"encoded fps {probe.fps} does not match {plan.fps}")


def pixel_format_for(width: int, height: int) -> str:
    return "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"


def output_path_for(output_dir: Path, dataset: str, video: str, window: FrameWindow) -> Path:
    return (
        output_dir
        / safe_component(dataset, "dataset")
        / safe_component(video, "video")
        / f"frames-{window.start_frame:08d}-{window.end_frame:08d}.mp4"
    ).resolve()


def clip_from_row(row: Mapping[str, Any], dataset: str) -> ClipSpec:
    row_dataset = safe_component(row.get("video_dataset"), "video_dataset")
    if row_dataset != dataset:
        raise ValueError(f"parquet video_dataset={row_dataset!r}, expected {dataset!r}")
    video = safe_component(row.get("video"), "video")
    clip = safe_component(row.get("clip"), "clip")
    start = strict_int(row.get("start_frame"), "start_frame")
    end = strict_int(row.get("end_frame"), "end_frame")
    n_frames = strict_int(row.get("n_frames"), "n_frames")
    width = strict_int(row.get("w"), "w")
    height = strict_int(row.get("h"), "h")
    fps = float(row.get("fps"))
    if start < 0 or end < start or n_frames != end - start + 1:
        raise ValueError("parquet frame range is not a closed, consistent interval")
    if width <= 0 or height <= 0:
        raise ValueError("parquet dimensions must be positive")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("parquet fps must be finite and positive")
    return ClipSpec(dataset, video, clip, start, end, n_frames, fps, width, height)


def load_clip_specs(parquet_path: Path, dataset: str) -> tuple[list[ClipSpec], int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise MaterializationError("pyarrow is required: pip install pyarrow") from exc
    required = {
        "video_dataset",
        "video",
        "clip",
        "start_frame",
        "end_frame",
        "n_frames",
        "fps",
        "w",
        "h",
    }
    parquet = pq.ParquetFile(parquet_path)
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise MaterializationError(f"{parquet_path} is missing columns: {sorted(missing)}")
    by_key: dict[str, ClipSpec] = {}
    row_count = 0
    columns = sorted(required)
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            row_count += 1
            try:
                spec = clip_from_row(row, dataset)
            except (TypeError, ValueError) as exc:
                raise MaterializationError(f"invalid {dataset} parquet row {row_count - 1}: {exc}") from exc
            existing = by_key.get(spec.key)
            if existing is not None and existing != spec:
                raise MaterializationError(f"conflicting metadata for clip {spec.key}")
            by_key[spec.key] = spec
    if not row_count or not by_key:
        raise MaterializationError(f"no annotation rows found in {parquet_path}")
    clips = sorted(
        by_key.values(),
        key=lambda item: (item.video, item.start_frame, item.end_frame, item.clip),
    )
    return clips, row_count


def build_video_plans(clips: Sequence[ClipSpec], max_frames: int) -> dict[str, VideoPlan]:
    grouped: dict[str, list[ClipSpec]] = defaultdict(list)
    for clip in clips:
        grouped[clip.video].append(clip)
    plans: dict[str, VideoPlan] = {}
    for video in sorted(grouped):
        video_clips = sorted(grouped[video], key=lambda item: (item.start_frame, item.end_frame, item.clip))
        first = video_clips[0]
        for clip in video_clips[1:]:
            if (clip.dataset, clip.fps, clip.width, clip.height) != (
                first.dataset,
                first.fps,
                first.width,
                first.height,
            ):
                raise MaterializationError(f"inconsistent source-video metadata for {first.source_video_id}")
        clip_windows = {
            clip.key: partition_window(clip.start_frame, clip.end_frame, max_frames)
            for clip in video_clips
        }
        unique_windows = sorted({window for windows in clip_windows.values() for window in windows})
        plans[video] = VideoPlan(
            dataset=first.dataset,
            video=video,
            fps=first.fps,
            width=first.width,
            height=first.height,
            clips=tuple(video_clips),
            clip_windows=clip_windows,
            windows=tuple(unique_windows),
        )
    return plans


def _pillow_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise MaterializationError("Pillow is required: pip install Pillow") from exc
    return Image


def validate_and_transform_frame(
    payload: bytes,
    plan: VideoPlan,
    member_name: str,
) -> bytes:
    Image = _pillow_image()
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "JPEG":
                raise FrameValidationError("image_format_mismatch", f"{member_name} is not a JPEG")
            actual_size = image.size
            expected_size = (plan.width, plan.height)
            if plan.dataset != "vipseg":
                if actual_size != expected_size:
                    raise FrameValidationError(
                        "frame_dimension_mismatch",
                        f"{member_name} is {actual_size[0]}x{actual_size[1]}, expected "
                        f"{plan.width}x{plan.height}",
                    )
                return payload

            image.load()
            source_width, source_height = actual_size
            if source_height <= 0:
                raise FrameValidationError("frame_dimension_mismatch", f"{member_name} has zero height")
            official_size = (int(720 * source_width / source_height), 720)
            if official_size != expected_size:
                raise FrameValidationError(
                    "vipseg_official_resize_mismatch",
                    f"official change2_720p maps {member_name} from {source_width}x{source_height} "
                    f"to {official_size[0]}x{official_size[1]}, but parquet declares "
                    f"{plan.width}x{plan.height}",
                )
            resampling = getattr(Image, "Resampling", Image)
            transformed = image.resize(official_size, resampling.BILINEAR)
            output = io.BytesIO()
            transformed.save(output, format="JPEG")
            transformed_payload = output.getvalue()
        with Image.open(io.BytesIO(transformed_payload)) as check:
            check.load()
            if check.size != expected_size:
                raise FrameValidationError(
                    "vipseg_transformed_dimension_mismatch",
                    f"stored VIPSeg frame is {check.size}, expected {expected_size}",
                )
        return transformed_payload
    except FrameValidationError:
        raise
    except Exception as exc:
        raise FrameValidationError("image_decode_failed", f"cannot decode {member_name}: {exc}") from exc


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def remove_cache_directory(path: Path, scratch_root: Path) -> None:
    resolved = path.resolve()
    root = scratch_root.resolve()
    if resolved == root or not _is_below(resolved, root):
        raise RuntimeError(f"refusing to remove cache outside scratch root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def remove_tree_below(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    root = parent.resolve()
    if resolved == root or not _is_below(resolved, root):
        raise RuntimeError(f"refusing to remove directory outside its owned parent: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _encode_video_cache(
    cache: VideoCache,
    windows: Sequence[FrameWindow],
    states: dict[FrameWindow, WindowState],
    staging_dir: Path,
    output_dir: Path,
    scratch_root: Path,
    budget: ScratchBudget,
    media_tool: MediaTool,
) -> VideoEncodingResult:
    result = dict(states)
    try:
        for window in windows:
            bad_frames = [
                index
                for index in range(window.start_frame, window.end_frame + 1)
                if index in cache.frame_errors or index not in cache.stored_frames
            ]
            final_path = output_path_for(output_dir, cache.plan.dataset, cache.plan.video, window)
            if bad_frames:
                reason, detail = cache.frame_errors.get(
                    bad_frames[0],
                    ("missing_frame", f"source frame {bad_frames[0]} was not present"),
                )
                result[window] = WindowState(
                    "failed",
                    final_path,
                    reason=reason,
                    detail=(
                        f"{detail}; affected_frames={len(bad_frames)}, "
                        f"examples={bad_frames[:10]}, omitted={max(0, len(bad_frames) - 10)}"
                    ),
                )
                continue
            relative = final_path.relative_to(output_dir.resolve())
            staged_path = (staging_dir / relative).resolve()
            local_media_dir = cache.directory / "local-mp4"
            local_media_path = local_media_dir / (
                f"window-{window.start_frame:08d}-{window.end_frame:08d}.mp4"
            )
            reserved_media_bytes = 0
            try:
                local_media_dir.mkdir(parents=True, exist_ok=True)
                media_tool.encode(
                    cache.directory / "frame-%08d.jpg",
                    window.start_frame,
                    window.n_frames,
                    cache.plan.fps,
                    pixel_format_for(cache.plan.width, cache.plan.height),
                    local_media_path,
                )
                validate_probe(media_tool.probe(local_media_path), window, cache.plan)
                local_media_size = local_media_path.stat().st_size
                if local_media_size <= 0:
                    raise MaterializationError(f"encoded local media is empty: {local_media_path}")
                budget.reserve(local_media_size)
                reserved_media_bytes = local_media_size
                receipt = upload_local_media_to_staging(local_media_path, staged_path)
                validate_probe(media_tool.probe(staged_path), window, cache.plan)
                result[window] = WindowState(
                    "encoded",
                    final_path,
                    staged_path=staged_path,
                    uploaded_size_bytes=receipt.size_bytes,
                    uploaded_sha256=receipt.sha256,
                )
            except Exception as exc:
                staged_path.unlink(missing_ok=True)
                result[window] = WindowState(
                    "failed",
                    final_path,
                    reason="media_encode_or_probe_failed",
                    detail=str(exc),
                )
            finally:
                local_media_path.unlink(missing_ok=True)
                if reserved_media_bytes:
                    budget.release(reserved_media_bytes)
        return VideoEncodingResult(cache.plan.video, result)
    finally:
        remove_cache_directory(cache.directory, scratch_root)
        budget.release(cache.bytes_used)


class ExtractionCoordinator:
    """Bound extracted-frame storage and concurrent ffmpeg work by source video."""

    def __init__(
        self,
        plans: Mapping[str, VideoPlan],
        initial_states: Mapping[str, Mapping[FrameWindow, WindowState]],
        scratch_root: Path,
        staging_dir: Path,
        output_dir: Path,
        workers: int,
        max_inflight_videos: int,
        budget: ScratchBudget,
        media_tool: MediaTool,
    ):
        self.plans = dict(plans)
        self.states = {video: dict(values) for video, values in initial_states.items()}
        self.scratch_root = scratch_root.resolve()
        self.staging_dir = staging_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.max_inflight_videos = max_inflight_videos
        self.budget = budget
        self.media_tool = media_tool
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="videotrack-ffmpeg")
        self.caches: dict[str, VideoCache] = {}
        self.pending: dict[Future[VideoEncodingResult], str] = {}
        self.results: dict[str, VideoEncodingResult] = {}
        self.fatal_errors: list[str] = []
        self.selected_members_seen = 0
        self.cached_frames = 0
        self.cached_bytes = 0
        for video, plan in sorted(self.plans.items()):
            needed_windows = [window for window in plan.windows if window not in self.states[video]]
            required = {
                frame
                for window in needed_windows
                for frame in range(window.start_frame, window.end_frame + 1)
            }
            if required:
                directory = (self.scratch_root / plan.dataset / safe_component(video, "video")).resolve()
                if not _is_below(directory, self.scratch_root):
                    raise MaterializationError(f"unsafe scratch path: {directory}")
                directory.mkdir(parents=True, exist_ok=False)
                self.caches[video] = VideoCache(plan, required, directory)

    @property
    def requested_videos(self) -> set[str]:
        return set(self.caches)

    def requested_frames(self, video: str) -> set[int]:
        cache = self.caches.get(video)
        return set() if cache is None else cache.required_frames

    def accept(self, video: str, logical_frame: int, payload: bytes, member_name: str) -> None:
        cache = self.caches.get(video)
        if cache is None or logical_frame not in cache.required_frames:
            return
        if logical_frame in cache.seen_frames:
            self.fatal_errors.append(
                f"duplicate selected source frame {cache.plan.dataset}::{video}::{logical_frame}: {member_name}"
            )
            return
        cache.seen_frames.add(logical_frame)
        self.selected_members_seen += 1
        try:
            stored_payload = validate_and_transform_frame(payload, cache.plan, member_name)
            self.budget.reserve(len(stored_payload))
            destination = cache.directory / f"frame-{logical_frame:08d}.jpg"
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("xb") as stream:
                    stream.write(stored_payload)
                os.replace(temporary, destination)
            except Exception:
                self.budget.release(len(stored_payload))
                raise
            finally:
                temporary.unlink(missing_ok=True)
            cache.stored_frames.add(logical_frame)
            cache.bytes_used += len(stored_payload)
            self.cached_frames += 1
            self.cached_bytes += len(stored_payload)
        except FrameValidationError as exc:
            cache.frame_errors[logical_frame] = (exc.reason, str(exc))
        if cache.seen_frames == cache.required_frames:
            self._submit(cache)

    def _submit(self, cache: VideoCache) -> None:
        if cache.submitted:
            return
        while len(self.pending) >= self.max_inflight_videos:
            self._collect_some()
        cache.submitted = True
        needed_windows = [window for window in cache.plan.windows if window not in self.states[cache.plan.video]]
        future = self.executor.submit(
            _encode_video_cache,
            cache,
            needed_windows,
            self.states[cache.plan.video],
            self.staging_dir,
            self.output_dir,
            self.scratch_root,
            self.budget,
            self.media_tool,
        )
        self.pending[future] = cache.plan.video

    def _collect_some(self) -> None:
        if not self.pending:
            return
        done, _ = wait(tuple(self.pending), return_when=FIRST_COMPLETED)
        for future in done:
            video = self.pending.pop(future)
            try:
                self.results[video] = future.result()
            except Exception as exc:
                self.fatal_errors.append(f"video worker failed for {video}: {exc}")

    def finish(self) -> dict[str, VideoEncodingResult]:
        for video in sorted(self.caches):
            cache = self.caches[video]
            if cache.submitted:
                continue
            missing = sorted(cache.required_frames - cache.seen_frames)
            for frame in missing:
                cache.frame_errors[frame] = (
                    "missing_frame",
                    f"source frame {frame} was not found in the audited archive mapping",
                )
            cache.seen_frames.update(missing)
            self._submit(cache)
        while self.pending:
            self._collect_some()
        self.executor.shutdown(wait=True)
        for video, states in self.states.items():
            if video not in self.results:
                self.results[video] = VideoEncodingResult(video, dict(states))
        return self.results

    def abort(self) -> None:
        while self.pending:
            self._collect_some()
        self.executor.shutdown(wait=True)
        for cache in self.caches.values():
            if not cache.submitted and cache.directory.exists():
                remove_cache_directory(cache.directory, self.scratch_root)
                self.budget.release(cache.bytes_used)


def _read_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    with archive.open(info, "r") as stream:
        payload = stream.read()
    if len(payload) != info.file_size:
        raise MaterializationError(f"short ZIP member read: {info.filename}")
    return payload


def extract_zip_sources(
    source: SourceDefinition,
    coordinator: ExtractionCoordinator,
    accounting: dict[str, Any],
) -> None:
    member_map: dict[tuple[str, int], tuple[zipfile.ZipFile, zipfile.ZipInfo]] = {}
    entries_scanned = 0
    with contextlib.ExitStack() as stack:
        archives = [stack.enter_context(zipfile.ZipFile(path, "r")) for path in source.archive_paths]
        for index, archive in enumerate(archives):
            expected_label = source.archive_labels[index] if source.archive_labels else None
            for info in archive.infolist():
                entries_scanned += 1
                if info.is_dir():
                    continue
                if source.kind == "dancetrack_zip":
                    parsed = parse_dancetrack_member(info.filename, expected_label)
                else:
                    parsed = parse_soccernet_member(info.filename)
                if parsed is None:
                    continue
                video, logical_frame = parsed
                if logical_frame not in coordinator.requested_frames(video):
                    continue
                identity = (video, logical_frame)
                if identity in member_map:
                    raise MaterializationError(f"duplicate selected ZIP frame mapping: {identity}")
                member_map[identity] = (archive, info)
        accounting["archive_directory_entries_scanned"] = entries_scanned
        accounting["archive_passes"] = 1
        reads = 0
        for video in sorted(coordinator.requested_videos):
            for logical_frame in sorted(coordinator.requested_frames(video)):
                resolved = member_map.get((video, logical_frame))
                if resolved is None:
                    continue
                archive, info = resolved
                coordinator.accept(video, logical_frame, _read_zip_member(archive, info), info.filename)
                reads += 1
        accounting["archive_random_member_reads"] = reads


def stream_selected_tar(
    archive: tarfile.TarFile,
    coordinator: ExtractionCoordinator,
    accounting: dict[str, Any],
) -> None:
    entries_scanned = 0
    selected_reads = 0
    for member in archive:
        entries_scanned += 1
        if not member.isfile():
            continue
        parsed = parse_mose_member(member.name)
        if parsed is None:
            continue
        video, logical_frame = parsed
        if logical_frame not in coordinator.requested_frames(video):
            continue
        extracted = archive.extractfile(member)
        if extracted is None:
            raise MaterializationError(f"cannot read TAR member: {member.name}")
        with extracted:
            payload = extracted.read()
        if len(payload) != member.size:
            raise MaterializationError(f"short TAR member read: {member.name}")
        coordinator.accept(video, logical_frame, payload, member.name)
        selected_reads += 1
    accounting["archive_directory_entries_scanned"] = entries_scanned
    accounting["archive_sequential_selected_reads"] = selected_reads
    accounting["archive_passes"] = 1


def extract_mose_outer_zip(
    source: SourceDefinition,
    coordinator: ExtractionCoordinator,
    accounting: dict[str, Any],
) -> None:
    with zipfile.ZipFile(source.archive_paths[0], "r") as outer:
        matches = [info for info in outer.infolist() if info.filename == MOSE_INNER_MEMBER and not info.is_dir()]
        if len(matches) != 1:
            raise MaterializationError(
                f"expected exactly one {MOSE_INNER_MEMBER!r} in MOSE outer ZIP, found {len(matches)}"
            )
        with outer.open(matches[0], "r") as inner_stream:
            with tarfile.open(fileobj=inner_stream, mode="r|gz") as archive:
                stream_selected_tar(archive, coordinator, accounting)
    accounting["nested_archive_member"] = MOSE_INNER_MEMBER


def load_sha256_sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\s]+)", line)
        if match is None:
            raise MaterializationError(f"invalid SHA256SUMS line {line_number} in {path}")
        digest, filename = match.groups()
        if filename in result:
            raise MaterializationError(f"duplicate SHA256SUMS filename {filename!r} in {path}")
        result[filename] = digest.lower()
    return result


def extract_mosev2_parts(
    source: SourceDefinition,
    coordinator: ExtractionCoordinator,
    accounting: dict[str, Any],
) -> None:
    expected_suffixes = (".aa", ".ab", ".ac")
    actual_suffixes = tuple(path.suffix for path in source.archive_paths)
    if actual_suffixes != expected_suffixes:
        raise MaterializationError(
            f"MOSEv2 split parts must be aa+ab+ac in order, got {actual_suffixes}"
        )
    if source.checksum_path is None:
        raise MaterializationError("MOSEv2 requires the official SHA256SUMS file")
    expected_sums = load_sha256_sums(source.checksum_path)
    required_names = tuple(path.name for path in source.archive_paths)
    missing = [name for name in required_names if name not in expected_sums]
    if missing:
        raise MaterializationError(f"MOSEv2 SHA256SUMS is missing split parts: {missing}")
    with ConcatenatedBinaryStream(source.archive_paths) as raw_stream:
        with io.BufferedReader(raw_stream, buffer_size=1024 * 1024) as buffered:
            with tarfile.open(fileobj=buffered, mode="r|gz") as archive:
                stream_selected_tar(archive, coordinator, accounting)
            while buffered.read(1024 * 1024):
                pass
        actual_sums = raw_stream.part_digests
        part_bytes = raw_stream.part_bytes_read
    checksum_accounting = {}
    mismatches = []
    for path, expected, actual, bytes_read in zip(
        source.archive_paths,
        (expected_sums[name] for name in required_names),
        actual_sums,
        part_bytes,
    ):
        matched = expected == actual and bytes_read == path.stat().st_size
        checksum_accounting[path.name] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "bytes_read": bytes_read,
            "expected_bytes": path.stat().st_size,
            "matched": matched,
        }
        if not matched:
            mismatches.append(path.name)
    accounting["multipart_sha256"] = checksum_accounting
    if mismatches:
        raise MaterializationError(f"MOSEv2 multipart SHA256 verification failed: {mismatches}")
    accounting["multipart_stream_order"] = [path.name for path in source.archive_paths]


def extract_vipseg_tar(
    source: SourceDefinition,
    coordinator: ExtractionCoordinator,
    accounting: dict[str, Any],
) -> None:
    by_video: dict[str, dict[int, tarfile.TarInfo]] = defaultdict(dict)
    entries_scanned = 0
    requested_videos = coordinator.requested_videos
    with tarfile.open(source.archive_paths[0], mode="r:") as archive:
        for member in archive.getmembers():
            entries_scanned += 1
            if not member.isfile():
                continue
            parsed = parse_vipseg_member(member.name)
            if parsed is None:
                continue
            video, original_number = parsed
            if video not in requested_videos:
                continue
            if original_number in by_video[video]:
                raise MaterializationError(
                    f"duplicate VIPSeg numeric frame for {video}: {original_number}"
                )
            by_video[video][original_number] = member
        reads = 0
        for video in sorted(requested_videos):
            ordered = [by_video[video][number] for number in sorted(by_video.get(video, {}))]
            for logical_frame in sorted(coordinator.requested_frames(video)):
                if logical_frame >= len(ordered):
                    continue
                member = ordered[logical_frame]
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise MaterializationError(f"cannot read VIPSeg TAR member: {member.name}")
                with extracted:
                    payload = extracted.read()
                if len(payload) != member.size:
                    raise MaterializationError(f"short VIPSeg TAR member read: {member.name}")
                coordinator.accept(video, logical_frame, payload, member.name)
                reads += 1
    accounting["archive_directory_entries_scanned"] = entries_scanned
    accounting["archive_random_member_reads"] = reads
    accounting["archive_passes"] = 1
    accounting["frame_index_mapping"] = "numeric filename sort -> zero-based local index"
    accounting["transformation"] = {
        "name": "VIPSeg/change2_720p.py",
        "operation": "Pillow BILINEAR resize",
        "size": "(int(720 * source_width / source_height), 720)",
        "validation": "transformed dimensions must exactly match parquet w/h",
    }


def failure_record(
    dataset: str,
    reason: str,
    detail: str,
    *,
    video: str | None = None,
    clip: str | None = None,
    window: FrameWindow | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"dataset": dataset, "reason": reason, "detail": detail}
    if video is not None:
        result["video"] = video
    if clip is not None:
        result["clip"] = clip
    if window is not None:
        result["source_start_frame"] = window.start_frame
        result["source_end_frame"] = window.end_frame
    return result


def prepare_initial_states(
    plans: Mapping[str, VideoPlan],
    output_dir: Path,
    resume: bool,
    media_tool: MediaTool,
) -> tuple[dict[str, dict[FrameWindow, WindowState]], list[dict[str, Any]]]:
    states: dict[str, dict[FrameWindow, WindowState]] = {video: {} for video in plans}
    failures: list[dict[str, Any]] = []
    for video, plan in sorted(plans.items()):
        for window in plan.windows:
            final_path = output_path_for(output_dir, plan.dataset, video, window)
            if not final_path.exists():
                continue
            if not final_path.is_file():
                detail = f"output path exists but is not a file: {final_path}"
                states[video][window] = WindowState(
                    "failed", final_path, reason="existing_output_invalid", detail=detail
                )
                failures.append(
                    failure_record(plan.dataset, "existing_output_invalid", detail, video=video, window=window)
                )
                continue
            if not resume:
                detail = f"output already exists; pass --resume to validate and reuse it: {final_path}"
                states[video][window] = WindowState(
                    "failed", final_path, reason="existing_output_requires_resume", detail=detail
                )
                failures.append(
                    failure_record(
                        plan.dataset,
                        "existing_output_requires_resume",
                        detail,
                        video=video,
                        window=window,
                    )
                )
                continue
            try:
                validate_probe(media_tool.probe(final_path), window, plan)
                states[video][window] = WindowState("reused", final_path)
            except Exception as exc:
                detail = f"resume validation failed for {final_path}: {exc}"
                states[video][window] = WindowState(
                    "failed", final_path, reason="existing_output_invalid", detail=detail
                )
                failures.append(
                    failure_record(plan.dataset, "existing_output_invalid", detail, video=video, window=window)
                )
    return states, failures


def build_index_entries(
    plans: Mapping[str, VideoPlan],
    results: Mapping[str, VideoEncodingResult],
) -> tuple[dict[str, dict[str, Any]], list[tuple[Path, Path]], list[dict[str, Any]]]:
    entries: dict[str, dict[str, Any]] = {}
    staged: dict[Path, Path] = {}
    failures: list[dict[str, Any]] = []
    for video, plan in sorted(plans.items()):
        states = results[video].states
        for clip in plan.clips:
            windows = plan.clip_windows[clip.key]
            invalid = [
                window
                for window in windows
                if states.get(window) is None or states[window].state not in {"encoded", "reused"}
            ]
            if invalid:
                for window in invalid:
                    state = states.get(window)
                    failures.append(
                        failure_record(
                            plan.dataset,
                            state.reason if state and state.reason else "window_not_materialized",
                            state.detail if state and state.detail else "required window was not materialized",
                            video=video,
                            clip=clip.clip,
                            window=window,
                        )
                    )
                continue
            entries[clip.key] = {
                "source_video_id": clip.source_video_id,
                "lineage_key": clip.lineage_key,
                "width": clip.width,
                "height": clip.height,
                "fps": clip.fps,
                "windows": [
                    {
                        "mode": "cropped",
                        "path": str(states[window].final_path),
                        "source_start_frame": window.start_frame,
                        "source_end_frame": window.end_frame,
                    }
                    for window in windows
                ],
            }
            for window in windows:
                state = states[window]
                if state.state == "encoded":
                    assert state.staged_path is not None
                    existing = staged.get(state.final_path)
                    if existing is not None and existing != state.staged_path:
                        raise MaterializationError(f"conflicting staged outputs for {state.final_path}")
                    staged[state.final_path] = state.staged_path
    staged_files = sorted(
        ((source, final) for final, source in staged.items()),
        key=lambda pair: str(pair[1]),
    )
    return entries, staged_files, failures


def process_dataset(
    source: SourceDefinition,
    output_dir: Path,
    scratch_root: Path,
    staging_dir: Path,
    max_window_frames: int,
    workers: int,
    max_inflight_videos: int,
    resume: bool,
    budget: ScratchBudget,
    media_tool: MediaTool,
) -> DatasetOutcome:
    accounting: dict[str, Any] = {
        "dataset": source.dataset,
        "source_kind": source.kind,
        "parquet": str(source.parquet_path.resolve()),
        "archives": [str(path.resolve()) for path in source.archive_paths],
        "checksum_file": (
            str(source.checksum_path.resolve()) if source.checksum_path is not None else None
        ),
        "frame_ranges": "zero-based inclusive",
        "max_window_frames": max_window_frames,
        "selected_frame_validation": (
            "Pillow full decode + official BILINEAR resize + exact dimensions"
            if source.dataset == "vipseg"
            else "Pillow JPEG header/dimensions; full decode by ffmpeg -xerror"
        ),
        "frame_transform_parallelism": (
            "serial archive-order VIPSeg transform; bounded multi-video ffmpeg encoding"
            if source.dataset == "vipseg"
            else "bounded multi-video ffmpeg encoding"
        ),
    }
    failures: list[dict[str, Any]] = []
    before: list[FileSnapshot] = []
    coordinator: ExtractionCoordinator | None = None
    try:
        before = [snapshot_file(path) for path in source.files]
        accounting["source_snapshot_before"] = [item.as_dict() for item in before]
        clips, row_count = load_clip_specs(source.parquet_path, source.dataset)
        plans = build_video_plans(clips, max_window_frames)
        initial_states, resume_failures = prepare_initial_states(plans, output_dir, resume, media_tool)
        failures.extend(resume_failures)
        accounting.update(
            {
                "annotation_rows": row_count,
                "unique_clips": len(clips),
                "source_videos": len(plans),
                "unique_windows_planned": sum(len(plan.windows) for plan in plans.values()),
                "window_references_planned": sum(
                    len(plan.clip_windows[clip.key]) for plan in plans.values() for clip in plan.clips
                ),
                "annotation_unique_frames": sum(
                    len(
                        {
                            frame
                            for clip in plan.clips
                            for frame in range(clip.start_frame, clip.end_frame + 1)
                        }
                    )
                    for plan in plans.values()
                ),
                "resume_windows_reused": sum(
                    state.state == "reused" for values in initial_states.values() for state in values.values()
                ),
                "resume_windows_failed": sum(
                    state.state == "failed" for values in initial_states.values() for state in values.values()
                ),
                "pixel_format_windows_planned": dict(
                    Counter(
                        pixel_format_for(plan.width, plan.height)
                        for plan in plans.values()
                        for _ in plan.windows
                    )
                ),
            }
        )
        coordinator = ExtractionCoordinator(
            plans,
            initial_states,
            scratch_root,
            staging_dir,
            output_dir,
            workers,
            max_inflight_videos,
            budget,
            media_tool,
        )
        accounting["archive_frames_requested"] = sum(
            len(coordinator.requested_frames(video)) for video in coordinator.requested_videos
        )
        accounting["archive_videos_requested"] = len(coordinator.requested_videos)
        if coordinator.requested_videos:
            if source.kind in {"dancetrack_zip", "soccernet_zip"}:
                extract_zip_sources(source, coordinator, accounting)
            elif source.kind == "mose_outer_zip":
                extract_mose_outer_zip(source, coordinator, accounting)
            elif source.kind == "mosev2_parts":
                extract_mosev2_parts(source, coordinator, accounting)
            elif source.kind == "vipseg_tar":
                extract_vipseg_tar(source, coordinator, accounting)
            else:
                raise MaterializationError(f"unsupported source kind: {source.kind}")
        else:
            accounting["archive_passes"] = 0
        results = coordinator.finish()
        accounting.update(
            {
                "archive_selected_members_seen": coordinator.selected_members_seen,
                "frames_cached": coordinator.cached_frames,
                "frame_bytes_cached_total": coordinator.cached_bytes,
            }
        )
        if coordinator.fatal_errors:
            raise MaterializationError("; ".join(coordinator.fatal_errors[:10]))
        after = [snapshot_file(path) for path in source.files]
        accounting["source_snapshot_after"] = [item.as_dict() for item in after]
        accounting["source_snapshot_unchanged"] = snapshots_equal(before, after)
        if not snapshots_equal(before, after):
            raise MaterializationError("one or more source files changed during materialization")
        entries, staged_files, index_failures = build_index_entries(plans, results)
        failures.extend(index_failures)
        accounting.update(
            {
                "successful_clips": len(entries),
                "failed_clips": len(clips) - len(entries),
                "encoded_unique_windows": len(staged_files),
                "indexed_window_references": sum(len(value["windows"]) for value in entries.values()),
            }
        )
        accounting["status"] = "complete" if not failures else "partial"
        return DatasetOutcome(source.dataset, entries, staged_files, accounting, failures)
    except Exception as exc:
        if coordinator is not None:
            coordinator.abort()
        accounting["status"] = "failed"
        accounting["fatal_error"] = str(exc)
        if before:
            try:
                after = [snapshot_file(path) for path in source.files]
                accounting["source_snapshot_after"] = [item.as_dict() for item in after]
                accounting["source_snapshot_unchanged"] = snapshots_equal(before, after)
            except Exception as snapshot_exc:
                accounting["source_snapshot_after_error"] = str(snapshot_exc)
        failures.append(failure_record(source.dataset, "source_failed", str(exc)))
        dataset_stage = staging_dir / source.dataset
        if dataset_stage.exists():
            remove_tree_below(dataset_stage, staging_dir)
        return DatasetOutcome(source.dataset, {}, [], accounting, failures, fatal=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_staged_media(staged_files: Sequence[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    installed: list[tuple[Path, Path]] = []
    try:
        for staged, final in sorted(staged_files, key=lambda pair: str(pair[1])):
            if not staged.is_file():
                raise MaterializationError(f"staged media is missing: {staged}")
            if final.exists():
                raise MaterializationError(f"output appeared during the run; refusing to overwrite: {final}")
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
            installed.append((final, staged))
    except Exception:
        for final, staged in reversed(installed):
            if final.exists():
                staged.parent.mkdir(parents=True, exist_ok=True)
                os.replace(final, staged)
        raise
    return installed


def rollback_installed_media(installed: Sequence[tuple[Path, Path]]) -> None:
    errors = []
    for final, staged in reversed(installed):
        try:
            if not final.exists():
                raise MaterializationError(f"installed media disappeared before rollback: {final}")
            staged.parent.mkdir(parents=True, exist_ok=True)
            os.replace(final, staged)
        except Exception as exc:
            errors.append(f"{final}: {exc}")
    if errors:
        raise MaterializationError("media rollback failed: " + "; ".join(errors))


def backup_metadata_files(
    paths: Sequence[Path],
    backup_dir: Path,
) -> list[tuple[Path, Path | None]]:
    backups = []
    backup_dir.mkdir(parents=True, exist_ok=False)
    for index, path in enumerate(paths):
        resolved = path.resolve()
        if resolved.exists() and not resolved.is_file():
            raise MaterializationError(f"metadata output exists but is not a file: {resolved}")
        if resolved.is_file():
            backup = backup_dir / f"{index:02d}-{resolved.name}.backup"
            shutil.copy2(resolved, backup)
            backups.append((resolved, backup))
        else:
            backups.append((resolved, None))
    return backups


def restore_metadata_files(backups: Sequence[tuple[Path, Path | None]]) -> None:
    errors = []
    for destination, backup in reversed(backups):
        try:
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
        except Exception as exc:
            errors.append(f"{destination}: {exc}")
    if errors:
        raise MaterializationError("metadata rollback failed: " + "; ".join(errors))


def _overlay_xy(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        raw_x, raw_y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        raw_x, raw_y = value
    else:
        raise ValueError("visible point is not an [x, y] pair")
    x, y = float(raw_x), float(raw_y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("visible point coordinates are not finite")
    return x, y


def load_overlay_candidates(
    source: SourceDefinition,
    index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise MaterializationError("pyarrow is required: pip install pyarrow") from exc
    columns = {
        "id",
        "video_dataset",
        "video",
        "clip",
        "start_frame",
        "end_frame",
        "n_frames",
        "w",
        "h",
        "exp",
        "points",
    }
    parquet = pq.ParquetFile(source.parquet_path)
    missing = columns - set(parquet.schema_arrow.names)
    if missing:
        raise MaterializationError(
            f"overlay audit columns missing from {source.parquet_path}: {sorted(missing)}"
        )
    selected_by_clip: dict[str, dict[str, Any]] = {}
    exclusions: Counter[str] = Counter()
    for batch in parquet.iter_batches(batch_size=256, columns=sorted(columns)):
        for row in batch.to_pylist():
            try:
                dataset = safe_component(row.get("video_dataset"), "video_dataset")
                if dataset != source.dataset:
                    raise ValueError(f"unexpected video_dataset {dataset!r}")
                video = safe_component(row.get("video"), "video")
                clip = safe_component(row.get("clip"), "clip")
                key = f"{dataset}::{clip}"
                entry = index.get(key)
                if entry is None:
                    exclusions["clip_not_indexed"] += 1
                    continue
                start = strict_int(row.get("start_frame"), "start_frame")
                end = strict_int(row.get("end_frame"), "end_frame")
                n_frames = strict_int(row.get("n_frames"), "n_frames")
                width = strict_int(row.get("w"), "w")
                height = strict_int(row.get("h"), "h")
                if end - start + 1 != n_frames:
                    raise ValueError("row frame bounds are inconsistent")
                if (width, height) != (strict_int(entry["width"], "width"), strict_int(entry["height"], "height")):
                    raise ValueError("row dimensions differ from materialized index")
                tracks = row.get("points")
                if not isinstance(tracks, list) or not tracks:
                    raise ValueError("row has no point tracks")
                normalized_tracks = []
                visible_offsets: set[int] = set()
                for track in tracks:
                    if not isinstance(track, Mapping):
                        raise ValueError("point track is not an object")
                    object_id = str(track.get("object_id", "")).strip()
                    if not object_id:
                        raise ValueError("point track object_id is empty")
                    values = track.get("points")
                    if not isinstance(values, list) or len(values) != n_frames:
                        raise ValueError("point track length differs from n_frames")
                    for offset, point in enumerate(values):
                        if point is not None:
                            _overlay_xy(point)
                            visible_offsets.add(offset)
                    normalized_tracks.append({"object_id": object_id, "points": values})
                ordered_visible = sorted(visible_offsets)
                if len(ordered_visible) < 3:
                    exclusions["fewer_than_three_visible_frames"] += 1
                    continue
                temporal_offsets = (
                    ordered_visible[0],
                    ordered_visible[len(ordered_visible) // 2],
                    ordered_visible[-1],
                )
                if len(set(temporal_offsets)) != 3:
                    exclusions["non_distinct_temporal_frames"] += 1
                    continue
                row_id = str(row.get("id", "")).strip()
                if not row_id:
                    raise ValueError("row id is empty")
                candidate = {
                    "dataset": dataset,
                    "video": video,
                    "clip": clip,
                    "key": key,
                    "row_id": row_id,
                    "expression": str(row.get("exp", "")).strip(),
                    "start_frame": start,
                    "end_frame": end,
                    "n_frames": n_frames,
                    "width": width,
                    "height": height,
                    "tracks": normalized_tracks,
                    "temporal_offsets": temporal_offsets,
                    "entry": entry,
                }
                existing = selected_by_clip.get(key)
                if existing is None or (row_id, video, clip) < (
                    existing["row_id"],
                    existing["video"],
                    existing["clip"],
                ):
                    selected_by_clip[key] = candidate
            except (KeyError, TypeError, ValueError) as exc:
                exclusions[f"invalid_candidate:{type(exc).__name__}"] += 1
    candidates = sorted(
        selected_by_clip.values(),
        key=lambda item: (item["video"], item["clip"], item["row_id"]),
    )
    return candidates, dict(sorted(exclusions.items()))


def _window_for_source_frame(entry: Mapping[str, Any], source_frame: int) -> Mapping[str, Any]:
    matches = [
        window
        for window in entry["windows"]
        if strict_int(window["source_start_frame"], "source_start_frame")
        <= source_frame
        <= strict_int(window["source_end_frame"], "source_end_frame")
    ]
    if len(matches) != 1:
        raise MaterializationError(
            f"source frame {source_frame} resolves to {len(matches)} materialized windows"
        )
    return matches[0]


def _object_color(object_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(object_id.encode("utf-8")).digest()
    return tuple(64 + value % 192 for value in digest[:3])


def render_overlay(
    candidate: Mapping[str, Any],
    temporal_role: str,
    offset: int,
    overlay_dir: Path,
    temporary_dir: Path,
    media_tool: MediaTool,
) -> dict[str, Any]:
    source_frame = strict_int(candidate["start_frame"], "start_frame") + offset
    window = _window_for_source_frame(candidate["entry"], source_frame)
    window_start = strict_int(window["source_start_frame"], "source_start_frame")
    window_end = strict_int(window["source_end_frame"], "source_end_frame")
    local_frame = source_frame - window_start
    video_path = Path(str(window["path"])).expanduser().resolve()
    identity = (
        f"{candidate['dataset']}\0{candidate['video']}\0{candidate['clip']}\0"
        f"{candidate['row_id']}\0{source_frame}\0{temporal_role}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    relative_output = (
        Path(safe_component(candidate["dataset"], "dataset"))
        / safe_component(candidate["video"], "video")
        / f"{safe_component(candidate['clip'], 'clip')}-{source_frame:08d}-{temporal_role}-{digest}.png"
    )
    output_path = (overlay_dir / relative_output).resolve()
    decoded = temporary_dir / f"decoded-{digest}.png"
    result: dict[str, Any] = {
        "dataset": candidate["dataset"],
        "video": candidate["video"],
        "clip": candidate["clip"],
        "row_id": candidate["row_id"],
        "expression": candidate["expression"],
        "temporal_role": temporal_role,
        "annotation_offset": offset,
        "source_frame": source_frame,
        "window_source_start_frame": window_start,
        "window_source_end_frame": window_end,
        "local_frame": local_frame,
        "video_path": str(video_path),
        "overlay_png": str(output_path),
    }
    try:
        if not video_path.is_file():
            raise MaterializationError(f"materialized MP4 is missing: {video_path}")
        media_tool.decode_frame(video_path, local_frame, decoded)
        Image = _pillow_image()
        from PIL import ImageDraw
        with Image.open(decoded) as raw_image:
            raw_image.load()
            image = raw_image.convert("RGB")
        expected_size = (strict_int(candidate["width"], "width"), strict_int(candidate["height"], "height"))
        if image.size != expected_size:
            raise MaterializationError(
                f"decoded frame is {image.size[0]}x{image.size[1]}, expected "
                f"{expected_size[0]}x{expected_size[1]}"
            )
        declared: list[tuple[str, float, float]] = []
        for track in candidate["tracks"]:
            point = track["points"][offset]
            if point is None:
                continue
            x, y = _overlay_xy(point)
            if x < 0 or y < 0 or x > expected_size[0] or y > expected_size[1]:
                raise MaterializationError(
                    f"GT point {track['object_id']} at ({x}, {y}) is outside declared dimensions"
                )
            declared.append((str(track["object_id"]), x, y))
        if not declared:
            raise MaterializationError("selected temporal frame has no visible GT points")
        draw = ImageDraw.Draw(image)
        radius = max(4, min(expected_size) // 120)
        rendered = 0
        for object_id, x, y in declared:
            color = _object_color(object_id)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
            label = f"GT/{object_id}"
            label_x = min(max(0, int(x + radius + 2)), max(0, expected_size[0] - 80))
            label_y = min(max(0, int(y - radius - 12)), max(0, expected_size[1] - 14))
            box = draw.textbbox((label_x, label_y), label)
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text((label_x, label_y), label, fill=color)
            rendered += 1
        if rendered != len(declared):
            raise MaterializationError(
                f"declared/rendered GT count differs: {len(declared)} != {rendered}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.tmp.png")
        try:
            image.save(temporary_output, format="PNG")
            with Image.open(temporary_output) as check:
                check.load()
                if check.size != expected_size:
                    raise MaterializationError("saved overlay PNG dimensions changed")
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)
        result.update(
            {
                "declared_point_count": len(declared),
                "rendered_point_count": rendered,
                "object_ids": [item[0] for item in declared],
                "status": "ok",
            }
        )
    except Exception as exc:
        result.update(
            {
                "declared_point_count": result.get("declared_point_count", 0),
                "rendered_point_count": 0,
                "status": "ground_truth_overlay_unresolved",
                "error": str(exc),
            }
        )
    finally:
        decoded.unlink(missing_ok=True)
    return result


def atomic_write_text(path: Path, content: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_overlay_html(path: Path, manifest: Mapping[str, Any]) -> None:
    rows = []
    for source in manifest["sources"]:
        for overlay in source["overlays"]:
            png = Path(overlay["overlay_png"])
            try:
                relative = png.relative_to(path.parent.resolve()).as_posix()
            except ValueError:
                relative = str(png)
            image = (
                f'<img src="{html.escape(relative, quote=True)}" width="480" loading="lazy">'
                if overlay["status"] == "ok"
                else html.escape(overlay.get("error", "unresolved"))
            )
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(overlay['dataset']))}</td>"
                f"<td>{html.escape(str(overlay['video']))}</td>"
                f"<td>{html.escape(str(overlay['clip']))}</td>"
                f"<td>{html.escape(str(overlay['temporal_role']))}</td>"
                f"<td>{overlay['source_frame']} / {overlay['local_frame']}</td>"
                f"<td>{overlay.get('declared_point_count', 0)} / {overlay.get('rendered_point_count', 0)}</td>"
                f"<td>{image}</td>"
                "</tr>"
            )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Molmo2 VideoTrack GT overlay audit</title>
<style>body{font-family:Arial,sans-serif;margin:20px;color:#202124}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #c8cdd2;padding:6px;text-align:left;vertical-align:top}th{background:#eef1f3}
img{height:auto;max-width:100%}.status{font-weight:bold}</style></head><body>
"""
    document += f'<p class="status">Status: {html.escape(str(manifest["status"]))}</p>'
    document += "<table><thead><tr><th>Dataset</th><th>Video</th><th>Clip</th><th>Role</th>"
    document += "<th>Source / local frame</th><th>Declared / rendered</th><th>GT overlay</th></tr></thead><tbody>"
    document += "".join(rows)
    document += "</tbody></table></body></html>\n"
    atomic_write_text(path, document)


def generate_overlay_audit(
    sources: Sequence[SourceDefinition],
    index: Mapping[str, Mapping[str, Any]],
    overlay_dir: Path,
    samples_per_source: int,
    media_tool: MediaTool,
) -> dict[str, Any]:
    overlay_dir = overlay_dir.expanduser().resolve()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = overlay_dir / f".decode-{os.getpid()}-{uuid.uuid4().hex}"
    temporary_dir.mkdir(exist_ok=False)
    source_results = []
    try:
        for source in sources:
            indexed_keys = sorted(key for key in index if key.startswith(source.dataset + "::"))
            if not indexed_keys:
                source_results.append(
                    {
                        "dataset": source.dataset,
                        "status": "skipped_no_indexed_clips",
                        "indexed_clips": 0,
                        "required_sample_clips": 0,
                        "selected_sample_clips": 0,
                        "candidate_exclusions": {},
                        "overlays": [],
                    }
                )
                continue
            try:
                candidates, exclusions = load_overlay_candidates(source, index)
            except Exception as exc:
                source_results.append(
                    {
                        "dataset": source.dataset,
                        "status": "ground_truth_overlay_unresolved",
                        "indexed_clips": len(indexed_keys),
                        "required_sample_clips": samples_per_source,
                        "selected_sample_clips": 0,
                        "candidate_exclusions": {},
                        "error": str(exc),
                        "overlays": [],
                    }
                )
                continue
            selected = candidates[:samples_per_source]
            overlays = []
            for candidate in selected:
                for role, offset in zip(("first", "middle", "last"), candidate["temporal_offsets"]):
                    overlays.append(
                        render_overlay(
                            candidate,
                            role,
                            offset,
                            overlay_dir,
                            temporary_dir,
                            media_tool,
                        )
                    )
            unresolved = [item for item in overlays if item["status"] != "ok"]
            enough = len(selected) == samples_per_source
            status = "complete" if enough and not unresolved else "ground_truth_overlay_unresolved"
            source_results.append(
                {
                    "dataset": source.dataset,
                    "status": status,
                    "indexed_clips": len(indexed_keys),
                    "eligible_candidate_clips": len(candidates),
                    "required_sample_clips": samples_per_source,
                    "selected_sample_clips": len(selected),
                    "candidate_exclusions": exclusions,
                    "overlays": overlays,
                }
            )
    finally:
        if temporary_dir.exists():
            remove_tree_below(temporary_dir, overlay_dir)
    unresolved_sources = [
        item for item in source_results if item["status"] == "ground_truth_overlay_unresolved"
    ]
    overlays = [overlay for source in source_results for overlay in source["overlays"]]
    manifest = {
        "format": "molmo2-videotrack-ground-truth-overlay-audit",
        "version": 1,
        "status": "complete" if not unresolved_sources else "ground_truth_overlay_unresolved",
        "selection": {
            "sample_clips_per_indexed_source": samples_per_source,
            "frames_per_clip": ["first_visible", "middle_visible", "last_visible"],
            "ordering": "video, clip, row_id",
        },
        "accounting": {
            "sources": len(source_results),
            "unresolved_sources": len(unresolved_sources),
            "overlays": len(overlays),
            "declared_points": sum(item.get("declared_point_count", 0) for item in overlays),
            "rendered_points": sum(item.get("rendered_point_count", 0) for item in overlays),
        },
        "sources": source_results,
    }
    manifest_path = (overlay_dir / "overlay_manifest.json").resolve()
    html_path = (overlay_dir / "index.html").resolve()
    manifest["manifest_path"] = str(manifest_path)
    manifest["html_path"] = str(html_path)
    atomic_write_json(manifest_path, manifest)
    write_overlay_html(html_path, manifest)
    return manifest


def source_definitions(args: argparse.Namespace) -> list[SourceDefinition]:
    root = args.dataset_root.resolve()

    def chosen(name: str, relative: str) -> Path:
        value = getattr(args, name)
        return (value if value is not None else root / relative).expanduser().resolve()

    definitions = {
        "dancetrack": SourceDefinition(
            "dancetrack",
            chosen("dancetrack_parquet", "data/dancetrack/dancetrack_point_tracks.parquet"),
            "dancetrack_zip",
            (
                chosen("dancetrack_train1_zip", "data/dancetrack/downloads/train1.zip"),
                chosen("dancetrack_train2_zip", "data/dancetrack/downloads/train2.zip"),
            ),
            ("train1", "train2"),
        ),
        "soccernet": SourceDefinition(
            "soccernet",
            chosen("soccernet_parquet", "data/soccernet/soccernet_point_tracks.parquet"),
            "soccernet_zip",
            (chosen("soccernet_train_zip", "data/soccernet/vedio/tracking/train.zip"),),
        ),
        "mose": SourceDefinition(
            "mose",
            chosen("mose_parquet", "data/mose/mose_point_tracks.parquet"),
            "mose_outer_zip",
            (chosen("mose_release_zip", "data/mose/vedio/MOSE_release.zip"),),
        ),
        "mosev2": SourceDefinition(
            "mosev2",
            chosen("mosev2_parquet", "data/mosev2/mosev2_point_tracks.parquet"),
            "mosev2_parts",
            (
                chosen("mosev2_part_aa", "data/mosev2/vedio/train.tar.gz.aa"),
                chosen("mosev2_part_ab", "data/mosev2/vedio/train.tar.gz.ab"),
                chosen("mosev2_part_ac", "data/mosev2/vedio/train.tar.gz.ac"),
            ),
            checksum_path=chosen("mosev2_checksums", "data/mosev2/vedio/SHA256SUMS"),
        ),
        "vipseg": SourceDefinition(
            "vipseg",
            chosen("vipseg_parquet", "data/vipseg/vipseg_point_tracks.parquet"),
            "vipseg_tar",
            (chosen("vipseg_tar", "data/vipseg/vedio/VIPSeg.tar"),),
        ),
    }
    return [definitions[name] for name in DATASET_ORDER if name in args.datasets]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create audited <=128-frame H.264 MP4 windows for Molmo2-VideoTrack."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=list(DATASET_ORDER))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("--overlay-samples-per-source", type=int, default=3)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-inflight-videos", type=int, default=None)
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    parser.add_argument("--max-window-frames", type=int, default=128)
    parser.add_argument("--max-scratch-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--max-failure-examples", type=int, default=200)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--resume", action="store_true")
    for name in (
        "dancetrack_parquet",
        "dancetrack_train1_zip",
        "dancetrack_train2_zip",
        "soccernet_parquet",
        "soccernet_train_zip",
        "mose_parquet",
        "mose_release_zip",
        "mosev2_parquet",
        "mosev2_part_aa",
        "mosev2_part_ab",
        "mosev2_part_ac",
        "mosev2_checksums",
        "vipseg_parquet",
        "vipseg_tar",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=None)
    args = parser.parse_args(argv)
    if args.workers <= 0 or args.ffmpeg_threads <= 0:
        parser.error("--workers and --ffmpeg-threads must be positive")
    if args.max_window_frames <= 0 or args.max_window_frames > 128:
        parser.error("--max-window-frames must be in [1, 128]")
    if args.max_scratch_bytes <= 0:
        parser.error("--max-scratch-bytes must be positive")
    if args.max_failure_examples < 0:
        parser.error("--max-failure-examples must be non-negative")
    if args.overlay_samples_per_source <= 0:
        parser.error("--overlay-samples-per-source must be positive")
    if args.max_inflight_videos is None:
        args.max_inflight_videos = args.workers * 2
    if args.max_inflight_videos <= 0:
        parser.error("--max-inflight-videos must be positive")
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.scratch_dir = args.scratch_dir.expanduser().resolve()
    args.index_path = (
        args.index_path.expanduser().resolve()
        if args.index_path is not None
        else args.output_dir.parent / "video_track_media.json"
    )
    args.report_path = (
        args.report_path.expanduser().resolve()
        if args.report_path is not None
        else args.index_path.with_name("video_track_media_report.json")
    )
    args.overlay_dir = (
        args.overlay_dir.expanduser().resolve()
        if args.overlay_dir is not None
        else args.index_path.parent / "ground_truth_overlay_audit"
    )
    args.datasets = tuple(dict.fromkeys(args.datasets))
    return args


def report_for(
    args: argparse.Namespace,
    outcomes: Sequence[DatasetOutcome],
    index: Mapping[str, Mapping[str, Any]],
    budget: ScratchBudget,
    overlay_audit: Mapping[str, Any],
) -> dict[str, Any]:
    failures = [failure for outcome in outcomes for failure in outcome.failures]
    reason_counts = Counter(failure["reason"] for failure in failures)
    window_counts = Counter(
        str(window["source_end_frame"] - window["source_start_frame"] + 1)
        for value in index.values()
        for window in value["windows"]
    )
    unique_paths = {
        window["path"]
        for value in index.values()
        for window in value["windows"]
    }
    failed_sources = sum(outcome.fatal for outcome in outcomes)
    overlay_failed = overlay_audit["status"] != "complete"
    if overlay_failed:
        reason_counts["ground_truth_overlay_unresolved"] += overlay_audit["accounting"][
            "unresolved_sources"
        ]
    status = (
        "failed"
        if overlay_failed
        else ("complete" if not failures and not failed_sources else ("partial" if index else "failed"))
    )
    examples = failures[: args.max_failure_examples]
    return {
        "format": "molmo2-videotrack-media-materialization-report",
        "version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "parameters": {
            "datasets": list(args.datasets),
            "dataset_root": str(args.dataset_root),
            "output_dir": str(args.output_dir),
            "index_path": str(args.index_path),
            "scratch_dir": str(args.scratch_dir),
            "overlay_dir": str(args.overlay_dir),
            "overlay_samples_per_source": args.overlay_samples_per_source,
            "workers": args.workers,
            "max_inflight_videos": args.max_inflight_videos,
            "ffmpeg_threads": args.ffmpeg_threads,
            "max_window_frames": args.max_window_frames,
            "max_scratch_bytes": args.max_scratch_bytes,
            "resume": args.resume,
        },
        "index_accounting": {
            "clips": len(index),
            "window_references": sum(len(value["windows"]) for value in index.values()),
            "unique_media_files": len(unique_paths),
            "window_frame_counts": dict(sorted(window_counts.items(), key=lambda item: int(item[0]))),
            "lineages": len({value["lineage_key"] for value in index.values()}),
        },
        "source_accounting": [outcome.accounting for outcome in outcomes],
        "failure_accounting": {
            "total": len(failures)
            + (
                overlay_audit["accounting"]["unresolved_sources"]
                if overlay_failed
                else 0
            ),
            "by_reason": dict(sorted(reason_counts.items())),
            "examples": examples,
            "examples_omitted": max(0, len(failures) - len(examples)),
        },
        "scratch_accounting": {
            "hard_limit_bytes": budget.limit,
            "peak_cached_frame_bytes": budget.peak,
            "remaining_cached_frame_bytes": budget.current,
        },
        "ground_truth_overlay_audit": {
            "status": overlay_audit["status"],
            "accounting": overlay_audit["accounting"],
            "manifest_path": overlay_audit["manifest_path"],
            "html_path": overlay_audit["html_path"],
            "sources": [
                {
                    "dataset": source["dataset"],
                    "status": source["status"],
                    "indexed_clips": source["indexed_clips"],
                    "selected_sample_clips": source["selected_sample_clips"],
                }
                for source in overlay_audit["sources"]
            ],
        },
        "io_complexity": {
            "dancetrack_soccernet": "one ZIP central-directory scan plus one random decompression per selected frame",
            "mose_mosev2": "one sequential compressed-TAR pass; selected frames dispatched once",
            "vipseg": "one uncompressed-TAR header scan plus one random read per selected local frame",
            "temporary_space": (
                "frame caches are hard bounded by max_scratch_bytes; each local encoded MP4 "
                "is budget-checked after encoding and removed immediately after sequential upload"
            ),
            "parallelism": (
                "ffmpeg encoding is bounded across source videos; VIPSeg official resize remains serial "
                "to preserve streaming cache semantics"
            ),
        },
        "converter_compatibility_assumptions": [
            "top-level keys are video_dataset::clip and each value contains a windows list",
            "source_start_frame/source_end_frame are zero-based inclusive and form an exact <=128-frame partition",
            "all windows of one source video retain source_video_id and lineage_key",
            "MOSE and MOSEv2 use mose-family lineage but never reuse MP4 because dimensions/fps are source-specific",
            "VIPSeg frames reproduce official change2_720p Pillow BILINEAR resizing before coordinate use",
            "ground-truth point overlays must use the exact encoded frame and transformed coordinate system",
        ],
    }


def validate_write_locations(args: argparse.Namespace) -> None:
    source_root = args.dataset_root.resolve()
    write_locations = {
        "output_dir": args.output_dir,
        "index_path": args.index_path,
        "report_path": args.report_path,
        "overlay_dir": args.overlay_dir,
        "scratch_dir": args.scratch_dir,
    }
    for label, path in write_locations.items():
        resolved = path.resolve()
        if resolved == source_root or _is_below(resolved, source_root):
            raise MaterializationError(
                f"{label} must be outside the read-only dataset root {source_root}: {resolved}"
            )
    if args.index_path == args.report_path:
        raise MaterializationError("--index-path and --report-path must be different")
    if args.output_dir == args.overlay_dir:
        raise MaterializationError("--output-dir and --overlay-dir must be different")
    scratch_anchor = Path(args.scratch_dir.anchor).resolve()
    if args.scratch_dir == scratch_anchor:
        raise MaterializationError("--scratch-dir cannot be a filesystem root")


def run(args: argparse.Namespace, media_tool: MediaTool | None = None) -> tuple[int, dict[str, Any]]:
    validate_write_locations(args)
    if media_tool is None:
        for command in (args.ffmpeg, args.ffprobe):
            if shutil.which(command) is None:
                raise SystemExit(f"required executable is not available: {command}")
        media_tool = FfmpegMediaTool(args.ffmpeg, args.ffprobe, args.ffmpeg_threads)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    run_id = f"molmo2-videotrack-{os.getpid()}-{uuid.uuid4().hex}"
    scratch_run = (args.scratch_dir / run_id).resolve()
    if not _is_below(scratch_run, args.scratch_dir):
        raise MaterializationError(f"unsafe scratch run path: {scratch_run}")
    staging_run = (args.output_dir.parent / f".{run_id}.staging").resolve()
    scratch_run.mkdir(parents=True, exist_ok=False)
    staging_run.mkdir(parents=True, exist_ok=False)
    budget = ScratchBudget(args.max_scratch_bytes)
    outcomes: list[DatasetOutcome] = []
    index: dict[str, dict[str, Any]] = {}
    installed_media: list[tuple[Path, Path]] = []
    metadata_backups: list[tuple[Path, Path | None]] = []
    try:
        definitions = source_definitions(args)
        for source in definitions:
            outcome = process_dataset(
                source,
                args.output_dir,
                scratch_run,
                staging_run / "media",
                args.max_window_frames,
                args.workers,
                args.max_inflight_videos,
                args.resume,
                budget,
                media_tool,
            )
            outcomes.append(outcome)
            for key, value in outcome.entries.items():
                if key in index and index[key] != value:
                    raise MaterializationError(f"conflicting index key across sources: {key}")
                index[key] = value
        staged_files = [pair for outcome in outcomes for pair in outcome.staged_files]
        metadata_backups = backup_metadata_files(
            (
                args.index_path,
                args.report_path,
                args.overlay_dir / "overlay_manifest.json",
                args.overlay_dir / "index.html",
            ),
            staging_run / "metadata-backups",
        )
        if index:
            installed_media = install_staged_media(staged_files)
            atomic_write_json(args.index_path, index)
        overlay_audit = generate_overlay_audit(
            definitions,
            index,
            args.overlay_dir,
            args.overlay_samples_per_source,
            media_tool,
        )
        report = report_for(args, outcomes, index, budget, overlay_audit)
        atomic_write_json(args.report_path, report)
        exit_code = 0 if report["status"] == "complete" else 2
        return exit_code, report
    except Exception as exc:
        rollback_errors = []
        try:
            rollback_installed_media(installed_media)
        except Exception as rollback_exc:
            rollback_errors.append(str(rollback_exc))
        try:
            restore_metadata_files(metadata_backups)
        except Exception as rollback_exc:
            rollback_errors.append(str(rollback_exc))
        if rollback_errors:
            raise MaterializationError(
                f"run failed ({exc}); rollback also failed: {'; '.join(rollback_errors)}"
            ) from exc
        raise
    finally:
        if scratch_run.exists():
            remove_cache_directory(scratch_run, args.scratch_dir)
        if staging_run.exists():
            remove_tree_below(staging_run, args.output_dir.parent)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    exit_code, report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "clips": report["index_accounting"]["clips"],
                "window_references": report["index_accounting"]["window_references"],
                "failures": report["failure_accounting"]["total"],
                "index": str(args.index_path),
                "report": str(args.report_path),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
