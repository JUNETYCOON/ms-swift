#!/usr/bin/env python3
"""Resolve LLaVA v1.5 mix665k images and convert it to ms-swift JSONL.

The converter is tailored to the standard ``llava_v1_5_mix665k.json`` image
prefixes while keeping every source path configurable. It performs four steps:

1. Stream the top-level LLaVA JSON array and collect unique image references.
2. Optionally extract missing OCR-VQA images from embedded parquet bytes.
3. Resolve every image from its dataset root and drop/report invalid rows.
4. Convert to ms-swift JSONL and create a deterministic train/validation split.

The default COCO and OCR-VQA roots point at the pre-extracted image directories
below ``dataset_ms-swift/llava-instruct``. Both flat image folders and retained
``train2017``/``images`` subdirectories are supported. GQA's
``images/train_balanced`` and ``images/val_balanced`` layouts are supported
directly. Every output ``images`` value is an absolute path to an existing
source image. Image extraction is disabled by default and is available only
through explicit opt-in flags. OCR-VQA parquet input and JPEG output use
separate configurable roots. Successful OCR parquet tasks are checkpointed so
an interrupted rerun only scans unfinished or changed shards.

Default output layout::

    /mnt/luojunkun/stage1/dataset_ms-swift/llava-instruct/
    |-- llava_v1_5_mix665k_sft_msswift.jsonl
    |-- llava_v1_5_mix665k_sft_msswift_train.jsonl
    |-- llava_v1_5_mix665k_sft_msswift_val.jsonl
    `-- prepare_report.json

Example::

    python /ms-swift/scripts/prepare_llava_instruct_swift.py \
        --extract-ocr-vqa --num-workers 100 --overwrite

The output is supervised fine-tuning data. Full-parameter training is selected
later with ``swift sft --train_type full``; it does not require another dataset
format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    wait,
)
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence, TextIO
from urllib.parse import unquote, urlparse


DEFAULT_INPUT_JSON = Path(
    "/mnt/luojunkun/stage1/dataset/llava-instruct/llava_v1_5_mix665k.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/llava-instruct"
)
DEFAULT_COCO_ROOT = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/llava-instruct/coco-train2017"
)
DEFAULT_OCR_VQA_ROOT = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/llava-instruct/ocrvqa-images"
)
DEFAULT_OCR_VQA_PARQUET_ROOT = Path(
    "/mnt/luojunkun/stage1/dataset/llava-instruct/datasets/ocr-vqa/data"
)
DEFAULT_GQA_ROOT = Path("/mnt/luojunkun/stage1/dataset_ms-swift/gqa")
DEFAULT_TEXTVQA_IMAGES_ROOT = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/textvqa/images"
)
DEFAULT_VISUALGENOME_ROOT = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/visualgenome"
)
DEFAULT_OUTPUT_FILE = "llava_v1_5_mix665k_sft_msswift.jsonl"
DEFAULT_OCR_CHECKPOINT_FILE = ".prepare_llava_ocr_checkpoint.json"
OCR_CHECKPOINT_VERSION = 1

READ_CHUNK_SIZE = 1024 * 1024
DIRECT_CHECK_LIMIT = 1000
IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}
IMAGE_TOKEN_RE = re.compile(r"<\s*image\s*>", flags=re.IGNORECASE)
OCR_IMAGE_FIELDS = ("image", "img", "image_data", "image_bytes", "bytes")
OCR_LOOKUP_FIELDS = (
    "image_id",
    "imageId",
    "imageid",
    "id",
    "image_name",
    "image_path",
    "image_file",
    "filename",
    "file_name",
    "path",
    "url",
    "image_url",
    "imageURL",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    prefixes: tuple[str, ...]


@dataclass
class ScanResult:
    stats: Counter
    references: dict[str, set[str]]
    unknown_references: set[str]


@dataclass
class ResolveResult:
    paths: dict[str, Path]
    owners: dict[str, str]
    missing: list[str]
    ambiguous: dict[str, list[str]]


class ConversionValidationError(RuntimeError):
    def __init__(self, message: str, stats: Counter, examples: list[str]) -> None:
        super().__init__(message)
        self.stats = stats
        self.examples = examples


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{size}B"


class ProgressLogger:
    """Print stage transitions and time-based heartbeats without a background thread."""

    def __init__(self, heartbeat_seconds: float) -> None:
        self.heartbeat_seconds = heartbeat_seconds
        self.run_started = time.monotonic()
        self.stage_started = self.run_started
        self.last_heartbeat = self.run_started
        self.stage = "startup"
        self.timings: dict[str, float] = {}

    @property
    def wait_timeout(self) -> float | None:
        return self.heartbeat_seconds if self.heartbeat_seconds > 0 else None

    def begin(self, stage: str, detail: str = "") -> None:
        now = time.monotonic()
        self.stage = stage
        self.stage_started = now
        self.last_heartbeat = now
        suffix = f" {detail}" if detail else ""
        print(
            f"[stage:{stage}] start time={time.strftime('%Y-%m-%d %H:%M:%S')}{suffix}",
            flush=True,
        )

    def heartbeat(self, detail: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if not force and (
            self.heartbeat_seconds <= 0
            or now - self.last_heartbeat < self.heartbeat_seconds
        ):
            return
        self.last_heartbeat = now
        suffix = f" {detail}" if detail else ""
        print(
            f"[heartbeat] time={time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"stage={self.stage} stage_elapsed={format_duration(now - self.stage_started)} "
            f"total_elapsed={format_duration(now - self.run_started)}{suffix}",
            flush=True,
        )

    def finish(self, detail: str = "") -> None:
        now = time.monotonic()
        elapsed = now - self.stage_started
        self.timings[self.stage] = self.timings.get(self.stage, 0.0) + elapsed
        suffix = f" {detail}" if detail else ""
        print(
            f"[stage:{self.stage}] complete elapsed={format_duration(elapsed)}{suffix}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LLaVA v1.5 mix665k to ms-swift multimodal SFT JSONL."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="JSONL filename below --output-dir, or an absolute path.",
    )
    parser.add_argument(
        "--train-output-file",
        type=Path,
        default=None,
        help="Train JSONL path after splitting. Defaults to '<output_stem>_train.jsonl'.",
    )
    parser.add_argument(
        "--val-output-file",
        type=Path,
        default=None,
        help="Validation JSONL path after splitting. Defaults to '<output_stem>_val.jsonl'.",
    )
    parser.add_argument("--report-file", default="prepare_report.json")
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=DEFAULT_COCO_ROOT,
        help="COCO image directory; flat and train2017 subdirectory layouts are supported.",
    )
    parser.add_argument(
        "--ocr-vqa-root",
        type=Path,
        default=DEFAULT_OCR_VQA_ROOT,
        help="OCR-VQA image directory; flat and images subdirectory layouts are supported.",
    )
    parser.add_argument(
        "--ocr-vqa-parquet-root",
        type=Path,
        default=DEFAULT_OCR_VQA_PARQUET_ROOT,
        help="Source directory containing OCR-VQA parquet shards with embedded images.",
    )
    parser.add_argument("--gqa-root", type=Path, default=DEFAULT_GQA_ROOT)
    parser.add_argument(
        "--textvqa-images-root", type=Path, default=DEFAULT_TEXTVQA_IMAGES_ROOT
    )
    parser.add_argument(
        "--visualgenome-root", type=Path, default=DEFAULT_VISUALGENOME_ROOT
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Worker processes for JSON conversion; 0 uses up to 16 CPUs.",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=0,
        help="Worker processes for source-record classification; 0 uses up to 16.",
    )
    parser.add_argument(
        "--resolve-workers",
        type=int,
        default=0,
        help="Worker processes for dataset image resolution; 0 uses up to 5.",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=0,
        help="Worker processes for OCR-VQA parquet extraction; 0 uses up to 8.",
    )
    parser.add_argument(
        "--split-workers",
        type=int,
        default=0,
        help="Worker processes for train/validation file writing; 0 uses up to 16.",
    )
    parser.add_argument(
        "--ocr-task-mode",
        choices=("shard", "row-group"),
        default="shard",
        help=(
            "OCR parquet task granularity. 'shard' avoids concurrent reads of the "
            "same large file and is recommended for shared storage."
        ),
    )
    parser.add_argument(
        "--ocr-output-shard-chars",
        type=int,
        default=2,
        help=(
            "Place extracted OCR images in prefix subdirectories using this many "
            "filename characters; 0 keeps a flat directory."
        ),
    )
    parser.add_argument(
        "--ocr-checkpoint-file",
        type=Path,
        default=None,
        help=(
            "Persistent OCR parquet task checkpoint. Defaults to "
            "'<ocr-vqa-root>/.prepare_llava_ocr_checkpoint.json'; relative paths "
            "are resolved below --output-dir."
        ),
    )
    parser.add_argument(
        "--reset-ocr-checkpoint",
        action="store_true",
        help="Ignore the saved OCR parquet task checkpoint and rebuild it.",
    )
    parser.add_argument(
        "--scan-chunk-size",
        type=int,
        default=1024,
        help="Source records per parallel scan task.",
    )
    parser.add_argument(
        "--worker-chunk-size",
        type=int,
        default=512,
        help="LLaVA records per process-pool task.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=256,
        help="Rows per OCR-VQA parquet batch.",
    )
    parser.add_argument(
        "--split-chunk-rows",
        type=int,
        default=4096,
        help="Clean JSONL rows per parallel train/validation split task.",
    )
    parser.add_argument(
        "--max-pending-tasks",
        type=int,
        default=0,
        help="Maximum queued scan/conversion tasks; 0 uses twice --num-workers.",
    )
    parser.add_argument(
        "--missing-image-policy",
        choices=("error", "skip"),
        default="skip",
        help="Fail before conversion when images are missing, or skip affected rows.",
    )
    parser.add_argument(
        "--invalid-record-policy",
        choices=("error", "skip"),
        default="skip",
        help="Fail on malformed conversations/token mismatches, or skip them.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio from the cleaned output JSONL. Use 0 to disable splitting.",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for train/val splitting.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N source records; intended for validation runs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N source records; 0 disables periodic logs.",
    )
    parser.add_argument(
        "--media-progress-every",
        type=int,
        default=1000,
        help=(
            "Print COCO/OCR-VQA image preparation progress every N images or rows "
            "inside worker processes; 0 disables worker media progress logs."
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=0.0,
        help="Print the active stage every N seconds during long operations; disabled by default.",
    )
    parser.add_argument(
        "--max-error-examples",
        type=int,
        default=30,
        help="Maximum missing/invalid examples stored in the report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output JSONL and report.",
    )
    parser.add_argument(
        "--overwrite-media",
        action="store_true",
        help="Rewrite COCO/OCR-VQA images that already exist.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Prepare and validate source image paths, then write only the report.",
    )
    coco_extraction = parser.add_mutually_exclusive_group()
    coco_extraction.add_argument(
        "--extract-coco",
        dest="extract_coco",
        action="store_true",
        help="Opt in to extracting missing COCO images from ZIP archives.",
    )
    coco_extraction.add_argument(
        "--no-extract-coco",
        dest="extract_coco",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    ocr_extraction = parser.add_mutually_exclusive_group()
    ocr_extraction.add_argument(
        "--extract-ocr-vqa",
        dest="extract_ocr_vqa",
        action="store_true",
        help="Opt in to extracting missing OCR-VQA images from parquet shards.",
    )
    ocr_extraction.add_argument(
        "--no-extract-ocr-vqa",
        dest="extract_ocr_vqa",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(extract_coco=False, extract_ocr_vqa=False)
    return parser.parse_args()


def split_output_path(value: Path | None, output_path: Path, output_dir: Path, suffix: str) -> Path:
    if value is None:
        return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
    value = value.expanduser()
    if not value.is_absolute():
        value = output_dir / value
    return value.absolute()


def resolve_cli_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    args.input_json = args.input_json.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.coco_root = args.coco_root.expanduser().resolve()
    args.ocr_vqa_root = args.ocr_vqa_root.expanduser().resolve()
    args.ocr_vqa_parquet_root = args.ocr_vqa_parquet_root.expanduser().resolve()
    args.gqa_root = args.gqa_root.expanduser().resolve()
    args.textvqa_images_root = args.textvqa_images_root.expanduser().resolve()
    args.visualgenome_root = args.visualgenome_root.expanduser().resolve()

    if args.ocr_checkpoint_file is None:
        args.ocr_checkpoint_file = args.ocr_vqa_root / DEFAULT_OCR_CHECKPOINT_FILE
    else:
        args.ocr_checkpoint_file = args.ocr_checkpoint_file.expanduser()
        if not args.ocr_checkpoint_file.is_absolute():
            args.ocr_checkpoint_file = args.output_dir / args.ocr_checkpoint_file
        args.ocr_checkpoint_file = args.ocr_checkpoint_file.absolute()

    output_path = Path(args.output_file).expanduser()
    if not output_path.is_absolute():
        output_path = args.output_dir / output_path
    output_path = output_path.absolute()

    report_path = Path(args.report_file).expanduser()
    if not report_path.is_absolute():
        report_path = args.output_dir / report_path
    report_path = report_path.absolute()

    train_path = split_output_path(args.train_output_file, output_path, args.output_dir, "train")
    val_path = split_output_path(args.val_output_file, output_path, args.output_dir, "val")
    return args.input_json, output_path, report_path, train_path, val_path


def validate_args(
    args: argparse.Namespace,
    output_path: Path,
    report_path: Path,
    train_path: Path,
    val_path: Path,
) -> None:
    if not args.input_json.is_file():
        raise FileNotFoundError(f"LLaVA JSON does not exist: {args.input_json}")
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError("--output-file must end in .jsonl")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be 0 or greater")
    for option in ("scan_workers", "resolve_workers", "ocr_workers", "split_workers"):
        if getattr(args, option) < 0:
            raise ValueError(f"--{option.replace('_', '-')} must be 0 or greater")
    if args.scan_chunk_size <= 0:
        raise ValueError("--scan-chunk-size must be greater than zero")
    if args.worker_chunk_size <= 0:
        raise ValueError("--worker-chunk-size must be greater than zero")
    if args.parquet_batch_size <= 0:
        raise ValueError("--parquet-batch-size must be greater than zero")
    if args.split_chunk_rows <= 0:
        raise ValueError("--split-chunk-rows must be greater than zero")
    if not 0 <= args.ocr_output_shard_chars <= 8:
        raise ValueError("--ocr-output-shard-chars must be in the range [0, 8]")
    if args.max_pending_tasks < 0:
        raise ValueError("--max-pending-tasks must be 0 or greater")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be 0 or greater")
    if args.media_progress_every < 0:
        raise ValueError("--media-progress-every must be 0 or greater")
    if args.heartbeat_seconds < 0:
        raise ValueError("--heartbeat-seconds must be 0 or greater")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in the range [0, 1)")
    if args.max_error_examples <= 0:
        raise ValueError("--max-error-examples must be greater than zero")
    if args.extract_ocr_vqa and not args.ocr_vqa_parquet_root.is_dir():
        raise FileNotFoundError(
            f"OCR-VQA parquet root does not exist: {args.ocr_vqa_parquet_root}"
        )
    if args.ocr_checkpoint_file.exists() and not args.ocr_checkpoint_file.is_file():
        raise ValueError(
            f"--ocr-checkpoint-file must be a file path: {args.ocr_checkpoint_file}"
        )
    distinct_paths = [output_path, report_path, args.ocr_checkpoint_file]
    if args.val_ratio > 0 and not args.check_only:
        distinct_paths.extend([train_path, val_path])
    if len(set(distinct_paths)) != len(distinct_paths):
        raise ValueError(
            "--output-file, --report-file, --train-output-file, "
            "--val-output-file, and --ocr-checkpoint-file must be different"
        )
    output_paths = [output_path]
    if args.val_ratio > 0:
        output_paths.extend([train_path, val_path])
    for path in output_paths:
        if path.exists() and not args.overwrite and not args.check_only:
            raise FileExistsError(f"Output already exists; pass --overwrite: {path}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report already exists; pass --overwrite: {report_path}")


def dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="coco",
            root=args.coco_root,
            prefixes=("coco", "coco2017"),
        ),
        DatasetSpec(
            name="ocr_vqa",
            root=args.ocr_vqa_root,
            prefixes=("ocr_vqa", "ocr-vqa", "ocrvqa"),
        ),
        DatasetSpec(
            name="gqa",
            root=args.gqa_root,
            prefixes=("gqa",),
        ),
        DatasetSpec(
            name="textvqa",
            root=args.textvqa_images_root,
            prefixes=("textvqa", "text_vqa", "text-vqa"),
        ),
        DatasetSpec(
            name="visualgenome",
            root=args.visualgenome_root,
            prefixes=("vg", "visualgenome", "visual_genome", "visual-genome"),
        ),
    ]


def normalized_reference(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().replace("\\", "/")
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme == "file":
        text = unquote(parsed.path)
    return text


def image_references(record: dict[str, Any]) -> list[str]:
    value = record.get("images")
    if value is None:
        value = record.get("image")
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [reference for item in values if (reference := normalized_reference(item))]


def classify_reference(reference: str, specs: Sequence[DatasetSpec]) -> str | None:
    normalized = reference.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        for spec in specs:
            try:
                path.resolve().relative_to(spec.root)
                return spec.name
            except (OSError, ValueError):
                continue

    parts = PurePosixPath(normalized.lstrip("/")).parts
    lowered = {part.lower() for part in parts[:3]}
    for spec in specs:
        if lowered.intersection(spec.prefixes):
            return spec.name
    return None


def iter_json_array(stream: TextIO) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading it in RAM."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    eof = False

    def read_more() -> bool:
        nonlocal buffer, position, eof
        if eof:
            return False
        chunk = stream.read(READ_CHUNK_SIZE)
        if not chunk:
            eof = True
            return False
        buffer = buffer[position:] + chunk
        position = 0
        return True

    while True:
        while position >= len(buffer) and read_more():
            pass
        while position < len(buffer) and buffer[position].isspace():
            position += 1

        if not started:
            if position >= len(buffer):
                raise ValueError("Input JSON is empty")
            if buffer[position] != "[":
                raise ValueError("Expected a top-level JSON array")
            position += 1
            started = True
            continue

        while True:
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) or not read_more():
                break

        if position < len(buffer) and buffer[position] == "]":
            return
        if position >= len(buffer) and eof:
            raise ValueError("Unexpected end of top-level JSON array")

        while True:
            try:
                value, end = decoder.raw_decode(buffer, position)
                position = end
                break
            except json.JSONDecodeError as exc:
                if not read_more():
                    raise ValueError(f"Invalid or truncated JSON: {exc}") from None
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected each top-level item to be an object, got {type(value).__name__}"
            )
        yield value


def iter_jsonl(stream: TextIO) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(stream, start=1):
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected a JSON object on line {line_number}, got {type(value).__name__}"
            )
        yield value


def iter_source_records(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        first = ""
        while not first:
            character = stream.read(1)
            if not character:
                raise ValueError(f"Input file is empty: {path}")
            if not character.isspace():
                first = character
        stream.seek(0)
        iterator = iter_json_array(stream) if first == "[" else iter_jsonl(stream)
        for index, record in enumerate(iterator):
            if limit is not None and index >= limit:
                break
            yield record


_SCAN_SPECS: tuple[DatasetSpec, ...] = ()


def init_scan_worker(specs: Sequence[DatasetSpec]) -> None:
    global _SCAN_SPECS
    _SCAN_SPECS = tuple(specs)


def scan_record_chunk(
    records: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, set[str]], set[str]]:
    references: dict[str, set[str]] = {spec.name: set() for spec in _SCAN_SPECS}
    unknown_references: set[str] = set()
    stats: Counter = Counter()

    for record in records:
        stats["records"] += 1
        refs = image_references(record)
        if refs:
            stats["multimodal_records"] += 1
        else:
            stats["text_only_records"] += 1
        stats["image_references"] += len(refs)
        for reference in refs:
            owner = classify_reference(reference, _SCAN_SPECS)
            if owner is None:
                unknown_references.add(reference)
                stats["unknown_image_references"] += 1
            else:
                references[owner].add(reference)

    return dict(stats), references, unknown_references


def iter_source_chunks(
    path: Path, limit: int | None, chunk_size: int
) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    for record in iter_source_records(path, limit):
        chunk.append(record)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def scan_source(
    path: Path,
    specs: Sequence[DatasetSpec],
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> ScanResult:
    references: dict[str, set[str]] = {spec.name: set() for spec in specs}
    unknown_references: set[str] = set()
    stats: Counter = Counter()
    chunks = iter_source_chunks(path, args.limit, args.scan_chunk_size)
    next_progress = args.progress_every

    def merge_result(
        result: tuple[dict[str, int], dict[str, set[str]], set[str]],
    ) -> None:
        nonlocal next_progress
        chunk_stats, chunk_references, chunk_unknown = result
        stats.update(chunk_stats)
        for name, items in chunk_references.items():
            references[name].update(items)
        unknown_references.update(chunk_unknown)
        if args.progress_every and stats["records"] >= next_progress:
            unique_count = sum(len(items) for items in references.values()) + len(
                unknown_references
            )
            print(
                f"[scan] records={stats['records']:,} "
                f"image_refs={stats['image_references']:,} unique={unique_count:,}",
                flush=True,
            )
            while next_progress <= stats["records"]:
                next_progress += args.progress_every

    if args.scan_workers == 1:
        init_scan_worker(specs)
        for chunk in chunks:
            merge_result(scan_record_chunk(chunk))
            progress.heartbeat(f"records={stats['records']:,}")
    else:
        max_pending = min(
            max(1, args.max_pending_tasks),
            max(1, args.scan_workers * 2),
            128,
        )
        with ProcessPoolExecutor(
            max_workers=args.scan_workers,
            initializer=init_scan_worker,
            initargs=(tuple(specs),),
        ) as executor:
            pending: set[Future] = set()
            for _ in range(max_pending):
                try:
                    pending.add(executor.submit(scan_record_chunk, next(chunks)))
                except StopIteration:
                    break

            while pending:
                done, pending = wait(
                    pending,
                    timeout=progress.wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    progress.heartbeat(
                        f"records={stats['records']:,} pending_chunks={len(pending):,}",
                        force=True,
                    )
                    continue
                for future in done:
                    merge_result(future.result())
                    try:
                        pending.add(executor.submit(scan_record_chunk, next(chunks)))
                    except StopIteration:
                        pass
                progress.heartbeat(
                    f"records={stats['records']:,} pending_chunks={len(pending):,}"
                )

    stats["unique_image_references"] = sum(
        len(items) for items in references.values()
    ) + len(unknown_references)
    print(
        f"[scan] complete records={stats['records']:,} "
        f"multimodal={stats['multimodal_records']:,} "
        f"text_only={stats['text_only_records']:,} "
        f"unique_images={stats['unique_image_references']:,}",
        flush=True,
    )
    return ScanResult(stats, references, unknown_references)


def strip_known_prefix(reference: str, spec: DatasetSpec) -> tuple[str, ...]:
    parts = list(PurePosixPath(reference.replace("\\", "/").lstrip("/")).parts)
    for index, part in enumerate(parts[:3]):
        if part.lower() in spec.prefixes:
            return tuple(parts[index + 1 :])
    return tuple(parts)


def safe_relative_parts(parts: Sequence[str]) -> tuple[str, ...]:
    safe = tuple(part for part in parts if part not in ("", "."))
    if any(part == ".." for part in safe):
        return ()
    return safe


def direct_candidates(reference: str, spec: DatasetSpec) -> list[Path]:
    raw_path = Path(reference)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)

    parts = safe_relative_parts(strip_known_prefix(reference, spec))
    if not parts:
        return candidates

    basename = parts[-1]
    if spec.name in {"coco", "ocr_vqa"}:
        candidates.append(spec.root / basename)
    candidates.append(spec.root.joinpath(*parts))

    if spec.name == "coco":
        if parts[0].lower() == "images" and len(parts) > 1:
            candidates.append(spec.root.joinpath(*parts[1:]))
    elif spec.name == "ocr_vqa":
        candidates.append(spec.root / "images" / basename)
    elif spec.name == "gqa":
        image_parts = parts[1:] if parts[0].lower() == "images" and len(parts) > 1 else parts
        first_image_part = image_parts[0].lower()
        if first_image_part in {"train_balanced", "val_balanced"}:
            candidates.append(spec.root / "images" / Path(*image_parts))
        else:
            candidates.append(
                spec.root / "images" / "train_balanced" / Path(*image_parts)
            )
            candidates.append(
                spec.root / "images" / "val_balanced" / Path(*image_parts)
            )
        candidates.append(spec.root / "images" / basename)
    elif spec.name == "textvqa":
        split_aliases = {
            "train_images": "train",
            "val_images": "validation",
            "validation_images": "validation",
            "test_images": "test",
        }
        first = parts[0].lower()
        if first in split_aliases and len(parts) > 1:
            candidates.append(spec.root / split_aliases[first] / Path(*parts[1:]))
        if first == "images" and len(parts) > 1:
            candidates.append(spec.root.joinpath(*parts[1:]))
        candidates.append(spec.root / basename)
    elif spec.name == "visualgenome":
        if parts[0].lower() in {"images", "image"} and len(parts) > 1:
            candidates.append(spec.root.joinpath(*parts[1:]))

    return list(dict.fromkeys(candidates))


def existing_direct_path(reference: str, spec: DatasetSpec) -> Path | None:
    for candidate in direct_candidates(reference, spec):
        if candidate.suffix.lower() in IMAGE_SUFFIXES and candidate.is_file():
            return candidate.absolute()
    return None


def coco_relative_path(reference: str, spec: DatasetSpec) -> Path | None:
    parts = safe_relative_parts(strip_known_prefix(reference, spec))
    if not parts:
        return None
    if parts[0].lower() == "images" and len(parts) > 1:
        parts = parts[1:]
    path = Path(*parts)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return path


def chunk_sequence(values: Sequence[Any], count: int) -> list[list[Any]]:
    if not values:
        return []
    count = max(1, min(count, len(values)))
    size = (len(values) + count - 1) // count
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def temporary_sibling(path: Path, create_parent: bool = True) -> Path:
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        return Path(stream.name)


def replace_atomically(temporary: Path, target: Path) -> None:
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_bytes_atomically(target: Path, data: bytes) -> None:
    temporary = temporary_sibling(target, create_parent=False)
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
        replace_atomically(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_zip_job(
    job: tuple[str, list[tuple[str, str, int]], bool, int]
) -> dict[str, int]:
    archive_path_text, members, overwrite, media_progress_every = job
    started = time.monotonic()
    archive_name = Path(archive_path_text).name
    total = len(members)
    stats: Counter = Counter()
    with zipfile.ZipFile(archive_path_text) as archive:
        for index, (member_name, target_text, member_size) in enumerate(members, start=1):
            stats["processed"] += 1
            target = Path(target_text)
            if target.exists() and not overwrite:
                stats["reused"] += 1
                stats["reused_source_bytes"] += member_size
                if media_progress_every and index % media_progress_every == 0:
                    print(
                        f"[coco:extract] archive={archive_name} "
                        f"images={index:,}/{total:,} written={stats['written']:,} "
                        f"reused={stats['reused']:,} "
                        f"written_size={format_size(stats['written_source_bytes'])} "
                        f"elapsed={format_duration(time.monotonic() - started)}",
                        flush=True,
                    )
                continue
            temporary = temporary_sibling(target)
            try:
                with archive.open(member_name, "r") as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                replace_atomically(temporary, target)
                stats["written"] += 1
                stats["written_source_bytes"] += member_size
            finally:
                if temporary.exists():
                    temporary.unlink()
            if media_progress_every and index % media_progress_every == 0:
                print(
                    f"[coco:extract] archive={archive_name} "
                    f"images={index:,}/{total:,} written={stats['written']:,} "
                    f"reused={stats['reused']:,} "
                    f"written_size={format_size(stats['written_source_bytes'])} "
                    f"elapsed={format_duration(time.monotonic() - started)}",
                    flush=True,
                )
    return dict(stats)


def extract_required_coco(
    references: set[str],
    spec: DatasetSpec,
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> Counter:
    stats: Counter = Counter()
    missing: dict[str, Path] = {}
    total = len(references)
    print(
        f"[coco] checking existing images references={total:,} root={spec.root}",
        flush=True,
    )
    for index, reference in enumerate(references, start=1):
        if existing_direct_path(reference, spec) is not None:
            stats["already_available"] += 1
        else:
            relative = coco_relative_path(reference, spec)
            if relative is not None:
                missing[reference] = relative
        if index % 1000 == 0:
            progress.heartbeat(
                f"checked={index:,}/{total:,} available={stats['already_available']:,} "
                f"missing={len(missing):,}"
            )
        if args.progress_every and index % args.progress_every == 0:
            print(
                f"[coco] checked={index:,}/{total:,} "
                f"available={stats['already_available']:,} missing={len(missing):,}",
                flush=True,
            )

    stats["requested_missing_before_extract"] = len(missing)
    print(
        f"[coco] check complete available={stats['already_available']:,} "
        f"missing={len(missing):,}",
        flush=True,
    )
    if not missing:
        return stats
    if not args.extract_coco:
        print("[coco] extraction is disabled; using existing images only", flush=True)
        return stats

    print(f"[coco] discovering ZIP archives directly below {spec.root}", flush=True)
    archives = sorted(path for path in spec.root.glob("*.zip") if path.is_file())
    if not archives:
        print(f"[coco] no ZIP archives found below {spec.root}", flush=True)
        return stats
    for archive_path in archives:
        print(
            f"[coco] archive={archive_path} size={format_size(archive_path.stat().st_size)}",
            flush=True,
        )

    by_exact: dict[str, str] = {}
    by_name: dict[str, list[str]] = defaultdict(list)
    for reference, relative in missing.items():
        normalized = relative.as_posix().lower()
        by_exact[normalized] = reference
        by_name[relative.name.lower()].append(reference)

    assignments: dict[Path, list[tuple[str, Path, int]]] = defaultdict(list)
    assigned: set[str] = set()
    for archive_index, archive_path in enumerate(archives, start=1):
        print(
            f"[coco] indexing archive={archive_index}/{len(archives)} path={archive_path}",
            flush=True,
        )
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            print(
                f"[coco] archive members={len(infos):,} path={archive_path.name}",
                flush=True,
            )
            for member_index, info in enumerate(infos, start=1):
                if member_index % 1000 == 0:
                    progress.heartbeat(
                        f"archive={archive_index}/{len(archives)} "
                        f"members={member_index:,}/{len(infos):,} matched={len(assigned):,}"
                    )
                if args.progress_every and member_index % args.progress_every == 0:
                    print(
                        f"[coco] archive={archive_index}/{len(archives)} "
                        f"members={member_index:,}/{len(infos):,} "
                        f"matched_before_current={len(assigned):,}",
                        flush=True,
                    )
                if info.is_dir():
                    continue
                member = PurePosixPath(info.filename)
                if member.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                member_normalized = member.as_posix().lstrip("./").lower()
                reference = by_exact.get(member_normalized)
                if reference is None:
                    matches = by_name.get(member.name.lower(), [])
                    if len(matches) == 1:
                        reference = matches[0]
                if reference is None or reference in assigned:
                    continue
                target = spec.root / missing[reference]
                try:
                    target.absolute().relative_to(spec.root)
                except ValueError:
                    raise ValueError(f"Unsafe COCO extraction target: {target}") from None
                assignments[archive_path].append((info.filename, target, info.file_size))
                assigned.add(reference)

    jobs: list[tuple[str, list[tuple[str, str, int]], bool, int]] = []
    for archive_path, members in assignments.items():
        partitions = chunk_sequence(members, min(args.num_workers, len(members)))
        for partition in partitions:
            jobs.append(
                (
                    str(archive_path),
                    [
                        (member, str(target), member_size)
                        for member, target, member_size in partition
                    ],
                    args.overwrite_media,
                    args.media_progress_every,
                )
            )

    print(
        f"[coco] archives={len(archives):,} matched={len(assigned):,} "
        f"workers={min(args.num_workers, len(jobs)) if jobs else 0} "
        f"media_progress_every={args.media_progress_every:,}",
        flush=True,
    )
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.num_workers, len(jobs))) as executor:
            pending = {executor.submit(extract_zip_job, job) for job in jobs}
            completed = 0
            while pending:
                done, pending = wait(
                    pending,
                    timeout=progress.wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    progress.heartbeat(
                        f"extract_tasks={completed:,}/{len(jobs):,} "
                        f"pending={len(pending):,}",
                        force=True,
                    )
                    continue
                for future in done:
                    result = future.result()
                    stats.update(result)
                    completed += 1
                    print(
                        f"[coco] extract_tasks={completed:,}/{len(jobs):,} "
                        f"processed={stats['processed']:,}/{len(assigned):,} "
                        f"written={stats['written']:,} reused={stats['reused']:,} "
                        f"written_size={format_size(stats['written_source_bytes'])}",
                        flush=True,
                    )
    stats["archive_matches"] = len(assigned)
    stats["unmatched_in_archives"] = len(missing) - len(assigned)
    print(
        f"[coco] extracted={stats['written']:,} reused={stats['reused']:,} "
        f"unmatched={stats['unmatched_in_archives']:,}",
        flush=True,
    )
    return stats


def normalized_lookup_keys(value: Any) -> set[str]:
    if isinstance(value, (int, float, bool)):
        text = str(value)
    else:
        text = normalized_reference(value)
    if not text:
        return set()
    parsed = urlparse(text)
    path_text = unquote(parsed.path) if parsed.scheme else text
    path = PurePosixPath(path_text.replace("\\", "/"))
    keys = {text.lower(), path.as_posix().lower(), path.name.lower()}
    if path.stem:
        keys.add(path.stem.lower())
        if path.stem.isdigit():
            keys.add(str(int(path.stem)))
    return {key for key in keys if key}


def ocr_target_for_reference(
    reference: str,
    spec: DatasetSpec,
    shard_chars: int = 0,
) -> Path:
    parts = safe_relative_parts(strip_known_prefix(reference, spec))
    basename = parts[-1] if parts else PurePosixPath(reference).name
    if shard_chars:
        raw_prefix = Path(basename).stem[:shard_chars].lower()
        prefix = re.sub(r"[^a-z0-9_-]", "_", raw_prefix) or "_"
        target = spec.root / prefix / basename
    else:
        target = spec.root / basename
    try:
        target.absolute().relative_to(spec.root)
    except ValueError:
        raise ValueError(f"Unsafe OCR-VQA extraction target: {target}") from None
    return target


OcrLookupItem = tuple[str, str]
OcrLookupValue = OcrLookupItem | list[OcrLookupItem]


def ocr_reference_lookup_keys(reference: str) -> set[str]:
    stem = PurePosixPath(reference.replace("\\", "/")).stem.lower()
    if not stem:
        return set()
    keys = {stem}
    if stem.isdigit():
        keys.add(str(int(stem)))
    return keys


def ocr_needed_lookup(
    references: set[str],
    spec: DatasetSpec,
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> tuple[dict[str, OcrLookupValue], set[str]]:
    lookup: dict[str, OcrLookupValue] = {}
    print(
        f"[ocr-vqa] resolving existing images before parquet extraction "
        f"references={len(references):,} root={spec.root}",
        flush=True,
    )
    existing, missing_items, _ = resolve_for_spec(references, spec, args, progress)
    missing = set(missing_items)
    target_directories: set[Path] = set()
    for reference in missing:
        target = ocr_target_for_reference(
            reference,
            spec,
            args.ocr_output_shard_chars,
        )
        target_directories.add(target.parent)
        item = (reference, str(target))
        # The canonical file ID is sufficient. Generic components such as
        # "images" would create one huge collision list.
        for key in ocr_reference_lookup_keys(reference):
            current = lookup.get(key)
            if current is None:
                lookup[key] = item
            elif isinstance(current, tuple):
                if current != item:
                    lookup[key] = [current, item]
            elif item not in current:
                current.append(item)
    for directory in target_directories:
        directory.mkdir(parents=True, exist_ok=True)
    print(
        f"[ocr-vqa] existing resolution complete available={len(existing):,} "
        f"missing={len(missing):,} lookup_keys={len(lookup):,} "
        f"target_directories={len(target_directories):,}",
        flush=True,
    )
    return dict(lookup), missing


def image_bytes_from_row(row: dict[str, Any]) -> tuple[bytes | None, str]:
    for field in OCR_IMAGE_FIELDS:
        value = row.get(field)
        if isinstance(value, dict):
            data = value.get("bytes") or value.get("data")
            path_hint = normalized_reference(value.get("path"))
        else:
            data = value
            path_hint = ""
        if isinstance(data, memoryview):
            data = data.tobytes()
        if isinstance(data, bytearray):
            data = bytes(data)
        if isinstance(data, bytes) and data:
            return data, path_hint
    return None, ""


def ocr_row_lookup_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in OCR_LOOKUP_FIELDS:
        keys.update(normalized_lookup_keys(row.get(field)))
    for field in ("image", "img", "image_data"):
        value = row.get(field)
        if isinstance(value, dict):
            for nested in ("path", "filename", "file_name", "id", "url"):
                keys.update(normalized_lookup_keys(value.get(nested)))
    return keys


_OCR_LOOKUP: dict[str, OcrLookupValue] = {}
_OCR_BATCH_SIZE = 256
_OCR_OVERWRITE = False
_OCR_MEDIA_PROGRESS_EVERY = 1000


def init_ocr_worker(
    lookup: dict[str, OcrLookupValue],
    batch_size: int,
    overwrite: bool,
    media_progress_every: int,
) -> None:
    global _OCR_LOOKUP, _OCR_BATCH_SIZE, _OCR_OVERWRITE, _OCR_MEDIA_PROGRESS_EVERY
    _OCR_LOOKUP = lookup
    _OCR_BATCH_SIZE = batch_size
    _OCR_OVERWRITE = overwrite
    _OCR_MEDIA_PROGRESS_EVERY = media_progress_every


def print_ocr_task_progress(
    task_name: str,
    total_rows: int,
    stats: Counter,
    started: float,
) -> None:
    elapsed = max(time.monotonic() - started, 0.001)
    completed_images = stats["written"] + stats["reused"]
    print(
        f"[ocr-vqa:jpg] task={task_name} "
        f"rows={stats['rows']:,}/{total_rows:,} "
        f"matched={stats['matched_items']:,} "
        f"written={stats['written']:,} reused={stats['reused']:,} "
        f"no_bytes={stats['matched_without_bytes']:,} "
        f"row_rate={stats['rows'] / elapsed:.1f}/s "
        f"image_rate={completed_images / elapsed:.1f}/s "
        f"elapsed={format_duration(elapsed)}",
        flush=True,
    )


def extract_ocr_parquet(job: tuple[str, int | None]) -> dict[str, Any]:
    path_text, row_group = job
    started = time.monotonic()
    shard_name = Path(path_text).name
    task_name = f"{shard_name}#rg{row_group}" if row_group is not None else shard_name
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "OCR-VQA parquet extraction requires pyarrow. Install it with "
            "`pip install pyarrow`, then rerun this script."
        ) from exc

    stats: Counter = Counter()
    matched_references: set[str] = set()
    parquet_file = pq.ParquetFile(path_text)
    if row_group is None:
        total_rows = parquet_file.metadata.num_rows if parquet_file.metadata else 0
        row_groups = None
    else:
        total_rows = parquet_file.metadata.row_group(row_group).num_rows
        row_groups = [row_group]
    available_columns = set(parquet_file.schema_arrow.names)
    selected_columns = [
        field
        for field in (*OCR_LOOKUP_FIELDS, *OCR_IMAGE_FIELDS)
        if field in available_columns
    ]
    print(
        f"[ocr-vqa:jpg] start task={task_name} rows={total_rows:,} "
        f"columns={','.join(selected_columns)}",
        flush=True,
    )
    for batch in parquet_file.iter_batches(
        batch_size=_OCR_BATCH_SIZE,
        columns=selected_columns,
        row_groups=row_groups,
    ):
        for row in batch.to_pylist():
            stats["rows"] += 1
            keys = ocr_row_lookup_keys(row)
            matches: dict[str, str] = {}
            for key in keys:
                value = _OCR_LOOKUP.get(key)
                if value is None:
                    continue
                items = (value,) if isinstance(value, tuple) else value
                for reference, target in items:
                    matches.setdefault(reference, target)
            if not matches:
                if (
                    _OCR_MEDIA_PROGRESS_EVERY
                    and stats["rows"] % _OCR_MEDIA_PROGRESS_EVERY == 0
                ):
                    print_ocr_task_progress(task_name, total_rows, stats, started)
                continue

            stats["matched_items"] += len(matches)
            data, _ = image_bytes_from_row(row)
            if data is None:
                stats["matched_without_bytes"] += len(matches)
                if (
                    _OCR_MEDIA_PROGRESS_EVERY
                    and stats["rows"] % _OCR_MEDIA_PROGRESS_EVERY == 0
                ):
                    print_ocr_task_progress(task_name, total_rows, stats, started)
                continue
            for reference, target_text in matches.items():
                target = Path(target_text)
                if target.exists() and not _OCR_OVERWRITE:
                    stats["reused"] += 1
                else:
                    write_bytes_atomically(target, data)
                    stats["written"] += 1
                matched_references.add(reference)
            if (
                _OCR_MEDIA_PROGRESS_EVERY
                and stats["rows"] % _OCR_MEDIA_PROGRESS_EVERY == 0
            ):
                print_ocr_task_progress(task_name, total_rows, stats, started)
    return {
        "source_file": path_text,
        "row_group": row_group,
        "elapsed_seconds": time.monotonic() - started,
        "stats": dict(stats),
        "matched_references": sorted(matched_references),
    }


def discover_ocr_parquet_files(root: Path) -> list[Path]:
    direct = set(root.glob("*.parquet"))
    data_dir = root / "data"
    if data_dir.is_dir():
        direct.update(data_dir.glob("*.parquet"))
    files = sorted(path for path in direct if path.is_file())
    if files:
        return files

    # Fallback for uncommon layouts without traversing extracted image trees.
    discovered: list[Path] = []
    skipped_media_dirs = {"images", "image", "imgs", "photos", "__pycache__", ".git"}
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames if name.lower() not in skipped_media_dirs
        ]
        for filename in filenames:
            if filename.lower().endswith(".parquet"):
                discovered.append(Path(current) / filename)
    return sorted(discovered)


def ocr_parquet_jobs(
    parquet_files: Sequence[Path],
    task_mode: str,
) -> list[tuple[str, int | None]]:
    if task_mode == "shard":
        return [(str(path), None) for path in parquet_files]

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "OCR-VQA parquet extraction requires pyarrow. Install it with "
            "`pip install pyarrow`, then rerun this script."
        ) from exc

    jobs: list[tuple[str, int | None]] = []
    for path in parquet_files:
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        if metadata is None or metadata.num_row_groups == 0:
            jobs.append((str(path), None))
            continue
        jobs.extend((str(path), index) for index in range(metadata.num_row_groups))
    return jobs


def ocr_coverage_id(references: set[str]) -> str:
    digest = hashlib.sha256()
    for reference in sorted(references):
        digest.update(reference.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def ocr_job_key(job: tuple[str, int | None]) -> str:
    path_text, row_group = job
    source = str(Path(path_text).resolve())
    task = "all" if row_group is None else f"row-group:{row_group}"
    return f"{source}::{task}"


def ocr_job_source_signature(job: tuple[str, int | None]) -> dict[str, Any]:
    path = Path(job[0]).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def new_ocr_checkpoint(
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "version": OCR_CHECKPOINT_VERSION,
        "task_mode": args.ocr_task_mode,
        "ocr_vqa_root": str(spec.root),
        "parquet_root": str(args.ocr_vqa_parquet_root),
        "coverages": {},
        "tasks": {},
    }


def load_ocr_checkpoint(
    path: Path,
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fresh = new_ocr_checkpoint(spec, args)
    if args.reset_ocr_checkpoint:
        print(
            f"[ocr-vqa:checkpoint] reset requested path={path}",
            flush=True,
        )
        return fresh
    if not path.is_file():
        print(
            f"[ocr-vqa:checkpoint] no saved checkpoint path={path}",
            flush=True,
        )
        return fresh

    try:
        with path.open("r", encoding="utf-8") as stream:
            checkpoint = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[ocr-vqa:checkpoint] could not read checkpoint; rebuilding "
            f"path={path} error={exc}",
            flush=True,
        )
        return fresh

    expected = {
        key: fresh[key]
        for key in ("version", "task_mode", "ocr_vqa_root", "parquet_root")
    }
    if not isinstance(checkpoint, dict) or any(
        checkpoint.get(key) != value for key, value in expected.items()
    ):
        print(
            f"[ocr-vqa:checkpoint] context changed; rebuilding path={path}",
            flush=True,
        )
        return fresh
    if not isinstance(checkpoint.get("coverages"), dict) or not isinstance(
        checkpoint.get("tasks"), dict
    ):
        print(
            f"[ocr-vqa:checkpoint] invalid checkpoint structure; rebuilding path={path}",
            flush=True,
        )
        return fresh

    print(
        f"[ocr-vqa:checkpoint] loaded path={path} "
        f"completed_tasks={len(checkpoint['tasks']):,}",
        flush=True,
    )
    return checkpoint


def ocr_checkpoint_task_status(
    checkpoint: dict[str, Any],
    coverage_sets: dict[str, set[str]],
    job: tuple[str, int | None],
    source_signature: dict[str, Any],
    missing: set[str],
) -> tuple[bool, str]:
    entry = checkpoint["tasks"].get(ocr_job_key(job))
    if not isinstance(entry, dict):
        return False, "not_checkpointed"
    if entry.get("source") != source_signature:
        return False, "source_changed"

    coverage_id = entry.get("coverage_id")
    coverage = coverage_sets.get(coverage_id)
    if coverage is None:
        return False, "invalid_coverage"
    if not missing.issubset(coverage):
        return False, "new_missing_references"

    matched = entry.get("matched_references")
    if not isinstance(matched, list) or any(
        not isinstance(reference, str) for reference in matched
    ):
        return False, "invalid_matches"
    if missing.intersection(matched):
        return False, "missing_previous_output"
    return True, "reused"


def update_ocr_checkpoint(
    checkpoint: dict[str, Any],
    job: tuple[str, int | None],
    source_signature: dict[str, Any],
    coverage_id: str,
    result: dict[str, Any],
) -> None:
    coverage = set(checkpoint["coverages"][coverage_id])
    matched_references = set(result["matched_references"])
    previous = checkpoint["tasks"].get(ocr_job_key(job))
    if isinstance(previous, dict) and previous.get("source") == source_signature:
        previous_coverage = checkpoint["coverages"].get(
            previous.get("coverage_id")
        )
        previous_matches = previous.get("matched_references")
        if isinstance(previous_coverage, list) and all(
            isinstance(reference, str) for reference in previous_coverage
        ):
            coverage.update(previous_coverage)
        if isinstance(previous_matches, list) and all(
            isinstance(reference, str) for reference in previous_matches
        ):
            matched_references.update(previous_matches)

    combined_coverage_id = ocr_coverage_id(coverage)
    checkpoint["coverages"][combined_coverage_id] = sorted(coverage)
    checkpoint["tasks"][ocr_job_key(job)] = {
        "source": source_signature,
        "row_group": job[1],
        "coverage_id": combined_coverage_id,
        "matched_references": sorted(matched_references),
        "stats": result["stats"],
    }


def write_ocr_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    referenced_coverages = {
        entry.get("coverage_id")
        for entry in checkpoint["tasks"].values()
        if isinstance(entry, dict)
    }
    serialized_checkpoint = dict(checkpoint)
    serialized_checkpoint["coverages"] = {
        coverage_id: references
        for coverage_id, references in checkpoint["coverages"].items()
        if coverage_id in referenced_coverages
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                serialized_checkpoint,
                stream,
                ensure_ascii=False,
                sort_keys=True,
            )
            stream.write("\n")
        replace_atomically(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_required_ocr(
    references: set[str],
    spec: DatasetSpec,
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> Counter:
    lookup, missing = ocr_needed_lookup(references, spec, args, progress)
    stats: Counter = Counter(
        {
            "already_available": len(references) - len(missing),
            "requested_missing_before_extract": len(missing),
        }
    )
    print(
        f"[ocr-vqa] lookup complete available={stats['already_available']:,} "
        f"missing={len(missing):,} lookup_keys={len(lookup):,}",
        flush=True,
    )
    if not missing:
        return stats
    if not args.extract_ocr_vqa:
        print("[ocr-vqa] extraction is disabled; using existing images only", flush=True)
        return stats

    print(
        f"[ocr-vqa] discovering parquet shards below {args.ocr_vqa_parquet_root}",
        flush=True,
    )
    parquet_files = discover_ocr_parquet_files(args.ocr_vqa_parquet_root)
    if not parquet_files:
        print(
            f"[ocr-vqa] no parquet shards found below {args.ocr_vqa_parquet_root}",
            flush=True,
        )
        return stats
    for parquet_path in parquet_files:
        print(
            f"[ocr-vqa] shard={parquet_path.name} "
            f"size={format_size(parquet_path.stat().st_size)}",
            flush=True,
        )

    all_jobs = ocr_parquet_jobs(parquet_files, args.ocr_task_mode)
    if not all_jobs:
        print("[ocr-vqa] parquet shards contain no row groups", flush=True)
        return stats
    checkpoint = load_ocr_checkpoint(args.ocr_checkpoint_file, spec, args)
    if args.reset_ocr_checkpoint:
        write_ocr_checkpoint(args.ocr_checkpoint_file, checkpoint)
    coverage_id = ocr_coverage_id(missing)
    checkpoint["coverages"][coverage_id] = sorted(missing)
    coverage_sets = {
        saved_coverage_id: set(saved_references)
        for saved_coverage_id, saved_references in checkpoint["coverages"].items()
        if isinstance(saved_coverage_id, str)
        and isinstance(saved_references, list)
        and all(isinstance(reference, str) for reference in saved_references)
    }
    jobs: list[tuple[str, int | None]] = []
    job_signatures: dict[tuple[str, int | None], dict[str, Any]] = {}
    checkpoint_statuses: Counter = Counter()
    for job in all_jobs:
        signature = ocr_job_source_signature(job)
        job_signatures[job] = signature
        reusable, status = ocr_checkpoint_task_status(
            checkpoint,
            coverage_sets,
            job,
            signature,
            missing,
        )
        checkpoint_statuses[status] += 1
        if not reusable:
            jobs.append(job)

    stats["checkpoint_skipped_tasks"] = checkpoint_statuses["reused"]
    stats["checkpoint_pending_tasks"] = len(jobs)
    stats["checkpoint_total_tasks"] = len(all_jobs)
    print(
        f"[ocr-vqa:checkpoint] total_tasks={len(all_jobs):,} "
        f"skipped={stats['checkpoint_skipped_tasks']:,} pending={len(jobs):,} "
        + " ".join(
            f"{name}={count:,}"
            for name, count in sorted(checkpoint_statuses.items())
            if name != "reused" and count
        ),
        flush=True,
    )
    if not jobs:
        stats["matched_references"] = 0
        stats["unmatched_in_parquet"] = len(missing)
        print(
            f"[ocr-vqa] all parquet tasks reused from checkpoint; "
            f"remaining_missing={len(missing):,}",
            flush=True,
        )
        return stats

    worker_count = min(args.ocr_workers, len(jobs))
    print(
        f"[ocr-vqa] shards={len(parquet_files):,} "
        f"tasks={len(jobs):,}/{len(all_jobs):,} "
        f"task_mode={args.ocr_task_mode} "
        f"missing={len(missing):,} workers={worker_count} "
        f"media_progress_every={args.media_progress_every:,}",
        flush=True,
    )
    matched: set[str] = set()
    extraction_started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_ocr_worker,
        initargs=(
            lookup,
            args.parquet_batch_size,
            args.overwrite_media,
            args.media_progress_every,
        ),
    ) as executor:
        future_jobs = {
            executor.submit(extract_ocr_parquet, job): job
            for job in jobs
        }
        pending = set(future_jobs)
        completed = 0
        while pending:
            done, pending = wait(
                pending,
                timeout=progress.wait_timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                progress.heartbeat(
                    f"completed_tasks={completed:,}/{len(jobs):,} "
                    f"pending={len(pending):,} rows={stats['rows']:,} "
                    f"matched={stats['matched_items']:,} written={stats['written']:,}",
                    force=True,
                )
                continue
            for future in done:
                result = future.result()
                job = future_jobs[future]
                update_ocr_checkpoint(
                    checkpoint,
                    job,
                    job_signatures[job],
                    coverage_id,
                    result,
                )
                write_ocr_checkpoint(args.ocr_checkpoint_file, checkpoint)
                completed += 1
                stats.update(result["stats"])
                matched.update(result["matched_references"])
                extraction_elapsed = max(
                    time.monotonic() - extraction_started,
                    0.001,
                )
                image_rate = len(matched) / extraction_elapsed
                remaining_images = max(0, len(missing) - len(matched))
                eta = (
                    format_duration(remaining_images / image_rate)
                    if image_rate > 0
                    else "unknown"
                )
                print(
                    f"[ocr-vqa] completed_tasks={completed:,}/{len(jobs):,} "
                    f"shard={Path(result['source_file']).name} "
                    f"row_group={result['row_group']} "
                    f"elapsed={format_duration(result['elapsed_seconds'])} "
                    f"rows={stats['rows']:,} matched_items={stats['matched_items']:,} "
                    f"written={stats['written']:,} "
                    f"reused={stats['reused']:,} matched_refs={len(matched):,} "
                    f"image_rate={image_rate:.1f}/s eta={eta}",
                    flush=True,
                )
    stats["matched_references"] = len(matched)
    stats["unmatched_in_parquet"] = len(missing - matched)
    print(
        f"[ocr-vqa] extracted={stats['written']:,} reused={stats['reused']:,} "
        f"unmatched={stats['unmatched_in_parquet']:,}",
        flush=True,
    )
    return stats


def image_files(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in {".git", "__pycache__", ".cache"}
        )
        for filename in sorted(filenames):
            path = Path(current) / filename
            if path.suffix.lower() in IMAGE_SUFFIXES:
                yield path


def suffix_match_score(reference: str, path: Path, root: Path, spec: DatasetSpec) -> int:
    expected = [part.lower() for part in strip_known_prefix(reference, spec)]
    try:
        actual = [part.lower() for part in path.relative_to(root).parts]
    except ValueError:
        actual = [path.name.lower()]
    score = 0
    if expected and actual and expected[-1] == actual[-1]:
        score += 100
    elif expected and actual and Path(expected[-1]).stem == Path(actual[-1]).stem:
        score += 60
    for expected_part, actual_part in zip(reversed(expected[:-1]), reversed(actual[:-1])):
        aliases = {
            "train_images": "train",
            "val_images": "validation",
            "validation_images": "validation",
            "test_images": "test",
        }
        if expected_part == actual_part or aliases.get(expected_part) == actual_part:
            score += 10
        elif expected_part in {"images", "image"}:
            continue
        else:
            break
    return score


def known_image_directories(spec: DatasetSpec) -> list[Path]:
    relative_directories = {
        "coco": ("", "train2017", "images", "images/train2017"),
        "ocr_vqa": ("", "images"),
        "gqa": ("images/train_balanced", "images/val_balanced", "images", ""),
        "textvqa": (
            "",
            "train",
            "validation",
            "test",
            "train_images",
            "val_images",
            "validation_images",
            "test_images",
        ),
        "visualgenome": (
            "",
            "VG_100K",
            "VG_100K_2",
            "images",
            "images/VG_100K",
            "images/VG_100K_2",
        ),
    }
    directories = [
        spec.root if not relative else spec.root / Path(relative)
        for relative in relative_directories.get(spec.name, ("",))
    ]
    return list(dict.fromkeys(directories))


def resolve_indexed_layout(
    references: set[str],
    spec: DatasetSpec,
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Index known image directories once instead of stat-ing every reference."""
    by_name: dict[str, set[str]] = defaultdict(set)
    by_stem: dict[str, set[str]] = defaultdict(set)
    for reference in references:
        name = PurePosixPath(reference.replace("\\", "/")).name.lower()
        by_name[name].add(reference)
        by_stem[Path(name).stem].add(reference)

    candidates: dict[str, list[Path]] = defaultdict(list)
    scanned_files = 0
    for directory in known_image_directories(spec):
        if not directory.is_dir():
            continue
        print(
            f"[resolve:{spec.name}] indexing image directory path={directory}",
            flush=True,
        )
        with os.scandir(directory) as entries:
            for entry in entries:
                if Path(entry.name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if not entry.is_file(follow_symlinks=True):
                    continue
                scanned_files += 1
                name = entry.name.lower()
                matched = by_name.get(name, set()) | by_stem.get(
                    Path(name).stem, set()
                )
                if matched:
                    path = Path(entry.path).absolute()
                    for reference in matched:
                        candidates[reference].append(path)
                if scanned_files % 1000 == 0:
                    progress.heartbeat(
                        f"indexed_image_files={scanned_files:,} "
                        f"matched_references={len(candidates):,}"
                    )
                if args.progress_every and scanned_files % args.progress_every == 0:
                    print(
                        f"[resolve:{spec.name}] indexed_files={scanned_files:,} "
                        f"matched_references={len(candidates):,}",
                        flush=True,
                    )

    resolved: dict[str, Path] = {}
    ambiguous: dict[str, list[str]] = {}
    for reference, matches in candidates.items():
        ranked = sorted(
            (
                (suffix_match_score(reference, path, spec.root, spec), str(path), path)
                for path in matches
            ),
            key=lambda item: (-item[0], item[1]),
        )
        best_score = ranked[0][0]
        best = [item[2] for item in ranked if item[0] == best_score]
        resolved[reference] = best[0]
        if len(best) > 1:
            ambiguous[reference] = [str(path) for path in best]

    print(
        f"[resolve:{spec.name}] directory index complete "
        f"scanned_files={scanned_files:,} "
        f"resolved={len(resolved):,} ambiguous={len(ambiguous):,}",
        flush=True,
    )
    return resolved, ambiguous


def resolve_for_spec(
    references: set[str],
    spec: DatasetSpec,
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> tuple[dict[str, Path], list[str], dict[str, list[str]]]:
    if references and not spec.root.is_dir():
        return {}, sorted(references), {}

    resolved: dict[str, Path] = {}
    ambiguous: dict[str, list[str]] = {}
    if references:
        indexed_resolved, indexed_ambiguous = resolve_indexed_layout(
            references, spec, args, progress
        )
        resolved.update(indexed_resolved)
        ambiguous.update(indexed_ambiguous)

    unresolved: set[str] = set(references) - set(resolved)
    direct_references = sorted(unresolved)
    direct_total = len(direct_references)
    if direct_total > DIRECT_CHECK_LIMIT:
        print(
            f"[resolve:{spec.name}] direct check skipped references={direct_total:,} "
            f"limit={DIRECT_CHECK_LIMIT:,}; using one recursive directory scan",
            flush=True,
        )
    else:
        print(
            f"[resolve:{spec.name}] direct check references={direct_total:,} "
            f"pre_resolved={len(resolved):,} root={spec.root}",
            flush=True,
        )
        for index, reference in enumerate(direct_references, start=1):
            path = existing_direct_path(reference, spec)
            if path is not None:
                resolved[reference] = path
                unresolved.remove(reference)
            if index % 1000 == 0:
                progress.heartbeat(
                    f"direct_checked={index:,}/{direct_total:,} "
                    f"resolved={len(resolved):,} unresolved={len(unresolved):,}"
                )
            if args.progress_every and index % args.progress_every == 0:
                print(
                    f"[resolve:{spec.name}] direct_checked={index:,}/{direct_total:,} "
                    f"resolved={len(resolved):,} unresolved={len(unresolved):,}",
                    flush=True,
                )

        print(
            f"[resolve:{spec.name}] direct check complete resolved={len(resolved):,} "
            f"unresolved={len(unresolved):,}",
            flush=True,
        )

    if not unresolved or not spec.root.is_dir():
        return resolved, sorted(unresolved), ambiguous

    by_name: dict[str, set[str]] = defaultdict(set)
    by_stem: dict[str, set[str]] = defaultdict(set)
    for reference in unresolved:
        name = PurePosixPath(reference.replace("\\", "/")).name.lower()
        by_name[name].add(reference)
        by_stem[Path(name).stem].add(reference)

    candidates: dict[str, list[Path]] = defaultdict(list)
    scanned_files = 0
    candidate_links = 0
    print(
        f"[resolve:{spec.name}] recursive fallback scan start root={spec.root}",
        flush=True,
    )
    for path in image_files(spec.root):
        scanned_files += 1
        name = path.name.lower()
        matched = by_name.get(name, set()) | by_stem.get(path.stem.lower(), set())
        for reference in matched:
            candidates[reference].append(path.absolute())
            candidate_links += 1
        if scanned_files % 1000 == 0:
            progress.heartbeat(
                f"scanned_image_files={scanned_files:,} "
                f"candidate_references={len(candidates):,} candidate_links={candidate_links:,}"
            )
        if args.progress_every and scanned_files % args.progress_every == 0:
            print(
                f"[resolve:{spec.name}] scanned_image_files={scanned_files:,} "
                f"candidate_references={len(candidates):,} "
                f"candidate_links={candidate_links:,}",
                flush=True,
            )

    print(
        f"[resolve:{spec.name}] recursive scan complete "
        f"scanned_image_files={scanned_files:,} "
        f"candidate_references={len(candidates):,}",
        flush=True,
    )

    sorted_unresolved = sorted(unresolved)
    for index, reference in enumerate(sorted_unresolved, start=1):
        matches = candidates.get(reference, [])
        if not matches:
            continue
        ranked = sorted(
            (
                (suffix_match_score(reference, path, spec.root, spec), str(path), path)
                for path in matches
            ),
            key=lambda item: (-item[0], item[1]),
        )
        best_score = ranked[0][0]
        best = [item[2] for item in ranked if item[0] == best_score]
        resolved[reference] = best[0]
        if len(best) > 1:
            ambiguous[reference] = [str(path) for path in best]
        if index % 1000 == 0:
            progress.heartbeat(
                f"ranked={index:,}/{len(sorted_unresolved):,} "
                f"resolved={len(resolved):,} ambiguous={len(ambiguous):,}"
            )
        if args.progress_every and index % args.progress_every == 0:
            print(
                f"[resolve:{spec.name}] ranked={index:,}/{len(sorted_unresolved):,} "
                f"resolved={len(resolved):,} ambiguous={len(ambiguous):,}",
                flush=True,
            )

    missing = sorted(unresolved - set(resolved))
    return resolved, missing, ambiguous


def resolve_spec_job(
    job: tuple[DatasetSpec, set[str], int],
) -> tuple[
    str,
    dict[str, Path],
    list[str],
    dict[str, list[str]],
    float,
]:
    spec, references, progress_every = job
    started = time.monotonic()
    worker_args = argparse.Namespace(progress_every=progress_every)
    worker_progress = ProgressLogger(0)
    paths, missing, ambiguous = resolve_for_spec(
        references,
        spec,
        worker_args,
        worker_progress,
    )
    return (
        spec.name,
        paths,
        missing,
        ambiguous,
        time.monotonic() - started,
    )


def resolve_all_images(
    scan: ScanResult,
    specs: Sequence[DatasetSpec],
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> ResolveResult:
    paths: dict[str, Path] = {}
    owners: dict[str, str] = {}
    missing: list[str] = []
    ambiguous: dict[str, list[str]] = {}

    def merge_spec_result(
        spec: DatasetSpec,
        spec_paths: dict[str, Path],
        spec_missing: list[str],
        spec_ambiguous: dict[str, list[str]],
    ) -> None:
        paths.update(spec_paths)
        owners.update({reference: spec.name for reference in spec_paths})
        missing.extend(spec_missing)
        ambiguous.update(spec_ambiguous)
        print(
            f"[resolve:{spec.name}] requested={len(scan.references[spec.name]):,} "
            f"resolved={len(spec_paths):,} missing={len(spec_missing):,} "
            f"ambiguous={len(spec_ambiguous):,}",
            flush=True,
        )

    spec_by_name = {spec.name: spec for spec in specs}
    if args.resolve_workers == 1:
        for spec in specs:
            references = scan.references[spec.name]
            progress.begin(
                f"resolve:{spec.name}",
                f"references={len(references):,} root={spec.root}",
            )
            if references and not spec.root.is_dir():
                print(f"[{spec.name}] source root is missing: {spec.root}", flush=True)
            spec_paths, spec_missing, spec_ambiguous = resolve_for_spec(
                references, spec, args, progress
            )
            merge_spec_result(spec, spec_paths, spec_missing, spec_ambiguous)
            progress.finish(
                f"resolved={len(spec_paths):,} missing={len(spec_missing):,}"
            )
    else:
        jobs = [
            (spec, scan.references[spec.name], args.progress_every)
            for spec in specs
        ]
        worker_count = min(args.resolve_workers, len(jobs))
        progress.begin(
            "resolve:datasets",
            f"datasets={len(jobs):,} workers={worker_count}",
        )
        for spec, references, _ in jobs:
            if references and not spec.root.is_dir():
                print(f"[{spec.name}] source root is missing: {spec.root}", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            pending = {
                executor.submit(resolve_spec_job, job): job[0].name for job in jobs
            }
            completed = 0
            while pending:
                done, remaining = wait(
                    set(pending),
                    timeout=progress.wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    progress.heartbeat(
                        f"completed_datasets={completed:,}/{len(jobs):,} "
                        f"pending={len(pending):,}",
                        force=True,
                    )
                    continue
                for future in done:
                    pending.pop(future)
                    name, spec_paths, spec_missing, spec_ambiguous, elapsed = (
                        future.result()
                    )
                    spec = spec_by_name[name]
                    merge_spec_result(
                        spec,
                        spec_paths,
                        spec_missing,
                        spec_ambiguous,
                    )
                    progress.timings[f"resolve:{name}"] = elapsed
                    completed += 1
                    print(
                        f"[resolve] completed_datasets={completed:,}/{len(jobs):,} "
                        f"dataset={name} elapsed={format_duration(elapsed)}",
                        flush=True,
                    )
                pending = {future: pending[future] for future in remaining}
        progress.finish(
            f"resolved={len(paths):,} missing={len(set(missing)):,}"
        )

    if scan.unknown_references:
        remaining = set(scan.unknown_references)
        for spec in specs:
            if not remaining:
                break
            progress.begin(
                f"resolve:unknown:{spec.name}",
                f"references={len(remaining):,} root={spec.root}",
            )
            found, _, found_ambiguous = resolve_for_spec(
                remaining, spec, args, progress
            )
            for reference, path in found.items():
                if reference not in paths:
                    paths[reference] = path
                    owners[reference] = spec.name
            remaining -= set(found)
            ambiguous.update(found_ambiguous)
            progress.finish(
                f"found={len(found):,} remaining={len(remaining):,}"
            )
        missing.extend(sorted(remaining))

    return ResolveResult(paths, owners, sorted(set(missing)), ambiguous)


def source_image_paths(resolved: ResolveResult) -> dict[str, str]:
    return {
        reference: str(source_path.absolute())
        for reference, source_path in resolved.paths.items()
    }


def normalize_content(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return IMAGE_TOKEN_RE.sub("<image>", value).strip()


def source_identifier(record: dict[str, Any], fallback: int) -> str:
    for key in ("id", "sample_id", "question_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"source_index:{fallback}"


def convert_record(
    record: dict[str, Any], source_index: int, image_paths: dict[str, str]
) -> tuple[str | None, str | None, str | None]:
    identifier = source_identifier(record, source_index)
    refs = image_references(record)
    missing_refs = [reference for reference in refs if reference not in image_paths]
    if missing_refs:
        return None, "skipped_missing_image", (
            f"{identifier}: unresolved image(s): {missing_refs!r}"
        )

    conversations = record.get("conversations")
    if conversations is None:
        conversations = record.get("messages")
    if not isinstance(conversations, list) or not conversations:
        return None, "invalid_record", (
            f"{identifier}: conversations/messages must be a non-empty list"
        )

    messages: list[dict[str, str]] = []
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            return None, "invalid_record", (
                f"{identifier}: turn {turn_index} is not an object"
            )
        source_role = turn.get("from", turn.get("role"))
        role = ROLE_MAP.get(str(source_role).strip().lower())
        if role is None:
            return None, "invalid_record", (
                f"{identifier}: unsupported role {source_role!r} at turn {turn_index}"
            )
        content = normalize_content(turn.get("value", turn.get("content")))
        if content is None:
            return None, "invalid_record", (
                f"{identifier}: non-string content at turn {turn_index}"
            )
        if not content:
            return None, "invalid_record", (
                f"{identifier}: empty content at turn {turn_index}"
            )
        messages.append({"role": role, "content": content})

    if not any(message["role"] == "user" for message in messages):
        return None, "invalid_record", f"{identifier}: conversation has no user turn"
    if not any(message["role"] == "assistant" for message in messages):
        return None, "invalid_record", (
            f"{identifier}: conversation has no assistant turn"
        )

    token_count = sum(message["content"].count("<image>") for message in messages)
    if refs and token_count == 0:
        first_user = next(
            index for index, message in enumerate(messages) if message["role"] == "user"
        )
        prefix = "<image>" * len(refs)
        messages[first_user]["content"] = f"{prefix}\n{messages[first_user]['content']}"
        token_count = len(refs)
    if token_count != len(refs):
        return None, "invalid_record", (
            f"{identifier}: <image> token count ({token_count}) does not match "
            f"image count ({len(refs)})"
        )

    output: dict[str, Any] = {"messages": messages}
    if refs:
        output["images"] = [image_paths[reference] for reference in refs]
    return json.dumps(output, ensure_ascii=False, separators=(",", ":")), None, None


_CONVERT_IMAGE_PATHS: dict[str, str] = {}


def init_convert_worker(image_paths: dict[str, str]) -> None:
    global _CONVERT_IMAGE_PATHS
    _CONVERT_IMAGE_PATHS = image_paths


def convert_chunk(
    chunk: list[tuple[int, dict[str, Any]]]
) -> tuple[list[str], dict[str, int], list[str]]:
    lines: list[str] = []
    stats: Counter = Counter()
    errors: list[str] = []
    for source_index, record in chunk:
        stats["records_seen"] += 1
        line, error_kind, error = convert_record(
            record, source_index, _CONVERT_IMAGE_PATHS
        )
        if error is not None:
            stats[error_kind or "invalid_record"] += 1
            errors.append(error)
            continue
        lines.append(line)
        stats["written"] += 1
        if image_references(record):
            stats["multimodal_written"] += 1
        else:
            stats["text_only_written"] += 1
    return lines, dict(stats), errors


def iter_record_chunks(
    path: Path, limit: int | None, chunk_size: int
) -> Iterator[list[tuple[int, dict[str, Any]]]]:
    chunk: list[tuple[int, dict[str, Any]]] = []
    for source_index, record in enumerate(iter_source_records(path, limit)):
        chunk.append((source_index, record))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def bounded_parallel_results(
    executor: ProcessPoolExecutor,
    chunks: Iterable[list[tuple[int, dict[str, Any]]]],
    max_pending: int,
    progress: ProgressLogger,
) -> Iterator[tuple[list[str], dict[str, int], list[str]]]:
    iterator = iter(chunks)
    pending: deque[Future] = deque()
    for _ in range(max_pending):
        try:
            pending.append(executor.submit(convert_chunk, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.popleft()
        while True:
            try:
                result = future.result(timeout=progress.wait_timeout)
                break
            except FuturesTimeoutError:
                progress.heartbeat(
                    f"waiting_for_next_chunk pending_chunks={len(pending) + 1:,}",
                    force=True,
                )
        yield result
        try:
            pending.append(executor.submit(convert_chunk, next(iterator)))
        except StopIteration:
            pass


def convert_dataset(
    input_path: Path,
    output_path: Path,
    image_paths: dict[str, str],
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> tuple[Counter, list[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(output_path)
    stats: Counter = Counter()
    errors: list[str] = []
    chunks = iter_record_chunks(input_path, args.limit, args.worker_chunk_size)
    print(
        f"[convert] temporary_output={temporary} final_output={output_path}",
        flush=True,
    )

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            if args.num_workers == 1:
                init_convert_worker(image_paths)
                results = (convert_chunk(chunk) for chunk in chunks)
                executor = None
            else:
                executor = ProcessPoolExecutor(
                    max_workers=args.num_workers,
                    initializer=init_convert_worker,
                    initargs=(image_paths,),
                )
                results = bounded_parallel_results(
                    executor, chunks, args.max_pending_tasks, progress
                )

            try:
                for lines, chunk_stats, chunk_errors in results:
                    for line in lines:
                        output.write(line)
                        output.write("\n")
                    stats.update(chunk_stats)
                    output.flush()
                    temporary_size = temporary.stat().st_size
                    remaining = args.max_error_examples - len(errors)
                    if remaining > 0:
                        errors.extend(chunk_errors[:remaining])
                    progress.heartbeat(
                        f"records={stats['records_seen']:,} written={stats['written']:,} "
                        f"temporary_size={format_size(temporary_size)}"
                    )
                    if (
                        args.progress_every
                        and stats["records_seen"] // args.progress_every
                        > (stats["records_seen"] - chunk_stats["records_seen"])
                        // args.progress_every
                    ):
                        print(
                            f"[convert] records={stats['records_seen']:,} "
                            f"written={stats['written']:,} "
                            f"missing={stats['skipped_missing_image']:,} "
                            f"invalid={stats['invalid_record']:,} "
                            f"temporary_size={format_size(temporary_size)}",
                            flush=True,
                        )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

        if stats["invalid_record"] and args.invalid_record_policy == "error":
            raise ConversionValidationError(
                f"Conversion found {stats['invalid_record']:,} invalid records; "
                "see the report or rerun with --invalid-record-policy skip.",
                stats,
                errors,
            )
        replace_atomically(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        f"[convert] complete written={stats['written']:,} output={output_path}",
        flush=True,
    )
    return stats, errors


_SPLIT_VAL_INDICES: frozenset[int] = frozenset()


def init_split_worker(val_indices: frozenset[int]) -> None:
    global _SPLIT_VAL_INDICES
    _SPLIT_VAL_INDICES = val_indices


def index_jsonl_chunks(
    path: Path,
    chunk_rows: int,
    progress_every: int,
) -> tuple[list[tuple[int, int, int, int]], int]:
    chunks: list[tuple[int, int, int, int]] = []
    total_rows = 0
    chunk_start_offset = 0
    chunk_start_row = 0
    rows_in_chunk = 0
    next_progress = progress_every

    with path.open("rb") as stream:
        while True:
            line_start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            if rows_in_chunk == 0:
                chunk_start_offset = line_start
                chunk_start_row = total_rows
            total_rows += 1
            rows_in_chunk += 1
            if rows_in_chunk >= chunk_rows:
                chunks.append(
                    (
                        chunk_start_offset,
                        stream.tell(),
                        chunk_start_row,
                        rows_in_chunk,
                    )
                )
                rows_in_chunk = 0
            if progress_every and total_rows >= next_progress:
                print(f"[split:index] rows={total_rows:,}", flush=True)
                while next_progress <= total_rows:
                    next_progress += progress_every

        if rows_in_chunk:
            chunks.append(
                (
                    chunk_start_offset,
                    stream.tell(),
                    chunk_start_row,
                    rows_in_chunk,
                )
            )
    return chunks, total_rows


def split_jsonl_chunk(
    job: tuple[int, str, int, int, int, int, str, str],
) -> tuple[int, int, int]:
    (
        chunk_index,
        input_text,
        start_offset,
        end_offset,
        start_row,
        expected_rows,
        train_part_text,
        val_part_text,
    ) = job
    train_rows = 0
    val_rows = 0
    row_index = start_row

    with (
        Path(input_text).open("rb") as source,
        Path(train_part_text).open("wb") as train_stream,
        Path(val_part_text).open("wb") as val_stream,
    ):
        source.seek(start_offset)
        while source.tell() < end_offset:
            line = source.readline()
            if not line:
                break
            if not line.strip():
                continue
            if not line.endswith(b"\n"):
                line += b"\n"
            if row_index in _SPLIT_VAL_INDICES:
                val_stream.write(line)
                val_rows += 1
            else:
                train_stream.write(line)
                train_rows += 1
            row_index += 1

    if train_rows + val_rows != expected_rows:
        raise RuntimeError(
            f"Split chunk {chunk_index} expected {expected_rows:,} rows but read "
            f"{train_rows + val_rows:,}"
        )
    return chunk_index, train_rows, val_rows


def split_clean_jsonl(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    val_ratio: float,
    seed: int,
    progress_every: int,
    workers: int,
    chunk_rows: int,
    progress: ProgressLogger,
) -> dict[str, Any]:
    chunks, total_rows = index_jsonl_chunks(
        input_path,
        chunk_rows,
        progress_every,
    )
    if total_rows == 0:
        raise ValueError(f"Cleaned output JSONL has no rows: {input_path}")
    val_rows_target = max(1, math.floor(total_rows * val_ratio))
    val_indices = frozenset(
        random.Random(seed).sample(range(total_rows), val_rows_target)
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train_temporary = temporary_sibling(train_path)
    val_temporary = temporary_sibling(val_path)
    train_parts = [temporary_sibling(train_path) for _ in chunks]
    val_parts = [temporary_sibling(val_path) for _ in chunks]
    jobs = [
        (
            index,
            str(input_path),
            start_offset,
            end_offset,
            start_row,
            rows,
            str(train_parts[index]),
            str(val_parts[index]),
        )
        for index, (start_offset, end_offset, start_row, rows) in enumerate(chunks)
    ]
    train_rows = 0
    val_rows = 0
    next_progress = progress_every
    worker_count = max(1, min(workers, len(jobs)))
    try:
        if worker_count == 1:
            init_split_worker(val_indices)
            results: Iterable[tuple[int, int, int]] = (
                split_jsonl_chunk(job) for job in jobs
            )
            for _, chunk_train_rows, chunk_val_rows in results:
                train_rows += chunk_train_rows
                val_rows += chunk_val_rows
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=init_split_worker,
                initargs=(val_indices,),
            ) as executor:
                pending = {executor.submit(split_jsonl_chunk, job) for job in jobs}
                completed = 0
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=progress.wait_timeout,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        progress.heartbeat(
                            f"completed_chunks={completed:,}/{len(jobs):,} "
                            f"pending={len(pending):,}",
                            force=True,
                        )
                        continue
                    for future in done:
                        _, chunk_train_rows, chunk_val_rows = future.result()
                        train_rows += chunk_train_rows
                        val_rows += chunk_val_rows
                        completed += 1
                    progress.heartbeat(
                        f"completed_chunks={completed:,}/{len(jobs):,} "
                        f"rows={train_rows + val_rows:,}/{total_rows:,}"
                    )
                    processed_rows = train_rows + val_rows
                    if progress_every and processed_rows >= next_progress:
                        print(
                            f"[split:write] rows={processed_rows:,}/{total_rows:,} "
                            f"train={train_rows:,} val={val_rows:,}",
                            flush=True,
                        )
                        while next_progress <= processed_rows:
                            next_progress += progress_every

        with (
            train_temporary.open("wb") as train_stream,
            val_temporary.open("wb") as val_stream,
        ):
            for train_part, val_part in zip(train_parts, val_parts):
                with train_part.open("rb") as part_stream:
                    shutil.copyfileobj(part_stream, train_stream, length=1024 * 1024)
                with val_part.open("rb") as part_stream:
                    shutil.copyfileobj(part_stream, val_stream, length=1024 * 1024)

        if train_rows + val_rows != total_rows or val_rows != val_rows_target:
            raise RuntimeError(
                f"Split totals are inconsistent: total={total_rows:,} "
                f"train={train_rows:,} val={val_rows:,} "
                f"target_val={val_rows_target:,}"
            )
        replace_atomically(train_temporary, train_path)
        replace_atomically(val_temporary, val_path)
    finally:
        if train_temporary.exists():
            train_temporary.unlink()
        if val_temporary.exists():
            val_temporary.unlink()
        for part in train_parts + val_parts:
            if part.exists():
                part.unlink()

    print(
        f"[split] complete total={total_rows:,} train={train_rows:,} "
        f"val={val_rows:,} val_ratio={val_rows / total_rows:.4f} "
        f"workers={worker_count} chunks={len(jobs):,}",
        flush=True,
    )
    return {
        "input_jsonl": input_path,
        "train_jsonl": train_path,
        "val_jsonl": val_path,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "val_ratio": val_rows / total_rows,
        "seed": seed,
        "workers": worker_count,
        "chunks": len(jobs),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(child) for child in value]
    return value


def write_report(path: Path, report: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(json_ready(report), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        replace_atomically(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run() -> int:
    args = parse_args()
    _, output_path, report_path, train_path, val_path = resolve_cli_paths(args)
    validate_args(args, output_path, report_path, train_path, val_path)
    args.num_workers = args.num_workers or min(16, max(1, os.cpu_count() or 1))
    args.scan_workers = args.scan_workers or min(16, args.num_workers)
    args.resolve_workers = args.resolve_workers or min(5, args.num_workers)
    args.ocr_workers = args.ocr_workers or min(8, args.num_workers)
    args.split_workers = args.split_workers or min(16, args.num_workers)
    args.max_pending_tasks = args.max_pending_tasks or max(1, args.num_workers * 2)
    specs = dataset_specs(args)
    progress = ProgressLogger(args.heartbeat_seconds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[input] {args.input_json}", flush=True)
    print(f"[output:clean] {output_path}", flush=True)
    if args.val_ratio > 0:
        print(f"[output:train] {train_path}", flush=True)
        print(f"[output:val] {val_path}", flush=True)
    print(
        f"[workers] scan={args.scan_workers} resolve={args.resolve_workers} "
        f"ocr={args.ocr_workers} convert={args.num_workers} "
        f"split={args.split_workers}",
        flush=True,
    )
    print(
        "[mode] existing images only: resolve absolute paths in place; no dataset "
        "is moved, duplicated, linked, or extracted unless extraction is explicitly enabled",
        flush=True,
    )
    print(
        f"[config] progress_every={args.progress_every:,} "
        f"media_progress_every={args.media_progress_every:,} "
        f"heartbeat_seconds={args.heartbeat_seconds:g} "
        f"scan_chunk_size={args.scan_chunk_size:,} "
        f"worker_chunk_size={args.worker_chunk_size:,} "
        f"split_chunk_rows={args.split_chunk_rows:,} "
        f"ocr_task_mode={args.ocr_task_mode} "
        f"ocr_output_shard_chars={args.ocr_output_shard_chars} "
        f"extract_coco={args.extract_coco} "
        f"extract_ocr_vqa={args.extract_ocr_vqa}",
        flush=True,
    )
    for spec in specs:
        print(
            f"[root:{spec.name}] exists={spec.root.is_dir()} path={spec.root}",
            flush=True,
        )
    print(
        f"[root:ocr_vqa_parquet] exists={args.ocr_vqa_parquet_root.is_dir()} "
        f"path={args.ocr_vqa_parquet_root}",
        flush=True,
    )
    print(
        f"[checkpoint:ocr_vqa] path={args.ocr_checkpoint_file} "
        f"reset={args.reset_ocr_checkpoint}",
        flush=True,
    )

    report: dict[str, Any] = {
        "input_json": args.input_json,
        "output_jsonl": output_path,
        "train_jsonl": train_path if args.val_ratio > 0 else None,
        "val_jsonl": val_path if args.val_ratio > 0 else None,
        "output_dir": args.output_dir,
        "workers": {
            "scan": args.scan_workers,
            "resolve": args.resolve_workers,
            "ocr": args.ocr_workers,
            "convert": args.num_workers,
            "split": args.split_workers,
        },
        "mode": "pre-extracted-media-resolution-and-conversion",
        "roots": {
            **{spec.name: spec.root for spec in specs},
            "ocr_vqa_parquet": args.ocr_vqa_parquet_root,
        },
        "ocr_checkpoint_file": args.ocr_checkpoint_file,
        "status": "running",
    }

    progress.begin("scan", f"input={args.input_json}")
    scan = scan_source(args.input_json, specs, args, progress)
    progress.finish(f"records={scan.stats['records']:,}")
    unique_counts = {
        name: len(references) for name, references in scan.references.items()
    }
    print(
        "[scan] unique_by_dataset "
        + " ".join(f"{name}={count:,}" for name, count in unique_counts.items())
        + f" unknown={len(scan.unknown_references):,}",
        flush=True,
    )
    report["scan"] = {
        "stats": scan.stats,
        "unique_by_dataset": unique_counts,
        "unknown_unique": len(scan.unknown_references),
    }

    spec_by_name = {spec.name: spec for spec in specs}
    if args.extract_coco:
        progress.begin(
            "coco_extract",
            f"references={len(scan.references['coco']):,} "
            f"root={spec_by_name['coco'].root}",
        )
        coco_stats = extract_required_coco(
            scan.references["coco"], spec_by_name["coco"], args, progress
        )
        progress.finish(
            f"available={coco_stats['already_available']:,} "
            f"written={coco_stats['written']:,} reused={coco_stats['reused']:,}"
        )
    else:
        coco_stats = Counter({"extraction_enabled": 0})
        print("[coco] extraction skipped; resolving existing files directly", flush=True)

    if args.extract_ocr_vqa:
        progress.begin(
            "ocr_vqa_extract",
            f"references={len(scan.references['ocr_vqa']):,} "
            f"source={args.ocr_vqa_parquet_root} "
            f"target={spec_by_name['ocr_vqa'].root}",
        )
        ocr_stats = extract_required_ocr(
            scan.references["ocr_vqa"], spec_by_name["ocr_vqa"], args, progress
        )
        progress.finish(
            f"available={ocr_stats['already_available']:,} "
            f"written={ocr_stats['written']:,} reused={ocr_stats['reused']:,}"
        )
    else:
        ocr_stats = Counter({"extraction_enabled": 0})
        print(
            "[ocr-vqa] extraction skipped; resolving existing files directly",
            flush=True,
        )
    report["extraction"] = {"coco": coco_stats, "ocr_vqa": ocr_stats}

    resolved = resolve_all_images(scan, specs, args, progress)
    missing_set = set(resolved.missing)
    missing_by_dataset = {
        name: len(references & missing_set)
        for name, references in scan.references.items()
    }
    missing_by_dataset["unknown"] = len(scan.unknown_references & missing_set)
    print(
        "[resolution] missing_by_dataset "
        + " ".join(
            f"{name}={count:,}" for name, count in missing_by_dataset.items()
        ),
        flush=True,
    )
    report["resolution"] = {
        "resolved": len(resolved.paths),
        "missing": len(resolved.missing),
        "missing_by_dataset": missing_by_dataset,
        "ambiguous": len(resolved.ambiguous),
        "missing_examples": resolved.missing[: args.max_error_examples],
        "ambiguous_examples": dict(
            list(resolved.ambiguous.items())[: args.max_error_examples]
        ),
    }

    if resolved.missing and args.missing_image_policy == "error":
        report["status"] = "failed_missing_images"
        report["stage_seconds"] = progress.timings
        write_report(report_path, report, args.overwrite)
        missing_summary = ", ".join(
            f"{name}={count:,}"
            for name, count in missing_by_dataset.items()
            if count
        )
        raise RuntimeError(
            f"Could not resolve {len(resolved.missing):,} unique images. "
            f"Missing by dataset: {missing_summary}. "
            f"See {report_path}. Check the source archives/parquet schema and directory "
            "roots; use --missing-image-policy skip only if dropping affected samples "
            "is acceptable."
        )

    progress.begin("source_paths", "writing existing absolute source paths")
    converted_image_paths = source_image_paths(resolved)
    progress.finish(f"paths={len(converted_image_paths):,}")

    if args.check_only:
        report["status"] = "check_complete"
        report["stage_seconds"] = progress.timings
        write_report(report_path, report, args.overwrite)
        print(f"[done] check complete report={report_path}", flush=True)
        return 0

    split_stats: dict[str, Any] | None = None
    try:
        progress.begin(
            "convert",
            f"records={scan.stats['records']:,} output={output_path}",
        )
        conversion_stats, conversion_errors = convert_dataset(
            args.input_json, output_path, converted_image_paths, args, progress
        )
        progress.finish(
            f"written={conversion_stats['written']:,} output={output_path}"
        )
        if args.val_ratio > 0:
            progress.begin(
                "split",
                f"clean={output_path} train={train_path} val={val_path} val_ratio={args.val_ratio:g}",
            )
            split_stats = split_clean_jsonl(
                output_path,
                train_path,
                val_path,
                args.val_ratio,
                args.split_seed,
                args.progress_every,
                args.split_workers,
                args.split_chunk_rows,
                progress,
            )
            progress.finish(
                f"train={split_stats['train_rows']:,} val={split_stats['val_rows']:,}"
            )
    except ConversionValidationError as exc:
        report["status"] = "failed_conversion"
        report["conversion"] = {
            "stats": exc.stats,
            "error_examples": exc.examples,
        }
        report["conversion_error"] = str(exc)
        report["stage_seconds"] = progress.timings
        write_report(report_path, report, args.overwrite)
        raise
    except Exception as exc:
        report["status"] = "failed_conversion"
        report["conversion_error"] = str(exc)
        report["stage_seconds"] = progress.timings
        write_report(report_path, report, args.overwrite)
        raise

    report["conversion"] = {
        "stats": conversion_stats,
        "error_examples": conversion_errors,
    }
    report["dropped"] = {
        "missing_image_rows": conversion_stats["skipped_missing_image"],
        "invalid_record_rows": conversion_stats["invalid_record"],
        "error_examples": conversion_errors,
    }
    if split_stats is not None:
        report["split"] = split_stats
    report["status"] = "complete"
    report["stage_seconds"] = progress.timings
    write_report(report_path, report, args.overwrite)
    print(f"[done] report={report_path}", flush=True)
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
