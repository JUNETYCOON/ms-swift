#!/usr/bin/env python3
"""Convert the official VLM-R1 full-SFT data to ms-swift JSONL.

Default source:
    /mnt/luojunkun/stage1/dataset/vlm-r1/sft_related/mllm_rec_json.json

Default output:
    /mnt/luojunkun/stage1/dataset_ms-swift/vlm-r1/
    |-- vlm_r1_sft_grounding_msswift.jsonl
    |-- conversion_report.json
    `-- images/

Only images referenced by the selected annotation are extracted from image ZIP
archives. The official REC assistant JSON, for example ``bbox_2d`` and
``label``, is converted to ms-swift's ``<ref-object>``, ``<bbox>``, and
``objects`` grounding representation. Ordinary QA and multi-image records are
converted to standard ``messages`` plus ``images`` records.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
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


DEFAULT_INPUT_DIR = Path("/mnt/luojunkun/stage1/dataset/vlm-r1")
DEFAULT_OUTPUT_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/vlm-r1")
DEFAULT_ANNOTATION = Path("sft_related/mllm_rec_json.json")
DEFAULT_OUTPUT_FILE = "vlm_r1_sft_grounding_msswift.jsonl"
DEFAULT_REPORT_FILE = "conversion_report.json"
DEFAULT_SYSTEM = "You are a helpful assistant."

READ_CHUNK_SIZE = 1024 * 1024
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
ANNOTATION_ARCHIVE_PREFIXES = ("rec_jsons_",)
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}
IMAGE_TOKEN_RE = re.compile(r"<\s*image\s*>", flags=re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
QUERY_AFTER_COLON_RE = re.compile(r":\s*(.+?)\s*[.?!]?\s*$", flags=re.DOTALL)
PATH_MARKERS = {
    "train2014",
    "test",
    "images",
    "image",
    "flickr30k-images",
    "dont_specify",
    "gui_multi-image",
}


@dataclass
class ScanResult:
    stats: Counter
    image_references: set[str]


@dataclass(frozen=True)
class ArchiveImage:
    archive: Path
    member: str
    target: Path


@dataclass
class MediaResult:
    paths: dict[str, Path]
    missing: list[str]
    ambiguous: dict[str, list[str]]
    stats: Counter


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
            f"[heartbeat] stage={self.stage} "
            f"stage_elapsed={format_duration(now - self.stage_started)} "
            f"total_elapsed={format_duration(now - self.run_started)}{suffix}",
            flush=True,
        )

    def finish(self, detail: str = "") -> None:
        elapsed = time.monotonic() - self.stage_started
        self.timings[self.stage] = self.timings.get(self.stage, 0.0) + elapsed
        suffix = f" {detail}" if detail else ""
        print(
            f"[stage:{self.stage}] complete elapsed={format_duration(elapsed)}{suffix}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VLM-R1 SFT/REC annotations to ms-swift multimodal JSONL."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION,
        help="JSON/JSONL annotation. Relative paths are resolved below --input-dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--report-file", default=DEFAULT_REPORT_FILE)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Extracted image directory. Default: <output-dir>/images.",
    )
    parser.add_argument(
        "--image-archives",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Image ZIP archives. Default: infer relevant non-annotation ZIPs from "
            "image paths, falling back to every ZIP below --input-dir."
        ),
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Extraction/conversion processes; 0 uses up to 16 CPUs.",
    )
    parser.add_argument("--worker-chunk-size", type=int, default=512)
    parser.add_argument("--max-pending-tasks", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-error-examples", type=int, default=30)
    parser.add_argument(
        "--missing-image-policy",
        choices=("error", "skip"),
        default="error",
    )
    parser.add_argument(
        "--ambiguous-image-policy",
        choices=("error", "first"),
        default="error",
    )
    parser.add_argument(
        "--invalid-record-policy",
        choices=("error", "skip"),
        default="error",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def resolve_output_path(value: str, output_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.absolute() if path.is_absolute() else (output_dir / path).absolute()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    annotation = args.annotation_file.expanduser()
    if not annotation.is_absolute():
        annotation = args.input_dir / annotation
    annotation = annotation.resolve()
    output_path = resolve_output_path(args.output_file, args.output_dir)
    report_path = resolve_output_path(args.report_file, args.output_dir)
    image_dir = (args.image_dir or (args.output_dir / "images")).expanduser().resolve()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not annotation.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation}")
    with annotation.open("rb") as stream:
        prefix = stream.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"Annotation is only a Git LFS pointer: {annotation}. Complete the Hugging Face download first."
        )
    if annotation.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError("--annotation-file must be .json or .jsonl")
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError("--output-file must end in .jsonl")
    if output_path == report_path:
        raise ValueError("Output and report paths must differ")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be 0 or greater")
    if args.worker_chunk_size <= 0:
        raise ValueError("--worker-chunk-size must be greater than zero")
    if args.max_pending_tasks < 0:
        raise ValueError("--max-pending-tasks must be 0 or greater")
    if args.progress_every < 0 or args.heartbeat_seconds < 0:
        raise ValueError("Progress values must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if args.max_error_examples <= 0:
        raise ValueError("--max-error-examples must be greater than zero")
    if output_path.exists() and not args.overwrite and not args.check_only:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_path}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {report_path}")
    return annotation, output_path, report_path, image_dir


def normalized_reference(value: Any) -> str:
    return value.strip().replace("\\", "/") if isinstance(value, str) else ""


def image_references(record: dict[str, Any]) -> list[str]:
    value = record.get("images")
    if value is None:
        value = record.get("image")
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [reference for item in values if (reference := normalized_reference(item))]


def iter_json_array(stream: TextIO) -> Iterator[dict[str, Any]]:
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
            if position >= len(buffer) or buffer[position] != "[":
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
            raise ValueError("Unexpected end of JSON array")
        while True:
            try:
                value, end = decoder.raw_decode(buffer, position)
                position = end
                break
            except json.JSONDecodeError as exc:
                if not read_more():
                    raise ValueError(f"Invalid or truncated JSON: {exc}") from None
        if not isinstance(value, dict):
            raise ValueError("Every top-level JSON item must be an object")
        yield value


def iter_jsonl(stream: TextIO) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(stream, start=1):
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        yield value


def iter_source_records(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        first = ""
        while not first:
            character = stream.read(1)
            if not character:
                raise ValueError(f"Annotation is empty: {path}")
            if not character.isspace():
                first = character
        stream.seek(0)
        iterator = iter_json_array(stream) if first == "[" else iter_jsonl(stream)
        for index, record in enumerate(iterator):
            if limit is not None and index >= limit:
                break
            yield record


def scan_source(path: Path, args: argparse.Namespace, progress: ProgressLogger) -> ScanResult:
    stats: Counter = Counter()
    references: set[str] = set()
    for record in iter_source_records(path, args.limit):
        stats["records"] += 1
        refs = image_references(record)
        references.update(refs)
        stats["image_references"] += len(refs)
        stats["multimodal_records" if refs else "text_only_records"] += 1
        if args.progress_every and stats["records"] % args.progress_every == 0:
            print(
                f"[scan] records={stats['records']:,} "
                f"image_refs={stats['image_references']:,} unique_images={len(references):,}",
                flush=True,
            )
        if stats["records"] % 1000 == 0:
            progress.heartbeat(
                f"records={stats['records']:,} unique_images={len(references):,}"
            )
    stats["unique_images"] = len(references)
    print(
        f"[scan] complete records={stats['records']:,} "
        f"multimodal={stats['multimodal_records']:,} "
        f"text_only={stats['text_only_records']:,} unique_images={len(references):,}",
        flush=True,
    )
    return ScanResult(stats=stats, image_references=references)


def path_parts(value: str) -> tuple[str, ...]:
    return tuple(part for part in PurePosixPath(value.replace("\\", "/")).parts if part not in {"", "/", "."})


def matching_keys(value: str) -> set[str]:
    parts = path_parts(value)
    if not parts:
        return set()
    lowered = [part.lower() for part in parts]
    keys = {lowered[-1]}
    for count in (2, 3):
        if len(lowered) >= count:
            keys.add("/".join(lowered[-count:]))
    for index, part in enumerate(lowered):
        if part in PATH_MARKERS:
            keys.add("/".join(lowered[index:]))
    return keys


def suffix_score(reference: str, candidate: str) -> int:
    expected = [part.lower() for part in path_parts(reference)]
    actual = [part.lower() for part in path_parts(candidate)]
    score = 0
    for left, right in zip(reversed(expected), reversed(actual)):
        if left != right:
            break
        score += 100
    return score


def image_files(root: Path, skipped_roots: set[Path]) -> Iterator[Path]:
    if not root.is_dir():
        return
    skipped_names = {".cache", ".git", "__pycache__"}
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current).resolve()
        dirnames[:] = [
            name
            for name in dirnames
            if name not in skipped_names
            and (current_path / name).resolve() not in skipped_roots
        ]
        for filename in filenames:
            path = Path(current) / filename
            if path.suffix.lower() in IMAGE_SUFFIXES:
                yield path.absolute()


def archive_matches_reference(archive: Path, reference: str) -> bool:
    stem = archive.stem.lower()
    return stem in {part.lower() for part in path_parts(reference)}


def discover_archives(
    input_dir: Path,
    configured: list[Path] | None,
    references: set[str],
) -> list[Path]:
    if configured is not None:
        paths = [path if path.is_absolute() else input_dir / path for path in configured]
    else:
        paths = [
            path
            for path in input_dir.glob("*.zip")
            if not path.name.lower().startswith(ANNOTATION_ARCHIVE_PREFIXES)
        ]
        matched = {
            path: {
                reference
                for reference in references
                if archive_matches_reference(path, reference)
            }
            for path in paths
        }
        covered = set().union(*matched.values()) if matched else set()
        if references and covered == references:
            paths = [path for path in paths if matched[path]]
        else:
            paths.sort(key=lambda path: (not bool(matched[path]), path.name.lower()))
    resolved = sorted({path.expanduser().resolve() for path in paths})
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("Image archive missing: " + ", ".join(map(str, missing)))
    lfs_pointers = []
    for path in resolved:
        with path.open("rb") as stream:
            if stream.read(64).startswith(b"version https://git-lfs.github.com/spec"):
                lfs_pointers.append(path)
    if lfs_pointers:
        raise RuntimeError(
            "Image archives are only Git LFS pointers; complete the Hugging Face "
            "download first: " + ", ".join(map(str, lfs_pointers))
        )
    return resolved


def safe_archive_target(image_dir: Path, archive: Path, member: str) -> Path:
    parts = [part for part in PurePosixPath(member.replace("\\", "/")).parts if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe ZIP member: {archive}!/{member}")
    target = image_dir.joinpath(*parts).resolve()
    try:
        target.relative_to(image_dir)
    except ValueError:
        raise ValueError(f"Unsafe extraction target: {target}") from None
    return target


def temporary_sibling(path: Path) -> Path:
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


def extract_archive_job(job: tuple[str, list[tuple[str, str]], bool]) -> dict[str, int]:
    archive_text, members, overwrite = job
    stats: Counter = Counter()
    with zipfile.ZipFile(archive_text) as archive:
        for member, target_text in members:
            target = Path(target_text)
            if target.is_file() and not overwrite:
                stats["reused"] += 1
                continue
            temporary = temporary_sibling(target)
            try:
                with archive.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                replace_atomically(temporary, target)
                stats["written"] += 1
            finally:
                if temporary.exists():
                    temporary.unlink()
    return dict(stats)


def partition(values: Sequence[Any], count: int) -> list[list[Any]]:
    if not values:
        return []
    count = max(1, min(count, len(values)))
    size = (len(values) + count - 1) // count
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def resolve_media(
    references: set[str],
    input_dir: Path,
    image_dir: Path,
    archives: list[Path],
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> MediaResult:
    stats: Counter = Counter()
    best: dict[str, tuple[int, Path, str]] = {}
    ambiguous: dict[str, set[str]] = defaultdict(set)

    for reference in references:
        raw = Path(reference)
        candidates = [raw] if raw.is_absolute() else [input_dir / raw, image_dir / raw]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                best[reference] = (100000, candidate.absolute(), str(candidate))
                break

    unresolved = references - set(best)
    if unresolved:
        wanted: dict[str, set[str]] = defaultdict(set)
        for reference in unresolved:
            for key in matching_keys(reference):
                wanted[key].add(reference)
        skipped_roots = {image_dir} if image_dir.is_relative_to(input_dir) else set()
        scanned = 0
        for root in dict.fromkeys((image_dir, input_dir)):
            for path in image_files(root, skipped_roots if root == input_dir else set()):
                scanned += 1
                refs: set[str] = set()
                for key in matching_keys(str(path.relative_to(root))):
                    refs.update(wanted.get(key, ()))
                for reference in refs:
                    score = suffix_score(reference, str(path))
                    current = best.get(reference)
                    if current is None or score > current[0]:
                        best[reference] = (score, path, str(path))
                        ambiguous.pop(reference, None)
                    elif score == current[0] and path != current[1]:
                        ambiguous[reference].update((current[2], str(path)))
                if scanned % 1000 == 0:
                    progress.heartbeat(
                        f"existing_files_scanned={scanned:,} resolved={len(best):,}/{len(references):,}"
                    )
                if args.progress_every and scanned % args.progress_every == 0:
                    print(
                        f"[media] existing_files_scanned={scanned:,} "
                        f"resolved={len(best):,}/{len(references):,}",
                        flush=True,
                    )
        stats["existing_files_scanned"] = scanned

    unresolved = references - set(best)
    archive_choices: dict[str, tuple[int, ArchiveImage]] = {}
    if unresolved:
        wanted = defaultdict(set)
        for reference in unresolved:
            for key in matching_keys(reference):
                wanted[key].add(reference)
        for archive_index, archive_path in enumerate(archives, start=1):
            print(
                f"[media] indexing_archive={archive_index}/{len(archives)} "
                f"path={archive_path} size={format_size(archive_path.stat().st_size)}",
                flush=True,
            )
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                for member_index, info in enumerate(infos, start=1):
                    if info.is_dir() or PurePosixPath(info.filename).suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    refs: set[str] = set()
                    for key in matching_keys(info.filename):
                        refs.update(wanted.get(key, ()))
                    for reference in refs:
                        score = suffix_score(reference, info.filename)
                        target = safe_archive_target(image_dir, archive_path, info.filename)
                        choice = ArchiveImage(archive_path, info.filename, target)
                        current = archive_choices.get(reference)
                        if current is None or score > current[0]:
                            archive_choices[reference] = (score, choice)
                            ambiguous.pop(reference, None)
                        elif score == current[0] and choice != current[1]:
                            ambiguous[reference].update(
                                (
                                    f"{current[1].archive}!/{current[1].member}",
                                    f"{archive_path}!/{info.filename}",
                                )
                            )
                    if member_index % 1000 == 0:
                        progress.heartbeat(
                            f"archive={archive_index}/{len(archives)} "
                            f"members={member_index:,}/{len(infos):,} "
                            f"matched={len(archive_choices):,}/{len(unresolved):,}"
                        )
                    if args.progress_every and member_index % args.progress_every == 0:
                        print(
                            f"[media] archive={archive_index}/{len(archives)} "
                            f"members={member_index:,}/{len(infos):,} "
                            f"matched={len(archive_choices):,}/{len(unresolved):,}",
                            flush=True,
                        )

    selected: dict[tuple[Path, str, Path], ArchiveImage] = {}
    for reference, (_, choice) in archive_choices.items():
        if reference in ambiguous and args.ambiguous_image_policy == "error":
            continue
        selected[(choice.archive, choice.member, choice.target)] = choice
    by_archive: dict[Path, list[ArchiveImage]] = defaultdict(list)
    for choice in selected.values():
        by_archive[choice.archive].append(choice)

    jobs: list[tuple[str, list[tuple[str, str]], bool]] = []
    for archive_path, choices in by_archive.items():
        choices.sort(key=lambda item: item.member)
        for group in partition(choices, args.num_workers):
            jobs.append(
                (
                    str(archive_path),
                    [(choice.member, str(choice.target)) for choice in group],
                    args.overwrite_images,
                )
            )

    if jobs:
        print(
            f"[media] extracting unique_files={len(selected):,} jobs={len(jobs):,} "
            f"workers={min(args.num_workers, len(jobs))}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=min(args.num_workers, len(jobs))) as executor:
            pending = {executor.submit(extract_archive_job, job) for job in jobs}
            completed = 0
            while pending:
                done, pending = wait(
                    pending,
                    timeout=progress.wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    progress.heartbeat(
                        f"extract_jobs={completed:,}/{len(jobs):,} pending={len(pending):,}",
                        force=True,
                    )
                    continue
                for future in done:
                    stats.update(future.result())
                    completed += 1
                    print(
                        f"[media] extract_jobs={completed:,}/{len(jobs):,} "
                        f"written={stats['written']:,} reused={stats['reused']:,}",
                        flush=True,
                    )

    for reference, (_, choice) in archive_choices.items():
        if reference in ambiguous and args.ambiguous_image_policy == "error":
            continue
        if choice.target.is_file():
            best[reference] = (1000, choice.target.absolute(), f"{choice.archive}!/{choice.member}")

    paths = {reference: value[1] for reference, value in best.items()}
    ambiguous_output = {
        reference: sorted(values) for reference, values in ambiguous.items() if values
    }
    missing = sorted(references - set(paths) - set(ambiguous_output))
    stats["requested_images"] = len(references)
    stats["resolved_images"] = len(paths)
    stats["missing_images"] = len(missing)
    stats["ambiguous_images"] = len(ambiguous_output)
    return MediaResult(paths, missing, ambiguous_output, stats)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.strip().split())
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def source_messages(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    value = record.get("messages")
    if value is None:
        value = record.get("conversations")
    return value if isinstance(value, list) else None


def assistant_value(record: dict[str, Any]) -> Any:
    messages = source_messages(record) or []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", message.get("from", ""))).lower()
        if ROLE_MAP.get(role) == "assistant":
            return message.get("content", message.get("value"))
    return record.get("solution")


def user_value(record: dict[str, Any]) -> str:
    messages = source_messages(record) or []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", message.get("from", ""))).lower()
        if ROLE_MAP.get(role) == "user":
            value = message.get("content", message.get("value"))
            return value if isinstance(value, str) else clean_text(value)
    return clean_text(record.get("problem"))


def numeric_box(value: Any) -> list[float | int] | None:
    if not isinstance(value, (list, tuple)) or len(value) not in {2, 4}:
        return None
    output: list[float | int] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            return None
        coordinate_float = float(coordinate)
        if not math.isfinite(coordinate_float):
            return None
        output.append(coordinate if isinstance(coordinate, int) else coordinate_float)
    if len(output) == 4:
        output[0], output[2] = sorted((output[0], output[2]))
        output[1], output[3] = sorted((output[1], output[3]))
    return output


def json_candidates(text: str) -> Iterator[Any]:
    answer = ANSWER_RE.search(text)
    if answer:
        text = answer.group(1)
    fenced = JSON_FENCE_RE.findall(text)
    candidates = fenced or [text]
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            yield json.loads(candidate)
            continue
        except json.JSONDecodeError:
            pass
        start = candidate.find("[")
        end = candidate.rfind("]")
        if 0 <= start < end:
            try:
                yield json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue


def parsed_grounding(value: Any) -> tuple[list[list[float | int]], list[str]] | None:
    payloads: list[Any]
    if isinstance(value, str):
        payloads = list(json_candidates(value))
    else:
        payloads = [value]
    for payload in payloads:
        box = numeric_box(payload)
        if box is not None:
            return [box], []
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list) or not payload:
            continue
        boxes: list[list[float | int]] = []
        labels: list[str] = []
        valid = True
        for item in payload:
            if isinstance(item, dict):
                raw_box = item.get("bbox_2d", item.get("bbox", item.get("box")))
                label = clean_text(item.get("label", item.get("ref", item.get("name"))))
            else:
                raw_box = item
                label = ""
            parsed = numeric_box(raw_box)
            if parsed is None:
                valid = False
                break
            boxes.append(parsed)
            labels.append(label)
        if valid and boxes:
            return boxes, labels
    return None


def query_phrase(record: dict[str, Any], user: str) -> str:
    phrase = clean_text(record.get("normal_caption"))
    if phrase:
        return phrase
    cleaned = IMAGE_TOKEN_RE.sub("", user).strip()
    match = QUERY_AFTER_COLON_RE.search(cleaned)
    if match:
        return clean_text(match.group(1)).rstrip(" .?!")
    return clean_text(cleaned).rstrip(" .?!")


def normalized_existing_objects(
    record: dict[str, Any], image_count: int
) -> tuple[list[list[float | int]], list[str], str, list[int]] | None:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return None
    raw_boxes = objects.get("bbox")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        return None
    boxes = [numeric_box(box) for box in raw_boxes]
    if any(box is None for box in boxes):
        raise ValueError("objects.bbox contains an invalid box")
    raw_refs = objects.get("ref", [])
    if not isinstance(raw_refs, list):
        raise ValueError("objects.ref must be a list")
    refs = [clean_text(value) for value in raw_refs]
    bbox_type = str(objects.get("bbox_type") or "real")
    if bbox_type not in {"real", "norm1"}:
        raise ValueError("objects.bbox_type must be real or norm1")
    image_ids = objects.get("image_id")
    if image_ids is None:
        image_ids = [0] * len(boxes)
    if (
        not isinstance(image_ids, list)
        or len(image_ids) != len(boxes)
        or any(not isinstance(value, int) or isinstance(value, bool) for value in image_ids)
        or any(value < 0 or value >= image_count for value in image_ids)
    ):
        raise ValueError("objects.image_id is invalid for the image count")
    return [box for box in boxes if box is not None], refs, bbox_type, image_ids


def grounding_payload(
    record: dict[str, Any], image_count: int
) -> tuple[list[list[float | int]], list[str], str, list[int]] | None:
    existing = normalized_existing_objects(record, image_count)
    if existing is not None:
        return existing
    parsed = parsed_grounding(assistant_value(record))
    if parsed is None:
        parsed = parsed_grounding(record.get("solution"))
    if parsed is None:
        return None
    boxes, labels = parsed
    user = user_value(record)
    fallback = query_phrase(record, user)
    unique_labels = list(dict.fromkeys(label for label in labels if label))
    refs = unique_labels or [fallback or "object"]
    return boxes, refs, "real", [0] * len(boxes)


def canonical_grounding_record(
    record: dict[str, Any], images: list[str], system: str
) -> dict[str, Any] | None:
    if not images:
        if record.get("objects") is not None:
            raise ValueError("grounding objects require at least one image")
        return None
    payload = grounding_payload(record, len(images))
    if payload is None:
        return None
    boxes, refs, bbox_type, image_ids = payload
    image_tokens = "<image>" * len(images)
    if len(refs) == 1:
        user_content = f"{image_tokens}Locate <ref-object>."
    else:
        ref_tokens = " ".join("<ref-object>" for _ in refs)
        user_content = f"{image_tokens}Locate these objects: {ref_tokens}."
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(
        (
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "<bbox>" * len(boxes)},
        )
    )
    objects: dict[str, Any] = {
        "ref": refs,
        "bbox": boxes,
        "bbox_type": bbox_type,
    }
    if bbox_type == "real":
        objects["image_id"] = image_ids
    return {"messages": messages, "images": images, "objects": objects}


def standard_record(record: dict[str, Any], images: list[str]) -> dict[str, Any]:
    source = source_messages(record)
    if not source:
        problem = clean_text(record.get("problem"))
        answer = clean_text(record.get("answer", record.get("solution")))
        if not problem or not answer:
            raise ValueError("record has neither messages nor problem/answer")
        source = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": answer},
        ]
    messages: list[dict[str, str]] = []
    for index, message in enumerate(source):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} is not an object")
        raw_role = str(message.get("role", message.get("from", ""))).lower()
        role = ROLE_MAP.get(raw_role)
        if role is None:
            raise ValueError(f"unsupported role {raw_role!r}")
        value = message.get("content", message.get("value"))
        if isinstance(value, str):
            content = IMAGE_TOKEN_RE.sub("<image>", value).strip()
        elif value is not None:
            content = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            content = ""
        if not content:
            raise ValueError(f"message {index} has empty content")
        messages.append({"role": role, "content": content})
    token_count = sum(message["content"].count("<image>") for message in messages)
    if images and token_count == 0:
        user_index = next(
            (index for index, message in enumerate(messages) if message["role"] == "user"),
            None,
        )
        if user_index is None:
            raise ValueError("multimodal record has no user message")
        messages[user_index]["content"] = (
            "<image>" * len(images) + messages[user_index]["content"]
        )
        token_count = len(images)
    if token_count != len(images):
        raise ValueError(
            f"<image> token count {token_count} does not match image count {len(images)}"
        )
    return {"messages": messages, **({"images": images} if images else {})}


def validate_output_record(record: dict[str, Any]) -> None:
    messages = record["messages"]
    content = "\n".join(message["content"] for message in messages)
    images = record.get("images", [])
    if content.count("<image>") != len(images):
        raise ValueError("generated image token count mismatch")
    objects = record.get("objects")
    if objects is None:
        return
    if content.count("<ref-object>") != len(objects.get("ref", [])):
        raise ValueError("generated ref token count mismatch")
    if content.count("<bbox>") != len(objects.get("bbox", [])):
        raise ValueError("generated bbox token count mismatch")
    image_ids = objects.get("image_id", [])
    if objects.get("bbox_type") == "real" and len(image_ids) != len(objects["bbox"]):
        raise ValueError("generated image_id count mismatch")


_WORKER_IMAGE_PATHS: dict[str, str] = {}
_WORKER_SYSTEM = DEFAULT_SYSTEM


def init_convert_worker(image_paths: dict[str, str], system: str) -> None:
    global _WORKER_IMAGE_PATHS, _WORKER_SYSTEM
    _WORKER_IMAGE_PATHS = image_paths
    _WORKER_SYSTEM = system


def source_identifier(record: dict[str, Any], index: int) -> str:
    for key in ("id", "question_id", "image_id"):
        if key in record:
            return str(record[key])
    return f"source_index:{index}"


def convert_source_record(
    record: dict[str, Any], index: int
) -> tuple[str | None, str | None, str | None]:
    identifier = source_identifier(record, index)
    refs = image_references(record)
    missing = [reference for reference in refs if reference not in _WORKER_IMAGE_PATHS]
    if missing:
        return None, "skipped_missing_image", f"{identifier}: missing images {missing!r}"
    images = [_WORKER_IMAGE_PATHS[reference] for reference in refs]
    try:
        output = canonical_grounding_record(record, images, _WORKER_SYSTEM)
        kind = "grounding_written"
        if output is None:
            output = standard_record(record, images)
            kind = "qa_written"
        validate_output_record(output)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, "invalid_record", f"{identifier}: {exc}"
    return json.dumps(output, ensure_ascii=False, separators=(",", ":")), kind, None


def convert_chunk(
    chunk: list[tuple[int, dict[str, Any]]]
) -> tuple[list[str], dict[str, int], list[str]]:
    lines: list[str] = []
    stats: Counter = Counter()
    errors: list[str] = []
    for index, record in chunk:
        stats["records_seen"] += 1
        line, kind, error = convert_source_record(record, index)
        if error is not None:
            stats[kind or "invalid_record"] += 1
            errors.append(error)
            continue
        lines.append(line or "")
        stats["written"] += 1
        stats[kind or "qa_written"] += 1
    return lines, dict(stats), errors


def iter_record_chunks(
    path: Path, limit: int | None, chunk_size: int
) -> Iterator[list[tuple[int, dict[str, Any]]]]:
    chunk: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(iter_source_records(path, limit)):
        chunk.append((index, record))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def bounded_results(
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
                    f"waiting_for_chunk pending_chunks={len(pending) + 1:,}", force=True
                )
        yield result
        try:
            pending.append(executor.submit(convert_chunk, next(iterator)))
        except StopIteration:
            pass


def convert_dataset(
    annotation: Path,
    output_path: Path,
    image_paths: dict[str, str],
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> tuple[Counter, list[str]]:
    temporary = output_path.with_name(f"{output_path.name}.inprogress")
    if temporary.exists() and not args.overwrite:
        raise FileExistsError(
            f"Incomplete output exists; pass --overwrite after checking that no other "
            f"conversion is running: {temporary}"
        )
    stats: Counter = Counter()
    examples: list[str] = []
    chunks = iter_record_chunks(annotation, args.limit, args.worker_chunk_size)
    print(f"[convert] inprogress_output={temporary}", flush=True)
    print(
        f"[monitor] {shlex.join(['watch', '-n', '2', 'ls', '-lh', str(temporary)])}",
        flush=True,
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            if args.num_workers == 1:
                init_convert_worker(image_paths, args.system)
                results = (convert_chunk(chunk) for chunk in chunks)
                executor = None
            else:
                executor = ProcessPoolExecutor(
                    max_workers=args.num_workers,
                    initializer=init_convert_worker,
                    initargs=(image_paths, args.system),
                )
                results = bounded_results(
                    executor, chunks, args.max_pending_tasks, progress
                )
            try:
                for lines, chunk_stats, chunk_errors in results:
                    for line in lines:
                        output.write(line + "\n")
                    output.flush()
                    stats.update(chunk_stats)
                    remaining = args.max_error_examples - len(examples)
                    if remaining > 0:
                        examples.extend(chunk_errors[:remaining])
                    size = temporary.stat().st_size
                    progress.heartbeat(
                        f"records={stats['records_seen']:,} written={stats['written']:,} "
                        f"size={format_size(size)}"
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
                            f"grounding={stats['grounding_written']:,} "
                            f"qa={stats['qa_written']:,} invalid={stats['invalid_record']:,} "
                            f"size={format_size(size)}",
                            flush=True,
                        )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
        if stats["invalid_record"] and args.invalid_record_policy == "error":
            raise ConversionValidationError(
                f"Found {stats['invalid_record']:,} invalid records", stats, examples
            )
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            print(f"[convert] partial_output_retained={temporary}", flush=True)
        raise
    return stats, examples


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(child) for child in value]
    return value


def write_report(path: Path, report: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {path}")
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
    annotation, output_path, report_path, image_dir = validate_args(args)
    args.num_workers = args.num_workers or min(16, max(1, os.cpu_count() or 1))
    args.max_pending_tasks = args.max_pending_tasks or args.num_workers * 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(args.heartbeat_seconds)

    print(f"[input] {args.input_dir}", flush=True)
    print(f"[annotation] {annotation}", flush=True)
    print(f"[output] {output_path}", flush=True)
    print(f"[image_dir] {image_dir}", flush=True)
    print(f"[workers] {args.num_workers}", flush=True)

    report: dict[str, Any] = {
        "input_dir": args.input_dir,
        "annotation": annotation,
        "output_jsonl": output_path,
        "image_dir": image_dir,
        "workers": args.num_workers,
        "status": "running",
    }

    progress.begin("scan", f"annotation={annotation}")
    scan = scan_source(annotation, args, progress)
    progress.finish(f"records={scan.stats['records']:,}")
    report["scan"] = scan.stats

    archives = discover_archives(
        args.input_dir, args.image_archives, scan.image_references
    )
    print(f"[media] archives={len(archives):,}", flush=True)
    for archive in archives:
        print(f"[media] archive={archive}", flush=True)
    progress.begin("media", f"unique_images={len(scan.image_references):,}")
    media = resolve_media(
        scan.image_references,
        args.input_dir,
        image_dir,
        archives,
        args,
        progress,
    )
    progress.finish(
        f"resolved={len(media.paths):,} missing={len(media.missing):,} "
        f"ambiguous={len(media.ambiguous):,}"
    )
    report["media"] = {
        "stats": media.stats,
        "missing_examples": media.missing[: args.max_error_examples],
        "ambiguous_examples": dict(
            list(media.ambiguous.items())[: args.max_error_examples]
        ),
    }

    if media.ambiguous and args.ambiguous_image_policy == "error":
        report["status"] = "failed_ambiguous_images"
        report["stage_seconds"] = progress.timings
        write_report(report_path, report, args.overwrite)
        raise RuntimeError(
            f"Found {len(media.ambiguous):,} ambiguous image references; see {report_path}"
        )
    if media.missing and args.missing_image_policy == "error":
        report["status"] = "failed_missing_images"
        report["stage_seconds"] = progress.timings
        write_report(report_path, report, args.overwrite)
        raise RuntimeError(
            f"Could not resolve {len(media.missing):,} images; see {report_path}"
        )

    image_paths = {reference: str(path.absolute()) for reference, path in media.paths.items()}
    if args.check_only:
        report["status"] = "check_complete"
        report["stage_seconds"] = progress.timings
        write_report(report_path, report, args.overwrite)
        print(f"[done] check complete report={report_path}", flush=True)
        return 0

    try:
        progress.begin("convert", f"records={scan.stats['records']:,}")
        conversion_stats, examples = convert_dataset(
            annotation, output_path, image_paths, args, progress
        )
        progress.finish(f"written={conversion_stats['written']:,}")
    except ConversionValidationError as exc:
        report["status"] = "failed_conversion"
        report["conversion"] = {"stats": exc.stats, "error_examples": exc.examples}
        report["stage_seconds"] = progress.timings
        write_report(report_path, report, args.overwrite)
        raise
    except Exception as exc:
        report["status"] = "failed_conversion"
        report["conversion_error"] = str(exc)
        report["stage_seconds"] = progress.timings
        write_report(report_path, report, args.overwrite)
        raise

    report["conversion"] = {"stats": conversion_stats, "error_examples": examples}
    report["status"] = "complete"
    report["stage_seconds"] = progress.timings
    write_report(report_path, report, args.overwrite)
    print(f"[done] output={output_path}", flush=True)
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
