#!/usr/bin/env python3
"""Convert Visual Genome annotations to aligned ms-swift JSONL.

The converter reads Visual Genome annotations from JSON or JSON ZIP files and
joins every QA or region label to its image through image_data.json's image_id
mapping. Existing VG_100K/VG_100K_2 directories are used directly when found;
otherwise images.zip/images2.zip are extracted. Missing images are never paired
with another label; they are skipped and reported by default. The default
output uses ms-swift SFT multimodal user/assistant messages.

Examples:
    python prepare_visualgenome_swift.py --input-dir /data/VisualGenome/master
    python prepare_visualgenome_swift.py --input-dir /data/VisualGenome/master \
        --images-dir /data/VisualGenome/extracted
    python prepare_visualgenome_swift.py --input-dir /data/VisualGenome/master \
        --missing-image-policy error --tasks qa regions
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, TextIO
from urllib.parse import unquote, urlparse


IMAGE_FOLDER_NAMES = ("VG_100K", "VG_100K_2")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
READ_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class JsonSource:
    path: Path
    member: str | None = None

    def describe(self) -> str:
        if self.member is None:
            return str(self.path)
        return f"{self.path}!/{self.member}"


@dataclass(frozen=True)
class ArchiveMember:
    archive_index: int
    member: str
    file_size: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert Visual Genome annotations into aligned ms-swift PT/SFT JSONL files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir,
        help="Visual Genome root. Subdirectories such as master/ are searched recursively.",
    )
    parser.add_argument(
        "--image-archives",
        nargs="+",
        type=Path,
        default=None,
        help="Image ZIP files. By default, images.zip and images2.zip are discovered recursively.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing extracted VG_100K and/or VG_100K_2 folders. "
            "These images are referenced in place and ZIP archives are not opened. "
            "By default, --output-dir, --extract-dir, and --input-dir are searched."
        ),
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="Image extraction directory. Default: --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="JSONL output directory. Default: <script directory>/ms_swift.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("qa", "regions"),
        default=("qa", "regions"),
        help="Datasets to create: visual question answering and/or grounded region captioning.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("pt", "sft"),
        default="sft",
        help="sft writes user/assistant turns; pt writes assistant-only pre-training text.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Fraction of image IDs assigned to validation. Set to 0 for train files only.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic image-level splitting.")
    parser.add_argument(
        "--missing-image-policy",
        choices=("skip", "error"),
        default="skip",
        help="skip writes only aligned available images and a report; error requires all image_data rows.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write image paths relative to each JSONL file instead of absolute paths.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum output rows per task, useful for creating a test subset.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress every N image-data rows or task labels. Set to 0 to disable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL files and report.")
    parser.add_argument(
        "--qa-prompt",
        default="<image>\n{question}",
        help="QA user prompt used in sft mode.",
    )
    parser.add_argument(
        "--region-prompt",
        default="<image>\nDescribe the region <bbox>.",
        help="Region user prompt used in sft mode.",
    )
    parser.add_argument(
        "--qa-pt-template",
        default="<image>Question: {question}\nAnswer: {answer}",
        help="Assistant-only QA text used in pt mode.",
    )
    parser.add_argument(
        "--region-pt-template",
        default="<image>Region <bbox>: {phrase}",
        help="Assistant-only grounded region text used in pt mode.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.images_dir is not None and args.image_archives is not None:
        raise SystemExit("--images-dir and --image-archives cannot be used together")
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be in the range [0, 1)")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be greater than zero")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be greater than or equal to zero")
    if args.dataset_mode == "pt":
        if "qa" in args.tasks and not all(
            token in args.qa_pt_template for token in ("<image>", "{question}", "{answer}")
        ):
            raise SystemExit("--qa-pt-template must contain '<image>', '{question}', and '{answer}'")
        if "regions" in args.tasks and not all(
            token in args.region_pt_template for token in ("<image>", "<bbox>", "{phrase}")
        ):
            raise SystemExit("--region-pt-template must contain '<image>', '<bbox>', and '{phrase}'")
    else:
        if "qa" in args.tasks and ("<image>" not in args.qa_prompt or "{question}" not in args.qa_prompt):
            raise SystemExit("--qa-prompt must contain '<image>' and '{question}'")
        if "regions" in args.tasks and (
            "<image>" not in args.region_prompt or "<bbox>" not in args.region_prompt
        ):
            raise SystemExit("--region-prompt must contain '<image>' and '<bbox>'")


def find_named_files(root: Path, filename: str) -> list[Path]:
    matches: list[Path] = []
    skipped_dirs = {".git", "__pycache__", "ms_swift", "ms_swift_parquet", *IMAGE_FOLDER_NAMES}
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in skipped_dirs]
        if filename in filenames:
            matches.append(Path(current) / filename)
    return sorted(matches)


def find_json_source(root: Path, filename: str) -> JsonSource:
    plain_files = find_named_files(root, filename)
    if plain_files:
        return JsonSource(plain_files[0])

    archive_name = f"{filename}.zip"
    for archive in find_named_files(root, archive_name):
        with zipfile.ZipFile(archive) as zf:
            members = [name for name in zf.namelist() if PurePosixPath(name).name == filename]
        if members:
            return JsonSource(archive, members[0])

    raise FileNotFoundError(
        f"Could not find {filename} or {archive_name} below {root}. Check --input-dir."
    )


def find_image_archives(root: Path) -> list[Path]:
    archives = []
    for filename in ("images.zip", "images2.zip"):
        archives.extend(find_named_files(root, filename))
    archives = sorted(set(archives), key=lambda path: (path.name != "images.zip", str(path)))
    if not archives:
        raise FileNotFoundError(f"Could not find images.zip or images2.zip below {root}")
    return archives


def find_extracted_image_dirs(root: Path) -> dict[str, Path]:
    """Find VG image folders without descending into their large contents."""
    if not root.is_dir():
        return {}
    root = root.resolve()
    if root.name in IMAGE_FOLDER_NAMES:
        return {root.name: root}

    matches: dict[str, Path] = {}
    skipped_dirs = {".git", "__pycache__"}
    for current, dirnames, _ in os.walk(root):
        dirnames.sort()
        current_path = Path(current)
        for folder_name in IMAGE_FOLDER_NAMES:
            if folder_name not in dirnames:
                continue
            candidate = (current_path / folder_name).resolve()
            previous = matches.get(folder_name)
            if previous is not None and previous != candidate:
                raise ValueError(
                    f"Found multiple {folder_name} directories below {root}: "
                    f"{previous}, {candidate}. Use --images-dir to select one root."
                )
            matches[folder_name] = candidate
        dirnames[:] = [
            name for name in dirnames if name not in skipped_dirs and name not in IMAGE_FOLDER_NAMES
        ]
    return matches


def resolve_extracted_image_dirs(
    configured: Path | None,
    search_roots: list[Path],
) -> dict[str, Path]:
    if configured is not None:
        root = configured.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {root}")
        image_dirs = find_extracted_image_dirs(root)
        if not image_dirs:
            expected = ", ".join(IMAGE_FOLDER_NAMES)
            raise FileNotFoundError(f"Could not find {expected} below --images-dir {root}")
        return image_dirs

    image_dirs: dict[str, Path] = {}
    seen_roots: set[Path] = set()
    for raw_root in search_roots:
        root = raw_root.expanduser().resolve()
        if root in seen_roots or not root.is_dir():
            continue
        seen_roots.add(root)
        for folder_name, path in find_extracted_image_dirs(root).items():
            image_dirs.setdefault(folder_name, path)
        if len(image_dirs) == len(IMAGE_FOLDER_NAMES):
            break
    return image_dirs


def resolve_image_archives(args: argparse.Namespace) -> list[Path]:
    if args.image_archives is None:
        return find_image_archives(args.input_dir)
    archives = [path.expanduser().resolve() for path in args.image_archives]
    missing = [path for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Image archive does not exist: " + ", ".join(str(path) for path in missing)
        )
    if len(set(archives)) != len(archives):
        raise ValueError("--image-archives contains duplicate paths")
    return archives


@contextmanager
def open_json_source(source: JsonSource) -> Iterator[TextIO]:
    if source.member is None:
        with source.path.open("r", encoding="utf-8-sig") as stream:
            yield stream
        return
    with zipfile.ZipFile(source.path) as zf:
        with zf.open(source.member, "r") as raw_stream:
            with io.TextIOWrapper(raw_stream, encoding="utf-8-sig") as stream:
                yield stream


def iter_json_array(stream: TextIO) -> Iterator[dict]:
    """Stream objects from a top-level JSON array using only stdlib."""
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
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) or not read_more():
                break
        if not started:
            if position >= len(buffer) or buffer[position] != "[":
                raise ValueError("Expected a top-level JSON array")
            position += 1
            started = True
            continue
        while True:
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
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
            except json.JSONDecodeError:
                if not read_more():
                    raise ValueError("Invalid or truncated JSON input") from None
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object, got {type(value).__name__}")
        yield value


def url_image_parts(url: str) -> tuple[str, str] | None:
    parts = PurePosixPath(unquote(urlparse(url).path)).parts
    for folder_name in IMAGE_FOLDER_NAMES:
        if folder_name in parts:
            index = parts.index(folder_name)
            if index + 1 < len(parts):
                return folder_name, parts[index + 1]
    return None


class ImageArchives:
    """Index and safely extract multiple Visual Genome image archives."""

    def __init__(self, archive_paths: list[Path]) -> None:
        self.archive_paths = archive_paths
        self.zips: list[zipfile.ZipFile] = []
        self.members: dict[tuple[str, str], ArchiveMember] = {}
        self.members_by_filename: dict[str, ArchiveMember] = {}
        self.image_members: list[ArchiveMember] = []

    def __enter__(self) -> "ImageArchives":
        try:
            for archive_index, archive_path in enumerate(self.archive_paths):
                zf = zipfile.ZipFile(archive_path)
                self.zips.append(zf)
                count = 0
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    path = PurePosixPath(info.filename)
                    if path.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    location = ArchiveMember(archive_index, info.filename, info.file_size)
                    self.image_members.append(location)
                    count += 1

                    previous_filename = self.members_by_filename.get(path.name)
                    if previous_filename is not None and previous_filename != location:
                        raise ValueError(f"Duplicate image filename across archives: {path.name}")
                    self.members_by_filename[path.name] = location

                    for folder_name in IMAGE_FOLDER_NAMES:
                        if folder_name in path.parts:
                            key = (folder_name, path.name)
                            previous = self.members.get(key)
                            if previous is not None and previous != location:
                                raise ValueError(f"Duplicate image archive member for {key}")
                            self.members[key] = location
                            break
                print(f"[info] indexed {count:,} images from {archive_path}")
        except BaseException:
            self.close()
            raise
        if not self.image_members:
            self.close()
            raise ValueError("No image files found in the selected archives")
        print(f"[info] indexed total images={len(self.image_members):,}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        for zf in self.zips:
            zf.close()
        self.zips.clear()

    def resolve_member(self, image: dict) -> ArchiveMember | None:
        url_parts = url_image_parts(str(image.get("url") or ""))
        if url_parts is not None:
            member = self.members.get(url_parts)
            if member is not None:
                return member
            member = self.members_by_filename.get(url_parts[1])
            if member is not None:
                return member
        image_id = image.get("image_id", image.get("id"))
        if image_id is not None:
            for suffix in IMAGE_SUFFIXES:
                member = self.members_by_filename.get(f"{image_id}{suffix}")
                if member is not None:
                    return member
        return None

    def extract(self, destination: Path) -> dict[ArchiveMember, Path]:
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
        paths: dict[ArchiveMember, Path] = {}
        target_owners: dict[Path, ArchiveMember] = {}
        extracted = 0
        skipped = 0

        for location in self.image_members:
            zf = self.zips[location.archive_index]
            target = (destination / location.member).resolve()
            try:
                target.relative_to(destination)
            except ValueError:
                raise ValueError(f"Unsafe path in image archive: {location.member}") from None
            previous = target_owners.get(target)
            if previous is not None and previous != location:
                raise ValueError(f"Multiple archive members map to extraction path: {target}")
            target_owners[target] = location
            paths[location] = target

            if target.is_file() and target.stat().st_size == location.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.part")
            try:
                with zf.open(location.member, "r") as src, temporary.open("wb") as dst:
                    while True:
                        chunk = src.read(READ_CHUNK_SIZE)
                        if not chunk:
                            break
                        dst.write(chunk)
                actual_size = temporary.stat().st_size
                if actual_size != location.file_size:
                    raise IOError(
                        f"Extracted size mismatch for {location.member}: "
                        f"expected {location.file_size}, got {actual_size}"
                    )
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            extracted += 1
            if extracted % 10000 == 0:
                print(f"[extract] files={extracted:,} existing={skipped:,}")

        print(f"[extract] complete files={extracted:,} existing={skipped:,} destination={destination}")
        return paths


def print_image_mapping_progress(
    processed: int,
    matched: int,
    missing: int,
    progress_every: int,
) -> None:
    if progress_every > 0 and processed % progress_every == 0:
        print(
            f"[progress] stage=image-map processed={processed:,} "
            f"matched={matched:,} missing={missing:,}",
            flush=True,
        )


def load_image_paths(
    source: JsonSource,
    archives: ImageArchives,
    extracted_paths: dict[ArchiveMember, Path],
    missing_policy: str,
    progress_every: int,
) -> tuple[dict[int, Path], list[int]]:
    image_paths: dict[int, Path] = {}
    missing_ids: list[int] = []
    processed = 0
    with open_json_source(source) as stream:
        for image in iter_json_array(stream):
            processed += 1
            raw_image_id = image.get("image_id", image.get("id"))
            if raw_image_id is None:
                raise ValueError("image_data.json row has no image_id")
            image_id = int(raw_image_id)
            member = archives.resolve_member(image)
            path = extracted_paths.get(member) if member is not None else None
            if path is None or not path.is_file():
                missing_ids.append(image_id)
            else:
                previous = image_paths.get(image_id)
                if previous is not None and previous != path:
                    raise ValueError(f"Conflicting image paths for image_id={image_id}: {previous}, {path}")
                image_paths[image_id] = path
            print_image_mapping_progress(processed, len(image_paths), len(missing_ids), progress_every)

    if missing_ids and missing_policy == "error":
        preview = ", ".join(str(image_id) for image_id in missing_ids[:10])
        raise RuntimeError(
            f"Missing {len(missing_ids):,} images referenced by image_data.json. First IDs: {preview}"
        )
    if missing_ids:
        print(
            f"[warning] missing images={len(missing_ids):,}; their QA/region labels will be skipped "
            "and listed in the conversion report"
        )
    print(f"[info] matched image IDs={len(image_paths):,}")
    return image_paths, missing_ids


def resolve_directory_image(image: dict, image_dirs: dict[str, Path]) -> Path | None:
    url_parts = url_image_parts(str(image.get("url") or ""))
    if url_parts is not None:
        folder_name, filename = url_parts
        directory = image_dirs.get(folder_name)
        if directory is not None:
            candidate = directory / filename
            if candidate.is_file():
                return candidate.resolve()

        for fallback_dir in image_dirs.values():
            candidate = fallback_dir / filename
            if candidate.is_file():
                return candidate.resolve()

    image_id = image.get("image_id", image.get("id"))
    if image_id is not None:
        for suffix in IMAGE_SUFFIXES:
            filename = f"{image_id}{suffix}"
            for directory in image_dirs.values():
                candidate = directory / filename
                if candidate.is_file():
                    return candidate.resolve()
    return None


def load_image_paths_from_directories(
    source: JsonSource,
    image_dirs: dict[str, Path],
    missing_policy: str,
    progress_every: int,
) -> tuple[dict[int, Path], list[int]]:
    image_paths: dict[int, Path] = {}
    missing_ids: list[int] = []
    processed = 0
    with open_json_source(source) as stream:
        for image in iter_json_array(stream):
            processed += 1
            raw_image_id = image.get("image_id", image.get("id"))
            if raw_image_id is None:
                raise ValueError("image_data.json row has no image_id")
            image_id = int(raw_image_id)
            path = resolve_directory_image(image, image_dirs)
            if path is None:
                missing_ids.append(image_id)
            else:
                previous = image_paths.get(image_id)
                if previous is not None and previous != path:
                    raise ValueError(f"Conflicting image paths for image_id={image_id}: {previous}, {path}")
                image_paths[image_id] = path
            print_image_mapping_progress(processed, len(image_paths), len(missing_ids), progress_every)

    if missing_ids and missing_policy == "error":
        preview = ", ".join(str(image_id) for image_id in missing_ids[:10])
        raise RuntimeError(
            f"Missing {len(missing_ids):,} images referenced by image_data.json. First IDs: {preview}"
        )
    if missing_ids:
        print(
            f"[warning] missing images={len(missing_ids):,}; their QA/region labels will be skipped "
            "and listed in the conversion report"
        )
    print(f"[info] matched image IDs={len(image_paths):,}")
    return image_paths, missing_ids


def is_validation_image(image_id: int, ratio: float, seed: int) -> bool:
    if ratio == 0:
        return False
    digest = hashlib.blake2b(f"{seed}:{image_id}".encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64) < ratio


def format_image_path(image_path: Path, jsonl_path: Path, relative: bool) -> str:
    if relative:
        return os.path.relpath(image_path, jsonl_path.parent).replace(os.sep, "/")
    return str(image_path.resolve())


class JsonlTaskWriter:
    def __init__(
        self,
        task: str,
        output_dir: Path,
        val_ratio: float,
        seed: int,
        relative_paths: bool,
        overwrite: bool,
    ) -> None:
        self.task = task
        self.output_dir = output_dir
        self.val_ratio = val_ratio
        self.seed = seed
        self.relative_paths = relative_paths
        self.paths = {"train": output_dir / f"visualgenome_{task}_train.jsonl"}
        if val_ratio > 0:
            self.paths["val"] = output_dir / f"visualgenome_{task}_val.jsonl"
        if not overwrite:
            existing = [path for path in self.paths.values() if path.exists()]
            if existing:
                raise FileExistsError(
                    "Output already exists: " + ", ".join(str(path) for path in existing)
                )
        self.temporary_paths = {
            split: path.with_name(f"{path.name}.tmp") for split, path in self.paths.items()
        }
        self.streams: dict[str, TextIO] = {}
        self.counts = {split: 0 for split in self.paths}

    def __enter__(self) -> "JsonlTaskWriter":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for path in self.temporary_paths.values():
            if path.exists():
                path.unlink()
        self.streams = {
            split: path.open("w", encoding="utf-8", newline="\n")
            for split, path in self.temporary_paths.items()
        }
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for stream in self.streams.values():
            stream.close()
        if exc_type is None:
            for split, temporary in self.temporary_paths.items():
                os.replace(temporary, self.paths[split])
        else:
            for temporary in self.temporary_paths.values():
                if temporary.exists():
                    temporary.unlink()

    def write(self, image_id: int, image_path: Path, record: dict) -> None:
        image_tokens = sum(message["content"].count("<image>") for message in record["messages"])
        if image_tokens != 1:
            raise ValueError(
                f"image_id={image_id} has {image_tokens} <image> tokens; exactly one is required"
            )
        objects = record.get("objects")
        if objects is not None:
            bbox_tokens = sum(message["content"].count("<bbox>") for message in record["messages"])
            if bbox_tokens != len(objects["bbox"]):
                raise ValueError(
                    f"image_id={image_id} has {bbox_tokens} <bbox> tokens but "
                    f"{len(objects['bbox'])} boxes"
                )
        split = "val" if is_validation_image(image_id, self.val_ratio, self.seed) else "train"
        record["images"] = [format_image_path(image_path, self.paths[split], self.relative_paths)]
        self.streams[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.counts[split] += 1


def get_matched_image_id(parent_image_id: object, annotation: dict, name: str) -> int:
    annotation_image_id = annotation.get("image_id")
    if annotation_image_id is None and parent_image_id is None:
        raise ValueError(f"{name} has no image_id")
    image_id = int(annotation_image_id if annotation_image_id is not None else parent_image_id)
    if parent_image_id is not None and int(parent_image_id) != image_id:
        raise ValueError(f"{name} image_id mismatch: parent={parent_image_id}, annotation={image_id}")
    return image_id


def print_task_progress(
    task: str,
    image_rows: int,
    labels: int,
    written: int,
    invalid: int,
    missing_image: int,
    progress_every: int,
) -> None:
    if progress_every > 0 and labels % progress_every == 0:
        print(
            f"[progress] task={task} image_rows={image_rows:,} labels={labels:,} "
            f"written={written:,} invalid={invalid:,} missing_image={missing_image:,}",
            flush=True,
        )


def convert_qa(
    source: JsonSource,
    image_paths: dict[int, Path],
    writer: JsonlTaskWriter,
    args: argparse.Namespace,
) -> dict:
    written = 0
    invalid_labels = 0
    missing_image_labels = 0
    source_image_rows = 0
    processed_labels = 0
    with writer, open_json_source(source) as stream:
        for image_entry in iter_json_array(stream):
            source_image_rows += 1
            parent_image_id = image_entry.get("image_id", image_entry.get("id"))
            for qa in image_entry.get("qas") or []:
                if args.max_samples is not None and written >= args.max_samples:
                    break
                processed_labels += 1
                try:
                    question = str(qa.get("q", qa.get("question", ""))).strip()
                    answer = str(qa.get("a", qa.get("answer", ""))).strip()
                    if not question or not answer:
                        invalid_labels += 1
                        continue
                    qa_id = qa.get("qa_id")
                    if qa_id is None:
                        raise ValueError("QA annotation has no qa_id")
                    image_id = get_matched_image_id(parent_image_id, qa, f"QA {qa_id}")
                    image_path = image_paths.get(image_id)
                    if image_path is None:
                        missing_image_labels += 1
                        continue
                    if args.dataset_mode == "pt":
                        messages = [{
                            "role": "assistant",
                            "content": args.qa_pt_template.format(question=question, answer=answer),
                        }]
                    else:
                        messages = [
                            {"role": "user", "content": args.qa_prompt.format(question=question)},
                            {"role": "assistant", "content": answer},
                        ]
                    record = {
                        "image_id": image_id,
                        "qa_id": int(qa_id),
                        "messages": messages,
                    }
                    writer.write(image_id, image_path, record)
                    written += 1
                finally:
                    print_task_progress(
                        "qa",
                        source_image_rows,
                        processed_labels,
                        written,
                        invalid_labels,
                        missing_image_labels,
                        args.progress_every,
                    )
            if args.max_samples is not None and written >= args.max_samples:
                break
    print(
        f"[info] task=qa image_rows={source_image_rows:,} labels={processed_labels:,} "
        f"rows={written:,} invalid={invalid_labels:,} missing_image={missing_image_labels:,}"
    )
    return {
        "rows": written,
        "train_rows": writer.counts["train"],
        "val_rows": writer.counts.get("val", 0),
        "source_image_rows": source_image_rows,
        "processed_labels": processed_labels,
        "invalid_labels": invalid_labels,
        "missing_image_labels": missing_image_labels,
    }


def convert_regions(
    source: JsonSource,
    image_paths: dict[int, Path],
    writer: JsonlTaskWriter,
    args: argparse.Namespace,
) -> dict:
    written = 0
    invalid_labels = 0
    missing_image_labels = 0
    source_image_rows = 0
    processed_labels = 0
    with writer, open_json_source(source) as stream:
        for image_entry in iter_json_array(stream):
            source_image_rows += 1
            parent_image_id = image_entry.get("image_id", image_entry.get("id"))
            for region in image_entry.get("regions") or []:
                if args.max_samples is not None and written >= args.max_samples:
                    break
                processed_labels += 1
                try:
                    phrase = str(region.get("phrase", "")).strip()
                    try:
                        x = float(region["x"])
                        y = float(region["y"])
                        width = float(region["width"])
                        height = float(region["height"])
                    except (KeyError, TypeError, ValueError):
                        invalid_labels += 1
                        continue
                    if not phrase or width <= 0 or height <= 0:
                        invalid_labels += 1
                        continue
                    region_id = region.get("region_id")
                    if region_id is None:
                        raise ValueError("Region annotation has no region_id")
                    image_id = get_matched_image_id(parent_image_id, region, f"region {region_id}")
                    image_path = image_paths.get(image_id)
                    if image_path is None:
                        missing_image_labels += 1
                        continue
                    if args.dataset_mode == "pt":
                        messages = [{
                            "role": "assistant",
                            "content": args.region_pt_template.format(phrase=phrase),
                        }]
                    else:
                        messages = [
                            {"role": "user", "content": args.region_prompt},
                            {"role": "assistant", "content": phrase},
                        ]
                    record = {
                        "image_id": image_id,
                        "region_id": int(region_id),
                        "messages": messages,
                        "objects": {
                            "ref": [],
                            "bbox": [[x, y, x + width, y + height]],
                            "bbox_type": "real",
                            "image_id": [0],
                        },
                    }
                    writer.write(image_id, image_path, record)
                    written += 1
                finally:
                    print_task_progress(
                        "regions",
                        source_image_rows,
                        processed_labels,
                        written,
                        invalid_labels,
                        missing_image_labels,
                        args.progress_every,
                    )
            if args.max_samples is not None and written >= args.max_samples:
                break
    print(
        f"[info] task=regions image_rows={source_image_rows:,} labels={processed_labels:,} "
        f"rows={written:,} invalid={invalid_labels:,} missing_image={missing_image_labels:,}"
    )
    return {
        "rows": written,
        "train_rows": writer.counts["train"],
        "val_rows": writer.counts.get("val", 0),
        "source_image_rows": source_image_rows,
        "processed_labels": processed_labels,
        "invalid_labels": invalid_labels,
        "missing_image_labels": missing_image_labels,
    }


def write_report(path: Path, report: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[write] {path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.input_dir = args.input_dir.expanduser().resolve()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    output_dir = (args.output_dir or (Path(__file__).resolve().parent / "ms_swift")).expanduser().resolve()
    extract_dir = (args.extract_dir or output_dir).expanduser().resolve()

    image_data_source = find_json_source(args.input_dir, "image_data.json")
    print(f"[read] {image_data_source.describe()}")

    image_dirs: dict[str, Path] = {}
    if args.images_dir is not None:
        image_dirs = resolve_extracted_image_dirs(args.images_dir, [])
    elif args.image_archives is None:
        image_dirs = resolve_extracted_image_dirs(
            None,
            [output_dir, extract_dir, args.input_dir],
        )

    archive_paths: list[Path] = []
    if image_dirs:
        for folder_name in IMAGE_FOLDER_NAMES:
            if folder_name in image_dirs:
                print(f"[read] extracted images {folder_name}={image_dirs[folder_name]}")
        image_paths, missing_image_ids = load_image_paths_from_directories(
            image_data_source,
            image_dirs,
            args.missing_image_policy,
            args.progress_every,
        )
        image_source_mode = "directories"
    else:
        archive_paths = resolve_image_archives(args)
        for archive_path in archive_paths:
            print(f"[read] {archive_path}")
        with ImageArchives(archive_paths) as archives:
            extracted_paths = archives.extract(extract_dir)
            image_paths, missing_image_ids = load_image_paths(
                image_data_source,
                archives,
                extracted_paths,
                args.missing_image_policy,
                args.progress_every,
            )
        image_source_mode = "archives"

    writer_kwargs = {
        "output_dir": output_dir,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "relative_paths": args.relative_paths,
        "overwrite": args.overwrite,
    }
    task_stats = {}
    if "qa" in args.tasks:
        source = find_json_source(args.input_dir, "question_answers.json")
        print(f"[read] {source.describe()}")
        task_stats["qa"] = convert_qa(
            source,
            image_paths,
            JsonlTaskWriter("qa", **writer_kwargs),
            args,
        )
    if "regions" in args.tasks:
        source = find_json_source(args.input_dir, "region_descriptions.json")
        print(f"[read] {source.describe()}")
        task_stats["regions"] = convert_regions(
            source,
            image_paths,
            JsonlTaskWriter("regions", **writer_kwargs),
            args,
        )

    report = {
        "dataset_mode": args.dataset_mode,
        "input_dir": str(args.input_dir),
        "image_source": image_source_mode,
        "extract_dir": str(extract_dir) if image_source_mode == "archives" else None,
        "image_archives": [str(path) for path in archive_paths],
        "image_directories": {
            folder_name: str(image_dirs[folder_name])
            for folder_name in IMAGE_FOLDER_NAMES
            if folder_name in image_dirs
        },
        "progress_every": args.progress_every,
        "available_image_ids": len(image_paths),
        "missing_image_ids_count": len(missing_image_ids),
        "missing_image_ids": missing_image_ids,
        "tasks": task_stats,
    }
    write_report(output_dir / "visualgenome_conversion_report.json", report, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
