#!/usr/bin/env python3
"""Convert RoboVQA JSON annotations and extracted clips to ms-swift JSONL.

Expected local layout:
    /mnt/luojunkun/stage1/dataset/robovqa/
      README.md
      robovqa_reasoning_*.json
      robovqa_understanding.json
      clips/

The converter reads the JSON annotations, resolves referenced media below
clips/, preserves the original id, and writes user/assistant records in the
ms-swift multimodal SFT format.

Example:
    python prepare_robovqa_swift.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


DEFAULT_INPUT_DIR = Path("/mnt/luojunkun/stage1/dataset/robovqa")
DEFAULT_OUTPUT_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/robovqa")
DEFAULT_JSON_PATTERNS = ("robovqa_reasoning_*.json", "robovqa_understanding.json")

QUESTION_KEYS = ("question", "query", "instruction", "prompt", "problem")
ANSWER_KEYS = ("answer", "answers", "correct_answer", "response", "output", "target")
CATEGORY_KEYS = (
    "category",
    "categories",
    "class",
    "label",
    "labels",
    "tag",
    "tags",
    "task",
    "task_type",
    "question_type",
    "reasoning_type",
    "skill",
)
ID_KEYS = ("id", "uid", "qid", "question_id", "question_idx", "sample_id", "video_id", "clip_id")

IMAGE_KEYS = ("image", "images", "image_path", "image_paths", "frame", "frames", "frame_path", "frame_paths")
VIDEO_KEYS = (
    "video",
    "videos",
    "video_path",
    "video_paths",
    "clip",
    "clips",
    "clip_path",
    "clip_paths",
    "clip_name",
    "episode",
    "trajectory",
)
AUDIO_KEYS = ("audio", "audios", "audio_path", "audio_paths", "sound", "sounds")
GENERIC_MEDIA_KEYS = ("media", "media_path", "media_paths", "file", "files", "path", "paths", "url", "urls")
CONVERSATION_KEYS = ("conversations", "conversation", "messages")
METADATA_MEDIA_KEYS = ("video_location", "image_location", "audio_location")

CONTAINER_KEYS = {
    "data",
    "dataset",
    "annotations",
    "annotation",
    "items",
    "samples",
    "records",
    "questions",
    "qa",
    "qas",
    "qa_pairs",
    "examples",
    "entries",
    "instances",
    "metadata",
    "info",
    "config",
    "features",
    "citation",
    "license",
}

MEDIA_SUFFIXES = {
    "images": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"},
    "videos": {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"},
    "audios": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"},
}
MEDIA_TOKENS = {"images": "<image>", "videos": "<video>", "audios": "<audio>"}
TASK_BLOCK_PATTERN = re.compile(r"<task:([^>]+)>(.*?)(?=<task:[^>]+>|$)", re.DOTALL)
PRED_PATTERN = re.compile(r"<PRED>(.*?)</PRED>", re.DOTALL)
PRED_ANSWER_PATTERN = re.compile(r"<PRED:ANSWER>(.*?)</PRED:ANSWER>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(
    r"<answer\b[^>]*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL
)
THINK_OPEN_PATTERN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_PATTERN = re.compile(r"</think\s*>", re.IGNORECASE)
ANSWER_OPEN_PATTERN = re.compile(r"<answer\b[^>]*>", re.IGNORECASE)
ANSWER_CLOSE_PATTERN = re.compile(r"</answer\s*>", re.IGNORECASE)
ANSWER_TAG_PATTERN = re.compile(r"</?answer\b[^>]*>", re.IGNORECASE)
THINK_TAG_PATTERN = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
MALFORMED_REASONING_TAG_PATTERN = re.compile(
    r"<\s*/?\s*(?:think|answer)\b", re.IGNORECASE
)
RESIDUAL_REASONING_PATTERN = re.compile(
    r"tags?\s+as\s+per\s+(?:the\s+)?requirements|"
    r"since\s+(?:it(?:'s|\s+is)|this\s+is)\s+(?:not\s+)?a\s+yes/no\s+question|"
    r"\b(?:we|i)\s+(?:need|should|must)\s+(?:to\s+)?(?:answer|respond|provide)\b|"
    r"\bthe\s+(?:user|question)\s+(?:asks|is\s+asking)\b",
    re.IGNORECASE,
)
FORMAT_REQUEST_PATTERN = re.compile(
    r"\s*Please\s+answer(?:\s+the\s+question)?\s+in\s+the\s+following\s+format\s*:\s*"
    r"<think\b[^>]*>.*?</think\s*>\s*<answer\b[^>]*>.*?</answer\s*>\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class SourceRecord:
    source_path: Path
    source_index: int
    context: dict[str, Any]
    record: dict[str, Any]


@dataclass(frozen=True)
class JsonSource:
    path: Path
    rows: int


@dataclass
class ConversionStats:
    read_rows: int = 0
    written_rows: int = 0
    skipped_missing_media: int = 0
    skipped_invalid_text: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RoboVQA JSON annotations to ms-swift multimodal SFT JSONL."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="RoboVQA dataset root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="ms-swift output directory.")
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=None,
        help="Directory containing extracted clips. Default: <input-dir>/clips.",
    )
    parser.add_argument(
        "--archive-extract-dir",
        type=Path,
        default=None,
        help="Where clips_part_*.tar.gz archives are extracted. Default: <output-dir>/robovqa_clips.",
    )
    parser.add_argument(
        "--json-files",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Annotation JSON files. Default: robovqa_reasoning_*.json and "
            "robovqa_understanding.json below --input-dir."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="robovqa_train_sft.jsonl",
        help="Output JSONL filename written below --output-dir.",
    )
    parser.add_argument(
        "--pt-template",
        default="{media_tokens}\nQuestion: {question}\nAnswer: {answer}",
        help=(
            "Assistant-only template. Available fields: {media_tokens}, {question}, "
            "{answer}, {category}, {source_file}, {source_shard}, {id}."
        ),
    )
    parser.add_argument(
        "--user-template",
        default="{media_tokens}\nQuestion: {question}",
        help=(
            "SFT user template. Available fields: {media_tokens}, {question}, "
            "{category}, {source_file}, {source_shard}, {id}."
        ),
    )
    parser.add_argument(
        "--missing-media-policy",
        choices=("skip", "error", "text"),
        default="skip",
        help="How to handle rows whose referenced media cannot be resolved.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write media paths relative to the JSONL file instead of absolute paths.",
    )
    parser.add_argument(
        "--reasoning-policy",
        choices=("answer-only", "preserve"),
        default="answer-only",
        help=(
            "How to handle assistant <think>/<answer> output. The default removes "
            "synthetic reasoning and supervises only the final answer."
        ),
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=512,
        help="Maximum answer-only target length; 0 disables the limit.",
    )
    parser.add_argument(
        "--overlong-answer-policy",
        choices=("compact", "drop", "error"),
        default="compact",
        help=(
            "How to handle final answers over --max-answer-chars. compact keeps "
            "complete leading/trailing sentences; drop skips the record."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit converted rows for quick tests.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Worker processes for row/media resolution. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--worker-chunk-size",
        type=int,
        default=256,
        help="Records sent to each worker task when --num-workers is greater than 1.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL/report files.")
    parser.add_argument("--skip-convert", action="store_true", help="Only validate JSON/clips metadata.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be greater than zero")
    if args.num_workers <= 0:
        raise SystemExit("--num-workers must be greater than zero")
    if args.worker_chunk_size <= 0:
        raise SystemExit("--worker-chunk-size must be greater than zero")
    if args.max_answer_chars < 0:
        raise SystemExit("--max-answer-chars must be greater than or equal to zero")
    if args.max_samples is not None and args.num_workers > 1:
        print("[warning] --max-samples is set; falling back to --num-workers 1 for exact limits")
        args.num_workers = 1
    if args.max_samples is not None and args.worker_chunk_size != 1:
        args.worker_chunk_size = 1
    for token in ("{media_tokens}", "{question}"):
        if token not in args.user_template:
            raise SystemExit(f"--user-template must contain {token!r}")


def load_readme(input_dir: Path) -> str | None:
    readme_path = input_dir / "README.md"
    if not readme_path.is_file():
        print(f"[warning] README.md not found: {readme_path}")
        return None
    text = readme_path.read_text(encoding="utf-8")
    print(f"[read] {readme_path}")
    lowered = text.lower()
    if "loader-ready" in lowered:
        print("[ok] README mentions loader-ready subset")
    if "clip" in lowered:
        print("[ok] README mentions clips")
    else:
        print("[warning] README does not mention clips")
    return text


def discover_json_files(input_dir: Path, selected: list[Path] | None) -> list[Path]:
    if selected:
        files = [(path if path.is_absolute() else input_dir / path).expanduser().resolve() for path in selected]
    else:
        files = []
        for pattern in DEFAULT_JSON_PATTERNS:
            files.extend(sorted(path for path in input_dir.glob(pattern) if path.is_file()))
        files = sorted(dict.fromkeys(files))
    if not files:
        patterns = ", ".join(DEFAULT_JSON_PATTERNS)
        raise FileNotFoundError(f"No RoboVQA JSON annotation files found below {input_dir}; patterns: {patterns}")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("JSON annotation file does not exist: " + ", ".join(map(str, missing)))
    for path in files:
        print(f"[read] {path}")
    return files


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def first_key_value(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, dict)) and not value:
            continue
        return value
    return None


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, (list, tuple)):
        texts = [text_value(item) for item in value]
        return "; ".join(text for text in texts if text)
    if isinstance(value, dict):
        for key in ("answer", "text", "value", "label", "content", "name"):
            text = text_value(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def looks_like_qa_record(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    has_question = any(key in keys for key in QUESTION_KEYS)
    has_answer = any(key in keys for key in ANSWER_KEYS)
    has_media_hint = any(key in keys for key in IMAGE_KEYS + VIDEO_KEYS + AUDIO_KEYS + GENERIC_MEDIA_KEYS)
    return has_question and has_answer and has_media_hint


def looks_like_loader_ready_text_record(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    has_text = "text" in keys
    has_media_hint = any(key in keys for key in IMAGE_KEYS + VIDEO_KEYS + AUDIO_KEYS + GENERIC_MEDIA_KEYS)
    return has_text and has_media_hint


def looks_like_conversation_record(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    has_conversation = any(key in keys for key in CONVERSATION_KEYS)
    has_media_hint = any(key in keys for key in IMAGE_KEYS + VIDEO_KEYS + AUDIO_KEYS + GENERIC_MEDIA_KEYS)
    return has_conversation and has_media_hint


def strip_xmlish_tags(value: str) -> str:
    text = re.sub(r"</?[^>]+>", " ", value)
    return " ".join(text.strip().split())


def clean_answer_text(value: str) -> str:
    value = value.strip()
    answer_match = PRED_ANSWER_PATTERN.search(value)
    if answer_match:
        value = answer_match.group(1)
    value = re.sub(r"^\s*A\s*:\s*", "", value, flags=re.IGNORECASE)
    return strip_xmlish_tags(value)


def clean_question_text(task_name: str, value: str) -> str:
    value = strip_xmlish_tags(value)
    if not value:
        return ""
    if task_name:
        return f"Task: {task_name}\n{value}"
    return value


def split_loader_ready_text(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand RoboVQA uid/text/video rows into one QA record per <task> block."""
    text = text_value(record.get("text"))
    if not text:
        return []

    blocks = list(TASK_BLOCK_PATTERN.finditer(text))
    if not blocks:
        blocks = [re.match(r"(?s)(.*)", text)]

    expanded: list[dict[str, Any]] = []
    base_id = text_value(first_key_value(record, ID_KEYS)) or text_value(record.get("uid"))
    for index, block in enumerate(blocks):
        if block is None:
            continue
        task_name = block.group(1).strip() if len(block.groups()) > 1 else ""
        body = block.group(2) if len(block.groups()) > 1 else block.group(1)
        pred_match = PRED_PATTERN.search(body)
        if not pred_match:
            continue

        question = clean_question_text(task_name, body[: pred_match.start()])
        answer = clean_answer_text(pred_match.group(1))
        if not question or not answer:
            continue

        item = dict(record)
        item["question"] = question
        item["answer"] = answer
        item["task"] = task_name
        if base_id:
            item["id"] = f"{base_id}:{index}"
        expanded.append(item)
    return expanded


def context_for_child(parent_context: dict[str, Any], key: str, child: Any) -> dict[str, Any]:
    context = dict(parent_context)
    if key not in CONTAINER_KEYS:
        if "parent_key" not in context:
            context["parent_key"] = key
        if "category_hint" not in context and isinstance(child, list):
            context["category_hint"] = key
        if "parent_id" not in context and isinstance(child, dict):
            context["parent_id"] = key
    return context


def iter_raw_records(value: Any, context: dict[str, Any] | None = None) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    context = context or {}
    if looks_like_conversation_record(value):
        yield value, context
        return
    if looks_like_qa_record(value):
        yield value, context
        return
    if looks_like_loader_ready_text_record(value):
        for record in split_loader_ready_text(value):
            yield record, context
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"metadata", "info", "config", "features", "citation", "license"}:
                continue
            yield from iter_raw_records(child, context_for_child(context, key, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_context = dict(context)
            child_context.setdefault("list_index", index)
            yield from iter_raw_records(item, child_context)


def iter_source_records(path: Path) -> Iterator[SourceRecord]:
    for index, (record, context) in enumerate(iter_raw_records(load_json(path))):
        yield SourceRecord(path, index, context, record)


def json_sources(paths: Iterable[Path]) -> list[JsonSource]:
    sources = []
    for path in paths:
        rows = sum(1 for _ in iter_source_records(path))
        if rows == 0:
            print(f"[warning] no QA/media records detected in {path}")
        else:
            print(f"[ok] {path.name} records={rows:,}")
        sources.append(JsonSource(path, rows))
    return sources


def flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, dict):
        values: list[str] = []
        for key in (
            "path",
            "file",
            "filename",
            "name",
            "id",
            "clip",
            "video",
            "image",
            "audio",
            "url",
            *METADATA_MEDIA_KEYS,
        ):
            values.extend(flatten_values(value.get(key)))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(flatten_values(item))
        return values
    if hasattr(value, "tolist"):
        return flatten_values(value.tolist())
    return []


def media_references(record: dict[str, Any]) -> dict[str, list[str]]:
    refs = {"images": [], "videos": [], "audios": [], "generic": []}
    for key in IMAGE_KEYS:
        refs["images"].extend(flatten_values(record.get(key)))
    for key in VIDEO_KEYS:
        refs["videos"].extend(flatten_values(record.get(key)))
    for key in AUDIO_KEYS:
        refs["audios"].extend(flatten_values(record.get(key)))
    for key in GENERIC_MEDIA_KEYS:
        refs["generic"].extend(flatten_values(record.get(key)))
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        refs["generic"].extend(flatten_values(metadata.get("video_location")))
        refs["generic"].extend(flatten_values(metadata.get("image_location")))
        refs["generic"].extend(flatten_values(metadata.get("audio_location")))

    for media_key, values in refs.items():
        deduped = []
        seen = set()
        for value in values:
            if value and value not in seen:
                deduped.append(value)
                seen.add(value)
        refs[media_key] = deduped
    return refs


def suffix_to_media_key(suffix: str) -> str | None:
    suffix = suffix.lower()
    for media_key, suffixes in MEDIA_SUFFIXES.items():
        if suffix in suffixes:
            return media_key
    return None


def add_media_path(
    path: Path,
    clips_dir: Path,
    index: dict[str, dict[str, Path]],
    counts: dict[str, int],
) -> None:
    media_key = suffix_to_media_key(path.suffix)
    if media_key is None:
        return
    resolved = path.resolve()
    try:
        relative = path.relative_to(clips_dir)
    except ValueError:
        relative = Path(path.name)
    keys = {
        path.name,
        path.stem,
        str(relative).replace(os.sep, "/"),
        str(relative.with_suffix("")).replace(os.sep, "/"),
    }
    for key in keys:
        index[media_key].setdefault(key, resolved)
    counts[media_key] += 1


def archive_stem(path: Path) -> str:
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def is_tar_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".tar.gz", ".tgz", ".tar"))


def safe_member_target(destination: Path, member_name: str) -> Path:
    target = (destination / member_name).resolve()
    destination = destination.resolve()
    try:
        target.relative_to(destination)
    except ValueError:
        raise ValueError(f"Unsafe path in tar archive: {member_name}") from None
    return target


def extract_media_archive(archive_path: Path, destination_root: Path) -> list[Path]:
    archive_destination = destination_root / archive_stem(archive_path)
    extracted: list[Path] = []
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            if suffix_to_media_key(suffix) is None:
                continue
            target = safe_member_target(archive_destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == member.size:
                extracted.append(target)
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            temporary = target.with_name(f"{target.name}.tmp")
            try:
                with source, temporary.open("wb") as stream:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            extracted.append(target)
    return extracted


def build_media_index(clips_dir: Path, archive_extract_dir: Path | None) -> dict[str, dict[str, Path]]:
    if not clips_dir.is_dir():
        raise FileNotFoundError(
            f"Clips directory does not exist: {clips_dir}. README says clips must be extracted before use."
        )

    index = {"images": {}, "videos": {}, "audios": {}}
    counts = {"images": 0, "videos": 0, "audios": 0}
    archives: list[Path] = []
    for path in clips_dir.rglob("*"):
        if not path.is_file():
            continue
        if is_tar_archive(path):
            archives.append(path)
            continue
        add_media_path(path, clips_dir, index, counts)

    if archives and archive_extract_dir is not None:
        archive_extract_dir.mkdir(parents=True, exist_ok=True)
        for archive_path in sorted(archives):
            print(f"[extract] {archive_path} -> {archive_extract_dir / archive_stem(archive_path)}")
            for extracted_path in extract_media_archive(archive_path, archive_extract_dir):
                add_media_path(extracted_path, archive_extract_dir, index, counts)
    elif archives:
        print(f"[warning] found {len(archives):,} clip archives but archive extraction is disabled")

    total = sum(counts.values())
    if total == 0:
        raise FileNotFoundError(
            f"No image/video/audio files found below clips directory: {clips_dir}. "
            "If clips are stored as clips_part_*.tar.gz, keep the default archive extraction "
            "or pass --archive-extract-dir."
        )
    print(
        f"[ok] indexed clips images={counts['images']:,} videos={counts['videos']:,} "
        f"audios={counts['audios']:,} archives={len(archives):,} from {clips_dir}"
    )
    return index


def candidate_ref_keys(ref: str, allowed_suffixes: set[str]) -> list[str]:
    normalized = ref.replace("\\", "/").strip()
    path = Path(normalized)
    keys = [normalized, path.name, path.stem]
    if path.suffix:
        keys.append(str(path.with_suffix("")).replace("\\", "/"))
    else:
        for suffix in sorted(allowed_suffixes):
            keys.append(f"{normalized}{suffix}")
            keys.append(f"{path.name}{suffix}")
    return [key for key in dict.fromkeys(keys) if key]


def resolve_ref_direct(ref: str, input_dir: Path, clips_dir: Path, allowed_suffixes: set[str]) -> Path | None:
    path = Path(ref)
    candidates = [path] if path.is_absolute() else [
        input_dir / path,
        clips_dir / path,
        input_dir / "clips" / path,
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in allowed_suffixes:
            return candidate.resolve()
        if not candidate.suffix:
            for suffix in allowed_suffixes:
                suffixed = candidate.with_suffix(suffix)
                if suffixed.is_file():
                    return suffixed.resolve()
    return None


def resolve_media_ref(
    ref: str,
    preferred_key: str | None,
    input_dir: Path,
    clips_dir: Path,
    index: dict[str, dict[str, Path]],
) -> tuple[str, Path] | None:
    media_keys = [preferred_key] if preferred_key in MEDIA_SUFFIXES else ["videos", "images", "audios"]
    if preferred_key is None:
        suffix_key = suffix_to_media_key(Path(ref).suffix)
        if suffix_key is not None:
            media_keys = [suffix_key]

    for media_key in media_keys:
        allowed_suffixes = MEDIA_SUFFIXES[media_key]
        direct = resolve_ref_direct(ref, input_dir, clips_dir, allowed_suffixes)
        if direct is not None:
            return media_key, direct
        for key in candidate_ref_keys(ref, allowed_suffixes):
            found = index[media_key].get(key)
            if found is not None:
                return media_key, found
    return None


def resolve_media_paths(
    refs: dict[str, list[str]],
    input_dir: Path,
    clips_dir: Path,
    index: dict[str, dict[str, Path]],
) -> dict[str, list[Path]]:
    paths = {"images": [], "videos": [], "audios": []}
    seen: set[Path] = set()

    for media_key in ("images", "videos", "audios"):
        for ref in refs[media_key]:
            resolved = resolve_media_ref(ref, media_key, input_dir, clips_dir, index)
            if resolved is None:
                continue
            actual_key, path = resolved
            if path not in seen:
                paths[actual_key].append(path)
                seen.add(path)

    for ref in refs["generic"]:
        resolved = resolve_media_ref(ref, None, input_dir, clips_dir, index)
        if resolved is None:
            continue
        media_key, path = resolved
        if path not in seen:
            paths[media_key].append(path)
            seen.add(path)
    return paths


def media_token_block(media_paths: dict[str, list[Path]]) -> str:
    tokens = []
    for media_key in ("images", "videos", "audios"):
        tokens.extend(MEDIA_TOKENS[media_key] for _ in media_paths[media_key])
    return "\n".join(tokens)


def format_media_path(media_path: Path, jsonl_path: Path, relative: bool) -> str:
    if relative:
        return os.path.relpath(media_path, jsonl_path.parent).replace(os.sep, "/")
    return str(media_path.resolve())


def source_shard(path: Path) -> str:
    stem = path.stem
    match = re.match(r"robovqa_(.+)", stem)
    return match.group(1) if match else stem


def task_metadata_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return []
    task_metadata = metadata.get("task_metadata")
    if isinstance(task_metadata, dict):
        return [task_metadata]
    if isinstance(task_metadata, list):
        return [item for item in task_metadata if isinstance(item, dict)]
    return []


def first_task_metadata_value(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for item in task_metadata_items(record):
        value = first_key_value(item, keys)
        if value is not None:
            return value
    return None


def record_id(source_record: SourceRecord) -> Any:
    explicit = first_key_value(source_record.record, ID_KEYS)
    if explicit is not None:
        return explicit
    metadata_id = first_task_metadata_value(source_record.record, ("uid", "video_id", "id"))
    if metadata_id is not None:
        return f"{metadata_id}:{source_record.source_index}"
    parent_id = source_record.context.get("parent_id")
    if parent_id is not None:
        return parent_id
    return f"{source_record.source_path.stem}:{source_record.source_index}"


def record_category(source_record: SourceRecord) -> str:
    explicit = text_value(first_key_value(source_record.record, CATEGORY_KEYS))
    if explicit:
        return explicit
    task = text_value(first_task_metadata_value(source_record.record, ("task", "task_type")))
    if task:
        return task
    hint = text_value(source_record.context.get("category_hint"))
    if hint:
        return hint
    shard = source_shard(source_record.source_path)
    if shard.startswith("reasoning_"):
        return "reasoning"
    if shard == "understanding":
        return "understanding"
    return shard


def conversation_items(record: dict[str, Any]) -> list[Any]:
    value = first_key_value(record, CONVERSATION_KEYS)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return []


def conversation_role(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    role = text_value(item.get("role") or item.get("from") or item.get("speaker")).lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def conversation_content(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    return text_value(item.get("content") or item.get("value") or item.get("text"))


def clean_user_prompt(content: str) -> str:
    """Remove the CoT-format request after converting the target to answer-only."""
    return FORMAT_REQUEST_PATTERN.sub("", content).strip()


def has_reasoning_markup(content: str) -> bool:
    """Return whether an answer contains balanced or orphan reasoning tags."""
    return bool(
        MALFORMED_REASONING_TAG_PATTERN.search(content)
        or THINK_OPEN_PATTERN.search(content)
        or THINK_CLOSE_PATTERN.search(content)
        or ANSWER_OPEN_PATTERN.search(content)
        or ANSWER_CLOSE_PATTERN.search(content)
    )


def has_residual_reasoning(content: str) -> bool:
    """Detect synthetic answer-format deliberation that must not enter SFT data."""
    return bool(
        has_reasoning_markup(content)
        or RESIDUAL_REASONING_PATTERN.search(content)
    )


def has_unclosed_final_answer_tag(content: str) -> bool:
    """Return whether the last answer opening tag has no following close tag."""
    answer_opens = list(ANSWER_OPEN_PATTERN.finditer(content))
    if not answer_opens:
        return False
    return ANSWER_CLOSE_PATTERN.search(content, answer_opens[-1].end()) is None


def extract_assistant_answer(content: str) -> str:
    """Extract a final answer without retaining synthetic chain-of-thought text."""
    answer_opens = list(ANSWER_OPEN_PATTERN.finditer(content))
    if answer_opens:
        last_open = answer_opens[-1]
        answer_close = ANSWER_CLOSE_PATTERN.search(content, last_open.end())
        answer_end = answer_close.start() if answer_close else len(content)
        answer = content[last_open.end() : answer_end]
    else:
        orphan_think_closes = list(THINK_CLOSE_PATTERN.finditer(content))
        if orphan_think_closes:
            answer = content[orphan_think_closes[-1].end() :]
        else:
            answer = THINK_PATTERN.sub("", content)
    answer = THINK_PATTERN.sub("", answer)
    answer = THINK_TAG_PATTERN.sub("", answer)
    answer = ANSWER_TAG_PATTERN.sub("", answer)
    answer = " ".join(answer.split())
    if not answer:
        raise ValueError("assistant answer is empty after removing reasoning")
    if has_residual_reasoning(answer):
        raise ValueError("assistant answer contains residual reasoning")
    return answer


def compact_answer(answer: str, max_answer_chars: int) -> str:
    """Keep complete sentences from both ends of a verbose direct description."""
    if len(answer) <= max_answer_chars:
        return answer
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer)
        if sentence.strip()
    ]
    if len(sentences) <= 1:
        clipped = answer[:max_answer_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
        return clipped or answer[:max_answer_chars]

    selected: list[tuple[int, str]] = []
    left = 0
    right = len(sentences) - 1
    take_left = True
    used = 0
    while left <= right:
        index = left if take_left else right
        sentence = sentences[index]
        added = len(sentence) + (1 if selected else 0)
        if used + added <= max_answer_chars:
            selected.append((index, sentence))
            used += added
        if take_left:
            left += 1
        else:
            right -= 1
        take_left = not take_left
    if not selected:
        return answer[:max_answer_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return " ".join(sentence for _, sentence in sorted(selected))


def clean_assistant_answer(
    content: str,
    max_answer_chars: int = 512,
    overlong_policy: str = "error",
) -> str:
    answer = extract_assistant_answer(content)
    if max_answer_chars and len(answer) > max_answer_chars:
        if overlong_policy == "compact":
            return compact_answer(answer, max_answer_chars)
        raise ValueError(
            f"assistant answer exceeds --max-answer-chars: {len(answer)} > {max_answer_chars}"
        )
    return answer


def apply_reasoning_policy(
    messages: list[dict[str, str]],
    reasoning_policy: str,
    max_answer_chars: int,
    overlong_answer_policy: str,
) -> list[dict[str, str]]:
    if reasoning_policy == "preserve":
        return messages
    cleaned: list[dict[str, str]] = []
    for message in messages:
        content = message["content"]
        if message["role"] == "user":
            content = clean_user_prompt(content)
        elif message["role"] == "assistant":
            content = clean_assistant_answer(
                content, max_answer_chars, overlong_answer_policy
            )
        if not content:
            raise ValueError(f"{message['role']} message is empty after reasoning cleanup")
        cleaned.append({"role": message["role"], "content": content})
    return cleaned


def normalized_messages(
    record: dict[str, Any], media_tokens: str, args: argparse.Namespace
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in conversation_items(record):
        role = conversation_role(item)
        content = conversation_content(item)
        if role not in {"system", "user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})

    if not any(message["role"] == "user" for message in messages):
        raise ValueError("conversation has no user message")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("conversation has no assistant message")

    if media_tokens:
        first_user = next(message for message in messages if message["role"] == "user")
        existing_tokens = sum(first_user["content"].count(token) for token in MEDIA_TOKENS.values())
        if existing_tokens == 0:
            first_user["content"] = f"{media_tokens}\n{first_user['content']}".strip()
    return apply_reasoning_policy(
        messages,
        args.reasoning_policy,
        args.max_answer_chars,
        args.overlong_answer_policy,
    )


def make_conversation_record(
    source_record: SourceRecord,
    media_paths: dict[str, list[Path]],
    jsonl_path: Path,
    args: argparse.Namespace,
) -> dict:
    raw = source_record.record
    category = record_category(source_record)
    output_id = record_id(source_record)
    media_tokens = media_token_block(media_paths)
    messages = normalized_messages(raw, media_tokens, args)

    output: dict[str, Any] = {
        "id": output_id,
        "source": "robovqa",
        "source_file": source_record.source_path.name,
        "source_shard": source_shard(source_record.source_path),
        "source_index": source_record.source_index,
        "category": category,
        "messages": messages,
    }
    for media_key in ("images", "videos", "audios"):
        if media_paths[media_key]:
            output[media_key] = [
                format_media_path(path, jsonl_path, args.relative_paths) for path in media_paths[media_key]
            ]

    for media_key, token in MEDIA_TOKENS.items():
        expected = len(media_paths[media_key])
        actual = sum(message["content"].count(token) for message in messages)
        if actual != expected:
            raise ValueError(f"{token} count mismatch: content={actual}, {media_key}={expected}")
    return output


def make_record(
    source_record: SourceRecord,
    media_paths: dict[str, list[Path]],
    jsonl_path: Path,
    args: argparse.Namespace,
) -> dict:
    raw = source_record.record
    if looks_like_conversation_record(raw):
        return make_conversation_record(source_record, media_paths, jsonl_path, args)

    question = text_value(first_key_value(raw, QUESTION_KEYS))
    answer = text_value(first_key_value(raw, ANSWER_KEYS))
    if not question or not answer:
        raise ValueError("question or answer is empty")
    if args.reasoning_policy == "answer-only":
        question = clean_user_prompt(question)
        answer = clean_assistant_answer(
            answer, args.max_answer_chars, args.overlong_answer_policy
        )

    category = record_category(source_record)
    output_id = record_id(source_record)
    media_tokens = media_token_block(media_paths)
    user_content = args.user_template.format(
        media_tokens=media_tokens,
        question=question,
        category=category,
        source_file=source_record.source_path.name,
        source_shard=source_shard(source_record.source_path),
        id=output_id,
    ).strip()

    output: dict[str, Any] = {
        "id": output_id,
        "source": "robovqa",
        "source_file": source_record.source_path.name,
        "source_shard": source_shard(source_record.source_path),
        "source_index": source_record.source_index,
        "category": category,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
    }
    for media_key in ("images", "videos", "audios"):
        if media_paths[media_key]:
            output[media_key] = [
                format_media_path(path, jsonl_path, args.relative_paths) for path in media_paths[media_key]
            ]

    for media_key, token in MEDIA_TOKENS.items():
        expected = len(media_paths[media_key])
        actual = user_content.count(token)
        if actual != expected:
            raise ValueError(f"{token} count mismatch: content={actual}, {media_key}={expected}")
    return output


def open_output_jsonl(path: Path, overwrite: bool) -> TextIO:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    return temporary.open("w", encoding="utf-8", newline="\n")


def commit_output_jsonl(stream: TextIO, final_path: Path) -> None:
    temporary = Path(stream.name)
    stream.close()
    os.replace(temporary, final_path)
    print(f"[write] {final_path}")


def cleanup_output_jsonl(stream: TextIO) -> None:
    temporary = Path(stream.name)
    stream.close()
    if temporary.exists():
        temporary.unlink()


def empty_media_paths() -> dict[str, list[Path]]:
    return {"images": [], "videos": [], "audios": []}


def has_media(media_paths: dict[str, list[Path]]) -> bool:
    return any(media_paths[key] for key in ("images", "videos", "audios"))


def iter_chunks(items: Iterable[SourceRecord], chunk_size: int) -> Iterator[list[SourceRecord]]:
    chunk: list[SourceRecord] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def add_stats(target: ConversionStats, source: ConversionStats) -> None:
    target.read_rows += source.read_rows
    target.skipped_missing_media += source.skipped_missing_media
    target.skipped_invalid_text += source.skipped_invalid_text


def iter_all_source_records(sources: Iterable[JsonSource]) -> Iterator[SourceRecord]:
    for source in sources:
        yield from iter_source_records(source.path)


def process_source_record(
    source_record: SourceRecord,
    input_dir: Path,
    clips_dir: Path,
    jsonl_path: Path,
    media_index: dict[str, dict[str, Path]],
    args: argparse.Namespace,
    stats: ConversionStats,
    lines: list[str],
) -> None:
    stats.read_rows += 1
    media_paths = resolve_media_paths(
        media_references(source_record.record),
        input_dir,
        clips_dir,
        media_index,
    )
    if not has_media(media_paths):
        if args.missing_media_policy == "error":
            raise ValueError(
                f"Missing media for {source_record.source_path} record_index={source_record.source_index}"
            )
        if args.missing_media_policy == "skip":
            stats.skipped_missing_media += 1
            return
        media_paths = empty_media_paths()
    try:
        output_record = make_record(source_record, media_paths, jsonl_path, args)
    except ValueError:
        stats.skipped_invalid_text += 1
        return
    lines.append(json.dumps(output_record, ensure_ascii=False, separators=(",", ":")))
    stats.written_rows += 1


def process_source_chunk_with_context(
    source_records: list[SourceRecord],
    input_dir: Path,
    clips_dir: Path,
    jsonl_path: Path,
    media_index: dict[str, dict[str, Path]],
    args: argparse.Namespace,
) -> tuple[ConversionStats, list[str]]:
    stats = ConversionStats()
    lines: list[str] = []
    for source_record in source_records:
        process_source_record(source_record, input_dir, clips_dir, jsonl_path, media_index, args, stats, lines)
    return stats, lines


_WORKER_INPUT_DIR: Path | None = None
_WORKER_CLIPS_DIR: Path | None = None
_WORKER_JSONL_PATH: Path | None = None
_WORKER_MEDIA_INDEX: dict[str, dict[str, Path]] | None = None
_WORKER_ARGS: argparse.Namespace | None = None


def init_worker(
    input_dir: Path,
    clips_dir: Path,
    jsonl_path: Path,
    media_index: dict[str, dict[str, Path]],
    args: argparse.Namespace,
) -> None:
    global _WORKER_INPUT_DIR, _WORKER_CLIPS_DIR, _WORKER_JSONL_PATH, _WORKER_MEDIA_INDEX, _WORKER_ARGS
    _WORKER_INPUT_DIR = input_dir
    _WORKER_CLIPS_DIR = clips_dir
    _WORKER_JSONL_PATH = jsonl_path
    _WORKER_MEDIA_INDEX = media_index
    _WORKER_ARGS = args


def process_source_chunk(source_records: list[SourceRecord]) -> tuple[ConversionStats, list[str]]:
    if (
        _WORKER_INPUT_DIR is None
        or _WORKER_CLIPS_DIR is None
        or _WORKER_JSONL_PATH is None
        or _WORKER_MEDIA_INDEX is None
        or _WORKER_ARGS is None
    ):
        raise RuntimeError("Worker context was not initialized")
    return process_source_chunk_with_context(
        source_records,
        _WORKER_INPUT_DIR,
        _WORKER_CLIPS_DIR,
        _WORKER_JSONL_PATH,
        _WORKER_MEDIA_INDEX,
        _WORKER_ARGS,
    )


def write_chunk_lines(
    stream: TextIO,
    lines: list[str],
    stats: ConversionStats,
    max_samples: int | None,
) -> None:
    for line in lines:
        if max_samples is not None and stats.written_rows >= max_samples:
            break
        stream.write(line + "\n")
        stats.written_rows += 1
        if stats.written_rows % 10000 == 0:
            print(f"[convert] rows={stats.written_rows:,}")


def convert_sources(
    sources: list[JsonSource],
    input_dir: Path,
    clips_dir: Path,
    output_dir: Path,
    archive_extract_dir: Path | None,
    args: argparse.Namespace,
) -> ConversionStats:
    media_index = build_media_index(clips_dir, archive_extract_dir)
    jsonl_path = output_dir / args.output_name
    stats = ConversionStats()
    stream = open_output_jsonl(jsonl_path, args.overwrite)
    try:
        chunks = iter_chunks(iter_all_source_records(sources), args.worker_chunk_size)
        if args.num_workers > 1:
            print(f"[parallel] workers={args.num_workers} chunk_size={args.worker_chunk_size}")
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                initializer=init_worker,
                initargs=(input_dir, clips_dir, jsonl_path, media_index, args),
            ) as executor:
                for chunk_stats, lines in executor.map(process_source_chunk, chunks):
                    add_stats(stats, chunk_stats)
                    write_chunk_lines(stream, lines, stats, args.max_samples)
        else:
            for chunk in chunks:
                if args.max_samples is not None and stats.written_rows >= args.max_samples:
                    break
                chunk_stats, lines = process_source_chunk_with_context(
                    chunk,
                    input_dir,
                    clips_dir,
                    jsonl_path,
                    media_index,
                    args,
                )
                add_stats(stats, chunk_stats)
                write_chunk_lines(stream, lines, stats, args.max_samples)
        commit_output_jsonl(stream, jsonl_path)
    except BaseException:
        cleanup_output_jsonl(stream)
        raise

    print(
        f"[ok] read={stats.read_rows:,} written={stats.written_rows:,} "
        f"missing_media={stats.skipped_missing_media:,} invalid_text={stats.skipped_invalid_text:,}"
    )
    return stats


def write_report(path: Path, report: dict[str, Any], overwrite: bool) -> None:
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
    args.output_dir = args.output_dir.expanduser().resolve()
    clips_dir = (args.clips_dir or (args.input_dir / "clips")).expanduser().resolve()
    archive_extract_dir = (args.archive_extract_dir or (args.output_dir / "robovqa_clips")).expanduser().resolve()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_readme(args.input_dir)
    json_files = discover_json_files(args.input_dir, args.json_files)
    sources = json_sources(json_files)
    if not any(source.rows for source in sources):
        raise ValueError("No QA/media records were detected in the selected JSON files")

    conversion_stats: dict[str, int] = {}
    if args.skip_convert:
        build_media_index(clips_dir, archive_extract_dir)
    else:
        stats = convert_sources(sources, args.input_dir, clips_dir, args.output_dir, archive_extract_dir, args)
        conversion_stats = {
            "read_rows": stats.read_rows,
            "written_rows": stats.written_rows,
            "skipped_missing_media": stats.skipped_missing_media,
            "skipped_invalid_text": stats.skipped_invalid_text,
        }

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "clips_dir": str(clips_dir),
        "archive_extract_dir": str(archive_extract_dir),
        "json_files": [{"path": str(source.path), "detected_records": source.rows} for source in sources],
        "output_name": args.output_name,
        "num_workers": args.num_workers,
        "worker_chunk_size": args.worker_chunk_size,
        "reasoning_policy": args.reasoning_policy,
        "max_answer_chars": args.max_answer_chars,
        "overlong_answer_policy": args.overlong_answer_policy,
        "format": "ms-swift multimodal SFT",
        "schema": {
            "required": ["messages"],
            "media": ["images", "videos", "audios"],
            "metadata": ["id", "source", "source_file", "source_shard", "source_index", "category"],
        },
        "conversion": conversion_stats,
    }
    write_report(args.output_dir / "robovqa_conversion_report.json", report, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (json.JSONDecodeError, FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
