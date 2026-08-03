#!/usr/bin/env python3
"""Convert TextVQA data to ms-swift multimodal SFT JSONL.

Default input:
    /mnt/luojunkun/stage1/dataset/textvqa

Default output:
    /mnt/luojunkun/stage1/dataset_ms-swift/textvqa

The converter supports the original TextVQA JSON/OCR files and Hugging Face
style data/*.parquet shards. It writes one JSONL per split plus an images/
directory, using SFT user/assistant records:

{"messages": [{"role": "user", "content": "<image>..."}, {"role": "assistant", "content": "..."}],
 "images": ["/abs/path/to/output/images/train/<image_id>.jpg"]}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


DEFAULT_INPUT_DIR = Path("/mnt/luojunkun/stage1/dataset/textvqa")
DEFAULT_OUTPUT_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/textvqa")
DEFAULT_EXPECTED_COUNTS = {"train": 34602, "validation": 5000, "test": 5734}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
DATA_SUFFIXES = {".json", ".jsonl", ".parquet"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
    "testdev": "test",
    "test-dev": "test",
    "test_std": "test",
    "test-std": "test",
}
REQUIRED_FIELDS = {
    "image_id",
    "question_id",
    "question",
    "question_tokens",
    "image",
    "image_width",
    "image_height",
    "answers",
}


@dataclass(frozen=True)
class SourceRow:
    row: dict[str, Any]
    source_path: Path
    source_index: int
    split_hint: str | None


@dataclass(frozen=True)
class ZipMember:
    zip_path: Path
    member_name: str
    suffix: str


@dataclass
class MediaIndex:
    files: dict[str, Path]
    zip_members: dict[str, ZipMember]
    image_files: int = 0
    zip_image_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local TextVQA data into ms-swift multimodal SFT JSONL."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="TextVQA dataset root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="ms-swift output directory.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing shards. Default: <input-dir>/data if present, otherwise <input-dir>.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to convert. Accepted aliases include val/valid/test-dev.",
    )
    parser.add_argument(
        "--output-template",
        default="textvqa_{split}_sft_msswift.jsonl",
        help="Output JSONL filename template below --output-dir.",
    )
    parser.add_argument(
        "--pretrain-template",
        default="<image>\nQuestion: {question}\nAnswer: {answer}",
        help=(
            "Template for rows with answers. Available fields: question, answer, "
            "answers, ocr_tokens, image_id, question_id, split."
        ),
    )
    parser.add_argument(
        "--unanswered-template",
        default="<image>\nQuestion: {question}",
        help="Deprecated compatibility option. SFT output skips rows without answers.",
    )
    parser.add_argument(
        "--sft-question-template",
        default="<image>{question}",
        help="User prompt template used for SFT rows.",
    )
    parser.add_argument(
        "--answer-policy",
        choices=("majority", "first", "all"),
        default="majority",
        help="How to select the assistant answer text when multiple answers are available.",
    )
    parser.add_argument(
        "--skip-unanswered",
        action="store_true",
        default=True,
        help="Skip rows with no non-empty answer. Enabled by default for SFT output.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write image paths relative to each JSONL instead of absolute paths.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Reference source image files directly when possible instead of creating output images/.",
    )
    parser.add_argument(
        "--image-subdir",
        default="images",
        help="Image output subdirectory under --output-dir when images are copied/extracted.",
    )
    parser.add_argument(
        "--no-zip-images",
        action="store_true",
        help="Do not index or extract image files from zip archives.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL/report files.")
    parser.add_argument("--overwrite-images", action="store_true", help="Rewrite copied/extracted image files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit written rows per split for quick checks.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Parquet read batch size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Worker processes for row/image conversion. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--worker-chunk-size",
        type=int,
        default=256,
        help="Rows sent to each worker task when --num-workers is greater than 1.",
    )
    parser.add_argument("--count-tolerance", type=float, default=0.02, help="Allowed count drift ratio for warnings.")
    parser.add_argument("--strict-counts", action="store_true", help="Fail if split counts are outside tolerance.")
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Only inspect schema, data files, media, and counts; do not write JSONL.",
    )
    return parser.parse_args()


def normalize_split(value: Any) -> str | None:
    text = str(value).strip().lower().replace("_", "-") if value is not None else ""
    if not text:
        return None
    return SPLIT_ALIASES.get(text) or SPLIT_ALIASES.get(text.replace("-", ""))


def split_from_path(path: Path) -> str | None:
    tokens = re.split(r"[^A-Za-z0-9]+", path.as_posix().lower())
    for token in reversed(tokens):
        split = normalize_split(token)
        if split:
            return split
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(part for item in value if (part := clean_text(item)))
    if isinstance(value, dict):
        for key in ("answer", "text", "value", "label", "word", "content", "name"):
            text = clean_text(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
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


def text_list(value: Any) -> list[str]:
    output = []
    for item in ensure_list(value):
        text = clean_text(item)
        if text:
            output.append(text)
    return output


def selected_answer(answers: list[str], policy: str) -> str:
    non_empty = [answer for answer in answers if answer]
    if not non_empty:
        return ""
    if policy == "first":
        return non_empty[0]
    if policy == "all":
        deduped = list(dict.fromkeys(non_empty))
        return "; ".join(deduped)
    counts = Counter(non_empty)
    return counts.most_common(1)[0][0]


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items() if key != "bytes"}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "as_py"):
        return json_safe(value.as_py())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def read_text_limited(path: Path, max_lines: int) -> list[str]:
    if not path.is_file():
        return []
    lines = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for _, line in zip(range(max_lines), stream):
            lines.append(line.rstrip("\n"))
    print(f"[read] {path} first_lines={len(lines)}")
    return lines


def load_dataset_infos(input_dir: Path) -> dict[str, Any] | None:
    path = input_dir / "dataset_infos.json"
    if not path.is_file():
        print(f"[warning] dataset_infos.json not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    print(f"[read] {path}")
    return info


def schema_from_dataset_infos(info: dict[str, Any] | None) -> dict[str, Any]:
    if not info:
        return {}
    schemas: dict[str, Any] = {}
    for config_name, config in info.items():
        features = config.get("features", {}) if isinstance(config, dict) else {}
        splits = config.get("splits", {}) if isinstance(config, dict) else {}
        schemas[config_name] = {
            "features": sorted(features) if isinstance(features, dict) else features,
            "splits": splits,
        }
    return schemas


def discover_data_files(data_dir: Path) -> list[Path]:
    files = sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in DATA_SUFFIXES
    )
    data_files = [path for path in files if path.name not in {"dataset_infos.json"} and path.name.lower() != "readme.md"]
    print(f"[ok] data files={len(data_files):,} under {data_dir}")
    for path in data_files[:30]:
        print(f"[data] {path}")
    if len(data_files) > 30:
        print(f"[data] ... {len(data_files) - 30:,} more")
    return data_files


def is_ocr_file(path: Path) -> bool:
    name = path.name.lower()
    return "ocr" in name or "rosetta" in name


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def iter_json_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
        return

    data = load_json(path)
    if isinstance(data, dict):
        payload = data.get("data", data.get("annotations", data.get("rows", data.get("samples", data))))
    else:
        payload = data
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("image_id", key)
                yield row
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def iter_parquet_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet shards requires pyarrow. Install pyarrow or convert from TextVQA JSON files."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        columns = batch.column_names
        values = batch.to_pydict()
        for row_index in range(batch.num_rows):
            yield {column: values[column][row_index] for column in columns}


def iter_source_rows(data_files: Iterable[Path], batch_size: int) -> Iterator[SourceRow]:
    for path in data_files:
        if is_ocr_file(path):
            continue
        split_hint = split_from_path(path)
        iterator = iter_parquet_rows(path, batch_size) if path.suffix.lower() == ".parquet" else iter_json_rows(path)
        for index, row in enumerate(iterator):
            yield SourceRow(row=row, source_path=path, source_index=index, split_hint=split_hint)


def load_ocr_index(data_files: Iterable[Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in data_files:
        if path.suffix.lower() not in {".json", ".jsonl"} or not is_ocr_file(path):
            continue
        print(f"[read] OCR {path}")
        for row in iter_json_rows(path):
            image_id = clean_text(row.get("image_id"))
            if image_id:
                index[image_id] = {
                    "ocr_tokens": text_list(row.get("ocr_tokens")),
                    "ocr_info": json_safe(row.get("ocr_info", [])),
                    "source": str(path),
                }
    print(f"[ok] OCR images={len(index):,}")
    return index


def media_keys_for_path(path: Path, root: Path) -> list[str]:
    relative = path.relative_to(root) if path.is_relative_to(root) else path
    normalized_relative = relative.as_posix()
    without_suffix = relative.with_suffix("").as_posix()
    return list(
        dict.fromkeys(
            [
                path.name,
                path.stem,
                normalized_relative,
                without_suffix,
            ]
        )
    )


def index_zip_members(zip_path: Path, root: Path, media_index: MediaIndex) -> None:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                suffix = Path(member.filename).suffix.lower()
                if suffix not in IMAGE_SUFFIXES:
                    continue
                member_path = Path(member.filename)
                zip_member = ZipMember(zip_path=zip_path.resolve(), member_name=member.filename, suffix=suffix)
                keys = [
                    member_path.name,
                    member_path.stem,
                    member.filename.replace("\\", "/"),
                    str(Path(zip_path.relative_to(root)) / member_path).replace("\\", "/")
                    if zip_path.is_relative_to(root)
                    else member.filename.replace("\\", "/"),
                ]
                for key in dict.fromkeys(keys):
                    media_index.zip_members.setdefault(key, zip_member)
                media_index.zip_image_files += 1
    except zipfile.BadZipFile:
        print(f"[warning] bad zip skipped: {zip_path}")


def build_media_index(input_dir: Path, include_zips: bool) -> MediaIndex:
    media_index = MediaIndex(files={}, zip_members={})
    zip_paths = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            resolved = path.resolve()
            for key in media_keys_for_path(path, input_dir):
                media_index.files.setdefault(key, resolved)
            media_index.image_files += 1
        elif include_zips and suffix == ".zip":
            zip_paths.append(path)

    for zip_path in sorted(zip_paths):
        print(f"[zip] indexing {zip_path}")
        index_zip_members(zip_path, input_dir, media_index)

    print(
        f"[ok] media index image_files={media_index.image_files:,} "
        f"zip_image_files={media_index.zip_image_files:,} zips={len(zip_paths):,}"
    )
    return media_index


def image_payload(row: dict[str, Any]) -> dict[str, Any]:
    image = row.get("image")
    if isinstance(image, dict):
        return image
    return {}


def guess_image_suffix(data: bytes | None, fallback: str = ".jpg") -> str:
    if not data:
        return fallback
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith(b"GIF8"):
        return ".gif"
    return fallback


def candidate_image_keys(row: dict[str, Any]) -> list[str]:
    image_id = clean_text(row.get("image_id"))
    image = row.get("image")
    payload = image_payload(row)
    path_values = [
        row.get("image_path"),
        row.get("image_file"),
        row.get("file_name"),
        row.get("filename"),
        row.get("path"),
        payload.get("path"),
    ]
    keys = []
    for value in path_values:
        text = clean_text(value)
        if not text:
            continue
        path = Path(text.replace("\\", "/"))
        keys.extend([text.replace("\\", "/"), path.name, path.stem])
    if isinstance(image, str):
        path = Path(image.replace("\\", "/"))
        keys.extend([image.replace("\\", "/"), path.name, path.stem])
    if image_id:
        keys.append(image_id)
        for suffix in sorted(IMAGE_SUFFIXES):
            keys.append(f"{image_id}{suffix}")
    return [key for key in dict.fromkeys(keys) if key]


def target_image_path(row: dict[str, Any], split: str, output_dir: Path, image_subdir: str, suffix: str) -> Path:
    image_id = clean_text(row.get("image_id"))
    question_id = clean_text(row.get("question_id"))
    stem = image_id or question_id or "image"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "image"
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return output_dir / image_subdir / split / f"{stem}{suffix.lower()}"


def temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        return Path(stream.name)


def replace_file_atomically(source: Path, target: Path) -> None:
    try:
        os.replace(source, target)
    finally:
        if source.exists():
            source.unlink()


def write_bytes_atomically(target: Path, data: bytes) -> None:
    temporary = temporary_sibling(target)
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
        replace_file_atomically(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def copy_file_atomically(source: Path, target: Path) -> None:
    temporary = temporary_sibling(target)
    try:
        shutil.copy2(source, temporary)
        replace_file_atomically(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def copy_zip_member_atomically(member: ZipMember, target: Path) -> None:
    temporary = temporary_sibling(target)
    try:
        with zipfile.ZipFile(member.zip_path) as archive, archive.open(member.member_name) as source:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream)
        replace_file_atomically(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_path_from_payload(row: dict[str, Any], input_dir: Path) -> Path | None:
    payload = image_payload(row)
    path_value = clean_text(payload.get("path"))
    if not path_value and isinstance(row.get("image"), str):
        path_value = clean_text(row.get("image"))
    if not path_value:
        return None
    path = Path(path_value)
    candidates = [path] if path.is_absolute() else [input_dir / path, input_dir / "data" / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def write_image_bytes(row: dict[str, Any], split: str, args: argparse.Namespace) -> Path | None:
    payload = image_payload(row)
    data = payload.get("bytes")
    if data is None:
        return None
    if isinstance(data, memoryview):
        data = data.tobytes()
    if not isinstance(data, bytes):
        return None
    suffix = guess_image_suffix(data)
    target = target_image_path(row, split, args.output_dir, args.image_subdir, suffix)
    if args.overwrite_images or not target.exists():
        write_bytes_atomically(target, data)
    return target.resolve()


def resolve_indexed_image(
    row: dict[str, Any],
    split: str,
    media_index: MediaIndex,
    args: argparse.Namespace,
) -> tuple[Path | None, str | None]:
    for key in candidate_image_keys(row):
        path = media_index.files.get(key)
        if path is not None:
            if args.no_copy_images:
                return path.resolve(), None
            target = target_image_path(row, split, args.output_dir, args.image_subdir, path.suffix or ".jpg")
            if args.overwrite_images or not target.exists():
                copy_file_atomically(path, target)
            return target.resolve(), None

        member = media_index.zip_members.get(key)
        if member is not None:
            target = target_image_path(row, split, args.output_dir, args.image_subdir, member.suffix)
            if args.overwrite_images or not target.exists():
                copy_zip_member_atomically(member, target)
            return target.resolve(), None
    return None, "not_found"


def resolve_image_path(
    source: SourceRow,
    split: str,
    media_index: MediaIndex,
    args: argparse.Namespace,
) -> tuple[Path | None, str | None]:
    if not args.no_copy_images:
        path = write_image_bytes(source.row, split, args)
        if path is not None:
            return path, None

    direct = source_path_from_payload(source.row, args.input_dir)
    if direct is not None:
        if args.no_copy_images:
            return direct.resolve(), None
        target = target_image_path(source.row, split, args.output_dir, args.image_subdir, direct.suffix or ".jpg")
        if args.overwrite_images or not target.exists():
            copy_file_atomically(direct, target)
        return target.resolve(), None

    return resolve_indexed_image(source.row, split, media_index, args)


def image_is_available(row: dict[str, Any], media_index: MediaIndex, input_dir: Path) -> bool:
    payload = image_payload(row)
    if isinstance(payload.get("bytes"), (bytes, memoryview)):
        return True
    if source_path_from_payload(row, input_dir) is not None:
        return True
    return any(key in media_index.files or key in media_index.zip_members for key in candidate_image_keys(row))


def format_image_path(path: Path, jsonl_path: Path, relative: bool) -> str:
    if relative:
        return os.path.relpath(path, jsonl_path.parent).replace(os.sep, "/")
    return str(path.resolve())


def split_for_row(source: SourceRow) -> str | None:
    row_split = normalize_split(source.row.get("set_name")) or normalize_split(source.row.get("split"))
    return row_split or source.split_hint


def fields_present(row: dict[str, Any], ocr_tokens: list[str]) -> set[str]:
    present = {key for key, value in row.items() if value is not None}
    if row.get("image") is not None or candidate_image_keys(row):
        present.add("image")
    if ocr_tokens or "ocr_tokens" in row:
        present.add("ocr_tokens")
    return present


def make_record(
    source: SourceRow,
    split: str,
    image_path: Path,
    jsonl_path: Path,
    ocr_index: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    row = source.row
    image_id = clean_text(row.get("image_id"))
    question_id = row.get("question_id")
    question = clean_text(row.get("question"))
    answers = text_list(row.get("answers"))
    ocr_payload = ocr_index.get(image_id, {})
    ocr_tokens = text_list(row.get("ocr_tokens")) or text_list(ocr_payload.get("ocr_tokens"))
    answer = selected_answer(answers, args.answer_policy)

    if not image_id or question_id is None or not question:
        return None
    if not answer:
        return None

    user_content = args.sft_question_template.format(
        question=question,
        answer=answer,
        answers="; ".join(answers),
        ocr_tokens=", ".join(ocr_tokens),
        image_id=image_id,
        question_id=question_id,
        split=split,
    ).strip()

    record: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "images": [format_image_path(image_path, jsonl_path, args.relative_paths)],
    }
    if user_content.count("<image>") != len(record["images"]):
        raise ValueError(f"<image> token count mismatch for question_id={question_id}")
    return record


def empty_chunk_result() -> dict[str, Any]:
    return {
        "stats": defaultdict(Counter),
        "observed_schema": defaultdict(Counter),
        "missing_required_examples": defaultdict(list),
        "missing_image_examples": [],
        "set_name_mismatch_examples": [],
        "records": [],
        "warnings": [],
    }


def process_source_row(
    source: SourceRow,
    selected_splits: set[str],
    final_paths: dict[str, Path],
    media_index: MediaIndex,
    ocr_index: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    result: dict[str, Any],
) -> None:
    split = split_for_row(source)
    if split not in selected_splits:
        result["stats"][split or "unknown"]["skipped_split"] += 1
        return

    row = source.row
    image_id = clean_text(row.get("image_id"))
    ocr_tokens = text_list(row.get("ocr_tokens")) or text_list(ocr_index.get(image_id, {}).get("ocr_tokens"))
    present = fields_present(row, ocr_tokens)
    for field in present:
        result["observed_schema"][split][field] += 1
    missing = sorted(REQUIRED_FIELDS - present)
    if missing and len(result["missing_required_examples"][split]) < 5:
        result["missing_required_examples"][split].append(
            {
                "source_file": str(source.source_path),
                "source_index": source.source_index,
                "question_id": row.get("question_id"),
                "missing": missing,
            }
        )

    raw_set_name = normalize_split(row.get("set_name"))
    if raw_set_name and raw_set_name != split and len(result["set_name_mismatch_examples"]) < 20:
        result["set_name_mismatch_examples"].append(
            {
                "source_file": str(source.source_path),
                "source_index": source.source_index,
                "question_id": row.get("question_id"),
                "set_name": row.get("set_name"),
                "resolved_split": split,
            }
        )

    result["stats"][split]["rows_seen"] += 1

    if args.skip_convert:
        if not image_is_available(row, media_index, args.input_dir):
            result["stats"][split]["missing_image"] += 1
            if len(result["missing_image_examples"]) < 20:
                result["missing_image_examples"].append(
                    {
                        "split": split,
                        "source_file": str(source.source_path),
                        "source_index": source.source_index,
                        "image_id": image_id,
                        "question_id": row.get("question_id"),
                        "error": "not_found",
                    }
                )
            return
        answers = text_list(row.get("answers"))
        if (
            not image_id
            or row.get("question_id") is None
            or not clean_text(row.get("question"))
            or not selected_answer(answers, args.answer_policy)
        ):
            result["stats"][split]["invalid_record"] += 1
            return
        result["records"].append((split, None))
        return

    image_path, image_error = resolve_image_path(source, split, media_index, args)
    if image_path is None:
        result["stats"][split]["missing_image"] += 1
        if len(result["missing_image_examples"]) < 20:
            result["missing_image_examples"].append(
                {
                    "split": split,
                    "source_file": str(source.source_path),
                    "source_index": source.source_index,
                    "image_id": image_id,
                    "question_id": row.get("question_id"),
                    "error": image_error,
                }
            )
        return

    try:
        record = make_record(source, split, image_path, final_paths[split], ocr_index, args)
    except ValueError as exc:
        result["stats"][split]["invalid_record"] += 1
        result["warnings"].append(str(exc))
        return
    if record is None:
        result["stats"][split]["invalid_record"] += 1
        return

    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    result["records"].append((split, line))


def process_source_chunk_with_context(
    sources: list[SourceRow],
    selected_splits: set[str],
    final_paths: dict[str, Path],
    media_index: MediaIndex,
    ocr_index: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = empty_chunk_result()
    for source in sources:
        process_source_row(source, selected_splits, final_paths, media_index, ocr_index, args, result)
    result["stats"] = dict(result["stats"])
    result["observed_schema"] = dict(result["observed_schema"])
    result["missing_required_examples"] = dict(result["missing_required_examples"])
    return result


_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_OCR_INDEX: dict[str, dict[str, Any]] | None = None
_WORKER_MEDIA_INDEX: MediaIndex | None = None
_WORKER_SELECTED_SPLITS: set[str] | None = None
_WORKER_FINAL_PATHS: dict[str, Path] | None = None


def init_worker(
    args: argparse.Namespace,
    ocr_index: dict[str, dict[str, Any]],
    media_index: MediaIndex,
    selected_splits: set[str],
    final_paths: dict[str, Path],
) -> None:
    global _WORKER_ARGS, _WORKER_OCR_INDEX, _WORKER_MEDIA_INDEX, _WORKER_SELECTED_SPLITS, _WORKER_FINAL_PATHS
    _WORKER_ARGS = args
    _WORKER_OCR_INDEX = ocr_index
    _WORKER_MEDIA_INDEX = media_index
    _WORKER_SELECTED_SPLITS = selected_splits
    _WORKER_FINAL_PATHS = final_paths


def process_source_chunk(sources: list[SourceRow]) -> dict[str, Any]:
    if (
        _WORKER_ARGS is None
        or _WORKER_OCR_INDEX is None
        or _WORKER_MEDIA_INDEX is None
        or _WORKER_SELECTED_SPLITS is None
        or _WORKER_FINAL_PATHS is None
    ):
        raise RuntimeError("Worker context was not initialized")
    return process_source_chunk_with_context(
        sources,
        _WORKER_SELECTED_SPLITS,
        _WORKER_FINAL_PATHS,
        _WORKER_MEDIA_INDEX,
        _WORKER_OCR_INDEX,
        _WORKER_ARGS,
    )


def iter_chunks(items: Iterable[SourceRow], chunk_size: int) -> Iterator[list[SourceRow]]:
    chunk: list[SourceRow] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def merge_limited_examples(target: list[dict[str, Any]], incoming: list[dict[str, Any]], limit: int) -> None:
    if len(target) >= limit:
        return
    target.extend(incoming[: limit - len(target)])


def merge_chunk_result(
    result: dict[str, Any],
    stats: dict[str, Counter],
    observed_schema: dict[str, Counter],
    missing_required_examples: dict[str, list[dict[str, Any]]],
    missing_image_examples: list[dict[str, Any]],
    set_name_mismatch_examples: list[dict[str, Any]],
    outputs: dict[str, TextIO],
    args: argparse.Namespace,
) -> None:
    for split, counter in result["stats"].items():
        stats[split].update(counter)
    for split, counter in result["observed_schema"].items():
        observed_schema[split].update(counter)
    for split, examples in result["missing_required_examples"].items():
        merge_limited_examples(missing_required_examples[split], examples, 5)
    merge_limited_examples(missing_image_examples, result["missing_image_examples"], 20)
    merge_limited_examples(set_name_mismatch_examples, result["set_name_mismatch_examples"], 20)
    for warning in result["warnings"]:
        print(f"[warning] {warning}")

    for split, line in result["records"]:
        if args.limit is not None and stats[split]["written"] >= args.limit:
            stats[split]["skipped_limit"] += 1
            continue
        if not args.skip_convert and line is not None:
            outputs[split].write(line + "\n")
        stats[split]["written"] += 1
        if stats[split]["written"] % 10000 == 0:
            print(f"[convert] split={split} written={stats[split]['written']:,}")


def open_output(path: Path, overwrite: bool) -> TextIO:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    return temporary.open("w", encoding="utf-8", newline="\n")


def commit_output(stream: TextIO, final_path: Path) -> None:
    temporary = Path(stream.name)
    stream.close()
    os.replace(temporary, final_path)
    print(f"[write] {final_path}")


def cleanup_output(stream: TextIO) -> None:
    temporary = Path(stream.name)
    stream.close()
    if temporary.exists():
        temporary.unlink()


def close_outputs(outputs: dict[str, TextIO], final_paths: dict[str, Path], commit: bool) -> None:
    for split, stream in outputs.items():
        if commit:
            commit_output(stream, final_paths[split])
        else:
            cleanup_output(stream)


def validate_count(split: str, count: int, args: argparse.Namespace) -> dict[str, Any]:
    expected = DEFAULT_EXPECTED_COUNTS.get(split)
    if expected is None or args.limit is not None:
        return {"actual": count, "expected": expected, "status": "unchecked"}
    drift = abs(count - expected) / expected if expected else 0.0
    ok = drift <= args.count_tolerance
    status = "ok" if ok else "outside_tolerance"
    message = f"[count] {split} actual={count:,} expected={expected:,} drift={drift:.2%} status={status}"
    print(message)
    if args.strict_counts and not ok:
        raise ValueError(message)
    return {"actual": count, "expected": expected, "drift": drift, "status": status}


def write_report(path: Path, report: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[write] {path}")


def convert(args: argparse.Namespace) -> dict[str, Any]:
    readme_lines = read_text_limited(args.input_dir / "README.md", 200)
    dataset_info = load_dataset_infos(args.input_dir)
    dataset_info_lines = read_text_limited(args.input_dir / "dataset_infos.json", 400)

    data_dir = args.data_dir or (args.input_dir / "data" if (args.input_dir / "data").is_dir() else args.input_dir)
    data_files = discover_data_files(data_dir)
    if not data_files:
        raise FileNotFoundError(f"No data shards found below {data_dir}")

    selected_splits = {normalize_split(split) for split in args.splits}
    selected_splits.discard(None)
    if not selected_splits:
        raise ValueError("No valid splits selected")

    ocr_index = load_ocr_index(data_files)
    media_index = build_media_index(args.input_dir, include_zips=not args.no_zip_images)

    outputs: dict[str, TextIO] = {}
    final_paths: dict[str, Path] = {}
    for split in sorted(selected_splits):
        final_path = args.output_dir / args.output_template.format(split=split)
        final_paths[split] = final_path
        if not args.skip_convert:
            outputs[split] = open_output(final_path, args.overwrite)

    stats: dict[str, Counter] = defaultdict(Counter)
    observed_schema: dict[str, Counter] = defaultdict(Counter)
    missing_required_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_image_examples: list[dict[str, Any]] = []
    set_name_mismatch_examples: list[dict[str, Any]] = []

    try:
        row_chunks = iter_chunks(iter_source_rows(data_files, args.batch_size), args.worker_chunk_size)
        if args.num_workers > 1:
            print(f"[parallel] workers={args.num_workers} chunk_size={args.worker_chunk_size}")
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                initializer=init_worker,
                initargs=(args, ocr_index, media_index, selected_splits, final_paths),
            ) as executor:
                for result in executor.map(process_source_chunk, row_chunks):
                    merge_chunk_result(
                        result,
                        stats,
                        observed_schema,
                        missing_required_examples,
                        missing_image_examples,
                        set_name_mismatch_examples,
                        outputs,
                        args,
                    )
        else:
            for chunk in row_chunks:
                result = process_source_chunk_with_context(
                    chunk,
                    selected_splits,
                    final_paths,
                    media_index,
                    ocr_index,
                    args,
                )
                merge_chunk_result(
                    result,
                    stats,
                    observed_schema,
                    missing_required_examples,
                    missing_image_examples,
                    set_name_mismatch_examples,
                    outputs,
                    args,
                )

        if not args.skip_convert:
            close_outputs(outputs, final_paths, commit=True)
    except BaseException:
        if not args.skip_convert:
            close_outputs(outputs, final_paths, commit=False)
        raise

    count_checks = {split: validate_count(split, int(stats[split]["written"]), args) for split in sorted(selected_splits)}
    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "data_dir": str(data_dir),
        "data_files": [str(path) for path in data_files],
        "media": {
            "image_files": media_index.image_files,
            "zip_image_files": media_index.zip_image_files,
        },
        "readme_first_200_lines_present": bool(readme_lines),
        "dataset_infos_first_400_lines_present": bool(dataset_info_lines),
        "dataset_infos_schema": schema_from_dataset_infos(dataset_info),
        "expected_counts": DEFAULT_EXPECTED_COUNTS,
        "count_checks": count_checks,
        "stats": {split: dict(counter) for split, counter in sorted(stats.items())},
        "observed_schema": {split: sorted(counter) for split, counter in sorted(observed_schema.items())},
        "required_fields": sorted(REQUIRED_FIELDS),
        "missing_required_examples": missing_required_examples,
        "set_name_mismatch_examples": set_name_mismatch_examples,
        "missing_image_examples": missing_image_examples,
        "outputs": {split: str(path) for split, path in sorted(final_paths.items())},
        "num_workers": args.num_workers,
        "worker_chunk_size": args.worker_chunk_size,
        "format": "ms-swift multimodal SFT JSONL",
    }
    if not args.skip_convert:
        write_report(args.output_dir / "textvqa_conversion_report.json", report, args.overwrite)
    return report


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.num_workers <= 0:
        raise SystemExit("--num-workers must be greater than zero")
    if args.worker_chunk_size <= 0:
        raise SystemExit("--worker-chunk-size must be greater than zero")
    if args.limit is not None and args.num_workers > 1:
        print("[warning] --limit is set; falling back to --num-workers 1 for exact per-split limits")
        args.num_workers = 1
    if args.count_tolerance < 0:
        raise SystemExit("--count-tolerance must be greater than or equal to zero")
    if args.sft_question_template.count("<image>") != 1:
        raise SystemExit("--sft-question-template must contain exactly one '<image>' token")
    if "{question}" not in args.sft_question_template:
        raise SystemExit("--sft-question-template must contain '{question}'")
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.expanduser().resolve()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = convert(args)
    print("[summary]", json.dumps(report["count_checks"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (json.JSONDecodeError, FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
