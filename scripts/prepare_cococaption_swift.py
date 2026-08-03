#!/usr/bin/env python3
"""Convert COCO Caption-style Arrow/JSON data to ms-swift multimodal JSONL.

Default input:
    /mnt/luojunkun/stage1/dataset/robovqa

Default output:
    /mnt/luojunkun/stage1/dataset_ms-swift/robovqa

The converter supports HuggingFace cache/save_to_disk Arrow files such as:
    coco_2014_caption-train.arrow
    coco_2014_caption-validation.arrow

It also supports common COCO caption JSON layouts with an annotations list and
an images list. The output is ms-swift SFT JSONL:

{"messages":[{"role":"user","content":"<image>Describe the image."},
 {"role":"assistant","content":"a dog running on grass"}],
 "images":["/abs/path/to/image.jpg"]}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


DEFAULT_INPUT_DIR = Path("/mnt/luojunkun/stage1/dataset/robovqa")
DEFAULT_OUTPUT_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/robovqa")
DEFAULT_SPLITS = ("train", "validation", "val", "test")
DATA_SUFFIXES = {".arrow", ".json", ".jsonl"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
CAPTION_KEYS = ("caption", "captions", "text", "sentence", "sentences", "answer", "answers", "label")
IMAGE_KEYS = ("image", "images", "image_path", "image_paths", "file_name", "filename", "path", "url")
IMAGE_ID_KEYS = ("image_id", "imageid", "id", "img_id")
SKIP_JSON_NAMES = {"dataset_info.json", "state.json", "license", "readme.md"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    split: str
    suffix: str


@dataclass(frozen=True)
class SourceRow:
    row: dict[str, Any]
    source_path: Path
    source_index: int
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO Caption-style Arrow/JSON data to ms-swift multimodal JSONL."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Dataset root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="ms-swift output directory.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing .arrow/.json/.jsonl files. Default: --input-dir.",
    )
    parser.add_argument(
        "--image-dirs",
        nargs="*",
        type=Path,
        default=None,
        help="Image directories. Default: common image folders below --input-dir plus --input-dir.",
    )
    parser.add_argument(
        "--source-files",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit source files. Relative paths are resolved below --input-dir.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Splits to convert. Accepted aliases include val/valid/dev.",
    )
    parser.add_argument(
        "--output-template",
        default="cococaption_{split}_sft_msswift.jsonl",
        help="Output JSONL filename template below --output-dir.",
    )
    parser.add_argument("--prompt", default="Describe the image.", help="User prompt after the <image> token.")
    parser.add_argument("--system", default=None, help="Optional system message.")
    parser.add_argument(
        "--caption-column",
        default="auto",
        help="Caption column/key, or auto to detect caption/captions/text/sentence.",
    )
    parser.add_argument(
        "--image-column",
        default="auto",
        help="Image column/key, or auto to detect image/images/image_path/file_name/path.",
    )
    parser.add_argument(
        "--image-id-column",
        default="auto",
        help="Image id column/key used for output filenames, or auto.",
    )
    parser.add_argument(
        "--caption-policy",
        choices=("first", "all"),
        default="first",
        help="How to handle rows with multiple captions.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write image paths relative to each JSONL instead of absolute paths.",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Reference source image paths directly when possible instead of copying them.",
    )
    parser.add_argument("--image-subdir", default="images", help="Image output subdirectory under --output-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL/report files.")
    parser.add_argument("--overwrite-images", action="store_true", help="Rewrite extracted/copied image files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit written rows per split for quick checks.")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker processes. Use 1 to disable multiprocessing.")
    parser.add_argument(
        "--worker-chunk-size",
        type=int,
        default=256,
        help="Rows sent to each worker task when --num-workers is greater than 1.",
    )
    parser.add_argument("--progress-every", type=int, default=50000, help="Print progress every N source rows.")
    parser.add_argument("--max-missing-image-logs", type=int, default=20, help="Missing-image examples per split.")
    parser.add_argument("--skip-convert", action="store_true", help="Only inspect files and image index.")
    return parser.parse_args()


def normalize_split(value: Any) -> str | None:
    text = str(value).strip().lower().replace("_", "-") if value is not None else ""
    if not text:
        return None
    return SPLIT_ALIASES.get(text) or SPLIT_ALIASES.get(text.replace("-", ""))


def split_from_path(path: Path) -> str:
    tokens = re.split(r"[^A-Za-z0-9]+", path.as_posix().lower())
    for token in reversed(tokens):
        split = normalize_split(token)
        if split:
            return split
    return "train"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(part for item in value if (part := clean_text(item)))
    if isinstance(value, dict):
        for key in ("caption", "text", "sentence", "answer", "value", "label", "content", "name"):
            text = clean_text(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if hasattr(value, "tolist"):
        return clean_text(value.tolist())
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


def first_key_value(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def configured_or_first(row: dict[str, Any], configured: str, candidates: Iterable[str]) -> Any:
    if configured != "auto":
        return row.get(configured)
    return first_key_value(row, candidates)


def caption_text(row: dict[str, Any], args: argparse.Namespace) -> str:
    value = configured_or_first(row, args.caption_column, CAPTION_KEYS)
    captions = [clean_text(item) for item in ensure_list(value)]
    captions = [caption for caption in captions if caption]
    if not captions:
        return ""
    if args.caption_policy == "all":
        return "; ".join(dict.fromkeys(captions))
    return captions[0]


def image_value(row: dict[str, Any], args: argparse.Namespace) -> Any:
    return configured_or_first(row, args.image_column, IMAGE_KEYS)


def image_id_value(row: dict[str, Any], args: argparse.Namespace) -> str:
    if args.image_id_column != "auto":
        return clean_text(row.get(args.image_id_column))
    return clean_text(first_key_value(row, IMAGE_ID_KEYS))


def candidate_image_dirs(input_dir: Path, configured: list[Path] | None) -> list[Path]:
    if configured is not None:
        candidates = [path if path.is_absolute() else input_dir / path for path in configured]
    else:
        candidates = [
            input_dir / "images",
            input_dir / "image",
            input_dir / "train2014",
            input_dir / "val2014",
            input_dir / "test2014",
            input_dir / "test2015",
            input_dir,
        ]
    output: list[Path] = []
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved.is_dir() and resolved not in output:
            output.append(resolved)
    return output


def media_keys_for_path(path: Path, root: Path) -> list[str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    return [
        path.name,
        path.stem,
        relative.as_posix(),
        relative.with_suffix("").as_posix(),
    ]


def build_image_index(image_dirs: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    count = 0
    for root in image_dirs:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = path.resolve()
            for key in media_keys_for_path(path, root):
                index.setdefault(key, resolved)
            count += 1
    print(f"[ok] indexed images={count:,} dirs={len(image_dirs):,}")
    return index


def discover_source_files(input_dir: Path, data_dir: Path, selected: list[Path] | None) -> list[SourceFile]:
    if selected:
        paths = [(path if path.is_absolute() else input_dir / path).expanduser().resolve() for path in selected]
    else:
        paths = sorted(
            path
            for path in data_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in DATA_SUFFIXES
            and path.name.lower() not in SKIP_JSON_NAMES
        )
    if not paths:
        raise FileNotFoundError(f"No .arrow/.json/.jsonl files found below {data_dir}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Source file does not exist: " + ", ".join(map(str, missing)))
    files = [SourceFile(path=path, split=split_from_path(path), suffix=path.suffix.lower()) for path in paths]
    for source in files[:50]:
        print(f"[data] split={source.split} {source.path}")
    if len(files) > 50:
        print(f"[data] ... {len(files) - 50:,} more")
    return files


def import_datasets():
    try:
        from datasets import Dataset, Image
    except ImportError as exc:
        raise SystemExit("datasets is required to read Arrow files. Install it with: pip install datasets") from exc
    return Dataset, Image


def resolve_arrow_image_column(column_names: list[str], args: argparse.Namespace) -> str | None:
    if args.image_column != "auto":
        return args.image_column if args.image_column in column_names else None
    for key in IMAGE_KEYS:
        if key in column_names:
            return key
    return None


def iter_arrow_rows(path: Path, split: str, args: argparse.Namespace) -> Iterator[SourceRow]:
    Dataset, Image = import_datasets()
    dataset = Dataset.from_file(str(path))
    image_column = resolve_arrow_image_column(list(dataset.column_names), args)
    if image_column is not None:
        try:
            dataset = dataset.cast_column(image_column, Image(decode=False))
        except Exception:
            pass
    for index, row in enumerate(dataset):
        if isinstance(row, dict):
            yield SourceRow(row=row, source_path=path, source_index=index, split=split)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def iter_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def iter_coco_json_rows(data: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(data, dict):
        return
    annotations = data.get("annotations")
    images = data.get("images")
    if not isinstance(annotations, list):
        return
    images_by_id = {}
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict) and "id" in image:
                images_by_id[str(image["id"])] = image
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        row = dict(annotation)
        image_info = images_by_id.get(clean_text(row.get("image_id")))
        if isinstance(image_info, dict):
            row.setdefault("file_name", image_info.get("file_name"))
            row.setdefault("image", image_info.get("file_name"))
            row.setdefault("image_info", image_info)
        yield row


def looks_like_caption_row(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    has_caption = any(key in keys for key in CAPTION_KEYS)
    has_image = any(key in keys for key in IMAGE_KEYS) or any(key in keys for key in IMAGE_ID_KEYS)
    return has_caption and has_image


def iter_caption_rows(value: Any) -> Iterator[dict[str, Any]]:
    if looks_like_caption_row(value):
        yield value
        return
    if isinstance(value, dict):
        coco_rows = list(iter_coco_json_rows(value))
        if coco_rows:
            yield from coco_rows
            return
        for child in value.values():
            yield from iter_caption_rows(child)
    elif isinstance(value, list):
        for item in value:
            yield from iter_caption_rows(item)


def iter_json_rows(path: Path, split: str) -> Iterator[SourceRow]:
    iterator = iter_jsonl_rows(path) if path.suffix.lower() == ".jsonl" else iter_caption_rows(load_json(path))
    for index, row in enumerate(iterator):
        yield SourceRow(row=row, source_path=path, source_index=index, split=split)


def iter_source_rows(files: Iterable[SourceFile], args: argparse.Namespace) -> Iterator[SourceRow]:
    for source in files:
        if source.suffix == ".arrow":
            yield from iter_arrow_rows(source.path, source.split, args)
        else:
            yield from iter_json_rows(source.path, source.split)


def image_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        data = value.get("bytes")
        if isinstance(data, memoryview):
            return data.tobytes()
        if isinstance(data, bytes):
            return data
        if data is not None:
            try:
                return bytes(data)
            except (TypeError, ValueError):
                return b""
    try:
        from PIL import Image as PilImage
    except ImportError:
        PilImage = None
    if PilImage is not None and isinstance(value, PilImage.Image):
        from io import BytesIO

        stream = BytesIO()
        image_format = value.format or "PNG"
        if image_format.upper() == "JPEG" and value.mode not in {"RGB", "L"}:
            value = value.convert("RGB")
        value.save(stream, format=image_format)
        return stream.getvalue()
    return b""


def image_path_text(value: Any) -> str:
    if isinstance(value, dict):
        return clean_text(value.get("path") or value.get("file_name") or value.get("filename"))
    if isinstance(value, (str, Path)):
        return str(value)
    return ""


def infer_suffix(path_text: str, data: bytes, fallback: str = ".jpg") -> str:
    suffix = Path(path_text).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return ".jpg" if suffix == ".jpeg" else suffix
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if data.startswith(b"BM"):
        return ".bmp"
    return fallback


def direct_image_candidates(path_text: str, input_dir: Path, image_dirs: list[Path]) -> list[Path]:
    if not path_text:
        return []
    path = Path(path_text)
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute():
        candidates.append(input_dir / path)
        for image_dir in image_dirs:
            candidates.append(image_dir / path)
            candidates.append(image_dir / path.name)
    return candidates


def resolve_direct_image(path_text: str, input_dir: Path, image_dirs: list[Path]) -> Path | None:
    for candidate in direct_image_candidates(path_text, input_dir, image_dirs):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate.resolve()
    return None


def target_image_path(source: SourceRow, args: argparse.Namespace, data: bytes, path_text: str, source_path: Path | None) -> Path:
    suffix = infer_suffix(path_text, data, source_path.suffix if source_path else ".jpg")
    filename = Path(path_text).name if Path(path_text).suffix.lower() in IMAGE_SUFFIXES else ""
    if not filename and source_path is not None:
        filename = source_path.name
    if not filename:
        image_id = image_id_value(source.row, args)
        stem = image_id or f"{source.source_path.stem}_{source.source_index}"
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "image"
        filename = f"{stem}{suffix}"
    return (args.output_dir / args.image_subdir / source.split / filename).resolve()


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


def resolve_indexed_image(row: dict[str, Any], path_text: str, index: dict[str, Path], args: argparse.Namespace) -> Path | None:
    keys = []
    if path_text:
        path = Path(path_text.replace("\\", "/"))
        keys.extend([path_text.replace("\\", "/"), path.name, path.stem])
    image_id = image_id_value(row, args)
    if image_id:
        keys.append(image_id)
        for suffix in sorted(IMAGE_SUFFIXES):
            keys.append(f"{image_id}{suffix}")
    for key in dict.fromkeys(keys):
        found = index.get(key)
        if found is not None:
            return found
    return None


def prepare_image_path(
    source: SourceRow,
    args: argparse.Namespace,
    image_dirs: list[Path],
    image_index: dict[str, Path],
    stats: Counter,
) -> Path | None:
    value = image_value(source.row, args)
    data = image_bytes(value)
    path_text = image_path_text(value)
    if not path_text:
        path_text = clean_text(first_key_value(source.row, ("file_name", "filename", "image_path", "path")))

    source_path = resolve_direct_image(path_text, args.input_dir, image_dirs)
    if source_path is None:
        source_path = resolve_indexed_image(source.row, path_text, image_index, args)

    if source_path is not None and args.no_copy_images and not data:
        stats["images_referenced"] += 1
        return source_path

    target = target_image_path(source, args, data, path_text, source_path)
    if data:
        if args.overwrite_images or not target.exists():
            write_bytes_atomically(target, data)
            stats["images_written"] += 1
        else:
            stats["images_reused"] += 1
        return target

    if source_path is not None:
        if args.no_copy_images:
            stats["images_referenced"] += 1
            return source_path
        if args.overwrite_images or not target.exists():
            copy_file_atomically(source_path, target)
            stats["images_copied"] += 1
        else:
            stats["images_reused"] += 1
        return target

    if target.exists():
        stats["images_reused"] += 1
        return target

    stats["missing_images"] += 1
    return None


def format_image_path(path: Path, jsonl_path: Path, relative: bool) -> str:
    if relative:
        return os.path.relpath(path, jsonl_path.parent).replace(os.sep, "/")
    return str(path.resolve())


def build_record(source: SourceRow, image_path: Path, jsonl_path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    caption = caption_text(source.row, args)
    if not caption:
        return None
    user_content = f"<image>{args.prompt}".strip()
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.extend([
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": caption},
    ])
    return {
        "messages": messages,
        "images": [format_image_path(image_path, jsonl_path, args.relative_paths)],
    }


def empty_result() -> dict[str, Any]:
    return {"stats": Counter(), "missing_image_examples": [], "lines": []}


def process_source_row(
    source: SourceRow,
    jsonl_path: Path,
    args: argparse.Namespace,
    image_dirs: list[Path],
    image_index: dict[str, Path],
    result: dict[str, Any],
) -> None:
    result["stats"]["seen"] += 1
    image_path = prepare_image_path(source, args, image_dirs, image_index, result["stats"])
    if image_path is None:
        if len(result["missing_image_examples"]) < args.max_missing_image_logs:
            result["missing_image_examples"].append(
                {
                    "source_file": str(source.source_path),
                    "source_index": source.source_index,
                    "image_id": image_id_value(source.row, args),
                    "image": image_path_text(image_value(source.row, args)),
                }
            )
        return
    record = build_record(source, image_path, jsonl_path, args)
    if record is None:
        result["stats"]["skipped_rows"] += 1
        return
    result["lines"].append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def process_chunk_with_context(
    sources: list[SourceRow],
    jsonl_path: Path,
    args: argparse.Namespace,
    image_dirs: list[Path],
    image_index: dict[str, Path],
) -> dict[str, Any]:
    result = empty_result()
    for source in sources:
        process_source_row(source, jsonl_path, args, image_dirs, image_index, result)
    result["stats"] = dict(result["stats"])
    return result


_WORKER_JSONL_PATH: Path | None = None
_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_IMAGE_DIRS: list[Path] | None = None
_WORKER_IMAGE_INDEX: dict[str, Path] | None = None


def init_worker(
    jsonl_path: Path,
    args: argparse.Namespace,
    image_dirs: list[Path],
    image_index: dict[str, Path],
) -> None:
    global _WORKER_JSONL_PATH, _WORKER_ARGS, _WORKER_IMAGE_DIRS, _WORKER_IMAGE_INDEX
    _WORKER_JSONL_PATH = jsonl_path
    _WORKER_ARGS = args
    _WORKER_IMAGE_DIRS = image_dirs
    _WORKER_IMAGE_INDEX = image_index


def process_chunk(sources: list[SourceRow]) -> dict[str, Any]:
    if _WORKER_JSONL_PATH is None or _WORKER_ARGS is None or _WORKER_IMAGE_DIRS is None or _WORKER_IMAGE_INDEX is None:
        raise RuntimeError("Worker context was not initialized")
    return process_chunk_with_context(sources, _WORKER_JSONL_PATH, _WORKER_ARGS, _WORKER_IMAGE_DIRS, _WORKER_IMAGE_INDEX)


def iter_chunks(items: Iterable[SourceRow], chunk_size: int) -> Iterator[list[SourceRow]]:
    chunk: list[SourceRow] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def merge_missing_examples(target: list[dict[str, Any]], incoming: list[dict[str, Any]], limit: int) -> None:
    if len(target) >= limit:
        return
    target.extend(incoming[: limit - len(target)])


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


def merge_chunk_result(
    result: dict[str, Any],
    stream: TextIO,
    stats: Counter,
    missing_image_examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> bool:
    stats.update(result["stats"])
    merge_missing_examples(missing_image_examples, result["missing_image_examples"], args.max_missing_image_logs)
    for line in result["lines"]:
        if args.limit is not None and stats["written"] >= args.limit:
            return True
        stream.write(line + "\n")
        stats["written"] += 1
    return args.limit is not None and stats["written"] >= args.limit


def convert_split(
    split: str,
    sources: list[SourceFile],
    args: argparse.Namespace,
    image_dirs: list[Path],
    image_index: dict[str, Path],
) -> dict[str, Any]:
    jsonl_path = args.output_dir / args.output_template.format(split=split)
    stats: Counter = Counter()
    missing_image_examples: list[dict[str, Any]] = []
    stream = open_output(jsonl_path, args.overwrite)
    next_progress = args.progress_every if args.progress_every > 0 else 0

    try:
        selected_rows = (row for row in iter_source_rows(sources, args) if row.split == split)
        row_chunks = iter_chunks(selected_rows, args.worker_chunk_size)
        if args.num_workers > 1:
            print(f"[parallel] split={split} workers={args.num_workers} chunk_size={args.worker_chunk_size}")
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                initializer=init_worker,
                initargs=(jsonl_path, args, image_dirs, image_index),
            ) as executor:
                for result in executor.map(process_chunk, row_chunks):
                    done = merge_chunk_result(result, stream, stats, missing_image_examples, args)
                    while next_progress and stats["seen"] >= next_progress:
                        print(f"[progress] split={split} seen={stats['seen']:,} written={stats['written']:,}")
                        next_progress += args.progress_every
                    if done:
                        break
        else:
            for chunk in row_chunks:
                result = process_chunk_with_context(chunk, jsonl_path, args, image_dirs, image_index)
                done = merge_chunk_result(result, stream, stats, missing_image_examples, args)
                while next_progress and stats["seen"] >= next_progress:
                    print(f"[progress] split={split} seen={stats['seen']:,} written={stats['written']:,}")
                    next_progress += args.progress_every
                if done:
                    break
        commit_output(stream, jsonl_path)
    except BaseException:
        cleanup_output(stream)
        raise

    return {
        "split": split,
        "output": str(jsonl_path),
        "source_files": [str(source.path) for source in sources if source.split == split],
        "stats": dict(stats),
        "missing_image_examples": missing_image_examples,
    }


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


def validate_args(args: argparse.Namespace) -> None:
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_dir = (args.data_dir or args.input_dir).expanduser().resolve()
    args.splits = [normalize_split(split) for split in args.splits]
    args.splits = [split for split in args.splits if split]
    if not args.splits:
        raise SystemExit("No valid splits selected")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    if args.num_workers <= 0:
        raise SystemExit("--num-workers must be greater than zero")
    if args.worker_chunk_size <= 0:
        raise SystemExit("--worker-chunk-size must be greater than zero")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be greater than or equal to zero")
    if args.max_missing_image_logs < 0:
        raise SystemExit("--max-missing-image-logs must be greater than or equal to zero")
    if args.limit is not None and args.num_workers > 1:
        print("[warning] --limit is set; falling back to --num-workers 1 for exact per-split limits")
        args.num_workers = 1
    if args.limit is not None:
        args.worker_chunk_size = 1


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = candidate_image_dirs(args.input_dir, args.image_dirs)
    sources = discover_source_files(args.input_dir, args.data_dir, args.source_files)
    selected_splits = set(args.splits)
    sources = [source for source in sources if source.split in selected_splits]
    if not sources:
        raise FileNotFoundError(f"No source files matched selected splits: {sorted(selected_splits)}")

    image_index = build_image_index(image_dirs)
    report: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "image_dirs": [str(path) for path in image_dirs],
        "format": "ms-swift multimodal SFT JSONL",
        "prompt": args.prompt,
        "num_workers": args.num_workers,
        "worker_chunk_size": args.worker_chunk_size,
        "splits": {},
    }

    if not args.skip_convert:
        for split in args.splits:
            split_sources = [source for source in sources if source.split == split]
            if not split_sources:
                print(f"[warning] no source files for split={split}")
                continue
            split_result = convert_split(split, split_sources, args, image_dirs, image_index)
            report["splits"][split] = split_result
            stats = split_result["stats"]
            print(
                f"[ok] split={split} seen={stats.get('seen', 0):,} "
                f"written={stats.get('written', 0):,} missing_images={stats.get('missing_images', 0):,}"
            )
    else:
        for split in args.splits:
            report["splits"][split] = {
                "source_files": [str(source.path) for source in sources if source.split == split],
            }

    write_report(args.output_dir / "cococaption_conversion_report.json", report, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (json.JSONDecodeError, FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
