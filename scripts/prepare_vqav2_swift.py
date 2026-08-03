#!/usr/bin/env python3
"""Convert local VQAv2 parquet shards to ms-swift multimodal JSONL.

Default input:
    /mnt/luojunkun/stage1/dataset/VQAv2

Default output:
    /mnt/luojunkun/stage1/dataset_ms-swift/VQAv2

The converter reads HuggingFace-style data/*.parquet shards and writes one
JSONL per split. Use ``--json-only --images-dir ...`` when images have already
been extracted; this avoids reading embedded image bytes and never copies or
writes media files. The default output is SFT-style multimodal data:

{"messages": [{"role": "user", "content": "<image>..."}, {"role": "assistant", "content": "..."}],
 "images": ["/abs/path/to/output/images/train/COCO_train2014_000000000009.jpg"]}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


DEFAULT_INPUT_DIR = Path("/mnt/luojunkun/stage1/dataset/VQAv2")
DEFAULT_OUTPUT_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/VQAv2")
DEFAULT_SPLITS = ("train", "validation", "testdev", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "testdev": "testdev",
    "test-dev": "testdev",
    "test_dev": "testdev",
    "test": "test",
}
COCO_IMAGE_DIR = {
    "train": "train2014",
    "validation": "val2014",
    "testdev": "test2015",
    "test": "test2015",
}
REQUIRED_FIELDS = {"question", "image_id", "question_id", "image"}
JSON_ONLY_COLUMNS = (
    "question",
    "image_id",
    "question_id",
    "answers",
    "multiple_choice_answer",
    "answer_type",
    "question_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local VQAv2 parquet shards into ms-swift multimodal JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"VQAv2 dataset root. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"ms-swift output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing parquet shards. Default: <input-dir>/data.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to convert. Accepted aliases include val and test-dev.",
    )
    parser.add_argument(
        "--mode",
        choices=("pretrain", "sft"),
        default="sft",
        help="pretrain writes assistant-only samples; sft writes user/assistant QA samples.",
    )
    parser.add_argument(
        "--answer-policy",
        choices=("multiple_choice", "majority", "first", "all"),
        default="multiple_choice",
        help="How to choose the answer text when labels are available.",
    )
    parser.add_argument(
        "--pretrain-template",
        default="<image>\nQuestion: {question}\nAnswer: {answer}",
        help=(
            "Template used for answered rows in pretrain mode. Available fields: "
            "question, answer, answers, image_id, question_id, answer_type, question_type, split."
        ),
    )
    parser.add_argument(
        "--unanswered-template",
        default="<image>\nQuestion: {question}",
        help="Template used for unlabeled rows in pretrain mode unless --skip-unanswered is set.",
    )
    parser.add_argument(
        "--sft-question-template",
        default="<image>{question}",
        help="User prompt template used in sft mode.",
    )
    parser.add_argument(
        "--skip-unanswered",
        action="store_true",
        help="Skip rows without answers. Useful if you only want train/validation labels.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write image paths relative to each JSONL instead of absolute paths.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=(
            "Root containing already-extracted images. Default: --input-dir. "
            "Used by --json-only and source-image lookup."
        ),
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help=(
            "Read annotation columns only and reference existing files below "
            "--images-dir; never extract, copy, or rewrite images."
        ),
    )
    parser.add_argument(
        "--trust-image-paths",
        action="store_true",
        help=(
            "With --json-only, construct image paths without per-row existence "
            "checks. Use only after a checked run reports zero missing images."
        ),
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help=(
            "Reference source image paths directly when possible. Image bytes stored "
            "inside parquet are still extracted into --output-dir/--image-subdir."
        ),
    )
    parser.add_argument(
        "--image-subdir",
        default="images",
        help="Image output subdirectory under --output-dir.",
    )
    parser.add_argument(
        "--output-template",
        default="vqav2_{split}_sft_msswift.jsonl",
        help="Output JSONL filename template below --output-dir.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL/report files.")
    parser.add_argument("--overwrite-images", action="store_true", help="Rewrite extracted/copied image files.")
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
    parser.add_argument(
        "--max-pending-chunks",
        type=int,
        default=None,
        help="Maximum submitted worker chunks kept in memory. Default: --num-workers.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress every N source rows. Set 0 to disable.",
    )
    parser.add_argument(
        "--log-every-batch",
        action="store_true",
        help=(
            "Print one progress line after every completed worker chunk. "
            "This overrides --progress-every."
        ),
    )
    parser.add_argument(
        "--max-missing-image-logs",
        type=int,
        default=20,
        help="Maximum missing-image examples to print per split.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Only inspect parquet schema/row counts; do not write JSONL or images.",
    )
    return parser.parse_args()


def import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to read parquet files. Install it with: pip install pyarrow") from exc
    return pq


def normalize_split(value: Any) -> str:
    text = str(value).strip().lower().replace("_", "-")
    split = SPLIT_ALIASES.get(text) or SPLIT_ALIASES.get(text.replace("-", ""))
    if not split:
        raise SystemExit(f"Unsupported split {value!r}. Known splits: {', '.join(DEFAULT_SPLITS)}")
    return split


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(part for item in value if (part := clean_text(item)))
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


def answer_text_from_item(item: Any) -> str:
    if isinstance(item, dict):
        return clean_text(item.get("answer"))
    return clean_text(item)


def answer_list(row: dict[str, Any]) -> list[str]:
    answers = []
    for item in ensure_list(row.get("answers")):
        text = answer_text_from_item(item)
        if text:
            answers.append(text)
    return answers


def pick_answer(row: dict[str, Any], answer_policy: str) -> str:
    answers = answer_list(row)
    multiple_choice = clean_text(row.get("multiple_choice_answer"))

    if answer_policy == "multiple_choice" and multiple_choice:
        return multiple_choice
    if answer_policy == "majority" and answers:
        return Counter(answers).most_common(1)[0][0]
    if answer_policy == "first" and answers:
        return answers[0]
    if answer_policy == "all" and answers:
        return "; ".join(dict.fromkeys(answers))

    if multiple_choice:
        return multiple_choice
    if answers:
        return Counter(answers).most_common(1)[0][0]
    return ""


def parquet_files_for_split(data_dir: Path, split: str) -> list[Path]:
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split={split} below {data_dir}")
    return files


def iter_parquet_rows(
    files: Iterable[Path],
    batch_size: int,
    columns: tuple[str, ...] | None = None,
) -> Iterator[dict[str, Any]]:
    pq = import_pyarrow_parquet()
    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        selected_columns = None
        if columns is not None:
            schema_names = set(parquet_file.schema_arrow.names)
            selected_columns = [column for column in columns if column in schema_names]
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=selected_columns,
        ):
            yield from batch.to_pylist()


def image_bytes(image: Any) -> bytes:
    if not isinstance(image, dict):
        return b""
    data = image.get("bytes")
    if data is None:
        return b""
    if isinstance(data, memoryview):
        return data.tobytes()
    return bytes(data)


def image_path_value(image: Any) -> str:
    if isinstance(image, dict):
        return clean_text(image.get("path"))
    return clean_text(image)


def infer_suffix(path_text: str, data: bytes) -> str:
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
    return ".jpg"


def coco_basename(split: str, image_id: Any, suffix: str) -> str:
    try:
        image_id_int = int(image_id)
    except (TypeError, ValueError):
        return f"{clean_text(image_id) or 'unknown'}{suffix}"
    return f"COCO_{COCO_IMAGE_DIR[split]}_{image_id_int:012d}{suffix}"


def source_image_candidates(path_text: str, images_dir: Path, split: str) -> list[Path]:
    if not path_text:
        return []

    path = Path(path_text)
    basename = path.name
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute():
        candidates.append(images_dir / path)
    candidates.extend(
        [
            images_dir / basename,
            images_dir / split / basename,
            images_dir / COCO_IMAGE_DIR[split] / basename,
            images_dir / "images" / basename,
            images_dir / "images" / split / basename,
            images_dir / "images" / COCO_IMAGE_DIR[split] / basename,
        ]
    )
    return list(dict.fromkeys(candidates))


@lru_cache(maxsize=262144)
def resolve_source_image(path_text: str, images_dir: Path, split: str) -> Path | None:
    for candidate in source_image_candidates(path_text, images_dir, split):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate.resolve()
    return None


def resolve_existing_image(
    row: dict[str, Any],
    split: str,
    images_dir: Path,
    path_text: str,
) -> Path | None:
    if path_text:
        source_path = resolve_source_image(path_text, images_dir, split)
        if source_path is not None:
            return source_path
    canonical_name = coco_basename(split, row.get("image_id"), ".jpg")
    return resolve_source_image(canonical_name, images_dir, split)


def split_image_dir_candidates(images_dir: Path, split: str) -> list[Path]:
    candidates = [
        images_dir / split,
        images_dir / COCO_IMAGE_DIR[split],
        images_dir / "images" / split,
        images_dir / "images" / COCO_IMAGE_DIR[split],
        images_dir,
    ]
    if images_dir.name.lower() in {split.lower(), COCO_IMAGE_DIR[split].lower()}:
        candidates.insert(0, images_dir)
    return list(dict.fromkeys(candidate.resolve() for candidate in candidates))


def trusted_image_path(
    row: dict[str, Any],
    split: str,
    args: argparse.Namespace,
    path_text: str,
) -> Path:
    filename = Path(path_text).name
    if Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
        filename = coco_basename(split, row.get("image_id"), ".jpg")
    return (args.split_image_dirs[split] / filename).absolute()


def output_image_path(row: dict[str, Any], split: str, args: argparse.Namespace, data: bytes, path_text: str) -> Path:
    suffix = infer_suffix(path_text, data)
    filename = Path(path_text).name if Path(path_text).suffix.lower() in IMAGE_SUFFIXES else ""
    if not filename:
        filename = coco_basename(split, row.get("image_id"), suffix)
    return (args.output_dir / args.image_subdir / split / filename).resolve()


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


def prepare_image_path(row: dict[str, Any], split: str, args: argparse.Namespace, stats: Counter) -> Path | None:
    image = row.get("image")
    path_text = image_path_value(image)

    if args.json_only and args.trust_image_paths:
        stats["images_referenced"] += 1
        stats["images_unchecked"] += 1
        return trusted_image_path(row, split, args, path_text)

    source_path = resolve_existing_image(row, split, args.images_dir, path_text)

    if args.json_only:
        if source_path is not None:
            stats["images_referenced"] += 1
            return source_path
        stats["missing_images"] += 1
        return None

    data = image_bytes(image)

    if source_path is not None and args.no_copy_images and not data:
        stats["images_referenced"] += 1
        return source_path

    target_path = output_image_path(row, split, args, data, path_text)

    if data:
        if args.overwrite_images or not target_path.exists():
            write_bytes_atomically(target_path, data)
            stats["images_written"] += 1
        else:
            stats["images_reused"] += 1
        return target_path

    if source_path is not None:
        if args.no_copy_images:
            stats["images_referenced"] += 1
            return source_path
        if args.overwrite_images or not target_path.exists():
            copy_file_atomically(source_path, target_path)
            stats["images_copied"] += 1
        else:
            stats["images_reused"] += 1
        return target_path

    if target_path.exists():
        stats["images_reused"] += 1
        return target_path

    stats["missing_images"] += 1
    return None


def path_for_json(image_path: Path, jsonl_path: Path, relative: bool) -> str:
    if relative:
        return os.path.relpath(image_path, jsonl_path.parent).replace(os.sep, "/")
    return str(image_path)


def template_fields(row: dict[str, Any], split: str, answer: str) -> dict[str, str]:
    answers = answer_list(row)
    return {
        "question": clean_text(row.get("question")),
        "answer": answer,
        "answers": "; ".join(dict.fromkeys(answers)),
        "image_id": clean_text(row.get("image_id")),
        "question_id": clean_text(row.get("question_id")),
        "answer_type": clean_text(row.get("answer_type")),
        "question_type": clean_text(row.get("question_type")),
        "split": split,
    }


def build_record(
    row: dict[str, Any],
    image_path: Path,
    jsonl_path: Path,
    split: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    question = clean_text(row.get("question"))
    if not question:
        return None

    answer = pick_answer(row, args.answer_policy)
    if not answer and args.skip_unanswered:
        return None

    image_value = path_for_json(image_path, jsonl_path, args.relative_paths)
    fields = template_fields(row, split, answer)

    if args.mode == "sft":
        if not answer:
            return None
        return {
            "messages": [
                {"role": "user", "content": args.sft_question_template.format(**fields)},
                {"role": "assistant", "content": answer},
            ],
            "images": [image_value],
        }

    template = args.pretrain_template if answer else args.unanswered_template
    return {
        "messages": [{"role": "assistant", "content": template.format(**fields)}],
        "images": [image_value],
    }


def inspect_split(files: list[Path], split: str) -> dict[str, Any]:
    pq = import_pyarrow_parquet()
    total_rows = 0
    schema_names: list[str] | None = None
    for path in files:
        parquet_file = pq.ParquetFile(path)
        total_rows += parquet_file.metadata.num_rows
        names = parquet_file.schema_arrow.names
        schema_names = names if schema_names is None else schema_names

    missing = sorted(REQUIRED_FIELDS - set(schema_names or []))
    if missing:
        raise ValueError(f"split={split} parquet schema missing fields: {missing}")

    return {
        "split": split,
        "num_shards": len(files),
        "num_rows": total_rows,
        "schema": schema_names,
    }


def output_jsonl_path(split: str, args: argparse.Namespace) -> Path:
    return args.output_dir / args.output_template.format(split=split)


def iter_chunks(items: Iterable[dict[str, Any]], chunk_size: int) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def row_is_eligible(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if not clean_text(row.get("question")):
        return False
    answer = pick_answer(row, args.answer_policy)
    if args.mode == "sft":
        return bool(answer)
    return bool(answer) or not args.skip_unanswered


def process_row(
    row: dict[str, Any],
    split: str,
    jsonl_path: Path,
    args: argparse.Namespace,
    stats: Counter,
    missing_image_examples: list[dict[str, str]],
) -> str | None:
    stats["seen"] += 1
    if not row_is_eligible(row, args):
        stats["skipped_rows"] += 1
        return None

    image_path = prepare_image_path(row, split, args, stats)
    if image_path is None:
        if len(missing_image_examples) < args.max_missing_image_logs:
            missing_image_examples.append(
                {
                    "question_id": clean_text(row.get("question_id")),
                    "image_id": clean_text(row.get("image_id")),
                    "image": image_path_value(row.get("image")),
                }
            )
        return None

    record = build_record(row, image_path, jsonl_path, split, args)
    if record is None:
        # Eligibility is checked before image resolution; this only guards
        # custom templates or future format extensions.
        stats["skipped_rows"] += 1
        return None

    return json.dumps(record, ensure_ascii=False)


def process_chunk_with_context(
    rows: list[dict[str, Any]],
    split: str,
    jsonl_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stats: Counter = Counter()
    missing_image_examples: list[dict[str, str]] = []
    lines: list[str] = []
    for row in rows:
        line = process_row(row, split, jsonl_path, args, stats, missing_image_examples)
        if line is not None:
            lines.append(line)
    return {
        "stats": dict(stats),
        "missing_image_examples": missing_image_examples,
        "lines": lines,
    }


_WORKER_SPLIT: str | None = None
_WORKER_JSONL_PATH: Path | None = None
_WORKER_ARGS: argparse.Namespace | None = None


def init_worker(split: str, jsonl_path: Path, args: argparse.Namespace) -> None:
    global _WORKER_SPLIT, _WORKER_JSONL_PATH, _WORKER_ARGS
    _WORKER_SPLIT = split
    _WORKER_JSONL_PATH = jsonl_path
    _WORKER_ARGS = args


def process_chunk(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if _WORKER_SPLIT is None or _WORKER_JSONL_PATH is None or _WORKER_ARGS is None:
        raise RuntimeError("Worker context was not initialized")
    return process_chunk_with_context(rows, _WORKER_SPLIT, _WORKER_JSONL_PATH, _WORKER_ARGS)


def iter_parallel_results(
    executor: ProcessPoolExecutor,
    row_chunks: Iterable[list[dict[str, Any]]],
    max_pending_chunks: int,
) -> Iterator[dict[str, Any]]:
    """Process chunks in input order without eagerly materializing the dataset."""
    pending = deque()
    try:
        for chunk in row_chunks:
            pending.append(executor.submit(process_chunk, chunk))
            if len(pending) >= max_pending_chunks:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        while pending:
            pending.popleft().cancel()


def merge_missing_examples(
    target: list[dict[str, str]],
    incoming: list[dict[str, str]],
    limit: int,
) -> None:
    if len(target) >= limit:
        return
    target.extend(incoming[: limit - len(target)])


def write_chunk_result(
    result: dict[str, Any],
    out_file: TextIO,
    stats: Counter,
    missing_image_examples: list[dict[str, str]],
    args: argparse.Namespace,
) -> bool:
    stats.update(result["stats"])
    merge_missing_examples(missing_image_examples, result["missing_image_examples"], args.max_missing_image_logs)
    for line in result["lines"]:
        if args.limit is not None and stats["written"] >= args.limit:
            return True
        out_file.write(line + "\n")
        stats["written"] += 1
    return args.limit is not None and stats["written"] >= args.limit


def log_batch_result(
    split: str,
    result: dict[str, Any],
    out_file: TextIO,
    stats: Counter,
    batch_written: int,
    args: argparse.Namespace,
) -> None:
    if not args.log_every_batch:
        return
    out_file.flush()
    batch_stats = result["stats"]
    print(
        f"[batch] split={split} batch={stats['batches']:,} "
        f"batch_seen={batch_stats.get('seen', 0):,} "
        f"batch_written={batch_written:,} "
        f"batch_missing={batch_stats.get('missing_images', 0):,} "
        f"batch_skipped={batch_stats.get('skipped_rows', 0):,} "
        f"total_seen={stats['seen']:,} total_written={stats['written']:,}",
        flush=True,
    )


def convert_split(split: str, args: argparse.Namespace, files: list[Path]) -> dict[str, Any]:
    jsonl_path = output_jsonl_path(split, args)
    if jsonl_path.exists() and not args.overwrite:
        raise FileExistsError(f"{jsonl_path} already exists. Use --overwrite to replace it.")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    missing_image_examples: list[dict[str, str]] = []
    next_progress = args.progress_every if args.progress_every > 0 else 0

    with jsonl_path.open("w", encoding="utf-8") as out_file:
        selected_columns = JSON_ONLY_COLUMNS if args.json_only else None
        rows = iter_parquet_rows(files, args.batch_size, selected_columns)
        row_chunks = iter_chunks(rows, args.worker_chunk_size)
        if args.num_workers > 1:
            print(
                f"[parallel] split={split} workers={args.num_workers} "
                f"chunk_size={args.worker_chunk_size} pending={args.max_pending_chunks}",
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                initializer=init_worker,
                initargs=(split, jsonl_path, args),
            ) as executor:
                for result in iter_parallel_results(executor, row_chunks, args.max_pending_chunks):
                    written_before = stats["written"]
                    done = write_chunk_result(result, out_file, stats, missing_image_examples, args)
                    stats["batches"] += 1
                    log_batch_result(
                        split,
                        result,
                        out_file,
                        stats,
                        stats["written"] - written_before,
                        args,
                    )
                    while next_progress and stats["seen"] >= next_progress:
                        out_file.flush()
                        print(
                            f"[progress] split={split} seen={stats['seen']:,} written={stats['written']:,}",
                            flush=True,
                        )
                        next_progress += args.progress_every
                    if done:
                        break
        else:
            for chunk in row_chunks:
                result = process_chunk_with_context(chunk, split, jsonl_path, args)
                written_before = stats["written"]
                done = write_chunk_result(result, out_file, stats, missing_image_examples, args)
                stats["batches"] += 1
                log_batch_result(
                    split,
                    result,
                    out_file,
                    stats,
                    stats["written"] - written_before,
                    args,
                )
                while next_progress and stats["seen"] >= next_progress:
                    out_file.flush()
                    print(
                        f"[progress] split={split} seen={stats['seen']:,} written={stats['written']:,}",
                        flush=True,
                    )
                    next_progress += args.progress_every
                if done:
                    break

    return {
        "split": split,
        "output": str(jsonl_path),
        "stats": dict(stats),
        "missing_image_examples": missing_image_examples,
    }


def validate_args(args: argparse.Namespace) -> None:
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_dir = (args.data_dir or args.input_dir / "data").expanduser().resolve()
    args.images_dir = (args.images_dir or args.input_dir).expanduser().resolve()
    args.splits = [normalize_split(split) for split in args.splits]

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.num_workers <= 0:
        raise SystemExit("--num-workers must be greater than zero")
    if args.worker_chunk_size <= 0:
        raise SystemExit("--worker-chunk-size must be greater than zero")
    if args.max_pending_chunks is not None and args.max_pending_chunks <= 0:
        raise SystemExit("--max-pending-chunks must be greater than zero")
    if args.max_pending_chunks is None:
        args.max_pending_chunks = args.num_workers
    if args.limit is not None and args.num_workers > 1:
        print("[warning] --limit is set; falling back to --num-workers 1 for exact per-split limits")
        args.num_workers = 1
        args.max_pending_chunks = 1
    if args.limit is not None and args.worker_chunk_size != 1:
        args.worker_chunk_size = 1
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be greater than or equal to zero")
    if args.log_every_batch:
        args.progress_every = 0
    if args.max_missing_image_logs < 0:
        raise SystemExit("--max-missing-image-logs must be greater than or equal to zero")
    if args.json_only and args.overwrite_images:
        raise SystemExit("--json-only cannot be combined with --overwrite-images")
    if args.trust_image_paths and not args.json_only:
        raise SystemExit("--trust-image-paths requires --json-only")
    if args.json_only and not args.skip_convert and not args.images_dir.is_dir():
        raise SystemExit(f"--images-dir does not exist: {args.images_dir}")

    args.split_image_dirs: dict[str, Path] = {}
    if args.trust_image_paths and not args.skip_convert:
        for split in args.splits:
            image_dir = next(
                (
                    candidate
                    for candidate in split_image_dir_candidates(args.images_dir, split)
                    if candidate.is_dir()
                ),
                None,
            )
            if image_dir is None:
                raise SystemExit(
                    f"Could not locate an image directory for split={split} "
                    f"below {args.images_dir}"
                )
            args.split_image_dirs[split] = image_dir

    if args.mode == "pretrain":
        if args.pretrain_template.count("<image>") != 1:
            raise SystemExit("--pretrain-template must contain exactly one '<image>' token")
        if args.unanswered_template.count("<image>") != 1:
            raise SystemExit("--unanswered-template must contain exactly one '<image>' token")
    elif args.sft_question_template.count("<image>") != 1:
        raise SystemExit("--sft-question-template must contain exactly one '<image>' token")


def write_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    report_path = args.output_dir / "vqav2_msswift_report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"{report_path} already exists. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    print(f"[write] report={report_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    print(f"[info] input_dir={args.input_dir}")
    print(f"[info] data_dir={args.data_dir}")
    print(f"[info] images_dir={args.images_dir}")
    print(f"[info] output_dir={args.output_dir}")
    print(
        f"[info] json_only={args.json_only} "
        f"trust_image_paths={args.trust_image_paths} "
        f"log_every_batch={args.log_every_batch}"
    )
    for split, image_dir in args.split_image_dirs.items():
        print(f"[info] split_image_dir[{split}]={image_dir}")

    report: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "data_dir": str(args.data_dir),
        "images_dir": str(args.images_dir),
        "output_dir": str(args.output_dir),
        "format": "ms-swift multimodal JSONL",
        "mode": args.mode,
        "json_only": args.json_only,
        "trust_image_paths": args.trust_image_paths,
        "split_image_dirs": {
            split: str(image_dir)
            for split, image_dir in args.split_image_dirs.items()
        },
        "log_every_batch": args.log_every_batch,
        "num_workers": args.num_workers,
        "worker_chunk_size": args.worker_chunk_size,
        "max_pending_chunks": args.max_pending_chunks,
        "splits": {},
    }

    split_files: dict[str, list[Path]] = {}
    for split in args.splits:
        files = parquet_files_for_split(args.data_dir, split)
        split_files[split] = files
        split_report = inspect_split(files, split)
        report["splits"][split] = split_report
        print(
            f"[inspect] split={split} shards={split_report['num_shards']} "
            f"rows={split_report['num_rows']:,}"
        )

    if not args.skip_convert:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for split in args.splits:
            split_result = convert_split(split, args, split_files[split])
            report["splits"][split]["conversion"] = split_result
            stats = split_result["stats"]
            print(
                f"[write] split={split} output={split_result['output']} "
                f"written={stats.get('written', 0):,} seen={stats.get('seen', 0):,} "
                f"missing_images={stats.get('missing_images', 0):,}"
            )

    write_report(report, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
