#!/usr/bin/env python3
"""Convert local GQA parquet data to ms-swift multimodal JSONL.

The default output is SFT-style ms-swift multimodal data:

{"messages": [{"role": "user", "content": "<image>..."}, {"role": "assistant", "content": "..."}],
 "images": ["/abs/path/to/image.jpg"]}

The source is expected to follow the local GQA layout:

GQA/
  train_balanced_instructions/*.parquet
  train_balanced_images/*.parquet
  val_balanced_instructions/*.parquet
  val_balanced_images/*.parquet

or the same folders below GQA/master/.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


DEFAULT_GQA_ROOT = Path("/mnt/nfs_a/cvat-data/zhw/vlm/vlm_train/GQA")
DEFAULT_CONFIGS = ("train_balanced", "val_balanced")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GQA parquet shards to ms-swift multimodal JSONL."
    )
    parser.add_argument(
        "--gqa-root",
        type=Path,
        default=DEFAULT_GQA_ROOT,
        help=f"GQA dataset root. Default: {DEFAULT_GQA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <gqa-root>/gqa-swift",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help=(
            "GQA configs to convert, without _instructions/_images suffix. "
            "Examples: train_balanced val_balanced train_all val_all"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "sft"],
        default="sft",
        help="pretrain writes assistant-only samples; sft writes user/assistant QA samples.",
    )
    parser.add_argument(
        "--answer-source",
        choices=["fullAnswer", "answer"],
        default="fullAnswer",
        help="Which GQA answer field to use when present.",
    )
    parser.add_argument(
        "--pretrain-template",
        default="<image>\nQuestion: {question}\nAnswer: {answer}",
        help="Template used when --mode pretrain.",
    )
    parser.add_argument(
        "--sft-question-template",
        default="<image>{question}",
        help="User prompt template used when --mode sft.",
    )
    parser.add_argument(
        "--include-unanswered",
        action="store_true",
        help="Write samples without answers. By default they are skipped.",
    )
    parser.add_argument(
        "--unanswered-template",
        default="<image>\nQuestion: {question}",
        help="Template used for unanswered pretrain samples when --include-unanswered.",
    )
    parser.add_argument(
        "--image-subdir",
        default="images",
        help="Image output subdirectory under --output-dir.",
    )
    parser.add_argument(
        "--jsonl-suffix",
        default="msswift",
        help="Suffix used in output JSONL filenames.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write image paths relative to each JSONL file instead of absolute paths.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Rewrite image files even if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit valid written samples per config. Useful for quick checks.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print conversion progress every N instruction rows.",
    )
    parser.add_argument(
        "--max-missing-image-logs",
        type=int,
        default=20,
        help="Maximum missing-image examples to print per config.",
    )
    return parser.parse_args()


def parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    return files


def iter_parquet_rows(files: Iterable[Path], batch_size: int = 8192) -> Iterable[dict[str, Any]]:
    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def pick_answer(row: dict[str, Any], answer_source: str) -> str:
    primary = clean_text(row.get(answer_source))
    if primary:
        return primary
    fallback = "answer" if answer_source == "fullAnswer" else "fullAnswer"
    return clean_text(row.get(fallback))


def valid_instruction(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if not clean_text(row.get("question")):
        return False
    if not clean_text(row.get("imageId")):
        return False
    if not args.include_unanswered and not pick_answer(row, args.answer_source):
        return False
    return True


def collect_needed_image_ids(
    instruction_files: list[Path],
    args: argparse.Namespace,
) -> set[str] | None:
    if args.limit is None:
        return None

    needed: set[str] = set()
    valid_count = 0
    for row in iter_parquet_rows(instruction_files):
        if not valid_instruction(row, args):
            continue
        needed.add(clean_text(row["imageId"]))
        valid_count += 1
        if valid_count >= args.limit:
            break
    return needed


def image_suffix(image: dict[str, Any], image_id: str) -> str:
    image_path = clean_text(image.get("path"))
    suffix = Path(image_path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return ".jpg" if suffix == ".jpeg" else suffix

    image_bytes = image.get("bytes") or b""
    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if image_bytes.startswith(b"BM"):
        return ".bmp"
    raise ValueError(f"Cannot infer image type for image id {image_id!r}")


def extract_images(
    image_files: list[Path],
    image_output_dir: Path,
    needed_ids: set[str] | None,
    overwrite: bool,
) -> tuple[dict[str, Path], Counter]:
    image_output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, Path] = {}
    stats: Counter = Counter()

    for row in iter_parquet_rows(image_files):
        image_id = clean_text(row.get("id"))
        if not image_id:
            stats["images_missing_id"] += 1
            continue
        if needed_ids is not None and image_id not in needed_ids:
            continue

        image = row.get("image") or {}
        image_bytes = image.get("bytes")
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        if not image_bytes:
            stats["images_missing_bytes"] += 1
            continue

        try:
            suffix = image_suffix(image, image_id)
        except ValueError:
            stats["images_unknown_type"] += 1
            continue

        output_path = image_output_dir / f"{image_id}{suffix}"
        if overwrite or not output_path.exists():
            output_path.write_bytes(image_bytes)
            stats["images_written"] += 1
        else:
            stats["images_reused"] += 1
        image_paths[image_id] = output_path.resolve()

        if needed_ids is not None and needed_ids.issubset(image_paths):
            break

    if needed_ids is not None:
        stats["needed_images"] = len(needed_ids)
        stats["needed_images_found"] = len(set(image_paths) & needed_ids)
    else:
        stats["images_found"] = len(image_paths)
    return image_paths, stats


def image_path_for_json(image_path: Path, jsonl_path: Path, relative: bool) -> str:
    if relative:
        return os.path.relpath(image_path, jsonl_path.parent).replace(os.sep, "/")
    return str(image_path)


def build_record(
    row: dict[str, Any],
    image_path: Path,
    jsonl_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    question = clean_text(row.get("question"))
    answer = pick_answer(row, args.answer_source)
    image_value = image_path_for_json(image_path, jsonl_path, args.relative_paths)

    if args.mode == "sft":
        if not answer:
            return None
        return {
            "messages": [
                {"role": "user", "content": args.sft_question_template.format(question=question)},
                {"role": "assistant", "content": answer},
            ],
            "images": [image_value],
        }

    if answer:
        content = args.pretrain_template.format(question=question, answer=answer)
    elif args.include_unanswered:
        content = args.unanswered_template.format(question=question)
    else:
        return None
    return {
        "messages": [{"role": "assistant", "content": content}],
        "images": [image_value],
    }


def resolve_data_root(gqa_root: Path, config: str) -> Path:
    direct_instruction_dir = gqa_root / f"{config}_instructions"
    direct_image_dir = gqa_root / f"{config}_images"
    if direct_instruction_dir.is_dir() and direct_image_dir.is_dir():
        return gqa_root

    master_dir = gqa_root / "master"
    master_instruction_dir = master_dir / f"{config}_instructions"
    master_image_dir = master_dir / f"{config}_images"
    if master_instruction_dir.is_dir() and master_image_dir.is_dir():
        return master_dir

    raise FileNotFoundError(
        "Missing GQA parquet directories. Expected either "
        f"{direct_instruction_dir} and {direct_image_dir}, or "
        f"{master_instruction_dir} and {master_image_dir}."
    )


def convert_config(config: str, args: argparse.Namespace) -> Counter:
    gqa_root = args.gqa_root.expanduser().resolve()
    data_root = resolve_data_root(gqa_root, config)
    output_dir = (args.output_dir or (gqa_root / "gqa-swift")).expanduser().resolve()
    instruction_dir = data_root / f"{config}_instructions"
    image_dir = data_root / f"{config}_images"

    instruction_files = parquet_files(instruction_dir)
    image_files = parquet_files(image_dir)

    jsonl_path = output_dir / f"gqa_{config}_{args.mode}_{args.jsonl_suffix}.jsonl"
    image_output_dir = output_dir / args.image_subdir / config
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    needed_ids = collect_needed_image_ids(instruction_files, args)
    print(f"[{config}] data root: {data_root}")
    print(f"[{config}] extracting images to {image_output_dir}")
    image_paths, image_stats = extract_images(
        image_files=image_files,
        image_output_dir=image_output_dir,
        needed_ids=needed_ids,
        overwrite=args.overwrite_images,
    )

    stats: Counter = Counter(image_stats)
    missing_logs = 0
    print(f"[{config}] writing {jsonl_path}")
    with jsonl_path.open("w", encoding="utf-8") as out_file:
        for row in iter_parquet_rows(instruction_files):
            stats["instruction_rows"] += 1
            if args.progress_every and stats["instruction_rows"] % args.progress_every == 0:
                print(
                    f"[{config}] rows={stats['instruction_rows']} "
                    f"written={stats['written']} skipped={stats['skipped']}"
                )

            if not valid_instruction(row, args):
                stats["skipped_invalid_instruction"] += 1
                stats["skipped"] += 1
                continue

            image_id = clean_text(row.get("imageId"))
            image_path = image_paths.get(image_id)
            if image_path is None:
                stats["missing_image"] += 1
                stats["skipped"] += 1
                if missing_logs < args.max_missing_image_logs:
                    print(f"[{config}] missing image for imageId={image_id}")
                    missing_logs += 1
                continue

            record = build_record(row, image_path, jsonl_path, args)
            if record is None:
                stats["skipped_unanswered"] += 1
                stats["skipped"] += 1
                continue

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            if args.limit is not None and stats["written"] >= args.limit:
                break

    stats_path = jsonl_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as stats_file:
        json.dump(dict(stats), stats_file, ensure_ascii=False, indent=2, sort_keys=True)
        stats_file.write("\n")

    print(f"[{config}] done: written={stats['written']} stats={stats_path}")
    return stats


def main() -> None:
    args = parse_args()
    total: Counter = Counter()
    for config in args.configs:
        total.update(convert_config(config, args))
    print("[total]", json.dumps(dict(total), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
