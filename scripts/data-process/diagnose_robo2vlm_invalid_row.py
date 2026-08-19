#!/usr/bin/env python3
"""Locate Robo2VLM rows skipped by the converter without rescanning image bytes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from scripts import prepare_robo2vlm_swift as converter
except ImportError:
    import prepare_robo2vlm_swift as converter


IMAGE_INDEX = re.compile(r"^(\d{9})_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("/mnt/luojunkun/stage1/dataset_ms-swift/robo2vlm/robo2vlm_train.jsonl"),
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("/mnt/luojunkun/stage1/dataset/robo2vlm/data"),
    )
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--expected-rows", type=int, default=678034)
    return parser.parse_args()


def find_first_missing_index(jsonl_path: Path) -> tuple[int, dict[str, Any] | None, dict[str, Any]]:
    previous: dict[str, Any] | None = None
    with jsonl_path.open("r", encoding="utf-8-sig") as stream:
        for zero_based_line, line in enumerate(stream):
            record = json.loads(line)
            images = record.get("images")
            if not isinstance(images, list) or len(images) != 1:
                raise ValueError(f"Invalid images field at line {zero_based_line + 1}")
            match = IMAGE_INDEX.match(Path(str(images[0])).name)
            if match is None:
                raise ValueError(f"Image filename has no source index at line {zero_based_line + 1}")
            source_index = int(match.group(1))
            if source_index != zero_based_line:
                if source_index < zero_based_line:
                    raise ValueError(
                        f"Non-monotonic source index at line {zero_based_line + 1}: {source_index}"
                    )
                return zero_based_line, previous, record
            previous = record
    raise ValueError("No source-index gap found in JSONL")


def read_parquet_row(parquet_dir: Path, split: str, source_index: int) -> tuple[Path, int, dict[str, Any]]:
    pq = converter.import_pyarrow()
    remaining = source_index
    for path in converter.parquet_files_for_split(parquet_dir, split):
        parquet_file = pq.ParquetFile(path)
        rows = parquet_file.metadata.num_rows
        if remaining >= rows:
            remaining -= rows
            continue
        table = parquet_file.read(columns=list(converter.REQUIRED_FIELDS[:-1]))
        return path, remaining, table.slice(remaining, 1).to_pylist()[0]
    raise IndexError(f"Source index {source_index} is outside the {split} parquet rows")


def converter_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_mode="sft",
        pt_template="<image>\nQuestion: {question}\nChoices:\n{choices}\nAnswer: {answer}",
        user_template="<image>\nQuestion: {question}\nChoices:\n{choices}",
        choice_format="{label}. {choice}",
        answer_format="{label}. {choice}",
        relative_paths=False,
    )


def find_existing_image(
    jsonl_path: Path, following: dict[str, Any], source_index: int, row_id: Any
) -> Path:
    following_images = following.get("images")
    if not isinstance(following_images, list) or len(following_images) != 1:
        raise ValueError("Following record has no unique image path")
    following_path = Path(str(following_images[0]))
    if not following_path.is_absolute():
        following_path = jsonl_path.parent / following_path
    stem = f"{source_index:09d}_{converter.clean_id(row_id, f'train_{source_index}')}"
    suffixes = [following_path.suffix.lower(), ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]
    for suffix in dict.fromkeys(suffixes):
        candidate = following_path.parent / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No existing image found for source index {source_index}: {stem}")


def source_image_index(record: dict[str, Any]) -> int:
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Record has no unique image path")
    match = IMAGE_INDEX.match(Path(str(images[0])).name)
    if match is None:
        raise ValueError("Record image filename has no source index")
    return int(match.group(1))


def repair_jsonl(
    jsonl_path: Path,
    missing_index: int,
    inserted_record: dict[str, Any],
    expected_rows: int,
) -> None:
    temporary = jsonl_path.with_name(f"{jsonl_path.name}.repair.tmp")
    if temporary.exists():
        raise FileExistsError(f"Repair temporary file already exists: {temporary}")
    output_rows = 0
    try:
        with jsonl_path.open("r", encoding="utf-8-sig") as source, temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            for line in source:
                if output_rows == missing_index:
                    destination.write(
                        json.dumps(inserted_record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    output_rows += 1
                record = json.loads(line)
                source_index = source_image_index(record)
                if source_index != output_rows:
                    raise ValueError(
                        f"Source index mismatch while repairing output row {output_rows}: {source_index}"
                    )
                destination.write(line if line.endswith("\n") else f"{line}\n")
                output_rows += 1
        if output_rows != expected_rows:
            raise ValueError(f"Repaired row count is {output_rows}; expected {expected_rows}")
        temporary.replace(jsonl_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    missing_index, previous, following = find_first_missing_index(args.jsonl)
    parquet_path, shard_index, row = read_parquet_row(args.parquet_dir, "train", missing_index)

    image_path = find_existing_image(args.jsonl, following, missing_index, row.get("id"))

    make_record_error = None
    record = None
    try:
        record = converter.make_record(
            row,
            image_path,
            args.jsonl,
            converter_args(),
        )
    except ValueError as error:
        make_record_error = str(error)

    print(
        json.dumps(
            {
                "missing_source_index": missing_index,
                "jsonl_line_to_insert_before": missing_index + 1,
                "previous_id": previous.get("id") if previous else None,
                "following_id": following.get("id"),
                "parquet_path": str(parquet_path),
                "parquet_shard_row_index": shard_index,
                "image_path": str(image_path),
                "row": row,
                "normalized_choices": converter.normalize_choices(row.get("choices")),
                "make_record_error": make_record_error,
                "record_if_valid": record,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    if args.repair:
        if record is None:
            raise ValueError(f"Cannot repair invalid record: {make_record_error}")
        repair_jsonl(args.jsonl, missing_index, record, args.expected_rows)
        print(
            json.dumps(
                {
                    "repaired": True,
                    "jsonl": str(args.jsonl),
                    "inserted_source_index": missing_index,
                    "inserted_id": record.get("id"),
                    "expected_rows": args.expected_rows,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
