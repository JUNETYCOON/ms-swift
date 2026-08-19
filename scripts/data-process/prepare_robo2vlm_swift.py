#!/usr/bin/env python3
"""Validate Robo2VLM parquet shards and convert them to ms-swift JSONL.

The expected Robo2VLM layout is:
    /mnt/luojunkun/stage1/dataset/robo2vlm/
      README.md
      data/train-00000-of-00262.parquet ... train-00261-of-00262.parquet
      data/test-00000-of-00003.parquet ... test-00002-of-00003.parquet

By default the converter writes SFT-style ms-swift records:
    {"messages": [{"role": "user", "content": "<image>..."}, {"role": "assistant", "content": "..."}], "images": ["/abs/path.jpg"]}

Examples:
    python prepare_robo2vlm_swift.py
    python prepare_robo2vlm_swift.py --max-samples 100 --overwrite
    python prepare_robo2vlm_swift.py --overwrite
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


DEFAULT_INPUT_DIR = Path("/mnt/luojunkun/stage1/dataset/robo2vlm")
DEFAULT_OUTPUT_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/robo2vlm")
REQUIRED_FIELDS = ("id", "question", "choices", "correct_answer", "image")
SPLITS = ("train", "test")
EXPECTED_SHARDS = {"train": 262, "test": 3}


@dataclass(frozen=True)
class SplitInfo:
    name: str
    files: list[Path]
    expected_rows: int | None
    actual_rows: int


@dataclass
class ConversionStats:
    read_rows: int = 0
    written_rows: int = 0
    skipped_missing_image: int = 0
    skipped_invalid_text: int = 0
    reused_image_paths: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Robo2VLM parquet shards to ms-swift multimodal JSONL."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Robo2VLM dataset root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="ms-swift output root.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Dataset splits to validate and convert.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("pt", "sft"),
        default="sft",
        help="pt writes assistant-only pre-training records; sft writes user/assistant turns.",
    )
    parser.add_argument(
        "--pt-template",
        default="<image>\nQuestion: {question}\nChoices:\n{choices}\nAnswer: {answer}",
        help="Assistant-only template for --dataset-mode pt.",
    )
    parser.add_argument(
        "--user-template",
        default="<image>\nQuestion: {question}\nChoices:\n{choices}",
        help="User prompt template for --dataset-mode sft.",
    )
    parser.add_argument(
        "--choice-format",
        default="{label}. {choice}",
        help="Per-choice format. Available fields: {index}, {label}, {choice}.",
    )
    parser.add_argument(
        "--answer-format",
        default="{label}. {choice}",
        help="Correct-answer format. Available fields: {index}, {label}, {choice}.",
    )
    parser.add_argument(
        "--missing-image-policy",
        choices=("skip", "error"),
        default="skip",
        help="How to handle rows whose image cannot be resolved or extracted.",
    )
    parser.add_argument(
        "--invalid-text-policy",
        choices=("skip", "error"),
        default="error",
        help="How to handle rows with invalid questions, choices, or answers.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory for images extracted from parquet bytes. Default: <output-dir>/robo2vlm_images.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write image paths relative to each JSONL file instead of absolute paths.",
    )
    parser.add_argument(
        "--reuse-existing-images",
        action="store_true",
        help=(
            "Reuse image paths from an existing robo2vlm_<split>.jsonl. "
            "This avoids reading embedded parquet image bytes when only text conversion changed."
        ),
    )
    parser.add_argument(
        "--validate-reused-images",
        action="store_true",
        help="Stat every reused image path. Disabled by default because it is costly on network storage.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Parquet record batch size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit converted rows per split.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL/report files.")
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Only validate README/parquet metadata; do not write JSONL.",
    )
    return parser.parse_args()


def import_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit(
            "pyarrow is required to read parquet files. Install it with: pip install pyarrow"
        ) from error
    return pq


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be greater than zero")
    if args.dataset_mode == "pt":
        for token in ("<image>", "{question}", "{choices}", "{answer}"):
            if token not in args.pt_template:
                raise SystemExit(f"--pt-template must contain {token!r}")
    else:
        for token in ("<image>", "{question}", "{choices}"):
            if token not in args.user_template:
                raise SystemExit(f"--user-template must contain {token!r}")
    try:
        args.choice_format.format(index=0, label="A", choice="example")
        args.answer_format.format(index=0, label="A", choice="example")
    except (KeyError, ValueError) as error:
        raise SystemExit(f"Invalid choice/answer format: {error}") from error


def load_readme_text(input_dir: Path) -> str:
    readme_path = input_dir / "README.md"
    if not readme_path.is_file():
        raise FileNotFoundError(f"README.md not found: {readme_path}")
    text = readme_path.read_text(encoding="utf-8")
    print(f"[read] {readme_path}")
    return text


def extract_front_matter(text: str) -> str | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[1:index])
    return None


def load_yaml_front_matter(text: str) -> dict[str, Any] | None:
    front_matter = extract_front_matter(text)
    if not front_matter:
        return None
    try:
        import yaml
    except ImportError:
        return None
    data = yaml.safe_load(front_matter)
    return data if isinstance(data, dict) else None


def find_feature_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        features = value.get("features")
        if isinstance(features, list):
            names = [item.get("name") for item in features if isinstance(item, dict)]
            names = [name for name in names if isinstance(name, str)]
            if set(REQUIRED_FIELDS).issubset(names):
                return names
        for child in value.values():
            names = find_feature_names(child)
            if names:
                return names
    elif isinstance(value, list):
        for item in value:
            names = find_feature_names(item)
            if names:
                return names
    return []


def find_split_counts(value: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(value, dict):
        splits = value.get("splits")
        if isinstance(splits, list):
            for split in splits:
                if not isinstance(split, dict):
                    continue
                name = split.get("name")
                num_examples = split.get("num_examples")
                if isinstance(name, str) and isinstance(num_examples, int):
                    counts[name] = num_examples
        for child in value.values():
            counts.update(find_split_counts(child))
    elif isinstance(value, list):
        for item in value:
            counts.update(find_split_counts(item))
    return counts


def regex_split_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    pattern = re.compile(
        r"(?ms)^\s*-\s*name:\s*(train|test)\s*$"
        r"(?:(?!^\s*-\s*name:\s*\S+\s*$).)*?"
        r"^\s*num_examples:\s*([0-9_]+)\s*$"
    )
    for match in pattern.finditer(text):
        counts[match.group(1)] = int(match.group(2).replace("_", ""))
    return counts


def validate_readme_schema(text: str) -> dict[str, int]:
    metadata = load_yaml_front_matter(text)
    feature_names = find_feature_names(metadata) if metadata is not None else []
    if feature_names:
        missing = [field for field in REQUIRED_FIELDS if field not in feature_names]
        if missing:
            raise ValueError(f"README schema missing required fields: {missing}")
        print(f"[ok] README schema fields include: {', '.join(REQUIRED_FIELDS)}")
    else:
        missing = [field for field in REQUIRED_FIELDS if not re.search(rf"\b{re.escape(field)}\b", text)]
        if missing:
            raise ValueError(
                "Could not confirm required fields from README.md. Missing text markers: "
                + ", ".join(missing)
            )
        print(
            "[ok] README mentions required schema fields; install pyyaml for stricter YAML metadata parsing"
        )

    split_counts = find_split_counts(metadata) if metadata is not None else {}
    if not split_counts:
        split_counts = regex_split_counts(text)
    if split_counts:
        for split in SPLITS:
            if split in split_counts:
                print(f"[ok] README metadata split={split} num_examples={split_counts[split]:,}")
    else:
        print("[warning] README split num_examples metadata was not found; row-count metadata check skipped")
    return split_counts


def parquet_files_for_split(data_dir: Path, split: str) -> list[Path]:
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split={split} below {data_dir}")
    expected_shards = EXPECTED_SHARDS.get(split)
    if expected_shards is not None and len(files) != expected_shards:
        raise ValueError(
            f"Expected {expected_shards} {split} parquet shards, found {len(files)} below {data_dir}"
        )
    return files


def validate_parquet_files(
    input_dir: Path,
    split_counts: dict[str, int],
    selected_splits: Iterable[str],
) -> dict[str, SplitInfo]:
    pq = import_pyarrow()
    data_dir = input_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    split_infos: dict[str, SplitInfo] = {}
    reference_schema = None
    reference_path: Path | None = None
    for split in selected_splits:
        files = parquet_files_for_split(data_dir, split)
        actual_rows = 0
        for path in files:
            parquet_file = pq.ParquetFile(path)
            schema = parquet_file.schema_arrow.remove_metadata()
            names = set(schema.names)
            missing = [field for field in REQUIRED_FIELDS if field not in names]
            if missing:
                raise ValueError(f"{path} parquet schema missing fields: {missing}")
            if reference_schema is None:
                reference_schema = schema
                reference_path = path
            elif not schema.equals(reference_schema, check_metadata=False):
                raise ValueError(f"Parquet schema mismatch: {path} differs from {reference_path}")
            actual_rows += parquet_file.metadata.num_rows

        expected_rows = split_counts.get(split)
        if expected_rows is not None and actual_rows != expected_rows:
            raise ValueError(
                f"Row count mismatch for {split}: README num_examples={expected_rows:,}, "
                f"parquet rows={actual_rows:,}"
            )
        print(
            f"[ok] split={split} shards={len(files):,} rows={actual_rows:,} "
            f"schema=consistent"
        )
        split_infos[split] = SplitInfo(split, files, expected_rows, actual_rows)

    return split_infos


def clean_id(value: Any, fallback: str) -> str:
    text = str(value if value is not None else fallback).strip() or fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:120] or fallback


def image_suffix(data: bytes | None, path_hint: str | None) -> str:
    if path_hint:
        suffix = Path(path_hint).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
            return suffix
    if data:
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


def resolve_path_hint(path_hint: str, input_dir: Path) -> Path | None:
    path = Path(path_hint)
    candidates = [path] if path.is_absolute() else [
        input_dir / path,
        input_dir / "data" / path,
        input_dir / "data" / "data" / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def extract_image_value(
    image_value: Any,
    input_dir: Path,
    image_dir: Path,
    split: str,
    row_index: int,
    row_id: Any,
) -> Path | None:
    image_bytes: bytes | None = None
    path_hint: str | None = None

    if isinstance(image_value, dict):
        raw_bytes = image_value.get("bytes")
        if raw_bytes is not None:
            image_bytes = bytes(raw_bytes)
        raw_path = image_value.get("path")
        if raw_path:
            path_hint = str(raw_path)
    elif isinstance(image_value, (bytes, bytearray, memoryview)):
        image_bytes = bytes(image_value)
    elif isinstance(image_value, str):
        path_hint = image_value

    if path_hint and image_bytes is None:
        resolved = resolve_path_hint(path_hint, input_dir)
        if resolved is not None:
            return resolved

    if image_bytes is None:
        return None

    suffix = image_suffix(image_bytes, path_hint)
    filename = f"{row_index:09d}_{clean_id(row_id, f'{split}_{row_index}')}{suffix}"
    output_path = image_dir / split / filename
    if output_path.is_file() and output_path.stat().st_size == len(image_bytes):
        return output_path.resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp")
    try:
        temporary.write_bytes(image_bytes)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path.resolve()


def normalize_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        choices: list[str] = []
        for item in value:
            if isinstance(item, (list, tuple)) or hasattr(item, "tolist"):
                choices.extend(normalize_choices(item))
            elif text := " ".join(str(item).split()):
                choices.append(text)
        return choices
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if converted is not value:
            return normalize_choices(converted)
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
    if isinstance(parsed, (list, tuple)) or hasattr(parsed, "tolist"):
        return normalize_choices(parsed)
    normalized = " ".join(str(parsed).split())
    return [normalized] if normalized else []


def format_choices(choices: list[str], choice_format: str) -> str:
    lines = []
    for index, choice in enumerate(choices):
        label = chr(ord("A") + index) if index < 26 else str(index + 1)
        lines.append(choice_format.format(index=index, label=label, choice=choice))
    return "\n".join(lines)


def choice_label(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else str(index + 1)


def resolve_answer_index(correct_answer: Any, choices: list[str]) -> int | None:
    if correct_answer is None:
        return None
    answer_index: int | None = None
    if isinstance(correct_answer, int) and not isinstance(correct_answer, bool):
        answer_index = correct_answer
    text = str(correct_answer).strip()
    if answer_index is None and text.isdigit():
        answer_index = int(text)
    if answer_index is None and len(text) == 1 and text.isalpha():
        answer_index = ord(text.upper()) - ord("A")
    if answer_index is None:
        for index, choice in enumerate(choices):
            if text == choice:
                answer_index = index
                break
    return answer_index


def deduplicate_choices(choices: list[str], correct_answer: Any) -> tuple[list[str], Any]:
    answer_index = resolve_answer_index(correct_answer, choices)
    unique_choices: list[str] = []
    canonical_indices: dict[str, int] = {}
    index_remap: list[int] = []
    for choice in choices:
        key = choice.casefold()
        canonical_index = canonical_indices.get(key)
        if canonical_index is None:
            canonical_index = len(unique_choices)
            canonical_indices[key] = canonical_index
            unique_choices.append(choice)
        index_remap.append(canonical_index)

    if answer_index is not None and 0 <= answer_index < len(index_remap):
        correct_answer = index_remap[answer_index]
    elif isinstance(correct_answer, str):
        canonical_index = canonical_indices.get(correct_answer.strip().casefold())
        if canonical_index is not None:
            correct_answer = canonical_index
    return unique_choices, correct_answer


def resolve_answer(correct_answer: Any, choices: list[str], answer_format: str) -> str:
    answer_index = resolve_answer_index(correct_answer, choices)
    text = "" if correct_answer is None else str(correct_answer).strip()
    if answer_index is not None and 0 <= answer_index < len(choices):
        return answer_format.format(
            index=answer_index,
            label=choice_label(answer_index),
            choice=choices[answer_index],
        )
    return text


def format_image_path(image_path: Path, jsonl_path: Path, relative: bool) -> str:
    if relative:
        return os.path.relpath(image_path, jsonl_path.parent).replace(os.sep, "/")
    return os.path.normpath(str(image_path))


def make_record(row: dict[str, Any], image_path: Path, jsonl_path: Path, args: argparse.Namespace) -> dict:
    question = str(row.get("question") or "").strip()
    choices, correct_answer = deduplicate_choices(
        normalize_choices(row.get("choices")), row.get("correct_answer")
    )
    if not 2 <= len(choices) <= 26:
        raise ValueError(f"multiple-choice row has {len(choices)} choices; expected 2..26")
    answer = resolve_answer(correct_answer, choices, args.answer_format)
    choices_text = format_choices(choices, args.choice_format)
    if not question or not choices_text or not answer:
        raise ValueError("question, choices, or correct_answer is empty")

    fields = {"question": question, "choices": choices_text, "answer": answer}
    if args.dataset_mode == "pt":
        messages = [{"role": "assistant", "content": args.pt_template.format(**fields)}]
    else:
        messages = [
            {"role": "user", "content": args.user_template.format(**fields)},
            {"role": "assistant", "content": answer},
        ]
    image_tokens = sum(message["content"].count("<image>") for message in messages)
    if image_tokens != 1:
        raise ValueError(f"record has {image_tokens} <image> tokens; exactly one is required")

    return {
        "id": row.get("id"),
        "messages": messages,
        "images": [format_image_path(image_path, jsonl_path, args.relative_paths)],
    }


def iter_parquet_rows(
    files: list[Path], batch_size: int, *, include_image: bool = True
) -> Iterator[dict[str, Any]]:
    pq = import_pyarrow()
    columns = list(REQUIRED_FIELDS if include_image else REQUIRED_FIELDS[:-1])
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield row


def iter_existing_images(
    jsonl_path: Path, *, validate_files: bool
) -> Iterator[tuple[str, Path]]:
    with jsonl_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid existing JSON at {jsonl_path}:{line_number}: {error}") from error
            images = record.get("images") if isinstance(record, dict) else None
            if not isinstance(images, list) or len(images) != 1 or not str(images[0]).strip():
                raise ValueError(f"Existing row must contain exactly one image: {jsonl_path}:{line_number}")
            image_path = Path(str(images[0]))
            if not image_path.is_absolute():
                image_path = jsonl_path.parent / image_path
            image_path = Path(os.path.normpath(str(image_path)))
            if validate_files and not image_path.is_file():
                raise FileNotFoundError(f"Existing image does not exist: {image_path}")
            yield str(record.get("id")), image_path


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


def convert_split(
    split_info: SplitInfo,
    input_dir: Path,
    output_dir: Path,
    image_dir: Path,
    args: argparse.Namespace,
) -> ConversionStats:
    jsonl_path = output_dir / f"robo2vlm_{split_info.name}.jsonl"
    stats = ConversionStats()
    existing_images = None
    if args.reuse_existing_images:
        if not jsonl_path.is_file():
            raise FileNotFoundError(
                f"Cannot use --reuse-existing-images; existing JSONL not found: {jsonl_path}"
            )
        existing_images = iter_existing_images(
            jsonl_path, validate_files=args.validate_reused_images
        )
    stream = open_output_jsonl(jsonl_path, args.overwrite)
    try:
        for row in iter_parquet_rows(
            split_info.files,
            args.batch_size,
            include_image=existing_images is None,
        ):
            if args.max_samples is not None and stats.written_rows >= args.max_samples:
                break
            stats.read_rows += 1
            if existing_images is not None:
                try:
                    existing_id, image_path = next(existing_images)
                except StopIteration:
                    raise ValueError(
                        f"Existing JSONL ended before parquet split={split_info.name} "
                        f"row={stats.read_rows}"
                    ) from None
                if existing_id != str(row.get("id")):
                    raise ValueError(
                        f"Existing JSONL/parquet ID mismatch at split={split_info.name} "
                        f"row={stats.read_rows}: {existing_id!r} != {row.get('id')!r}"
                    )
                stats.reused_image_paths += 1
            else:
                image_path = extract_image_value(
                    row.get("image"),
                    input_dir,
                    image_dir,
                    split_info.name,
                    stats.read_rows - 1,
                    row.get("id"),
                )
            if image_path is None:
                if args.missing_image_policy == "error":
                    raise ValueError(f"Missing image for split={split_info.name} row={stats.read_rows}")
                stats.skipped_missing_image += 1
                continue
            try:
                record = make_record(row, image_path, jsonl_path, args)
            except ValueError as error:
                stats.skipped_invalid_text += 1
                if args.invalid_text_policy == "error":
                    raise ValueError(
                        f"Invalid text for split={split_info.name} row={stats.read_rows} "
                        f"id={row.get('id')!r}: {error}"
                    ) from error
                continue
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats.written_rows += 1
            if stats.written_rows % 10000 == 0:
                print(f"[convert] split={split_info.name} rows={stats.written_rows:,}")
        if existing_images is not None and args.max_samples is None:
            try:
                extra_id, _ = next(existing_images)
            except StopIteration:
                pass
            else:
                raise ValueError(
                    f"Existing JSONL has rows beyond parquet split={split_info.name}: "
                    f"first extra id={extra_id!r}"
                )
        commit_output_jsonl(stream, jsonl_path)
    except BaseException:
        cleanup_output_jsonl(stream)
        raise

    print(
        f"[ok] split={split_info.name} read={stats.read_rows:,} written={stats.written_rows:,} "
        f"missing_image={stats.skipped_missing_image:,} invalid_text={stats.skipped_invalid_text:,} "
        f"reused_image_paths={stats.reused_image_paths:,}"
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
    image_dir = (args.image_dir or (args.output_dir / "robo2vlm_images")).expanduser().resolve()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    readme_text = load_readme_text(args.input_dir)
    split_counts = validate_readme_schema(readme_text)
    split_infos = validate_parquet_files(args.input_dir, split_counts, args.splits)

    conversion_stats: dict[str, dict[str, int]] = {}
    if not args.skip_convert:
        for split in args.splits:
            stats = convert_split(split_infos[split], args.input_dir, args.output_dir, image_dir, args)
            conversion_stats[split] = {
                "read_rows": stats.read_rows,
                "written_rows": stats.written_rows,
                "skipped_missing_image": stats.skipped_missing_image,
                "skipped_invalid_text": stats.skipped_invalid_text,
                "reused_image_paths": stats.reused_image_paths,
            }

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "image_dir": str(image_dir),
        "dataset_mode": args.dataset_mode,
        "choice_format": args.choice_format,
        "answer_format": args.answer_format,
        "invalid_text_policy": args.invalid_text_policy,
        "reuse_existing_images": args.reuse_existing_images,
        "validate_reused_images": args.validate_reused_images,
        "required_fields": list(REQUIRED_FIELDS),
        "splits": {
            name: {
                "shards": len(info.files),
                "expected_rows_from_readme": info.expected_rows,
                "actual_parquet_rows": info.actual_rows,
            }
            for name, info in split_infos.items()
        },
        "conversion": conversion_stats,
    }
    write_report(args.output_dir / "robo2vlm_conversion_report.json", report, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
