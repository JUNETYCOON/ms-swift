#!/usr/bin/env python3
"""Convert Visual Genome region JSONL records to ms-swift grounding JSONL.

The input is the region output produced by prepare_visualgenome_swift.py. A
source record such as a region caption with one real bounding box can be
converted into either of the two canonical grounding directions:

locate:
    user:      <image>\nLocate <ref-object>.
    assistant: <bbox>

describe:
    user:      <image>\nDescribe the region <bbox>.
    assistant: <ref-object>

grounded:
    user:      <image>\nDescribe the image with a grounded region.
    assistant: <ref-object><bbox>

The original region phrase is stored in objects.ref and the original boxes are
stored in objects.bbox. Images are referenced in place and are not copied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


DEFAULT_INPUT_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/visualgenome")
DEFAULT_OUTPUT_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/visualgenome_grounding")
DEFAULT_SPLITS = ("train", "val")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Visual Genome region JSONL to ms-swift grounding JSONL."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Source JSONL directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Grounding output directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to convert. Accepted aliases include validation/dev for val.",
    )
    parser.add_argument(
        "--input-template",
        default="visualgenome_regions_{split}.jsonl",
        help="Input filename template below --input-dir.",
    )
    parser.add_argument(
        "--output-template",
        default="visualgenome_regions_grounding_{split}.jsonl",
        help="Output filename template below --output-dir.",
    )
    parser.add_argument(
        "--task",
        choices=("locate", "describe", "grounded", "both"),
        default="locate",
        help=(
            "Grounding direction. grounded emits both the region phrase and bbox in the assistant answer; "
            "both writes separate locate and describe rows."
        ),
    )
    parser.add_argument("--system", default="You are a helpful assistant.", help="Optional system message.")
    parser.add_argument(
        "--locate-prompt",
        default="<image>\nLocate <ref-object>.",
        help="Prompt for phrase-to-box grounding. Must contain one <image> and one <ref-object>.",
    )
    parser.add_argument(
        "--describe-prompt",
        default="<image>\nDescribe the region {bbox_tokens}.",
        help="Prompt for box-to-phrase grounding. Must contain one <image> and {bbox_tokens}.",
    )
    parser.add_argument(
        "--grounded-prompt",
        default="<image>\nDescribe the image with a grounded region.",
        help="Prompt for phrase-and-box output. Must contain one <image> and no grounding placeholders.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write image paths relative to each output JSONL.",
    )
    parser.add_argument(
        "--invalid-policy",
        choices=("skip", "error"),
        default="skip",
        help="How to handle malformed source rows.",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Limit valid source rows per split.")
    parser.add_argument("--max-error-logs", type=int, default=20, help="Invalid-row examples retained per split.")
    parser.add_argument("--progress-every", type=int, default=50000, help="Print progress every N source rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL and report files.")
    return parser.parse_args()


def normalize_split(value: Any) -> str:
    text = str(value).strip().lower().replace("_", "-")
    split = SPLIT_ALIASES.get(text)
    if split is None:
        raise SystemExit(f"Unsupported split {value!r}; expected train or val")
    return split


def validate_args(args: argparse.Namespace) -> None:
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.splits = list(dict.fromkeys(normalize_split(split) for split in args.splits))
    if args.max_records is not None and args.max_records <= 0:
        raise SystemExit("--max-records must be greater than zero")
    if args.max_error_logs < 0:
        raise SystemExit("--max-error-logs must be greater than or equal to zero")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be greater than or equal to zero")
    if args.locate_prompt.count("<image>") != 1 or args.locate_prompt.count("<ref-object>") != 1:
        raise SystemExit("--locate-prompt must contain exactly one <image> and one <ref-object>")
    if args.locate_prompt.count("<bbox>") != 0:
        raise SystemExit("--locate-prompt must not contain <bbox>")
    if args.describe_prompt.count("<image>") != 1 or args.describe_prompt.count("{bbox_tokens}") != 1:
        raise SystemExit("--describe-prompt must contain exactly one <image> and one {bbox_tokens}")
    if "<ref-object>" in args.describe_prompt:
        raise SystemExit("--describe-prompt must not contain <ref-object>")
    if args.grounded_prompt.count("<image>") != 1:
        raise SystemExit("--grounded-prompt must contain exactly one <image>")
    if "<ref-object>" in args.grounded_prompt or "<bbox>" in args.grounded_prompt:
        raise SystemExit("--grounded-prompt must not contain <ref-object> or <bbox>")


def source_path(split: str, args: argparse.Namespace) -> Path:
    return args.input_dir / args.input_template.format(split=split)


def output_path(split: str, args: argparse.Namespace) -> Path:
    return args.output_dir / args.output_template.format(split=split)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("JSONL row must be an object")
            yield line_number, value


def first_assistant_content(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def region_phrase(record: dict[str, Any]) -> str:
    phrase = first_assistant_content(record)
    objects = record.get("objects")
    refs = objects.get("ref") if isinstance(objects, dict) else None
    placeholder_free = phrase.replace("<ref-object>", "").replace("<bbox>", "").strip()
    if not placeholder_free and isinstance(refs, list) and len(refs) == 1:
        phrase = str(refs[0]).strip()
    if not phrase:
        phrase = str(record.get("phrase") or "").strip()
    if not phrase:
        raise ValueError("region phrase is empty")
    if any(token in phrase for token in ("<image>", "<bbox>", "<ref-object>")):
        raise ValueError("region phrase contains a reserved multimodal token")
    return " ".join(phrase.split())


def normalized_box(value: Any, bbox_type: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) not in {2, 4}:
        raise ValueError("each bbox must contain two or four coordinates")
    box: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError("bbox coordinates must be numeric")
        coordinate = float(coordinate)
        if not math.isfinite(coordinate):
            raise ValueError("bbox coordinates must be finite")
        if bbox_type == "norm1" and not 0 <= coordinate <= 1:
            raise ValueError("norm1 bbox coordinates must be in the range [0, 1]")
        box.append(coordinate)
    if len(box) == 4:
        box[0], box[2] = sorted((box[0], box[2]))
        box[1], box[3] = sorted((box[1], box[3]))
    return box


def normalized_objects(record: dict[str, Any], image_count: int) -> tuple[list[list[float]], str, list[int]]:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("objects is missing or is not an object")
    bbox_type = str(objects.get("bbox_type") or "real")
    if bbox_type not in {"real", "norm1"}:
        raise ValueError("bbox_type must be 'real' or 'norm1'")
    raw_boxes = objects.get("bbox")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError("objects.bbox must be a non-empty list")
    boxes = [normalized_box(box, bbox_type) for box in raw_boxes]

    if bbox_type == "norm1":
        return boxes, bbox_type, []

    raw_image_ids = objects.get("image_id")
    if raw_image_ids is None:
        image_ids = [0] * len(boxes)
    elif isinstance(raw_image_ids, list) and len(raw_image_ids) == len(boxes):
        image_ids = []
        for image_id in raw_image_ids:
            if isinstance(image_id, bool) or not isinstance(image_id, int):
                raise ValueError("objects.image_id values must be integers")
            image_ids.append(image_id)
    else:
        raise ValueError("objects.image_id length must equal objects.bbox length")
    if any(image_id < 0 or image_id >= image_count for image_id in image_ids):
        raise ValueError("objects.image_id contains an out-of-range image index")
    return boxes, bbox_type, image_ids


def normalized_images(record: dict[str, Any], source_file: Path, target_file: Path, relative: bool) -> list[str]:
    images = record.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str) or not images[0].strip():
        raise ValueError("Visual Genome region rows must contain exactly one image path")
    image_path = Path(images[0]).expanduser()
    if not image_path.is_absolute():
        image_path = (source_file.parent / image_path).resolve()
    else:
        image_path = image_path.resolve()
    if relative:
        return [os.path.relpath(image_path, target_file.parent).replace(os.sep, "/")]
    return [str(image_path)]


def messages(system: str, user: str, assistant: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if system:
        result.append({"role": "system", "content": system})
    result.extend(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    )
    return result


def grounding_objects(phrase: str, boxes: list[list[float]], bbox_type: str, image_ids: list[int]) -> dict[str, Any]:
    objects: dict[str, Any] = {"ref": [phrase], "bbox": boxes, "bbox_type": bbox_type}
    if bbox_type == "real":
        objects["image_id"] = image_ids
    return objects


def output_record(
    source: dict[str, Any],
    output_messages: list[dict[str, str]],
    images: list[str],
    objects: dict[str, Any],
) -> dict[str, Any]:
    record = {key: value for key, value in source.items() if key not in {"messages", "images", "objects"}}
    record.update({"messages": output_messages, "images": images, "objects": objects})
    return record


def validate_output(record: dict[str, Any]) -> None:
    content = "\n".join(message["content"] for message in record["messages"])
    if content.count("<image>") != len(record["images"]):
        raise ValueError("generated <image> count does not match images")
    if content.count("<ref-object>") != len(record["objects"]["ref"]):
        raise ValueError("generated <ref-object> count does not match objects.ref")
    if content.count("<bbox>") != len(record["objects"]["bbox"]):
        raise ValueError("generated <bbox> count does not match objects.bbox")


def convert_record(
    source: dict[str, Any],
    source_file: Path,
    target_file: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    phrase = region_phrase(source)
    images = normalized_images(source, source_file, target_file, args.relative_paths)
    boxes, bbox_type, image_ids = normalized_objects(source, len(images))
    objects = grounding_objects(phrase, boxes, bbox_type, image_ids)
    bbox_tokens = "".join("<bbox>" for _ in boxes)
    records: list[dict[str, Any]] = []

    if args.task in {"locate", "both"}:
        records.append(
            output_record(source, messages(args.system, args.locate_prompt, bbox_tokens), images, objects)
        )
    if args.task in {"describe", "both"}:
        records.append(
            output_record(
                source,
                messages(
                    args.system,
                    args.describe_prompt.format(bbox_tokens=bbox_tokens),
                    "<ref-object>",
                ),
                images,
                objects,
            )
        )
    if args.task == "grounded":
        records.append(
            output_record(
                source,
                messages(args.system, args.grounded_prompt, "<ref-object>" + bbox_tokens),
                images,
                objects,
            )
        )
    for record in records:
        validate_output(record)
    return records


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


def convert_split(split: str, args: argparse.Namespace) -> dict[str, Any]:
    input_file = source_path(split, args)
    target_file = output_path(split, args)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_file}")

    stats: Counter = Counter()
    errors: list[dict[str, Any]] = []
    next_progress = args.progress_every if args.progress_every > 0 else 0
    stream = open_output(target_file, args.overwrite)
    try:
        try:
            iterator = iter_jsonl(input_file)
            for line_number, source in iterator:
                stats["read_rows"] += 1
                try:
                    records = convert_record(source, input_file, target_file, args)
                except (KeyError, TypeError, ValueError) as error:
                    stats["invalid_rows"] += 1
                    if len(errors) < args.max_error_logs:
                        errors.append({"line": line_number, "error": str(error)})
                    if args.invalid_policy == "error":
                        raise ValueError(f"{input_file}:{line_number}: {error}") from error
                    continue
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stats["written_rows"] += 1
                stats["valid_source_rows"] += 1
                while next_progress and stats["read_rows"] >= next_progress:
                    print(
                        f"[progress] split={split} read={stats['read_rows']:,} "
                        f"written={stats['written_rows']:,} invalid={stats['invalid_rows']:,}",
                        flush=True,
                    )
                    next_progress += args.progress_every
                if args.max_records is not None and stats["valid_source_rows"] >= args.max_records:
                    break
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {input_file}: {error}") from error
        commit_output(stream, target_file)
    except BaseException:
        cleanup_output(stream)
        raise

    print(
        f"[ok] split={split} read={stats['read_rows']:,} written={stats['written_rows']:,} "
        f"invalid={stats['invalid_rows']:,}"
    )
    return {
        "input": str(input_file),
        "output": str(target_file),
        "stats": dict(stats),
        "invalid_examples": errors,
    }


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
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    report: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "task": args.task,
        "format": "ms-swift grounding JSONL",
        "splits": {},
    }
    for split in args.splits:
        report["splits"][split] = convert_split(split, args)
    write_report(args.output_dir / "visualgenome_regions_grounding_report.json", report, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
