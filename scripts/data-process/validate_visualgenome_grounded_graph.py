#!/usr/bin/env python3
"""Validate generated Visual Genome grounded-caption JSONL files."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_visualgenome_swift import find_json_source, is_validation_image, iter_json_array, open_json_source


DEFAULT_INPUT_DIR = Path("/mnt/luojunkun/stage1/dataset/VisualGenome")
DEFAULT_DATASET_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/visualgenome_grounded_graph")


def image_sizes(input_dir: Path) -> dict[int, tuple[int, int]]:
    result = {}
    source = find_json_source(input_dir, "image_data.json")
    with open_json_source(source) as stream:
        for row in iter_json_array(stream):
            image_id = int(row.get("image_id", row.get("id")))
            result[image_id] = (int(row["width"]), int(row["height"]))
    return result


def assistant_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    values = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    return values[-1] if len(values) == 1 and isinstance(values[-1], str) else ""


def repaired_interval(start: float, end: float, limit: int) -> tuple[float, float]:
    bounded_start = max(0.0, min(start, limit))
    bounded_end = max(0.0, min(end, limit))
    if bounded_start < bounded_end:
        return bounded_start, bounded_end
    if start >= limit and limit > 0:
        return float(limit - 1), float(limit)
    if end <= 0 and limit > 0:
        return 0.0, 1.0
    return bounded_start, bounded_end


def validate_row(
    row: dict[str, Any], split: str, sizes: dict[int, tuple[int, int]], stats: Counter, repair_bounds: bool
) -> None:
    image_id = int(row["image_id"])
    int(row["region_id"])
    expected_split = "val" if is_validation_image(image_id, 0.01, 42) else "train"
    if split != expected_split:
        raise ValueError(f"image_id={image_id} belongs to {expected_split}, not {split}")
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1 or not str(images[0]).endswith(f"/{image_id}.jpg"):
        raise ValueError("images must contain the matching Visual Genome JPEG")
    messages = row.get("messages")
    text = assistant_text(messages)
    if not text:
        raise ValueError("exactly one non-empty assistant message is required")
    image_tokens = sum(
        str(message.get("content") or "").count("<image>")
        for message in messages
        if isinstance(message, dict)
    )
    if image_tokens != 1:
        raise ValueError(f"expected one <image>, found {image_tokens}")

    objects = row.get("objects")
    if not isinstance(objects, dict) or objects.get("bbox_type") != "real":
        raise ValueError("objects.bbox_type must be real")
    refs = objects.get("ref")
    boxes = objects.get("bbox")
    image_ids = objects.get("image_id")
    if not isinstance(refs, list) or not refs or any(not str(value).strip() for value in refs):
        raise ValueError("objects.ref must contain non-empty values")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("objects.bbox must be non-empty")
    if text.count("<ref-object>") != len(refs):
        raise ValueError("<ref-object> count does not match objects.ref")
    if text.count("<bbox>") != len(boxes):
        raise ValueError("<bbox> count does not match objects.bbox")
    if not isinstance(image_ids, list) or image_ids != [0] * len(boxes):
        raise ValueError("objects.image_id must contain one zero per bbox")

    width, height = sizes[image_id]
    for index, box in enumerate(boxes):
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"invalid bbox shape: {box!r}")
        coordinates = [float(value) for value in box]
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"non-finite bbox: {box!r}")
        x1, y1, x2, y2 = coordinates
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            if not repair_bounds:
                raise ValueError(f"bbox outside {width}x{height}: {box!r}")
            x1, x2 = repaired_interval(x1, x2, width)
            y1, y2 = repaired_interval(y1, y2, height)
            repaired = [x1, y1, x2, y2]
            if not (repaired[0] < repaired[2] and repaired[1] < repaired[3]):
                raise ValueError(f"bbox cannot be repaired within {width}x{height}: {box!r}")
            boxes[index] = repaired
            stats["repaired_boxes"] += 1
    stats["refs"] += len(refs)
    stats["boxes"] += len(boxes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--progress-every", type=int, default=250000)
    parser.add_argument("--repair-bounds", action="store_true")
    args = parser.parse_args()
    sizes = image_sizes(args.input_dir.expanduser().resolve())
    dataset_dir = args.dataset_dir.expanduser().resolve()
    stats: Counter = Counter()
    region_ids: set[int] = set()
    errors = []
    replacements: list[tuple[Path, Path]] = []
    for split in ("train", "val"):
        path = dataset_dir / f"visualgenome_regions_{split}.jsonl"
        temporary = path.with_name(f"{path.name}.bounded.tmp")
        output = temporary.open("w", encoding="utf-8", newline="\n") if args.repair_bounds else None
        try:
            stream = path.open("r", encoding="utf-8-sig")
            for line_number, line in enumerate(stream, start=1):
                row = None
                stats[f"{split}_rows"] += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("row is not a JSON object")
                    region_id = int(row["region_id"])
                    if region_id in region_ids:
                        raise ValueError(f"duplicate region_id={region_id}")
                    region_ids.add(region_id)
                    validate_row(row, split, sizes, stats, args.repair_bounds)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    stats["invalid_rows"] += 1
                    if len(errors) < 20:
                        errors.append({"file": str(path), "line": line_number, "error": str(error)})
                if output is not None and row is not None:
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                total = stats["train_rows"] + stats["val_rows"]
                if args.progress_every and total % args.progress_every == 0:
                    print(f"[validate] rows={total:,} invalid={stats['invalid_rows']:,}", flush=True)
            stream.close()
        finally:
            if output is not None:
                output.close()
        if args.repair_bounds:
            replacements.append((temporary, path))
    report = {"status": "valid" if not errors else "invalid", "stats": dict(stats), "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if stats["invalid_rows"]:
        for temporary, _ in replacements:
            temporary.unlink(missing_ok=True)
        raise SystemExit(1)
    for temporary, path in replacements:
        os.replace(temporary, path)


if __name__ == "__main__":
    main()
