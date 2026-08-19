#!/usr/bin/env python3
"""Normalize ms-swift grounding bbox/point coordinates to bbox_type=norm1000.

Reads a JSONL file in ms-swift format, converts objects.bbox from real or
norm1 to 0..1000 coordinates, writes a new JSONL, and emits a processing
report. All original record fields are preserved; only objects.bbox and
objects.bbox_type are rewritten.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - report a clear error when missing
    Image = None


def finite_float(value: Any, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: non-numeric coordinate {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: non-finite coordinate {value!r}")
    return number


def box_kind(box: Any) -> str:
    if not isinstance(box, list):
        raise ValueError("bbox entry must be a list")
    if len(box) == 2:
        return "point"
    if len(box) == 4:
        return "box"
    raise ValueError(f"bbox entry must contain 2 or 4 values, got {len(box)}")


def image_size(path: str, image_root: Path | None) -> tuple[float, float]:
    if Image is None:
        raise ValueError("Pillow is required to convert real-coordinate bbox")
    candidate = Path(path)
    if not candidate.is_absolute() and image_root is not None:
        candidate = image_root / candidate
    with Image.open(candidate) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions for {path}")
    return float(width), float(height)


def normalize_box(
    box: list[Any],
    bbox_type: str,
    image: str | None,
    image_root: Path | None,
) -> list[float]:
    kind = box_kind(box)
    values = [finite_float(value, "bbox coordinate") for value in box]
    normalized: list[float]

    if bbox_type == "norm1000":
        normalized = values
    elif bbox_type == "norm1":
        normalized = [value * 1000.0 for value in values]
    elif bbox_type == "real":
        if not image:
            raise ValueError("real-coordinate bbox requires images")
        width, height = image_size(image, image_root)
        if kind == "point":
            normalized = [values[0] / width * 1000.0, values[1] / height * 1000.0]
        else:
            normalized = [
                values[0] / width * 1000.0,
                values[1] / height * 1000.0,
                values[2] / width * 1000.0,
                values[3] / height * 1000.0,
            ]
    else:
        raise ValueError(f"unsupported bbox_type: {bbox_type!r}")

    if any(value < 0.0 or value > 1000.0 for value in normalized):
        raise ValueError(f"norm1000 coordinate out of range: {normalized}")
    if kind == "box" and (normalized[2] < normalized[0] or normalized[3] < normalized[1]):
        raise ValueError(f"inverted xyxy box after normalization: {normalized}")
    return [round(value, 6) for value in normalized]


def process_row(record: dict[str, Any], image_root: Path | None) -> dict[str, Any]:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return {"record": record, "status": "no_bbox"}
    raw_boxes = objects.get("bbox")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        return {"record": record, "status": "no_bbox"}

    bbox_type = str(objects.get("bbox_type") or "real").lower()
    images = record.get("images") or []
    image_ids = objects.get("image_id") or []
    if not isinstance(images, list):
        images = []
    if not isinstance(image_ids, list):
        image_ids = []

    normalized_boxes: list[list[float]] = []
    for index, box in enumerate(raw_boxes):
        image = None
        if bbox_type == "real":
            image_index = int(image_ids[index]) if index < len(image_ids) else 0
            image = str(images[image_index]) if images and image_index < len(images) else None
        normalized_boxes.append(normalize_box(box, bbox_type, image, image_root))

    normalized = dict(record)
    normalized_objects = dict(objects)
    normalized_objects["bbox"] = normalized_boxes
    normalized_objects["bbox_type"] = "norm1000"
    normalized["objects"] = normalized_objects
    return {"record": normalized, "status": "converted", "source_type": bbox_type}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source ms-swift JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output norm1000 JSONL")
    parser.add_argument("--report", type=Path, required=True, help="Processing report JSON")
    parser.add_argument("--image-root", type=Path, default=None, help="Base directory for relative image paths")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output/report files")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}")
    if args.report.exists() and not args.overwrite:
        raise SystemExit(f"report already exists: {args.report}")

    counts = Counter()
    by_source_type = Counter()
    by_error = Counter()
    errors: list[dict[str, Any]] = []

    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                outcome = process_row(record, args.image_root)
            except Exception as exc:
                counts["rejected"] += 1
                by_error[type(exc).__name__] += 1
                errors.append({"line_no": line_no, "error": str(exc), "error_type": type(exc).__name__})
                continue
            counts[outcome["status"]] += 1
            if outcome["status"] == "converted":
                by_source_type[outcome["source_type"]] += 1
                target.write(json.dumps(outcome["record"], ensure_ascii=False) + "\n")
            else:
                target.write(json.dumps(outcome["record"], ensure_ascii=False) + "\n")

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "image_root": str(args.image_root) if args.image_root else None,
        "counts": dict(counts),
        "by_source_bbox_type": dict(by_source_type),
        "by_error_type": dict(by_error),
        "errors": errors[:50],
        "total_errors": len(errors),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
