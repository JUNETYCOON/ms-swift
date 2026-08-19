#!/usr/bin/env python3
"""Attach objects.bbox (bbox_type=norm1000) to Molmo2-VideoTrack JSONL.

Input is the ms-swift JSONL produced by prepare_molmo_pixmo_spatial_swift.py
(messages + videos, assistant text contains point-track lines such as
"Object 0, frame 0 (0.000s): [123, 456]"). The adapter parses every point,
replaces the coordinate with a <bbox> token, and writes the same media/split
mapping with objects.bbox + bbox_type=norm1000.

The converter is deterministic, streams both files, uses a bounded process
pool, preserves source ordering, and validates placeholder/coordinate counts,
media existence and train/val video isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


POINT_LINE_RE = re.compile(
    r"^(?P<prefix>Object \d+, frame \d+ \([^)]*\)): \[(?P<x>-?\d+(?:\.\d+)?), (?P<y>-?\d+(?:\.\d+)?)\]$"
)
NO_POINT_TEXT = "No visible track points were annotated."


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def convert_record(payload: tuple[str, int, dict[str, Any]]) -> dict[str, Any]:
    split, line_no, record = payload

    def rejected(reason: str, detail: str = "") -> dict[str, Any]:
        return {
            "status": "rejected",
            "split": split,
            "line_no": line_no,
            "reason": reason,
            "detail": detail,
            "record": record,
        }

    videos = record.get("videos") or []
    if len(videos) != 1:
        return rejected("invalid_video_count", f"videos={len(videos)}")

    messages = record.get("messages") or []
    assistant_idx = None
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            assistant_idx = index
            break
    if assistant_idx is None:
        return rejected("missing_assistant_message")

    assistant = messages[assistant_idx].get("content")
    if not isinstance(assistant, str):
        return rejected("assistant_content_not_string")

    bbox: list[list[float]] = []
    new_lines: list[str] = []
    for line in assistant.split("\n"):
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        match = POINT_LINE_RE.match(stripped)
        if match:
            x, y = match.group("x"), match.group("y")
            if not _finite(x) or not _finite(y):
                return rejected("non_finite_coordinate", stripped)
            value_x, value_y = float(x), float(y)
            if not (0.0 <= value_x <= 1000.0 and 0.0 <= value_y <= 1000.0):
                return rejected("coordinate_out_of_range", stripped)
            bbox.append([round(value_x, 6), round(value_y, 6)])
            new_lines.append(f"{match.group('prefix')}: <bbox>")
            continue
        if stripped == NO_POINT_TEXT:
            new_lines.append(line)
            continue
        return rejected("unparsed_assistant_line", stripped[:300])

    new_record = dict(record)
    new_messages = [dict(message) for message in messages]
    new_messages[assistant_idx]["content"] = "\n".join(new_lines)
    new_record["messages"] = new_messages
    new_record["objects"] = {"ref": [], "bbox": bbox, "bbox_type": "norm1000"}

    object_ids: list[int] = []
    frames: list[int] = []
    for line in assistant.split("\n"):
        match = re.match(r"^Object (\d+), frame (\d+) ", line)
        if match:
            object_ids.append(int(match.group(1)))
            frames.append(int(match.group(2)))

    return {
        "status": "no_bbox" if not bbox else "converted",
        "split": split,
        "line_no": line_no,
        "record": new_record,
        "meta": {
            "video": videos[0],
            "bbox_count": len(bbox),
            "object_ids": object_ids,
            "frames": frames,
        },
    }


def route_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return convert_record(payload)


def iter_records(path: Path, split: str):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield split, line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield {
                    "status": "rejected",
                    "split": split,
                    "line_no": line_no,
                    "reason": "invalid_json",
                    "detail": str(exc),
                    "record": None,
                }


def process_file(path: Path, split: str, num_workers: int):
    payloads = iter_records(path, split)
    if num_workers <= 1:
        for payload in payloads:
            yield route_payload(payload)
        return
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        yield from executor.map(route_payload, payloads, chunksize=32)


def sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_outputs(
    train_path: Path,
    eval_path: Path,
    status_counter: Counter,
    report: dict[str, Any],
) -> None:
    placeholder_errors: list[str] = []
    range_errors: list[str] = []
    missing_media: list[str] = []
    train_videos: set[str] = set()
    eval_videos: set[str] = set()
    bbox_total = 0
    converted_rows = 0

    for path, split, video_set in (
        (train_path, "train", train_videos),
        (eval_path, "eval", eval_videos),
    ):
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                record = json.loads(line)
                objects = record.get("objects")
                if not isinstance(objects, dict) or objects.get("bbox_type") != "norm1000":
                    range_errors.append(f"{split}:{line_no}:missing_bbox_type")
                    continue
                bbox = objects.get("bbox") or []
                bbox_total += len(bbox)
                if bbox:
                    converted_rows += 1
                for index, box in enumerate(bbox):
                    if len(box) not in {2, 4}:
                        range_errors.append(f"{split}:{line_no}:bbox_len={len(box)}")
                        continue
                    if any(not _finite(v) or not (0.0 <= float(v) <= 1000.0) for v in box):
                        range_errors.append(f"{split}:{line_no}:box_index={index}:out_of_range")
                assistant = next(
                    (m.get("content") for m in record.get("messages", []) if m.get("role") == "assistant"),
                    "",
                )
                bbox_tokens = assistant.count("<bbox>")
                if bbox_tokens != len(bbox):
                    placeholder_errors.append(f"{split}:{line_no}:tokens={bbox_tokens}:bbox={len(bbox)}")
                for video in record.get("videos") or []:
                    video_set.add(video)
                    if not os.path.exists(video):
                        missing_media.append(f"{split}:{line_no}:{video}")

    report["validation"] = {
        "bbox_rows": converted_rows,
        "bbox_total": bbox_total,
        "placeholder_errors": placeholder_errors[:50],
        "placeholder_error_count": len(placeholder_errors),
        "range_errors": range_errors[:50],
        "range_error_count": len(range_errors),
        "missing_media": missing_media[:50],
        "missing_media_count": len(missing_media),
        "train_video_count": len(train_videos),
        "eval_video_count": len(eval_videos),
        "cross_split_video_overlap": sorted(train_videos & eval_videos)[:50],
        "cross_split_overlap_count": len(train_videos & eval_videos),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--eval-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.train_input.exists() or not args.eval_input.exists():
        raise SystemExit("train/eval input files must exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": args.output_dir / "ready_train_norm1000.jsonl",
        "eval": args.output_dir / "ready_eval_norm1000.jsonl",
        "rejected": args.output_dir / "ready_norm1000_rejected.jsonl",
        "metadata": args.output_dir / "ready_norm1000_track_metadata.jsonl",
        "report": args.output_dir / "ready_norm1000_conversion_report.json",
    }
    if any(path.exists() for path in outputs.values()) and not args.overwrite:
        raise SystemExit(f"outputs already exist, pass --overwrite: {args.output_dir}")

    counters: Counter = Counter()
    rejections: list[dict[str, Any]] = []

    with outputs["train"].open("w", encoding="utf-8") as train_out, \
         outputs["eval"].open("w", encoding="utf-8") as eval_out, \
         outputs["rejected"].open("w", encoding="utf-8") as rejected_out, \
         outputs["metadata"].open("w", encoding="utf-8") as meta_out:
        for outcome in process_file(args.train_input, "train", args.num_workers):
            _write_outcome(outcome, train_out, eval_out, rejected_out, meta_out, counters, rejections)
        for outcome in process_file(args.eval_input, "eval", args.num_workers):
            _write_outcome(outcome, train_out, eval_out, rejected_out, meta_out, counters, rejections)

    report: dict[str, Any] = {
        "dataset": "Molmo2-VideoTrack",
        "converter": Path(__file__).name,
        "inputs": {
            "train": str(args.train_input),
            "eval": str(args.eval_input),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "counts": {f"{split}:{status}": count for (split, status), count in sorted(counters.items())},
        "rejections": rejections[:50],
        "rejection_count": len(rejections),
    }
    validate_outputs(outputs["train"], outputs["eval"], counters, report)
    report["output_sha256"] = {
        name: sha256_hex(path)
        for name, path in outputs.items()
        if path.exists()
    }
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _write_outcome(
    outcome: dict[str, Any],
    train_out,
    eval_out,
    rejected_out,
    meta_out,
    counters: Counter,
    rejections: list[dict[str, Any]],
) -> None:
    key = (outcome["split"], outcome["status"])
    counters[key] += 1
    if outcome["status"] == "rejected":
        rejected_out.write(json.dumps(outcome, ensure_ascii=False) + "\n")
        rejections.append(outcome)
        return
    target = eval_out if outcome["split"] == "eval" else train_out
    target.write(json.dumps(outcome["record"], ensure_ascii=False) + "\n")
    meta_out.write(json.dumps(outcome["meta"], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
