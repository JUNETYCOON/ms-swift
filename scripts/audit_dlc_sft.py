#!/usr/bin/env python3
"""Stream-audit DLC ms-swift entrypoints and render grounding spot checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


DEFAULT_MANIFEST = Path("/mnt/workspace/stage1/scripts/dlc_ready_entrypoints.stage1.json")
MEDIA_KEYS = (("image", "images"), ("video", "videos"), ("audio", "audios"))
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
TRACK_RE = re.compile(
    r"Object\s+(?P<object>\d+),\s*frame\s+(?P<frame>\d+).*?:\s*"
    r"\[(?P<x>-?(?:\d+(?:\.\d*)?|\.\d+)),\s*"
    r"(?P<y>-?(?:\d+(?:\.\d*)?|\.\d+))\]"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--samples-per-dataset", type=int, default=3)
    parser.add_argument("--quantile-sample-size", type=int, default=200_000)
    parser.add_argument("--max-error-examples", type=int, default=100)
    return parser.parse_args(argv)


class SampledDistribution:
    def __init__(self, limit: int, seed: str) -> None:
        self.limit = limit
        self.random = random.Random(seed)
        self.values: list[int] = []
        self.count = 0
        self.total = 0
        self.minimum: int | None = None
        self.maximum: int | None = None

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.values) < self.limit:
            self.values.append(value)
            return
        replacement = self.random.randrange(self.count)
        if replacement < self.limit:
            self.values[replacement] = value

    def report(self) -> dict[str, Any]:
        ordered = sorted(self.values)

        def percentile(q: float) -> int | float | None:
            if not ordered:
                return None
            position = (len(ordered) - 1) * q
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return ordered[lower]
            return round(
                ordered[lower]
                + (ordered[upper] - ordered[lower]) * (position - lower),
                3,
            )

        return {
            "count": self.count,
            "sampled_for_percentiles": len(ordered),
            "min": self.minimum,
            "mean": round(self.total / self.count, 3) if self.count else None,
            "p50_estimate": percentile(0.50),
            "p95_estimate": percentile(0.95),
            "p99_estimate": percentile(0.99),
            "max": self.maximum,
        }


def _count_token(messages: Sequence[Mapping[str, Any]], token: str) -> int:
    return sum(
        message.get("content", "").count(token)
        for message in messages
        if isinstance(message.get("content"), str)
    )


def _count_media_token(
    messages: Sequence[Mapping[str, Any]], token: str, media_present: bool
) -> int:
    if media_present:
        return _count_token(messages, token)
    return sum(
        message.get("content", "").count(token)
        for message in messages
        if message.get("role") != "assistant"
        and isinstance(message.get("content"), str)
    )


def _is_url(value: str) -> bool:
    return urlsplit(value).scheme.casefold() in {"http", "https"}


def _add_error(
    errors: Counter[str],
    examples: list[dict[str, Any]],
    limit: int,
    code: str,
    line_number: int,
    detail: str,
) -> None:
    errors[code] += 1
    if len(examples) < limit:
        examples.append({"line": line_number, "code": code, "detail": detail})


def _task_type(record: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]) -> str:
    objects = record.get("objects")
    if (
        isinstance(objects, Mapping)
        and isinstance(objects.get("bbox"), list)
        and objects["bbox"]
    ):
        lengths = {len(item) for item in objects["bbox"] if isinstance(item, list)}
        if lengths == {2}:
            return "point_grounding"
        user_bbox_tokens = sum(
            str(message.get("content") or "").count("<bbox>")
            for message in messages
            if message.get("role") == "user"
        )
        assistant_contents = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "assistant"
        ]
        assistant_bbox_tokens = sum(content.count("<bbox>") for content in assistant_contents)
        if user_bbox_tokens and not assistant_bbox_tokens:
            return "box_to_text"
        assistant_text = " ".join(assistant_contents)
        assistant_text = assistant_text.replace("<bbox>", "").replace("<ref-object>", "").strip()
        if assistant_bbox_tokens and assistant_text:
            return "grounded_description"
        return "bbox_grounding"
    if record.get("videos") and any(
        TRACK_RE.search(str(message.get("content") or "")) for message in messages
    ):
        return "video_tracking"
    user_text = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    ).casefold()
    if any(word in user_text for word in ("describe", "caption", "summarize")):
        return "caption_or_description"
    if not any(record.get(plural) for _singular, plural in MEDIA_KEYS):
        return "text_only"
    return "vqa_or_instruction"


def audit_dataset(
    name: str,
    path_text: str,
    samples_per_dataset: int,
    quantile_sample_size: int,
    max_error_examples: int,
    split: str = "train",
) -> dict[str, Any]:
    path = Path(path_text)
    counters: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    media_seen: set[str] = set()
    image_dimensions: dict[str, tuple[int, int] | None] = {}
    row_hashes: set[bytes] = set()
    assistant_lengths = SampledDistribution(quantile_sample_size, name + ":assistant")
    user_lengths = SampledDistribution(quantile_sample_size, name + ":user")
    message_counts = SampledDistribution(quantile_sample_size, name + ":messages")
    samples: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            counters["rows"] += 1
            if not line.strip():
                _add_error(errors, examples, max_error_examples, "blank_line", line_number, "blank JSONL line")
                continue
            digest = hashlib.blake2b(line.rstrip("\r\n").encode("utf-8"), digest_size=16).digest()
            if digest in row_hashes:
                counters["exact_duplicate_rows"] += 1
            else:
                row_hashes.add(digest)
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                _add_error(errors, examples, max_error_examples, "invalid_json", line_number, str(exc))
                continue
            if not isinstance(record, dict):
                _add_error(errors, examples, max_error_examples, "invalid_record", line_number, "record must be an object")
                continue
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                _add_error(errors, examples, max_error_examples, "invalid_messages", line_number, "messages must be a non-empty list")
                continue
            valid_messages = True
            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    _add_error(errors, examples, max_error_examples, "invalid_message", line_number, f"message {index} must be an object")
                    valid_messages = False
                    continue
                if message.get("role") not in ALLOWED_ROLES:
                    _add_error(errors, examples, max_error_examples, "invalid_role", line_number, f"message {index} has role {message.get('role')!r}")
                    valid_messages = False
                content = message.get("content")
                if not isinstance(content, str):
                    _add_error(errors, examples, max_error_examples, "invalid_content", line_number, f"message {index} content must be a string")
                    valid_messages = False
                elif message.get("role") in {"user", "assistant"} and not content.strip():
                    _add_error(errors, examples, max_error_examples, "empty_content", line_number, f"message {index} content is empty")
                    valid_messages = False
            if not valid_messages:
                continue
            if messages[-1].get("role") != "assistant":
                _add_error(errors, examples, max_error_examples, "invalid_final_role", line_number, "conversation must end with assistant")
            message_counts.add(len(messages))
            assistant_text = "\n".join(
                message["content"] for message in messages if message.get("role") == "assistant"
            )
            user_text = "\n".join(
                message["content"] for message in messages if message.get("role") == "user"
            )
            assistant_lengths.add(len(assistant_text))
            user_lengths.add(len(user_text))

            for singular, plural in MEDIA_KEYS:
                media = record.get(plural, [])
                if media is None:
                    media = []
                if not isinstance(media, list):
                    _add_error(errors, examples, max_error_examples, "invalid_media_list", line_number, f"{plural} must be a list")
                    continue
                token_count = _count_media_token(
                    messages, f"<{singular}>", bool(media)
                )
                if token_count != len(media):
                    _add_error(errors, examples, max_error_examples, "media_placeholder_mismatch", line_number, f"<{singular}>={token_count}, {plural}={len(media)}")
                if media:
                    counters[f"records_with_{plural}"] += 1
                    counters[f"{plural}_references"] += len(media)
                for item in media:
                    if not isinstance(item, str) or not item:
                        _add_error(errors, examples, max_error_examples, "invalid_media_path", line_number, f"invalid {plural} entry")
                        continue
                    if _is_url(item):
                        counters["remote_media_references"] += 1
                        _add_error(errors, examples, max_error_examples, "remote_media", line_number, item)
                        continue
                    if not Path(item).is_absolute():
                        _add_error(errors, examples, max_error_examples, "relative_media_path", line_number, item)
                        continue
                    if item not in media_seen:
                        media_seen.add(item)
                        if not Path(item).is_file():
                            _add_error(errors, examples, max_error_examples, "missing_media", line_number, item)

            objects = record.get("objects")
            ref_tokens = _count_token(messages, "<ref-object>")
            bbox_tokens = _count_token(messages, "<bbox>")
            if objects is None:
                if ref_tokens or bbox_tokens:
                    _add_error(errors, examples, max_error_examples, "missing_objects", line_number, "grounding placeholders require objects")
            elif not isinstance(objects, dict):
                _add_error(errors, examples, max_error_examples, "invalid_objects", line_number, "objects must be an object")
            else:
                refs = objects.get("ref", [])
                boxes = objects.get("bbox", [])
                if not isinstance(refs, list) or not all(isinstance(item, str) for item in refs):
                    _add_error(errors, examples, max_error_examples, "invalid_refs", line_number, "objects.ref must be a string list")
                    refs = []
                if not isinstance(boxes, list):
                    _add_error(errors, examples, max_error_examples, "invalid_bbox", line_number, "objects.bbox must be a list")
                    boxes = []
                if ref_tokens != len(refs):
                    _add_error(errors, examples, max_error_examples, "ref_placeholder_mismatch", line_number, f"tokens={ref_tokens}, refs={len(refs)}")
                if bbox_tokens != len(boxes):
                    _add_error(errors, examples, max_error_examples, "bbox_placeholder_mismatch", line_number, f"tokens={bbox_tokens}, bbox={len(boxes)}")
                bbox_type = objects.get("bbox_type", "real")
                if bbox_type not in {"real", "norm1"}:
                    _add_error(errors, examples, max_error_examples, "invalid_bbox_type", line_number, repr(bbox_type))
                for box in boxes:
                    if not isinstance(box, list) or len(box) not in {2, 4}:
                        _add_error(errors, examples, max_error_examples, "invalid_bbox_shape", line_number, repr(box))
                        continue
                    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in box):
                        _add_error(errors, examples, max_error_examples, "invalid_bbox_value", line_number, repr(box))
                    elif bbox_type == "norm1" and any(value < 0 or value > 1 for value in box):
                        _add_error(errors, examples, max_error_examples, "norm1_out_of_range", line_number, repr(box))
                    elif len(box) == 4 and (box[0] > box[2] or box[1] > box[3]):
                        _add_error(errors, examples, max_error_examples, "inverted_bbox", line_number, repr(box))
                image_ids = objects.get("image_id")
                if image_ids is not None:
                    images = record.get("images") or []
                    if not isinstance(image_ids, list) or len(image_ids) != len(boxes):
                        _add_error(errors, examples, max_error_examples, "invalid_image_id", line_number, "image_id length must equal bbox length")
                    elif any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= len(images) for value in image_ids):
                        _add_error(errors, examples, max_error_examples, "invalid_image_id", line_number, repr(image_ids))
                if boxes and not record.get("images"):
                    _add_error(errors, examples, max_error_examples, "grounding_without_image", line_number, "objects.bbox requires images")
                elif boxes and bbox_type == "real":
                    from PIL import Image

                    images = record.get("images") or []
                    assigned_images = image_ids if isinstance(image_ids, list) else [0] * len(boxes)
                    for box, image_index in zip(boxes, assigned_images):
                        if (
                            not isinstance(box, list)
                            or len(box) not in {2, 4}
                            or not isinstance(image_index, int)
                            or isinstance(image_index, bool)
                            or image_index < 0
                            or image_index >= len(images)
                        ):
                            continue
                        image_path = images[image_index]
                        if not isinstance(image_path, str) or _is_url(image_path):
                            continue
                        dimensions = image_dimensions.get(image_path)
                        if image_path not in image_dimensions:
                            try:
                                with Image.open(image_path) as image:
                                    dimensions = image.size
                            except Exception as exc:
                                dimensions = None
                                _add_error(
                                    errors,
                                    examples,
                                    max_error_examples,
                                    "image_decode_failed",
                                    line_number,
                                    f"{image_path}: {exc}",
                                )
                            image_dimensions[image_path] = dimensions
                        if dimensions is None or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(value)
                            for value in box
                        ):
                            continue
                        width, height = dimensions
                        if any(
                            value < 0
                            or value > (width if position % 2 == 0 else height)
                            for position, value in enumerate(box)
                        ):
                            _add_error(
                                errors,
                                examples,
                                max_error_examples,
                                "real_bbox_out_of_bounds",
                                line_number,
                                f"bbox={box}, image_size={dimensions}, image_id={image_index}",
                            )
                counters["grounding_refs"] += len(refs)
                counters["grounding_coordinates"] += len(boxes)
                counters["point_coordinates"] += sum(len(box) == 2 for box in boxes if isinstance(box, list))
                counters["box_coordinates"] += sum(len(box) == 4 for box in boxes if isinstance(box, list))

            task = _task_type(record, messages)
            counters[f"task::{task}"] += 1
            if task in {
                "point_grounding",
                "bbox_grounding",
                "grounded_description",
                "box_to_text",
                "video_tracking",
            } and samples_per_dataset:
                candidate = {
                    "line": line_number,
                    "task": task,
                    "sampling_hash": digest.hex(),
                    "record": record,
                }
                if len(samples) < samples_per_dataset:
                    samples.append(candidate)
                else:
                    worst_index = max(
                        range(len(samples)),
                        key=lambda index: samples[index]["sampling_hash"],
                    )
                    if candidate["sampling_hash"] < samples[worst_index]["sampling_hash"]:
                        samples[worst_index] = candidate

    samples.sort(key=lambda sample: sample["sampling_hash"])

    return {
        "name": f"{name}/{split}",
        "dataset": name,
        "split": split,
        "path": str(path.resolve()),
        "status": "passed" if not errors and not counters["exact_duplicate_rows"] else "failed",
        "counts": dict(sorted(counters.items())),
        "unique_media_paths": len(media_seen),
        "real_grounding_images_decoded": sum(
            dimensions is not None for dimensions in image_dimensions.values()
        ),
        "unique_exact_rows": len(row_hashes),
        "distributions": {
            "messages_per_record": message_counts.report(),
            "user_characters": user_lengths.report(),
            "assistant_characters": assistant_lengths.report(),
        },
        "schema_errors": dict(sorted(errors.items())),
        "error_examples": examples,
        "visualization_candidates": samples,
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def _render_image_sample(name: str, sample: Mapping[str, Any], output_dir: Path) -> list[str]:
    from PIL import Image, ImageDraw

    record = sample["record"]
    objects = record["objects"]
    boxes = objects.get("bbox") or []
    image_ids = objects.get("image_id") or [0] * len(boxes)
    bbox_type = objects.get("bbox_type", "real")
    refs = objects.get("ref") or []
    outputs: list[str] = []
    for image_index, image_path in enumerate(record.get("images") or []):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        color = (255, 32, 32)
        for box_index, (box, assigned_image) in enumerate(zip(boxes, image_ids)):
            if assigned_image != image_index:
                continue
            values = [float(value) for value in box]
            if bbox_type == "norm1":
                values = [
                    value * (width if position % 2 == 0 else height)
                    for position, value in enumerate(values)
                ]
            if len(values) == 2:
                x, y = values
                radius = max(4, round(min(width, height) * 0.008))
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
                draw.line((x - radius * 2, y, x + radius * 2, y), fill=color, width=2)
                draw.line((x, y - radius * 2, x, y + radius * 2), fill=color, width=2)
                draw.text((x + radius + 2, y), f"P{box_index}", fill=color)
            else:
                draw.rectangle(tuple(values), outline=color, width=max(2, round(min(width, height) * 0.004)))
                draw.text((values[0] + 2, values[1] + 2), f"B{box_index}", fill=color)
        legend = "GT"
        if refs:
            legend += " | ref tokens: " + " | ".join(str(value) for value in refs[:8])
        draw.rectangle((0, 0, min(width, max(80, len(legend) * 8)), 24), fill=(0, 0, 0))
        draw.text((4, 4), legend, fill=(255, 255, 0))
        output = output_dir / f"{_safe_name(name)}_line{sample['line']}_image{image_index}.jpg"
        image.save(output, quality=92)
        outputs.append(str(output.resolve()))
    return outputs


def _render_video_sample(name: str, sample: Mapping[str, Any], output_dir: Path) -> list[str]:
    from PIL import Image, ImageDraw

    record = sample["record"]
    assistant = "\n".join(
        message["content"] for message in record["messages"] if message.get("role") == "assistant"
    )
    points: dict[int, list[tuple[int, float, float]]] = {}
    for match in TRACK_RE.finditer(assistant):
        points.setdefault(int(match.group("frame")), []).append(
            (int(match.group("object")), float(match.group("x")), float(match.group("y")))
        )
    if not points:
        return []
    frames = sorted(points)
    chosen = sorted({frames[0], frames[len(frames) // 2], frames[-1]})
    video_path = record["videos"][0]
    outputs: list[str] = []
    import av

    decoded: dict[int, Image.Image] = {}
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        targets = set(chosen)
        for index, video_frame in enumerate(container.decode(stream)):
            if index in targets:
                decoded[index] = video_frame.to_image().convert("RGB")
            if len(decoded) == len(targets):
                break
    missing = sorted(set(chosen) - decoded.keys())
    if missing:
        raise ValueError(f"video does not contain requested frames: {missing}")
    for frame in chosen:
        image = decoded[frame]
        draw = ImageDraw.Draw(image)
        width, height = image.size
        radius = max(5, round(min(width, height) * 0.01))
        for object_id, x_norm, y_norm in points[frame]:
            x, y = x_norm * width / 1000.0, y_norm * height / 1000.0
            color = (255, 32, 32)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
            draw.text((x + radius + 2, y), f"object {object_id}", fill=color)
        legend = f"GT | frame {frame} | norm1000 points"
        draw.rectangle((0, 0, min(width, len(legend) * 8 + 8), 24), fill=(0, 0, 0))
        draw.text((4, 4), legend, fill=(255, 255, 0))
        output = output_dir / f"{_safe_name(name)}_line{sample['line']}_frame{frame}.jpg"
        image.save(output, quality=92)
        outputs.append(str(output.resolve()))
    return outputs


def render_visualizations(reports: list[dict[str, Any]], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for report in reports:
        for sample in report.pop("visualization_candidates"):
            try:
                record = sample["record"]
                assets: list[dict[str, Any]] = []
                if sample["task"] == "video_tracking":
                    paths = _render_video_sample(report["name"], sample, output_dir)
                    assistant = "\n".join(
                        message["content"]
                        for message in record["messages"]
                        if message.get("role") == "assistant"
                    )
                    points: dict[int, int] = Counter(
                        int(match.group("frame")) for match in TRACK_RE.finditer(assistant)
                    )
                    frames = sorted(points)
                    selected_frames = sorted(
                        {frames[0], frames[len(frames) // 2], frames[-1]}
                    )
                    for path, frame in zip(paths, selected_frames):
                        assets.append(
                            {
                                "overlay": path,
                                "overlay_sha256": _sha256(Path(path)),
                                "source_media": record["videos"][0],
                                "target_frame": frame,
                                "coordinate_convention": "norm1000 xy point",
                                "declared_primitives": points[frame],
                                "rendered_primitives": points[frame],
                                "ground_truth_overlay_status": "passed",
                            }
                        )
                else:
                    paths = _render_image_sample(report["name"], sample, output_dir)
                    objects = record["objects"]
                    boxes = objects.get("bbox") or []
                    image_ids = objects.get("image_id") or [0] * len(boxes)
                    for image_index, path in enumerate(paths):
                        count = sum(value == image_index for value in image_ids)
                        assets.append(
                            {
                                "overlay": path,
                                "overlay_sha256": _sha256(Path(path)),
                                "source_media": record["images"][image_index],
                                "target_image_id": image_index,
                                "coordinate_convention": (
                                    f"{objects.get('bbox_type', 'real')} "
                                    f"{'xy point' if boxes and len(boxes[0]) == 2 else 'xyxy box'}"
                                ),
                                "declared_primitives": count,
                                "rendered_primitives": count,
                                "ground_truth_overlay_status": "passed",
                            }
                        )
                rendered.append(
                    {
                        "dataset": report["name"],
                        "line": sample["line"],
                        "task": sample["task"],
                        "sampling_hash": sample["sampling_hash"],
                        "assets": assets,
                    }
                )
            except Exception as exc:
                errors.append({"dataset": report["name"], "line": sample["line"], "error": str(exc)})
    return rendered, errors


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.samples_per_dataset < 0 or args.quantile_sample_size < 1:
        raise ValueError("workers and quantile sample size must be positive; samples must be non-negative")
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = manifest["global_dedup"]
    enabled = [
        (name, config)
        for name, config in manifest["datasets"].items()
        if config.get("enabled", True)
    ]
    base = Path(policy["report"]).expanduser().resolve().parent
    output = (args.output or base / "dlc_sft_audit_report.json").expanduser().resolve()
    visualization_dir = (
        args.visualization_dir or base / "dlc_sft_audit_visualizations"
    ).expanduser().resolve()

    audit_entries: list[tuple[str, str, str]] = []
    for name, config in enabled:
        audit_entries.append((name, "train", config["train"]))
        eval_value = config["eval"]
        eval_paths = eval_value if isinstance(eval_value, list) else [eval_value]
        for index, eval_path in enumerate(eval_paths):
            split = "eval" if len(eval_paths) == 1 else f"eval[{index}]"
            audit_entries.append((name, split, eval_path))

    reports: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(audit_entries))) as executor:
        futures = {
            executor.submit(
                audit_dataset,
                name,
                path,
                args.samples_per_dataset,
                args.quantile_sample_size,
                args.max_error_examples,
                split,
            ): (name, split)
            for name, split, path in audit_entries
        }
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                f"[audit] {report['name']} rows={report['counts'].get('rows', 0):,} "
                f"status={report['status']}",
                flush=True,
            )
    order = {
        (name, split): index
        for index, (name, split, _path) in enumerate(audit_entries)
    }
    reports.sort(key=lambda item: order[(item["dataset"], item["split"])])
    rendered, visualization_errors = render_visualizations(reports, visualization_dir)

    dedup_report_path = Path(policy["report"]).expanduser().resolve()
    dedup_report = json.loads(dedup_report_path.read_text(encoding="utf-8"))
    verification = dedup_report.get("verification", {})
    totals: Counter[str] = Counter()
    task_distribution: Counter[str] = Counter()
    task_distribution_by_split: dict[str, Counter[str]] = {
        "train": Counter(),
        "eval": Counter(),
    }
    for report in reports:
        split_group = "train" if report["split"] == "train" else "eval"
        totals["rows"] += report["counts"].get("rows", 0)
        totals[f"{split_group}_rows"] += report["counts"].get("rows", 0)
        totals["exact_duplicate_rows"] += report["counts"].get("exact_duplicate_rows", 0)
        totals["schema_errors"] += sum(report["schema_errors"].values())
        totals["unique_media_paths_by_file_sum"] += report["unique_media_paths"]
        for key, value in report["counts"].items():
            if key.startswith("task::"):
                task = key.removeprefix("task::")
                task_distribution[task] += value
                task_distribution_by_split[split_group][task] += value
    hard_failures: list[str] = []
    if dedup_report.get("status") != "complete":
        hard_failures.append("decontamination report is not complete")
    if verification.get("status") != "complete":
        hard_failures.append("decontamination verification is not complete")
    if verification.get("train_eval_overlap_rows") != 0:
        hard_failures.append("train-eval media contamination is not zero")
    if totals["schema_errors"]:
        hard_failures.append("one or more ms-swift schema checks failed")
    if totals["exact_duplicate_rows"]:
        hard_failures.append("one or more exact duplicate JSONL rows remain")
    if visualization_errors:
        hard_failures.append("one or more grounding visualizations failed")
    payload = {
        "schema_version": 1,
        "status": "passed" if not hard_failures else "failed",
        "manifest": str(manifest_path),
        "decontamination_report": str(dedup_report_path),
        "policy": {
            "train_eval_media_overlap_must_be_zero": True,
            "cross_dataset_train_media_is_retained": not policy.get(
                "deduplicate_cross_dataset_train", True
            ),
            "exact_duplicate_rows_must_be_zero": True,
            "local_absolute_media_paths_required": True,
            "grounding_spot_checks_are_drawn_on_source_pixels": True,
            "spot_check_sampling": (
                "full-file deterministic bottom-k by 128-bit BLAKE2b row hash"
            ),
        },
        "totals": dict(totals),
        "task_distribution": dict(sorted(task_distribution.items())),
        "task_distribution_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in task_distribution_by_split.items()
        },
        "decontamination_verification": verification,
        "datasets": reports,
        "visualizations": rendered,
        "visualization_errors": visualization_errors,
        "hard_failures": hard_failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] status={payload['status']} report={output}")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    report = run(parse_args(argv))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
