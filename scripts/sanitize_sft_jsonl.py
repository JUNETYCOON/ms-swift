#!/usr/bin/env python3
"""Stream-clean an ms-swift JSONL while preserving the first valid row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


MEDIA_KEYS = (("image", "images"), ("video", "videos"), ("audio", "audios"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rejected-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _fingerprint(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    rows = 0
    with path.open("rb") as stream:
        for line in stream:
            rows += 1
            hasher.update(line)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "rows": rows,
        "sha256": hasher.hexdigest(),
    }


def _token_count(messages: Sequence[Mapping[str, Any]], token: str) -> int:
    return sum(
        message.get("content", "").count(token)
        for message in messages
        if isinstance(message.get("content"), str)
    )


def _media_token_count(
    messages: Sequence[Mapping[str, Any]], token: str, media_present: bool
) -> int:
    if media_present:
        return _token_count(messages, token)
    return sum(
        message.get("content", "").count(token)
        for message in messages
        if message.get("role") != "assistant"
        and isinstance(message.get("content"), str)
    )


class RecordValidator:
    def __init__(self) -> None:
        self.image_dimensions: dict[str, tuple[int, int] | None] = {}

    def validate(self, record: Any) -> list[dict[str, str]]:
        errors: list[dict[str, str]] = []
        if not isinstance(record, dict):
            return [{"code": "invalid_record", "detail": "record must be an object"}]
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages or not all(
            isinstance(message, dict) for message in messages
        ):
            return [{"code": "invalid_messages", "detail": "messages must be a non-empty object list"}]

        for singular, plural in MEDIA_KEYS:
            media = record.get(plural, [])
            if media is None:
                media = []
            if not isinstance(media, list):
                errors.append({"code": "invalid_media_list", "detail": f"{plural} must be a list"})
                continue
            tokens = _media_token_count(messages, f"<{singular}>", bool(media))
            if tokens != len(media):
                errors.append(
                    {
                        "code": "media_placeholder_mismatch",
                        "detail": f"<{singular}>={tokens}, {plural}={len(media)}",
                    }
                )

        objects = record.get("objects")
        if not isinstance(objects, dict) or objects.get("bbox_type", "real") != "real":
            return errors
        boxes = objects.get("bbox") or []
        images = record.get("images") or []
        image_ids = objects.get("image_id")
        assigned = image_ids if isinstance(image_ids, list) else [0] * len(boxes)
        for box, image_index in zip(boxes, assigned):
            if (
                not isinstance(box, list)
                or len(box) not in {2, 4}
                or not isinstance(image_index, int)
                or isinstance(image_index, bool)
                or image_index < 0
                or image_index >= len(images)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in box
                )
            ):
                continue
            image_path = images[image_index]
            if not isinstance(image_path, str):
                continue
            dimensions = self.image_dimensions.get(image_path)
            if image_path not in self.image_dimensions:
                try:
                    from PIL import Image

                    with Image.open(image_path) as image:
                        dimensions = image.size
                except Exception as exc:
                    dimensions = None
                    errors.append(
                        {
                            "code": "image_decode_failed",
                            "detail": f"{image_path}: {exc}",
                        }
                    )
                self.image_dimensions[image_path] = dimensions
            if dimensions is None:
                continue
            width, height = dimensions
            if any(
                value < 0 or value > (width if position % 2 == 0 else height)
                for position, value in enumerate(box)
            ):
                errors.append(
                    {
                        "code": "real_bbox_out_of_bounds",
                        "detail": f"bbox={box}, image_size={dimensions}, image_id={image_index}",
                    }
                )
        return errors


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rejected = (
        args.rejected_output.expanduser().resolve()
        if args.rejected_output
        else output.with_name(f"{output.stem}_rejected.jsonl")
    )
    report_output = (
        args.report_output.expanduser().resolve()
        if args.report_output
        else output.with_name(f"{output.stem}_sanitization_report.json")
    )
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if not source.is_file():
        raise FileNotFoundError(source)
    if len({source, output, rejected, report_output}) != 4:
        raise ValueError("input, output, rejected output, and report output must differ")
    for path in (output, rejected, report_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {path}; pass --overwrite")

    output_tmp = _temporary_sibling(output)
    rejected_tmp = _temporary_sibling(rejected)
    report_tmp = _temporary_sibling(report_output)
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    validator = RecordValidator()
    try:
        with (
            source.open("r", encoding="utf-8-sig") as input_stream,
            output_tmp.open("w", encoding="utf-8", newline="\n") as output_stream,
            rejected_tmp.open("w", encoding="utf-8", newline="\n") as rejected_stream,
        ):
            for line_number, line in enumerate(input_stream, 1):
                counts["source_rows"] += 1
                normalized = line.rstrip("\r\n")
                digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()
                reasons: list[dict[str, str]] = []
                if digest in seen:
                    reasons.append({"code": "exact_duplicate_row", "detail": "same JSONL bytes seen earlier"})
                else:
                    seen.add(digest)
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        reasons.append({"code": "invalid_json", "detail": str(exc)})
                    else:
                        reasons.extend(validator.validate(record))
                if reasons:
                    counts["rejected_rows"] += 1
                    for reason in reasons:
                        counts[f"rejected::{reason['code']}"] += 1
                    payload = {
                        "source": str(source),
                        "line": line_number,
                        "row_blake2b_128": digest.hex(),
                        "reasons": reasons,
                    }
                    rejected_stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    if len(examples) < 100:
                        examples.append(payload)
                else:
                    output_stream.write(normalized + "\n")
                    counts["retained_rows"] += 1
                if args.progress_every and line_number % args.progress_every == 0:
                    print(
                        f"[sanitize] rows={line_number:,} keep={counts['retained_rows']:,} "
                        f"reject={counts['rejected_rows']:,}",
                        flush=True,
                    )

        payload = {
            "schema_version": 1,
            "status": "passed",
            "policy": {
                "exact_duplicate_rows": "keep_first",
                "real_bbox_out_of_bounds": "reject",
                "assistant_only_html_media_tags_without_media": "treat_as_text",
            },
            "counts": dict(sorted(counts.items())),
            "source": _fingerprint(source),
            "output": _fingerprint(output_tmp),
            "rejected": _fingerprint(rejected_tmp),
            "examples": examples,
        }
        payload["output"]["path"] = str(output)
        payload["rejected"]["path"] = str(rejected)
        with report_tmp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(output_tmp, output)
        os.replace(rejected_tmp, rejected)
        os.replace(report_tmp, report_output)
        print(
            f"[done] source={counts['source_rows']:,} retained={counts['retained_rows']:,} "
            f"rejected={counts['rejected_rows']:,} report={report_output}",
            flush=True,
        )
        return payload
    except BaseException:
        for path in (output_tmp, rejected_tmp, report_tmp):
            if path.exists():
                path.unlink()
        raise


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
