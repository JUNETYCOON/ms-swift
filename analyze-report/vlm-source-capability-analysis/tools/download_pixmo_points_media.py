#!/usr/bin/env python3
"""Validate PixMo-Points source URLs locally in deterministic draw order."""

from __future__ import annotations

import hashlib
import json
import struct
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATASET_ROOT = HERE.parent / "sub-dataset" / "PixMo-Points"
TARGET = 200
BATCH_SIZE = 32
MAX_BYTES = 30 * 1024 * 1024


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def download(url: str) -> dict[str, Any]:
    if not url:
        return {"error": "missing_media_url"}
    request = urllib.request.Request(
        url, headers={"User-Agent": "source-dataset-audit/1.0"}
    )
    error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = response.read(MAX_BYTES + 1)
                if len(payload) > MAX_BYTES:
                    return {"error": "remote_media_over_30mb"}
                return {
                    "payload": payload,
                    "content_type": response.headers.get("Content-Type"),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
        except urllib.error.HTTPError as exc:
            error = exc
            if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        time.sleep(0.25 * (attempt + 1))
    if isinstance(error, urllib.error.HTTPError):
        return {"error": f"remote_http_error:{error.code}"}
    return {"error": f"remote_fetch_error:{type(error).__name__}:{str(error)[:160]}"}


def jpeg_dimensions(payload: bytes) -> tuple[int, int] | None:
    if not payload.startswith(b"\xff\xd8"):
        return None
    offset = 2
    while offset + 9 <= len(payload):
        if payload[offset] != 0xFF:
            offset += 1
            continue
        marker = payload[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(payload):
            return None
        length = struct.unpack(">H", payload[offset : offset + 2])[0]
        if length < 2 or offset + length > len(payload):
            return None
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            height, width = struct.unpack(">HH", payload[offset + 3 : offset + 7])
            return width, height
        offset += length
    return None


def inspect_image(payload: bytes) -> dict[str, Any]:
    if payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 24:
        width, height = struct.unpack(">II", payload[16:24])
        return {"image_format": "png", "width": width, "height": height}
    jpeg = jpeg_dimensions(payload)
    if jpeg:
        return {"image_format": "jpeg", "width": jpeg[0], "height": jpeg[1]}
    if payload[:6] in {b"GIF87a", b"GIF89a"} and len(payload) >= 10:
        width, height = struct.unpack("<HH", payload[6:10])
        return {"image_format": "gif", "width": width, "height": height}
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP" and len(payload) >= 30:
        chunk = payload[12:16]
        if chunk == b"VP8X":
            width = 1 + int.from_bytes(payload[24:27], "little")
            height = 1 + int.from_bytes(payload[27:30], "little")
            return {"image_format": "webp", "width": width, "height": height}
        if chunk == b"VP8 " and len(payload) >= 30:
            marker = payload.find(b"\x9d\x01\x2a", 20)
            if marker >= 0 and marker + 7 <= len(payload):
                width, height = struct.unpack("<HH", payload[marker + 3 : marker + 7])
                return {
                    "image_format": "webp",
                    "width": width & 0x3FFF,
                    "height": height & 0x3FFF,
                }
    return {"error": "unsupported_or_invalid_image_header"}


def point_count(raw: dict[str, Any]) -> int:
    points = raw.get("points") or []
    if points and isinstance(points[0], list):
        return sum(len(group or []) for group in points)
    return len(points)


def main() -> int:
    candidates = load_jsonl(DATASET_ROOT / "candidate-records.jsonl")
    summary = json.loads(
        (DATASET_ROOT / "sampling-summary.json").read_text(encoding="utf-8")
    )
    media_dir = DATASET_ROOT / "media"
    record_dir = DATASET_ROOT / "records"
    media_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    decisions = []
    seen_hashes = set()
    next_draw = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        while next_draw < len(candidates) and len(manifest) < TARGET:
            batch = candidates[next_draw : next_draw + BATCH_SIZE]
            fetched = list(
                executor.map(lambda item: download(str(item.get("source_media_url") or "")), batch)
            )
            for candidate, result in zip(batch, fetched):
                locator = {
                    key: candidate[key]
                    for key in ("draw_order", "population_index", "shard", "shard_row", "split")
                }
                if len(manifest) >= TARGET:
                    decisions.append({**locator, "reason": "reserve_candidate_not_needed"})
                    continue
                if result.get("error"):
                    decisions.append(
                        {
                            **locator,
                            "reason": result["error"],
                            "source_media_url": candidate.get("source_media_url"),
                        }
                    )
                    continue
                payload = result["payload"]
                digest = hashlib.sha256(payload).hexdigest()
                expected = str(candidate.get("expected_sha256") or "").lower()
                if expected and expected != digest:
                    decisions.append(
                        {
                            **locator,
                            "reason": "source_sha256_mismatch",
                            "expected_sha256": expected,
                            "actual_sha256": digest,
                        }
                    )
                    continue
                if digest in seen_hashes:
                    decisions.append(
                        {**locator, "reason": "duplicate_media_sha256", "sha256": digest}
                    )
                    continue
                inspected = inspect_image(payload)
                if inspected.get("error"):
                    decisions.append(
                        {**locator, "reason": inspected["error"], "sha256": digest}
                    )
                    continue
                seen_hashes.add(digest)
                sample_id = f"pixmo-points-{len(manifest) + 1:03d}"
                extension = "jpg" if inspected["image_format"] == "jpeg" else inspected["image_format"]
                media_path = f"media/{sample_id}.{extension}"
                record_path = f"records/{sample_id}.json"
                media_target = DATASET_ROOT / media_path
                media_temporary = media_target.with_suffix(media_target.suffix + ".tmp")
                media_temporary.write_bytes(payload)
                media_temporary.replace(media_target)
                raw = candidate.get("raw_record") or {}
                write_json(
                    DATASET_ROOT / record_path,
                    {"source_locator": candidate["source_uri"], "raw_record": raw},
                )
                count = point_count(raw)
                manifest.append(
                    {
                        **locator,
                        "sample_id": sample_id,
                        "source_uri": candidate["source_uri"],
                        "source_media_url": candidate.get("source_media_url"),
                        "media_archive_path": media_path,
                        "record_archive_path": record_path,
                        "sha256": digest,
                        "byte_length": len(payload),
                        **inspected,
                        "http_etag": result.get("etag"),
                        "http_last_modified": result.get("last_modified"),
                        "label": raw.get("label"),
                        "collection_method": raw.get("collection_method"),
                        "annotated_count": raw.get("count"),
                        "point_annotation_count": count,
                        "input_preview": f"Locate or count: {raw.get('label') or '<missing>'}",
                        "output_preview": f"points={count}; count={raw.get('count')}",
                    }
                )
            next_draw += len(batch)

    for candidate in candidates[next_draw:]:
        decisions.append(
            {
                **{
                    key: candidate[key]
                    for key in ("draw_order", "population_index", "shard", "shard_row", "split")
                },
                "reason": "reserve_candidate_not_needed",
            }
        )

    reasons = Counter(item["reason"] for item in decisions)
    summary.update(
        {
            "selected_count": len(manifest),
            "candidate_count": len(candidates),
            "invalid_or_duplicate_count": sum(
                reason != "reserve_candidate_not_needed" for reason in reasons.elements()
            ),
            "reserve_candidate_count": reasons.get("reserve_candidate_not_needed", 0),
            "incomplete_visual_sample": len(manifest) < TARGET,
            "incomplete_reason": (
                None if len(manifest) >= TARGET else "fewer than 200 source-URL images passed hash/header validation"
            ),
            "media_validation_location": "local downloader using original source URLs",
            "candidate_decision_reason_counts": dict(reasons.most_common()),
        }
    )
    write_jsonl(DATASET_ROOT / "sampling-manifest.jsonl", manifest)
    write_jsonl(DATASET_ROOT / "sampling-candidate-decisions.jsonl", decisions)
    write_json(DATASET_ROOT / "sampling-summary.json", summary)
    print(
        json.dumps(
            {
                "selected": len(manifest),
                "decisions": len(decisions),
                "reasons": dict(reasons.most_common()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
