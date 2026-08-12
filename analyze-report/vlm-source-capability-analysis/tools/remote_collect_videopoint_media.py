#!/usr/bin/env python3
"""Collect deterministic generated-video previews for Molmo2-VideoPoint.

The complete generated-video ID population is read from the canonical source
Parquet. Candidate IDs are sampled once with a fixed seed. Source tar streams
are scanned read-only; selected source videos are hashed and decoded with PyAV.
Only derived JPEG previews and traceable source records are returned.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import av
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path("/mnt/luojunkun/stage1/dataset/Molmo2-VideoPoint")
PARQUET_PATH = ROOT / "data" / "train-00000-of-00001.parquet"
ARCHIVE_DIR = ROOT / "generated_videos"
SEED = 2026080508
CANDIDATES = 400
TARGET = 200


def retry(operation, attempts: int = 10):
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except (FileNotFoundError, OSError) as exc:
            error = exc
            time.sleep(min(0.25 * (attempt + 1), 2.0))
    assert error is not None
    raise error


def safe_value(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "<max-depth>"
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, str):
        return value if len(value) <= 4_000 else value[:4_000] + "<truncated>"
    if isinstance(value, dict):
        return {str(key): safe_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [safe_value(item, depth + 1) for item in value[:30]]
        if len(value) > 30:
            items.append({"truncated_items": len(value) - 30})
        return items
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def source_files() -> list[dict[str, Any]]:
    output = []
    for current, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(name for name in dirs if name not in {".cache", "__pycache__"})
        current_path = Path(current)
        for name in sorted(files):
            path = current_path / name
            stat = retry(path.stat)
            output.append(
                {
                    "relative_path": path.relative_to(ROOT).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return output


def fingerprint(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(files, key=lambda item: item["relative_path"]):
        digest.update(
            f"{row['relative_path']}\t{row['size']}\t{row['mtime_ns']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def generated_video_population() -> list[dict[str, Any]]:
    parquet = retry(lambda: pq.ParquetFile(PARQUET_PATH))
    first_rows: dict[str, int] = {}
    offset = 0
    for batch in parquet.iter_batches(
        batch_size=8192, columns=["video_id", "video_source"]
    ):
        sources = batch.column(batch.schema.get_field_index("video_source"))
        mask = pc.equal(sources, "generated")
        indices = pc.indices_nonzero(mask).to_pylist()
        ids = batch.column(batch.schema.get_field_index("video_id"))
        for index in indices:
            video_id = str(ids[index].as_py() or "")
            if video_id and video_id not in first_rows:
                first_rows[video_id] = offset + index
        offset += batch.num_rows
    parquet.close()
    return [
        {"video_id": video_id, "source_row": source_row}
        for video_id, source_row in sorted(first_rows.items())
    ]


def read_rows(source_rows: list[int]) -> dict[int, dict[str, Any]]:
    parquet = retry(lambda: pq.ParquetFile(PARQUET_PATH))
    wanted = set(source_rows)
    output = {}
    offset = 0
    for group_index in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        group_rows = [row for row in wanted if offset <= row < offset + count]
        if group_rows:
            table = retry(lambda group_index=group_index: parquet.read_row_group(group_index))
            for source_row in group_rows:
                output[source_row] = table.slice(source_row - offset, 1).to_pylist()[0]
                wanted.remove(source_row)
        offset += count
        if not wanted:
            break
    parquet.close()
    return output


def normalized_member_id(name: str) -> str:
    path = PurePosixPath(name)
    if len(path.parts) < 2:
        return path.stem
    return f"{path.parts[0]}/{path.stem}"


def decode_preview(payload: bytes) -> dict[str, Any]:
    try:
        with av.open(io.BytesIO(payload), mode="r") as container:
            streams = list(container.streams.video)
            if not streams:
                return {"error": "video_has_no_video_stream"}
            stream = streams[0]
            image = None
            for frame in container.decode(stream):
                image = frame.to_image()
                break
            if image is None:
                return {"error": "video_has_no_decodable_frame"}
            preview = io.BytesIO()
            image.convert("RGB").save(preview, format="JPEG", quality=88, optimize=True)
            rate = float(stream.average_rate) if stream.average_rate else None
            duration = None
            if container.duration is not None:
                duration = float(container.duration / av.time_base)
            return {
                "preview": preview.getvalue(),
                "width": image.width,
                "height": image.height,
                "video_codec": stream.codec_context.name,
                "video_fps": rate,
                "video_frames_declared": stream.frames or None,
                "video_duration_seconds": duration,
            }
    except Exception as exc:
        return {"error": f"video_decode_error:{type(exc).__name__}:{str(exc)[:180]}"}


def add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    tar.addfile(info, io.BytesIO(payload))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )


def main() -> int:
    before_files = source_files()
    before_digest = fingerprint(before_files)
    population = generated_video_population()
    candidate_indices = random.Random(SEED).sample(
        range(len(population)), min(CANDIDATES, len(population))
    )
    candidates = [
        {
            "draw_order": draw_order,
            "population_index": population_index,
            **population[population_index],
        }
        for draw_order, population_index in enumerate(candidate_indices)
    ]
    by_id = {item["video_id"]: item for item in candidates}
    rows = read_rows([item["source_row"] for item in candidates])

    media_results: dict[str, dict[str, Any]] = {}
    duplicate_members: Counter[str] = Counter()
    archive_counts: dict[str, int] = {}
    for archive_path in sorted(ARCHIVE_DIR.glob("*.tar.gz")):
        relative_archive = archive_path.relative_to(ROOT).as_posix()
        regular_members = 0
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                regular_members += 1
                video_id = normalized_member_id(member.name)
                if video_id not in by_id:
                    continue
                if video_id in media_results:
                    duplicate_members[video_id] += 1
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    media_results[video_id] = {"error": "tar_member_stream_unavailable"}
                    continue
                payload = stream.read()
                source_hash = hashlib.sha256(payload).hexdigest()
                decoded = decode_preview(payload)
                media_results[video_id] = {
                    **decoded,
                    "source_archive": relative_archive,
                    "source_member": member.name,
                    "source_video_sha256": source_hash,
                    "source_video_byte_length": len(payload),
                }
        archive_counts[relative_archive] = regular_members

    manifest = []
    decisions = []
    selected_payloads = []
    seen_video_hashes = set()
    for candidate in candidates:
        video_id = candidate["video_id"]
        public_locator = {
            "draw_order": candidate["draw_order"],
            "population_index": candidate["population_index"],
            "video_id": video_id,
            "source_row": candidate["source_row"],
            "source_shard": "data/train-00000-of-00001.parquet",
        }
        media = media_results.get(video_id)
        if media is None:
            decisions.append({**public_locator, "reason": "generated_video_member_not_found"})
            continue
        if media.get("error"):
            decisions.append(
                {
                    **public_locator,
                    "reason": media["error"],
                    "source_archive": media.get("source_archive"),
                    "source_member": media.get("source_member"),
                }
            )
            continue
        video_hash = media["source_video_sha256"]
        if video_hash in seen_video_hashes:
            decisions.append(
                {**public_locator, "reason": "duplicate_source_video_sha256", "source_video_sha256": video_hash}
            )
            continue
        if len(manifest) >= TARGET:
            decisions.append({**public_locator, "reason": "reserve_candidate_not_needed"})
            continue
        seen_video_hashes.add(video_hash)
        sample_id = f"molmo2-videopoint-{len(manifest) + 1:03d}"
        preview_path = f"derived-preview/{sample_id}.jpg"
        record_path = f"media-records/{sample_id}.json"
        preview_hash = hashlib.sha256(media["preview"]).hexdigest()
        row = rows[candidate["source_row"]]
        item = {
            **public_locator,
            "sample_id": sample_id,
            "preview_archive_path": preview_path,
            "record_archive_path": record_path,
            "preview_sha256": preview_hash,
            "source_video_sha256": video_hash,
            "source_video_byte_length": media["source_video_byte_length"],
            "source_archive": media["source_archive"],
            "source_member": media["source_member"],
            "source_uri": f"{ROOT}/{media['source_archive']}#{media['source_member']}",
            "preview_generation": "PyAV first decodable video frame; JPEG quality=88; source video not copied",
            "preview_width": media["width"],
            "preview_height": media["height"],
            "video_codec": media["video_codec"],
            "video_fps": media["video_fps"],
            "video_frames_declared": media["video_frames_declared"],
            "video_duration_seconds": media["video_duration_seconds"],
            "input_preview": str(row.get("question") or "")[:1_000],
            "output_preview": (
                f"label={row.get('label')}; count={row.get('count')}; "
                f"timestamp_groups={len(row.get('points') or [])}"
            )[:1_000],
        }
        manifest.append(item)
        selected_payloads.append((item, media["preview"], row))

    after_files = source_files()
    after_digest = fingerprint(after_files)
    summary = {
        "dataset": "Molmo2-VideoPoint",
        "source_root": str(ROOT),
        "sampling_unit": "unique_video",
        "population_total": len(population),
        "population_definition": "unique video_id where canonical source video_source == generated",
        "annotation_population_total": 658340,
        "seed": SEED,
        "candidate_count": len(candidates),
        "selected_count": len(manifest),
        "invalid_or_duplicate_count": sum(
            item["reason"] != "reserve_candidate_not_needed" for item in decisions
        ),
        "reserve_candidate_count": sum(
            item["reason"] == "reserve_candidate_not_needed" for item in decisions
        ),
        "sampling_method": "fixed-seed uniform permutation over all unique generated video IDs from canonical source Parquet",
        "archive_regular_member_counts": archive_counts,
        "duplicate_candidate_member_count": sum(duplicate_members.values()),
        "source_video_bytes_archived": False,
        "preview_only": True,
        "read_only_source": True,
        "incomplete_visual_sample": len(manifest) < TARGET,
        "source_fingerprint_before": before_digest,
        "source_fingerprint_after": after_digest,
        "source_unchanged": before_digest == after_digest,
    }
    evidence = {
        "dataset": "Molmo2-VideoPoint",
        "access_mode": "ssh",
        "source_root": str(ROOT),
        "fingerprint_algorithm": "sha256 over sorted relative path, size, and mtime_ns",
        "before_digest": before_digest,
        "after_digest": after_digest,
        "unchanged": before_digest == after_digest,
        "collector_transport": "collector code via stdin; tar artifacts via stdout",
        "temporary_workspace": None,
        "read_operations": ["Parquet metadata read", "sequential source tar read", "source video decode"],
    }

    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as output:
        add_bytes(output, "sampling-manifest.jsonl", jsonl_bytes(manifest))
        add_bytes(output, "sampling-candidate-decisions.jsonl", jsonl_bytes(decisions))
        add_bytes(output, "media-sampling-summary.json", json_bytes(summary))
        add_bytes(output, "source-media-evidence-entry.json", json_bytes(evidence))
        for item, preview, row in selected_payloads:
            add_bytes(output, item["preview_archive_path"], preview)
            add_bytes(
                output,
                item["record_archive_path"],
                json_bytes(
                    {
                        "source_locator": (
                            f"{PARQUET_PATH}#row={item['source_row']}"
                        ),
                        "source_media_locator": item["source_uri"],
                        "raw_record": safe_value(row),
                    }
                ),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
