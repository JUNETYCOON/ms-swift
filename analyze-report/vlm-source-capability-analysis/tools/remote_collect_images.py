#!/usr/bin/env python3
"""Collect source-only image samples and full statistics as a tar stream."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pyarrow.compute as pc
from PIL import Image


SOURCE_BASE = Path("/mnt/luojunkun/stage1/dataset")
TARGET = 200
CANDIDATES = 400
CONFIGS = {
    "PixMo-Cap": {
        "source_dir": "pixmo-cap",
        "files": "data/*.parquet",
        "mode": "remote_image",
        "seed": 2026080501,
        "url_field": "image_url",
        "identity_field": "image_url",
        "text_fields": ("caption",),
        "sampling_unit": "logical_record",
        "task": "description.long_caption",
        "candidate_count": 800,
    },
    "PixMo-Points": {
        "source_dir": "pixmo-points",
        "files": "data/*.parquet",
        "mode": "remote_image",
        "seed": 2026080502,
        "url_field": "image_url",
        "identity_field": "image_sha256",
        "expected_sha_field": "image_sha256",
        "text_fields": ("label",),
        "sampling_unit": "logical_record",
        "task": "pointing_or_counting",
        "candidate_count": 800,
        "full_scan_excluded_fields": ("points",),
    },
    "SpatialVLM": {
        "source_dir": "spatialvlm",
        "files": "data/*.parquet",
        "mode": "embedded_chat",
        "seed": 2026080503,
        "sampling_unit": "logical_record",
        "task": "spatial_vqa",
        "candidate_count": 400,
    },
}


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


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )


def add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    tar.addfile(info, io.BytesIO(payload))


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


def source_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in {".cache", "__pycache__"})
        current_path = Path(current)
        for name in sorted(files):
            path = current_path / name
            stat = retry(path.stat)
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return rows


def fingerprint(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(files, key=lambda item: item["relative_path"]):
        digest.update(
            f"{row['relative_path']}\t{row['size']}\t{row['mtime_ns']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def split_name(relative: str) -> str:
    name = Path(relative).name.lower()
    if name.startswith("validation"):
        return "validation"
    if name.startswith("test"):
        return "test"
    if name.startswith("val"):
        return "val"
    if name.startswith("train"):
        return "train"
    return "unspecified"


def parquet_index(root: Path, pattern: str) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(root.glob(pattern)):
        parquet = retry(lambda path=path: pq.ParquetFile(path))
        reports.append(
            {
                "path": path,
                "relative": path.relative_to(root).as_posix(),
                "split": split_name(path.name),
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
                "schema": parquet.schema_arrow,
            }
        )
        parquet.close()
    return reports


def candidate_locations(
    index: list[dict[str, Any]], seed: int, candidate_count: int
) -> list[dict[str, Any]]:
    total = sum(item["rows"] for item in index)
    rng = random.Random(seed)
    draws = rng.sample(range(total), min(candidate_count, total))
    locations = []
    for draw_order, global_row in enumerate(draws):
        remaining = global_row
        for shard in index:
            if remaining >= shard["rows"]:
                remaining -= shard["rows"]
                continue
            locations.append(
                {
                    "draw_order": draw_order,
                    "population_index": global_row,
                    "shard": shard["relative"],
                    "shard_row": remaining,
                    "split": shard["split"],
                }
            )
            break
    return locations


def read_candidate_rows(root: Path, locations: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in locations:
        grouped[item["shard"]].append(item)
    output: dict[int, dict[str, Any]] = {}
    for relative, targets in grouped.items():
        parquet = retry(lambda: pq.ParquetFile(root / relative))
        remaining = {item["shard_row"]: item for item in targets}
        offset = 0
        for group_index in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group_index).num_rows
            wanted = [row for row in remaining if offset <= row < offset + count]
            if wanted:
                table = retry(lambda group_index=group_index: parquet.read_row_group(group_index))
                for row_index in wanted:
                    locator = remaining.pop(row_index)
                    output[locator["draw_order"]] = {
                        "locator": locator,
                        "record": table.slice(row_index - offset, 1).to_pylist()[0],
                    }
            offset += count
            if not remaining:
                break
        parquet.close()
    return output


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return " ".join(str(item.get("text") or "") for item in content if isinstance(item, dict)).strip()


def classify_spatial(question: str) -> list[str]:
    text = question.lower()
    labels = []
    rules = {
        "spatial.directional_relation": ("left", "right", "above", "below", "under", "over", "behind", "front"),
        "spatial.metric_size": ("measure", "width", "height", "tall", "size", "feet", "meter", "inch"),
        "spatial.distance": ("distance", "distant", "far", "close to", "near"),
        "spatial.depth_order": ("closer", "closest", "farther", "farthest", "depth"),
        "spatial.relative_scale": ("larger", "smaller", "bigger", "shorter", "longer"),
        "vqa.object_attribute": ("what color", "what appliance", "what object", "what are"),
    }
    for label, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            labels.append(label)
    return labels or ["vqa.general"]


def full_statistics(cfg: dict[str, Any], root: Path, index: list[dict[str, Any]]) -> dict[str, Any]:
    field_nulls: Counter[str] = Counter()
    field_non_nulls: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    unique_media: set[str] = set()
    text_chars = 0
    qa_pairs = 0
    point_total = 0
    record_count = 0
    collection_methods: Counter[str] = Counter()
    representative_records = []

    if index and index[0]["rows"]:
        first = retry(lambda: pq.ParquetFile(index[0]["path"]))
        representative_records = [
            {"source": f"{index[0]['relative']}#row={row_index}", "record": safe_value(row)}
            for row_index, row in enumerate(first.read_row_group(0).slice(0, 3).to_pylist())
        ]
        first.close()

    if cfg["mode"] == "remote_image":
        scan_columns = list(
            dict.fromkeys(
                [
                    cfg["url_field"],
                    cfg["identity_field"],
                    *cfg["text_fields"],
                    "collection_method",
                    "count",
                    "points",
                ]
            )
        )
        scan_columns = [
            column
            for column in scan_columns
            if column not in cfg.get("full_scan_excluded_fields", ())
        ]
        available = {field.name for field in index[0]["schema"]}
        scan_columns = [column for column in scan_columns if column in available]
    else:
        scan_columns = None

    for shard in index:
        parquet = retry(lambda shard=shard: pq.ParquetFile(shard["path"]))
        split_counts[shard["split"]] += shard["rows"]
        for batch in parquet.iter_batches(batch_size=8192, columns=scan_columns):
            if cfg["mode"] == "remote_image":
                record_count += batch.num_rows
                columns = {
                    name: batch.column(batch.schema.get_field_index(name))
                    for name in batch.schema.names
                }
                for field, column in columns.items():
                    field_nulls[field] += column.null_count
                    field_non_nulls[field] += len(column) - column.null_count

                identity_column = columns.get(cfg["identity_field"])
                if identity_column is not None:
                    unique_media.update(
                        str(value) for value in pc.unique(identity_column).to_pylist() if value
                    )
                for field in cfg["text_fields"]:
                    column = columns.get(field)
                    if column is not None:
                        length_sum = pc.sum(pc.utf8_length(column)).as_py()
                        text_chars += int(length_sum or 0)

                if cfg["task"] == "description.long_caption":
                    task_counts["description"] += batch.num_rows
                    capability_counts["general.long_description"] += batch.num_rows
                else:
                    method_column = columns["collection_method"]
                    for value in pc.value_counts(method_column).to_pylist():
                        collection_methods[str(value["values"] or "<missing>")] += int(value["counts"])
                    if "points" in columns:
                        lengths = pc.list_value_length(columns["points"])
                        point_total += int(pc.sum(lengths).as_py() or 0)
                continue

            rows = batch.to_pylist()
            for row in rows:
                record_count += 1
                for field, value in row.items():
                    (field_nulls if value is None else field_non_nulls)[field] += 1

                images = row.get("images") or []
                for image in images:
                    if isinstance(image, dict) and image.get("bytes"):
                        unique_media.add(hashlib.sha256(image["bytes"]).hexdigest())
                messages = row.get("messages") or []
                pending_question = ""
                for message in messages:
                    role = str(message.get("role") or "")
                    text = message_text(message.get("content"))
                    text_chars += len(text)
                    if role == "user":
                        pending_question = text
                    elif role == "assistant":
                        qa_pairs += 1
                        task_counts["spatial_vqa"] += 1
                        for label in classify_spatial(pending_question):
                            capability_counts[label] += 1
        parquet.close()

    if cfg["task"] == "pointing_or_counting":
        counting = collection_methods.get("counting", 0)
        task_counts["vqa.counting"] = counting
        task_counts["pointing"] = record_count - counting
        capability_counts["perception.counting"] = counting
        capability_counts["grounding.pointing"] = record_count - counting

    return {
        "scope": "full source population",
        "source_records": record_count,
        "records_by_split": dict(sorted(split_counts.items())),
        "unique_media_identities": len(unique_media),
        "qa_pairs": qa_pairs,
        "text_characters": text_chars,
        "supervised_token_estimate": round(text_chars / 4),
        "point_annotations": (
            point_total if "points" not in cfg.get("full_scan_excluded_fields", ()) else None
        ),
        "point_annotations_evidence": (
            "full-population statistic"
            if "points" not in cfg.get("full_scan_excluded_fields", ())
            else "not scanned over the full population; estimate from deterministic candidates"
        ),
        "task_counts": dict(task_counts.most_common()),
        "capability_counts": dict(capability_counts.most_common()),
        "collection_method_counts": dict(collection_methods.most_common()),
        "field_null_counts": dict(sorted(field_nulls.items())),
        "field_non_null_counts": dict(sorted(field_non_nulls.items())),
        "representative_records": representative_records,
    }


def download_image(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "source-dataset-audit/1.0"})
    error = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = response.read(30 * 1024 * 1024 + 1)
                if len(payload) > 30 * 1024 * 1024:
                    return {"error": "remote_media_over_30mb"}
                return {
                    "payload": payload,
                    "content_type": response.headers.get("Content-Type"),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
            time.sleep(0.25 * (attempt + 1))
    return {"error": f"remote_fetch_error:{type(error).__name__}:{str(error)[:180]}"}


def image_from_record(cfg: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if cfg["mode"] == "remote_image":
        url = str(row.get(cfg["url_field"]) or "")
        if not url:
            return {"error": "missing_media_url"}
        result = download_image(url)
        result["source_media_url"] = url
        return result
    images = row.get("images") or []
    if not images or not isinstance(images[0], dict) or not images[0].get("bytes"):
        return {"error": "missing_embedded_image_bytes"}
    return {
        "payload": images[0]["bytes"],
        "source_image_path": images[0].get("path"),
        "content_type": None,
    }


def decode_image(payload: bytes) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            return {
                "width": image.width,
                "height": image.height,
                "image_format": (image.format or "bin").lower(),
                "image_mode": image.mode,
            }
    except Exception as exc:
        return {"error": f"image_decode_error:{type(exc).__name__}:{str(exc)[:180]}"}


def preview_fields(cfg: dict[str, Any], row: dict[str, Any]) -> tuple[str, str]:
    if cfg["task"] == "description.long_caption":
        return "Describe the image in detail.", str(row.get("caption") or "")
    if cfg["task"] == "pointing_or_counting":
        question = f"Locate or count: {row.get('label') or '<missing>'}"
        answer = f"points={len(row.get('points') or [])}; count={row.get('count')}"
        return question, answer
    messages = row.get("messages") or []
    question = ""
    answer = ""
    for message in messages:
        role = str(message.get("role") or "")
        text = message_text(message.get("content"))
        if role == "user" and not question:
            question = text
        elif role == "assistant" and question and not answer:
            answer = text
            break
    return question, answer


def collect(dataset: str) -> None:
    cfg = CONFIGS[dataset]
    root = SOURCE_BASE / cfg["source_dir"]
    before_files = source_files(root)
    before_digest = fingerprint(before_files)
    index = parquet_index(root, cfg["files"])
    population_total = sum(item["rows"] for item in index)
    stats = full_statistics(cfg, root, index)
    locations = candidate_locations(index, cfg["seed"], cfg["candidate_count"])
    loaded = read_candidate_rows(root, locations)

    if cfg["task"] == "pointing_or_counting":
        candidate_point_counts = []
        for draw_order in range(len(locations)):
            points = loaded[draw_order]["record"].get("points") or []
            if points and isinstance(points[0], list):
                candidate_point_counts.append(sum(len(group or []) for group in points))
            else:
                candidate_point_counts.append(len(points))
        stats["candidate_point_annotation_estimate"] = {
            "evidence_type": "Sample estimate",
            "sample_size": len(candidate_point_counts),
            "seed": cfg["seed"],
            "minimum": min(candidate_point_counts) if candidate_point_counts else None,
            "maximum": max(candidate_point_counts) if candidate_point_counts else None,
            "mean": (
                sum(candidate_point_counts) / len(candidate_point_counts)
                if candidate_point_counts
                else None
            ),
            "zero_point_records": sum(value == 0 for value in candidate_point_counts),
        }

    if os.environ.get("PIXMO_METADATA_ONLY") == "1":
        after_files = source_files(root)
        after_digest = fingerprint(after_files)
        schema_variants = []
        schema_counter: Counter[str] = Counter(str(item["schema"]) for item in index)
        for schema, count in schema_counter.items():
            fields = next(item["schema"] for item in index if str(item["schema"]) == schema)
            schema_variants.append(
                {
                    "file_count": count,
                    "fields": [
                        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                        for field in fields
                    ],
                    "arrow_schema": schema,
                }
            )
        source_schema = {
            "dataset": dataset,
            "observed_from": "original source Parquet only",
            "schema_variants": schema_variants,
            "field_null_counts": stats["field_null_counts"],
            "field_non_null_counts": stats["field_non_null_counts"],
            "representative_records": stats["representative_records"],
        }
        candidate_records = []
        pending_decisions = []
        for draw_order in range(len(locations)):
            locator = loaded[draw_order]["locator"]
            row = loaded[draw_order]["record"]
            candidate_records.append(
                {
                    **locator,
                    "source_uri": f"{root}/{locator['shard']}#row={locator['shard_row']}",
                    "source_media_url": row.get(cfg["url_field"]),
                    "expected_sha256": row.get(cfg.get("expected_sha_field")),
                    "raw_record": safe_value(row),
                }
            )
            pending_decisions.append({**locator, "reason": "media_validation_pending"})
        summary = {
            "dataset": dataset,
            "source_root": str(root),
            "sampling_unit": cfg["sampling_unit"],
            "population_total": population_total,
            "population_by_split": stats["records_by_split"],
            "seed": cfg["seed"],
            "candidate_count": len(locations),
            "selected_count": 0,
            "invalid_or_duplicate_count": len(locations),
            "reserve_candidate_count": 0,
            "sampling_method": "fixed-seed uniform permutation over all canonical source Parquet rows; media validation performed locally from source URLs",
            "read_only_source": True,
            "incomplete_visual_sample": True,
            "incomplete_reason": "media_validation_pending",
            "source_fingerprint_before": before_digest,
            "source_fingerprint_after": after_digest,
            "source_unchanged": before_digest == after_digest,
            "full_statistics": stats,
            "media_storage": "remote URL",
        }
        evidence = {
            "dataset": dataset,
            "access_mode": "ssh",
            "source_root": str(root),
            "fingerprint_algorithm": "sha256 over sorted relative path, size, and mtime_ns",
            "before_digest": before_digest,
            "after_digest": after_digest,
            "unchanged": before_digest == after_digest,
            "collector_transport": "collector code via stdin; metadata tar stream via stdout",
            "temporary_workspace": None,
            "read_operations": ["directory enumeration", "Parquet footer/read", "candidate metadata read"],
        }
        layout = "\n".join(
            f"file\t{item['relative_path']}\t{item['size']}\t{item['mtime_ns']}"
            for item in before_files
        ) + "\n"
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
            add_bytes(tar, "source-layout.txt", layout.encode("utf-8"))
            add_bytes(tar, "source-schema.json", json_bytes(source_schema))
            add_bytes(tar, "sampling-summary.json", json_bytes(summary))
            add_bytes(tar, "full-statistics.json", json_bytes(stats))
            add_bytes(tar, "source-evidence-entry.json", json_bytes(evidence))
            add_bytes(tar, "sampling-manifest.jsonl", b"")
            add_bytes(tar, "candidate-records.jsonl", jsonl_bytes(candidate_records))
            add_bytes(tar, "sampling-candidate-decisions.jsonl", jsonl_bytes(pending_decisions))
            for metadata in before_files:
                relative = metadata["relative_path"]
                path = root / relative
                if path.name.lower() in {"readme.md", "dataset_infos.json", ".gitattributes"} and metadata["size"] <= 2_000_000:
                    add_bytes(tar, f"source-metadata/{relative}", retry(path.read_bytes))
        return

    selected = []
    decisions = []
    seen_hashes: set[str] = set()
    next_draw = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        while next_draw < len(locations) and len(selected) < TARGET:
            draw_orders = list(range(next_draw, min(next_draw + 32, len(locations))))
            fetched = list(
                executor.map(
                    lambda draw: image_from_record(cfg, loaded[draw]["record"]),
                    draw_orders,
                )
            )
            for draw_order, media in zip(draw_orders, fetched):
                locator = loaded[draw_order]["locator"]
                row = loaded[draw_order]["record"]
                if len(selected) >= TARGET:
                    decisions.append({**locator, "reason": "reserve_candidate_not_needed"})
                    continue
                if media.get("error"):
                    decisions.append(
                        {
                            **locator,
                            "reason": media["error"],
                            "source_media_url": media.get("source_media_url"),
                        }
                    )
                    continue
                payload = media["payload"]
                digest = hashlib.sha256(payload).hexdigest()
                expected_field = cfg.get("expected_sha_field")
                expected = str(row.get(expected_field) or "") if expected_field else ""
                if expected and expected.lower() != digest:
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
                    decisions.append({**locator, "reason": "duplicate_media_sha256", "sha256": digest})
                    continue
                decoded = decode_image(payload)
                if decoded.get("error"):
                    decisions.append({**locator, "reason": decoded["error"], "sha256": digest})
                    continue
                seen_hashes.add(digest)
                sample_id = f"{dataset.lower().replace(' ', '-').replace('_', '-')}-{len(selected) + 1:03d}"
                extension = "jpg" if decoded["image_format"] in {"jpeg", "jpg"} else decoded["image_format"]
                media_path = f"media/{sample_id}.{extension}"
                record_path = f"records/{sample_id}.json"
                question, answer = preview_fields(cfg, row)
                public = {
                    **locator,
                    "sample_id": sample_id,
                    "source_uri": f"{root}/{locator['shard']}#row={locator['shard_row']}",
                    "media_archive_path": media_path,
                    "record_archive_path": record_path,
                    "sha256": digest,
                    "byte_length": len(payload),
                    **decoded,
                    "source_media_url": media.get("source_media_url"),
                    "source_image_path": media.get("source_image_path"),
                    "http_etag": media.get("etag"),
                    "http_last_modified": media.get("last_modified"),
                    "input_preview": question[:1_000],
                    "output_preview": answer[:1_000],
                }
                selected.append({"public": public, "payload": payload, "record": row})
            next_draw += len(draw_orders)

    for draw_order in range(next_draw, len(locations)):
        decisions.append({**loaded[draw_order]["locator"], "reason": "reserve_candidate_not_needed"})

    if len(selected) < TARGET:
        for item in decisions:
            if item["reason"] == "reserve_candidate_not_needed":
                item["reason"] = "uninspected_after_candidate_processing"

    after_files = source_files(root)
    after_digest = fingerprint(after_files)
    schema_variants = []
    schema_counter: Counter[str] = Counter(str(item["schema"]) for item in index)
    for schema, count in schema_counter.items():
        fields = next(item["schema"] for item in index if str(item["schema"]) == schema)
        schema_variants.append(
            {
                "file_count": count,
                "fields": [
                    {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                    for field in fields
                ],
                "arrow_schema": schema,
            }
        )
    source_schema = {
        "dataset": dataset,
        "observed_from": "source Parquet only",
        "schema_variants": schema_variants,
        "field_null_counts": stats["field_null_counts"],
        "field_non_null_counts": stats["field_non_null_counts"],
        "representative_records": stats["representative_records"],
    }
    summary = {
        "dataset": dataset,
        "source_root": str(root),
        "sampling_unit": cfg["sampling_unit"],
        "population_total": population_total,
        "population_by_split": stats["records_by_split"],
        "seed": cfg["seed"],
        "candidate_count": len(locations),
        "selected_count": len(selected),
        "invalid_or_duplicate_count": sum(item["reason"] != "reserve_candidate_not_needed" for item in decisions),
        "reserve_candidate_count": sum(item["reason"] == "reserve_candidate_not_needed" for item in decisions),
        "sampling_method": "fixed-seed uniform permutation over all canonical source Parquet rows",
        "read_only_source": True,
        "source_fingerprint_before": before_digest,
        "source_fingerprint_after": after_digest,
        "source_unchanged": before_digest == after_digest,
        "full_statistics": stats,
        "media_storage": "remote URL" if cfg["mode"] == "remote_image" else "embedded Parquet bytes",
    }
    evidence = {
        "dataset": dataset,
        "access_mode": "ssh",
        "source_root": str(root),
        "fingerprint_algorithm": "sha256 over sorted relative path, size, and mtime_ns",
        "before_digest": before_digest,
        "after_digest": after_digest,
        "unchanged": before_digest == after_digest,
        "collector_transport": "collector code via stdin; tar stream via stdout",
        "temporary_workspace": None,
        "read_operations": ["directory enumeration", "Parquet footer/read", "media URL read" if cfg["mode"] == "remote_image" else "embedded media read"],
    }
    layout = "\n".join(
        f"file\t{item['relative_path']}\t{item['size']}\t{item['mtime_ns']}" for item in before_files
    ) + "\n"

    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
        add_bytes(tar, "source-layout.txt", layout.encode("utf-8"))
        add_bytes(tar, "source-schema.json", json_bytes(source_schema))
        add_bytes(tar, "sampling-summary.json", json_bytes(summary))
        add_bytes(tar, "full-statistics.json", json_bytes(stats))
        add_bytes(tar, "source-evidence-entry.json", json_bytes(evidence))
        add_bytes(
            tar,
            "sampling-manifest.jsonl",
            b"".join(json.dumps(item["public"], ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for item in selected),
        )
        add_bytes(
            tar,
            "sampling-candidate-decisions.jsonl",
            b"".join(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for item in decisions),
        )
        for metadata in before_files:
            relative = metadata["relative_path"]
            path = root / relative
            if path.name.lower() in {"readme.md", "dataset_infos.json", ".gitattributes"} and metadata["size"] <= 2_000_000:
                add_bytes(tar, f"source-metadata/{relative}", retry(path.read_bytes))
        for item in selected:
            public = item["public"]
            add_bytes(tar, public["media_archive_path"], item["payload"])
            record = {"source_locator": public["source_uri"], "raw_record": safe_value(item["record"])}
            add_bytes(tar, public["record_archive_path"], json_bytes(record))


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIGS:
        print(f"usage: {Path(sys.argv[0]).name} DATASET; choices={sorted(CONFIGS)}", file=sys.stderr)
        return 2
    collect(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
