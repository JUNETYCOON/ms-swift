#!/usr/bin/env python3
"""Collect source-only Molmo2 annotation statistics and deterministic records.

The script is streamed to ``python3 -`` on the source host. It performs only
reads below SOURCE_BASE and emits a tar stream to stdout. Media extraction is
handled separately because availability differs by source and archive.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


SOURCE_BASE = Path("/mnt/luojunkun/stage1/dataset")
CANDIDATES = 400
RECORD_TARGET = 200
CONFIGS = {
    "Molmo2-VideoCapQA": {
        "source_dir": "Molmo2-VideoCapQA",
        "canonical_files": "data/*.parquet",
        "schema_files": "data/*.parquet",
        "mode": "capqa",
        "seed": 2026080504,
        "sampling_unit": "logical_record",
        "media_status": "source package has annotations and URL mapping but no video bytes",
    },
    "Molmo2-VideoSubtitleQA": {
        "source_dir": "Molmo2-VideoSubtitleQA",
        "canonical_files": "data/*.parquet",
        "schema_files": "data/*.parquet",
        "mode": "subtitleqa",
        "seed": 2026080505,
        "sampling_unit": "logical_record",
        "media_status": "source package has annotations and URL mapping but no video bytes",
    },
    "Molmo2-VideoPoint": {
        "source_dir": "Molmo2-VideoPoint",
        "canonical_files": "data/train-*.parquet",
        "schema_files": "**/*.parquet",
        "mode": "videopoint",
        "seed": 2026080506,
        "sampling_unit": "logical_record",
        "media_status": "mixed external videos and locally packaged generated-video tar archives",
    },
    "Molmo2-VideoTrack": {
        "source_dir": "Molmo2-VideoTrack",
        "canonical_files": "data/*/*_point_tracks.parquet",
        "schema_files": "data/*/*.parquet",
        "mode": "videotrack",
        "seed": 2026080507,
        "sampling_unit": "logical_record",
        "media_status": "source video availability varies by contributing dataset; several sources are not packaged",
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
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


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
                    "suffix": path.suffix.lower(),
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


def split_hint(relative: str) -> str:
    lower = relative.lower()
    for split in ("validation", "train", "test", "val"):
        if split in lower:
            return split
    return "unspecified"


def parquet_index(root: Path, pattern: str) -> list[dict[str, Any]]:
    output = []
    for path in sorted(root.glob(pattern)):
        parquet = retry(lambda path=path: pq.ParquetFile(path))
        output.append(
            {
                "path": path,
                "relative": path.relative_to(root).as_posix(),
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
                "schema": parquet.schema_arrow,
                "split": split_hint(path.name),
            }
        )
        parquet.close()
    return output


def schema_report(
    dataset: str,
    root: Path,
    schema_index: list[dict[str, Any]],
    canonical_paths: set[str],
) -> dict[str, Any]:
    variants = []
    representative_records = []
    for item in schema_index:
        parquet = retry(lambda item=item: pq.ParquetFile(item["path"]))
        samples = []
        if item["rows"] and len(representative_records) < 3 and item["relative"] in canonical_paths:
            first = retry(lambda: parquet.read_row_group(0))
            samples = [safe_value(row) for row in first.slice(0, 3).to_pylist()]
            representative_records.extend(
                {
                    "source": f"{item['relative']}#row={row_index}",
                    "record": record,
                }
                for row_index, record in enumerate(samples)
            )
        variants.append(
            {
                "relative_path": item["relative"],
                "canonical_population": item["relative"] in canonical_paths,
                "rows": item["rows"],
                "row_groups": item["row_groups"],
                "fields": [
                    {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                    for field in item["schema"]
                ],
                "arrow_schema": str(item["schema"]),
                "representative_records": samples,
            }
        )
        parquet.close()
    return {
        "dataset": dataset,
        "observed_from": "original source Parquet records",
        "schema_variants": variants,
        "representative_records": representative_records[:3],
        "derived_fields": [],
    }


def add_counter(counter: Counter[str], values: list[Any]) -> None:
    for value in values:
        counter[str(value) if value is not None else "<missing>"] += 1


def text_stats(column: Any) -> tuple[int, int, int]:
    lengths = pc.utf8_length(column)
    chars = int(pc.sum(lengths).as_py() or 0)
    empty = int(pc.sum(pc.equal(lengths, 0)).as_py() or 0)
    non_null = len(column) - column.null_count
    return chars, empty, non_null


def list_length_sum(column: Any) -> int:
    try:
        return int(pc.sum(pc.list_value_length(column)).as_py() or 0)
    except Exception:
        return 0


def classify_category(category: str) -> list[str]:
    text = category.lower()
    labels = []
    rules = {
        "perception.object": ("object", "presence", "appearance", "animal"),
        "perception.action": ("action", "gesture", "interaction"),
        "temporal.event_sequence": ("sequence", "temporal", "event"),
        "reasoning.spatial": ("spatial", "location", "position", "reference"),
        "reasoning.causal": ("causal", "causality", "explanation", "why"),
        "reasoning.counting": ("count", "quantity", "number"),
        "reasoning.comparative": ("comparative", "comparison"),
        "multimodal.dialogue_alignment": ("dialogue", "subtitle", "alignment"),
        "tracking.referring_expression": ("track", "referring", "reference"),
    }
    for label, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            labels.append(label)
    return labels or ["vqa.general"]


def full_statistics(
    cfg: dict[str, Any], root: Path, canonical: list[dict[str, Any]], schema_index: list[dict[str, Any]]
) -> dict[str, Any]:
    mode = cfg["mode"]
    records_by_file: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    alignment_types: Counter[str] = Counter()
    video_sources: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter()
    field_nulls: Counter[str] = Counter()
    field_empty_strings: Counter[str] = Counter()
    unique_videos: set[str] = set()
    unique_clips: set[str] = set()
    text_characters = 0
    qa_pairs = 0
    negative_answers = 0
    point_frames = 0
    point_annotations = 0
    track_point_frames = 0
    segment_entries = 0
    unsure_count = 0
    non_null_count_values = 0

    for item in canonical:
        parquet = retry(lambda item=item: pq.ParquetFile(item["path"]))
        records_by_file[item["relative"]] += item["rows"]
        for batch in parquet.iter_batches(batch_size=2048):
            columns = {
                name: batch.column(batch.schema.get_field_index(name))
                for name in batch.schema.names
            }
            for name, column in columns.items():
                field_nulls[name] += column.null_count

            video_column = columns.get("video_id") or columns.get("video")
            if video_column is not None:
                unique_videos.update(str(value) for value in video_column.to_pylist() if value)
            if columns.get("clip") is not None:
                unique_clips.update(str(value) for value in columns["clip"].to_pylist() if value)

            if mode == "capqa" and "qa_list" in columns:
                for qa_list in columns["qa_list"].to_pylist():
                    for qa in qa_list or []:
                        qa_pairs += 1
                        category = str((qa or {}).get("Category") or "<missing>")
                        categories[category] += 1
                        for label in classify_category(category):
                            capability_counts[label] += 1
                        question = str((qa or {}).get("Question") or "")
                        answer = str((qa or {}).get("Answer") or "")
                        text_characters += len(question) + len(answer)
                        if not question:
                            field_empty_strings["Question"] += 1
                        if not answer:
                            field_empty_strings["Answer"] += 1
                        negative_answers += len((qa or {}).get("NegativeAnswers") or [])
                continue

            for field in ("Question", "Answer", "question", "label", "exp"):
                column = columns.get(field)
                if column is not None:
                    chars, empty, _ = text_stats(column)
                    text_characters += chars
                    field_empty_strings[field] += empty

            category_column = columns.get("Category") or columns.get("category")
            if category_column is not None:
                values = category_column.to_pylist()
                add_counter(categories, values)
                for category in values:
                    for label in classify_category(str(category or "<missing>")):
                        capability_counts[label] += 1

            if mode == "capqa":
                qa_pairs += batch.num_rows
                negative_answers += list_length_sum(columns["NegativeAnswers"])
            elif mode == "subtitleqa":
                qa_pairs += batch.num_rows
                negative_answers += list_length_sum(columns["NegativeAnswers"])
                add_counter(alignment_types, columns["AlignmentType"].to_pylist())
                capability_counts["multimodal.subtitle_visual_alignment"] += batch.num_rows
            elif mode == "videopoint":
                add_counter(video_sources, columns["video_source"].to_pylist())
                point_frames += list_length_sum(columns["points"])
                try:
                    flattened = pc.list_flatten(columns["points"])
                    point_annotations += list_length_sum(flattened)
                except Exception:
                    pass
                if columns.get("annotator_unsure") is not None:
                    unsure_count += int(pc.sum(pc.cast(columns["annotator_unsure"], "int64")).as_py() or 0)
                if columns.get("count") is not None:
                    non_null_count_values += len(columns["count"]) - columns["count"].null_count
                capability_counts["pointing.video_temporal"] += batch.num_rows
            elif mode == "videotrack":
                track_point_frames += list_length_sum(columns["points"])
                segment_entries += list_length_sum(columns["segments"])
                capability_counts["tracking.referring_expression"] += batch.num_rows
                capability_counts["temporal.object_persistence"] += batch.num_rows
        parquet.close()

    physical_schema_rows = sum(item["rows"] for item in schema_index)
    canonical_rows = sum(records_by_file.values())
    task_counts: dict[str, int]
    if mode == "capqa":
        task_counts = {"video_qa.multiple_choice": qa_pairs}
    elif mode == "subtitleqa":
        task_counts = {"video_qa.subtitle_grounded": qa_pairs}
    elif mode == "videopoint":
        task_counts = {"pointing.video_temporal": canonical_rows}
    else:
        task_counts = {"tracking.referring_expression": canonical_rows}

    return {
        "scope": "full canonical source metadata population",
        "canonical_source_records": canonical_rows,
        "physical_parquet_rows_all_views": physical_schema_rows,
        "duplicate_or_noncanonical_view_rows": physical_schema_rows - canonical_rows,
        "records_by_file": dict(sorted(records_by_file.items())),
        "unique_video_ids": len(unique_videos),
        "unique_clip_ids": len(unique_clips),
        "supervision_units": qa_pairs if qa_pairs else canonical_rows,
        "qa_pairs": qa_pairs,
        "negative_answer_options": negative_answers,
        "text_characters": text_characters,
        "supervised_token_estimate_chars_div_4": round(text_characters / 4),
        "point_timestamp_entries": point_frames,
        "point_annotations": point_annotations,
        "track_point_frame_entries": track_point_frames,
        "segment_entries": segment_entries,
        "annotator_unsure_count": unsure_count,
        "non_null_count_values": non_null_count_values,
        "task_counts": task_counts,
        "category_counts": dict(categories.most_common()),
        "alignment_type_counts": dict(alignment_types.most_common()),
        "video_source_counts": dict(video_sources.most_common()),
        "capability_counts": dict(capability_counts.most_common()),
        "field_null_counts": dict(sorted(field_nulls.items())),
        "field_empty_string_counts": dict(sorted(field_empty_strings.items())),
        "evidence_type": "Full-population statistic",
    }


def candidate_locations(
    index: list[dict[str, Any]], seed: int, count: int
) -> list[dict[str, Any]]:
    total = sum(item["rows"] for item in index)
    draws = random.Random(seed).sample(range(total), min(count, total))
    output = []
    for draw_order, population_index in enumerate(draws):
        remaining = population_index
        for item in index:
            if remaining >= item["rows"]:
                remaining -= item["rows"]
                continue
            output.append(
                {
                    "draw_order": draw_order,
                    "population_index": population_index,
                    "shard": item["relative"],
                    "shard_row": remaining,
                    "split": item["split"],
                }
            )
            break
    return output


def read_candidate_rows(
    root: Path, locations: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in locations:
        grouped[item["shard"]].append(item)
    output = {}
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


def record_identity(mode: str, row: dict[str, Any]) -> str:
    if mode == "videotrack":
        return f"{row.get('video_dataset')}:{row.get('clip') or row.get('video')}"
    return str(row.get("video_id") or "")


def previews(mode: str, row: dict[str, Any]) -> tuple[str, str, list[str]]:
    if mode == "capqa" and row.get("qa_list"):
        qa = (row.get("qa_list") or [{}])[0] or {}
        category = str(qa.get("Category") or "<missing>")
        return str(qa.get("Question") or ""), str(qa.get("Answer") or ""), classify_category(category)
    if mode in {"capqa", "subtitleqa"}:
        category = str(row.get("Category") or "<missing>")
        return str(row.get("Question") or ""), str(row.get("Answer") or ""), classify_category(category)
    if mode == "videopoint":
        category = str(row.get("category") or "<missing>")
        answer = f"label={row.get('label')}; count={row.get('count')}; timestamp_groups={len(row.get('points') or [])}"
        return str(row.get("question") or ""), answer, ["pointing.video_temporal", *classify_category(category)]
    answer = f"track point frames={len(row.get('points') or [])}; start={row.get('start_frame')}; end={row.get('end_frame')}"
    return str(row.get("exp") or ""), answer, ["tracking.referring_expression", "temporal.object_persistence"]


def media_failure_reason(mode: str) -> str:
    if mode in {"capqa", "subtitleqa"}:
        return "source_video_not_packaged_external_download_required"
    if mode == "videopoint":
        return "source_media_join_deferred_mixed_local_and_external_sources"
    return "source_media_join_deferred_partial_packaging"


def collect(dataset: str) -> None:
    cfg = CONFIGS[dataset]
    root = SOURCE_BASE / cfg["source_dir"]
    before_files = source_files(root)
    before_digest = fingerprint(before_files)
    canonical = parquet_index(root, cfg["canonical_files"])
    schema_index = parquet_index(root, cfg["schema_files"])
    canonical_paths = {item["relative"] for item in canonical}
    schema = schema_report(dataset, root, schema_index, canonical_paths)
    stats = full_statistics(cfg, root, canonical, schema_index)
    schema["field_null_counts_full_canonical"] = stats["field_null_counts"]
    schema["field_empty_string_counts_full_canonical"] = stats[
        "field_empty_string_counts"
    ]

    locations = candidate_locations(canonical, cfg["seed"], CANDIDATES)
    loaded = read_candidate_rows(root, locations)
    record_samples = []
    decisions = []
    records_to_archive = []
    seen_identities = set()
    for draw_order in range(len(locations)):
        locator = loaded[draw_order]["locator"]
        row = loaded[draw_order]["record"]
        identity = record_identity(cfg["mode"], row)
        if len(record_samples) >= RECORD_TARGET:
            decisions.append({**locator, "reason": "reserve_record_candidate_not_needed"})
            continue
        if not identity:
            decisions.append({**locator, "reason": "missing_video_or_clip_identity"})
            continue
        if identity in seen_identities:
            decisions.append(
                {**locator, "reason": "duplicate_media_identity_within_draw", "media_identity": identity}
            )
            continue
        seen_identities.add(identity)
        sample_id = f"{dataset.lower()}-record-{len(record_samples) + 1:03d}"
        record_path = f"records/{sample_id}.json"
        question, answer, capabilities = previews(cfg["mode"], row)
        public = {
            **locator,
            "sample_id": sample_id,
            "record_archive_path": record_path,
            "source_uri": f"{root}/{locator['shard']}#row={locator['shard_row']}",
            "media_identity": identity,
            "input_preview": question[:1_000],
            "output_preview": answer[:1_000],
            "capabilities": capabilities,
            "visual_acceptance": False,
            "media_status": cfg["media_status"],
        }
        record_samples.append(public)
        records_to_archive.append((record_path, public["source_uri"], row))
        decisions.append(
            {
                **locator,
                "reason": media_failure_reason(cfg["mode"]),
                "media_identity": identity,
                "record_archive_path": record_path,
            }
        )

    after_files = source_files(root)
    after_digest = fingerprint(after_files)
    summary = {
        "dataset": dataset,
        "source_root": str(root),
        "sampling_unit": cfg["sampling_unit"],
        "population_total": sum(item["rows"] for item in canonical),
        "physical_parquet_rows_all_views": sum(item["rows"] for item in schema_index),
        "seed": cfg["seed"],
        "candidate_count": len(locations),
        "selected_count": 0,
        "record_sample_count": len(record_samples),
        "invalid_or_duplicate_count": sum(
            item["reason"] != "reserve_record_candidate_not_needed" for item in decisions
        ),
        "reserve_candidate_count": sum(
            item["reason"] == "reserve_record_candidate_not_needed" for item in decisions
        ),
        "sampling_method": "fixed-seed uniform permutation over canonical source Parquet rows; unique media identity retained for record inspection",
        "read_only_source": True,
        "incomplete_visual_sample": True,
        "incomplete_reason": cfg["media_status"],
        "source_fingerprint_before": before_digest,
        "source_fingerprint_after": after_digest,
        "source_unchanged": before_digest == after_digest,
        "full_statistics": stats,
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
        "read_operations": ["directory enumeration", "Parquet footer/schema read", "Parquet metadata streaming"],
    }
    layout = "\n".join(
        f"file\t{item['relative_path']}\t{item['size']}\t{item['mtime_ns']}"
        for item in before_files
    ) + "\n"

    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
        add_bytes(tar, "source-layout.txt", layout.encode("utf-8"))
        add_bytes(tar, "source-schema.json", json_bytes(schema))
        add_bytes(tar, "sampling-summary.json", json_bytes(summary))
        add_bytes(tar, "full-statistics.json", json_bytes(stats))
        add_bytes(tar, "source-evidence-entry.json", json_bytes(evidence))
        add_bytes(tar, "sampling-manifest.jsonl", b"")
        add_bytes(tar, "record-sampling-manifest.jsonl", jsonl_bytes(record_samples))
        add_bytes(tar, "sampling-candidate-decisions.jsonl", jsonl_bytes(decisions))
        for metadata in before_files:
            relative = metadata["relative_path"]
            path = root / relative
            if path.name.lower() in {"readme.md", "dataset_infos.json", ".gitattributes", "sha256sums"} and metadata["size"] <= 2_000_000:
                add_bytes(tar, f"source-metadata/{relative}", retry(path.read_bytes))
        for record_path, source_uri, row in records_to_archive:
            add_bytes(
                tar,
                record_path,
                json_bytes({"source_locator": source_uri, "raw_record": safe_value(row)}),
            )


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIGS:
        print(f"usage: {Path(sys.argv[0]).name} DATASET; choices={sorted(CONFIGS)}", file=sys.stderr)
        return 2
    collect(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
