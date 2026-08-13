#!/usr/bin/env python3
"""Audit converted ms-swift datasets and collect deterministic row samples."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse, urlsplit, urlunsplit

from PIL import Image, ImageDraw


DEFAULT_MANIFEST = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/curated_dataset_entrypoints.json"
)
DEFAULT_OUTPUT = Path("/mnt/workspace/converted-dataset-audit-20260810")
CONVERTED_ROOT = Path("/mnt/luojunkun/stage1/dataset_ms-swift")
TARGET_ROWS = 100
CANDIDATE_ROWS = 400
BASE_SEED = 20260810
CHUNK_SIZE = 8 * 1024 * 1024
MAX_MEDIA_PRESENCE_CHECKS = 0
REMOTE_SCHEMES = {"http", "https", "s3", "gs", "oss"}
PLACEHOLDERS = {"images": "<image>", "videos": "<video>", "audios": "<audio>"}

DISPLAY_NAMES = {
    "ai2d": "AI2D",
    "chartqa": "ChartQA",
    "gqa": "GQA",
    "textvqa": "TextVQA",
    "visualgenome-qa": "VisualGenome-QA",
    "visualgenome-regions": "VisualGenome-Regions",
    "vlm-r1": "VLM-R1",
    "vqav2": "VQAv2",
    "robo2vlm": "Robo2VLM",
    "robovqa": "RoboVQA",
    "spatialvlm": "SpatialVLM",
    "molmo2-video-capqa": "Molmo2-VideoCapQA",
    "molmo2-video-point": "Molmo2-VideoPoint",
    "molmo2-video-subtitleqa": "Molmo2-VideoSubtitleQA",
    "molmo2-video-track": "Molmo2-VideoTrack",
    "pixmo-cap": "PixMo-Cap",
    "pixmo-points": "PixMo-Points",
    "llava": "LLaVA-Instruct",
    "coco": "COCO",
}

TASK_TYPES = {
    "ai2d": "diagram understanding / description",
    "chartqa": "chart VQA",
    "gqa": "compositional VQA",
    "textvqa": "scene-text VQA",
    "visualgenome-qa": "short-answer VQA",
    "visualgenome-regions": "region description / grounding text",
    "vlm-r1": "grounding",
    "vqav2": "general VQA",
    "robo2vlm": "embodied multiple-choice VQA",
    "robovqa": "embodied video QA",
    "spatialvlm": "spatial VQA / grounding",
    "molmo2-video-capqa": "video VQA / caption QA",
    "molmo2-video-point": "video pointing",
    "molmo2-video-subtitleqa": "video subtitle QA",
    "molmo2-video-track": "video tracking",
    "pixmo-cap": "long description",
    "pixmo-points": "pointing / counting",
    "llava": "mixed instruction tuning",
    "coco": "multi-label image classification / caption-style instruction",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    os.replace(temporary, path)


def safe_value(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "<max-depth>"
    if isinstance(value, str):
        return value if len(value) <= 20_000 else value[:20_000] + "<truncated>"
    if isinstance(value, dict):
        return {str(key): safe_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [safe_value(item, depth + 1) for item in value[:200]]
        if len(value) > 200:
            items.append({"truncated_items": len(value) - 200})
        return items
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def field_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("datasets"), dict):
        raise ValueError("Manifest must contain a datasets object")
    return manifest


def enabled_configs(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    priority = manifest.get("global_dedup", {}).get("training_priority") or []
    enabled = {
        name: config
        for name, config in manifest["datasets"].items()
        if isinstance(config, dict) and config.get("enabled", True)
    }
    if set(priority) != set(enabled):
        raise ValueError("Manifest training priority does not match enabled datasets")
    return [(name, enabled[name]) for name in priority]


def as_paths(value: Any) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    return [Path(item).expanduser().resolve() for item in values if item]


def dataset_inputs(config: dict[str, Any]) -> list[tuple[str, Path]]:
    source_train = config.get("source_train") or config.get("split_train")
    rows = [("source_train", path) for path in as_paths(source_train)]
    eval_paths = as_paths(config.get("eval"))
    rows.extend(("eval" if len(eval_paths) == 1 else f"eval_{index}", path) for index, path in enumerate(eval_paths))
    return rows


def metadata_row(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    row: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists:
        stat = path.stat()
        row.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return row


def metadata_digest(rows: list[dict[str, Any]]) -> str:
    canonical = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in sorted(rows, key=lambda item: item["path"])
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    output = []
    for item in content:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, dict):
            value = item.get("text")
            if isinstance(value, str):
                output.append(value)
    return "\n".join(output)


def message_preview(messages: Any) -> tuple[str, str]:
    if not isinstance(messages, list):
        return "", ""
    question = ""
    answer = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        text = text_from_content(message.get("content"))
        if role == "user" and not question:
            question = text
        elif role == "assistant" and not answer:
            answer = text
    return question, answer


def normalize_media(value: Any) -> tuple[list[str], bool]:
    if value is None:
        return [], False
    invalid = not isinstance(value, list)
    values = value if isinstance(value, list) else [value]
    output = []
    for item in values:
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, dict) and isinstance(item.get("path"), str):
            output.append(item["path"])
        else:
            invalid = True
    return output, invalid


def is_remote(reference: str) -> bool:
    return urlparse(reference).scheme.lower() in REMOTE_SCHEMES


def canonical_http_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid HTTP URL: {value!r}")
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    default_port = (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    )
    netloc = hostname if not port or default_port else f"{hostname}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", "", ""))


def pixmo_downloaded_path(dataset_name: str, reference: str) -> Path | None:
    if dataset_name not in {"pixmo-cap", "pixmo-points"}:
        return None
    try:
        identity = hashlib.sha256(canonical_http_url(reference).encode("utf-8")).hexdigest()
    except ValueError:
        return None
    image_root = CONVERTED_ROOT / dataset_name / "images" / identity[:2]
    matches = sorted(image_root.glob(f"{identity}.*"))
    return next((path for path in matches if path.is_file()), None)


def resolve_reference(reference: str, jsonl_path: Path) -> str:
    if is_remote(reference):
        return reference
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = jsonl_path.parent / path
    # Avoid Path.resolve() here: on OSS/FUSE-backed dataset roots it can turn
    # every media reference into a blocking filesystem lookup. Existence and
    # decoding are checked later in batched media snapshots and sample archiving.
    return os.path.abspath(os.fspath(path))


def placeholder_count(
    messages: Any,
    placeholder: str,
    *,
    ignore_assistant_text: bool = False,
) -> int:
    if not isinstance(messages, list):
        return 0
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        role = str(message.get("role") or "").strip().casefold()
        if not (ignore_assistant_text and role == "assistant"):
            count += text_from_content(content).count(placeholder)
        if isinstance(content, list):
            expected_type = placeholder.strip("<>")
            count += sum(
                1
                for item in content
                if isinstance(item, dict) and item.get("type") == expected_type
            )
    return count


def validate_objects(objects: Any, messages: Any) -> list[str]:
    if objects is None:
        return []
    if not isinstance(objects, dict):
        return ["objects_not_object"]
    errors = []
    bboxes = objects.get("bbox")
    refs = objects.get("ref")
    if bboxes is not None:
        if not isinstance(bboxes, list):
            errors.append("bbox_not_array")
        else:
            for box in bboxes:
                if (
                    not isinstance(box, (list, tuple))
                    or len(box) not in {2, 4}
                    or any(not isinstance(value, (int, float)) for value in box)
                ):
                    errors.append("invalid_point_or_bbox")
                    break
            if bboxes and not objects.get("bbox_type"):
                errors.append("missing_bbox_type")
    if refs is not None and not isinstance(refs, list):
        errors.append("ref_not_array")
    if isinstance(refs, list) and placeholder_count(messages, "<ref-object>") != len(refs):
        errors.append("ref_placeholder_mismatch")
    if isinstance(bboxes, list) and placeholder_count(messages, "<bbox>") != len(bboxes):
        errors.append("bbox_placeholder_mismatch")
    return errors


def validate_row(row: Any, jsonl_path: Path) -> tuple[list[str], dict[str, list[str]]]:
    if not isinstance(row, dict):
        return ["row_not_object"], {key: [] for key in PLACEHOLDERS}
    errors = []
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("missing_or_invalid_messages")
    else:
        roles = []
        for message in messages:
            if not isinstance(message, dict):
                errors.append("message_not_object")
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant", "tool"}:
                errors.append("invalid_message_role")
            else:
                roles.append(role)
            if not isinstance(content, (str, list)):
                errors.append("invalid_message_content")
        if "user" not in roles:
            errors.append("missing_user_message")
        if "assistant" not in roles:
            errors.append("missing_assistant_message")

    media: dict[str, list[str]] = {}
    for field, placeholder in PLACEHOLDERS.items():
        values, invalid = normalize_media(row.get(field))
        media[field] = [resolve_reference(value, jsonl_path) for value in values]
        if invalid:
            errors.append(f"{field}_not_ordered_array")
        placeholders = placeholder_count(
            messages,
            placeholder,
            ignore_assistant_text=True,
        )
        if placeholders != len(values):
            errors.append(f"{field}_placeholder_mismatch")
    errors.extend(validate_objects(row.get("objects"), messages))
    return sorted(set(errors)), media


def candidate_priority(seed: int, path: Path, line_number: int) -> int:
    payload = f"{seed}\0{path}\0{line_number}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def push_candidate(
    heap: list[tuple[int, int, dict[str, Any]]],
    priority: int,
    sequence: int,
    candidate: dict[str, Any],
) -> None:
    item = (-priority, sequence, candidate)
    if len(heap) < CANDIDATE_ROWS:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def inspect_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    output = []
    for index, (name, config) in enumerate(enabled_configs(manifest)):
        files = []
        for split, path in dataset_inputs(config):
            record = None
            error = None
            if path.is_file():
                try:
                    with path.open("rb") as stream:
                        for line_number, line in enumerate(stream, 1):
                            if line.strip():
                                record = json.loads(line)
                                break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            files.append(
                {
                    "split": split,
                    **metadata_row(path),
                    "first_line": line_number if record is not None else None,
                    "keys": sorted(record) if isinstance(record, dict) else None,
                    "first_record": safe_value(record),
                    "error": error,
                }
            )
        output.append(
            {
                "name": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "seed": BASE_SEED * 100 + index,
                "canonical_train": metadata_row(Path(config["train"])),
                "files": files,
            }
        )
    return {"generated_at": utc_now(), "datasets": output}


def scan_dataset(
    name: str,
    config: dict[str, Any],
    dataset_index: int,
    output_root: Path,
    status: dict[str, Any],
) -> dict[str, Any]:
    display_name = DISPLAY_NAMES.get(name, name)
    dataset_dir = output_root / "sub-dataset" / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    inputs = dataset_inputs(config)
    input_before = [metadata_row(path) | {"split": split} for split, path in inputs]
    missing_inputs = [row["path"] for row in input_before if not row["exists"]]
    if missing_inputs:
        raise FileNotFoundError(f"{name}: missing inputs: {missing_inputs}")

    seed = BASE_SEED * 100 + dataset_index
    row_count = 0
    rows_by_split: Counter[str] = Counter()
    json_errors: Counter[str] = Counter()
    format_errors: Counter[str] = Counter()
    field_types: dict[str, Counter[str]] = defaultdict(Counter)
    field_presence: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    text_quality_counts: Counter[str] = Counter()
    media_references: dict[str, set[str]] = {key: set() for key in PLACEHOLDERS}
    media_rows: Counter[str] = Counter()
    text_only_rows = 0
    valid_format_rows = 0
    candidate_heap: list[tuple[int, int, dict[str, Any]]] = []
    representative_records = []
    file_reports = []

    for split, path in inputs:
        started = time.monotonic()
        file_hasher = hashlib.sha256()
        file_rows = 0
        # The converted root is FUSE-backed; a large buffer avoids thousands of
        # tiny remote reads while preserving the exact byte stream and hash.
        with path.open("rb", buffering=16 * 1024 * 1024) as stream:
            for line_number, line in enumerate(stream, 1):
                file_hasher.update(line)
                if not line.strip():
                    json_errors["blank_line"] += 1
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    json_errors[type(exc).__name__] += 1
                    continue
                file_rows += 1
                row_count += 1
                rows_by_split[split] += 1
                if not isinstance(row, dict):
                    format_errors["row_not_object"] += 1
                    continue
                for field, value in row.items():
                    field_presence[field] += 1
                    field_types[field][field_type(value)] += 1
                messages = row.get("messages")
                if isinstance(messages, list):
                    row_texts = []
                    for message in messages:
                        if isinstance(message, dict):
                            role_counts[str(message.get("role") or "<missing>")] += 1
                            row_texts.append(text_from_content(message.get("content")))
                    combined_text = "\n".join(row_texts)
                    if "\ufffd" in combined_text:
                        text_quality_counts["replacement_character_rows"] += 1
                    if any(token in combined_text for token in ("鈥", "锟", "Ã", "Â", "ðŸ", "銆")):
                        text_quality_counts["possible_mojibake_rows"] += 1
                    if any(not text.strip() for text in row_texts):
                        text_quality_counts["empty_message_content_rows"] += 1
                errors, media = validate_row(row, path)
                for error in errors:
                    format_errors[error] += 1
                if not errors:
                    valid_format_rows += 1
                media_count = sum(len(values) for values in media.values())
                if media_count == 0:
                    text_only_rows += 1
                for media_type, references in media.items():
                    if references:
                        media_rows[media_type] += 1
                        media_references[media_type].update(references)
                if len(representative_records) < 3:
                    representative_records.append(
                        {
                            "source": f"{path}#line={line_number}",
                            "record": safe_value(row),
                            "format_errors": errors,
                            "media": media,
                        }
                    )
                priority = candidate_priority(seed, path, line_number)
                candidate = {
                    "source_path": str(path),
                    "source_file": path.name,
                    "split": split,
                    "line_number": line_number,
                    "population_index": row_count - 1,
                    "priority": f"{priority:032x}",
                    "record": row,
                    "format_errors": errors,
                    "media": media,
                }
                push_candidate(candidate_heap, priority, row_count, candidate)
                if row_count % 250_000 == 0:
                    print(
                        f"[{name}] rows={row_count:,} unique_media="
                        f"{sum(len(values) for values in media_references.values()):,}",
                        flush=True,
                    )
        file_reports.append(
            {
                "split": split,
                "path": str(path),
                "rows": file_rows,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": file_hasher.hexdigest(),
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        )

    candidates = [item[2] for item in candidate_heap]
    candidates.sort(key=lambda item: item["priority"])
    selected_candidates = candidates[: min(TARGET_ROWS, len(candidates))]
    selected_ids = {
        (item["source_path"], item["line_number"]) for item in selected_candidates
    }
    decisions = []
    for draw_order, candidate in enumerate(candidates):
        locator = {
            "draw_order": draw_order,
            "source_path": candidate["source_path"],
            "line_number": candidate["line_number"],
            "population_index": candidate["population_index"],
        }
        if (candidate["source_path"], candidate["line_number"]) in selected_ids:
            continue
        decisions.append({**locator, "reason": "reserve_candidate_not_needed"})

    print(f"[{name}] auditing full unique media path population", flush=True)
    media_snapshot = audit_media_population(media_references)
    sample_rows = archive_samples(name, selected_candidates, dataset_dir)
    metadata_reports = archive_metadata_reports([path for _, path in inputs], dataset_dir)

    selected_manifest = []
    for draw_order, sample in enumerate(sample_rows):
        selected_manifest.append(
            {
                "sample_id": sample["sample_id"],
                "draw_order": draw_order,
                "population_index": sample["population_index"],
                "source_path": sample["source_path"],
                "line_number": sample["line_number"],
                "split": sample["split"],
                "record_archive_path": sample["record_archive_path"],
                "media_status": sample["media_status"],
                "media_assets": sample["media_assets"],
                "ground_truth_overlay": sample.get("ground_truth_overlay") or [],
                "format_errors": sample["format_errors"],
                "input_preview": sample["input_preview"],
                "output_preview": sample["output_preview"],
            }
        )

    input_after = [metadata_row(path) | {"split": split} for split, path in inputs]
    before_digest = metadata_digest(input_before)
    after_digest = metadata_digest(input_after)
    canonical_train = metadata_row(Path(config["train"]).expanduser().resolve())
    summary = {
        "dataset": name,
        "display_name": display_name,
        "task_type": TASK_TYPES.get(name, "multimodal instruction tuning"),
        "sampling_unit": "logical_record",
        "population_total": row_count,
        "population_by_split": dict(rows_by_split),
        "seed": seed,
        "candidate_count": len(candidates),
        "selected_count": len(selected_manifest),
        "sampling_method": (
            "deterministic SHA-256 rank over every stable source_train/eval JSONL row; "
            "first 100 ranked logical rows displayed without media-conditioned replacement"
        ),
        "read_only_source": True,
        "source_fingerprint_before": before_digest,
        "source_fingerprint_after": after_digest,
        "source_unchanged": before_digest == after_digest,
        "canonical_train": canonical_train,
        "canonical_train_ready": bool(canonical_train["exists"] and canonical_train.get("size", 0) > 0),
        "format": {
            "json_error_counts": dict(json_errors),
            "format_error_counts": dict(format_errors.most_common()),
            "valid_format_rows": valid_format_rows,
            "valid_format_ratio": valid_format_rows / row_count if row_count else 0.0,
            "text_only_rows": text_only_rows,
            "rows_with_media_by_type": dict(media_rows),
            "text_quality_counts": dict(text_quality_counts),
        },
        "media_snapshot": media_snapshot,
        "sample_media_statuses": dict(Counter(row["media_status"] for row in sample_rows)),
        "ground_truth_overlay_statuses": dict(
            Counter(
                overlay.get("status", "unknown")
                for row in sample_rows
                for overlay in (row.get("ground_truth_overlay") or [])
            )
        ),
        "metadata_reports": metadata_reports,
    }
    schema = {
        "dataset": name,
        "observed_from": [str(path) for _, path in inputs],
        "field_presence_counts": dict(field_presence),
        "field_type_counts": {
            field: dict(counter) for field, counter in sorted(field_types.items())
        },
        "message_role_counts": dict(role_counts),
        "text_quality_counts": dict(text_quality_counts),
        "representative_records": representative_records,
    }
    layout_lines = [
        f"{row['split']}\t{row['path']}\t{row.get('size', 0)}\t{row.get('mtime_ns', 0)}"
        for row in input_before
    ]
    layout_lines.append(
        f"canonical_train\t{canonical_train['path']}\t"
        f"{canonical_train.get('size', 0)}\t{canonical_train.get('mtime_ns', 0)}"
    )
    (dataset_dir / "source-layout.txt").write_text("\n".join(layout_lines) + "\n", encoding="utf-8")
    atomic_json(dataset_dir / "sampling-summary.json", summary)
    atomic_json(dataset_dir / "source-schema.json", schema)
    atomic_json(dataset_dir / "format-audit.json", summary["format"])
    atomic_json(dataset_dir / "media-audit.json", media_snapshot)
    atomic_json(dataset_dir / "input-files.json", file_reports)
    write_jsonl(dataset_dir / "sampling-manifest.jsonl", selected_manifest)
    write_jsonl(dataset_dir / "sampling-candidate-decisions.jsonl", decisions)
    write_jsonl(dataset_dir / "displayed-samples.jsonl", sample_rows)

    evidence = {
        "dataset": name,
        "access_mode": "ssh",
        "source_root": str(CONVERTED_ROOT),
        "declared_population_files": [str(path) for _, path in inputs],
        "fingerprint_algorithm": "sha256 over sorted path, exists, size, mtime_ns",
        "before_digest": before_digest,
        "after_digest": after_digest,
        "unchanged": before_digest == after_digest,
        "collector_transport": "uploaded collector; artifacts staged outside source root",
        "temporary_workspace": str(output_root),
        "read_operations": ["JSONL full scan", "media path existence", "sample media decode"],
    }
    return {
        "name": name,
        "display_name": display_name,
        "task_type": TASK_TYPES.get(name),
        "summary": summary,
        "evidence": evidence,
        "input_files": file_reports,
    }


def audit_media_population(media_references: dict[str, set[str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for media_type, references in media_references.items():
        remote = sorted(value for value in references if is_remote(value))
        local = sorted(value for value in references if not is_remote(value))
        present_count = 0
        missing_count = 0
        missing_examples = []
        unreadable_examples = []
        checked_local = local[:MAX_MEDIA_PRESENCE_CHECKS]
        unchecked_local = max(0, len(local) - len(checked_local))

        def check_path(raw_path: str) -> tuple[str, bool, str | None]:
            try:
                return raw_path, Path(raw_path).is_file(), None
            except OSError as exc:
                return raw_path, False, f"{type(exc).__name__}: {exc}"

        workers = min(32, max(1, (os.cpu_count() or 4) * 2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for raw_path, exists, error in executor.map(check_path, checked_local, chunksize=1024):
                if exists:
                    present_count += 1
                else:
                    missing_count += 1
                    if len(missing_examples) < 20:
                        missing_examples.append(raw_path)
                    if error and len(unreadable_examples) < 20:
                        unreadable_examples.append({"path": raw_path, "error": error})
        result[media_type] = {
            "unique_references": len(references),
            "local_references": len(local),
            "remote_references": len(remote),
            "local_presence_check_mode": (
                "complete_reference_stat" if unchecked_local == 0 else "bounded_reference_stat"
            ),
            "local_presence_checked": len(checked_local),
            "local_presence_unchecked": unchecked_local,
            "local_present": present_count,
            "local_missing": missing_count,
            "local_presence_ratio": present_count / len(checked_local) if checked_local else None,
            "remote_examples": remote[:20],
            "missing_examples": missing_examples,
            "unreadable_reference_count": len(unreadable_examples),
            "unreadable_reference_examples": unreadable_examples,
        }
    return result


def archive_metadata_reports(input_paths: list[Path], dataset_dir: Path) -> dict[str, Any]:
    source_dir = input_paths[0].parent
    candidates = sorted(
        {
            *source_dir.glob("*report*.json"),
            *source_dir.glob("*.stats.json"),
        },
        key=lambda path: path.name,
    )
    output: dict[str, Any] = {}
    for path in candidates:
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        relative = Path("source-metadata") / path.name
        destination = dataset_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            value = {"parse_error": f"{type(exc).__name__}: {exc}"}
        output[path.name] = {
            "source_path": str(path),
            "archive_path": relative.as_posix(),
            "size": path.stat().st_size,
            "content": safe_value(value),
        }
    return output


def copy_image(source: Path, dataset_dir: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_key = str(source)
    if source_key in cache:
        return dict(cache[source_key])
    if not source.is_file():
        result = {"source": source_key, "status": "missing"}
        cache[source_key] = result
        return dict(result)
    try:
        staging_dir = dataset_dir / "media"
        staging_dir.mkdir(parents=True, exist_ok=True)
        temporary = staging_dir / f".{os.getpid()}-{hashlib.sha256(source_key.encode('utf-8')).hexdigest()}.tmp"
        hasher = hashlib.sha256()
        byte_length = 0
        with source.open("rb") as stream, temporary.open("wb") as output_stream:
            for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                hasher.update(chunk)
                output_stream.write(chunk)
                byte_length += len(chunk)
        digest = hasher.hexdigest()
        with Image.open(temporary) as image:
            image.load()
            width, height = image.size
            image_format = (image.format or source.suffix.lstrip(".") or "bin").lower()
        extension = source.suffix.lower() or (".jpg" if image_format == "jpeg" else f".{image_format}")
        relative = Path("media") / f"{digest}{extension}"
        destination = dataset_dir / relative
        if destination.is_file():
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, destination)
        result = {
            "source": source_key,
            "status": "available",
            "archive_path": relative.as_posix(),
            "sha256": digest,
            "byte_length": byte_length,
            "width": width,
            "height": height,
            "format": image_format,
        }
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)  # type: ignore[name-defined]
        except Exception:
            pass
        result = {
            "source": source_key,
            "status": "decode_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    cache[source_key] = result
    return dict(result)


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def image_ids_for_boxes(objects: dict[str, Any], box_count: int) -> list[int]:
    raw = objects.get("image_id")
    if isinstance(raw, list) and len(raw) == box_count and all(isinstance(item, int) for item in raw):
        return list(raw)
    if raw is None:
        return [0] * box_count
    return []


def normalize_box_or_point(
    values: list[int | float],
    *,
    width: int,
    height: int,
    bbox_type: str,
) -> tuple[float, ...] | None:
    if len(values) not in {2, 4}:
        return None
    nums = [float(value) for value in values]
    if bbox_type == "norm1":
        scale = [width, height] if len(nums) == 2 else [width, height, width, height]
        nums = [value * scale[index] for index, value in enumerate(nums)]
    elif bbox_type in {"norm1000", "qwen1000"}:
        scale = [width / 1000.0, height / 1000.0] if len(nums) == 2 else [
            width / 1000.0,
            height / 1000.0,
            width / 1000.0,
            height / 1000.0,
        ]
        nums = [value * scale[index] for index, value in enumerate(nums)]
    elif bbox_type != "real":
        return None
    return tuple(nums)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str) -> None:
    x, y = xy
    pad = 3
    try:
        bbox = draw.textbbox((x, y), text)
        background = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    except Exception:
        background = (x, y, x + 9 * len(text), y + 16)
    draw.rectangle(background, fill=(255, 255, 255))
    draw.text((x, y), text, fill=(210, 20, 20))


def render_image_gt_overlay(
    image_asset: dict[str, Any],
    primitives: list[tuple[int, list[int | float]]],
    objects: dict[str, Any],
    dataset_dir: Path,
    sample_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    source_path = image_asset.get("archive_path")
    if not source_path:
        return None, {"status": "ground_truth_overlay_unresolved", "reason": "image_asset_missing_archive_path"}
    bbox_type = str(objects.get("bbox_type") or "real")
    source = dataset_dir / source_path
    if not source.is_file():
        return None, {"status": "ground_truth_overlay_unresolved", "reason": "image_archive_missing"}
    rendered = 0
    unresolved = 0
    try:
        with Image.open(source) as original:
            image = original.convert("RGB")
            width, height = image.size
        draw = ImageDraw.Draw(image)
        draw_label(draw, (10, 10), "GT")
        for primitive_index, box in primitives:
            normalized = normalize_box_or_point(box, width=width, height=height, bbox_type=bbox_type)
            if normalized is None:
                unresolved += 1
                continue
            if len(normalized) == 2:
                x, y = normalized
                radius = max(4, int(min(width, height) * 0.008))
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 0, 0), width=4)
                draw.line((x - radius * 2, y, x + radius * 2, y), fill=(255, 0, 0), width=2)
                draw.line((x, y - radius * 2, x, y + radius * 2), fill=(255, 0, 0), width=2)
                draw_label(draw, (x + radius + 3, y + radius + 3), f"GT p{primitive_index}")
            else:
                x1, y1, x2, y2 = normalized
                draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=max(3, int(min(width, height) * 0.004)))
                draw_label(draw, (x1, max(0, y1 - 18)), f"GT box{primitive_index}")
            rendered += 1
        if rendered == 0:
            return None, {
                "status": "ground_truth_overlay_unresolved",
                "reason": "no_supported_image_primitives",
                "declared_primitives": len(primitives),
                "rendered_primitives": rendered,
                "unresolved_primitives": unresolved,
                "bbox_type": bbox_type,
            }
        overlay_relative = Path("derived-preview") / f"{sample_id}-gt-overlay.jpg"
        overlay_path = dataset_dir / overlay_relative
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(overlay_path, format="JPEG", quality=92)
        digest = sha256_path(overlay_path)
        asset = {
            "type": "images",
            "source": image_asset.get("source"),
            "status": "available",
            "archive_path": overlay_relative.as_posix(),
            "sha256": digest,
            "width": width,
            "height": height,
            "format": "jpeg",
            "derived_preview": "GT overlay rendered on a copy of the archived original image",
            "gt_overlay": True,
            "clean_archive_path": source_path,
        }
        evidence = {
            "status": "rendered",
            "media_type": "images",
            "target_media_archive_path": source_path,
            "overlay_path": overlay_relative.as_posix(),
            "overlay_sha256": digest,
            "coordinate_convention": bbox_type,
            "declared_primitives": len(primitives),
            "rendered_primitives": rendered,
            "unresolved_primitives": unresolved,
            "image_width": width,
            "image_height": height,
        }
        return asset, evidence
    except Exception as exc:
        return None, {"status": "ground_truth_overlay_unresolved", "reason": f"{type(exc).__name__}: {exc}"}


def ffprobe_video(source: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,avg_frame_rate",
        "-of",
        "json",
        str(source),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    return json.loads(completed.stdout)


TRACK_POINT_RE = re.compile(
    r"(?:Object\s+(?P<object>\d+),\s*)?frame\s+(?P<frame>\d+)\s*(?:\((?P<time>[0-9.]+)s\))?\s*:\s*"
    r"\[(?P<x>-?\d+(?:\.\d+)?),\s*(?P<y>-?\d+(?:\.\d+)?)\]",
    re.IGNORECASE,
)


def parse_video_track_points(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    outputs = [
        text_from_content(message.get("content"))
        for message in messages
        if isinstance(message, dict) and str(message.get("role") or "") == "assistant"
    ]
    points = []
    for text in outputs:
        for match in TRACK_POINT_RE.finditer(text):
            points.append(
                {
                    "object": int(match.group("object") or 0),
                    "frame": int(match.group("frame")),
                    "time_seconds": float(match.group("time")) if match.group("time") is not None else None,
                    "x": float(match.group("x")),
                    "y": float(match.group("y")),
                }
            )
    return points


def representative_video_points(points: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    if len(points) <= limit:
        return points
    wanted = {
        round(index * (len(points) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [points[index] for index in sorted(wanted)]


def render_video_gt_overlay(
    source: Path,
    points: list[dict[str, Any]],
    dataset_dir: Path,
    sample_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not source.is_file():
        return None, {"status": "ground_truth_overlay_unresolved", "reason": "video_missing"}
    selected = representative_video_points(points)
    if not selected:
        return None, {"status": "ground_truth_overlay_unresolved", "reason": "no_parseable_frame_points"}
    try:
        probe = ffprobe_video(source)
        video_stream = next(
            (
                stream
                for stream in probe.get("streams", [])
                if isinstance(stream, dict) and stream.get("codec_type") == "video"
            ),
            {},
        )
        width = int(video_stream.get("width") or 0)
        height = int(video_stream.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError("video width/height unavailable")
        frame_dir = dataset_dir / "derived-preview" / f"{sample_id}-gt-video-frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        rendered_frames = []
        for index, point in enumerate(selected):
            frame_path = frame_dir / f"frame-{index:02d}.jpg"
            if point.get("time_seconds") is not None:
                seek_args = ["-ss", f"{float(point['time_seconds']):.6f}"]
                selector_note = f"time={point['time_seconds']:.6f}s"
            else:
                seek_args = []
                selector_note = f"frame={point['frame']}"
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                *seek_args,
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(frame_path),
            ]
            if not frame_path.is_file():
                subprocess.run(command, capture_output=True, timeout=120, check=True)
            with Image.open(frame_path) as original:
                image = original.convert("RGB")
                frame_width, frame_height = image.size
            draw = ImageDraw.Draw(image)
            x = point["x"] / 1000.0 * frame_width
            y = point["y"] / 1000.0 * frame_height
            radius = max(4, int(min(frame_width, frame_height) * 0.012))
            draw_label(draw, (10, 10), "GT")
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 0, 0), width=4)
            draw.line((x - radius * 2, y, x + radius * 2, y), fill=(255, 0, 0), width=2)
            draw.line((x, y - radius * 2, x, y + radius * 2), fill=(255, 0, 0), width=2)
            draw_label(draw, (x + radius + 3, y + radius + 3), f"GT f{point['frame']}")
            image.save(frame_path, format="JPEG", quality=92)
            rendered_frames.append(
                {
                    "path": frame_path,
                    "selector": selector_note,
                    "frame": point["frame"],
                    "object": point["object"],
                    "x_norm1000": point["x"],
                    "y_norm1000": point["y"],
                }
            )
        images = [Image.open(item["path"]).convert("RGB") for item in rendered_frames]
        tile_width = max(image.size[0] for image in images)
        tile_height = max(image.size[1] for image in images)
        columns = min(3, len(images))
        rows = (len(images) + columns - 1) // columns
        contact = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
        for index, image in enumerate(images):
            contact.paste(image, ((index % columns) * tile_width, (index // columns) * tile_height))
            image.close()
        overlay_relative = Path("derived-preview") / f"{sample_id}-gt-video-overlay.jpg"
        overlay_path = dataset_dir / overlay_relative
        contact.save(overlay_path, format="JPEG", quality=92)
        digest = sha256_path(overlay_path)
        asset = {
            "type": "videos",
            "source": str(source),
            "status": "available",
            "archive_path": overlay_relative.as_posix(),
            "preview_sha256": digest,
            "preview_width": contact.size[0],
            "preview_height": contact.size[1],
            "derived_preview": "GT overlay on parsed video frame points as a contact sheet",
            "gt_overlay": True,
        }
        evidence = {
            "status": "rendered",
            "media_type": "videos",
            "target_media": str(source),
            "overlay_path": overlay_relative.as_posix(),
            "overlay_sha256": digest,
            "coordinate_convention": "norm1000_points_from_assistant_text",
            "declared_primitives": len(points),
            "rendered_primitives": len(rendered_frames),
            "frame_selection": [
                {key: value for key, value in item.items() if key != "path"}
                for item in rendered_frames
            ],
        }
        return asset, evidence
    except Exception as exc:
        return None, {"status": "ground_truth_overlay_unresolved", "reason": f"{type(exc).__name__}: {exc}"}


def video_preview(source: Path, dataset_dir: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_key = str(source)
    if source_key in cache:
        return dict(cache[source_key])
    if not source.is_file():
        result = {"source": source_key, "status": "missing"}
        cache[source_key] = result
        return dict(result)
    try:
        probe = ffprobe_video(source)
        duration = float((probe.get("format") or {}).get("duration") or 0.0)
        if duration <= 0:
            raise ValueError("video duration is unavailable")
        identity = hashlib.sha256(
            f"{source}\0{source.stat().st_size}\0{source.stat().st_mtime_ns}".encode("utf-8")
        ).hexdigest()
        relative = Path("derived-preview") / f"{identity}.jpg"
        destination = dataset_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            interval = max(duration / 6.0, 0.04)
            vf = (
                f"fps=1/{interval:.6f},scale=320:-2:flags=lanczos,"
                "tile=3x2:nb_frames=6:padding=4:margin=4:color=white"
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-vf",
                    vf,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(destination),
                ],
                capture_output=True,
                timeout=120,
                check=True,
            )
        with Image.open(destination) as image:
            image.load()
            preview_size = image.size
        preview_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        result = {
            "source": source_key,
            "status": "available",
            "archive_path": relative.as_posix(),
            "preview_sha256": preview_hash,
            "preview_width": preview_size[0],
            "preview_height": preview_size[1],
            "source_size": source.stat().st_size,
            "source_mtime_ns": source.stat().st_mtime_ns,
            "duration_seconds": duration,
            "probe": probe,
            "derived_preview": "six uniformly sampled frames in a 3x2 JPEG contact sheet",
        }
    except Exception as exc:
        result = {
            "source": source_key,
            "status": "decode_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    cache[source_key] = result
    return dict(result)


def archive_samples(
    dataset_name: str,
    candidates: list[dict[str, Any]],
    dataset_dir: Path,
) -> list[dict[str, Any]]:
    image_cache: dict[str, dict[str, Any]] = {}
    video_cache: dict[str, dict[str, Any]] = {}
    output = []
    for index, candidate in enumerate(candidates, 1):
        sample_id = f"{dataset_name}-{index:03d}"
        record_path = Path("records") / f"{sample_id}.json"
        record = candidate["record"]
        atomic_json(
            dataset_dir / record_path,
            {
                "source_path": candidate["source_path"],
                "line_number": candidate["line_number"],
                "raw_record": record,
            },
        )
        assets = []
        all_references = []
        for media_type in PLACEHOLDERS:
            references = candidate["media"].get(media_type) or []
            for media_index, reference in enumerate(references):
                all_references.append({"type": media_type, "reference": reference})
                if media_index >= 4:
                    continue
                if is_remote(reference):
                    downloaded = (
                        pixmo_downloaded_path(dataset_name, reference)
                        if media_type == "images"
                        else None
                    )
                    if downloaded is not None:
                        asset = copy_image(downloaded, dataset_dir, image_cache)
                        asset.update(
                            {
                                "type": media_type,
                                "source_remote": reference,
                                "resolved_downloaded_path": str(downloaded),
                            }
                        )
                        assets.append(asset)
                    else:
                        assets.append(
                            {
                                "type": media_type,
                                "source": reference,
                                "status": "remote_reference_not_downloaded",
                            }
                        )
                elif media_type == "images":
                    assets.append({"type": media_type, **copy_image(Path(reference), dataset_dir, image_cache)})
                elif media_type == "videos":
                    assets.append({"type": media_type, **video_preview(Path(reference), dataset_dir, video_cache)})
                else:
                    source = Path(reference)
                    assets.append(
                        {
                            "type": media_type,
                            "source": reference,
                            "status": "available" if source.is_file() else "missing",
                        }
                    )
        gt_overlays = []
        gt_evidence = []
        if isinstance(record, dict):
            objects = record.get("objects")
            if isinstance(objects, dict) and isinstance(objects.get("bbox"), list) and objects.get("bbox"):
                boxes = [
                    box
                    for box in objects.get("bbox") or []
                    if isinstance(box, list) and len(box) in {2, 4}
                ]
                image_ids = image_ids_for_boxes(objects, len(boxes))
                if not image_ids:
                    gt_evidence.append(
                        {
                            "status": "ground_truth_overlay_unresolved",
                            "reason": "image_id_missing_or_length_mismatch",
                            "declared_primitives": len(boxes),
                        }
                    )
                else:
                    image_assets = [
                        asset
                        for asset in assets
                        if asset.get("type") == "images" and asset.get("status") == "available"
                    ]
                    for image_index, image_asset in enumerate(image_assets):
                        primitives = [
                            (primitive_index, box)
                            for primitive_index, (box, target_image_id) in enumerate(zip(boxes, image_ids))
                            if target_image_id == image_index
                        ]
                        if not primitives:
                            continue
                        overlay, evidence = render_image_gt_overlay(
                            image_asset,
                            primitives,
                            objects,
                            dataset_dir,
                            f"{sample_id}-image{image_index}",
                        )
                        gt_evidence.append(evidence)
                        if overlay is not None:
                            gt_overlays.append(overlay)
                    if not any(item.get("status") == "rendered" for item in gt_evidence):
                        gt_evidence.append(
                            {
                                "status": "ground_truth_overlay_unresolved",
                                "reason": "no_available_image_asset_for_declared_objects",
                                "declared_primitives": len(boxes),
                            }
                        )
            video_points = parse_video_track_points(record)
            if video_points:
                local_videos = [
                    item["reference"]
                    for item in all_references
                    if item.get("type") == "videos"
                    and isinstance(item.get("reference"), str)
                    and not is_remote(str(item["reference"]))
                ]
                if local_videos:
                    overlay, evidence = render_video_gt_overlay(
                        Path(local_videos[0]),
                        video_points,
                        dataset_dir,
                        f"{sample_id}-video0",
                    )
                    gt_evidence.append(evidence)
                    if overlay is not None:
                        gt_overlays.append(overlay)
                else:
                    gt_evidence.append(
                        {
                            "status": "ground_truth_overlay_unresolved",
                            "reason": "parseable_video_points_but_no_local_video",
                            "declared_primitives": len(video_points),
                        }
                    )
        if gt_overlays:
            assets = gt_overlays + assets
        statuses = [asset["status"] for asset in assets]
        if not all_references:
            media_status = "text_only"
        elif statuses and all(status == "available" for status in statuses):
            media_status = "available"
        elif any(status == "available" for status in statuses):
            media_status = "partial"
        elif any(status == "remote_reference_not_downloaded" for status in statuses):
            media_status = "remote_not_downloaded"
        elif any(status == "decode_error" for status in statuses):
            media_status = "decode_error"
        else:
            media_status = "missing"
        question, answer = message_preview(record.get("messages") if isinstance(record, dict) else None)
        output.append(
            {
                "sample_id": sample_id,
                "population_index": candidate["population_index"],
                "source_path": candidate["source_path"],
                "source_file": candidate["source_file"],
                "line_number": candidate["line_number"],
                "split": candidate["split"],
                "record_archive_path": record_path.as_posix(),
                "format_errors": candidate["format_errors"],
                "media_status": media_status,
                "media_references": all_references,
                "media_assets": assets,
                "ground_truth_overlay": gt_evidence,
                "input_preview": question,
                "output_preview": answer,
                "raw_record": record,
                "metadata_preview": {
                    key: value
                    for key, value in record.items()
                    if key not in {"messages", "images", "videos", "audios", "objects"}
                }
                if isinstance(record, dict)
                else {},
                "objects": record.get("objects") if isinstance(record, dict) else None,
            }
        )
        if index % 20 == 0:
            print(f"[{dataset_name}] archived samples={index}/{len(candidates)}", flush=True)
    return output


def excluded_inventory(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    formal_names = set(manifest["datasets"])
    formal_roots = {
        Path(config.get("source_train") or config.get("train") or "").parent.name
        for config in manifest["datasets"].values()
        if isinstance(config, dict)
    }
    rows = []
    try:
        children = sorted(CONVERTED_ROOT.iterdir(), key=lambda path: path.name.casefold())
    except OSError:
        list(CONVERTED_ROOT.parent.iterdir())
        children = sorted(CONVERTED_ROOT.iterdir(), key=lambda path: path.name.casefold())
    for child in children:
        if child.name in formal_roots:
            continue
        reason = "not listed in curated_dataset_entrypoints.json"
        if child.name.startswith("."):
            reason = "repository/tooling or temporary artifact"
        elif "backup" in child.name.casefold():
            reason = "backup conversion"
        elif "smoke" in child.name.casefold():
            reason = "smoke conversion"
        elif child.name in {"random_sample_report"}:
            reason = "derived report directory"
        elif child.is_file():
            reason = "root metadata or temporary file"
        rows.append(
            {
                "path": str(child),
                "name": child.name,
                "kind": "directory" if child.is_dir() else "file",
                "reason": reason,
            }
        )
    disabled = [
        {
            "path": str(Path(config.get("source_train") or config.get("train") or "").parent),
            "name": name,
            "kind": "manifest_dataset",
            "reason": config.get("reason") or config.get("status") or "disabled in manifest",
        }
        for name, config in manifest["datasets"].items()
        if isinstance(config, dict) and not config.get("enabled", True)
    ]
    return disabled + rows


def process_snapshot() -> list[str]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,stat=,etime=,cmd="],
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError:
        return []
    keywords = ("download_pixmo_media", "global_media_dedup", "prepare_molmo")
    return [line.strip() for line in completed.stdout.splitlines() if any(key in line for key in keywords)]


def run(manifest_path: Path, output_root: Path, selected_names: set[str] | None = None) -> None:
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to(CONVERTED_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Output root must remain outside the converted source root")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    configs = enabled_configs(manifest)
    if selected_names:
        known = {name for name, _ in configs}
        unknown = selected_names - known
        if unknown:
            raise ValueError(f"Unknown or disabled datasets: {sorted(unknown)}")
        configs = [(name, config) for name, config in configs if name in selected_names]
    status: dict[str, Any] = {
        "state": "running",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "datasets_total": len(configs),
        "datasets_completed": 0,
        "current_dataset": None,
    }
    atomic_json(output_root / "collection-status.json", status)
    registry = []
    evidence_entries = []
    active_processes_before = process_snapshot()
    try:
        for index, (name, config) in enumerate(configs):
            status["current_dataset"] = name
            atomic_json(output_root / "collection-status.json", status)
            print(f"[dataset {index + 1}/{len(configs)}] {name}", flush=True)
            result = scan_dataset(name, config, index, output_root, status)
            registry.append(result)
            evidence_entries.append(result["evidence"])
            status["datasets_completed"] = index + 1
            atomic_json(output_root / "collection-status.json", status)
    except BaseException as exc:
        status.update(
            {
                "state": "failed",
                "finished_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_json(output_root / "collection-status.json", status)
        raise

    active_processes_after = process_snapshot()
    source_evidence = {
        "version": 1,
        "datasets": evidence_entries,
        "media_snapshot_note": (
            "PixMo downloaders may change media availability during collection; read-only "
            "evidence covers declared JSONL populations, while media counts are timestamped snapshots."
        ),
    }
    top_summary = {
        "version": 1,
        "generated_at": utc_now(),
        "manifest": str(manifest_path),
        "dataset_count": len(registry),
        "target_rows_per_dataset": TARGET_ROWS,
        "displayed_rows_total": sum(item["summary"]["selected_count"] for item in registry),
        "datasets": [
            {
                "name": item["name"],
                "display_name": item["display_name"],
                "task_type": item["task_type"],
                **item["summary"],
            }
            for item in registry
        ],
        "excluded_inventory": excluded_inventory(manifest),
        "active_processes_before": active_processes_before,
        "active_processes_after": active_processes_after,
    }
    atomic_json(output_root / "dataset-registry.json", registry)
    atomic_json(output_root / "overall-summary.json", top_summary)
    atomic_json(output_root / "source-readonly-evidence.json", source_evidence)
    shutil.copyfile(manifest_path, output_root / "curated_dataset_entrypoints.json")
    status.update(
        {
            "state": "completed",
            "finished_at": utc_now(),
            "current_dataset": None,
        }
    )
    atomic_json(output_root / "collection-status.json", status)
    print(
        f"[completed] datasets={len(registry)} displayed_rows="
        f"{top_summary['displayed_rows_total']}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.inspect_only:
        print(json.dumps(inspect_manifest(args.manifest), ensure_ascii=False, indent=2))
        return 0
    run(args.manifest, args.output, set(args.datasets) if args.datasets else None)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
