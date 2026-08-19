#!/usr/bin/env python3
"""Refresh post-audit entrypoint and PixMo media evidence without resampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from PIL import Image


PIXMO_DATASETS = ("pixmo-cap", "pixmo-points")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


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


def downloaded_path(converted_root: Path, dataset: str, reference: str) -> Path | None:
    identity = hashlib.sha256(canonical_http_url(reference).encode("utf-8")).hexdigest()
    image_root = converted_root / dataset / "images" / identity[:2]
    for path in sorted(image_root.glob(f"{identity}.*")):
        if path.is_file():
            return path
    return None


def archive_image(source: Path, remote: str, dataset_dir: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    with Image.open(source) as image:
        image.load()
        width, height = image.size
        image_format = (image.format or source.suffix.lstrip(".") or "bin").lower()
    extension = source.suffix.lower() or (".jpg" if image_format == "jpeg" else f".{image_format}")
    relative = Path("media") / f"{digest}{extension}"
    destination = dataset_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        shutil.copyfile(source, destination)
    return {
        "type": "images",
        "source": str(source),
        "source_remote": remote,
        "resolved_downloaded_path": str(source),
        "status": "available",
        "archive_path": relative.as_posix(),
        "sha256": digest,
        "byte_length": source.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
    }


def refresh_pixmo_dataset(
    output_root: Path, converted_root: Path, dataset: str
) -> dict[str, Any]:
    dataset_dir = output_root / "sub-dataset" / dataset
    displayed_path = dataset_dir / "displayed-samples.jsonl"
    rows = load_jsonl(displayed_path)
    before = Counter(str(row.get("media_status")) for row in rows)
    for row in rows:
        assets = []
        references = [
            item.get("reference")
            for item in row.get("media_references") or []
            if item.get("type") == "images" and isinstance(item.get("reference"), str)
        ]
        for reference in references[:4]:
            try:
                source = downloaded_path(converted_root, dataset, reference)
                if source is None:
                    assets.append(
                        {
                            "type": "images",
                            "source": reference,
                            "status": "remote_reference_not_downloaded",
                        }
                    )
                else:
                    assets.append(archive_image(source, reference, dataset_dir))
            except Exception as exc:
                assets.append(
                    {
                        "type": "images",
                        "source": reference,
                        "status": "decode_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        statuses = [asset["status"] for asset in assets]
        if not references:
            media_status = "text_only"
        elif statuses and all(status == "available" for status in statuses):
            media_status = "available"
        elif any(status == "available" for status in statuses):
            media_status = "partial"
        elif any(status == "decode_error" for status in statuses):
            media_status = "decode_error"
        else:
            media_status = "remote_not_downloaded"
        row["media_assets"] = assets
        row["media_status"] = media_status

    after = Counter(str(row.get("media_status")) for row in rows)
    write_jsonl(displayed_path, rows)

    manifest_path = dataset_dir / "sampling-manifest.jsonl"
    manifest = load_jsonl(manifest_path)
    refreshed = {row["sample_id"]: row for row in rows}
    for item in manifest:
        current = refreshed[item["sample_id"]]
        item["media_status"] = current["media_status"]
        item["media_assets"] = current["media_assets"]
    write_jsonl(manifest_path, manifest)

    summary_path = dataset_dir / "sampling-summary.json"
    summary = load_json(summary_path)
    summary["sample_media_statuses"] = dict(after)
    summary["post_download_sample_refresh"] = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "sample_ids_unchanged": True,
        "before": dict(before),
        "after": dict(after),
    }
    write_json(summary_path, summary)
    return summary["post_download_sample_refresh"]


def collect_post_audit_status(
    output_root: Path, converted_root: Path, overall: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    checked_at = datetime.now(timezone.utc).isoformat()
    manifest = load_json(Path(overall["manifest"]))
    validator = load_json(output_root / "entrypoint-validation.json")
    names = {item["name"] for item in overall["datasets"]}
    entrypoints = {}
    for name, config in manifest["datasets"].items():
        if name not in names:
            continue
        train = Path(config["train"])
        entrypoints[name] = {
            "path": str(train),
            "exists": train.is_file(),
            "size": train.stat().st_size if train.is_file() else None,
        }
    entrypoint_status = {
        "checked_at": checked_at,
        "official_validator_returncode": validator.get("returncode"),
        "official_validator_passed": validator.get("returncode") == 0,
        "ready_count": sum(bool(row["exists"] and row["size"]) for row in entrypoints.values()),
        "dataset_count": len(entrypoints),
        "eval_count": len(validator.get("eval_entries") or []),
        "datasets": entrypoints,
    }

    pixmo = {}
    for dataset in PIXMO_DATASETS:
        download = load_json(converted_root / dataset / "media_download_report.json")
        localization = load_json(converted_root / dataset / "localization_report.json")
        pixmo[dataset] = {
            "checked_at": checked_at,
            "download_report": download,
            "localization_report": localization,
        }
    return entrypoint_status, {"checked_at": checked_at, "datasets": pixmo}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--converted-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    converted_root = args.converted_root.resolve()
    overall_path = output_root / "overall-summary.json"
    overall = load_json(overall_path)

    present_names = {item["name"] for item in overall["datasets"]}
    refresh_results = {
        dataset: refresh_pixmo_dataset(output_root, converted_root, dataset)
        for dataset in PIXMO_DATASETS
        if dataset in present_names
    }
    entrypoints, media = collect_post_audit_status(output_root, converted_root, overall)
    media["sample_refresh"] = refresh_results
    write_json(output_root / "post-audit-entrypoint-status.json", entrypoints)
    write_json(output_root / "post-audit-media-status.json", media)

    for item in overall["datasets"]:
        status = entrypoints["datasets"].get(item["name"])
        item["post_audit_canonical_train_ready"] = bool(
            status and status.get("exists") and status.get("size")
        )
        if item["name"] in refresh_results:
            current = load_json(
                output_root / "sub-dataset" / item["name"] / "sampling-summary.json"
            )
            item["sample_media_statuses"] = current["sample_media_statuses"]
            item["post_download_sample_refresh"] = refresh_results[item["name"]]
    overall["post_audit_entrypoint_status"] = {
        key: value for key, value in entrypoints.items() if key != "datasets"
    }
    overall["post_audit_media_status_file"] = "post-audit-media-status.json"
    write_json(overall_path, overall)

    print(
        json.dumps(
            {
                "entrypoints": {
                    "ready": entrypoints["ready_count"],
                    "total": entrypoints["dataset_count"],
                    "validator_passed": entrypoints["official_validator_passed"],
                },
                "sample_refresh": refresh_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
