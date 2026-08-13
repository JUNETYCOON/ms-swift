#!/usr/bin/env python3
"""Restore GT overlay media assets from overlay evidence after post-refresh steps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def overlay_asset(dataset_dir: Path, evidence: dict[str, Any]) -> dict[str, Any] | None:
    if evidence.get("status") != "rendered" or not evidence.get("overlay_path"):
        return None
    path = dataset_dir / str(evidence["overlay_path"])
    if not path.is_file():
        return None
    with Image.open(path) as image:
        image.load()
        width, height = image.size
    media_type = str(evidence.get("media_type") or "images")
    asset: dict[str, Any] = {
        "type": media_type,
        "source": evidence.get("target_media") or evidence.get("target_media_archive_path") or "",
        "status": "available",
        "archive_path": str(evidence["overlay_path"]),
        "gt_overlay": True,
        "derived_preview": "GT overlay restored from ground_truth_overlay evidence",
    }
    if media_type == "videos":
        asset.update({"preview_sha256": sha256_file(path), "preview_width": width, "preview_height": height})
    else:
        asset.update(
            {
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
                "format": path.suffix.lstrip(".").lower() or "jpg",
                "clean_archive_path": evidence.get("target_media_archive_path"),
            }
        )
    return asset


def restore_dataset(dataset_dir: Path) -> dict[str, Any]:
    displayed_path = dataset_dir / "displayed-samples.jsonl"
    manifest_path = dataset_dir / "sampling-manifest.jsonl"
    summary_path = dataset_dir / "sampling-summary.json"
    rows = load_jsonl(displayed_path)
    restored = 0
    for row in rows:
        existing = [
            asset
            for asset in (row.get("media_assets") or [])
            if isinstance(asset, dict) and not asset.get("gt_overlay")
        ]
        overlays = []
        for evidence in row.get("ground_truth_overlay") or []:
            if isinstance(evidence, dict):
                asset = overlay_asset(dataset_dir, evidence)
                if asset is not None:
                    overlays.append(asset)
        if overlays:
            row["media_assets"] = overlays + existing
            row["media_status"] = "available" if not existing else row.get("media_status", "available")
            restored += len(overlays)
    write_jsonl(displayed_path, rows)

    refreshed = {row["sample_id"]: row for row in rows}
    manifest = load_jsonl(manifest_path)
    for item in manifest:
        row = refreshed[item["sample_id"]]
        item["media_status"] = row.get("media_status")
        item["media_assets"] = row.get("media_assets")
        item["ground_truth_overlay"] = row.get("ground_truth_overlay") or []
    write_jsonl(manifest_path, manifest)

    summary = load_json(summary_path)
    summary["sample_media_statuses"] = dict(Counter(str(row.get("media_status")) for row in rows))
    summary["ground_truth_overlay_statuses"] = dict(
        Counter(
            overlay.get("status", "unknown")
            for row in rows
            for overlay in (row.get("ground_truth_overlay") or [])
            if isinstance(overlay, dict)
        )
    )
    summary["restored_gt_overlay_assets"] = restored
    write_json(summary_path, summary)
    return {"dataset": dataset_dir.name, "restored_gt_overlay_assets": restored}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    results = [
        restore_dataset(path)
        for path in sorted((args.root / "sub-dataset").iterdir())
        if (path / "displayed-samples.jsonl").is_file()
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
