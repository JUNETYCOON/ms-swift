#!/usr/bin/env python3
"""Build a deterministic multimodal dataset for Stage 1 DLC performance tests."""

from __future__ import annotations

import argparse
import heapq
import json
import random
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def first_image_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        if row.get("images") and not row.get("videos") and not row.get("audios"):
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def largest_distinct_video_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    by_video: dict[str, tuple[int, dict[str, Any]]] = {}
    for row in iter_jsonl(path):
        videos = row.get("videos") or []
        if not videos:
            continue
        video = str(videos[0])
        if video in by_video:
            continue
        video_path = Path(video)
        try:
            size = video_path.stat().st_size
        except OSError:
            continue
        by_video[video] = (size, row)
    largest = heapq.nlargest(limit, by_video.values(), key=lambda item: item[0])
    return [row for _, row in largest]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/mnt/luojunkun/stage1/ms-swift/scripts/dlc_ready_entrypoints.stage1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/luojunkun/stage1/perf/stage1_perf_mix.jsonl"),
    )
    parser.add_argument("--images-per-dataset", type=int, default=32)
    parser.add_argument("--video-count", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    image_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    per_dataset: dict[str, int] = {}

    for name, config in manifest["datasets"].items():
        if not config.get("enabled", True):
            continue
        source = Path(config["train"])
        if name == "molmo2-video-track":
            rows = largest_distinct_video_rows(source, args.video_count)
            video_rows.extend(rows)
        else:
            rows = first_image_rows(source, args.images_per_dataset)
            image_rows.extend(rows)
        per_dataset[name] = len(rows)

    selected = image_rows + video_rows
    random.Random(args.seed).shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "output": str(args.output),
        "rows": len(selected),
        "image_rows": len(image_rows),
        "video_rows": len(video_rows),
        "per_dataset": per_dataset,
        "seed": args.seed,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
