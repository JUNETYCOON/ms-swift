#!/usr/bin/env python3
"""Sample a fixed N-row Molmo2-VideoTrack smoke set from ready_train_norm1000."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_jsonl", type=Path)
    parser.add_argument("source_videos", type=Path)
    parser.add_argument("sample_count", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("sampled_jsonl", type=Path)
    parser.add_argument("video_list", type=Path)
    args = parser.parse_args()

    source_prefix = str(args.source_videos).replace("\\", "/").rstrip("/")
    rows = []
    for row in iter_rows(args.source_jsonl):
        videos = row.get("videos") or []
        rels = []
        for video in videos:
            video = str(video).replace("\\", "/")
            if video.startswith(source_prefix + "/"):
                rels.append(video[len(source_prefix) + 1 :])
        if rels:
            rows.append((row, sorted(set(rels))))

    if len(rows) < args.sample_count:
        raise SystemExit(
            f"not enough norm1000 rows with videos: {len(rows)} < {args.sample_count}"
        )

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(rows)), args.sample_count))
    selected = [rows[i] for i in indices]

    args.sampled_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.sampled_jsonl.open("w", encoding="utf-8") as output:
        for row, _ in selected:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    video_rels = sorted({rel for _, rels in selected for rel in rels})
    args.video_list.parent.mkdir(parents=True, exist_ok=True)
    args.video_list.write_text("".join(f"{rel}\n" for rel in video_rels), encoding="utf-8")

    print(
        f"[sample] rows={len(selected)} unique_videos={len(video_rels)} "
        f"seed={args.seed} -> {args.sampled_jsonl}"
    )
    print(f"[sample] video_list -> {args.video_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
