"""Rewrite a ms-swift JSONL to node-local video paths, dropping failed files."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    source_jsonl, source_prefix, local_prefix, fail_log, output_jsonl = sys.argv[1:6]
    failed: set[str] = set()
    fail_path = Path(fail_log)
    if fail_path.exists():
        for line in fail_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line:
                failed.add(line)

    kept = 0
    dropped = 0
    with open(source_jsonl, encoding="utf-8") as source, open(output_jsonl, "w", encoding="utf-8") as output:
        for line in source:
            line = line.rstrip("\n")
            if not line:
                continue
            item = json.loads(line)
            videos = item.get("videos") or []
            rels = []
            drop = False
            for video in videos:
                if video.startswith(source_prefix + "/"):
                    rel = video[len(source_prefix) + 1 :]
                    rels.append(rel)
                    if rel in failed:
                        drop = True
            if drop:
                dropped += 1
                continue
            if rels:
                item["videos"] = [f"{local_prefix}/{rel}" for rel in rels]
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[dataset-prep] kept={kept} dropped={dropped} failed_files={len(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
