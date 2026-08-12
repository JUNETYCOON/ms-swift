#!/usr/bin/env python3
"""Run final source fingerprints and build unified read-only evidence."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARTIFACTS = ROOT / "artifacts"
REMOTE_HOST = "8.146.226.25"
GROUPS = [
    ["COCO", "VQAv2", "GQA"],
    ["VisualGenome", "ai2d", "llava-instruct"],
    ["textvqa", "Chartqa", "robo2vlm"],
    ["robovqa", "vlm-r1"],
    [
        "pixmo-cap",
        "pixmo-points",
        "spatialvlm",
        "Molmo2-VideoCapQA",
        "Molmo2-VideoPoint",
        "Molmo2-VideoSubtitleQA",
        "Molmo2-VideoTrack",
    ],
]
REPORT_NAMES = {
    "COCO": "COCO",
    "VQAv2": "VQAv2",
    "GQA": "GQA",
    "VisualGenome": "VisualGenome",
    "ai2d": "AI2D",
    "llava-instruct": "LLaVA-Instruct",
    "textvqa": "TextVQA",
    "Chartqa": "ChartQA",
    "robo2vlm": "Robo2VLM",
    "robovqa": "RoboVQA",
    "vlm-r1": "VLM-R1",
    "pixmo-cap": "PixMo-Cap",
    "pixmo-points": "PixMo-Points",
    "spatialvlm": "SpatialVLM",
    "Molmo2-VideoCapQA": "Molmo2-VideoCapQA",
    "Molmo2-VideoPoint": "Molmo2-VideoPoint",
    "Molmo2-VideoSubtitleQA": "Molmo2-VideoSubtitleQA",
    "Molmo2-VideoTrack": "Molmo2-VideoTrack",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def before_entries() -> dict[str, dict]:
    output: dict[str, dict] = {}
    paths = sorted(ARTIFACTS.glob("source-probe-before-legacy-*.json"))
    paths.append(ARTIFACTS / "new-source-probe.json")
    for path in paths:
        for row in load(path).get("datasets", []):
            output[row["dataset"]] = row
    return output


def run_group(script: bytes, datasets: list[str]) -> list[dict]:
    process = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            REMOTE_HOST,
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            "python3",
            "-",
            *datasets,
            "--sample-count",
            "0",
            "--workers",
            "4",
        ],
        input=script,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=1800,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
    payload = json.loads(process.stdout.decode("utf-8"))
    return payload["datasets"]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=path.stem + "-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    script = (HERE / "remote_probe.py").read_bytes()
    after_rows: list[dict] = []
    for group in GROUPS:
        print("probing " + ", ".join(group), flush=True)
        after_rows.extend(run_group(script, group))
    after = {
        "version": 1,
        "source_root": "/mnt/luojunkun/stage1/dataset",
        "datasets": after_rows,
    }
    atomic_json(ARTIFACTS / "source-probe-after.json", after)

    before = before_entries()
    after_map = {row["dataset"]: row for row in after_rows}
    evidence_rows = []
    for group in GROUPS:
        for source_name in group:
            before_row = before[source_name]
            after_row = after_map[source_name]
            before_digest = before_row["fingerprint"]
            after_digest = after_row["fingerprint"]
            evidence_rows.append(
                {
                    "dataset": REPORT_NAMES[source_name],
                    "source_dataset_directory": source_name,
                    "access_mode": "ssh",
                    "source_root": after_row["source_root"],
                    "fingerprint_algorithm": after_row["fingerprint_algorithm"],
                    "before_digest": before_digest,
                    "after_digest": after_digest,
                    "before_file_count": before_row["file_count"],
                    "after_file_count": after_row["file_count"],
                    "before_total_file_bytes": before_row.get("total_file_bytes", sum(item["size"] for item in before_row.get("files", []))),
                    "after_total_file_bytes": after_row.get("total_file_bytes", sum(item["size"] for item in after_row.get("files", []))),
                    "unchanged": before_digest == after_digest,
                    "collector_transport": "collector code via SSH stdin; JSON evidence via stdout",
                    "source_writes": [],
                    "read_operations": [
                        "directory enumeration",
                        "file stat",
                        "Parquet footer/schema read",
                        "authorized source record/media read for analysis",
                    ],
                }
            )
    evidence = {
        "version": 1,
        "source_root": "/mnt/luojunkun/stage1/dataset",
        "source_only_contract": True,
        "fingerprint_time": "2026-08-06",
        "all_unchanged": all(row["unchanged"] for row in evidence_rows),
        "datasets": sorted(evidence_rows, key=lambda row: row["dataset"]),
        "scope_note": "Fingerprints cover each declared dataset directory; excluded root entries are recorded in inventory.csv.",
    }
    atomic_json(ROOT / "source-readonly-evidence.json", evidence)
    if not evidence["all_unchanged"]:
        changed = [row["dataset"] for row in evidence_rows if not row["unchanged"]]
        raise RuntimeError("source fingerprints changed: " + ", ".join(changed))
    print(f"all {len(evidence_rows)} source fingerprints unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
