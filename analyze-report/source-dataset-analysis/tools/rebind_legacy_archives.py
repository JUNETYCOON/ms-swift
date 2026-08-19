#!/usr/bin/env python3
"""Rebind source-only legacy sample archives to the current source mount.

The historical archives were collected from the same relative source files on
an older mount path. This script proves path/size continuity against the
current before probes, copies the local archives, and rewrites only source
locators. It never connects to or writes on the remote host.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPORT_ROOT = HERE.parent
REPO_ROOT = REPORT_ROOT.parents[1]
LEGACY_ROOT = REPO_ROOT / "analyze-report" / "vlm-dataset-src-anal" / "sub-dataset"
ARTIFACT_ROOT = REPORT_ROOT / "artifacts"

OLD_BASE = "/mnt/oss-data/luojunkun/stage1/dataset"
NEW_BASE = "/mnt/luojunkun/stage1/dataset"

DATASETS = {
    "COCO": {"legacy": "COCO", "probe": "COCO", "unit": "unique_image"},
    "VQAv2": {"legacy": "VQAv2", "probe": "VQAv2", "unit": "joined_instruction_row"},
    "VisualGenome": {
        "legacy": "VisualGenome",
        "probe": "VisualGenome",
        "unit": "archive_member",
    },
    "GQA": {"legacy": "GQA", "probe": "GQA", "unit": "joined_instruction_row"},
    "TextVQA": {
        "legacy": "TextVQA",
        "probe": "textvqa",
        "unit": "joined_instruction_row",
    },
    "ChartQA": {
        "legacy": "ChartQA",
        "probe": "Chartqa",
        "unit": "joined_instruction_row",
    },
    "AI2D": {"legacy": "AI2D", "probe": "ai2d", "unit": "unique_image"},
    "LLaVA-Instruct": {
        "legacy": "LLaVA-style",
        "probe": "llava-instruct",
        "unit": "joined_instruction_row",
    },
    "VLM-R1": {"legacy": "VLM-R1", "probe": "vlm-r1", "unit": "logical_record"},
    "Robo2VLM": {
        "legacy": "Robo2VLM",
        "probe": "robo2vlm",
        "unit": "joined_instruction_row",
    },
    "RoboVQA": {"legacy": "RoboVQA", "probe": "robovqa", "unit": "unique_video"},
}


def load_probes() -> dict[str, dict[str, Any]]:
    datasets: dict[str, dict[str, Any]] = {}
    for path in sorted(ARTIFACT_ROOT.glob("*probe*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for dataset in payload.get("datasets", []):
            datasets[dataset["dataset"]] = dataset
    return datasets


def layout_files(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if not parts or parts[0] != "file" or len(parts) < 2:
            continue
        size = None
        if len(parts) >= 3 and parts[2].strip():
            try:
                size = int(parts[2])
            except ValueError:
                size = None
        rows.append({"relative_path": parts[1], "size": size})
    return rows


def compare_layout(
    historical: list[dict[str, Any]], current: list[dict[str, Any]]
) -> dict[str, Any]:
    matches = []
    missing = []
    size_mismatches = []
    size_unrecorded = []
    for old in historical:
        candidates = [
            item
            for item in current
            if item["relative_path"] == old["relative_path"]
            or item["relative_path"].endswith("/" + old["relative_path"])
        ]
        if not candidates:
            missing.append(old)
            continue
        if old["size"] is None:
            size_unrecorded.append(
                {
                    "historical": old,
                    "current": {
                        "relative_path": candidates[0]["relative_path"],
                        "size": candidates[0]["size"],
                    },
                }
            )
            continue
        same_size = [item for item in candidates if item["size"] == old["size"]]
        if same_size:
            matches.append(
                {
                    "historical_relative_path": old["relative_path"],
                    "current_relative_path": same_size[0]["relative_path"],
                    "size": old["size"],
                }
            )
        else:
            size_mismatches.append(
                {
                    "historical": old,
                    "current_candidates": [
                        {"relative_path": item["relative_path"], "size": item["size"]}
                        for item in candidates
                    ],
                }
            )
    return {
        "historical_layout_file_count": len(historical),
        "path_and_size_match_count": len(matches),
        "size_unrecorded_but_path_match_count": len(size_unrecorded),
        "missing_count": len(missing),
        "size_mismatch_count": len(size_mismatches),
        "all_historical_paths_present": not missing,
        "all_recorded_sizes_match": not size_mismatches,
        "matches": matches,
        "size_unrecorded_path_matches": size_unrecorded,
        "missing": missing,
        "size_mismatches": size_mismatches,
    }


def rewrite_source_locators(target: Path) -> None:
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text.replace(OLD_BASE, NEW_BASE)
        if updated != text:
            path.write_text(updated, encoding="utf-8", newline="\n")


def update_summary(
    target: Path,
    dataset: str,
    unit: str,
    evidence: dict[str, Any],
) -> None:
    path = target / "sampling-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    historical_root = summary.get("source_root")
    summary.update(
        {
            "dataset": dataset,
            "historical_source_root": historical_root,
            "source_root": str(historical_root).replace(OLD_BASE, NEW_BASE),
            "sampling_unit": unit,
            "source_rebound_from_historical_archive": True,
            "current_source_fingerprint_before": evidence["current_source_fingerprint"],
            "current_source_rebind_evidence": "current-source-rebind.json",
            "current_source_paths_present": evidence["comparison"][
                "all_historical_paths_present"
            ],
            "current_source_recorded_sizes_match": evidence["comparison"][
                "all_recorded_sizes_match"
            ],
        }
    )
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    probes = load_probes()
    for dataset, config in DATASETS.items():
        legacy = LEGACY_ROOT / config["legacy"]
        target = REPORT_ROOT / "sub-dataset" / dataset
        probe = probes[config["probe"]]
        comparison = compare_layout(
            layout_files(legacy / "source-layout.txt"), probe.get("files", [])
        )
        if not comparison["all_historical_paths_present"]:
            raise RuntimeError(f"historical source paths missing for {dataset}")
        if not comparison["all_recorded_sizes_match"]:
            raise RuntimeError(f"historical source sizes changed for {dataset}")

        shutil.copytree(legacy, target, dirs_exist_ok=True, copy_function=shutil.copy2)
        rewrite_source_locators(target)
        evidence = {
            "version": 1,
            "dataset": dataset,
            "evidence_boundary": (
                "The archived sample bytes were collected read-only from the historical mount. "
                "This rebind proves that every historical source-layout path remains present on "
                "the current source mount and that every historically recorded file size matches. "
                "It does not claim a full-file content hash for multi-gigabyte source containers."
            ),
            "historical_archive_source_base": OLD_BASE,
            "current_source_base": NEW_BASE,
            "current_source_root": probe["source_root"],
            "current_source_fingerprint_algorithm": probe["fingerprint_algorithm"],
            "current_source_fingerprint": probe["fingerprint"],
            "comparison": comparison,
        }
        (target / "current-source-rebind.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        update_summary(target, dataset, config["unit"], evidence)
        print(
            f"{dataset}: copied {comparison['historical_layout_file_count']} source-layout files; "
            f"{comparison['path_and_size_match_count']} path+size matches; "
            f"{comparison['size_unrecorded_but_path_match_count']} path-only matches"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
