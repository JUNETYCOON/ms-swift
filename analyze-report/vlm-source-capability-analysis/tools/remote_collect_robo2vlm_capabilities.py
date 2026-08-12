#!/usr/bin/env python3
"""Classify canonical Robo2VLM source questions with deterministic text rules."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path("/mnt/luojunkun/stage1/dataset/robo2vlm")

RULES = [
    (
        "cross_view_3d_correspondence",
        re.compile(r"left image|right image|same 3d location|corresponding to the same|ext[12] camera", re.I),
    ),
    (
        "depth_distance_estimation",
        re.compile(r"farthest|closest|nearer|farther|depth|distance from the camera", re.I),
    ),
    (
        "trajectory_language_alignment",
        re.compile(r"trajectory|language instruction best describes|path shown", re.I),
    ),
    (
        "success_judgment",
        re.compile(r"successfully completed|has the robot successfully|task successful|achieved the task", re.I),
    ),
    (
        "next_motion_prediction",
        re.compile(r"move next|next move|direction .*robot.*move|arrow correctly shows", re.I),
    ),
    (
        "obstacle_reachability",
        re.compile(r"obstacle|blocking the robot|reachable|reach the", re.I),
    ),
    (
        "goal_state_recognition",
        re.compile(r"goal state|configuration shows|desired state|final state", re.I),
    ),
    (
        "affordance_feasibility",
        re.compile(r"is it possible|can the robot|able to (?:pick|place|move|reach|open|close)", re.I),
    ),
    (
        "spatial_relation_localization",
        re.compile(r"which (?:point|side|location)|to the left|to the right|above|below|relative position", re.I),
    ),
    (
        "robot_action_object_understanding",
        re.compile(r"robot|task|instruction|action|gripper|arm", re.I),
    ),
]


def compact(value: Any) -> str:
    value = value if isinstance(value, str) else str(value or "")
    return value if len(value) <= 500 else value[:500] + "<truncated>"


def main() -> int:
    paths = sorted((ROOT / "data").glob("*.parquet"))
    primary_counts: Counter[str] = Counter()
    multilabel_counts: Counter[str] = Counter()
    rows_by_split: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_questions: set[str] = set()
    records = 0
    empty_questions = 0

    for path in paths:
        split = path.name.split("-", 1)[0]
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=32768, columns=["id", "question", "choices", "correct_answer"]
        ):
            for row in batch.to_pylist():
                records += 1
                rows_by_split[split] += 1
                question = compact(row.get("question"))
                unique_questions.add(question)
                empty_questions += int(not question.strip())
                matches = [name for name, pattern in RULES if pattern.search(question)]
                primary = matches[0] if matches else "other_unclassified"
                primary_counts[primary] += 1
                for name in matches:
                    multilabel_counts[name] += 1
                if len(examples[primary]) < 3:
                    examples[primary].append(
                        {
                            "id": compact(row.get("id")),
                            "question": question,
                            "choices": compact(row.get("choices")),
                            "correct_answer": row.get("correct_answer"),
                            "source_shard": path.name,
                        }
                    )
        parquet.close()

    result = {
        "dataset": "Robo2VLM",
        "source_root": str(ROOT),
        "source_only_contract": True,
        "evidence_type": "full canonical source question scan with versioned deterministic rules",
        "canonical_rule": "direct data/*.parquet only; nested data/data copies excluded",
        "canonical_records": records,
        "records_by_split": dict(rows_by_split),
        "unique_question_texts": len(unique_questions),
        "empty_questions": empty_questions,
        "primary_capability_counts": dict(primary_counts.most_common()),
        "multilabel_capability_counts": dict(multilabel_counts.most_common()),
        "primary_unclassified_rate": primary_counts["other_unclassified"] / records,
        "rule_order": [
            {"capability": name, "pattern": pattern.pattern} for name, pattern in RULES
        ],
        "representative_source_records_by_primary_capability": dict(examples),
        "classification_limit": (
            "Rules operate on source question text. Counts are full-population deterministic "
            "classifications, but semantic precision is estimated rather than human-adjudicated."
        ),
    }
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
