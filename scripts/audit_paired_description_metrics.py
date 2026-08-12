#!/usr/bin/env python3
"""Aggregate description metrics on the exact shared sample population."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_METRICS = ("cider", "rouge_l", "bleu_4", "chrf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two benchmark results.csv files on shared description IDs. "
            "Rows with mismatched questions or references are rejected."
        )
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("ours", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--task", default="description")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    return parser.parse_args()


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def parse_score(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    score = float(text)
    if not math.isfinite(score):
        raise ValueError(f"non-finite metric value: {value!r}")
    return score


def load_rows(path: Path, task: str) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    rows: dict[str, dict[str, str]] = {}
    stats: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        required = {"sample_id", "task", "question", "reference"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"missing required columns in {path}: {missing}")
        for line_number, row in enumerate(reader, 2):
            stats["all_rows"] += 1
            if normalized_text(row.get("task")).lower() != task.lower():
                continue
            stats["task_rows"] += 1
            sample_id = normalized_text(row.get("sample_id"))
            if not sample_id:
                raise ValueError(f"empty sample_id at {path}:{line_number}")
            if sample_id in rows:
                raise ValueError(f"duplicate sample_id {sample_id!r} at {path}:{line_number}")
            rows[sample_id] = row
    return rows, {"path": str(path.resolve()), **stats}


def available_metrics(
    baseline_rows: Iterable[dict[str, str]],
    ours_rows: Iterable[dict[str, str]],
    requested: list[str],
) -> tuple[list[str], list[str]]:
    baseline_fields = set().union(*(row.keys() for row in baseline_rows))
    ours_fields = set().union(*(row.keys() for row in ours_rows))
    available = [metric for metric in requested if metric in baseline_fields and metric in ours_fields]
    unavailable = [metric for metric in requested if metric not in available]
    return available, unavailable


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    baseline, baseline_stats = load_rows(args.baseline.expanduser().resolve(), args.task)
    ours, ours_stats = load_rows(args.ours.expanduser().resolve(), args.task)
    shared_ids = sorted(set(baseline) & set(ours))
    if not shared_ids:
        raise ValueError("no shared description sample IDs")

    available, unavailable = available_metrics(baseline.values(), ours.values(), args.metrics)
    if not available:
        raise ValueError(f"none of the requested metrics are present: {args.metrics}")

    mismatches: list[dict[str, str]] = []
    metric_pairs: dict[str, list[tuple[float, float]]] = {metric: [] for metric in available}
    missing_values: Counter[str] = Counter()
    for sample_id in shared_ids:
        baseline_row = baseline[sample_id]
        ours_row = ours[sample_id]
        for field in ("question", "reference"):
            if normalized_text(baseline_row.get(field)) != normalized_text(ours_row.get(field)):
                mismatches.append({"sample_id": sample_id, "field": field})
        for metric in available:
            baseline_score = parse_score(baseline_row.get(metric))
            ours_score = parse_score(ours_row.get(metric))
            if baseline_score is None or ours_score is None:
                missing_values[metric] += 1
                continue
            metric_pairs[metric].append((baseline_score, ours_score))

    if mismatches:
        preview = mismatches[:10]
        raise ValueError(
            f"shared IDs contain {len(mismatches)} question/reference mismatches; first={preview}"
        )

    metrics: dict[str, Any] = {}
    for metric, pairs in metric_pairs.items():
        if not pairs:
            continue
        baseline_mean = sum(left for left, _ in pairs) / len(pairs)
        ours_mean = sum(right for _, right in pairs) / len(pairs)
        outcomes = Counter(
            "ours_higher" if right > left else "baseline_higher" if left > right else "tied"
            for left, right in pairs
        )
        metrics[metric] = {
            "paired_rows": len(pairs),
            "baseline": baseline_mean,
            "ours": ours_mean,
            "delta": ours_mean - baseline_mean,
            "missing_pairs": missing_values[metric],
            "paired_outcomes": dict(outcomes),
        }

    return {
        "schema_version": 1,
        "task": args.task,
        "pairing_policy": "exact shared sample_id with identical normalized question and reference",
        "baseline": baseline_stats,
        "ours": ours_stats,
        "shared_task_rows": len(shared_ids),
        "baseline_only_task_rows": len(set(baseline) - set(ours)),
        "ours_only_task_rows": len(set(ours) - set(baseline)),
        "metrics": metrics,
        "unavailable_requested_metrics": unavailable,
    }


def main() -> int:
    args = parse_args()
    report = aggregate(args)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
