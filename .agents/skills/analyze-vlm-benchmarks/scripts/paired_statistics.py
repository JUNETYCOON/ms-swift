#!/usr/bin/env python3
"""Compute binary paired VLM accuracy statistics from JSONL rows."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{number}")
            rows.append(value)
    return rows


def binary(value: object, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip().casefold() in {"0", "1", "false", "true"}:
        return int(value.strip().casefold() in {"1", "true"})
    raise ValueError(f"{field} must be binary, got {value!r}")


def outcome(baseline: int, ours: int) -> str:
    return {
        (1, 1): "both_correct",
        (0, 1): "ours_only",
        (1, 0): "baseline_only",
        (0, 0): "both_wrong",
    }[(baseline, ours)]


def exact_mcnemar(baseline_only: int, ours_only: int) -> float:
    discordant = baseline_only + ours_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, ours_only)
    logs = [
        math.lgamma(discordant + 1) - math.lgamma(k + 1) - math.lgamma(discordant - k + 1)
        - discordant * math.log(2)
        for k in range(tail + 1)
    ]
    maximum = max(logs)
    lower = math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)
    return min(1.0, 2 * lower)


def bootstrap_ci(counts: Counter, n: int, seed: int, iterations: int) -> list[float]:
    rng = random.Random(seed)
    binomial = getattr(rng, "binomialvariate", None)

    def draw_binomial(trials: int, probability: float) -> int:
        if trials <= 0 or probability <= 0:
            return 0
        if probability >= 1:
            return trials
        if binomial is not None:
            return binomial(trials, probability)
        return sum(rng.random() < probability for _ in range(trials))

    p_ours_only = counts["ours_only"] / n
    p_baseline_only = counts["baseline_only"] / n
    conditional_baseline = p_baseline_only / (1 - p_ours_only) if p_ours_only < 1 else 0.0
    values = []
    for _ in range(iterations):
        ours_only = draw_binomial(n, p_ours_only)
        baseline_only = draw_binomial(n - ours_only, conditional_baseline)
        values.append((ours_only - baseline_only) / n)
    values.sort()

    def percentile(percent: float) -> float:
        position = (len(values) - 1) * percent
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    return [percentile(0.025), percentile(0.975)]


def category_rows(records: list[tuple[int, int, str]]) -> list[dict]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for baseline, ours, category in records:
        groups[category].append((baseline, ours))
    result = []
    for category in sorted(groups):
        pairs = groups[category]
        count = len(pairs)
        baseline_accuracy = sum(pair[0] for pair in pairs) / count
        ours_accuracy = sum(pair[1] for pair in pairs) / count
        result.append({
            "category": category,
            "num_samples": count,
            "baseline_accuracy": baseline_accuracy,
            "ours_accuracy": ours_accuracy,
            "delta": ours_accuracy - baseline_accuracy,
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-field", default="baseline_correct")
    parser.add_argument("--ours-field", default="ours_correct")
    parser.add_argument("--category-field", default="category")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    if not rows:
        raise SystemExit("input has no rows")
    records = []
    outcomes = Counter()
    for row in rows:
        baseline = binary(row.get(args.baseline_field), args.baseline_field)
        ours = binary(row.get(args.ours_field), args.ours_field)
        category = str(row.get(args.category_field) or "unknown")
        records.append((baseline, ours, category))
        outcomes[outcome(baseline, ours)] += 1
    n = len(records)
    baseline_accuracy = sum(record[0] for record in records) / n
    ours_accuracy = sum(record[1] for record in records) / n
    result = {
        "schema_version": 1,
        "num_samples": n,
        "baseline_accuracy": baseline_accuracy,
        "ours_accuracy": ours_accuracy,
        "accuracy_delta": ours_accuracy - baseline_accuracy,
        "paired_outcomes": {key: outcomes[key] for key in ("both_correct", "ours_only", "baseline_only", "both_wrong")},
        "bootstrap": {
            "method": "paired nonparametric bootstrap over empirical outcome rows",
            "seed": args.seed,
            "iterations": args.bootstrap_iterations,
            "delta_95_ci": bootstrap_ci(outcomes, n, args.seed, args.bootstrap_iterations),
        },
        "mcnemar_exact_p": exact_mcnemar(outcomes["baseline_only"], outcomes["ours_only"]),
        "categories": category_rows(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("num_samples", "baseline_accuracy", "ours_accuracy", "accuracy_delta", "mcnemar_exact_p")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
