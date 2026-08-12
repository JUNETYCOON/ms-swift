#!/usr/bin/env python3
"""Select deterministic representative and targeted paired benchmark rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable


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


def paired_outcome(row: dict, baseline_field: str, ours_field: str) -> str:
    baseline = binary(row.get(baseline_field), baseline_field)
    ours = binary(row.get(ours_field), ours_field)
    return {
        (1, 1): "both_correct",
        (0, 1): "ours_only",
        (1, 0): "baseline_only",
        (0, 0): "both_wrong",
    }[(baseline, ours)]


def stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode("utf-8")).hexdigest()


def stratified_select(
    rows: list[dict],
    target: int,
    stratum: Callable[[dict], str],
    sample_id: Callable[[dict], str],
    seed: int,
    namespace: str,
) -> list[dict]:
    if target <= 0:
        return []
    if target > len(rows):
        raise ValueError(f"cannot select {target} rows from {len(rows)}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    total = len(rows)
    exact = {name: target * len(values) / total for name, values in groups.items()}
    allocation = {name: math.floor(value) for name, value in exact.items()}
    remaining = target - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda name: (
            -(exact[name] - allocation[name]),
            stable_key(seed, f"{namespace}:quota", name),
        ),
    )
    for name in order[:remaining]:
        allocation[name] += 1
    selected = []
    for name in sorted(groups):
        ordered = sorted(
            groups[name],
            key=lambda row: stable_key(seed, f"{namespace}:{name}", sample_id(row)),
        )
        selected.extend(ordered[:allocation[name]])
    return sorted(selected, key=lambda row: stable_key(seed, f"{namespace}:final", sample_id(row)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-id-field", default="sample_id")
    parser.add_argument("--category-field", default="category")
    parser.add_argument("--baseline-field", default="baseline_correct")
    parser.add_argument("--ours-field", default="ours_correct")
    parser.add_argument("--representative", type=int, default=50)
    parser.add_argument("--targeted", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--allow-short", action="store_true")
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    sample_id = lambda row: str(row.get(args.sample_id_field))
    ids = [sample_id(row) for row in rows]
    if any(value in {"", "None"} for value in ids) or len(set(ids)) != len(ids):
        raise SystemExit("sample IDs must be present and unique")
    requested = args.representative + args.targeted
    if len(rows) < requested and not args.allow_short:
        raise SystemExit(f"need {requested} unique rows, found {len(rows)}; use --allow-short to keep all available rows")
    representative_target = min(args.representative, len(rows))
    targeted_target = min(args.targeted, max(0, len(rows) - representative_target))
    get_outcome = lambda row: paired_outcome(row, args.baseline_field, args.ours_field)
    get_category = lambda row: str(row.get(args.category_field) or "unknown")
    representative = stratified_select(
        rows,
        representative_target,
        lambda row: f"{get_category(row)}|{get_outcome(row)}",
        sample_id,
        args.seed,
        "representative",
    )
    chosen_ids = {sample_id(row) for row in representative}
    remaining_rows = [row for row in rows if sample_id(row) not in chosen_ids]
    targeted = []
    reasons = {}
    for priority in ("baseline_only", "ours_only", "both_wrong", "both_correct"):
        needed = targeted_target - len(targeted)
        if needed <= 0:
            break
        candidates = [row for row in remaining_rows if get_outcome(row) == priority]
        take = min(needed, len(candidates))
        picked = stratified_select(
            candidates,
            take,
            get_category,
            sample_id,
            args.seed,
            f"targeted:{priority}",
        ) if take else []
        picked_ids = {sample_id(row) for row in picked}
        for row in picked:
            reasons[sample_id(row)] = priority
        targeted.extend(picked)
        remaining_rows = [row for row in remaining_rows if sample_id(row) not in picked_ids]
    output_rows = []
    for cohort, selected in (("representative", representative), ("targeted_badcase", targeted)):
        for row in selected:
            value = dict(row)
            value["selection_cohort"] = cohort
            value["selection_seed"] = args.seed
            value["paired_outcome"] = get_outcome(row)
            value["selection_stratum"] = f"{get_category(row)}|{get_outcome(row)}"
            value["selection_reason"] = (
                "proportional_category_outcome_stratum"
                if cohort == "representative"
                else f"targeted_priority:{reasons[sample_id(row)]}"
            )
            output_rows.append(value)
    for index, row in enumerate(output_rows, 1):
        row["selection_order"] = index
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "seed": args.seed,
        "population": len(rows),
        "selected": len(output_rows),
        "representative": len(representative),
        "targeted_badcase": len(targeted),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
