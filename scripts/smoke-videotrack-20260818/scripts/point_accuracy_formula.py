#!/usr/bin/env python3
"""Compute point accuracy from a prediction JSONL without a GPU.

Each input row should contain ground-truth points (objects.bbox or objects.points)
and a prediction field such as response, prediction, pred, or output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
POINT_PAIR_RE = re.compile(rf"[\[(]\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*[\])]")
POINT_KEY_RE = re.compile(
    rf"(?:point_2d|click_point|point)\s*[\"']?\s*[:=]\s*"
    rf"[\[(]\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*[\])]",
    re.IGNORECASE,
)
PREDICTION_FIELDS = ("response", "prediction", "pred", "output")
POINT_KEYS = ("point_2d", "click_point", "point")


def threshold_key(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def normalize_point(point: Sequence[float], prediction_space: str) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    if prediction_space in {"norm1000", "normalized1000"}:
        x, y = x / 1000.0, y / 1000.0
    elif prediction_space not in {"norm1", "normalized"}:
        raise ValueError(f"Unsupported prediction_space: {prediction_space}")
    return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))


def point_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def coerce_point(value: Any) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def extract_gt_points(record: dict[str, Any]) -> list[tuple[float, float]]:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return []
    points: list[tuple[float, float]] = []
    for key in ("bbox", "points", "point_2d", "click_point", "point"):
        value = objects.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    points.append(coerce_point(item))
    return points


def structured_points(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, str):
        return []
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            return [coerce_point(value)]
        result: list[tuple[float, float]] = []
        for item in value:
            result.extend(structured_points(item))
        return result
    if not isinstance(value, dict):
        return []
    result = []
    for key in POINT_KEYS:
        if key in value:
            result.extend(structured_points(value[key]))
    if all(isinstance(value.get(key), (int, float)) for key in ("x", "y")):
        result.append((float(value["x"]), float(value["y"])))
    for item in value.values():
        if isinstance(item, (dict, list, tuple)):
            result.extend(structured_points(item))
    return result


def parse_prediction_points(prediction: Any) -> list[tuple[float, float]]:
    if isinstance(prediction, (dict, list, tuple)):
        points = structured_points(prediction)
        if points:
            return points
    text = str(prediction or "")
    if not text:
        return []
    points = [tuple(float(value) for value in match.groups()) for match in POINT_KEY_RE.finditer(text)]
    if not points:
        points = [tuple(float(value) for value in match.groups()) for match in POINT_PAIR_RE.finditer(text)]
    return points


def match_points(
    predictions: Sequence[Sequence[float]],
    ground_truth: Sequence[Sequence[float]],
) -> list[tuple[int, int, float]]:
    if not predictions or not ground_truth:
        return []
    used: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for gt_index, target in enumerate(ground_truth):
        best_index = None
        best_distance = 1e9
        for pred_index, prediction in enumerate(predictions):
            if pred_index in used:
                continue
            distance = point_distance(prediction, target)
            if distance < best_distance:
                best_distance = distance
                best_index = pred_index
        if best_index is not None:
            used.add(best_index)
            matches.append((best_index, gt_index, best_distance))
    return matches


def sample_scores(
    record: dict[str, Any],
    prediction: Any,
    prediction_space: str,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    gt_raw = extract_gt_points(record)
    pred_raw = parse_prediction_points(prediction)
    gt_points = [normalize_point(point, prediction_space) for point in gt_raw]
    pred_points = [normalize_point(point, prediction_space) for point in pred_raw]
    matches = match_points(pred_points, gt_points)
    best = [1.0] * len(gt_points)
    for _, gt_index, distance in matches:
        best[gt_index] = min(best[gt_index], distance)
    hit_counts = {threshold_key(value): sum(1 for distance in best if distance <= value) for value in thresholds}
    gt_counts = {threshold_key(value): len(gt_points) for value in thresholds}
    return {
        "gt_point_count": len(gt_points),
        "pred_point_count": len(pred_points),
        "matched_point_count": len(matches),
        "mean_point_distance": sum(best) / len(best) if best else None,
        "point_parse_success": bool(pred_points),
        "hit_counts": hit_counts,
        "gt_counts": gt_counts,
        "accuracies": {key: hit_counts[key] / gt_counts[key] if gt_counts[key] else 0.0 for key in hit_counts},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-space", default="norm1000")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.02, 0.05, 0.10])
    args = parser.parse_args()

    thresholds = tuple(sorted(args.thresholds))
    keys = [threshold_key(value) for value in thresholds]
    totals = {key: 0 for key in keys}
    hits = {key: 0 for key in keys}
    sample_rows: list[dict[str, Any]] = []
    gt_total = 0
    pred_total = 0
    parse_success = 0
    distance_sum = 0.0
    distance_count = 0

    with args.input.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prediction = next((record.get(field) for field in PREDICTION_FIELDS if record.get(field) is not None), "")
            scores = sample_scores(record, prediction, args.prediction_space, thresholds)
            gt_total += scores["gt_point_count"]
            pred_total += scores["pred_point_count"]
            parse_success += int(scores["point_parse_success"])
            if scores["mean_point_distance"] is not None:
                distance_sum += scores["mean_point_distance"] * scores["gt_point_count"]
                distance_count += scores["gt_point_count"]
            for key in keys:
                totals[key] += scores["gt_counts"][key]
                hits[key] += scores["hit_counts"][key]
            sample_rows.append(
                {
                    "line_number": line_no,
                    "id": record.get("id") or record.get("sample_id") or "",
                    "gt_point_count": scores["gt_point_count"],
                    "pred_point_count": scores["pred_point_count"],
                    "matched_point_count": scores["matched_point_count"],
                    "mean_point_distance": scores["mean_point_distance"],
                    "point_parse_success": scores["point_parse_success"],
                    **{f"point_acc_{key}": scores["accuracies"][key] for key in keys},
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = args.output_dir / "point_accuracy_samples.csv"
    with sample_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()) if sample_rows else [])
        if sample_rows:
            writer.writeheader()
            writer.writerows(sample_rows)

    summary = {
        "rows": len(sample_rows),
        "gt_point_count": gt_total,
        "pred_point_count": pred_total,
        "parse_success_rate": parse_success / len(sample_rows) if sample_rows else 0.0,
        "mean_point_distance": distance_sum / distance_count if distance_count else None,
        **{f"point_acc_{key}": hits[key] / totals[key] if totals[key] else 0.0 for key in keys},
    }
    summary_csv = args.output_dir / "point_accuracy_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        names = [f"point_acc@{value:.2f}" for value in thresholds]
        values = [summary[f"point_acc_{key}"] * 100 for key in keys]
        bars = ax.bar(names, values, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center")
        fig.suptitle("Point accuracy from prediction JSONL")
        fig.tight_layout()
        plot_path = args.output_dir / "point_accuracy.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
    except Exception as error:  # noqa: BLE001
        print(f"plot skipped: {error}", file=sys.stderr)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"sample CSV: {sample_csv}")
    print(f"summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
