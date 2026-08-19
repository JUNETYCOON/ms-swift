#!/usr/bin/env python3
"""Compute per-step point accuracy from ms-swift batch traces.

The batch_trace callback records each optimizer step's micro-loss and the
samples used by every rank. This script joins those records with GT point data,
applies the point-accuracy formula, and writes per-step CSV + a plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from point_accuracy_formula import (
    normalize_point,
    parse_prediction_points,
    point_distance,
    threshold_key,
)


def iter_jsonl(paths):
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"[warn] partial JSONL line: {path}:{line_no}", file=sys.stderr)


def assistant_text(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    parts = []
    for item in messages:
        if isinstance(item, dict) and item.get("role") == "assistant":
            parts.append(str(item.get("content", "")))
    return "\n".join(parts)


def gt_points(record: dict[str, Any]) -> list[list[float]]:
    objects = record.get("objects")
    if not isinstance(objects, dict):
        return []
    bbox = objects.get("bbox") or []
    if not isinstance(bbox, list):
        return []
    points = []
    for value in bbox:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                points.append([float(value[0]), float(value[1])])
            except (TypeError, ValueError):
                continue
    return points


def replace_bbox_placeholders(text: str, points: list[list[float]]) -> str:
    output = text
    index = 0
    while "<bbox>" in output and index < len(points):
        point = points[index]
        output = output.replace("<bbox>", f"({point[0]:g},{point[1]:g})", 1)
        index += 1
    return output


def prediction_source_and_text(sample: dict[str, Any], record: dict[str, Any]) -> tuple[str, str]:
    prediction = (
        sample.get("prediction_text")
        or sample.get("model_output")
        or sample.get("output_text")
        or ""
    )
    if prediction:
        return str(prediction), "model_output"
    text = sample.get("target_assistant_text") or assistant_text(record)
    points = sample.get("gt_points") or gt_points(record)
    return replace_bbox_placeholders(text, points), "target_reconstruction"


def sample_scores(
    prediction_text: str,
    gt_raw: list[list[float]],
    prediction_space: str,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    gt = [normalize_point(point, prediction_space) for point in gt_raw]
    pred = [normalize_point(point, prediction_space) for point in parse_prediction_points(prediction_text)]
    best = [1.0] * len(gt)
    used: set[int] = set()
    for gt_index, target in enumerate(gt):
        best_index = None
        best_distance = 1e9
        for pred_index, prediction in enumerate(pred):
            if pred_index in used:
                continue
            distance = point_distance(prediction, target)
            if distance < best_distance:
                best_distance = distance
                best_index = pred_index
        if best_index is not None:
            used.add(best_index)
            best[gt_index] = min(best[gt_index], best_distance)
    hit_counts = {threshold_key(value): sum(1 for distance in best if distance <= value) for value in thresholds}
    return {
        "gt_point_count": len(gt),
        "pred_point_count": len(pred),
        "mean_point_distance": sum(best) / len(best) if best else None,
        "hit_counts": hit_counts,
    }


def read_source_row(source_path: str, line_no: int, cache: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    key = (source_path, int(line_no))
    if key in cache:
        return cache[key]
    row: dict[str, Any] = {}
    try:
        with Path(source_path).open(encoding="utf-8", errors="replace") as handle:
            for current, line in enumerate(handle, 1):
                if current == int(line_no):
                    row = json.loads(line)
                    break
    except Exception as error:  # noqa: BLE001
        print(f"[warn] failed to read {source_path}:{line_no}: {error}", file=sys.stderr)
    cache[key] = row
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-space", default="norm1000")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.02, 0.05, 0.10])
    args = parser.parse_args()

    thresholds = tuple(sorted(args.thresholds))
    keys = [threshold_key(value) for value in thresholds]
    trace_paths = sorted(args.trace_dir.glob("batch_trace_rank*.jsonl"))
    if not trace_paths:
        trace_paths = sorted(args.trace_dir.glob("batch_trace*.jsonl"))
    if not trace_paths:
        print(f"[error] no batch_trace*.jsonl under {args.trace_dir}", file=sys.stderr)
        return 2

    rows = []
    for event in iter_jsonl(trace_paths):
        if event.get("event") not in ("step_loss", "batch_end", "micro_batch_end"):
            continue
        samples = event.get("samples") or []
        if not isinstance(samples, list):
            continue
        rows.append(
            {
                "trace_id": event.get("trace_id") or "",
                "optimizer_step": int(event.get("optimizer_step", event.get("global_step", 0))),
                "micro_step": int(event.get("micro_step_in_optimizer_step", 0)),
                "rank": int(event.get("rank", 0)),
                "micro_loss": event.get("micro_loss") or event.get("loss"),
                "samples": samples,
            }
        )
    rows.sort(key=lambda row: (row["optimizer_step"], row["rank"], row["micro_step"]))
    if not rows:
        print(f"[error] no trace events with samples under {args.trace_dir}", file=sys.stderr)
        return 2

    source_cache: dict[tuple[str, int], dict[str, Any]] = {}
    sample_rows: list[dict[str, Any]] = []
    step_agg: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "loss_sum": 0.0,
            "loss_max": float("-inf"),
            "loss_count": 0,
            "sample_count": 0,
            "gt_total": 0,
            "pred_total": 0,
            "distance_sum": 0.0,
            "distance_count": 0,
            "hits": {key: 0 for key in keys},
            "totals": {key: 0 for key in keys},
        }
    )

    for row in rows:
        step = row["optimizer_step"]
        loss = row["micro_loss"]
        if loss is not None:
            try:
                loss = float(loss)
            except (TypeError, ValueError):
                loss = None
        if loss is not None and math.isfinite(loss):
            agg = step_agg[step]
            agg["loss_sum"] += loss
            agg["loss_max"] = max(agg["loss_max"], loss)
            agg["loss_count"] += 1
        for sample in row["samples"]:
            if not isinstance(sample, dict):
                continue
            source = sample.get("source_jsonl") or ""
            line_no = sample.get("source_line_no")
            record = {}
            if source and line_no is not None:
                record = read_source_row(str(source), int(line_no), source_cache)
            prediction_text, prediction_source = prediction_source_and_text(sample, record)
            gt = sample.get("gt_points") or gt_points(record)
            scores = sample_scores(prediction_text, gt, args.prediction_space, thresholds)
            agg = step_agg[step]
            agg["sample_count"] += 1
            agg["gt_total"] += scores["gt_point_count"]
            agg["pred_total"] += scores["pred_point_count"]
            if scores["mean_point_distance"] is not None:
                agg["distance_sum"] += scores["mean_point_distance"] * scores["gt_point_count"]
                agg["distance_count"] += scores["gt_point_count"]
            for key in keys:
                agg["totals"][key] += scores["gt_point_count"]
                agg["hits"][key] += scores["hit_counts"][key]
            sample_rows.append(
                {
                    "step": step,
                    "rank": row["rank"],
                    "micro_step": row["micro_step"],
                    "trace_id": row["trace_id"],
                    "micro_loss": loss,
                    "source_jsonl": source,
                    "source_line_no": line_no,
                    "sample_id": sample.get("sample_id") or "",
                    "media_head": sample.get("media_head") or "",
                    "prediction_source": prediction_source,
                    "target_text": sample.get("target_text") or "",
                    "model_output": prediction_text if prediction_source == "model_output" else "",
                    "gt_point_count": scores["gt_point_count"],
                    "pred_point_count": scores["pred_point_count"],
                    "mean_point_distance": scores["mean_point_distance"],
                    **{f"point_acc_{key}": scores["hit_counts"][key] / scores["gt_point_count"]
                       if scores["gt_point_count"] else 0.0 for key in keys},
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = args.output_dir / "point_accuracy_samples.csv"
    with sample_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "step",
            "rank",
            "micro_step",
            "trace_id",
            "micro_loss",
            "source_jsonl",
            "source_line_no",
            "sample_id",
            "media_head",
            "prediction_source",
            "target_text",
            "model_output",
            "gt_point_count",
            "pred_point_count",
            "mean_point_distance",
            *[f"point_acc_{key}" for key in keys],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    step_csv = args.output_dir / "point_accuracy_by_step.csv"
    with step_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "step",
            "micro_loss_mean",
            "micro_loss_max",
            "loss_count",
            "sample_count",
            "gt_point_count",
            "pred_point_count",
            "mean_point_distance",
            *[f"point_acc_{key}" for key in keys],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(step_agg):
            agg = step_agg[step]
            writer.writerow(
                {
                    "step": step,
                    "micro_loss_mean": agg["loss_sum"] / agg["loss_count"] if agg["loss_count"] else "",
                    "micro_loss_max": agg["loss_max"] if agg["loss_max"] != float("-inf") else "",
                    "loss_count": agg["loss_count"],
                    "sample_count": agg["sample_count"],
                    "gt_point_count": agg["gt_total"],
                    "pred_point_count": agg["pred_total"],
                    "mean_point_distance": agg["distance_sum"] / agg["distance_count"] if agg["distance_count"] else "",
                    **{f"point_acc_{key}": agg["hits"][key] / agg["totals"][key] if agg["totals"][key] else 0.0
                       for key in keys},
                }
            )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = sorted(step_agg)
        if steps:
            fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            losses = [
                step_agg[step]["loss_sum"] / step_agg[step]["loss_count"] if step_agg[step]["loss_count"] else None
                for step in steps
            ]
            axes[0].plot(
                [step for step, value in zip(steps, losses) if value is not None],
                [value for value in losses if value is not None],
                marker="o",
                label="micro_loss_mean",
            )
            axes[0].set_ylabel("Micro loss")
            axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            axes[0].legend()
            colors = ["#4C78A8", "#F58518", "#54A24B"]
            for key, color in zip(keys, colors):
                values = [
                    step_agg[step]["hits"][key] / step_agg[step]["totals"][key] * 100
                    if step_agg[step]["totals"][key] else 0.0
                    for step in steps
                ]
                axes[1].plot(steps, values, marker="o", label=f"point_acc@{float(key.replace('_', '.')):.2f}", color=color)
            axes[1].set_ylabel("Point accuracy (%)")
            axes[1].set_ylim(0, 100)
            axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            axes[1].legend()
            axes[2].plot(steps, [step_agg[step]["sample_count"] for step in steps], marker="s", color="#72B7B2")
            axes[2].set_xlabel("Training step")
            axes[2].set_ylabel("Samples")
            axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            fig.suptitle("Per-step batch trace and point accuracy")
            fig.tight_layout()
            plot_path = args.output_dir / "point_accuracy_vs_step.png"
            fig.savefig(plot_path, dpi=180)
            plt.close(fig)
    except Exception as error:  # noqa: BLE001
        print(f"[warn] plot failed: {error}", file=sys.stderr)

    print(
        json.dumps(
            {
                "steps": len(step_agg),
                "samples": len(sample_rows),
                "sample_csv": str(sample_csv),
                "step_csv": str(step_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
