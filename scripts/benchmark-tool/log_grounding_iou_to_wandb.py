#!/usr/bin/env python3
"""Log grounding IoU reports to Weights & Biases.

This script consumes the JSON report produced by scripts/evaluate_grounding_iou.py
and logs scalar metrics with stable W&B names, so checkpoint-level grounding
quality can be shown as curves next to ms-swift training loss.

Example:
  python3 scripts/log_grounding_iou_to_wandb.py \
    --report /mnt/luojunkun/stage1/eval_results/ckpt-900/iou_report.json \
    --project ms-swift \
    --run-id 2y0wq9v1 \
    --resume allow \
    --step 900
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def flatten_grounding_report(report: dict[str, Any], prefix: str) -> dict[str, float | int]:
    metrics = report.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise ValueError("report['metrics'] must be a dict")

    logs: dict[str, float | int] = {}

    top_level_keys = [
        "rows",
        "filtered_non_box_output_rows",
        "rows_with_gt_boxes",
        "rows_without_gt_boxes",
        "rows_with_gt_points",
        "scored_rows",
        "coordinate_error_rows",
        "parse_success_rows",
        "parse_failure_rows",
        "parse_success_rate",
        "single_target_rows",
        "gt_boxes_total",
        "gt_boxes_scored",
        "pred_boxes_scored",
        "gt_points_total",
        "pred_points_total",
        "invalid_pred_boxes",
        "clipped_boxes",
        "reordered_boxes",
        "mean_iou_per_gt_box",
    ]
    for key in top_level_keys:
        value = _number(metrics.get(key))
        if value is not None:
            logs[f"{prefix}/{key}"] = value

    rows_with_gt = _number(metrics.get("rows_with_gt_boxes")) or 0
    parse_failure = _number(metrics.get("parse_failure_rows")) or 0
    coordinate_errors = _number(metrics.get("coordinate_error_rows")) or 0
    if rows_with_gt:
        logs[f"{prefix}/parse_failure_rate"] = parse_failure / rows_with_gt
        logs[f"{prefix}/coordinate_error_rate"] = coordinate_errors / rows_with_gt

    thresholds = metrics.get("thresholds") or {}
    if isinstance(thresholds, dict):
        for threshold, values in sorted(thresholds.items()):
            if not isinstance(values, dict):
                continue
            for key in [
                "true_positive_boxes",
                "false_positive_boxes",
                "false_negative_boxes",
                "box_precision",
                "box_recall",
                "box_f1",
                "all_targets_accuracy",
                "exact_set_accuracy",
                "complete_iou_accuracy",
                "c_iou",
                "single_target_accuracy",
            ]:
                value = _number(values.get(key))
                if value is not None:
                    logs[f"{prefix}/{key}@{threshold}"] = value

    # Optionally expose per-group metrics such as dataset_name or task subtype.
    groups = report.get("groups") or {}
    if isinstance(groups, dict):
        for group_field, group_values in groups.items():
            if not isinstance(group_values, dict):
                continue
            for group_name, group_metrics in group_values.items():
                if not isinstance(group_metrics, dict):
                    continue
                safe_group = str(group_name).replace("/", "_").replace(" ", "_")[:120]
                group_prefix = f"{prefix}/group/{group_field}/{safe_group}"
                miou = _number(group_metrics.get("mean_iou_per_gt_box"))
                parse_rate = _number(group_metrics.get("parse_success_rate"))
                scored = _number(group_metrics.get("scored_rows"))
                if miou is not None:
                    logs[f"{group_prefix}/mean_iou_per_gt_box"] = miou
                if parse_rate is not None:
                    logs[f"{group_prefix}/parse_success_rate"] = parse_rate
                if scored is not None:
                    logs[f"{group_prefix}/scored_rows"] = scored
                group_thresholds = group_metrics.get("thresholds") or {}
                if isinstance(group_thresholds, dict):
                    for threshold, values in sorted(group_thresholds.items()):
                        if not isinstance(values, dict):
                            continue
                        for key in (
                            "box_f1",
                            "single_target_accuracy",
                            "exact_set_accuracy",
                            "complete_iou_accuracy",
                            "c_iou",
                        ):
                            value = _number(values.get(key))
                            if value is not None:
                                logs[f"{group_prefix}/{key}@{threshold}"] = value
    return logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log grounding IoU report scalars to W&B.")
    parser.add_argument("--report", type=Path, required=True, help="JSON report from evaluate_grounding_iou.py")
    parser.add_argument("--prefix", default="grounding", help="Metric namespace in W&B.")
    parser.add_argument("--step", type=int, default=None, help="global_step/checkpoint step used as W&B x-axis.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "ms-swift"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--run-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", choices=("allow", "must", "never", "auto"), default="allow")
    parser.add_argument("--group", default=os.environ.get("WANDB_RUN_GROUP"))
    parser.add_argument("--job-type", default="grounding-eval")
    parser.add_argument("--dry-run", action="store_true", help="Print flattened metrics without importing wandb.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    logs = flatten_grounding_report(report, args.prefix)
    if args.step is not None:
        logs["train/global_step"] = args.step
        logs["global_step"] = args.step

    if args.dry_run:
        print(json.dumps(logs, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        name=args.run_name,
        resume=args.resume,
        group=args.group,
        job_type=args.job_type,
        config={
            "grounding_iou_report": str(args.report),
            "grounding_iou_input": report.get("input"),
            "grounding_iou_ground_truth": report.get("ground_truth"),
            "grounding_iou_args": report.get("arguments"),
        },
    )
    wandb.log(logs, step=args.step)
    if f"{args.prefix}/mean_iou_per_gt_box" in logs:
        run.summary[f"{args.prefix}/mean_iou_per_gt_box"] = logs[f"{args.prefix}/mean_iou_per_gt_box"]
    threshold_key = "0.5"
    for key in (
        f"{args.prefix}/box_f1@{threshold_key}",
        f"{args.prefix}/c_iou@{threshold_key}",
        f"{args.prefix}/complete_iou_accuracy@{threshold_key}",
        f"{args.prefix}/single_target_accuracy@{threshold_key}",
        f"{args.prefix}/exact_set_accuracy@{threshold_key}",
        f"{args.prefix}/parse_success_rate",
    ):
        if key in logs:
            run.summary[key] = logs[key]
    wandb.finish()
    print(json.dumps(logs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
