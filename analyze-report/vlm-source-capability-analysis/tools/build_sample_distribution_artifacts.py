#!/usr/bin/env python3
"""Build deterministic sample-distribution evidence from archived source manifests."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUB = ROOT / "sub-dataset"
ANALYSIS_DATE = "2026-08-06"

DATASETS = [
    "COCO",
    "VQAv2",
    "VisualGenome",
    "GQA",
    "TextVQA",
    "ChartQA",
    "AI2D",
    "LLaVA-Instruct",
    "VLM-R1",
    "Robo2VLM",
    "RoboVQA",
    "SpatialVLM",
    "PixMo-Cap",
    "PixMo-Points",
    "Molmo2-VideoCapQA",
    "Molmo2-VideoPoint",
    "Molmo2-VideoSubtitleQA",
    "Molmo2-VideoTrack",
]

MEDIA_TYPES = {
    "COCO": "image",
    "VQAv2": "image",
    "VisualGenome": "image",
    "GQA": "image",
    "TextVQA": "image",
    "ChartQA": "image",
    "AI2D": "image",
    "LLaVA-Instruct": "image",
    "VLM-R1": "image",
    "Robo2VLM": "robot_image",
    "RoboVQA": "robot_video",
    "SpatialVLM": "image",
    "PixMo-Cap": "remote_image",
    "PixMo-Points": "remote_image",
    "Molmo2-VideoCapQA": "video_id_without_media",
    "Molmo2-VideoPoint": "video",
    "Molmo2-VideoSubtitleQA": "video_id_without_media",
    "Molmo2-VideoTrack": "video_lineage_without_complete_media",
}

INPUT_KEYS = (
    "question",
    "user_prompt",
    "first_user",
    "input_preview",
    "first_question",
)
OUTPUT_KEYS = (
    "answer_preview",
    "assistant_raw",
    "first_assistant",
    "output_preview",
    "first_answer",
    "first_answer_text",
    "first_region_phrase",
    "label_names",
    "parsed_output",
)

LENGTH_BINS = (
    ("0", 0, 0),
    ("1-16", 1, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
    ("513+", 513, math.inf),
)

DENSITY_BINS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65+", 65, math.inf),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def manifest_rows(dataset: str) -> tuple[list[dict[str, Any]], str]:
    directory = SUB / dataset
    visual = directory / "sampling-manifest.jsonl"
    rows = load_jsonl(visual)
    if rows:
        return rows, visual.relative_to(ROOT).as_posix()
    records = directory / "record-sampling-manifest.jsonl"
    return load_jsonl(records), records.relative_to(ROOT).as_posix()


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return normalized_text(value)
    return ""


def source_split_values(row: dict[str, Any]) -> list[str]:
    value = row.get("task_splits") or row.get("split") or "unspecified"
    if isinstance(value, list):
        return sorted({normalized_text(item) for item in value}) or ["unspecified"]
    return [normalized_text(value)]


def flatten_labels(value: Any, prefix: str = "") -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, dict):
        output: list[str] = []
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, (dict, list)):
                output.extend(flatten_labels(item, name))
            else:
                output.append(f"{name}={normalized_text(item)}")
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(flatten_labels(item, prefix))
        return output
    return [f"{prefix}={normalized_text(value)}" if prefix else normalized_text(value)]


def task_values(dataset: str, row: dict[str, Any]) -> list[str]:
    for key in (
        "capabilities",
        "task_types",
        "category_values",
        "category",
        "collection_method",
        "first_task",
        "resolver",
    ):
        labels = flatten_labels(row.get(key), key)
        if labels:
            return sorted(set(labels))
    fallbacks = {
        "COCO": "object_multi_label_classification",
        "VisualGenome": "vqa_region_grounding",
        "LLaVA-Instruct": "multimodal_instruction",
        "VLM-R1": "referring_expression_grounding",
        "Molmo2-VideoPoint": "video_pointing",
    }
    return [fallbacks.get(dataset, "source_task_unspecified")]


def annotation_density(dataset: str, row: dict[str, Any]) -> tuple[int, str]:
    if dataset == "VisualGenome":
        return int(row.get("qa_count", 0)) + int(row.get("region_count", 0)), "QA + region rows on sampled image"
    for key, description in (
        ("question_count", "questions on sampled image"),
        ("task_count", "source tasks on sampled video"),
        ("bbox_count", "boxes in sampled grounding row"),
        ("point_annotation_count", "points in sampled source row"),
        ("conversation_turn_count", "conversation turns in sampled instruction row"),
    ):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return max(0, int(value)), description
    return 1, "one sampled supervision row (per-media total unavailable in manifest)"


def bin_name(value: int, bins: Iterable[tuple[str, int, float]]) -> str:
    for name, lower, upper in bins:
        if lower <= value <= upper:
            return name
    raise ValueError(f"no bin for {value}")


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def numeric_summary(values: list[float]) -> tuple[Any, Any, Any, Any, Any]:
    if not values:
        return "not_evaluable", "not_evaluable", "not_evaluable", "not_evaluable", "not_evaluable"
    return (
        round(min(values), 3),
        round(statistics.median(values), 3),
        round(percentile([round(value * 1000) for value in values], 0.9) / 1000, 3),
        round(max(values), 3),
        round(statistics.fmean(values), 3),
    )


def entropy_bits(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows generated for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    strata_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    length_summary_rows: list[dict[str, Any]] = []
    density_rows: list[dict[str, Any]] = []
    density_summary_rows: list[dict[str, Any]] = []
    media_summary_rows: list[dict[str, Any]] = []
    text_quality_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        rows, source_manifest = manifest_rows(dataset)
        summary = load_json(SUB / dataset / "sampling-summary.json")
        sample_total = len(rows)
        selected_count = int(summary.get("selected_count", 0))
        expected_count = selected_count or int(summary.get("record_sample_count", 0))
        if sample_total != expected_count:
            raise ValueError(f"{dataset}: sample manifest count does not match summary")

        dimensions: dict[str, Counter[str]] = {
            "split": Counter(),
            "task_or_capability": Counter(),
            "media_type": Counter({MEDIA_TYPES[dataset]: sample_total}),
        }
        input_lengths: list[int] = []
        output_lengths: list[int] = []
        input_texts: list[str] = []
        output_texts: list[str] = []
        densities: list[int] = []
        density_measures: Counter[str] = Counter()
        widths: list[float] = []
        heights: list[float] = []
        byte_lengths: list[float] = []
        durations: list[float] = []
        frame_rates: list[float] = []
        frame_counts: list[float] = []
        formats: Counter[str] = Counter()

        for row in rows:
            dimensions["split"].update(source_split_values(row))
            dimensions["task_or_capability"].update(task_values(dataset, row))
            input_text = first_text(row, INPUT_KEYS)
            output_text = first_text(row, OUTPUT_KEYS)
            input_texts.append(input_text)
            output_texts.append(output_text)
            input_lengths.append(len(input_text))
            output_lengths.append(len(output_text))
            density, measure = annotation_density(dataset, row)
            densities.append(density)
            density_measures[measure] += 1
            width = row.get("width") or row.get("preview_width")
            height = row.get("height") or row.get("preview_height")
            byte_length = row.get("byte_length") or row.get("source_video_byte_length") or row.get("video_size")
            fps = row.get("video_fps") or row.get("framerate")
            frames = row.get("video_frames_declared") or row.get("num_frames")
            duration = row.get("video_duration_seconds")
            if not isinstance(duration, (int, float)) and isinstance(frames, (int, float)) and isinstance(fps, (int, float)) and fps > 0:
                duration = frames / fps
            if isinstance(width, (int, float)) and width > 0:
                widths.append(float(width))
            if isinstance(height, (int, float)) and height > 0:
                heights.append(float(height))
            if isinstance(byte_length, (int, float)) and byte_length > 0:
                byte_lengths.append(float(byte_length))
            if isinstance(duration, (int, float)) and duration >= 0:
                durations.append(float(duration))
            if isinstance(fps, (int, float)) and fps > 0:
                frame_rates.append(float(fps))
            if isinstance(frames, (int, float)) and frames >= 0:
                frame_counts.append(float(frames))
            media_format = row.get("image_format") or row.get("video_codec")
            if media_format:
                formats[normalized_text(media_format)] += 1

        for dimension, counts in dimensions.items():
            denominator = sum(counts.values())
            for value, count in sorted(counts.items()):
                strata_rows.append({
                    "dataset": dataset,
                    "dimension": dimension,
                    "value": value,
                    "sample_count": count,
                    "dimension_total": denominator,
                    "proportion": count / denominator if denominator else 0,
                    "evidence_type": "deterministic archived source sample",
                    "source_manifest": source_manifest,
                    "seed": summary.get("seed"),
                    "population_sampling_method": summary.get("sampling_method") or "fixed-seed source permutation; see summary",
                    "analysis_date": ANALYSIS_DATE,
                })

        for side, values in (("input", input_lengths), ("output", output_lengths)):
            counts = Counter(bin_name(value, LENGTH_BINS) for value in values)
            for name, _, _ in LENGTH_BINS:
                length_rows.append({
                    "dataset": dataset,
                    "side": side,
                    "length_bin_characters": name,
                    "sample_count": counts[name],
                    "sample_total": len(values),
                    "proportion": counts[name] / len(values) if values else 0,
                    "evidence_type": "deterministic archived source sample estimate",
                    "source_manifest": source_manifest,
                    "analysis_date": ANALYSIS_DATE,
                })
            length_summary_rows.append({
                "dataset": dataset,
                "side": side,
                "sample_total": len(values),
                "mean_characters": round(statistics.fmean(values), 3) if values else 0,
                "median_characters": statistics.median(values) if values else 0,
                "p90_characters": percentile(values, 0.9),
                "max_characters": max(values, default=0),
                "evidence_type": "deterministic archived source sample estimate",
                "source_manifest": source_manifest,
                "analysis_date": ANALYSIS_DATE,
            })

        density_counts = Counter(bin_name(value, DENSITY_BINS) for value in densities)
        for name, _, _ in DENSITY_BINS:
            density_rows.append({
                "dataset": dataset,
                "annotation_count_bin": name,
                "sample_count": density_counts[name],
                "sample_total": len(densities),
                "proportion": density_counts[name] / len(densities) if densities else 0,
                "measure_notes": "; ".join(f"{key}: {value}" for key, value in density_measures.items()),
                "evidence_type": "deterministic archived source sample estimate",
                "source_manifest": source_manifest,
                "analysis_date": ANALYSIS_DATE,
            })
        density_summary_rows.append({
            "dataset": dataset,
            "sample_total": len(densities),
            "mean_annotations": round(statistics.fmean(densities), 3) if densities else 0,
            "median_annotations": statistics.median(densities) if densities else 0,
            "p90_annotations": percentile(densities, 0.9),
            "max_annotations": max(densities, default=0),
            "measure_notes": "; ".join(f"{key}: {value}" for key, value in density_measures.items()),
            "evidence_type": "deterministic archived source sample estimate",
            "source_manifest": source_manifest,
            "analysis_date": ANALYSIS_DATE,
        })

        width_stats = numeric_summary(widths)
        height_stats = numeric_summary(heights)
        byte_stats = numeric_summary(byte_lengths)
        duration_stats = numeric_summary(durations)
        fps_stats = numeric_summary(frame_rates)
        frame_stats = numeric_summary(frame_counts)
        media_summary_rows.append({
            "dataset": dataset,
            "sample_total": sample_total,
            "decoded_or_inspected_media": max(len(widths), len(durations), len(frame_counts)),
            "media_status": "not_evaluable: source media unavailable" if not widths and not durations and not frame_counts else "sample media inspected",
            "format_counts": "; ".join(f"{key}:{value}" for key, value in sorted(formats.items())) or "not_evaluable",
            "width_min": width_stats[0], "width_median": width_stats[1], "width_p90": width_stats[2], "width_max": width_stats[3],
            "height_min": height_stats[0], "height_median": height_stats[1], "height_p90": height_stats[2], "height_max": height_stats[3],
            "bytes_median": byte_stats[1], "bytes_p90": byte_stats[2],
            "duration_seconds_median": duration_stats[1], "duration_seconds_p90": duration_stats[2],
            "fps_median": fps_stats[1], "fps_p90": fps_stats[2],
            "frames_median": frame_stats[1], "frames_p90": frame_stats[2],
            "evidence_type": "deterministic archived source sample estimate",
            "source_manifest": source_manifest,
            "analysis_date": ANALYSIS_DATE,
        })
        text_quality_rows.append({
            "dataset": dataset,
            "sample_total": sample_total,
            "empty_input_count": sum(not value for value in input_texts),
            "empty_output_count": sum(not value for value in output_texts),
            "unique_input_count": len(set(input_texts)),
            "unique_output_count": len(set(output_texts)),
            "unique_input_rate": len(set(input_texts)) / sample_total if sample_total else 0,
            "unique_output_rate": len(set(output_texts)) / sample_total if sample_total else 0,
            "output_entropy_bits": round(entropy_bits(output_texts), 6),
            "evidence_type": "deterministic archived source sample estimate",
            "source_manifest": source_manifest,
            "analysis_date": ANALYSIS_DATE,
        })

    write_csv(ROOT / "sampling-strata.csv", strata_rows)
    write_csv(ROOT / "sample-length-distribution.csv", length_rows)
    write_csv(ROOT / "sample-length-summary.csv", length_summary_rows)
    write_csv(ROOT / "sample-annotation-density.csv", density_rows)
    write_csv(ROOT / "sample-annotation-density-summary.csv", density_summary_rows)
    write_csv(ROOT / "sample-media-summary.csv", media_summary_rows)
    write_csv(ROOT / "sample-text-quality-summary.csv", text_quality_rows)
    print(json.dumps({
        "status": "generated",
        "datasets": len(DATASETS),
        "samples": sum(row["sample_total"] for row in length_summary_rows if row["side"] == "input"),
        "files": [
            "sampling-strata.csv",
            "sample-length-distribution.csv",
            "sample-length-summary.csv",
            "sample-annotation-density.csv",
            "sample-annotation-density-summary.csv",
            "sample-media-summary.csv",
            "sample-text-quality-summary.csv",
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
