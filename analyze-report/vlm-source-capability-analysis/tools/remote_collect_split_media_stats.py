#!/usr/bin/env python3
"""Audit source split isolation for embedded-media and scene-lineage datasets."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


ROOT = Path("/mnt/luojunkun/stage1/dataset")


def split_name(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("validation"):
        return "validation"
    if name.startswith("test"):
        return "test"
    if name.startswith("val"):
        return "val"
    if name.startswith("train"):
        return "train"
    return "unspecified"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def spatialvlm() -> tuple[dict[str, set[str]], dict[str, Any]]:
    root = ROOT / "spatialvlm"
    sets: dict[str, set[str]] = defaultdict(set)
    paths: dict[str, set[str]] = defaultdict(set)
    row_counts: Counter[str] = Counter()
    missing_bytes = multi_image_rows = 0
    for path in sorted((root / "data").glob("*.parquet")):
        split = split_name(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=["images"]):
            for images in batch.column(0).to_pylist():
                row_counts[split] += 1
                if not isinstance(images, list) or not images:
                    missing_bytes += 1
                    continue
                multi_image_rows += int(len(images) != 1)
                for image in images:
                    if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
                        missing_bytes += 1
                        continue
                    sets[split].add(sha(image["bytes"]))
                    if image.get("path"):
                        paths[split].add(str(image["path"]))
        parquet.close()
    return dict(sets), {
        "row_counts": dict(row_counts),
        "unique_sha256_by_split": {key: len(value) for key, value in sets.items()},
        "unique_source_paths_by_split": {key: len(value) for key, value in paths.items()},
        "missing_image_bytes": missing_bytes,
        "rows_with_image_count_not_one": multi_image_rows,
        "source_path_intersection_train_test": len(
            paths.get("train", set()) & paths.get("test", set())
        ),
    }


def chartqa() -> tuple[dict[str, set[str]], dict[str, Any]]:
    root = ROOT / "Chartqa"
    sets: dict[str, set[str]] = defaultdict(set)
    row_counts: Counter[str] = Counter()
    missing_bytes = 0
    for path in sorted((root / "data").glob("*.parquet")):
        split = split_name(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=["image"]):
            image_bytes = pc.struct_field(batch.column(0), "bytes").to_pylist()
            for payload in image_bytes:
                row_counts[split] += 1
                if isinstance(payload, bytes):
                    sets[split].add(sha(payload))
                else:
                    missing_bytes += 1
        parquet.close()
    return dict(sets), {
        "row_counts": dict(row_counts),
        "unique_sha256_by_split": {key: len(value) for key, value in sets.items()},
        "missing_image_bytes": missing_bytes,
    }


def robo2vlm() -> tuple[dict[str, set[str]], dict[str, Any]]:
    root = ROOT / "robo2vlm"
    sets: dict[str, set[str]] = defaultdict(set)
    raw_ids: dict[str, set[str]] = defaultdict(set)
    row_counts: Counter[str] = Counter()
    missing_ids = 0
    for path in sorted((root / "data").glob("*.parquet")):
        split = split_name(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65536, columns=["id"]):
            for value in batch.column(0).to_pylist():
                row_counts[split] += 1
                if value is None or not str(value):
                    missing_ids += 1
                    continue
                value = str(value)
                raw_ids[split].add(value)
                sets[split].add(re.sub(r"_q\d+$", "", value))
        parquet.close()
    return dict(sets), {
        "canonical_rule": "direct data/*.parquet only; nested data/data copies excluded",
        "row_counts": dict(row_counts),
        "unique_raw_ids_by_split": {key: len(value) for key, value in raw_ids.items()},
        "unique_scene_lineages_by_split": {key: len(value) for key, value in sets.items()},
        "missing_ids": missing_ids,
    }


def coco() -> tuple[dict[str, set[str]], dict[str, Any]]:
    root = ROOT / "COCO" / "COCO-MODELSCOPE"
    label_names = [
        line.strip()
        for line in (root / "labels.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sets: dict[str, set[str]] = defaultdict(set)
    row_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    label_entries = empty_label_rows = invalid_label_ids = 0
    missing_or_non_single_images = 0
    schemas: dict[str, str] = {}
    for split in ("train", "test"):
        for path in sorted((root / split).glob("*.arrow")):
            with path.open("rb") as stream:
                reader = ipc.RecordBatchStreamReader(stream)
                schemas[split] = str(reader.schema)
                for batch in reader:
                    row_counts[split] += batch.num_rows
                    images = batch.column(batch.schema.get_field_index("images"))
                    lengths = pc.list_value_length(images).to_pylist()
                    missing_or_non_single_images += sum(value != 1 for value in lengths)
                    flattened = pc.list_flatten(images)
                    payloads = pc.struct_field(flattened, "bytes").to_pylist()
                    for payload in payloads:
                        if isinstance(payload, bytes):
                            sets[split].add(sha(payload))
                        else:
                            missing_or_non_single_images += 1
                    if "labels" in batch.schema.names:
                        labels = batch.column(batch.schema.get_field_index("labels"))
                        label_lengths = pc.list_value_length(labels).to_pylist()
                        empty_label_rows += sum(value == 0 for value in label_lengths)
                        values = pc.list_flatten(labels).to_pylist()
                        label_entries += len(values)
                        for value in values:
                            if isinstance(value, int) and 0 <= value < len(label_names):
                                label_counts[value] += 1
                            else:
                                invalid_label_ids += 1
    return dict(sets), {
        "row_counts": dict(row_counts),
        "unique_sha256_by_split": {key: len(value) for key, value in sets.items()},
        "label_name_count": len(label_names),
        "label_annotation_entries": label_entries,
        "empty_train_label_rows": empty_label_rows,
        "invalid_label_ids": invalid_label_ids,
        "missing_or_non_single_images": missing_or_non_single_images,
        "label_counts": {
            label_names[index]: count for index, count in label_counts.most_common()
        },
        "source_schema_by_split": schemas,
        "semantic_boundary": {
            "train": "image bytes -> list<int64> category IDs",
            "test": "image bytes only; no labels field",
            "not_present": ["caption", "question", "answer", "bbox", "segmentation"],
        },
    }


def pair_rows(dataset: str, sets: dict[str, set[str]], namespace: str) -> list[dict[str, Any]]:
    output = []
    for left, right in itertools.combinations(sorted(sets), 2):
        intersection = len(sets[left] & sets[right])
        output.append(
            {
                "dataset": dataset,
                "left_split": left,
                "right_split": right,
                "left_unique_media": len(sets[left]),
                "right_unique_media": len(sets[right]),
                "intersection": intersection,
                "namespace": namespace,
                "status": "overlap" if intersection else "isolated",
                "evidence_type": "full source population",
            }
        )
    return output


def main() -> int:
    spatial_sets, spatial_details = spatialvlm()
    chart_sets, chart_details = chartqa()
    robo_sets, robo_details = robo2vlm()
    coco_sets, coco_details = coco()
    result = {
        "source_root": str(ROOT),
        "source_only_contract": True,
        "details": {
            "SpatialVLM": spatial_details,
            "ChartQA": chart_details,
            "Robo2VLM": robo_details,
            "COCO": coco_details,
        },
        "split_media_audit": (
            pair_rows("SpatialVLM", spatial_sets, "source image SHA256")
            + pair_rows("ChartQA", chart_sets, "source image SHA256")
            + pair_rows("Robo2VLM", robo_sets, "scene lineage id (_qN removed)")
            + pair_rows("COCO", coco_sets, "source image SHA256")
        ),
    }
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
