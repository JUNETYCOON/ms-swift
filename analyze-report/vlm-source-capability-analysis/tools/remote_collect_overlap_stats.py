#!/usr/bin/env python3
"""Compute source-only lineage and split overlap statistics."""

from __future__ import annotations

import io
import itertools
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import pyarrow.parquet as pq


ROOT = Path("/mnt/luojunkun/stage1/dataset")
CHUNK_CHARS = 4 * 1024 * 1024


def iter_json_array(binary: BinaryIO) -> Iterator[Any]:
    stream = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False

    def refill(compact: bool = True) -> None:
        nonlocal buffer, position, eof
        if compact and position:
            buffer = buffer[position:]
            position = 0
        chunk = stream.read(CHUNK_CHARS)
        if chunk:
            buffer += chunk
        else:
            eof = True

    refill(False)
    while position >= len(buffer) or buffer[position].isspace():
        if position < len(buffer):
            position += 1
        elif eof:
            raise ValueError("empty JSON")
        else:
            refill()
    if buffer[position] != "[":
        raise ValueError("expected array")
    position += 1
    while True:
        while True:
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer):
                break
            if eof:
                raise ValueError("unterminated array")
            refill()
        if buffer[position] == "]":
            return
        while True:
            try:
                value, end = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if eof:
                    raise
                refill()
        yield value
        position = end
        if position >= CHUNK_CHARS:
            refill()


def iter_path(path: Path) -> Iterator[Any]:
    with path.open("rb") as stream:
        yield from iter_json_array(stream)


def normalize_numeric_id(value: Any) -> str | None:
    if value is None:
        return None
    matches = re.findall(r"\d+", Path(str(value)).stem)
    return str(int(matches[-1])) if matches else None


def parquet_values(paths: list[Path], field: str) -> set[str]:
    values: set[str] = set()
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65536, columns=[field]):
            for value in batch.column(0).to_pylist():
                if value is not None:
                    values.add(str(value))
        parquet.close()
    return values


def parquet_split_values(
    paths: list[Path], field: str, split_for_path
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        split = split_for_path(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65536, columns=[field]):
            output[split].update(
                str(value) for value in batch.column(0).to_pylist() if value is not None
            )
        parquet.close()
    return dict(output)


def llava_sets() -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    path = ROOT / "llava-instruct" / "llava_v1_5_mix665k.json"
    for row in iter_path(path):
        if not isinstance(row, dict) or not isinstance(row.get("image"), str):
            continue
        image = row["image"]
        prefix = image.split("/", 1)[0]
        stem = Path(image).stem
        if prefix == "coco":
            normalized = normalize_numeric_id(stem)
            if normalized:
                output["coco"].add(normalized)
        elif prefix in {"gqa", "textvqa", "vg"}:
            output[prefix].add(stem)
    return dict(output)


def vlm_r1_ids() -> set[str]:
    output: set[str] = set()
    path = ROOT / "vlm-r1" / "sft_related" / "mllm_rec_json.json"
    for row in iter_path(path):
        if not isinstance(row, dict):
            continue
        for image in row.get("images", []):
            normalized = normalize_numeric_id(image)
            if normalized:
                output.add(normalized)
    return output


def visual_genome_sets() -> tuple[set[str], set[str]]:
    path = ROOT / "VisualGenome" / "image_data.json.zip"
    image_ids: set[str] = set()
    coco_ids: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".json"))
        with archive.open(member) as stream:
            for row in iter_json_array(stream):
                if not isinstance(row, dict):
                    continue
                image_id = row.get("image_id", row.get("id"))
                if image_id is not None:
                    image_ids.add(str(image_id))
                coco_id = row.get("coco_id")
                if coco_id is not None:
                    coco_ids.add(str(coco_id))
    return image_ids, coco_ids


def split_pairs(dataset: str, sets: dict[str, set[str]], namespace: str) -> list[dict[str, Any]]:
    rows = []
    for left, right in itertools.combinations(sorted(sets), 2):
        intersection = len(sets[left] & sets[right])
        rows.append(
            {
                "dataset": dataset,
                "left_split": left,
                "right_split": right,
                "left_unique_media": len(sets[left]),
                "right_unique_media": len(sets[right]),
                "intersection": intersection,
                "namespace": namespace,
                "status": "overlap" if intersection else "isolated",
            }
        )
    return rows


def edge(
    left: str,
    right: str,
    left_values: set[str],
    right_values: set[str],
    namespace: str,
    relationship: str,
) -> dict[str, Any]:
    intersection = len(left_values & right_values)
    return {
        "left": left,
        "right": right,
        "left_unique": len(left_values),
        "right_unique": len(right_values),
        "intersection": intersection,
        "left_overlap_rate": intersection / len(left_values) if left_values else 0,
        "right_overlap_rate": intersection / len(right_values) if right_values else 0,
        "namespace": namespace,
        "relationship": relationship,
        "evidence_type": "full source lineage ID intersection",
    }


def main() -> int:
    llava = llava_sets()
    vlm = vlm_r1_ids()
    vg_images, vg_coco = visual_genome_sets()

    vqav2_paths = sorted((ROOT / "VQAv2" / "data").glob("*.parquet"))
    vqav2_splits = parquet_split_values(
        vqav2_paths, "image_id", lambda path: path.name.split("-", 1)[0]
    )
    vqav2_splits = {
        split: {value for item in values if (value := normalize_numeric_id(item))}
        for split, values in vqav2_splits.items()
    }
    vqav2_all = set().union(*vqav2_splits.values())

    gqa_paths = sorted((ROOT / "GQA").glob("*_all_instructions/*.parquet"))
    gqa_splits = parquet_split_values(
        gqa_paths,
        "imageId",
        lambda path: path.parent.name.replace("_all_instructions", ""),
    )
    gqa_all = set().union(*gqa_splits.values())

    textvqa_paths = sorted((ROOT / "textvqa" / "data").glob("*.parquet"))
    textvqa_splits = parquet_split_values(
        textvqa_paths, "image_id", lambda path: path.name.split("-", 1)[0]
    )
    textvqa_all = set().union(*textvqa_splits.values())

    capqa = parquet_values(
        sorted((ROOT / "Molmo2-VideoCapQA" / "data").glob("*.parquet")),
        "video_id",
    )
    subtitle = parquet_values(
        sorted((ROOT / "Molmo2-VideoSubtitleQA" / "data").glob("*.parquet")),
        "video_id",
    )
    point = parquet_values(
        sorted((ROOT / "Molmo2-VideoPoint" / "data").glob("train-*.parquet")),
        "video_id",
    )

    edges = [
        edge("LLaVA-Instruct:coco", "VQAv2", llava.get("coco", set()), vqav2_all, "COCO image_id", "derived source reuse"),
        edge("VLM-R1", "VQAv2", vlm, vqav2_all, "COCO image_id", "shared canonical media family"),
        edge("VisualGenome:coco_lineage", "VQAv2", vg_coco, vqav2_all, "COCO image_id", "cross-release lineage"),
        edge("LLaVA-Instruct:coco", "VLM-R1", llava.get("coco", set()), vlm, "COCO image_id", "shared canonical media family"),
        edge("LLaVA-Instruct:coco", "VisualGenome:coco_lineage", llava.get("coco", set()), vg_coco, "COCO image_id", "cross-release lineage"),
        edge("VLM-R1", "VisualGenome:coco_lineage", vlm, vg_coco, "COCO image_id", "cross-release lineage"),
        edge("LLaVA-Instruct:gqa", "GQA", llava.get("gqa", set()), gqa_all, "GQA/VisualGenome image_id", "derived source reuse"),
        edge("LLaVA-Instruct:textvqa", "TextVQA", llava.get("textvqa", set()), textvqa_all, "TextVQA image_id", "derived source reuse"),
        edge("LLaVA-Instruct:vg", "VisualGenome", llava.get("vg", set()), vg_images, "VisualGenome image_id", "derived source reuse"),
        edge("Molmo2-VideoCapQA", "Molmo2-VideoSubtitleQA", capqa, subtitle, "Molmo2 video_id", "shared video corpus"),
        edge("Molmo2-VideoCapQA", "Molmo2-VideoPoint", capqa, point, "Molmo2 video_id", "shared video corpus"),
        edge("Molmo2-VideoSubtitleQA", "Molmo2-VideoPoint", subtitle, point, "Molmo2 video_id", "shared video corpus"),
    ]

    split_audit = []
    split_audit.extend(split_pairs("VQAv2", vqav2_splits, "COCO image_id"))
    split_audit.extend(split_pairs("GQA", gqa_splits, "GQA image_id"))
    split_audit.extend(split_pairs("TextVQA", textvqa_splits, "TextVQA image_id"))

    result = {
        "source_root": str(ROOT),
        "source_only_contract": True,
        "entities": {
            "LLaVA-Instruct:coco": len(llava.get("coco", set())),
            "LLaVA-Instruct:gqa": len(llava.get("gqa", set())),
            "LLaVA-Instruct:textvqa": len(llava.get("textvqa", set())),
            "LLaVA-Instruct:vg": len(llava.get("vg", set())),
            "VLM-R1": len(vlm),
            "VisualGenome": len(vg_images),
            "VisualGenome:coco_lineage": len(vg_coco),
            "VQAv2": len(vqav2_all),
            "GQA": len(gqa_all),
            "TextVQA": len(textvqa_all),
            "Molmo2-VideoCapQA": len(capqa),
            "Molmo2-VideoSubtitleQA": len(subtitle),
            "Molmo2-VideoPoint": len(point),
        },
        "source_lineage_edges": edges,
        "split_media_audit": split_audit,
        "split_unique_media_counts": {
            "VQAv2": {key: len(value) for key, value in vqav2_splits.items()},
            "GQA": {key: len(value) for key, value in gqa_splits.items()},
            "TextVQA": {key: len(value) for key, value in textvqa_splits.items()},
        },
        "limitations": [
            "IDs are compared only inside an explicitly shared namespace.",
            "An ID intersection is lineage evidence, not a byte-level identity claim.",
            "Datasets without a shared source namespace are audited separately using archived source-media SHA256 samples.",
        ],
    }
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
