#!/usr/bin/env python3
"""Full-population lightweight stats for legacy source Parquet datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path("/mnt/luojunkun/stage1/dataset")


def text_chars(column: Any) -> tuple[int, int]:
    lengths = pc.utf8_length(column)
    return int(pc.sum(lengths).as_py() or 0), int(
        pc.sum(pc.equal(lengths, 0)).as_py() or 0
    )


def list_length_sum(column: Any) -> int:
    return int(pc.sum(pc.list_value_length(column)).as_py() or 0)


def add_values(counter: Counter[str], column: Any) -> None:
    for value in column.to_pylist():
        counter[str(value) if value is not None else "<missing>"] += 1


def parquet_rows(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        total += parquet.metadata.num_rows
        parquet.close()
    return total


def vqav2() -> dict[str, Any]:
    root = ROOT / "VQAv2"
    paths = sorted(root.glob("data/*.parquet"))
    unique_images = set()
    unique_questions = set()
    split_counts: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    question_types: Counter[str] = Counter()
    question_chars = answer_chars = answer_entries = 0
    empty_questions = empty_answers = 0
    for path in paths:
        split = path.name.split("-", 1)[0]
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=8192,
            columns=[
                "question_type",
                "multiple_choice_answer",
                "answers",
                "image_id",
                "answer_type",
                "question_id",
                "question",
            ],
        ):
            split_counts[split] += batch.num_rows
            columns = {name: batch.column(batch.schema.get_field_index(name)) for name in batch.schema.names}
            unique_images.update(value for value in columns["image_id"].to_pylist() if value is not None)
            unique_questions.update(value for value in columns["question_id"].to_pylist() if value is not None)
            chars, empty = text_chars(columns["question"])
            question_chars += chars
            empty_questions += empty
            chars, empty = text_chars(columns["multiple_choice_answer"])
            answer_chars += chars
            empty_answers += empty
            answer_entries += list_length_sum(columns["answers"])
            add_values(answer_types, columns["answer_type"])
            add_values(question_types, columns["question_type"])
        parquet.close()
    records = sum(split_counts.values())
    return {
        "dataset": "VQAv2",
        "evidence_type": "Full-population statistic",
        "canonical_records": records,
        "unique_media_ids": len(unique_images),
        "unique_question_ids": len(unique_questions),
        "supervision_units": records,
        "answer_annotation_entries": answer_entries,
        "records_by_split": dict(sorted(split_counts.items())),
        "question_characters": question_chars,
        "answer_characters": answer_chars,
        "supervised_token_estimate_chars_div_4": round((question_chars + answer_chars) / 4),
        "empty_questions": empty_questions,
        "empty_multiple_choice_answers": empty_answers,
        "answer_type_counts": dict(answer_types.most_common()),
        "question_type_counts": dict(question_types.most_common()),
    }


def gqa() -> dict[str, Any]:
    root = ROOT / "GQA"
    canonical_paths = sorted(root.glob("*_all_instructions/*.parquet"))
    balanced_paths = sorted(root.glob("*_balanced_instructions/*.parquet"))
    image_paths = sorted(root.glob("*_all_images/*.parquet"))
    balanced_image_paths = sorted(root.glob("*_balanced_images/*.parquet"))
    unique_images_from_instructions = set()
    unique_images_from_tables = set()
    unique_questions = set()
    split_counts: Counter[str] = Counter()
    structural: Counter[str] = Counter()
    semantic: Counter[str] = Counter()
    question_chars = answer_chars = 0
    empty_questions = empty_answers = 0
    answered_records = 0
    for path in canonical_paths:
        split = path.parts[-2].replace("_all_instructions", "")
        parquet = pq.ParquetFile(path)
        available = {field.name for field in parquet.schema_arrow}
        requested = [name for name in ("id", "imageId", "question", "answer", "types") if name in available]
        for batch in parquet.iter_batches(batch_size=8192, columns=requested):
            split_counts[split] += batch.num_rows
            columns = {name: batch.column(batch.schema.get_field_index(name)) for name in batch.schema.names}
            unique_questions.update(str(value) for value in columns["id"].to_pylist() if value)
            unique_images_from_instructions.update(str(value) for value in columns["imageId"].to_pylist() if value)
            chars, empty = text_chars(columns["question"])
            question_chars += chars
            empty_questions += empty
            if "answer" in columns:
                chars, empty = text_chars(columns["answer"])
                answer_chars += chars
                empty_answers += empty
                answered_records += len(columns["answer"]) - columns["answer"].null_count
            if "types" in columns:
                types = columns["types"]
                add_values(structural, pc.struct_field(types, "structural"))
                add_values(semantic, pc.struct_field(types, "semantic"))
        parquet.close()
    for path in image_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=16384, columns=["id"]):
            unique_images_from_tables.update(
                str(value) for value in batch.column(0).to_pylist() if value
            )
        parquet.close()
    records = sum(split_counts.values())
    return {
        "dataset": "GQA",
        "evidence_type": "Full-population statistic",
        "canonical_rule": "all instruction configs only; balanced configs are duplicate subsets",
        "canonical_records": records,
        "physical_balanced_duplicate_rows": parquet_rows(balanced_paths),
        "canonical_image_table_rows": parquet_rows(image_paths),
        "balanced_image_duplicate_rows": parquet_rows(balanced_image_paths),
        "unique_media_ids_instruction_union": len(unique_images_from_instructions),
        "unique_media_ids_image_table_union": len(unique_images_from_tables),
        "unique_question_ids": len(unique_questions),
        "answered_records": answered_records,
        "records_by_split": dict(sorted(split_counts.items())),
        "question_characters": question_chars,
        "answer_characters": answer_chars,
        "supervised_token_estimate_chars_div_4": round((question_chars + answer_chars) / 4),
        "empty_questions": empty_questions,
        "empty_answers": empty_answers,
        "structural_type_counts": dict(structural.most_common()),
        "semantic_type_counts": dict(semantic.most_common()),
    }


def textvqa() -> dict[str, Any]:
    root = ROOT / "textvqa"
    paths = sorted(root.glob("data/*.parquet"))
    unique_images = set()
    unique_questions = set()
    split_counts: Counter[str] = Counter()
    set_names: Counter[str] = Counter()
    question_chars = answer_chars = 0
    answer_entries = ocr_tokens = 0
    empty_questions = 0
    for path in paths:
        split = path.name.split("-", 1)[0]
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=8192,
            columns=["image_id", "question_id", "question", "answers", "set_name", "ocr_tokens"],
        ):
            split_counts[split] += batch.num_rows
            columns = {name: batch.column(batch.schema.get_field_index(name)) for name in batch.schema.names}
            unique_images.update(str(value) for value in columns["image_id"].to_pylist() if value)
            unique_questions.update(value for value in columns["question_id"].to_pylist() if value is not None)
            chars, empty = text_chars(columns["question"])
            question_chars += chars
            empty_questions += empty
            answer_entries += list_length_sum(columns["answers"])
            answers = pc.list_flatten(columns["answers"])
            answer_chars += int(pc.sum(pc.utf8_length(answers)).as_py() or 0)
            ocr_tokens += list_length_sum(columns["ocr_tokens"])
            add_values(set_names, columns["set_name"])
        parquet.close()
    records = sum(split_counts.values())
    return {
        "dataset": "TextVQA",
        "evidence_type": "Full-population statistic",
        "canonical_records": records,
        "unique_media_ids": len(unique_images),
        "unique_question_ids": len(unique_questions),
        "supervision_units": records,
        "answer_annotation_entries": answer_entries,
        "ocr_token_entries": ocr_tokens,
        "records_by_split": dict(sorted(split_counts.items())),
        "source_set_name_counts": dict(set_names.most_common()),
        "question_characters": question_chars,
        "answer_characters": answer_chars,
        "supervised_token_estimate_chars_div_4": round((question_chars + answer_chars) / 4),
        "empty_questions": empty_questions,
    }


def image_bytes(image_column: Any) -> Any:
    return pc.struct_field(image_column, "bytes")


def chartqa() -> dict[str, Any]:
    root = ROOT / "Chartqa"
    paths = sorted(root.glob("data/*.parquet"))
    unique_hashes = set()
    split_counts: Counter[str] = Counter()
    provenance: Counter[str] = Counter()
    question_chars = answer_chars = answer_entries = 0
    empty_questions = empty_answers = 0
    for path in paths:
        split = path.name.split("-", 1)[0]
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=512, columns=["image", "query", "label", "human_or_machine"]
        ):
            split_counts[split] += batch.num_rows
            columns = {name: batch.column(batch.schema.get_field_index(name)) for name in batch.schema.names}
            for payload in image_bytes(columns["image"]).to_pylist():
                if payload:
                    unique_hashes.add(hashlib.sha256(payload).hexdigest())
            chars, empty = text_chars(columns["query"])
            question_chars += chars
            empty_questions += empty
            labels = columns["label"]
            if pa.types.is_list(labels.type) or pa.types.is_large_list(labels.type):
                answer_entries += list_length_sum(labels)
                flattened = pc.list_flatten(labels)
                answer_chars += int(pc.sum(pc.utf8_length(flattened)).as_py() or 0)
                empty_answers += int(
                    pc.sum(pc.equal(pc.list_value_length(labels), 0)).as_py() or 0
                )
            else:
                chars, empty = text_chars(labels)
                answer_chars += chars
                empty_answers += empty
                answer_entries += len(labels) - labels.null_count
            add_values(provenance, columns["human_or_machine"])
        parquet.close()
    records = sum(split_counts.values())
    return {
        "dataset": "ChartQA",
        "evidence_type": "Full-population statistic",
        "canonical_records": records,
        "unique_media_sha256": len(unique_hashes),
        "supervision_units": records,
        "answer_entries": answer_entries,
        "records_by_split": dict(sorted(split_counts.items())),
        "human_or_machine_counts": dict(provenance.most_common()),
        "question_characters": question_chars,
        "answer_characters": answer_chars,
        "supervised_token_estimate_chars_div_4": round((question_chars + answer_chars) / 4),
        "empty_questions": empty_questions,
        "empty_answers": empty_answers,
    }


def robo2vlm() -> dict[str, Any]:
    root = ROOT / "robo2vlm"
    paths = sorted(root.glob("data/*.parquet"))
    duplicate_paths = sorted(root.glob("data/data/*.parquet"))
    unique_ids = set()
    scene_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    answer_indices: Counter[str] = Counter()
    question_chars = choice_chars = 0
    empty_questions = empty_choices = 0
    suffix = re.compile(r"_q\d+$")
    for path in paths:
        split = path.name.split("-", 1)[0]
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=8192, columns=["id", "question", "choices", "correct_answer"]
        ):
            split_counts[split] += batch.num_rows
            columns = {name: batch.column(batch.schema.get_field_index(name)) for name in batch.schema.names}
            for value in columns["id"].to_pylist():
                if value:
                    identity = str(value)
                    unique_ids.add(identity)
                    scene_counts[suffix.sub("", identity)] += 1
            chars, empty = text_chars(columns["question"])
            question_chars += chars
            empty_questions += empty
            chars, empty = text_chars(columns["choices"])
            choice_chars += chars
            empty_choices += empty
            add_values(answer_indices, columns["correct_answer"])
        parquet.close()
    records = sum(split_counts.values())
    annotation_counts = sorted(scene_counts.values())
    return {
        "dataset": "Robo2VLM",
        "evidence_type": "Full-population statistic",
        "canonical_rule": "direct data/*.parquet only; nested data/data copies excluded",
        "canonical_records": records,
        "physical_nested_duplicate_rows": parquet_rows(duplicate_paths),
        "unique_record_ids": len(unique_ids),
        "unique_scene_lineages_suffix_q_removed": len(scene_counts),
        "scenes_with_multiple_questions": sum(value > 1 for value in annotation_counts),
        "annotations_per_scene_mean": sum(annotation_counts) / len(annotation_counts),
        "annotations_per_scene_median": annotation_counts[len(annotation_counts) // 2],
        "annotations_per_scene_max": max(annotation_counts),
        "supervision_units": records,
        "records_by_split": dict(sorted(split_counts.items())),
        "question_characters": question_chars,
        "choice_characters": choice_chars,
        "supervised_token_estimate_chars_div_4": round((question_chars + choice_chars) / 4),
        "empty_questions": empty_questions,
        "empty_choices": empty_choices,
        "correct_answer_value_counts": dict(answer_indices.most_common()),
    }


def main() -> int:
    payload = {
        "source_root": str(ROOT),
        "evidence_scope": "full source metadata projection; no converted data",
        "datasets": [vqav2(), gqa(), textvqa(), chartqa(), robo2vlm()],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
