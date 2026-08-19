#!/usr/bin/env python3
"""Read-only full-population statistics for source JSON and ZIP datasets.

The script is sent to the server over SSH stdin. It writes one JSON document to
stdout and never creates files beneath the source root.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterator


ROOT = Path("/mnt/luojunkun/stage1/dataset")
CHUNK_CHARS = 4 * 1024 * 1024


def iter_json_array(binary: BinaryIO) -> Iterator[Any]:
    """Stream a top-level JSON array without loading the whole file."""
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
    while True:
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position < len(buffer):
            break
        if eof:
            raise ValueError("empty JSON stream")
        refill()
    if buffer[position] != "[":
        raise ValueError("expected a top-level JSON array")
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
                raise ValueError("unterminated JSON array")
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


def iter_json_array_path(path: Path) -> Iterator[Any]:
    with path.open("rb") as stream:
        yield from iter_json_array(stream)


def zip_json_rows(path: Path) -> tuple[str, Iterator[Any]]:
    archive = zipfile.ZipFile(path)
    members = [name for name in archive.namelist() if name.lower().endswith(".json")]
    if len(members) != 1:
        archive.close()
        raise ValueError(f"expected one JSON member in {path}, found {members}")
    member = members[0]

    def rows() -> Iterator[Any]:
        try:
            with archive.open(member) as stream:
                yield from iter_json_array(stream)
        finally:
            archive.close()

    return member, rows()


def text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def compact_sample(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "<max-depth>"
    if isinstance(value, dict):
        return {
            str(key): compact_sample(item, depth + 1)
            for key, item in list(value.items())[:30]
        }
    if isinstance(value, list):
        rows = [compact_sample(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            rows.append({"omitted_items": len(value) - 12})
        return rows
    if isinstance(value, str):
        value = re.sub(r"/data/[^/]+/dataset/", "/data/<redacted>/dataset/", value)
        return value if len(value) <= 800 else value[:800] + "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def distribution(values: Counter[int]) -> dict[str, Any]:
    total = sum(values.values())
    if not total:
        return {"count": 0}
    ordered = sorted(values.items())

    def percentile(fraction: float) -> int:
        target = max(1, math.ceil(total * fraction))
        seen = 0
        for value, count in ordered:
            seen += count
            if seen >= target:
                return value
        return ordered[-1][0]

    weighted_sum = sum(value * count for value, count in ordered)
    return {
        "count": total,
        "min": ordered[0][0],
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1][0],
        "mean": weighted_sum / total,
        "histogram": {str(key): value for key, value in ordered[:200]},
        "histogram_truncated": len(ordered) > 200,
    }


def signature(value: Any) -> bytes:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def llava() -> dict[str, Any]:
    path = ROOT / "llava-instruct" / "llava_v1_5_mix665k.json"
    prefixes: Counter[str] = Counter()
    turn_counts: Counter[int] = Counter()
    role_counts: Counter[str] = Counter()
    records_per_media: Counter[str] = Counter()
    schema_variants: Counter[str] = Counter()
    unique_ids: set[str] = set()
    unique_media: set[str] = set()
    samples: list[Any] = []
    record_count = 0
    text_only = 0
    empty_messages = 0
    empty_values = 0
    image_placeholder_mismatch = 0
    user_chars = assistant_chars = other_chars = 0

    for row in iter_json_array_path(path):
        record_count += 1
        if len(samples) < 3:
            samples.append(compact_sample(row))
        if not isinstance(row, dict):
            schema_variants[json_type(row)] += 1
            continue
        schema_variants[",".join(sorted(row))] += 1
        record_id = row.get("id")
        if record_id is not None:
            unique_ids.add(str(record_id))
        image = text(row.get("image"))
        if image:
            unique_media.add(image)
            records_per_media[image] += 1
            prefixes[image.split("/", 1)[0]] += 1
        else:
            text_only += 1
            prefixes["<text-only>"] += 1
        conversations = row.get("conversations")
        if not isinstance(conversations, list):
            conversations = []
        turn_counts[len(conversations)] += 1
        placeholder_count = 0
        if not conversations:
            empty_messages += 1
        for message in conversations:
            if not isinstance(message, dict):
                role_counts[json_type(message)] += 1
                continue
            role = text(message.get("from")) or "<missing>"
            value = text(message.get("value"))
            role_counts[role] += 1
            placeholder_count += value.count("<image>")
            if not value:
                empty_values += 1
            if role == "human":
                user_chars += len(value)
            elif role == "gpt":
                assistant_chars += len(value)
            else:
                other_chars += len(value)
        if placeholder_count != (1 if image else 0):
            image_placeholder_mismatch += 1

    return {
        "dataset": "LLaVA-Instruct",
        "source_path": str(path),
        "evidence_type": "full source JSON streaming statistic",
        "canonical_records": record_count,
        "unique_record_ids": len(unique_ids),
        "unique_media_paths": len(unique_media),
        "text_only_records": text_only,
        "records_by_media_prefix": dict(prefixes.most_common()),
        "conversation_turn_distribution": distribution(turn_counts),
        "records_per_media_distribution": distribution(Counter(records_per_media.values())),
        "role_counts": dict(role_counts.most_common()),
        "user_characters": user_chars,
        "assistant_characters": assistant_chars,
        "other_role_characters": other_chars,
        "all_text_token_estimate_chars_div_4": round(
            (user_chars + assistant_chars + other_chars) / 4
        ),
        "supervised_token_estimate_assistant_chars_div_4": round(assistant_chars / 4),
        "empty_conversation_records": empty_messages,
        "empty_message_values": empty_values,
        "image_placeholder_mismatch_records": image_placeholder_mismatch,
        "source_schema_variants": dict(schema_variants.most_common()),
        "representative_source_records": samples,
        "media_sets": {
            "coco_ids": sorted(
                {
                    match.group(1)
                    for item in unique_media
                    if item.startswith("coco/")
                    for match in [re.search(r"(\d{12})", Path(item).name)]
                    if match
                }
            ),
            "gqa_ids": sorted(
                {Path(item).stem for item in unique_media if item.startswith("gqa/")}
            ),
            "textvqa_ids": sorted(
                {Path(item).stem for item in unique_media if item.startswith("textvqa/")}
            ),
            "vg_ids": sorted(
                {Path(item).stem for item in unique_media if item.startswith("vg/")}
            ),
        },
    }


def parse_fenced_json(value: str) -> Any:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.I)
    return json.loads(match.group(1) if match else value)


def vlm_r1() -> dict[str, Any]:
    path = ROOT / "vlm-r1" / "sft_related" / "mllm_rec_json.json"
    image_counts: Counter[int] = Counter()
    message_counts: Counter[int] = Counter()
    bbox_counts: Counter[int] = Counter()
    role_counts: Counter[str] = Counter()
    schema_variants: Counter[str] = Counter()
    unique_images: set[str] = set()
    samples: list[Any] = []
    parse_failure_examples: list[Any] = []
    record_count = parse_failures = invalid_bbox = reversed_bbox = 0
    negative_bbox = large_bbox = empty_labels = 0
    user_chars = assistant_chars = 0
    coordinate_min: float | None = None
    coordinate_max: float | None = None

    for row in iter_json_array_path(path):
        record_count += 1
        if len(samples) < 3:
            samples.append(compact_sample(row))
        if not isinstance(row, dict):
            schema_variants[json_type(row)] += 1
            continue
        schema_variants[",".join(sorted(row))] += 1
        images = row.get("images") if isinstance(row.get("images"), list) else []
        image_counts[len(images)] += 1
        for item in images:
            if isinstance(item, str):
                unique_images.add(Path(item).name)
        messages = row.get("messages") if isinstance(row.get("messages"), list) else []
        message_counts[len(messages)] += 1
        assistant_outputs: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = text(message.get("role")) or "<missing>"
            content = text(message.get("content"))
            role_counts[role] += 1
            if role == "user":
                user_chars += len(content)
            elif role == "assistant":
                assistant_chars += len(content)
                assistant_outputs.append(content)
        boxes: list[Any] = []
        try:
            for output in assistant_outputs:
                parsed = parse_fenced_json(output)
                if isinstance(parsed, list):
                    boxes.extend(parsed)
                else:
                    raise ValueError("assistant JSON is not a list")
        except (ValueError, TypeError, json.JSONDecodeError):
            parse_failures += 1
            if len(parse_failure_examples) < 3:
                parse_failure_examples.append(
                    {
                        "source_record_index": record_count - 1,
                        "record": compact_sample(row),
                    }
                )
            bbox_counts[0] += 1
            continue
        bbox_counts[len(boxes)] += 1
        for item in boxes:
            if not isinstance(item, dict):
                invalid_bbox += 1
                continue
            bbox = item.get("bbox_2d")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(not isinstance(value, (int, float)) for value in bbox)
            ):
                invalid_bbox += 1
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            coordinate_min = min(bbox) if coordinate_min is None else min(coordinate_min, *bbox)
            coordinate_max = max(bbox) if coordinate_max is None else max(coordinate_max, *bbox)
            if x2 <= x1 or y2 <= y1:
                reversed_bbox += 1
            if min(bbox) < 0:
                negative_bbox += 1
            if max(bbox) > 10_000:
                large_bbox += 1
            if not text(item.get("label")).strip():
                empty_labels += 1

    image_ids = {
        match.group(1)
        for name in unique_images
        for match in [re.search(r"(\d{12})", name)]
        if match
    }
    return {
        "dataset": "VLM-R1",
        "source_path": str(path),
        "evidence_type": "full source JSON streaming statistic",
        "canonical_records": record_count,
        "unique_media_filenames": len(unique_images),
        "unique_coco_train2014_ids": len(image_ids),
        "image_count_distribution": distribution(image_counts),
        "message_count_distribution": distribution(message_counts),
        "bbox_count_distribution": distribution(bbox_counts),
        "role_counts": dict(role_counts.most_common()),
        "user_characters": user_chars,
        "assistant_characters": assistant_chars,
        "supervised_token_estimate_assistant_chars_div_4": round(assistant_chars / 4),
        "assistant_parse_failures": parse_failures,
        "structurally_invalid_bbox_count": invalid_bbox,
        "non_positive_area_bbox_count": reversed_bbox,
        "negative_coordinate_bbox_count": negative_bbox,
        "coordinate_over_10000_bbox_count": large_bbox,
        "empty_bbox_label_count": empty_labels,
        "observed_coordinate_min": coordinate_min,
        "observed_coordinate_max": coordinate_max,
        "coordinate_boundary_note": (
            "Full scan validates structure and ordering. Pixel-bound checks require joined "
            "image dimensions and are reported from the decoded deterministic media sample."
        ),
        "source_schema_variants": dict(schema_variants.most_common()),
        "representative_source_records": samples,
        "assistant_parse_failure_examples": parse_failure_examples,
        "coco_ids": sorted(image_ids),
    }


def visual_genome() -> dict[str, Any]:
    root = ROOT / "VisualGenome"
    image_member, image_rows = zip_json_rows(root / "image_data.json.zip")
    image_ids: set[str] = set()
    coco_ids: set[str] = set()
    flickr_ids: set[str] = set()
    dimensions: dict[str, tuple[int, int]] = {}
    image_samples: list[Any] = []
    missing_coco = missing_flickr = invalid_dimensions = 0
    image_count = 0
    for row in image_rows:
        image_count += 1
        if len(image_samples) < 3:
            image_samples.append(compact_sample(row))
        if not isinstance(row, dict):
            continue
        image_id = str(row.get("image_id", row.get("id", "")))
        if image_id:
            image_ids.add(image_id)
        width, height = row.get("width"), row.get("height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            dimensions[image_id] = (width, height)
        else:
            invalid_dimensions += 1
        coco_id = row.get("coco_id")
        if coco_id is None:
            missing_coco += 1
        else:
            coco_ids.add(str(coco_id))
        flickr_id = row.get("flickr_id")
        if flickr_id is None:
            missing_flickr += 1
        else:
            flickr_ids.add(str(flickr_id))

    qa_member, qa_rows = zip_json_rows(root / "question_answers.json.zip")
    qa_group_count = qa_count = empty_qa_groups = 0
    unique_qa_ids: set[str] = set()
    qa_per_image: Counter[int] = Counter()
    question_chars = answer_chars = 0
    empty_questions = empty_answers = 0
    missing_qa_image_join = 0
    qa_samples: list[Any] = []
    for group in qa_rows:
        qa_group_count += 1
        if len(qa_samples) < 3:
            qa_samples.append(compact_sample(group))
        if not isinstance(group, dict):
            empty_qa_groups += 1
            continue
        qas = group.get("qas") if isinstance(group.get("qas"), list) else []
        qa_per_image[len(qas)] += 1
        if not qas:
            empty_qa_groups += 1
        group_id = str(group.get("id", ""))
        if group_id and group_id not in image_ids:
            missing_qa_image_join += 1
        for qa in qas:
            qa_count += 1
            if not isinstance(qa, dict):
                continue
            qa_id = qa.get("qa_id")
            if qa_id is not None:
                unique_qa_ids.add(str(qa_id))
            question = text(qa.get("question"))
            answer = text(qa.get("answer"))
            question_chars += len(question)
            answer_chars += len(answer)
            empty_questions += int(not question.strip())
            empty_answers += int(not answer.strip())

    region_member, region_rows = zip_json_rows(root / "region_descriptions.json.zip")
    region_group_count = region_count = empty_region_groups = 0
    unique_region_ids: set[str] = set()
    regions_per_image: Counter[int] = Counter()
    phrase_chars = empty_phrases = 0
    missing_region_image_join = invalid_region = out_of_bounds_region = 0
    region_samples: list[Any] = []
    for group in region_rows:
        region_group_count += 1
        if len(region_samples) < 3:
            region_samples.append(compact_sample(group))
        if not isinstance(group, dict):
            empty_region_groups += 1
            continue
        regions = group.get("regions") if isinstance(group.get("regions"), list) else []
        regions_per_image[len(regions)] += 1
        if not regions:
            empty_region_groups += 1
        group_id = str(group.get("id", ""))
        if group_id and group_id not in image_ids:
            missing_region_image_join += 1
        for region in regions:
            region_count += 1
            if not isinstance(region, dict):
                invalid_region += 1
                continue
            region_id = region.get("region_id")
            if region_id is not None:
                unique_region_ids.add(str(region_id))
            phrase = text(region.get("phrase"))
            phrase_chars += len(phrase)
            empty_phrases += int(not phrase.strip())
            image_id = str(region.get("image_id", group_id))
            coords = [region.get(key) for key in ("x", "y", "width", "height")]
            if any(not isinstance(value, (int, float)) for value in coords):
                invalid_region += 1
                continue
            x, y, width, height = (float(value) for value in coords)
            if width <= 0 or height <= 0:
                invalid_region += 1
                continue
            size = dimensions.get(image_id)
            if size and (x < 0 or y < 0 or x + width > size[0] or y + height > size[1]):
                out_of_bounds_region += 1

    return {
        "dataset": "VisualGenome",
        "source_root": str(root),
        "evidence_type": "full source ZIP-member JSON streaming statistic",
        "canonical_image_records": image_count,
        "unique_image_ids": len(image_ids),
        "image_metadata_member": image_member,
        "valid_dimension_records": len(dimensions),
        "invalid_dimension_records": invalid_dimensions,
        "records_with_coco_id": image_count - missing_coco,
        "unique_coco_ids": len(coco_ids),
        "records_with_flickr_id": image_count - missing_flickr,
        "unique_flickr_ids": len(flickr_ids),
        "qa_group_records": qa_group_count,
        "qa_member": qa_member,
        "canonical_qa_records": qa_count,
        "unique_qa_ids": len(unique_qa_ids),
        "qa_per_image_distribution": distribution(qa_per_image),
        "empty_qa_groups": empty_qa_groups,
        "empty_questions": empty_questions,
        "empty_answers": empty_answers,
        "question_characters": question_chars,
        "answer_characters": answer_chars,
        "qa_token_estimate_chars_div_4": round((question_chars + answer_chars) / 4),
        "missing_qa_group_image_joins": missing_qa_image_join,
        "region_group_records": region_group_count,
        "region_member": region_member,
        "canonical_region_records": region_count,
        "unique_region_ids": len(unique_region_ids),
        "regions_per_image_distribution": distribution(regions_per_image),
        "empty_region_groups": empty_region_groups,
        "empty_region_phrases": empty_phrases,
        "region_phrase_characters": phrase_chars,
        "region_token_estimate_chars_div_4": round(phrase_chars / 4),
        "missing_region_group_image_joins": missing_region_image_join,
        "structurally_invalid_regions": invalid_region,
        "out_of_bounds_regions_against_metadata": out_of_bounds_region,
        "invalid_source_artifact": "images2.zip.1 is not a valid ZIP and is excluded",
        "representative_source_records": {
            "image_data": image_samples,
            "question_answers": qa_samples,
            "region_descriptions": region_samples,
        },
        "image_ids": sorted(image_ids),
        "coco_ids": sorted(coco_ids),
    }


def ai2d() -> dict[str, Any]:
    root = ROOT / "ai2d" / "ai2d"
    images = sorted(path.name for path in (root / "images").iterdir() if path.is_file())
    image_set = set(images)
    categories = json.loads((root / "categories.json").read_text(encoding="utf-8"))
    category_counts = Counter(str(categories.get(name, "<missing>")) for name in images)
    question_files = sorted((root / "questions").glob("*.json"))
    files_with_zero_questions = 0
    question_count = option_count = 0
    correct_index_invalid = empty_questions = empty_options = 0
    abc_label_counts: Counter[str] = Counter()
    questions_per_image: Counter[int] = Counter()
    category_question_counts: Counter[str] = Counter()
    unique_question_ids: set[str] = set()
    question_chars = answer_chars = 0
    samples: list[Any] = []
    question_image_names: set[str] = set()

    for path in question_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        if len(samples) < 3:
            samples.append(compact_sample(row))
        image_name = text(row.get("imageName")) or path.name[:-5]
        question_image_names.add(image_name)
        questions = row.get("questions") if isinstance(row.get("questions"), dict) else {}
        questions_per_image[len(questions)] += 1
        files_with_zero_questions += int(not questions)
        category = str(categories.get(image_name, "<missing>"))
        category_question_counts[category] += len(questions)
        for question_text, item in questions.items():
            question_count += 1
            question_value = text(question_text)
            question_chars += len(question_value)
            empty_questions += int(not question_value.strip())
            if not isinstance(item, dict):
                correct_index_invalid += 1
                continue
            question_id = item.get("questionId")
            if question_id is not None:
                unique_question_ids.add(str(question_id))
            abc_label_counts[str(item.get("abcLabel", "<missing>"))] += 1
            options = item.get("answerTexts") if isinstance(item.get("answerTexts"), list) else []
            option_count += len(options)
            empty_options += sum(not text(option).strip() for option in options)
            correct = item.get("correctAnswer")
            if not isinstance(correct, int) or correct < 0 or correct >= len(options):
                correct_index_invalid += 1
            else:
                answer_chars += len(text(options[correct]))

    annotation_counts: Counter[str] = Counter()
    annotation_schema_variants: Counter[str] = Counter()
    annotation_files = sorted((root / "annotations").glob("*.json"))
    annotation_parse_errors = 0
    for path in annotation_files:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            annotation_parse_errors += 1
            continue
        if not isinstance(row, dict):
            annotation_schema_variants[json_type(row)] += 1
            continue
        annotation_schema_variants[",".join(sorted(row))] += 1
        for key in ("arrows", "arrowHeads", "blobs", "text", "relationships"):
            value = row.get(key)
            if isinstance(value, dict):
                annotation_counts[key] += len(value)

    missing_question_file_images = image_set - question_image_names
    question_files_without_image = question_image_names - image_set
    images_without_any_question = len(missing_question_file_images) + files_with_zero_questions
    return {
        "dataset": "AI2D",
        "source_root": str(root),
        "evidence_type": "full source JSON-file statistic",
        "canonical_image_records": len(images),
        "question_files": len(question_files),
        "annotation_files": len(annotation_files),
        "canonical_question_records": question_count,
        "unique_question_ids": len(unique_question_ids),
        "answer_option_entries": option_count,
        "images_missing_question_file": len(missing_question_file_images),
        "question_files_with_zero_questions": files_with_zero_questions,
        "images_without_any_question": images_without_any_question,
        "question_files_without_image": len(question_files_without_image),
        "questions_per_question_file_distribution": distribution(questions_per_image),
        "category_image_counts": dict(category_counts.most_common()),
        "category_question_counts": dict(category_question_counts.most_common()),
        "abc_label_counts": dict(abc_label_counts.most_common()),
        "empty_question_texts": empty_questions,
        "empty_answer_options": empty_options,
        "invalid_correct_answer_indices": correct_index_invalid,
        "question_characters": question_chars,
        "correct_answer_characters": answer_chars,
        "supervised_token_estimate_chars_div_4": round((question_chars + answer_chars) / 4),
        "annotation_object_counts": dict(annotation_counts.most_common()),
        "annotation_schema_variants": dict(annotation_schema_variants.most_common()),
        "annotation_parse_errors": annotation_parse_errors,
        "representative_source_records": samples,
    }


def robo_task_family(task: str) -> str:
    lowered = task.lower()
    if "affordance" in lowered:
        return "affordance"
    if "success" in lowered:
        return "success_judgment"
    if "future" in lowered:
        return "future_prediction"
    if "past" in lowered or "description" in lowered:
        return "past_description"
    if "planning" in lowered or "remaining" in lowered or "next" in lowered:
        return "planning"
    return "other"


def robovqa() -> dict[str, Any]:
    root = ROOT / "robovqa"
    paths = sorted(root.glob("robovqa_*.json"))
    records_by_file: Counter[str] = Counter()
    unique_videos: set[str] = set()
    unique_uids: set[str] = set()
    task_counts: Counter[str] = Counter()
    task_families: Counter[str] = Counter()
    conversation_task_counts: Counter[str] = Counter()
    conversation_task_families: Counter[str] = Counter()
    conversation_kind_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    records_per_video: Counter[str] = Counter()
    supervision_per_video: Counter[str] = Counter()
    schema_by_file: dict[str, Counter[str]] = defaultdict(Counter)
    samples_by_file: dict[str, list[Any]] = defaultdict(list)
    supervision_signatures: set[bytes] = set()
    conversation_signatures: set[bytes] = set()
    matched_conversation_task_signatures: set[bytes] = set()
    record_count = supervision_units = 0
    empty_questions = empty_answers = empty_video = records_without_task_metadata = 0
    reasoning_task_matches = reasoning_task_no_match = reasoning_task_ambiguous = 0
    reasoning_missing_think_tags = reasoning_missing_answer_tags = 0
    question_chars = answer_chars = instruction_chars = 0
    user_chars = assistant_chars = 0

    for path in paths:
        for row in iter_json_array_path(path):
            record_count += 1
            records_by_file[path.name] += 1
            if len(samples_by_file[path.name]) < 1:
                samples_by_file[path.name].append(compact_sample(row))
            if not isinstance(row, dict):
                schema_by_file[path.name][json_type(row)] += 1
                continue
            schema_by_file[path.name][",".join(sorted(row))] += 1
            video = text(row.get("video"))
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if not video:
                video = text(metadata.get("video_location"))
            video_id = Path(video).stem if video else ""
            if video_id:
                unique_videos.add(video_id)
                records_per_video[video_id] += 1
            else:
                empty_video += 1
            conversations = row.get("conversations") if isinstance(row.get("conversations"), list) else []
            if conversations:
                conversation_signatures.add(signature(conversations))
            user_messages: list[str] = []
            assistant_messages: list[str] = []
            for message in conversations:
                if not isinstance(message, dict):
                    continue
                role = text(message.get("role")) or text(message.get("from"))
                content = text(message.get("content")) or text(message.get("value"))
                if role in {"user", "human"}:
                    user_chars += len(content)
                    user_messages.append(content)
                elif role in {"assistant", "gpt"}:
                    assistant_chars += len(content)
                    assistant_messages.append(content)
            task_metadata = metadata.get("task_metadata")
            if isinstance(task_metadata, dict):
                task_rows = [task_metadata]
            elif isinstance(task_metadata, list):
                task_rows = task_metadata
            else:
                task_rows = []
            if not task_rows:
                records_without_task_metadata += 1
            if path.name == "robovqa_understanding.json":
                conversation_kind_counts["understanding_description"] += 1
                conversation_task_counts["description.robot_objects_actions"] += 1
                conversation_task_families["embodied_scene_description"] += 1
            else:
                conversation_kind_counts["reasoning"] += 1
                user_prompt = "\n".join(user_messages).strip()
                candidates: list[tuple[int, dict[str, Any]]] = []
                for item in task_rows:
                    if not isinstance(item, dict):
                        continue
                    question_processed = text(item.get("question_processed")).strip()
                    if question_processed and user_prompt.startswith(question_processed):
                        candidates.append((len(question_processed), item))
                if candidates:
                    longest = max(length for length, _ in candidates)
                    matches = [item for length, item in candidates if length == longest]
                    reasoning_task_ambiguous += int(len(matches) > 1)
                    matched = matches[0]
                    task = text(matched.get("task")) or "<missing>"
                    family = robo_task_family(task)
                    conversation_task_counts[task] += 1
                    conversation_task_families[family] += 1
                    reasoning_task_matches += 1
                    matched_conversation_task_signatures.add(
                        signature(
                            [
                                video_id,
                                task,
                                text(matched.get("instruction_processed"))
                                or text(matched.get("instruction")),
                                text(matched.get("question_processed"))
                                or text(matched.get("question")),
                                text(matched.get("answer_processed"))
                                or text(matched.get("answer")),
                            ]
                        )
                    )
                else:
                    reasoning_task_no_match += 1
                    conversation_task_counts["<unmatched>"] += 1
                    conversation_task_families["unmatched"] += 1
                combined_assistant = "\n".join(assistant_messages)
                reasoning_missing_think_tags += int(
                    "<think>" not in combined_assistant
                    or "</think>" not in combined_assistant
                )
                reasoning_missing_answer_tags += int(
                    "<answer>" not in combined_assistant
                    or "</answer>" not in combined_assistant
                )
            for item in task_rows:
                supervision_units += 1
                if not isinstance(item, dict):
                    continue
                task = text(item.get("task")) or "<missing>"
                split = text(item.get("split")) or "<missing>"
                question = text(item.get("question_processed")) or text(item.get("question"))
                answer = text(item.get("answer_processed")) or text(item.get("answer"))
                instruction = text(item.get("instruction_processed")) or text(item.get("instruction"))
                item_video = text(item.get("video_id")) or Path(text(item.get("video"))).stem
                item_video = item_video or video_id
                if item_video:
                    unique_videos.add(item_video)
                    supervision_per_video[item_video] += 1
                uid = item.get("uid")
                if uid is not None:
                    unique_uids.add(str(uid))
                task_counts[task] += 1
                task_families[robo_task_family(task)] += 1
                split_counts[split] += 1
                question_chars += len(question)
                answer_chars += len(answer)
                instruction_chars += len(instruction)
                empty_questions += int(not question.strip())
                empty_answers += int(not answer.strip())
                if answer.strip().lower() in {"yes", "no", "true", "false"}:
                    answer_counts[answer.strip().lower()] += 1
                supervision_signatures.add(
                    signature([item_video, task, instruction, question, answer])
                )
        print(f"finished {path.name}: {records_by_file[path.name]}", file=sys.stderr)

    return {
        "dataset": "RoboVQA",
        "source_root": str(root),
        "evidence_type": "full source JSON streaming statistic",
        "canonical_source_records": record_count,
        "records_by_file": dict(records_by_file),
        "unique_video_ids_in_json": len(unique_videos),
        "unique_uids": len(unique_uids),
        "canonical_supervision_units": supervision_units,
        "unique_supervision_signatures": len(supervision_signatures),
        "duplicate_supervision_units_exact_signature": supervision_units
        - len(supervision_signatures),
        "unique_conversation_signatures": len(conversation_signatures),
        "records_per_video_distribution": distribution(Counter(records_per_video.values())),
        "supervision_per_video_distribution": distribution(
            Counter(supervision_per_video.values())
        ),
        "task_counts": dict(task_counts.most_common()),
        "task_family_counts": dict(task_families.most_common()),
        "conversation_kind_counts": dict(conversation_kind_counts.most_common()),
        "conversation_task_counts": dict(conversation_task_counts.most_common()),
        "conversation_task_family_counts": dict(
            conversation_task_families.most_common()
        ),
        "reasoning_conversation_task_matches": reasoning_task_matches,
        "reasoning_conversation_task_no_match": reasoning_task_no_match,
        "reasoning_conversation_task_ambiguous": reasoning_task_ambiguous,
        "unique_matched_conversation_task_signatures": len(
            matched_conversation_task_signatures
        ),
        "duplicate_reasoning_conversations_by_task_signature": reasoning_task_matches
        - len(matched_conversation_task_signatures),
        "reasoning_records_missing_think_tags": reasoning_missing_think_tags,
        "reasoning_records_missing_answer_tags": reasoning_missing_answer_tags,
        "split_counts": dict(split_counts.most_common()),
        "binary_answer_counts": dict(answer_counts.most_common()),
        "question_characters": question_chars,
        "answer_characters": answer_chars,
        "instruction_characters": instruction_chars,
        "conversation_user_characters": user_chars,
        "conversation_assistant_characters": assistant_chars,
        "supervised_token_estimate_chars_div_4": round(
            (question_chars + answer_chars + instruction_chars) / 4
        ),
        "empty_questions": empty_questions,
        "empty_answers": empty_answers,
        "records_without_video_identity": empty_video,
        "records_without_task_metadata": records_without_task_metadata,
        "source_schema_variants_by_file": {
            name: dict(counts.most_common()) for name, counts in schema_by_file.items()
        },
        "representative_source_records_by_file": dict(samples_by_file),
        "video_ids": sorted(unique_videos),
    }


COLLECTORS = {
    "LLaVA-Instruct": llava,
    "VLM-R1": vlm_r1,
    "VisualGenome": visual_genome,
    "AI2D": ai2d,
    "RoboVQA": robovqa,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in COLLECTORS:
        choices = ", ".join(COLLECTORS)
        raise SystemExit(f"usage: remote_collect_json_stats.py {{{choices}}}")
    result = COLLECTORS[sys.argv[1]]()
    result["source_only_contract"] = True
    result["source_root_boundary"] = str(ROOT)
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
