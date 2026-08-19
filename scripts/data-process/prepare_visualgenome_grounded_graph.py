#!/usr/bin/env python3
"""Build grounded Visual Genome captions from region-graph object annotations.

Unlike region-level caption conversion, this converter replaces each aligned
entity mention with ms-swift's independent ``<ref-object>`` and ``<bbox>``
placeholders. Bounding boxes come from the region graph's object annotations,
not from the enclosing region rectangle.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from prepare_visualgenome_swift import (
    JsonlTaskWriter,
    find_json_source,
    get_matched_image_id,
    iter_json_array,
    open_json_source,
    resolve_extracted_image_dirs,
    url_image_parts,
    write_report,
)


DEFAULT_INPUT_DIR = Path("/mnt/luojunkun/stage1/dataset/VisualGenome")
DEFAULT_IMAGES_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/visualgenome")
DEFAULT_OUTPUT_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/visualgenome_grounded_graph")
DEFAULT_PROMPT = "<image>\nDescribe the image with grounded objects."
NAME_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class Mention:
    start: int
    end: int
    text: str
    candidates: list[dict[str, Any]]
    boxes: list[list[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Visual Genome region graphs to ms-swift grounded-caption JSONL."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relative-paths", action="store_true")
    parser.add_argument("--max-regions", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--max-report-examples", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.input_dir = args.input_dir.expanduser().resolve()
    args.images_dir = args.images_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be in the range [0, 1)")
    if args.max_regions is not None and args.max_regions <= 0:
        raise SystemExit("--max-regions must be greater than zero")
    if args.progress_every < 0 or args.max_report_examples < 0:
        raise SystemExit("progress and report limits must be non-negative")
    if args.prompt.count("<image>") != 1:
        raise SystemExit("--prompt must contain exactly one <image>")
    if "<ref-object>" in args.prompt or "<bbox>" in args.prompt:
        raise SystemExit("--prompt must not contain grounding placeholders")
    if args.output_dir == args.images_dir:
        raise SystemExit("--output-dir must differ from --images-dir to prevent source overwrite")


def normalized_name(value: Any) -> str:
    text = NAME_RE.sub(" ", str(value or "").casefold()).strip()
    words = text.split()
    normalized = []
    for word in words:
        if len(word) > 3 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("es"):
            word = word[:-2]
        elif len(word) > 2 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        normalized.append(word)
    return " ".join(normalized)


def load_trusted_image_paths(
    source: Any, image_dirs: dict[str, Path]
) -> tuple[dict[int, Path], dict[int, tuple[int, int]]]:
    """Resolve paths from image_data URLs after the original conversion verified all files."""
    result: dict[int, Path] = {}
    sizes: dict[int, tuple[int, int]] = {}
    with open_json_source(source) as stream:
        for image in iter_json_array(stream):
            raw_image_id = image.get("image_id", image.get("id"))
            parts = url_image_parts(str(image.get("url") or ""))
            if raw_image_id is None or parts is None:
                raise ValueError("image_data row is missing image_id or a Visual Genome URL")
            folder_name, filename = parts
            directory = image_dirs.get(folder_name)
            if directory is None:
                raise FileNotFoundError(f"Missing extracted image directory: {folder_name}")
            image_id = int(raw_image_id)
            path = directory / filename
            previous = result.get(image_id)
            if previous is not None and previous != path:
                raise ValueError(f"Conflicting paths for image_id={image_id}: {previous}, {path}")
            result[image_id] = path
            sizes[image_id] = (int(image["width"]), int(image["height"]))
    print(f"[info] trusted image paths={len(result):,}", flush=True)
    return result, sizes


def bounded_interval(start: float, end: float, limit: int) -> tuple[float, float]:
    bounded_start = max(0.0, min(start, limit))
    bounded_end = max(0.0, min(end, limit))
    if bounded_start < bounded_end:
        return bounded_start, bounded_end
    if start >= limit and limit > 0:
        return float(limit - 1), float(limit)
    if end <= 0 and limit > 0:
        return 0.0, 1.0
    return bounded_start, bounded_end


def clamp_region_objects(region: dict[str, Any], width: int, height: int, stats: Counter) -> None:
    for value in region.get("objects") or []:
        if not isinstance(value, dict):
            continue
        box = object_box(value)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1, x2 = bounded_interval(x1, x2, width)
        y1, y2 = bounded_interval(y1, y2, height)
        clamped = [x1, y1, x2, y2]
        if clamped != box:
            stats["source_boxes_clamped"] += 1
        value["x"], value["y"] = clamped[0], clamped[1]
        value["w"], value["h"] = clamped[2] - clamped[0], clamped[3] - clamped[1]


def object_box(value: dict[str, Any]) -> list[float] | None:
    try:
        x = float(value["x"])
        y = float(value["y"])
        width = float(value["w"])
        height = float(value["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(number) for number in (x, y, width, height)) or width <= 0 or height <= 0:
        return None
    return [x, y, x + width, y + height]


def unique_objects(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen: set[Any] = set()
    for value in values:
        key = value.get("object_id")
        if key is None:
            box = object_box(value)
            key = (normalized_name(value.get("name")), tuple(box or []))
        if key in seen or object_box(value) is None:
            continue
        seen.add(key)
        result.append(value)
    return result


def candidate_objects(entity: dict[str, Any], objects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    synset = str(entity.get("synset_name") or "").strip()
    if synset:
        matched = unique_objects(
            value for value in objects if synset in (value.get("synsets") or [])
        )
        if matched:
            return matched, "synset"
    entity_name = normalized_name(entity.get("entity_name"))
    if entity_name:
        matched = unique_objects(
            value for value in objects if normalized_name(value.get("name")) == entity_name
        )
        if matched:
            return matched, "name"
    return [], "unmatched"


def object_name_patterns(name: Any) -> list[re.Pattern[str]]:
    words = str(name or "").casefold().strip().split()
    if not words or any(not NAME_RE.sub("", word) for word in words):
        return []
    forms = [words]
    last = words[-1]
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        plural = last[:-1] + "ies"
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        plural = last + "es"
    else:
        plural = last + "s"
    forms.append([*words[:-1], plural])
    return [
        re.compile(r"(?<!\w)" + r"\s+".join(re.escape(word) for word in form) + r"(?!\w)", re.IGNORECASE)
        for form in forms
    ]


def surface_name_mentions(phrase: str, objects: list[dict[str, Any]]) -> list[Mention]:
    by_span: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for value in unique_objects(objects):
        for pattern in object_name_patterns(value.get("name")):
            for match in pattern.finditer(phrase):
                by_span[(match.start(), match.end())].append(value)
    return [
        Mention(start, end, phrase[start:end], unique_objects(candidates), [])
        for (start, end), candidates in by_span.items()
    ]


def object_key(value: dict[str, Any]) -> Any:
    return value.get("object_id", (normalized_name(value.get("name")), tuple(object_box(value) or [])))


def is_plural_mention(value: str) -> bool:
    words = NAME_RE.sub(" ", value.casefold()).split()
    if not words or words[0] in {"a", "an", "one"}:
        return False
    last = words[-1]
    if last in {"people", "men", "women", "children", "feet", "teeth", "geese", "mice"}:
        return True
    return last.endswith("s") and not last.endswith(("ss", "us", "is"))


def region_box(region: dict[str, Any]) -> list[float] | None:
    try:
        x = float(region["x"])
        y = float(region["y"])
        width = float(region["width"])
        height = float(region["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(number) for number in (x, y, width, height)) or width <= 0 or height <= 0:
        return None
    return [x, y, x + width, y + height]


def region_overlap_score(value: dict[str, Any], target: list[float] | None) -> tuple[float, float]:
    box = object_box(value)
    if box is None or target is None:
        return (0.0, 0.0)
    intersection_width = max(0.0, min(box[2], target[2]) - max(box[0], target[0]))
    intersection_height = max(0.0, min(box[3], target[3]) - max(box[1], target[1]))
    intersection = intersection_width * intersection_height
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    union = box_area + target_area - intersection
    center_distance = abs((box[0] + box[2]) - (target[0] + target[2])) + abs(
        (box[1] + box[3]) - (target[1] + target[3])
    )
    return (intersection / union if union else 0.0, -center_distance)


def assign_repeated_candidates(mentions: list[Mention], region: dict[str, Any], stats: Counter) -> None:
    groups: dict[tuple[Any, ...], list[Mention]] = defaultdict(list)
    for mention in mentions:
        candidate_ids = tuple(object_key(value) for value in mention.candidates)
        groups[candidate_ids].append(mention)

    for group in groups.values():
        candidates = sorted(
            group[0].candidates,
            key=lambda value: (float(value.get("x", 0)), float(value.get("y", 0))),
        )
        if len(group) == 1:
            mention = group[0]
            if len(candidates) > 1 and not is_plural_mention(mention.text):
                candidates = [max(candidates, key=lambda value: region_overlap_score(value, region_box(region)))]
                stats["singular_multi_candidate_resolved"] += 1
            mention.boxes = [object_box(value) for value in candidates if object_box(value) is not None]
            continue
        assignments: list[list[dict[str, Any]]] = [[] for _ in group]
        for index, candidate in enumerate(candidates):
            assignments[min(index, len(group) - 1)].append(candidate)
        for index, mention in enumerate(group):
            selected = assignments[index] or candidates[:1]
            mention.boxes = [object_box(value) for value in selected if object_box(value) is not None]


def grounded_caption(region: dict[str, Any], stats: Counter) -> tuple[str, list[str], list[list[float]]] | None:
    phrase = str(region.get("phrase") or "")
    if not phrase.strip():
        stats["empty_phrase"] += 1
        return None
    objects = [value for value in (region.get("objects") or []) if isinstance(value, dict)]
    entities = [value for value in (region.get("synsets") or []) if isinstance(value, dict)]
    mentions: list[Mention] = []
    for entity in entities:
        try:
            start = int(entity["entity_idx_start"])
            end = int(entity["entity_idx_end"])
        except (KeyError, TypeError, ValueError):
            stats["invalid_entity_span"] += 1
            continue
        if not 0 <= start < end <= len(phrase):
            stats["invalid_entity_span"] += 1
            continue
        candidates, method = candidate_objects(entity, objects)
        stats[f"entity_match_{method}"] += 1
        if not candidates:
            continue
        mentions.append(Mention(start, end, phrase[start:end], candidates, []))

    for mention in surface_name_mentions(phrase, objects):
        if any(mention.start < existing.end and existing.start < mention.end for existing in mentions):
            continue
        mentions.append(mention)
        stats["entity_match_surface_name"] += 1

    if not mentions:
        stats["regions_without_aligned_entities"] += 1
        return None

    non_overlapping: list[Mention] = []
    cursor = -1
    for mention in sorted(mentions, key=lambda value: (value.start, -(value.end - value.start))):
        if mention.start < cursor:
            stats["overlapping_entities_skipped"] += 1
            continue
        non_overlapping.append(mention)
        cursor = mention.end
    assign_repeated_candidates(non_overlapping, region, stats)
    non_overlapping = [mention for mention in non_overlapping if mention.boxes]
    if not non_overlapping:
        stats["regions_without_valid_boxes"] += 1
        return None

    content: list[str] = []
    refs: list[str] = []
    boxes: list[list[float]] = []
    cursor = 0
    for mention in non_overlapping:
        content.append(phrase[cursor:mention.start])
        content.append("<ref-object>" + "<bbox>" * len(mention.boxes))
        refs.append(mention.text.strip())
        boxes.extend(mention.boxes)
        if len(mention.boxes) > 1:
            stats["multi_box_mentions"] += 1
        cursor = mention.end
    content.append(phrase[cursor:])
    result = "".join(content).strip()
    if result.count("<ref-object>") != len(refs) or result.count("<bbox>") != len(boxes):
        raise AssertionError("placeholder count mismatch")
    stats["grounded_refs"] += len(refs)
    stats["grounded_boxes"] += len(boxes)
    return result, refs, boxes


def make_record(
    region: dict[str, Any], image_id: int, system: str, prompt: str, stats: Counter
) -> dict[str, Any] | None:
    grounded = grounded_caption(region, stats)
    if grounded is None:
        return None
    content, refs, boxes = grounded
    region_id = int(region["region_id"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": content},
    ])
    return {
        "image_id": image_id,
        "region_id": region_id,
        "messages": messages,
        "objects": {
            "ref": refs,
            "bbox": boxes,
            "bbox_type": "real",
            "image_id": [0] * len(boxes),
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    image_source = find_json_source(args.input_dir, "image_data.json")
    graph_source = find_json_source(args.input_dir, "region_graphs.json")
    image_dirs = resolve_extracted_image_dirs(args.images_dir, [])
    image_paths, sizes = load_trusted_image_paths(image_source, image_dirs)
    stats: Counter = Counter()
    examples: list[dict[str, Any]] = []
    writer = JsonlTaskWriter(
        "regions",
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        relative_paths=args.relative_paths,
        overwrite=args.overwrite,
    )
    stop = False
    with writer, open_json_source(graph_source) as stream:
        for image_entry in iter_json_array(stream):
            parent_image_id = image_entry.get("image_id", image_entry.get("id"))
            for region in image_entry.get("regions") or []:
                stats["read_regions"] += 1
                try:
                    region_id = region.get("region_id", "unknown")
                    image_id = get_matched_image_id(parent_image_id, region, f"region {region_id}")
                    clamp_region_objects(region, *sizes[image_id], stats)
                    record = make_record(region, image_id, args.system, args.prompt, stats)
                except (KeyError, TypeError, ValueError):
                    stats["invalid_regions"] += 1
                    continue
                if record is not None:
                    image_id = record["image_id"]
                    image_path = image_paths.get(image_id)
                    if image_path is None:
                        stats["missing_image_regions"] += 1
                    else:
                        writer.write(image_id, image_path, record)
                        stats["written_regions"] += 1
                        if len(examples) < args.max_report_examples:
                            examples.append({
                                "region_id": record["region_id"],
                                "source_phrase": region.get("phrase"),
                                "assistant": record["messages"][-1]["content"],
                                "objects": record["objects"],
                            })
                if args.progress_every and stats["read_regions"] % args.progress_every == 0:
                    print(
                        f"[progress] read={stats['read_regions']:,} written={stats['written_regions']:,} "
                        f"skipped={stats['regions_without_aligned_entities']:,}",
                        flush=True,
                    )
                if args.max_regions is not None and stats["read_regions"] >= args.max_regions:
                    stop = True
                    break
            if stop:
                break

    report = {
        "format": "ms-swift grounded caption from Visual Genome region graphs",
        "source": graph_source.describe(),
        "output_dir": str(args.output_dir),
        "prompt": args.prompt,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "missing_image_ids": [],
        "stats": dict(sorted(stats.items())),
        "split_rows": writer.counts,
        "examples": examples,
    }
    write_report(args.output_dir / "visualgenome_grounded_graph_report.json", report, args.overwrite)


if __name__ == "__main__":
    main()
