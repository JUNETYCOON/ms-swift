#!/usr/bin/env python3
"""Remove overlapping academic sources from LLaVA and split complete images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO

from split.grouped_jsonl_split import load_record


DEFAULT_DATA_ROOT = Path("/mnt/luojunkun/stage1/dataset_ms-swift")
DEFAULT_LLAVA_DIR = DEFAULT_DATA_ROOT / "llava-instruct"
DEFAULT_INPUT = DEFAULT_LLAVA_DIR / "llava_v1_5_mix665k_sft_msswift.jsonl"
DEFAULT_TRAIN = DEFAULT_LLAVA_DIR / "llava_v1_5_mix665k_sft_msswift_train.jsonl"
DEFAULT_EVAL = DEFAULT_LLAVA_DIR / "llava_v1_5_mix665k_sft_msswift_val.jsonl"
DEFAULT_REPORT = DEFAULT_LLAVA_DIR / "llava_curation_report.json"
DEFAULT_DEDUP_INPUTS = (
    DEFAULT_DATA_ROOT / "VQAv2/vqav2_train_sft_msswift.jsonl",
    DEFAULT_DATA_ROOT / "VQAv2/vqav2_validation_sft_msswift.jsonl",
)
DEFAULT_EXCLUDED_SOURCES = ("gqa", "textvqa", "visualgenome")
IMAGE_TOKEN_PATTERN = re.compile(r"<\s*image\s*>", re.IGNORECASE)
COCO_STEM_PATTERN = re.compile(
    r"(?:COCO_(?:train|val|test)\d{4}_)?0*([0-9]+)$", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prefer standalone academic datasets over their LLaVA copies, remove "
            "exact VQAv2 question overlap, and split by complete image groups."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--exclude-source",
        nargs="+",
        choices=("gqa", "textvqa", "visualgenome", "ocrvqa"),
        default=list(DEFAULT_EXCLUDED_SOURCES),
        help="Standalone sources to remove from the lower-priority LLaVA mixture.",
    )
    parser.add_argument(
        "--dedup-jsonl",
        type=Path,
        nargs="*",
        default=list(DEFAULT_DEDUP_INPUTS),
        help="Higher-priority JSONL files indexed by canonical image ID and user question.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.input_json = args.input_json.expanduser().resolve()
    args.train_output = args.train_output.expanduser().resolve()
    args.eval_output = args.eval_output.expanduser().resolve()
    args.report_output = args.report_output.expanduser().resolve()
    args.dedup_jsonl = [path.expanduser().resolve() for path in args.dedup_jsonl]
    if not args.input_json.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {args.input_json}")
    for path in args.dedup_jsonl:
        if not path.is_file():
            raise FileNotFoundError(f"Dedup JSONL does not exist: {path}")
    if not 0 < args.eval_ratio < 1:
        raise ValueError("--eval-ratio must be in the range (0, 1)")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be greater than or equal to zero")
    outputs = (args.train_output, args.eval_output, args.report_output)
    if len(set(outputs)) != len(outputs):
        raise ValueError("Train, eval, and report outputs must be different")
    if args.input_json in outputs or any(path in outputs for path in args.dedup_jsonl):
        raise ValueError("Output paths must not overwrite an input JSONL")
    for output in outputs:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite.")


def temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def open_temporary(path: Path) -> TextIO:
    return temporary_sibling(path).open("w", encoding="utf-8", newline="\n")


def normalized_question(content: str) -> str:
    content = IMAGE_TOKEN_PATTERN.sub(" ", content)
    content = " ".join(content.casefold().split())
    return content.strip(" \t\r\n.?!")


def source_name(path_text: str) -> str:
    normalized = path_text.replace("\\", "/").casefold()
    if "/gqa/" in normalized:
        return "gqa"
    if "/textvqa/" in normalized:
        return "textvqa"
    if "/visualgenome/" in normalized:
        return "visualgenome"
    if "ocrvqa" in normalized or "ocr-vqa" in normalized or "ocr_vqa" in normalized:
        return "ocrvqa"
    if "coco" in normalized:
        return "coco"
    return "other"


def canonical_image_id(path_text: str, jsonl_path: Path) -> str:
    source = source_name(path_text)
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = jsonl_path.parent / path
    stem = path.stem
    if source == "coco":
        match = COCO_STEM_PATTERN.fullmatch(stem)
        if match:
            return f"coco:{int(match.group(1))}"
    if source in {"gqa", "textvqa", "visualgenome", "ocrvqa"}:
        return f"{source}:{stem.casefold()}"
    return os.path.abspath(os.path.normpath(str(path)))


def image_values(record: dict[str, Any]) -> list[str]:
    value = record.get("images", [])
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(item).strip() for item in values if item is not None and str(item).strip()]


def image_group(record: dict[str, Any], jsonl_path: Path) -> tuple[str, list[str]]:
    images = sorted(
        {canonical_image_id(value, jsonl_path) for value in image_values(record)}
    )
    if images:
        return json.dumps(images, ensure_ascii=False, separators=(",", ":")), images
    messages = record.get("messages")
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()
    return f"text:{digest}", []


def user_questions(record: dict[str, Any]) -> list[str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    questions: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and (question := normalized_question(content)):
            questions.append(question)
    return questions


def record_signatures(record: dict[str, Any], jsonl_path: Path) -> set[str]:
    _, images = image_group(record, jsonl_path)
    if not images:
        return set()
    media_key = json.dumps(images, ensure_ascii=False, separators=(",", ":"))
    return {f"{media_key}\0{question}" for question in user_questions(record)}


def record_sources(record: dict[str, Any]) -> set[str]:
    return {source_name(path) for path in image_values(record)}


def stable_is_eval(group: str, eval_ratio: float, seed: int) -> bool:
    threshold = int(eval_ratio * (1 << 64))
    digest = hashlib.blake2b(
        f"{seed}\0{group}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") < threshold


def build_dedup_index(
    paths: Iterable[Path], progress_every: int
) -> tuple[set[str], dict[str, int]]:
    signatures: set[str] = set()
    rows = 0
    rows_with_signatures = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                rows += 1
                record = load_record(line, path, line_number)
                values = record_signatures(record, path)
                if values:
                    rows_with_signatures += 1
                    signatures.update(values)
                if progress_every and rows % progress_every == 0:
                    print(
                        f"[index] rows={rows:,} signatures={len(signatures):,}",
                        flush=True,
                    )
    return signatures, {
        "rows": rows,
        "rows_with_signatures": rows_with_signatures,
        "unique_signatures": len(signatures),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    dedup_index, dedup_stats = build_dedup_index(
        args.dedup_jsonl, args.progress_every
    )
    excluded = set(args.exclude_source)
    stats: Counter[str] = Counter()
    excluded_by_source: Counter[str] = Counter()
    train_groups: set[str] = set()
    eval_groups: set[str] = set()
    train_images: set[str] = set()
    eval_images: set[str] = set()
    train_stream = open_temporary(args.train_output)
    eval_stream = open_temporary(args.eval_output)
    try:
        with args.input_json.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                stats["input_rows"] += 1
                record = load_record(line, args.input_json, line_number)
                matched_sources = record_sources(record) & excluded
                if matched_sources:
                    stats["excluded_source_rows"] += 1
                    for name in matched_sources:
                        excluded_by_source[name] += 1
                    continue
                signatures = record_signatures(record, args.input_json)
                if signatures & dedup_index:
                    stats["excluded_exact_overlap_rows"] += 1
                    continue
                group, images = image_group(record, args.input_json)
                output_line = line.rstrip("\r\n") + "\n"
                if stable_is_eval(group, args.eval_ratio, args.seed):
                    eval_stream.write(output_line)
                    stats["eval_rows"] += 1
                    eval_groups.add(group)
                    eval_images.update(images)
                else:
                    train_stream.write(output_line)
                    stats["train_rows"] += 1
                    train_groups.add(group)
                    train_images.update(images)
                stats["output_rows"] += 1
                if args.progress_every and stats["input_rows"] % args.progress_every == 0:
                    print(
                        f"[curate] input={stats['input_rows']:,} output={stats['output_rows']:,} "
                        f"source_drop={stats['excluded_source_rows']:,} "
                        f"exact_drop={stats['excluded_exact_overlap_rows']:,}",
                        flush=True,
                    )

        group_overlap = train_groups & eval_groups
        image_overlap = train_images & eval_images
        if group_overlap or image_overlap:
            raise ValueError(
                f"Leakage detected: group_overlap={len(group_overlap):,} "
                f"image_overlap={len(image_overlap):,}"
            )
        if not stats["train_rows"] or not stats["eval_rows"]:
            raise ValueError("Curation must leave at least one row in train and eval")
        if (
            stats["input_rows"]
            != stats["output_rows"]
            + stats["excluded_source_rows"]
            + stats["excluded_exact_overlap_rows"]
        ):
            raise ValueError("Curation row accounting is inconsistent")

        train_temporary = Path(train_stream.name)
        eval_temporary = Path(eval_stream.name)
        train_stream.close()
        eval_stream.close()
        os.replace(train_temporary, args.train_output)
        os.replace(eval_temporary, args.eval_output)
    except BaseException:
        for stream in (train_stream, eval_stream):
            temporary = Path(stream.name)
            stream.close()
            if temporary.exists():
                temporary.unlink()
        raise

    report = {
        "input_jsonl": str(args.input_json),
        "train_jsonl": str(args.train_output),
        "eval_jsonl": str(args.eval_output),
        "higher_priority_dedup_jsonl": [str(path) for path in args.dedup_jsonl],
        "excluded_sources": sorted(excluded),
        "dedup_key": "canonical_image_id + normalized_user_question",
        "seed": args.seed,
        "requested_eval_ratio": args.eval_ratio,
        "actual_eval_ratio": stats["eval_rows"] / stats["output_rows"],
        "dedup_index": dedup_stats,
        "rows": {
            "input_rows": stats["input_rows"],
            "output_rows": stats["output_rows"],
            "train_rows": stats["train_rows"],
            "eval_rows": stats["eval_rows"],
            "excluded_source_rows": stats["excluded_source_rows"],
            "excluded_exact_overlap_rows": stats["excluded_exact_overlap_rows"],
        },
        "excluded_rows_by_source": dict(sorted(excluded_by_source.items())),
        "groups": {
            "train_groups": len(train_groups),
            "eval_groups": len(eval_groups),
            "train_images": len(train_images),
            "eval_images": len(eval_images),
            "group_overlap": 0,
            "image_overlap": 0,
        },
    }
    write_report(args.report_output, report)
    return report


def main() -> None:
    args = parse_args()
    validate_args(args)
    report = run(args)
    rows = report["rows"]
    print(
        f"[done] input={rows['input_rows']:,} output={rows['output_rows']:,} "
        f"train={rows['train_rows']:,} eval={rows['eval_rows']:,} "
        f"source_drop={rows['excluded_source_rows']:,} "
        f"exact_drop={rows['excluded_exact_overlap_rows']:,}"
    )
    print(f"[write] {args.train_output}")
    print(f"[write] {args.eval_output}")
    print(f"[write] {args.report_output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
