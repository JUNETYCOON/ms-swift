#!/usr/bin/env python3
"""Create a deterministic, type-stratified RoboVQA evaluation sample."""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_train_sft_eval.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/robovqa/robovqa_eval_stratified_256.jsonl"
)
DEFAULT_REPORT = DEFAULT_OUTPUT.with_suffix(".report.json")


@dataclass(frozen=True)
class Candidate:
    line_number: int
    record: dict[str, Any]
    answer_type: str
    category: str
    video: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--yes-no-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def assistant_answer(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")
    answers = [
        message.get("content")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not answers or not isinstance(answers[-1], str):
        raise ValueError("record has no assistant answer")
    return answers[-1].strip()


def answer_type(record: dict[str, Any]) -> str:
    answer = assistant_answer(record).casefold().strip()
    answer = answer.strip(string.whitespace + string.punctuation)
    return "yes_no" if answer in {"yes", "no"} else "freeform"


def first_video(record: dict[str, Any]) -> str:
    videos = record.get("videos")
    if not isinstance(videos, list) or len(videos) != 1 or not isinstance(videos[0], str):
        raise ValueError("sample requires exactly one video")
    return videos[0]


def load_candidates(path: Path) -> list[Candidate]:
    candidates = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            candidates.append(
                Candidate(
                    line_number=line_number,
                    record=record,
                    answer_type=answer_type(record),
                    category=str(record.get("category") or "unknown"),
                    video=first_video(record),
                )
            )
    return candidates


def pick_candidate(
    candidates: Iterable[Candidate],
    used_videos: set[str],
) -> Candidate | None:
    for candidate in candidates:
        if candidate.video not in used_videos:
            return candidate
    return None


def select_type(
    candidates: list[Candidate],
    target: int,
    used_videos: set[str],
    rng: random.Random,
) -> list[Candidate]:
    by_category: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_category[candidate.category].append(candidate)
    for values in by_category.values():
        rng.shuffle(values)

    selected: list[Candidate] = []
    categories = sorted(by_category, key=lambda value: (-len(by_category[value]), value))
    if target >= len(categories):
        for category in categories:
            candidate = pick_candidate(by_category[category], used_videos)
            if candidate is None:
                continue
            selected.append(candidate)
            used_videos.add(candidate.video)

    selected_ids = {candidate.line_number for candidate in selected}
    remaining = [
        candidate
        for candidate in candidates
        if candidate.line_number not in selected_ids
    ]
    rng.shuffle(remaining)
    for candidate in remaining:
        if len(selected) >= target:
            break
        if candidate.video in used_videos:
            continue
        selected.append(candidate)
        used_videos.add(candidate.video)
    if len(selected) != target:
        raise ValueError(
            f"Unable to select {target} unique-video samples; selected {len(selected)}"
        )
    return selected


def temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def write_json(path: Path, value: Any) -> None:
    temporary = temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_size <= 1:
        raise ValueError("--sample-size must be greater than one")
    if not 0 < args.yes_no_ratio < 1:
        raise ValueError("--yes-no-ratio must be between zero and one")
    for path in (args.output_jsonl, args.report_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite.")

    candidates = load_candidates(args.input_jsonl)
    by_type: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_type[candidate.answer_type].append(candidate)

    yes_no_target = round(args.sample_size * args.yes_no_ratio)
    freeform_target = args.sample_size - yes_no_target
    rng = random.Random(args.seed)
    used_videos: set[str] = set()
    selected = select_type(by_type["yes_no"], yes_no_target, used_videos, rng)
    selected.extend(select_type(by_type["freeform"], freeform_target, used_videos, rng))
    rng.shuffle(selected)

    temporary = temporary_sibling(args.output_jsonl)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for candidate in selected:
                stream.write(
                    json.dumps(candidate.record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
        os.replace(temporary, args.output_jsonl)
    finally:
        if temporary.exists():
            temporary.unlink()

    report = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "sample_size": len(selected),
        "seed": args.seed,
        "yes_no_ratio": args.yes_no_ratio,
        "unique_videos": len(used_videos),
        "source_rows": len(candidates),
        "source_answer_types": dict(Counter(value.answer_type for value in candidates)),
        "sample_answer_types": dict(Counter(value.answer_type for value in selected)),
        "sample_categories": dict(sorted(Counter(value.category for value in selected).items())),
        "source_line_numbers": [value.line_number for value in selected],
        "sample_ids": [value.record.get("id") for value in selected],
    }
    write_json(args.report_json, report)
    return report


def main() -> None:
    args = parse_args()
    args.input_jsonl = args.input_jsonl.expanduser().resolve()
    args.output_jsonl = args.output_jsonl.expanduser().resolve()
    args.report_json = args.report_json.expanduser().resolve()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
