#!/usr/bin/env python3
"""Remove synthetic RoboVQA reasoning and split complete videos atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from prepare_robovqa_swift import (
    ANSWER_PATTERN,
    clean_assistant_answer,
    clean_user_prompt,
    extract_assistant_answer,
    has_reasoning_markup,
    has_residual_reasoning,
    has_unclosed_final_answer_tag,
)
from split.grouped_jsonl_split import load_record, record_group, record_values


DEFAULT_DATA_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/robovqa")
DEFAULT_INPUT = DEFAULT_DATA_DIR / "robovqa_train_sft.jsonl"
DEFAULT_TRAIN = DEFAULT_DATA_DIR / "robovqa_train_sft_train.jsonl"
DEFAULT_EVAL = DEFAULT_DATA_DIR / "robovqa_train_sft_eval.jsonl"
DEFAULT_REPORT = DEFAULT_DATA_DIR / "robovqa_reasoning_cleanup_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract RoboVQA final answers, remove synthetic <think> text, and split "
            "complete video groups between train and eval."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=512,
        help="Maximum answer length after removing reasoning; 0 disables.",
    )
    parser.add_argument(
        "--overlong-answer-policy",
        choices=("compact", "drop", "error"),
        default="compact",
        help="compact retains complete leading/trailing sentences within the limit.",
    )
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.input_json = args.input_json.expanduser().resolve()
    args.train_output = args.train_output.expanduser().resolve()
    args.eval_output = args.eval_output.expanduser().resolve()
    args.report_output = args.report_output.expanduser().resolve()
    if not args.input_json.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {args.input_json}")
    if not 0 < args.eval_ratio < 1:
        raise ValueError("--eval-ratio must be in the range (0, 1)")
    if args.max_answer_chars < 0:
        raise ValueError("--max-answer-chars must be greater than or equal to zero")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be greater than or equal to zero")
    outputs = (args.train_output, args.eval_output, args.report_output)
    if len(set(outputs)) != len(outputs):
        raise ValueError("Train, eval, and report outputs must be different")
    if args.input_json in outputs:
        raise ValueError("Output paths must not overwrite the source JSONL")
    for output in outputs:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite.")


def temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def open_temporary(path: Path) -> TextIO:
    return temporary_sibling(path).open("w", encoding="utf-8", newline="\n")


def stable_is_eval(group: str, eval_ratio: float, seed: int) -> bool:
    threshold = int(eval_ratio * (1 << 64))
    digest = hashlib.blake2b(
        f"{seed}\0{group}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") < threshold


def percentile(histogram: Counter[int], quantile: float) -> int:
    total = sum(histogram.values())
    if not total:
        return 0
    target = max(1, int(total * quantile + 0.999999))
    seen = 0
    for length, count in sorted(histogram.items()):
        seen += count
        if seen >= target:
            return length
    return max(histogram)


def clean_record(
    record: dict[str, Any],
    max_answer_chars: int,
    overlong_answer_policy: str,
    stats: Counter[str],
) -> dict[str, Any] | None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        stats["skipped_invalid_messages"] += 1
        return None

    cleaned_messages: list[dict[str, Any]] = []
    record_had_reasoning = False
    record_prompt_cleaned = False
    for message in messages:
        if not isinstance(message, dict):
            stats["skipped_invalid_messages"] += 1
            return None
        role = str(message.get("role") or "").strip()
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            stats["skipped_invalid_messages"] += 1
            return None

        output_message = dict(message)
        if role == "user":
            cleaned = clean_user_prompt(content)
            record_prompt_cleaned = record_prompt_cleaned or cleaned != content.strip()
            output_message["content"] = cleaned
        elif role == "assistant":
            stats["assistant_messages_seen"] += 1
            stats["assistant_chars_before"] += len(content)
            stats["assistant_max_chars_before"] = max(
                stats["assistant_max_chars_before"], len(content)
            )
            stats.setdefault("before_histogram", Counter())[len(content)] += 1
            had_reasoning = has_reasoning_markup(content) or has_residual_reasoning(content)
            record_had_reasoning = record_had_reasoning or had_reasoning
            if ANSWER_PATTERN.search(content):
                stats["answer_tags_extracted"] += 1
            unclosed_answer = has_unclosed_final_answer_tag(content)
            if unclosed_answer:
                stats["unclosed_answer_tags_seen"] += 1
            try:
                extracted = extract_assistant_answer(content)
                is_overlong = bool(
                    max_answer_chars and len(extracted) > max_answer_chars
                )
                if is_overlong:
                    stats["overlong_answers_seen"] += 1
                    if overlong_answer_policy == "drop":
                        stats["skipped_overlong_answer"] += 1
                        return None
                cleaned = clean_assistant_answer(
                    content, max_answer_chars, overlong_answer_policy
                )
                if unclosed_answer:
                    stats["unclosed_answer_tags_recovered"] += 1
                if is_overlong and overlong_answer_policy == "compact":
                    stats["compacted_overlong_answer"] += 1
            except ValueError as error:
                if "exceeds --max-answer-chars" in str(error):
                    stats["skipped_overlong_answer"] += 1
                elif "contains residual reasoning" in str(error):
                    stats["skipped_residual_reasoning"] += 1
                else:
                    stats["skipped_empty_answer"] += 1
                return None
            stats["assistant_chars_after"] += len(cleaned)
            stats["assistant_max_chars_after"] = max(
                stats["assistant_max_chars_after"], len(cleaned)
            )
            stats.setdefault("after_histogram", Counter())[len(cleaned)] += 1
            output_message["content"] = cleaned
        cleaned_messages.append(output_message)

    if not any(message.get("role") == "user" for message in cleaned_messages):
        stats["skipped_invalid_messages"] += 1
        return None
    if not any(message.get("role") == "assistant" for message in cleaned_messages):
        stats["skipped_invalid_messages"] += 1
        return None
    if record_had_reasoning:
        stats["records_with_reasoning_removed"] += 1
    if record_prompt_cleaned:
        stats["records_with_prompt_rewritten"] += 1

    output = dict(record)
    output["messages"] = cleaned_messages
    return output


def length_report(stats: Counter[str], suffix: str) -> dict[str, float | int]:
    histogram = stats.get(f"{suffix}_histogram", Counter())
    count = sum(histogram.values())
    chars = stats[f"assistant_chars_{suffix}"]
    return {
        "messages": count,
        "mean_chars": round(chars / count, 2) if count else 0,
        "p50_chars": percentile(histogram, 0.50),
        "p95_chars": percentile(histogram, 0.95),
        "p99_chars": percentile(histogram, 0.99),
        "max_chars": stats[f"assistant_max_chars_{suffix}"],
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
    train_stream = open_temporary(args.train_output)
    eval_stream = open_temporary(args.eval_output)
    stats: Counter[str] = Counter()
    train_groups: set[str] = set()
    eval_groups: set[str] = set()
    train_videos: set[str] = set()
    eval_videos: set[str] = set()
    try:
        with args.input_json.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                stats["input_rows"] += 1
                record = load_record(line, args.input_json, line_number)
                cleaned = clean_record(
                    record,
                    args.max_answer_chars,
                    args.overlong_answer_policy,
                    stats,
                )
                if cleaned is None:
                    continue
                try:
                    group = record_group(cleaned, "videos", args.input_json, None)
                    videos = record_values(cleaned, "videos", args.input_json)
                except ValueError as error:
                    raise ValueError(f"{args.input_json}:{line_number}: {error}") from error
                output_line = json.dumps(
                    cleaned, ensure_ascii=False, separators=(",", ":")
                ) + "\n"
                if stable_is_eval(group, args.eval_ratio, args.seed):
                    eval_stream.write(output_line)
                    stats["eval_rows"] += 1
                    eval_groups.add(group)
                    eval_videos.update(videos)
                else:
                    train_stream.write(output_line)
                    stats["train_rows"] += 1
                    train_groups.add(group)
                    train_videos.update(videos)
                stats["output_rows"] += 1
                if args.progress_every and stats["input_rows"] % args.progress_every == 0:
                    print(
                        f"[curate] input={stats['input_rows']:,} output={stats['output_rows']:,} "
                        f"train={stats['train_rows']:,} eval={stats['eval_rows']:,}",
                        flush=True,
                    )

        group_overlap = train_groups & eval_groups
        video_overlap = train_videos & eval_videos
        if group_overlap or video_overlap:
            raise ValueError(
                f"Leakage detected: group_overlap={len(group_overlap):,} "
                f"video_overlap={len(video_overlap):,}"
            )
        if not stats["train_rows"] or not stats["eval_rows"]:
            raise ValueError("Curation must leave at least one row in train and eval")

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
        "reasoning_policy": "answer-only",
        "max_answer_chars": args.max_answer_chars,
        "overlong_answer_policy": args.overlong_answer_policy,
        "seed": args.seed,
        "requested_eval_ratio": args.eval_ratio,
        "actual_eval_ratio": stats["eval_rows"] / stats["output_rows"],
        "rows": {
            key: stats[key]
            for key in (
                "input_rows",
                "output_rows",
                "train_rows",
                "eval_rows",
                "skipped_overlong_answer",
                "skipped_empty_answer",
                "skipped_residual_reasoning",
                "skipped_invalid_messages",
            )
        },
        "cleanup": {
            "records_with_reasoning_removed": stats["records_with_reasoning_removed"],
            "records_with_prompt_rewritten": stats["records_with_prompt_rewritten"],
            "answer_tags_extracted": stats["answer_tags_extracted"],
            "unclosed_answer_tags_seen": stats["unclosed_answer_tags_seen"],
            "unclosed_answer_tags_recovered": stats["unclosed_answer_tags_recovered"],
            "overlong_answers_seen": stats["overlong_answers_seen"],
            "compacted_overlong_answer": stats["compacted_overlong_answer"],
        },
        "assistant_length_before": length_report(stats, "before"),
        "assistant_length_after": length_report(stats, "after"),
        "groups": {
            "train_videos": len(train_videos),
            "eval_videos": len(eval_videos),
            "video_overlap": 0,
            "group_overlap": 0,
        },
    }
    write_report(args.report_output, report)
    return report


def main() -> None:
    args = parse_args()
    validate_args(args)
    report = run(args)
    rows = report["rows"]
    before = report["assistant_length_before"]
    after = report["assistant_length_after"]
    print(
        f"[done] input={rows['input_rows']:,} output={rows['output_rows']:,} "
        f"train={rows['train_rows']:,} eval={rows['eval_rows']:,}"
    )
    print(
        f"[length] mean={before['mean_chars']}->{after['mean_chars']} "
        f"p99={before['p99_chars']}->{after['p99_chars']} "
        f"max={before['max_chars']}->{after['max_chars']}"
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
