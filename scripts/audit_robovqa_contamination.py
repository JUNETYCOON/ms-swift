#!/usr/bin/env python3
"""Audit assistant answers for residual synthetic reasoning text."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_robovqa_swift import extract_assistant_answer


SUSPICIOUS_PATTERNS = {
    "reasoning_tag": re.compile(r"</?(?:think|answer)\b", re.IGNORECASE),
    "format_instruction": re.compile(
        r"tags?\s+as\s+per\s+(?:the\s+)?requirements|"
        r"since\s+(?:it(?:'s|\s+is)|this\s+is)\s+(?:not\s+)?a\s+yes/no\s+question",
        re.IGNORECASE,
    ),
    "assistant_meta_reasoning": re.compile(
        r"\b(?:we|i)\s+(?:need|should|must)\s+(?:to\s+)?(?:answer|respond|provide)\b|"
        r"\bthe\s+(?:user|question)\s+(?:asks|is\s+asking)\b",
        re.IGNORECASE,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--check-cleaner", action="store_true")
    return parser.parse_args()


def assistant_contents(record: dict[str, Any]) -> list[str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    return [
        str(message.get("content"))
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
    ]


def main() -> None:
    args = parse_args()
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    examples_by_pattern: dict[str, list[dict[str, Any]]] = {
        name: [] for name in SUSPICIOUS_PATTERNS
    }
    tagless_examples: list[dict[str, Any]] = []
    cleaner_failure_examples: list[dict[str, Any]] = []
    for path in args.jsonl:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                counts["rows"] += 1
                record = json.loads(line)
                matched: set[str] = set()
                contents = assistant_contents(record)
                if args.check_cleaner:
                    cleaner_failed = False
                    for content in contents:
                        try:
                            extract_assistant_answer(content)
                        except ValueError as error:
                            cleaner_failed = True
                            counts["cleaner_failures"] += 1
                            if len(cleaner_failure_examples) < args.max_examples:
                                cleaner_failure_examples.append(
                                    {
                                        "path": str(path),
                                        "line_number": line_number,
                                        "id": record.get("id"),
                                        "error": str(error),
                                        "assistant": content,
                                    }
                                )
                    if cleaner_failed:
                        counts["cleaner_failure_rows"] += 1
                for content in contents:
                    for name, pattern in SUSPICIOUS_PATTERNS.items():
                        if pattern.search(content):
                            matched.add(name)
                if not matched:
                    continue
                counts["contaminated_rows"] += 1
                if "reasoning_tag" not in matched:
                    counts["tagless_contaminated_rows"] += 1
                for name in matched:
                    counts[name] += 1
                example = {
                    "path": str(path),
                    "line_number": line_number,
                    "id": record.get("id"),
                    "patterns": sorted(matched),
                    "assistant": contents[-1] if contents else "",
                }
                if len(examples) < args.max_examples:
                    examples.append(example)
                if (
                    "reasoning_tag" not in matched
                    and len(tagless_examples) < args.max_examples
                ):
                    tagless_examples.append(example)
                for name in matched:
                    if len(examples_by_pattern[name]) < args.max_examples:
                        examples_by_pattern[name].append(example)
    report = {
        "rows": counts["rows"],
        "contaminated_rows": counts["contaminated_rows"],
        "tagless_contaminated_rows": counts["tagless_contaminated_rows"],
        "patterns": {
            name: counts[name] for name in SUSPICIOUS_PATTERNS
        },
        "examples": examples,
        "examples_by_pattern": examples_by_pattern,
        "tagless_examples": tagless_examples,
        "cleaner_failure_rows": counts["cleaner_failure_rows"],
        "cleaner_failures": counts["cleaner_failures"],
        "cleaner_failure_examples": cleaner_failure_examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
