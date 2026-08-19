#!/usr/bin/env python3
"""Convert AI2D annotations/questions to ms-swift multimodal JSONL.

Default output is SFT-style ms-swift data:

{"messages": [{"role": "user", "content": "<image>..."}, {"role": "assistant", "content": "..."}],
 "images": ["/abs/path/to/image.png"]}

Expected layout:

/mnt/luojunkun/stage1/dataset/ai2d/ai2d/
  annotations/*.json
  questions/*.json
  images/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
TEXT_KEYS = {"text", "label", "value", "content", "name"}
QUESTION_KEYS = ("question", "query", "prompt", "instruction")
CHOICE_KEYS = ("answerTexts", "answers", "choices", "options", "choiceTexts")
ANSWER_KEYS = ("correctAnswer", "correct_answer", "answer", "label", "target")
IMAGE_NAME_KEYS = ("image", "imageName", "image_name", "image_path", "file_name", "filename", "diagramName")

DEFAULT_AI2D_ROOT = Path("/mnt/luojunkun/stage1/dataset/ai2d/ai2d")
DEFAULT_OUTPUT = Path("/mnt/luojunkun/stage1/dataset_ms-swift/ai2d/ai2d_sft_msswift.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AI2D annotations/questions to ms-swift multimodal JSONL.")
    parser.add_argument("--ai2d-root", type=Path, default=DEFAULT_AI2D_ROOT, help="AI2D dataset root.")
    parser.add_argument("--annotations-dir", type=Path, default=None, help="Default: <root>/annotations.")
    parser.add_argument("--questions-dir", type=Path, default=None, help="Default: <root>/questions.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Default: <root>/images.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path.")
    parser.add_argument(
        "--mode",
        choices=("pretrain", "sft"),
        default="sft",
        help="sft writes user/assistant rows; pretrain writes assistant-only rows.",
    )
    parser.add_argument(
        "--sft-question-template",
        default="<image>{question}\nChoices: {choices}",
        help="User prompt template in SFT mode.",
    )
    parser.add_argument(
        "--pretrain-template",
        default="<image>\nQuestion: {question}\nChoices: {choices}\nAnswer: {answer}",
        help="Assistant-only template in pretrain mode.",
    )
    parser.add_argument("--fallback-prompt", default="<image>Describe the diagram.")
    parser.add_argument("--fallback-template", default="<image>\n{texts}")
    parser.add_argument("--image-prefix", default=None, help="Optional rewrite prefix for image paths.")
    parser.add_argument("--relative-paths", action="store_true", help="Write image paths relative to output JSONL.")
    parser.add_argument("--keep-id", action="store_true", help="Keep sample ids in output rows.")
    parser.add_argument("--keep-empty", action="store_true", help="Keep fallback rows with empty text.")
    parser.add_argument("--limit", type=int, default=0, help="Limit written rows for quick checks.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return " ".join(str(value).strip().split())
    if isinstance(value, dict):
        for key in ("text", "label", "value", "content", "name"):
            text = clean_text(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return "; ".join(part for item in value if (part := clean_text(item)))
    return " ".join(str(value).strip().split())


def text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        values = []
        for key in sorted(value):
            text = clean_text(value[key])
            if text:
                values.append(text)
        return values
    if isinstance(value, (list, tuple)):
        return [text for item in value if (text := clean_text(item))]
    text = clean_text(value)
    return [text] if text else []


def first_value(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple, dict)) and not value:
            continue
        return value
    return None


def build_image_index(image_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not image_dir.is_dir():
        return index
    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        resolved = path.resolve()
        relative = path.relative_to(image_dir)
        keys = {path.name, path.stem, relative.as_posix(), relative.with_suffix("").as_posix()}
        for key in keys:
            index.setdefault(key, resolved)
    return index


def candidate_image_names(annotation: dict[str, Any], annotation_path: Path) -> list[str]:
    names = []
    file_name = annotation_path.name
    if file_name.endswith(".json"):
        file_name = file_name[: -len(".json")]
    names.extend([file_name, Path(file_name).stem, f"{annotation_path.stem}.png"])
    for key in IMAGE_NAME_KEYS:
        value = clean_text(annotation.get(key))
        if not value:
            continue
        path = Path(value.replace("\\", "/"))
        names.extend([value.replace("\\", "/"), path.name, path.stem])
    expanded = []
    for name in names:
        expanded.append(name)
        if not Path(name).suffix:
            expanded.extend(f"{name}{suffix}" for suffix in IMAGE_EXTS)
    return [name for name in dict.fromkeys(expanded) if name]


def resolve_image(annotation: dict[str, Any], annotation_path: Path, image_index: dict[str, Path]) -> Path | None:
    for name in candidate_image_names(annotation, annotation_path):
        image_path = image_index.get(name)
        if image_path is not None:
            return image_path
    return None


def build_question_index(questions_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not questions_dir.is_dir():
        return index
    for path in questions_dir.rglob("*.json"):
        name = path.name
        if name.endswith(".json"):
            name = name[: -len(".json")]
        keys = {
            path.name,
            path.stem,
            name,
            Path(name).stem,
            path.relative_to(questions_dir).as_posix(),
            path.relative_to(questions_dir).with_suffix("").as_posix(),
        }
        for key in keys:
            index.setdefault(key, path.resolve())
    return index


def resolve_question_path(annotation_path: Path, annotation: dict[str, Any], question_index: dict[str, Path]) -> Path | None:
    for name in candidate_image_names(annotation, annotation_path):
        candidates = [name, f"{name}.json"]
        if Path(name).suffix:
            candidates.append(f"{Path(name).stem}.json")
        for candidate in candidates:
            path = question_index.get(candidate)
            if path is not None:
                return path
    return None


def output_image_path(image_path: Path, jsonl_path: Path, args: argparse.Namespace) -> str:
    if args.image_prefix:
        return f"{args.image_prefix.rstrip('/')}/{image_path.name}"
    if args.relative_paths:
        return os.path.relpath(image_path, jsonl_path.parent).replace(os.sep, "/")
    return str(image_path.resolve())


def collect_texts(data: Any) -> list[str]:
    texts: list[str] = []

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                if child_key == "questions":
                    continue
                visit(child_value, child_key)
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key)
            return
        if key in TEXT_KEYS:
            text = clean_text(value)
            if text:
                texts.append(text)

    visit(data)
    return list(dict.fromkeys(texts))


def iter_question_items(annotation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    questions = annotation.get("questions")
    if isinstance(questions, dict):
        return [(str(question_id), question) for question_id, question in questions.items() if isinstance(question, dict)]
    if isinstance(questions, list):
        return [
            (clean_text(question.get("questionId")) or str(index), question)
            for index, question in enumerate(questions)
            if isinstance(question, dict)
        ]
    if any(key in annotation for key in QUESTION_KEYS):
        return [("0", annotation)]
    return []


def choices_for_question(question: dict[str, Any]) -> list[str]:
    for key in CHOICE_KEYS:
        choices = text_list(question.get(key))
        if choices:
            return choices
    return []


def answer_for_question(question: dict[str, Any], choices: list[str]) -> str:
    raw_answer = first_value(question, ANSWER_KEYS)
    if raw_answer is None:
        return ""
    if choices:
        if isinstance(raw_answer, int) and 0 <= raw_answer < len(choices):
            return choices[raw_answer]
        answer_text = clean_text(raw_answer)
        if answer_text.isdigit():
            index = int(answer_text)
            if 0 <= index < len(choices):
                return choices[index]
            if 1 <= index <= len(choices):
                return choices[index - 1]
        if len(answer_text) == 1 and answer_text.upper().isalpha():
            index = ord(answer_text.upper()) - ord("A")
            if 0 <= index < len(choices):
                return choices[index]
        if answer_text:
            return answer_text
    return clean_text(raw_answer)


def question_text(question: dict[str, Any]) -> str:
    return clean_text(first_value(question, QUESTION_KEYS))


def choice_text(choices: list[str]) -> str:
    if not choices:
        return ""
    labels = [chr(ord("A") + index) for index in range(len(choices))]
    return " ".join(f"{label}. {choice}" for label, choice in zip(labels, choices))


def build_question_record(
    annotation_path: Path,
    question_id: str,
    question: dict[str, Any],
    image_path: Path,
    jsonl_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    query = question_text(question)
    choices = choices_for_question(question)
    answer = answer_for_question(question, choices)
    if not query or not answer:
        return None

    image_value = output_image_path(image_path, jsonl_path, args)
    if args.mode == "pretrain":
        content = args.pretrain_template.format(question=query, choices=choice_text(choices), answer=answer).strip()
        record: dict[str, Any] = {
            "messages": [{"role": "assistant", "content": content}],
            "images": [image_value],
        }
    else:
        record = {
            "messages": [
                {"role": "user", "content": args.sft_question_template.format(question=query, choices=choice_text(choices))},
                {"role": "assistant", "content": answer},
            ],
            "images": [image_value],
        }
    if args.keep_id:
        record["id"] = f"{annotation_path.stem}:{question_id}"
    return record


def build_fallback_record(
    annotation_path: Path,
    annotation: dict[str, Any],
    image_path: Path,
    jsonl_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    texts = collect_texts(annotation)
    if not texts and not args.keep_empty:
        return None
    answer = "\n".join(texts)
    if args.mode == "pretrain":
        record: dict[str, Any] = {
            "messages": [{"role": "assistant", "content": args.fallback_template.format(texts=answer).strip()}],
            "images": [output_image_path(image_path, jsonl_path, args)],
        }
    else:
        record = {
            "messages": [
                {"role": "user", "content": args.fallback_prompt},
                {"role": "assistant", "content": answer},
            ],
            "images": [output_image_path(image_path, jsonl_path, args)],
        }
    if args.keep_id:
        record["id"] = annotation_path.name
    return record


def convert(args: argparse.Namespace) -> Counter:
    ai2d_root = args.ai2d_root.expanduser().resolve()
    annotations_dir = (args.annotations_dir or ai2d_root / "annotations").expanduser().resolve()
    questions_dir = (args.questions_dir or ai2d_root / "questions").expanduser().resolve()
    image_dir = (args.image_dir or ai2d_root / "images").expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotations_dir}")
    if not questions_dir.is_dir():
        raise FileNotFoundError(f"Question directory does not exist: {questions_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    image_index = build_image_index(image_dir)
    question_index = build_question_index(questions_dir)
    annotation_files = sorted(annotations_dir.rglob("*.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter(
        annotation_files=len(annotation_files),
        question_files=len(set(question_index.values())),
        indexed_images=len(set(image_index.values())),
    )
    print(f"[read] annotations={annotations_dir} files={len(annotation_files):,}")
    print(f"[read] questions={questions_dir} files={stats['question_files']:,}")
    print(f"[read] images={image_dir} indexed={stats['indexed_images']:,}")
    print(f"[write] output={output_path}")

    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for annotation_path in annotation_files:
            stats["files_seen"] += 1
            try:
                annotation = read_json(annotation_path)
            except Exception as exc:
                stats[f"bad_json:{type(exc).__name__}"] += 1
                continue
            if not isinstance(annotation, dict):
                stats["skip_not_object"] += 1
                continue

            image_path = resolve_image(annotation, annotation_path, image_index)
            if image_path is None:
                stats["skip_no_image"] += 1
                continue

            question_doc = annotation
            question_path = resolve_question_path(annotation_path, annotation, question_index)
            if question_path is not None:
                try:
                    question_doc = read_json(question_path)
                    stats["question_files_read"] += 1
                except Exception as exc:
                    stats[f"bad_question_json:{type(exc).__name__}"] += 1
                    question_doc = annotation
            else:
                stats["missing_question_file"] += 1

            question_items = iter_question_items(question_doc) if isinstance(question_doc, dict) else []
            for question_id, question in question_items:
                record = build_question_record(annotation_path, question_id, question, image_path, output_path, args)
                if record is None:
                    stats["skip_bad_question"] += 1
                    continue
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["written"] += 1
                stats["question_rows"] += 1
                if args.limit and stats["written"] >= args.limit:
                    return stats

            if question_items:
                continue

            record = build_fallback_record(annotation_path, annotation, image_path, output_path, args)
            if record is None:
                stats["skip_no_text"] += 1
                continue
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["written"] += 1
            stats["fallback_rows"] += 1
            if args.limit and stats["written"] >= args.limit:
                return stats
    return stats


def validate_args(args: argparse.Namespace) -> None:
    if args.limit < 0:
        raise SystemExit("--limit must be greater than or equal to zero")
    if "<image>" not in args.pretrain_template:
        raise SystemExit("--pretrain-template must contain '<image>'")
    if args.sft_question_template.count("<image>") != 1:
        raise SystemExit("--sft-question-template must contain exactly one '<image>' token")
    if "<image>" not in args.fallback_prompt:
        raise SystemExit("--fallback-prompt must contain '<image>'")
    if "<image>" not in args.fallback_template:
        raise SystemExit("--fallback-template must contain '<image>'")


def main() -> None:
    args = parse_args()
    validate_args(args)
    stats = convert(args)
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
