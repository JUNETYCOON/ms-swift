#!/usr/bin/env python3
"""Convert local ChartQA data to ms-swift multimodal JSONL.

The default output is SFT-style ms-swift multimodal data:

{"messages": [{"role": "user", "content": "<image>..."}, {"role": "assistant", "content": "..."}],
 "images": ["/abs/path/to/chart.png"]}

Supported source layouts:

1. HuggingFace parquet layout:

/mnt/luojunkun/stage1/dataset/Chartqa/
  data/
    train-*.parquet
    val-*.parquet
    test-*.parquet

2. Original ChartQA JSON/image layout:

/mnt/luojunkun/stage1/dataset/Chartqa/
  ChartQA Dataset/
    train/train_human.json
    train/train_augmented.json
    train/png/*.png
    val/...
    test/...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_CHARTQA_ROOT = Path("/mnt/luojunkun/stage1/dataset/Chartqa")
DEFAULT_OUTPUT_DIR = Path("/mnt/luojunkun/stage1/dataset_ms-swift/chartqa")
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_SUBSETS = ("human", "augmented")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ChartQA JSON files or parquet shards to ms-swift multimodal JSONL."
    )
    parser.add_argument(
        "--chartqa-root",
        type=Path,
        default=DEFAULT_CHARTQA_ROOT,
        help=f"ChartQA dataset root. Default: {DEFAULT_CHARTQA_ROOT}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Parquet shard directory. Default: <chartqa-root>/data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--source-format",
        choices=("auto", "parquet", "json"),
        default="auto",
        help="auto uses parquet when <chartqa-root>/data/*.parquet exists, otherwise original JSON layout.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to convert. Accepted aliases include validation/valid/dev for val.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        choices=DEFAULT_SUBSETS,
        help="Original JSON-layout subsets to convert. Ignored in parquet mode.",
    )
    parser.add_argument(
        "--mode",
        choices=("pretrain", "sft"),
        default="sft",
        help="pretrain writes assistant-only samples; sft writes user/assistant QA samples.",
    )
    parser.add_argument(
        "--pretrain-template",
        default="<image>\nQuestion: {question}\nAnswer: {answer}",
        help="Template used when --mode pretrain.",
    )
    parser.add_argument(
        "--sft-question-template",
        default="<image>{question}",
        help="User prompt template used when --mode sft.",
    )
    parser.add_argument(
        "--include-empty-answer",
        action="store_true",
        help="Write samples without labels. By default they are skipped.",
    )
    parser.add_argument(
        "--jsonl-suffix",
        default="msswift",
        help="Suffix used in output JSONL filenames.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write image paths relative to each JSONL file instead of absolute paths.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy original-layout image files into --output-dir/--image-subdir and reference those copies.",
    )
    parser.add_argument(
        "--image-subdir",
        default="images",
        help="Image output subdirectory under --output-dir.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Rewrite copied/extracted image files even if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit valid written samples per split/source. Useful for quick checks.",
    )
    parser.add_argument("--batch-size", type=int, default=2048, help="Parquet read batch size.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print conversion progress every N source rows. Set 0 to disable.",
    )
    parser.add_argument(
        "--max-missing-image-logs",
        type=int,
        default=20,
        help="Maximum missing-image examples to print per split/source.",
    )
    return parser.parse_args()


def import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to read parquet files. Install it with: pip install pyarrow") from exc
    return pq


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(part for item in value if (part := clean_text(item)))
    if hasattr(value, "tolist"):
        return clean_text(value.tolist())
    return " ".join(str(value).strip().split())


def normalize_split(value: Any) -> str:
    text = str(value).strip().lower().replace("_", "-")
    split = SPLIT_ALIASES.get(text) or SPLIT_ALIASES.get(text.replace("-", ""))
    if not split:
        raise SystemExit(f"Unsupported split {value!r}. Known splits: {', '.join(DEFAULT_SPLITS)}")
    return split


def safe_stem(value: Any, fallback: str) -> str:
    stem = Path(clean_text(value)).stem or fallback
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or fallback


def load_json_rows(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as json_file:
        rows = json.load(json_file)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {json_path}")
    return rows


def iter_parquet_rows(files: Iterable[Path], batch_size: int) -> Iterator[dict[str, Any]]:
    pq = import_pyarrow_parquet()
    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def validate_args(args: argparse.Namespace) -> None:
    args.chartqa_root = args.chartqa_root.expanduser().resolve()
    args.data_dir = (args.data_dir or args.chartqa_root / "data").expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.splits = [normalize_split(split) for split in args.splits]

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be greater than or equal to zero")
    if args.max_missing_image_logs < 0:
        raise SystemExit("--max-missing-image-logs must be greater than or equal to zero")

    if args.mode == "pretrain":
        for token in ("<image>", "{question}", "{answer}"):
            if token not in args.pretrain_template:
                raise SystemExit(f"--pretrain-template must contain {token!r}")
        if args.pretrain_template.count("<image>") != 1:
            raise SystemExit("--pretrain-template must contain exactly one '<image>' token")
    else:
        for token in ("<image>", "{question}"):
            if token not in args.sft_question_template:
                raise SystemExit(f"--sft-question-template must contain {token!r}")
        if args.sft_question_template.count("<image>") != 1:
            raise SystemExit("--sft-question-template must contain exactly one '<image>' token")


def image_path_for_json(image_path: Path, jsonl_path: Path, relative: bool) -> str:
    if relative:
        return os.path.relpath(image_path, jsonl_path.parent).replace(os.sep, "/")
    return str(image_path)


def question_text(row: dict[str, Any]) -> str:
    for key in ("query", "question", "Question"):
        text = clean_text(row.get(key))
        if text:
            return text
    return ""


def answer_text(row: dict[str, Any]) -> str:
    for key in ("label", "answer", "answers", "Answer"):
        text = clean_text(row.get(key))
        if text:
            return text
    return ""


def build_record(
    row: dict[str, Any],
    image_path: Path,
    jsonl_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    question = question_text(row)
    answer = answer_text(row)
    if not question:
        return None
    if not answer and not args.include_empty_answer:
        return None

    image_value = image_path_for_json(image_path, jsonl_path, args.relative_paths)
    if args.mode == "sft":
        if not answer:
            return None
        return {
            "messages": [
                {"role": "user", "content": args.sft_question_template.format(question=question, answer=answer)},
                {"role": "assistant", "content": answer},
            ],
            "images": [image_value],
        }

    content = args.pretrain_template.format(question=question, answer=answer)
    return {
        "messages": [{"role": "assistant", "content": content}],
        "images": [image_value],
    }


def source_image_path(row: dict[str, Any], image_dir: Path) -> Path | None:
    imgname = clean_text(row.get("imgname") or row.get("image_path") or row.get("image"))
    if not imgname:
        return None

    image_path = Path(imgname)
    if image_path.is_absolute():
        return image_path

    candidate = image_dir / image_path
    if candidate.exists():
        return candidate

    basename_candidate = image_dir / image_path.name
    if basename_candidate.exists():
        return basename_candidate

    return candidate


def output_image_path_for_source(source_path: Path, split: str, subset: str, args: argparse.Namespace) -> Path:
    return args.output_dir / args.image_subdir / split / subset / source_path.name


def prepare_source_image_path(
    source_path: Path,
    split: str,
    subset: str,
    args: argparse.Namespace,
    stats: Counter,
) -> Path:
    source_path = source_path.resolve()
    if not args.copy_images:
        return source_path

    target_path = output_image_path_for_source(source_path, split, subset, args).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_images or not target_path.exists():
        shutil.copy2(source_path, target_path)
        stats["images_copied"] += 1
    else:
        stats["images_reused"] += 1
    return target_path


def convert_json_source(split: str, subset: str, args: argparse.Namespace) -> Counter:
    split_dir = args.chartqa_root / "ChartQA Dataset" / split
    json_path = split_dir / f"{split}_{subset}.json"
    image_dir = split_dir / "png"
    jsonl_path = args.output_dir / f"chartqa_{split}_{subset}_{args.mode}_{args.jsonl_suffix}.jsonl"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing source JSON: {json_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    rows = load_json_rows(json_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter(source_rows=len(rows))
    missing_logs = 0
    print(f"[json {split}/{subset}] writing {jsonl_path}")

    with jsonl_path.open("w", encoding="utf-8") as out_file:
        for row in rows:
            stats["rows_seen"] += 1
            if args.progress_every and stats["rows_seen"] % args.progress_every == 0:
                print(
                    f"[json {split}/{subset}] rows={stats['rows_seen']} "
                    f"written={stats['written']} skipped={stats['skipped']}"
                )

            image_path = source_image_path(row, image_dir)
            if image_path is None or not image_path.exists():
                stats["missing_image"] += 1
                stats["skipped"] += 1
                if missing_logs < args.max_missing_image_logs:
                    print(f"[json {split}/{subset}] missing image: {clean_text(row.get('imgname'))}")
                    missing_logs += 1
                continue

            prepared_image_path = prepare_source_image_path(image_path, split, subset, args, stats)
            record = build_record(row, prepared_image_path, jsonl_path, args)
            if record is None:
                stats["skipped_invalid_record"] += 1
                stats["skipped"] += 1
                continue

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            if args.limit is not None and stats["written"] >= args.limit:
                break

    write_stats(jsonl_path, stats)
    print(f"[json {split}/{subset}] done: written={stats['written']}")
    return stats


def parquet_files_for_split(data_dir: Path, split: str) -> list[Path]:
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split={split} below {data_dir}")
    return files


def image_payload(row: dict[str, Any]) -> dict[str, Any]:
    image = row.get("image")
    return image if isinstance(image, dict) else {}


def image_bytes(row: dict[str, Any]) -> bytes:
    data = image_payload(row).get("bytes")
    if data is None:
        return b""
    if isinstance(data, memoryview):
        return data.tobytes()
    return bytes(data)


def image_path_text(row: dict[str, Any]) -> str:
    payload = image_payload(row)
    image = row.get("image")
    image_text = image if isinstance(image, str) else None
    return clean_text(
        payload.get("path")
        or row.get("imgname")
        or row.get("image_path")
        or row.get("image_file")
        or image_text
    )


def infer_suffix(path_text: str, data: bytes) -> str:
    suffix = Path(path_text).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return ".jpg" if suffix == ".jpeg" else suffix
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if data.startswith(b"BM"):
        return ".bmp"
    return ".png"


def direct_image_candidates(path_text: str, chartqa_root: Path) -> list[Path]:
    if not path_text:
        return []
    path = Path(path_text)
    if path.is_absolute():
        return [path]
    return [
        chartqa_root / path,
        chartqa_root / "images" / path,
        chartqa_root / "ChartQA Dataset" / path,
        chartqa_root / path.name,
    ]


def resolve_direct_image(path_text: str, chartqa_root: Path) -> Path | None:
    for candidate in direct_image_candidates(path_text, chartqa_root):
        if candidate.exists():
            return candidate.resolve()
    return None


def parquet_output_image_path(row: dict[str, Any], split: str, row_index: int, args: argparse.Namespace) -> Path:
    path_text = image_path_text(row)
    data = image_bytes(row)
    suffix = infer_suffix(path_text, data)
    filename = Path(path_text).name if Path(path_text).suffix.lower() in IMAGE_SUFFIXES else ""
    if not filename:
        fallback = f"{split}_{row_index:08d}"
        filename = f"{safe_stem(row.get('imgname') or row.get('image_id'), fallback)}{suffix}"
    return (args.output_dir / args.image_subdir / split / filename).resolve()


def prepare_parquet_image_path(
    row: dict[str, Any],
    split: str,
    row_index: int,
    args: argparse.Namespace,
    stats: Counter,
) -> Path | None:
    path_text = image_path_text(row)
    data = image_bytes(row)
    direct_path = resolve_direct_image(path_text, args.chartqa_root)
    target_path = parquet_output_image_path(row, split, row_index, args)

    if data:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite_images or not target_path.exists():
            target_path.write_bytes(data)
            stats["images_written"] += 1
        else:
            stats["images_reused"] += 1
        return target_path

    if direct_path is not None:
        if not args.copy_images:
            stats["images_referenced"] += 1
            return direct_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite_images or not target_path.exists():
            shutil.copy2(direct_path, target_path)
            stats["images_copied"] += 1
        else:
            stats["images_reused"] += 1
        return target_path

    if target_path.exists():
        stats["images_reused"] += 1
        return target_path

    stats["missing_image"] += 1
    return None


def convert_parquet_split(split: str, args: argparse.Namespace) -> Counter:
    files = parquet_files_for_split(args.data_dir, split)
    jsonl_path = args.output_dir / f"chartqa_{split}_{args.mode}_{args.jsonl_suffix}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter(source_shards=len(files))
    missing_logs = 0
    print(f"[parquet {split}] shards={len(files)} writing {jsonl_path}")

    with jsonl_path.open("w", encoding="utf-8") as out_file:
        for row in iter_parquet_rows(files, args.batch_size):
            stats["rows_seen"] += 1
            if args.progress_every and stats["rows_seen"] % args.progress_every == 0:
                print(
                    f"[parquet {split}] rows={stats['rows_seen']} "
                    f"written={stats['written']} skipped={stats['skipped']}"
                )

            image_path = prepare_parquet_image_path(row, split, stats["rows_seen"], args, stats)
            if image_path is None:
                stats["skipped"] += 1
                if missing_logs < args.max_missing_image_logs:
                    print(f"[parquet {split}] missing image: {image_path_text(row)}")
                    missing_logs += 1
                continue

            record = build_record(row, image_path, jsonl_path, args)
            if record is None:
                stats["skipped_invalid_record"] += 1
                stats["skipped"] += 1
                continue

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            if args.limit is not None and stats["written"] >= args.limit:
                break

    write_stats(jsonl_path, stats)
    print(f"[parquet {split}] done: written={stats['written']}")
    return stats


def write_stats(jsonl_path: Path, stats: Counter) -> None:
    stats_path = jsonl_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as stats_file:
        json.dump(dict(stats), stats_file, ensure_ascii=False, indent=2, sort_keys=True)
        stats_file.write("\n")


def detect_source_format(args: argparse.Namespace) -> str:
    if args.source_format != "auto":
        return args.source_format
    if args.data_dir.exists() and any(args.data_dir.glob("*.parquet")):
        return "parquet"
    return "json"


def main() -> None:
    args = parse_args()
    validate_args(args)

    source_format = detect_source_format(args)
    print(f"[info] chartqa_root={args.chartqa_root}")
    print(f"[info] output_dir={args.output_dir}")
    print(f"[info] source_format={source_format}")
    if source_format == "parquet":
        print(f"[info] data_dir={args.data_dir}")

    total: Counter = Counter()
    if source_format == "parquet":
        for split in args.splits:
            total.update(convert_parquet_split(split, args))
    else:
        for split in args.splits:
            for subset in args.subsets:
                total.update(convert_json_source(split, subset, args))

    print("[total]", json.dumps(dict(total), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
