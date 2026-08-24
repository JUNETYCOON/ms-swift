#!/usr/bin/env python3
"""Resplit Visual Genome so GQA-shared images stay in train.

GQA and Visual Genome reuse the same photographs. The previous global eval
reservation dropped VG train rows whose image also appeared in GQA val. This
script rebuilds VG QA and region splits from the original train+val pool:

- Images present in GQA stay in each VG train file.
- Images unique to Visual Genome are hashed into val with ``val_ratio``.
- GQA official train/val files are left unchanged.

Example:

    python scripts/data-process/split/resplit_visualgenome_gqa_shared.py \
        --data-root /mnt/pengtaijun/vlm_fm/dataset/dataset_ms-swift \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from grouped_jsonl_split import image_stem_group, split_jsonl


DEFAULT_DATA_ROOT = Path("/mnt/pengtaijun/vlm_fm/dataset/dataset_ms-swift")
GQA_DIR_NAMES = ("gqa", "GQA")
VG_DIR_NAMES = ("visualgenome", "VisualGenome", "visual_genome")
GQA_PAIR_NAMES = (
    (
        "gqa_train_balanced_sft_msswift.jsonl",
        "gqa_val_balanced_sft_msswift.jsonl",
    ),
    (
        "gqa_train_balanced_sft_msswift_sanitized_source.jsonl",
        "gqa_val_balanced_sft_msswift_ready_eval.jsonl",
    ),
    (
        "gqa_train_balanced_sft_msswift_dlc_train.jsonl",
        "gqa_val_balanced_sft_msswift.jsonl",
    ),
)
VG_TASKS = (
    {
        "name": "visualgenome-qa",
        "inputs": (
            "visualgenome_qa_train.jsonl",
            "visualgenome_qa_val.jsonl",
        ),
        "train": "visualgenome_qa_train.jsonl",
        "eval": "visualgenome_qa_val.jsonl",
        "report": "visualgenome_qa_gqa_shared_split_report.json",
        "sync_train": (
            "visualgenome_qa_dlc_train.jsonl",
            "visualgenome_qa_sanitized_source.jsonl",
        ),
    },
    {
        "name": "visualgenome-regions",
        "inputs": (
            "visualgenome_regions_train.jsonl",
            "visualgenome_regions_val.jsonl",
        ),
        "train": "visualgenome_regions_train.jsonl",
        "eval": "visualgenome_regions_val.jsonl",
        "report": "visualgenome_regions_gqa_shared_split_report.json",
        "sync_train": (
            "visualgenome_regions_dlc_train.jsonl",
            "visualgenome_regions_sanitized_source.jsonl",
        ),
    },
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sync-source-train",
        action="store_true",
        default=True,
        help="Copy rebuilt VG train files onto dlc/sanitized source_train paths when present.",
    )
    parser.add_argument(
        "--no-sync-source-train",
        action="store_false",
        dest="sync_source_train",
    )
    return parser.parse_args(argv)


def existing_dirs(root: Path, names: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for name in names:
        path = (root / name).expanduser()
        if path.is_dir():
            found.append(path.resolve())
    return found


def resolve_jsonl(directory: Path, filename: str) -> Path | None:
    direct = directory / filename
    if direct.is_file():
        return direct.resolve()
    wanted = filename.casefold()
    for path in directory.glob("*.jsonl"):
        if path.name.casefold() == wanted:
            return path.resolve()
    return None


def existing_files(directories: Sequence[Path], filenames: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for filename in filenames:
        match: Path | None = None
        for directory in directories:
            match = resolve_jsonl(directory, filename)
            if match is not None:
                break
        if match is None:
            return []
        files.append(match)
    return files


def resolve_gqa_inputs(root: Path) -> list[Path]:
    directories = existing_dirs(root, GQA_DIR_NAMES)
    for filenames in GQA_PAIR_NAMES:
        files = existing_files(directories, filenames)
        if len(files) == len(filenames):
            return files
    discovered = sorted(
        {
            path.resolve()
            for directory in directories
            for path in directory.glob("*.jsonl")
            if path.is_file()
        }
    )
    if discovered:
        return discovered
    searched = ", ".join(str(root / name) for name in GQA_DIR_NAMES)
    raise FileNotFoundError(
        f"Could not find GQA JSONL files below {searched}. "
        "Pass a data root that contains the converted GQA train and val files."
    )


def collect_image_stems(paths: Sequence[Path], progress_every: int) -> set[str]:
    stems: set[str] = set()
    rows = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                rows += 1
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
                stems.add(image_stem_group(record, path))
                if progress_every and rows % progress_every == 0:
                    print(
                        f"[gqa-index] rows={rows:,} unique_images={len(stems):,}",
                        flush=True,
                    )
    print(f"[gqa-index] complete rows={rows:,} unique_images={len(stems):,}", flush=True)
    return stems


def replace_with_copy(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def resplit_task(
    *,
    vg_dir: Path,
    task: dict[str, Any],
    gqa_stems: set[str],
    val_ratio: float,
    seed: int,
    progress_every: int,
    overwrite: bool,
    sync_source_train: bool,
) -> dict[str, Any]:
    inputs = existing_files([vg_dir], task["inputs"])
    if not inputs:
        raise FileNotFoundError(
            f"{task['name']}: missing input JSONL files under {vg_dir}: {list(task['inputs'])}"
        )
    train_output = (vg_dir / task["train"]).resolve()
    eval_output = (vg_dir / task["eval"]).resolve()
    report_output = (vg_dir / task["report"]).resolve()
    destinations = (train_output, eval_output, report_output)
    if not overwrite:
        existing = [str(path) for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(
                "output exists; pass --overwrite: " + ", ".join(existing)
            )
    staging_dir = vg_dir / ".gqa-shared-resplit"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_train = staging_dir / f"{task['name']}.train.jsonl"
    staged_eval = staging_dir / f"{task['name']}.eval.jsonl"
    staged_report = staging_dir / f"{task['name']}.report.json"
    for path in (staged_train, staged_eval, staged_report):
        if path.exists():
            path.unlink()

    report = split_jsonl(
        input_paths=inputs,
        train_output=staged_train,
        eval_output=staged_eval,
        report_output=staged_report,
        group_resolver=image_stem_group,
        group_key_description="Visual Genome / GQA image_id or filename stem",
        eval_ratio=val_ratio,
        seed=seed,
        force_train_groups=gqa_stems,
        progress_every=progress_every,
        overwrite=True,
        report_extra={
            "policy": "gqa_shared_images_forced_train_unique_images_hashed_val",
            "gqa_unique_images": len(gqa_stems),
            "dataset": task["name"],
        },
    )
    train_output.parent.mkdir(parents=True, exist_ok=True)
    staged_train.replace(train_output)
    staged_eval.replace(eval_output)
    staged_report.replace(report_output)
    synced: list[str] = []
    if sync_source_train:
        for relative in task.get("sync_train") or ():
            destination = resolve_jsonl(vg_dir, relative)
            if destination is None or destination == train_output:
                continue
            replace_with_copy(train_output, destination)
            synced.append(str(destination))
    report["materialized"] = {
        "train": str(train_output),
        "eval": str(eval_output),
        "report": str(report_output),
        "synced_source_train": synced,
    }
    print(
        f"[split] {task['name']} train={report['rows']['train']:,} "
        f"eval={report['rows']['eval']:,} forced_train_rows={report['rows']['forced_train']:,}",
        flush=True,
    )
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be in the range (0, 1)")
    gqa_inputs = resolve_gqa_inputs(root)
    gqa_stems = collect_image_stems(gqa_inputs, args.progress_every)
    if not gqa_stems:
        raise ValueError("GQA inputs did not yield any image stems")
    vg_dirs = existing_dirs(root, VG_DIR_NAMES)
    vg_dir = vg_dirs[0] if vg_dirs else None
    reports = []
    skipped: list[str] = []
    for task in VG_TASKS:
        if vg_dir is None:
            skipped.append(task["name"])
            print(f"[skip] {task['name']} missing Visual Genome directory", flush=True)
            continue
        inputs = existing_files([vg_dir], task["inputs"])
        if not inputs:
            skipped.append(task["name"])
            print(f"[skip] {task['name']} missing {list(task['inputs'])} under {vg_dir}", flush=True)
            continue
        reports.append(
            resplit_task(
                vg_dir=vg_dir,
                task=task,
                gqa_stems=gqa_stems,
                val_ratio=args.val_ratio,
                seed=args.seed,
                progress_every=args.progress_every,
                overwrite=args.overwrite,
                sync_source_train=args.sync_source_train,
            )
        )
    if not reports:
        searched = ", ".join(str(root / name) for name in VG_DIR_NAMES)
        raise FileNotFoundError(
            "No Visual Genome train/val JSONL files were found under " + searched
        )
    summary = {
        "data_root": str(root),
        "gqa_inputs": [str(path) for path in gqa_inputs],
        "gqa_unique_images": len(gqa_stems),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "skipped_datasets": skipped,
        "datasets": reports,
    }
    summary_path = (vg_dir or root / "visualgenome") / "visualgenome_gqa_shared_split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[write] {summary_path}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
