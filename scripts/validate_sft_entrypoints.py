#!/usr/bin/env python3
"""Validate curated train/eval entrypoints before launching ms-swift SFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    "/mnt/luojunkun/stage1/dataset_ms-swift/curated_dataset_entrypoints.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reject duplicate dataset paths, train/eval mixing, and simultaneous use "
            "of source JSONL files with their curated split files."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--train-json",
        type=Path,
        nargs="*",
        default=None,
        help="Training JSONL paths. Default: curated train entries from the manifest.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        nargs="*",
        default=None,
        help="Evaluation JSONL paths. Default: curated eval entries from the manifest.",
    )
    return parser.parse_args()


def canonical(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or not isinstance(value.get("datasets"), dict):
        raise ValueError("Manifest must contain a datasets object")
    return value


def duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def validate_entries(
    manifest: dict[str, Any], train_values: list[str], eval_values: list[str]
) -> None:
    errors: list[str] = []
    for split, values in (("train", train_values), ("eval", eval_values)):
        duplicates = duplicate_values(values)
        if duplicates:
            errors.append(f"{split} contains repeated paths: {duplicates}")
        missing = [value for value in values if not Path(value).is_file()]
        if missing:
            errors.append(f"{split} paths do not exist: {missing}")

    overlap = sorted(set(train_values) & set(eval_values))
    if overlap:
        errors.append(f"The same JSONL is present in train and eval: {overlap}")

    selected = set(train_values) | set(eval_values)
    for dataset_name, config in manifest["datasets"].items():
        if not isinstance(config, dict):
            errors.append(f"Manifest dataset {dataset_name!r} must be an object")
            continue
        source_only = canonical(config["source_only"])
        train_path = canonical(config["train"])
        eval_path = canonical(config["eval"])
        split_selected = selected & {train_path, eval_path}
        if source_only in selected and split_selected:
            errors.append(
                f"{dataset_name}: source-only file cannot be combined with curated splits: "
                f"{source_only}"
            )
        if source_only in train_values:
            errors.append(
                f"{dataset_name}: uncurated source-only file is forbidden for training: "
                f"{source_only}"
            )
        if eval_path in train_values:
            errors.append(f"{dataset_name}: eval file was passed as training data: {eval_path}")
        if train_path in eval_values:
            errors.append(f"{dataset_name}: train file was passed as evaluation data: {train_path}")

    if errors:
        raise ValueError("\n".join(f"- {error}" for error in errors))


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    default_train = [canonical(config["train"]) for config in manifest["datasets"].values()]
    default_eval = [canonical(config["eval"]) for config in manifest["datasets"].values()]
    train_values = (
        [canonical(path) for path in args.train_json]
        if args.train_json is not None
        else default_train
    )
    eval_values = (
        [canonical(path) for path in args.eval_json]
        if args.eval_json is not None
        else default_eval
    )
    validate_entries(manifest, train_values, eval_values)
    print(f"[ok] manifest={manifest_path}")
    print(f"[ok] train_entries={len(train_values)} eval_entries={len(eval_values)}")
    for value in train_values:
        print(f"[train] {value}")
    for value in eval_values:
        print(f"[eval] {value}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        print(f"Error:\n{error}", file=sys.stderr)
        raise SystemExit(1) from None
