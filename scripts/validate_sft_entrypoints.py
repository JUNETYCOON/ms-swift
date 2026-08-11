#!/usr/bin/env python3
"""Validate curated train/eval entrypoints before launching ms-swift SFT."""

from __future__ import annotations

import argparse
import hashlib
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


def file_fingerprint(path: Path | str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    hasher = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hasher.hexdigest(),
    }


def validate_fingerprint(
    path: Path,
    evidence: Any,
    label: str,
    cache: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(evidence, dict):
        raise ValueError(f"{label}: dedup report has no file fingerprint")
    resolved = canonical(path)
    if canonical(evidence.get("path", "")) != resolved:
        raise ValueError(f"{label}: fingerprint path does not match: {resolved}")
    actual = cache.get(resolved)
    if actual is None:
        actual = file_fingerprint(path)
        cache[resolved] = actual
    for key in ("size", "sha256"):
        if evidence.get(key) != actual[key]:
            raise ValueError(
                f"{label}: fingerprint {key} changed after global dedup; rerun dedup"
            )


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or not isinstance(value.get("datasets"), dict):
        raise ValueError("Manifest must contain a datasets object")
    return value


def enabled_datasets(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, config in manifest["datasets"].items():
        if not isinstance(config, dict):
            raise ValueError(f"Manifest dataset {name!r} must be an object")
        if config.get("enabled", True):
            result[name] = config
    return result


def eval_paths(config: dict[str, Any]) -> list[str]:
    value = config.get("eval")
    values = value if isinstance(value, list) else [value]
    if not values or any(item is None for item in values):
        raise ValueError("Enabled manifest datasets require a non-empty eval path")
    return [canonical(item) for item in values]


def validate_global_dedup(
    manifest: dict[str, Any], manifest_path: Path
) -> None:
    policy = manifest.get("global_dedup")
    if not isinstance(policy, dict) or not policy.get("required", False):
        return
    report_value = policy.get("report")
    if not report_value:
        raise ValueError("global_dedup.required needs a report path")
    report_path = Path(report_value).expanduser()
    if not report_path.is_absolute():
        report_path = manifest_path.parent / report_path
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Global dedup report does not exist: {report_path}")
    with report_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    verification = report.get("verification") if isinstance(report, dict) else None
    if report.get("schema_version") != 2:
        raise ValueError("Global dedup report schema_version must be 2")
    if report.get("status") != "complete" or not isinstance(verification, dict):
        raise ValueError("Global dedup report is not complete")
    if verification.get("status") != "complete":
        raise ValueError("Global dedup post-filter verification did not complete")
    if verification.get("train_eval_overlap_rows") != 0:
        raise ValueError("Global dedup verification train_eval_overlap_rows is not zero")

    datasets = enabled_datasets(manifest)
    priority = policy.get("training_priority")
    if not isinstance(priority, list) or set(priority) != set(datasets):
        raise ValueError("Global dedup training_priority does not match enabled datasets")
    if report.get("training_priority") != priority:
        raise ValueError("Global dedup report training_priority does not match manifest")
    if canonical(report.get("manifest", "")) != str(manifest_path.resolve()):
        raise ValueError("Global dedup report manifest path does not match")
    fingerprint_cache: dict[str, dict[str, Any]] = {}
    validate_fingerprint(
        manifest_path,
        report.get("manifest_fingerprint"),
        "manifest",
        fingerprint_cache,
    )
    report_datasets = report.get("datasets")
    if not isinstance(report_datasets, dict):
        raise ValueError("Global dedup report has no datasets object")
    report_mtime = report_path.stat().st_mtime_ns
    if manifest_path.stat().st_mtime_ns > report_mtime:
        raise ValueError("Manifest changed after the global dedup report; rerun dedup")
    require_media_identity = bool(policy.get("require_media_identity", True))
    deduplicate_cross_dataset_train = policy.get(
        "deduplicate_cross_dataset_train", True
    )
    if not isinstance(deduplicate_cross_dataset_train, bool):
        raise ValueError(
            "global_dedup.deduplicate_cross_dataset_train must be a boolean"
        )
    report_policy = report.get("policy")
    if not isinstance(report_policy, dict):
        raise ValueError("Global dedup report has no policy object")
    if report_policy.get("require_media_identity") is not require_media_identity:
        raise ValueError("Global dedup report media identity policy does not match manifest")
    if (
        report_policy.get("deduplicate_cross_dataset_train", True)
        is not deduplicate_cross_dataset_train
    ):
        raise ValueError(
            "Global dedup report cross-dataset train policy does not match manifest"
        )
    if (
        deduplicate_cross_dataset_train
        and verification.get("cross_dataset_train_overlap_rows") != 0
    ):
        raise ValueError(
            "Global dedup verification cross_dataset_train_overlap_rows is not zero"
        )
    for name, config in datasets.items():
        evidence = report_datasets.get(name)
        if not isinstance(evidence, dict):
            raise ValueError(f"Global dedup report is missing dataset {name!r}")
        allow_text_only = config.get("allow_text_only", False)
        if not isinstance(allow_text_only, bool):
            raise ValueError(f"{name}: allow_text_only must be a boolean")
        if bool(evidence.get("allow_text_only", False)) is not allow_text_only:
            raise ValueError(f"{name}: report allow_text_only does not match manifest")
        use_video_stem_identity = config.get("use_video_stem_identity", True)
        if not isinstance(use_video_stem_identity, bool):
            raise ValueError(f"{name}: use_video_stem_identity must be a boolean")
        if evidence.get("use_video_stem_identity", True) is not use_video_stem_identity:
            raise ValueError(
                f"{name}: report use_video_stem_identity does not match manifest"
            )
        train_path = Path(canonical(config["train"]))
        source_value = config.get("source_train") or config.get("split_train")
        if not source_value:
            raise ValueError(f"{name}: global dedup requires source_train")
        source_path = Path(canonical(source_value))
        if canonical(evidence.get("train", "")) != str(train_path):
            raise ValueError(f"{name}: report train path does not match manifest")
        if canonical(evidence.get("source_train", "")) != str(source_path):
            raise ValueError(f"{name}: report source_train path does not match manifest")
        fingerprints = evidence.get("fingerprints")
        if not isinstance(fingerprints, dict):
            raise ValueError(f"{name}: report has no fingerprints object")
        validate_fingerprint(
            source_path,
            fingerprints.get("source_train"),
            f"{name} source_train",
            fingerprint_cache,
        )
        validate_fingerprint(
            train_path,
            fingerprints.get("train"),
            f"{name} train",
            fingerprint_cache,
        )
        dataset_eval_paths = [Path(value) for value in eval_paths(config)]
        eval_fingerprints = fingerprints.get("eval")
        if not isinstance(eval_fingerprints, list) or len(eval_fingerprints) != len(
            dataset_eval_paths
        ):
            raise ValueError(f"{name}: report eval fingerprints do not match eval paths")
        for index, (path, fingerprint) in enumerate(
            zip(dataset_eval_paths, eval_fingerprints)
        ):
            validate_fingerprint(
                path,
                fingerprint,
                f"{name} eval[{index}]",
                fingerprint_cache,
            )
        prerequisite_value = config.get("prerequisite_report")
        evidence_prerequisite = evidence.get("prerequisite_report")
        if prerequisite_value:
            prerequisite_path = Path(canonical(prerequisite_value))
            if canonical(evidence_prerequisite or "") != str(prerequisite_path):
                raise ValueError(f"{name}: report prerequisite path does not match manifest")
            validate_fingerprint(
                prerequisite_path,
                fingerprints.get("prerequisite_report"),
                f"{name} prerequisite_report",
                fingerprint_cache,
            )
        elif evidence_prerequisite is not None or fingerprints.get(
            "prerequisite_report"
        ) is not None:
            raise ValueError(f"{name}: unexpected prerequisite evidence in dedup report")
        counts = evidence.get("counts")
        if not isinstance(counts, dict):
            raise ValueError(f"{name}: report has no counts object")
        if require_media_identity:
            for key in ("eval_rows_without_identity", "train_rows_without_identity"):
                if counts.get(key, 0) != 0:
                    raise ValueError(f"{name}: global dedup report {key} is not zero")
        required_paths = [source_path, train_path, *dataset_eval_paths]
        if prerequisite_value:
            required_paths.append(Path(canonical(prerequisite_value)))
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(f"{name}: required entrypoint does not exist: {path}")
            if path.stat().st_mtime_ns > report_mtime:
                raise ValueError(
                    f"{name}: {path} changed after the global dedup report; rerun dedup"
                )


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
    enabled = enabled_datasets(manifest)
    policy = manifest.get("global_dedup")
    strict_global = isinstance(policy, dict) and policy.get("required", False)
    allowed_train = {canonical(config["train"]) for config in enabled.values()}
    allowed_eval = {
        value for config in enabled.values() for value in eval_paths(config)
    }
    if strict_global:
        unexpected_train = sorted(set(train_values) - allowed_train)
        unexpected_eval = sorted(set(eval_values) - allowed_eval)
        if unexpected_train:
            errors.append(
                f"Training paths are not globally deduplicated manifest entries: {unexpected_train}"
            )
        if unexpected_eval:
            errors.append(f"Evaluation paths are not manifest eval entries: {unexpected_eval}")

    for dataset_name, config in manifest["datasets"].items():
        if not isinstance(config, dict):
            errors.append(f"Manifest dataset {dataset_name!r} must be an object")
            continue
        forbidden_sources = {
            canonical(value)
            for key in ("source_only", "source_train", "split_train")
            if (value := config.get(key))
        }
        train_path = canonical(config["train"]) if config.get("train") else None
        dataset_eval = eval_paths(config) if config.get("eval") else []
        curated_paths = ({train_path} if train_path else set()) | set(dataset_eval)
        for source_path in forbidden_sources:
            if source_path in selected and selected & curated_paths:
                errors.append(
                    f"{dataset_name}: source file cannot be combined with curated splits: "
                    f"{source_path}"
                )
            if source_path in train_values:
                errors.append(
                    f"{dataset_name}: pre-dedup/source file is forbidden for training: "
                    f"{source_path}"
                )
        for eval_path in dataset_eval:
            if eval_path in train_values:
                errors.append(
                    f"{dataset_name}: eval file was passed as training data: {eval_path}"
                )
        if train_path and train_path in eval_values:
            errors.append(
                f"{dataset_name}: train file was passed as evaluation data: {train_path}"
            )

    if errors:
        raise ValueError("\n".join(f"- {error}" for error in errors))


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    validate_global_dedup(manifest, manifest_path)
    enabled = enabled_datasets(manifest)
    default_train = [canonical(config["train"]) for config in enabled.values()]
    default_eval = [value for config in enabled.values() for value in eval_paths(config)]
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
