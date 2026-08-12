#!/usr/bin/env python3
"""Verify Stage 1 grouped splits and the repaired SpatialVLM conversion."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DATA_ROOT = Path("/mnt/luojunkun/stage1/dataset_ms-swift")
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "curated_dataset_entrypoints.json"

SPLIT_EXPECTATIONS = {
    "ai2d": {
        "report": "ai2d/ai2d_split_report.json",
        "train": "ai2d/ai2d_pretrain_msswift_train.jsonl",
        "eval": "ai2d/ai2d_pretrain_msswift_eval.jsonl",
        "reserved_groups": 79,
        "group_key_contains": "SHA-256",
        "policy": {
            "hash_algorithm": "sha256",
            "source_plus_eval_forbidden": True,
            "training_entrypoint": "strict _train only",
        },
    },
    "vlm-r1": {
        "report": "vlm-r1/vlm_r1_grouped_split_report.json",
        "train": "vlm-r1/vlm_r1_sft_grounding_msswift_train.jsonl",
        "eval": "vlm-r1/vlm_r1_sft_grounding_msswift_eval.jsonl",
        "reserved_groups": 2825,
        "group_key_contains": "SHA-256",
        "policy": {"hash_algorithm": "sha256"},
    },
    "robo2vlm": {
        "report": "robo2vlm/robo2vlm_grouped_split_report.json",
        "train": "robo2vlm/robo2vlm_sft_train.jsonl",
        "eval": "robo2vlm/robo2vlm_sft_eval.jsonl",
        "reserved_groups": 1582,
        "source_overlap_groups": 5239,
        "group_key_contains": "trailing _qN removed",
        "policy": {
            "task_type": "multiple-choice VQA",
            "caption_label_forbidden": True,
            "source_test_is_not_an_independent_eval": True,
        },
    },
}

SPATIAL_REQUIRED_VALUES = {
    "configuration.spatialvlm_split_policy": (
        "preserve_official_train_test_and_reserve_test_image_sha256"
    ),
    "spatialvlm_official_split_audit.official_test_records_written_to_train": 0,
    "spatialvlm_official_split_audit.official_train_records_written_to_val": 0,
    "spatialvlm_official_split_audit.post_filter_train_test_hash_overlap": 0,
    "unique_media.cross_split_leakage": 0,
}


class VerificationError(ValueError):
    pass


ROBO2VLM_ID_RE = re.compile(r"_q\d+$", re.IGNORECASE)
ROBO2VLM_CHOICE_RE = re.compile(r"^([A-Z])\.\s+(.+)$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    return parser


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"required JSON file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"JSON root must be an object: {path}")
    return value


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise VerificationError(f"missing report field: {dotted_path}")
        current = current[part]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count


def verify_robo2vlm_choice_contract(path: Path) -> dict[str, int]:
    rows = 0
    choices = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise VerificationError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(record, dict):
                raise VerificationError(f"{path}:{line_number}: row is not an object")
            sample_id = str(record.get("id") or "").strip()
            if not ROBO2VLM_ID_RE.search(sample_id):
                raise VerificationError(
                    f"{path}:{line_number}: Robo2VLM id has no trailing _qN: {sample_id!r}"
                )
            messages = record.get("messages")
            if not isinstance(messages, list):
                raise VerificationError(f"{path}:{line_number}: messages is not a list")
            user_messages = [
                item.get("content")
                for item in messages
                if isinstance(item, dict) and item.get("role") == "user"
            ]
            assistant_messages = [
                item.get("content")
                for item in messages
                if isinstance(item, dict) and item.get("role") == "assistant"
            ]
            if len(user_messages) != 1 or len(assistant_messages) != 1:
                raise VerificationError(
                    f"{path}:{line_number}: expected one user and one assistant message"
                )
            prompt = str(user_messages[0] or "")
            answer = " ".join(str(assistant_messages[0] or "").split())
            if prompt.count("<image>") != 1 or "\nQuestion:" not in prompt:
                raise VerificationError(
                    f"{path}:{line_number}: malformed image/question prompt"
                )
            _prefix, separator, choices_text = prompt.rpartition("\nChoices:\n")
            if not separator:
                raise VerificationError(f"{path}:{line_number}: prompt has no Choices block")
            parsed_choices: dict[str, str] = {}
            for raw_line in choices_text.splitlines():
                choice_line = " ".join(raw_line.split())
                if not choice_line:
                    continue
                match = ROBO2VLM_CHOICE_RE.fullmatch(choice_line)
                if not match:
                    raise VerificationError(
                        f"{path}:{line_number}: malformed choice line: {choice_line!r}"
                    )
                label, text = match.groups()
                if label in parsed_choices:
                    raise VerificationError(
                        f"{path}:{line_number}: duplicate choice label {label}"
                    )
                try:
                    parsed_choice = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    parsed_choice = None
                if isinstance(parsed_choice, (list, tuple, set)):
                    raise VerificationError(
                        f"{path}:{line_number}: serialized choice collection was not expanded"
                    )
                parsed_choices[label] = text
            expected_labels = [chr(ord("A") + index) for index in range(len(parsed_choices))]
            if not 2 <= len(parsed_choices) <= 26 or list(parsed_choices) != expected_labels:
                raise VerificationError(
                    f"{path}:{line_number}: choices must be sequential A.. with 2..26 entries"
                )
            answer_match = ROBO2VLM_CHOICE_RE.fullmatch(answer)
            if not answer_match:
                raise VerificationError(
                    f"{path}:{line_number}: answer is not '<label>. <choice>': {answer!r}"
                )
            answer_label, answer_text = answer_match.groups()
            if parsed_choices.get(answer_label) != answer_text:
                raise VerificationError(
                    f"{path}:{line_number}: answer label/text does not match the choices"
                )
            choices += len(parsed_choices)
    if not rows:
        raise VerificationError(f"Robo2VLM split is empty: {path}")
    return {"rows": rows, "choices": choices}


def _canonical(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise VerificationError(f"{label}: got {actual!r}, expected {expected!r}")


def verify_grouped_split(
    name: str,
    expectation: Mapping[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    report_path = data_root / expectation["report"]
    train_path = data_root / expectation["train"]
    eval_path = data_root / expectation["eval"]
    report = _load_json(report_path)
    for path in (train_path, eval_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise VerificationError(f"{name}: split output is missing or empty: {path}")
        if path.stat().st_mtime_ns > report_path.stat().st_mtime_ns:
            raise VerificationError(f"{name}: split output is newer than its report: {path}")

    _require_equal(_canonical(report.get("train_jsonl", "")), _canonical(train_path), f"{name} train path")
    _require_equal(_canonical(report.get("eval_jsonl", "")), _canonical(eval_path), f"{name} eval path")
    if expectation["group_key_contains"].casefold() not in str(report.get("group_key", "")).casefold():
        raise VerificationError(
            f"{name}: group_key does not contain {expectation['group_key_contains']!r}"
        )
    _require_equal(_nested(report, "groups.train_eval_overlap"), 0, f"{name} group overlap")
    _require_equal(
        _nested(report, "reserved_eval.missing_groups"),
        0,
        f"{name} missing reserved groups",
    )
    _require_equal(
        _nested(report, "reserved_eval.reserve_only"),
        True,
        f"{name} reserve-only mode",
    )
    _require_equal(
        _nested(report, "reserved_eval.groups"),
        expectation["reserved_groups"],
        f"{name} reserved media groups",
    )
    _require_equal(
        _nested(report, "groups.eval"),
        expectation["reserved_groups"],
        f"{name} eval media groups",
    )
    source_overlap_groups = expectation.get("source_overlap_groups")
    if source_overlap_groups is not None:
        _require_equal(
            _nested(report, "groups.present_in_multiple_source_files"),
            source_overlap_groups,
            f"{name} source-overlap media groups",
        )
    for key, expected in expectation["policy"].items():
        _require_equal(
            _nested(report, f"dataset_policy.{key}"),
            expected,
            f"{name} policy {key}",
        )

    train_rows = _line_count(train_path)
    eval_rows = _line_count(eval_path)
    train_sha256 = _sha256(train_path)
    eval_sha256 = _sha256(eval_path)
    _require_equal(_nested(report, "rows.train"), train_rows, f"{name} train rows")
    _require_equal(_nested(report, "rows.eval"), eval_rows, f"{name} eval rows")
    _require_equal(
        _nested(report, "output_sha256.train"), train_sha256, f"{name} train SHA-256"
    )
    _require_equal(
        _nested(report, "output_sha256.eval"), eval_sha256, f"{name} eval SHA-256"
    )
    contract = None
    if name == "robo2vlm":
        contract = {
            "train": verify_robo2vlm_choice_contract(train_path),
            "eval": verify_robo2vlm_choice_contract(eval_path),
        }
    return {
        "status": "complete",
        "report": str(report_path),
        "train": {"path": str(train_path), "rows": train_rows, "sha256": train_sha256},
        "eval": {"path": str(eval_path), "rows": eval_rows, "sha256": eval_sha256},
        "reserved_groups": expectation["reserved_groups"],
        "source_overlap_groups": source_overlap_groups,
        "choice_contract": contract,
    }


def verify_spatialvlm(data_root: Path) -> dict[str, Any]:
    root = data_root / "spatialvlm"
    report_path = root / "conversion_report.json"
    train_path = root / "train.jsonl"
    eval_path = root / "val.jsonl"
    groups_path = root / "media_groups.tsv"
    report = _load_json(report_path)
    _require_equal(report.get("dataset"), "spatialvlm", "SpatialVLM report dataset")
    for dotted_path, expected in SPATIAL_REQUIRED_VALUES.items():
        _require_equal(_nested(report, dotted_path), expected, f"SpatialVLM {dotted_path}")
    test_source_rows = _nested(
        report, "spatialvlm_official_split_audit.official_test_source_rows"
    )
    test_hashes = _nested(
        report, "spatialvlm_official_split_audit.test_media_hashes_reserved"
    )
    if not isinstance(test_source_rows, int) or test_source_rows <= 0:
        raise VerificationError("SpatialVLM official test source row count must be positive")
    if not isinstance(test_hashes, int) or test_hashes <= 0:
        raise VerificationError("SpatialVLM reserved test image hash count must be positive")
    source_names = [Path(value).name.casefold() for value in report.get("input_files") or []]
    if not any(name.startswith("train-") for name in source_names):
        raise VerificationError("SpatialVLM report has no official train parquet")
    if not any(name.startswith(("test-", "val-", "validation-")) for name in source_names):
        raise VerificationError("SpatialVLM report has no official test/validation parquet")
    for key, path in (("train", train_path), ("val", eval_path)):
        if not path.is_file() or path.stat().st_size == 0:
            raise VerificationError(f"SpatialVLM {key} output is missing or empty: {path}")
        if path.stat().st_mtime_ns > report_path.stat().st_mtime_ns:
            raise VerificationError(f"SpatialVLM {key} output is newer than its report")
        _require_equal(
            _canonical(_nested(report, f"output_files.{key}")),
            _canonical(path),
            f"SpatialVLM {key} output path",
        )
        _require_equal(
            _nested(report, f"output_sha256.{key}"),
            _sha256(path),
            f"SpatialVLM {key} output SHA-256",
        )
    train_rows = _line_count(train_path)
    eval_rows = _line_count(eval_path)
    _require_equal(
        _nested(report, "counters.written_train"), train_rows, "SpatialVLM train rows"
    )
    _require_equal(
        _nested(report, "counters.written_val"), eval_rows, "SpatialVLM val rows"
    )

    if not groups_path.is_file():
        raise VerificationError(f"SpatialVLM media group manifest is missing: {groups_path}")
    if groups_path.stat().st_mtime_ns > report_path.stat().st_mtime_ns:
        raise VerificationError("SpatialVLM media group manifest is newer than its report")
    _require_equal(
        _canonical(_nested(report, "output_files.media_groups")),
        _canonical(groups_path),
        "SpatialVLM media group output path",
    )
    _require_equal(
        _nested(report, "output_sha256.media_groups"),
        _sha256(groups_path),
        "SpatialVLM media group SHA-256",
    )
    assignments: dict[str, str] = {}
    with groups_path.open("r", encoding="utf-8") as stream:
        _require_equal(stream.readline().rstrip("\n"), "split\tmedia_key", "SpatialVLM media header")
        for line_number, line in enumerate(stream, start=2):
            try:
                split, media_key = line.rstrip("\n").split("\t", 1)
            except ValueError as error:
                raise VerificationError(
                    f"SpatialVLM invalid media group row {line_number}"
                ) from error
            if split not in {"train", "val"} or not media_key:
                raise VerificationError(f"SpatialVLM invalid media group row {line_number}")
            previous = assignments.setdefault(media_key, split)
            if previous != split:
                raise VerificationError(
                    f"SpatialVLM media hash crosses train/val: {media_key}"
                )
    _require_equal(
        len(assignments), _nested(report, "unique_media.total"), "SpatialVLM unique media"
    )
    split_counts = {
        split: sum(1 for assigned_split in assignments.values() if assigned_split == split)
        for split in ("train", "val")
    }
    for split, count in split_counts.items():
        _require_equal(
            count,
            _nested(report, f"unique_media.{split}"),
            f"SpatialVLM {split} unique media",
        )
    return {
        "status": "complete",
        "report": str(report_path),
        "official_test_source_rows": test_source_rows,
        "test_media_hashes_reserved": test_hashes,
        "train": {"path": str(train_path), "rows": train_rows, "sha256": _sha256(train_path)},
        "eval": {"path": str(eval_path), "rows": eval_rows, "sha256": _sha256(eval_path)},
        "unique_media": len(assignments),
    }


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    policy = manifest.get("global_dedup")
    datasets = manifest.get("datasets")
    if not isinstance(policy, dict) or not isinstance(datasets, dict):
        raise VerificationError("curated manifest is missing global_dedup or datasets")
    _require_equal(policy.get("required"), True, "manifest global dedup requirement")
    _require_equal(
        policy.get("require_media_identity"), True, "manifest media identity requirement"
    )
    enabled = {
        name
        for name, config in datasets.items()
        if isinstance(config, dict) and config.get("enabled", True)
    }
    priority = policy.get("training_priority")
    if not isinstance(priority, list) or len(priority) != len(set(priority)):
        raise VerificationError("manifest training priority is missing or contains duplicates")
    _require_equal(set(priority), enabled, "manifest enabled datasets vs training priority")
    for name in priority:
        config = datasets[name]
        if "dedup_owner" in config:
            raise VerificationError(f"{name}: dedup_owner aliases are forbidden")
        train = str(config.get("train") or "")
        source = str(config.get("source_train") or config.get("split_train") or "")
        if "global_train" not in Path(train).name:
            raise VerificationError(f"{name}: train entry is not a global_train file: {train}")
        if not source or _canonical(source) == _canonical(train):
            raise VerificationError(f"{name}: source_train and train must be distinct")
    spatial = datasets.get("spatialvlm")
    if not isinstance(spatial, dict) or not spatial.get("enabled"):
        raise VerificationError("SpatialVLM is not enabled in the final manifest")
    for key, expected in SPATIAL_REQUIRED_VALUES.items():
        _require_equal(
            (spatial.get("prerequisite_values") or {}).get(key),
            expected,
            f"manifest SpatialVLM prerequisite {key}",
        )
    robo = datasets.get("robo2vlm")
    _require_equal(
        robo.get("task_type") if isinstance(robo, dict) else None,
        "multiple-choice VQA / embodied state understanding",
        "Robo2VLM task type",
    )
    return {
        "status": "complete",
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "enabled_datasets": len(enabled),
        "training_priority": priority,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def run(data_root: Path, manifest_path: Path) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    splits = {
        name: verify_grouped_split(name, expectation, data_root)
        for name, expectation in SPLIT_EXPECTATIONS.items()
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "generated_at": _utc_now(),
        "data_root": str(data_root),
        "manifest": verify_manifest(manifest_path),
        "grouped_splits": splits,
        "spatialvlm": verify_spatialvlm(data_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args.data_root, args.manifest)
        if args.output:
            _write_json_atomic(args.output.expanduser().resolve(), report)
            print(f"[report] {args.output.expanduser().resolve()}")
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, VerificationError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
