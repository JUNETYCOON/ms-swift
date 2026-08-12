#!/usr/bin/env python3
"""Command-line tools for profiling, validating, splitting, and adapting S1-UDF."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from universal_dataset.io import atomic_text_writer, iter_jsonl, json_line, parse_json_strict
    from universal_dataset.ms_swift import (
        MsSwiftConversionError, ms_swift_projection_warnings, ms_swift_to_record, record_to_ms_swift,
    )
    from universal_dataset.profile import profile_path
    from universal_dataset.split import canonical_asset_identity, split_jsonl
    from universal_dataset.validation import (
        MANIFEST_SCHEMA_PATH, SCHEMA_PATH, ensure_valid, validate_manifest, validate_record,
    )
else:
    from .io import atomic_text_writer, iter_jsonl, json_line, parse_json_strict
    from .ms_swift import (
        MsSwiftConversionError, ms_swift_projection_warnings, ms_swift_to_record, record_to_ms_swift,
    )
    from .profile import profile_path
    from .split import canonical_asset_identity, split_jsonl
    from .validation import MANIFEST_SCHEMA_PATH, SCHEMA_PATH, ensure_valid, validate_manifest, validate_record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Read container metadata and sample nested fields without extraction.")
    profile.add_argument("path", type=Path)
    profile.add_argument("--sample-rows", type=int, default=100)
    profile.add_argument("--max-depth", type=int, default=8)
    profile.add_argument("--max-files", type=int, default=10000)
    profile.add_argument("--workers", type=int, default=1)
    profile.add_argument("--include-examples", action="store_true")
    profile.add_argument("--relative-paths", action="store_true")
    profile.add_argument("--output", type=Path)

    validate = subparsers.add_parser("validate", help="Validate S1-UDF JSONL and cross-record leakage invariants.")
    validate.add_argument("input", type=Path)
    validate.add_argument("--check-assets", action="store_true")
    validate.add_argument("--no-json-schema", action="store_true")
    validate.add_argument("--max-errors", type=int, default=100)
    validate.add_argument("--report", type=Path)

    validate_manifest_parser = subparsers.add_parser(
        "validate-manifest", help="Validate an S1-UDF manifest and optionally its record files."
    )
    validate_manifest_parser.add_argument("input", type=Path)
    validate_manifest_parser.add_argument("--check-files", action="store_true")
    validate_manifest_parser.add_argument("--no-json-schema", action="store_true")
    validate_manifest_parser.add_argument("--max-errors", type=int, default=100)
    validate_manifest_parser.add_argument("--report", type=Path)

    split = subparsers.add_parser("split", help="Split JSONL deterministically by group_id.")
    split.add_argument("input", type=Path)
    split.add_argument("--train-output", type=Path, required=True)
    split.add_argument("--val-output", type=Path, required=True)
    split.add_argument("--val-ratio", type=float, default=0.02)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--no-asset-audit", action="store_true")
    split.add_argument("--overwrite", action="store_true")

    import_swift = subparsers.add_parser("import-ms-swift", help="Convert ms-swift JSONL to S1-UDF JSONL.")
    import_swift.add_argument("input", type=Path)
    import_swift.add_argument("--output", type=Path, required=True)
    import_swift.add_argument("--dataset-name", default="ms-swift")
    import_swift.add_argument("--drop-raw", action="store_true")
    import_swift.add_argument("--no-json-schema", action="store_true")
    import_swift.add_argument("--workers", type=int, default=1)
    import_swift.add_argument("--chunksize", type=int, default=64)
    import_swift.add_argument("--overwrite", action="store_true")

    export_swift = subparsers.add_parser("export-ms-swift", help="Convert S1-UDF conversations to ms-swift JSONL.")
    export_swift.add_argument("input", type=Path)
    export_swift.add_argument("--output", type=Path, required=True)
    export_swift.add_argument("--drop-ids", action="store_true")
    export_swift.add_argument("--answer-policy", choices=("error", "first", "highest-count"), default="error")
    export_swift.add_argument(
        "--annotation-policy",
        choices=("error", "prefer-conversations", "prefer-annotations"),
        default="error",
    )
    export_swift.add_argument("--caption-prompt")
    export_swift.add_argument("--unsupported-policy", choices=("error", "skip"), default="error")
    export_swift.add_argument("--report", type=Path)
    export_swift.add_argument("--max-error-examples", type=int, default=100)
    export_swift.add_argument("--no-json-schema", action="store_true")
    export_swift.add_argument("--workers", type=int, default=1)
    export_swift.add_argument("--chunksize", type=int, default=64)
    export_swift.add_argument("--overwrite", action="store_true")

    schema_path = subparsers.add_parser("schema-path", help="Print a canonical JSON Schema path.")
    schema_path.add_argument("--kind", choices=("record", "manifest"), default="record")
    return parser


def _write_json(path: Optional[Path], value: Dict[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path is None:
        print(text, end="")
    else:
        with atomic_text_writer(path, overwrite=True) as stream:
            stream.write(text)


def _import_ms_swift_worker(payload: Any) -> Any:
    line_number, value, source_name, dataset_name, preserve_raw, base_dir, check_schema = payload
    record = ms_swift_to_record(
        value,
        source_id=str(value.get("id") or value.get("sample_id") or "{}:{}".format(source_name, line_number)),
        source_dataset=dataset_name,
        preserve_raw=preserve_raw,
        media_base_dir=Path(base_dir),
    )
    ensure_valid(record, check_json_schema=check_schema, base_dir=Path(base_dir))
    return line_number, record


def _export_ms_swift_worker(payload: Any) -> Any:
    (
        line_number,
        record,
        include_ids,
        base_dir,
        check_schema,
        answer_policy,
        annotation_policy,
        caption_prompt,
    ) = payload
    ensure_valid(record, check_json_schema=check_schema, base_dir=Path(base_dir))
    try:
        values = record_to_ms_swift(
            record,
            include_ids=include_ids,
            base_dir=Path(base_dir),
            answer_policy=answer_policy,
            annotation_policy=annotation_policy,
            caption_prompt=caption_prompt,
        )
    except MsSwiftConversionError as error:
        return line_number, record.get("id"), None, [], str(error)
    projection_warnings = ms_swift_projection_warnings(record, answer_policy, annotation_policy)
    return line_number, record.get("id"), values, projection_warnings, None


def _validate(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    errors: List[Dict[str, Any]] = []
    record_ids = set()
    group_splits: Dict[str, str] = {}
    asset_groups: Dict[str, str] = {}
    records = 0
    groups = set()
    for line_number, record in iter_jsonl(input_path):
        if len(errors) >= args.max_errors:
            break
        records += 1
        identifier = record.get("id")
        if isinstance(identifier, str) and identifier in record_ids:
            errors.append({"line": line_number, "path": "$.id", "code": "duplicate_record_id", "message": repr(identifier)})
        if isinstance(identifier, str):
            record_ids.add(identifier)
        group_id = record.get("group_id")
        if isinstance(group_id, str):
            groups.add(group_id)
            split = record.get("split")
            if isinstance(split, str):
                previous = group_splits.setdefault(group_id, split)
                if previous != split:
                    errors.append(
                        {
                            "line": line_number,
                            "path": "$.split",
                            "code": "group_split_leakage",
                            "message": "group {!r} appears in {!r} and {!r}".format(group_id, previous, split),
                        }
                    )
            for asset in record.get("assets") or []:
                if len(errors) >= args.max_errors:
                    break
                if not isinstance(asset, dict):
                    continue
                try:
                    identity = canonical_asset_identity(asset, input_path)
                except ValueError as error:
                    errors.append({"line": line_number, "path": "$.assets", "code": "asset_identity", "message": str(error)})
                    continue
                previous_group = asset_groups.setdefault(identity, group_id)
                if previous_group != group_id:
                    errors.append(
                        {
                            "line": line_number,
                            "path": "$.group_id",
                            "code": "asset_group_conflict",
                            "message": "asset {} belongs to groups {!r} and {!r}".format(identity, previous_group, group_id),
                        }
                    )
        for issue in validate_record(
            record,
            check_json_schema=not args.no_json_schema,
            check_assets=args.check_assets,
            base_dir=input_path.parent,
        ):
            errors.append({"line": line_number, "path": issue.path, "code": issue.code, "message": issue.message})
            if len(errors) >= args.max_errors:
                break
        if len(errors) >= args.max_errors:
            break
    if records == 0 and not errors:
        errors.append(
            {
                "line": None,
                "path": "$",
                "code": "empty_dataset",
                "message": "input JSONL contains no records",
            }
        )
    report = {
        "status": "valid" if not errors else "invalid",
        "input": str(input_path),
        "records_scanned": records,
        "groups": len(groups),
        "unique_assets": len(asset_groups),
        "errors": errors,
        "errors_truncated": len(errors) >= args.max_errors,
    }
    _write_json(args.report, report)
    return 0 if not errors else 1


def _import_ms_swift(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    count = 0
    payloads = (
        (
            line_number,
            value,
            input_path.name,
            args.dataset_name,
            not args.drop_raw,
            str(input_path.parent),
            not args.no_json_schema,
        )
        for line_number, value in iter_jsonl(input_path)
    )
    executor: Optional[ProcessPoolExecutor] = None
    try:
        if args.workers == 1:
            converted = map(_import_ms_swift_worker, payloads)
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            converted = executor.map(_import_ms_swift_worker, payloads, chunksize=args.chunksize)
        with atomic_text_writer(args.output, overwrite=args.overwrite) as stream:
            for _, record in converted:
                stream.write(json_line(record))
                count += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    print(json.dumps({"status": "complete", "records": count, "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


def _validate_manifest_command(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    value = parse_json_strict(input_path.read_text(encoding="utf-8"))
    issues = validate_manifest(
        value,
        manifest_path=input_path,
        check_json_schema=not args.no_json_schema,
        check_files=args.check_files,
        max_errors=args.max_errors,
    )
    report = {
        "status": "valid" if not issues else "invalid",
        "input": str(input_path),
        "files_checked": bool(args.check_files),
        "errors": [
            {"path": issue.path, "code": issue.code, "message": issue.message} for issue in issues
        ],
        "errors_truncated": len(issues) >= args.max_errors,
    }
    _write_json(args.report, report)
    return 0 if not issues else 1


def _export_ms_swift(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    input_records = 0
    output_records = 0
    skipped_records = 0
    warnings: Counter = Counter()
    error_examples: List[Dict[str, Any]] = []
    fatal_error: Optional[str] = None
    executor: Optional[ProcessPoolExecutor] = None
    payloads = (
        (
            line_number,
            record,
            not args.drop_ids,
            str(input_path.parent),
            not args.no_json_schema,
            args.answer_policy,
            args.annotation_policy,
            args.caption_prompt,
        )
        for line_number, record in iter_jsonl(input_path)
    )
    try:
        if args.workers == 1:
            converted = map(_export_ms_swift_worker, payloads)
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            converted = executor.map(_export_ms_swift_worker, payloads, chunksize=args.chunksize)
        with atomic_text_writer(args.output, overwrite=args.overwrite) as stream:
            for line_number, record_id, values, projection_warnings, conversion_error in converted:
                input_records += 1
                if conversion_error is not None:
                    if len(error_examples) < args.max_error_examples:
                        error_examples.append(
                            {"line": line_number, "id": record_id, "error": conversion_error}
                        )
                    if args.unsupported_policy == "error":
                        fatal_error = "line {} record {!r}: {}".format(
                            line_number, record_id, conversion_error
                        )
                        raise MsSwiftConversionError(fatal_error)
                    skipped_records += 1
                    continue
                warnings.update(projection_warnings)
                for value in values or []:
                    stream.write(json_line(value))
                    output_records += 1
    except MsSwiftConversionError:
        pass
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    status = "failed" if fatal_error else ("complete_with_skips" if skipped_records else "complete")
    report = {
        "status": status,
        "input": str(input_path),
        "output": str(args.output.resolve()),
        "input_records": input_records,
        "output_records": output_records if not fatal_error else 0,
        "skipped_records": skipped_records,
        "projection_warnings": dict(sorted(warnings.items())),
        "error_examples": error_examples,
        "errors_truncated": len(error_examples) >= args.max_error_examples,
        "fatal_error": fatal_error,
        "policies": {
            "answer": args.answer_policy,
            "annotations": args.annotation_policy,
            "unsupported": args.unsupported_policy,
        },
    }
    if args.report is not None:
        _write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 1 if fatal_error else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "profile":
        if args.sample_rows <= 0 or args.max_depth <= 0 or args.max_files <= 0 or args.workers <= 0:
            raise SystemExit("profile limits must be greater than zero")
        report = profile_path(
            args.path,
            args.sample_rows,
            args.max_depth,
            args.max_files,
            include_examples=args.include_examples,
            relative_paths=args.relative_paths,
            workers=args.workers,
        )
        _write_json(args.output, report)
        return 0
    if args.command == "validate":
        if args.max_errors <= 0:
            raise SystemExit("--max-errors must be greater than zero")
        return _validate(args)
    if args.command == "validate-manifest":
        if args.max_errors <= 0:
            raise SystemExit("--max-errors must be greater than zero")
        return _validate_manifest_command(args)
    if args.command == "split":
        summary = split_jsonl(
            args.input,
            args.train_output,
            args.val_output,
            val_ratio=args.val_ratio,
            seed=args.seed,
            overwrite=args.overwrite,
            audit_assets=not args.no_asset_audit,
        )
        print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "import-ms-swift":
        if args.workers <= 0 or args.chunksize <= 0:
            raise SystemExit("--workers and --chunksize must be greater than zero")
        return _import_ms_swift(args)
    if args.command == "export-ms-swift":
        if args.max_error_examples <= 0 or args.workers <= 0 or args.chunksize <= 0:
            raise SystemExit("--max-error-examples, --workers, and --chunksize must be greater than zero")
        return _export_ms_swift(args)
    if args.command == "schema-path":
        print(MANIFEST_SCHEMA_PATH if args.kind == "manifest" else SCHEMA_PATH)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
