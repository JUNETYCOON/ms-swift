#!/usr/bin/env python3
"""Command-line tools for profiling, validating, splitting, and adapting S1-UDF."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from universal_dataset.io import atomic_text_writer, iter_jsonl, json_line, parse_json_strict
    from universal_dataset.adapter import (
        AdapterContext, CURRENT_UDF_VERSION, REGISTRY, TargetContext, load_adapter_plugins,
    )
    from universal_dataset.conversion import (
        ConversionConfigurationError, normalize_plugin_references, parse_adapter_options,
        run_source_adapter, run_target_adapter,
    )
    from universal_dataset.manifest import ManifestBuildError, build_manifest
    from universal_dataset.ms_swift import (
        MsSwiftConversionError, ms_swift_projection_warnings, ms_swift_to_record, record_to_ms_swift,
    )
    from universal_dataset.profile import profile_path
    from universal_dataset.split import AssetIdentityRegistry, split_jsonl
    from universal_dataset.validation import (
        MANIFEST_SCHEMA_PATH, SCHEMA_PATH, ensure_valid, validate_manifest, validate_record,
    )
else:
    from .io import atomic_text_writer, iter_jsonl, json_line, parse_json_strict
    from .adapter import AdapterContext, CURRENT_UDF_VERSION, REGISTRY, TargetContext, load_adapter_plugins
    from .conversion import (
        ConversionConfigurationError, normalize_plugin_references, parse_adapter_options,
        run_source_adapter, run_target_adapter,
    )
    from .manifest import ManifestBuildError, build_manifest
    from .ms_swift import (
        MsSwiftConversionError, ms_swift_projection_warnings, ms_swift_to_record, record_to_ms_swift,
    )
    from .profile import profile_path
    from .split import AssetIdentityRegistry, split_jsonl
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
    profile.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="RELATIVE_GLOB",
        help=(
            "Repeatable root-relative glob for explicitly excluded files; "
            "the patterns, counts, and examples are recorded in the report."
        ),
    )
    profile.add_argument(
        "--summary-only",
        action="store_true",
        help="Keep aggregated schemas and issue examples without repeated per-file reports.",
    )
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

    build_manifest_parser = subparsers.add_parser(
        "build-manifest", help="Build a checksummed manifest from validated S1-UDF record files."
    )
    build_manifest_parser.add_argument("record_files", nargs="+", type=Path)
    build_manifest_parser.add_argument("--output", type=Path, required=True)
    build_manifest_parser.add_argument("--dataset-name", required=True)
    build_manifest_parser.add_argument("--dataset-version")
    build_manifest_parser.add_argument("--description")
    build_manifest_parser.add_argument("--homepage")
    build_manifest_parser.add_argument("--license")
    build_manifest_parser.add_argument("--revision")
    build_manifest_parser.add_argument(
        "--source-schema", action="append", default=[], metavar="JSON",
        help="Repeatable JSON object describing an audited source schema.",
    )
    build_manifest_parser.add_argument("--provenance", metavar="JSON")
    build_manifest_parser.add_argument(
        "--record-split",
        action="append",
        default=[],
        metavar="PATH=SPLIT",
        help="Assign a record file to a split explicitly, including an empty split file.",
    )
    build_manifest_parser.add_argument("--no-record-validation", action="store_true")
    build_manifest_parser.add_argument("--check-assets", action="store_true")
    build_manifest_parser.add_argument("--overwrite", action="store_true")

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
    export_swift.add_argument(
        "--drop-ids",
        action="store_true",
        help="Omit the output record id; canonical group_id is always preserved.",
    )
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

    list_adapters = subparsers.add_parser(
        "list-adapters", help="List built-in and explicitly loaded adapter plugins."
    )
    list_adapters.add_argument("--plugin", action="append", default=[])
    list_adapters.add_argument("--direction", choices=("source", "target"))

    inspect_adapter = subparsers.add_parser(
        "inspect-adapter", help="Print one adapter's version, capabilities, and loss contract."
    )
    inspect_adapter.add_argument("name")
    inspect_adapter.add_argument("--direction", choices=("source", "target"), required=True)
    inspect_adapter.add_argument("--plugin", action="append", default=[])

    convert_source = subparsers.add_parser(
        "convert-source", help="Run a registered source adapter into validated S1-UDF JSONL."
    )
    convert_source.add_argument("source", type=Path)
    convert_source.add_argument("--adapter", required=True)
    convert_source.add_argument("--plugin", action="append", default=[])
    convert_source.add_argument("--output", type=Path, required=True)
    convert_source.add_argument("--dataset-name", required=True)
    convert_source.add_argument("--dataset-version")
    convert_source.add_argument("--drop-raw", action="store_true")
    convert_source.add_argument("--option", action="append", default=[], metavar="KEY=JSON")
    convert_source.add_argument("--workers", type=int, default=1)
    convert_source.add_argument("--batch-size", type=int, default=64)
    convert_source.add_argument("--max-pending-batches", type=int)
    convert_source.add_argument("--error-policy", choices=("error", "skip"), default="error")
    convert_source.add_argument("--max-diagnostic-examples", type=int, default=100)
    convert_source.add_argument("--report", type=Path)
    convert_source.add_argument("--no-json-schema", action="store_true")
    convert_source.add_argument("--overwrite", action="store_true")

    convert_target = subparsers.add_parser(
        "convert-target", help="Run a registered target adapter over validated S1-UDF JSONL."
    )
    convert_target.add_argument("input", type=Path)
    convert_target.add_argument("--adapter", required=True)
    convert_target.add_argument("--plugin", action="append", default=[])
    convert_target.add_argument("--output", type=Path, required=True)
    convert_target.add_argument("--option", action="append", default=[], metavar="KEY=JSON")
    convert_target.add_argument("--workers", type=int, default=1)
    convert_target.add_argument("--batch-size", type=int, default=64)
    convert_target.add_argument("--max-pending-batches", type=int)
    convert_target.add_argument("--error-policy", choices=("error", "skip"), default="error")
    convert_target.add_argument("--max-diagnostic-examples", type=int, default=100)
    convert_target.add_argument("--report", type=Path)
    convert_target.add_argument("--no-json-schema", action="store_true")
    convert_target.add_argument("--overwrite", action="store_true")

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
    asset_identities = AssetIdentityRegistry()
    episode_groups: Dict[str, str] = {}
    episode_splits: Dict[str, str] = {}
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
            episode = record.get("episode")
            if isinstance(episode, Mapping):
                episode_id = episode.get("id")
                if isinstance(episode_id, str) and episode_id:
                    previous_group = episode_groups.setdefault(episode_id, group_id)
                    if previous_group != group_id:
                        errors.append(
                            {
                                "line": line_number,
                                "path": "$.group_id",
                                "code": "episode_group_conflict",
                                "message": "episode {!r} belongs to groups {!r} and {!r}".format(
                                    episode_id, previous_group, group_id
                                ),
                            }
                        )
                    if isinstance(split, str):
                        previous_split = episode_splits.setdefault(episode_id, split)
                        if previous_split != split:
                            errors.append(
                                {
                                    "line": line_number,
                                    "path": "$.split",
                                    "code": "episode_split_leakage",
                                    "message": "episode {!r} appears in {!r} and {!r}".format(
                                        episode_id, previous_split, split
                                    ),
                                }
                            )
            for asset in record.get("assets") or []:
                if len(errors) >= args.max_errors:
                    break
                if not isinstance(asset, dict):
                    continue
                try:
                    identities = asset_identities.add(asset, input_path)
                except ValueError as error:
                    errors.append({"line": line_number, "path": "$.assets", "code": "asset_identity", "message": str(error)})
                    continue
                for identity in identities:
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
        "unique_assets": len(asset_identities),
        "unique_episodes": len(episode_groups),
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


def _parse_json_object(value: str, option_name: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ConversionConfigurationError("{} must be valid JSON: {}".format(option_name, error)) from error
    if not isinstance(parsed, dict):
        raise ConversionConfigurationError("{} must be a JSON object".format(option_name))
    return parsed


def _parse_record_splits(values: Sequence[str]) -> Dict[Path, str]:
    result: Dict[Path, str] = {}
    for value in values:
        if "=" not in value:
            raise ConversionConfigurationError(
                "--record-split must use PATH=SPLIT syntax: {!r}".format(value)
            )
        raw_path, raw_split = value.rsplit("=", 1)
        if not raw_path.strip() or not raw_split.strip():
            raise ConversionConfigurationError(
                "--record-split requires a non-empty path and split: {!r}".format(value)
            )
        path = Path(raw_path.strip()).expanduser().resolve()
        split = raw_split.strip()
        previous = result.setdefault(path, split)
        if previous != split:
            raise ConversionConfigurationError(
                "--record-split assigns conflicting splits to {}".format(path)
            )
    return result


def _build_manifest_command(args: argparse.Namespace) -> int:
    source_schemas = [_parse_json_object(value, "--source-schema") for value in args.source_schema]
    provenance = _parse_json_object(args.provenance, "--provenance") if args.provenance else None
    value = build_manifest(
        args.record_files,
        args.output,
        args.dataset_name,
        dataset_version=args.dataset_version,
        description=args.description,
        homepage=args.homepage,
        license_name=args.license,
        revision=args.revision,
        source_schemas=source_schemas,
        provenance=provenance,
        record_splits=_parse_record_splits(args.record_split),
        validate_records=not args.no_record_validation,
        check_assets=args.check_assets,
    )
    with atomic_text_writer(args.output, overwrite=args.overwrite) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.expanduser().resolve()),
                "records": value["statistics"]["records"],
                "files": len(value["record_files"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _load_cli_adapters(plugins: Sequence[str]) -> None:
    if __package__ in (None, ""):
        __import__("universal_dataset.builtin_adapters")
    else:
        __import__("{}.builtin_adapters".format(__package__), fromlist=["*"])
    load_adapter_plugins(normalize_plugin_references(plugins))


def _list_adapters_command(args: argparse.Namespace) -> int:
    _load_cli_adapters(args.plugin)
    print(
        json.dumps(
            {"adapters": [item.as_dict() for item in REGISTRY.specs(args.direction)]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _inspect_adapter_command(args: argparse.Namespace) -> int:
    _load_cli_adapters(args.plugin)
    print(json.dumps(REGISTRY.spec(args.name, args.direction).as_dict(), ensure_ascii=False, indent=2))
    return 0


def _convert_source_command(args: argparse.Namespace) -> int:
    source = args.source.expanduser().resolve()
    source_root = source if source.is_dir() else source.parent
    context = AdapterContext(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        source_root=source_root,
        preserve_raw=not args.drop_raw,
        options=parse_adapter_options(args.option),
        udf_version=CURRENT_UDF_VERSION,
    )
    report = run_source_adapter(
        source,
        args.output,
        args.adapter,
        context,
        plugins=args.plugin,
        workers=args.workers,
        batch_size=args.batch_size,
        max_pending_batches=args.max_pending_batches,
        error_policy=args.error_policy,
        max_diagnostic_examples=args.max_diagnostic_examples,
        report_path=args.report,
        overwrite=args.overwrite,
        check_json_schema=not args.no_json_schema,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 1 if report["status"] == "failed" else 0


def _convert_target_command(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    context = TargetContext(
        output_path=args.output.expanduser().resolve(),
        input_base_dir=input_path.parent,
        options=parse_adapter_options(args.option),
        udf_version=CURRENT_UDF_VERSION,
    )
    report = run_target_adapter(
        input_path,
        args.output,
        args.adapter,
        context,
        plugins=args.plugin,
        workers=args.workers,
        batch_size=args.batch_size,
        max_pending_batches=args.max_pending_batches,
        error_policy=args.error_policy,
        max_diagnostic_examples=args.max_diagnostic_examples,
        report_path=args.report,
        overwrite=args.overwrite,
        check_json_schema=not args.no_json_schema,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 1 if report["status"] == "failed" else 0


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
            summary_only=args.summary_only,
            exclude_patterns=args.exclude,
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
    if args.command == "build-manifest":
        return _build_manifest_command(args)
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
    if args.command == "list-adapters":
        return _list_adapters_command(args)
    if args.command == "inspect-adapter":
        return _inspect_adapter_command(args)
    if args.command == "convert-source":
        return _convert_source_command(args)
    if args.command == "convert-target":
        return _convert_target_command(args)
    if args.command == "schema-path":
        print(MANIFEST_SCHEMA_PATH if args.kind == "manifest" else SCHEMA_PATH)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
