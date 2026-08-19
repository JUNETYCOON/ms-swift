"""Deterministic bounded-parallel runners for registered source and target adapters."""

from __future__ import annotations

import importlib
import json
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .adapter import (
    AdapterContext,
    AdapterDiagnostic,
    AdapterResult,
    CURRENT_UDF_VERSION,
    REGISTRY,
    TargetAdapterResult,
    TargetContext,
    load_adapter_plugins,
)
from .io import atomic_text_writer, iter_jsonl, json_line
from .validation import DatasetValidationError, ensure_valid


class ConversionConfigurationError(ValueError):
    pass


class _AbortConversion(RuntimeError):
    pass


_WORKER_ADAPTER: Any = None
_WORKER_CONTEXT: Any = None
_WORKER_CHECK_SCHEMA = True


def _load_adapters(plugins: Sequence[str]) -> None:
    importlib.import_module("{}.builtin_adapters".format(__package__))
    load_adapter_plugins(plugins)


def normalize_plugin_references(plugins: Sequence[str]) -> Tuple[str, ...]:
    values: List[str] = []
    for reference in plugins:
        candidate = Path(reference).expanduser()
        if candidate.suffix.lower() == ".py" or candidate.exists():
            values.append(str(candidate.resolve()))
        else:
            values.append(reference)
    return tuple(values)


def parse_adapter_options(values: Sequence[str]) -> Dict[str, Any]:
    """Parse repeatable KEY=JSON options without inventing string coercions."""

    result: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ConversionConfigurationError(
                "Adapter option must use KEY=JSON syntax: {!r}".format(item)
            )
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key or key in result:
            raise ConversionConfigurationError(
                "Adapter option keys must be non-empty and unique: {!r}".format(key)
            )
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ConversionConfigurationError(
                "Adapter option {!r} is not valid JSON: {}".format(key, error)
            ) from error
    return result


def _source_initializer(
    adapter_name: str,
    plugins: Sequence[str],
    context: AdapterContext,
    check_schema: bool,
) -> None:
    global _WORKER_ADAPTER, _WORKER_CONTEXT, _WORKER_CHECK_SCHEMA
    _load_adapters(plugins)
    _WORKER_ADAPTER = REGISTRY.create_source(adapter_name)
    _WORKER_CONTEXT = context
    _WORKER_CHECK_SCHEMA = check_schema


def _target_initializer(
    adapter_name: str,
    plugins: Sequence[str],
    context: TargetContext,
    check_schema: bool,
) -> None:
    global _WORKER_ADAPTER, _WORKER_CONTEXT, _WORKER_CHECK_SCHEMA
    _load_adapters(plugins)
    _WORKER_ADAPTER = REGISTRY.create_target(adapter_name)
    _WORKER_CONTEXT = context
    _WORKER_CHECK_SCHEMA = check_schema


def _diagnostic_from_exception(
    error: Exception,
    stage: str,
    source_locator: str,
    record_id: Optional[str] = None,
) -> AdapterDiagnostic:
    if isinstance(error, DatasetValidationError):
        return AdapterDiagnostic(
            stage=stage,
            code="record_validation",
            severity="error",
            message=str(error),
            source_locator=source_locator,
            record_id=record_id,
            adapter=_WORKER_ADAPTER.spec.name,
            details={
                "issues": [
                    {"path": issue.path, "code": issue.code, "message": issue.message}
                    for issue in error.issues[:20]
                ],
                "issues_truncated": len(error.issues) > 20,
            },
        )
    return AdapterDiagnostic(
        stage=stage,
        code="adapter_exception",
        severity="error",
        message="{}: {}".format(type(error).__name__, error),
        source_locator=source_locator,
        record_id=record_id,
        adapter=_WORKER_ADAPTER.spec.name,
    )


def _normalize_diagnostics(
    diagnostics: Sequence[AdapterDiagnostic], warnings: Sequence[str], source_locator: str,
    record_id: Optional[str] = None,
) -> List[AdapterDiagnostic]:
    values: List[AdapterDiagnostic] = []
    for item in diagnostics:
        if not isinstance(item, AdapterDiagnostic):
            raise TypeError("Adapter diagnostics must contain AdapterDiagnostic values")
        values.append(
            replace(
                item,
                source_locator=item.source_locator or source_locator,
                record_id=item.record_id or record_id,
                adapter=item.adapter or _WORKER_ADAPTER.spec.name,
            )
        )
    for warning in warnings:
        values.append(
            AdapterDiagnostic(
                stage="convert",
                code="adapter_warning",
                severity="warning",
                message=str(warning),
                source_locator=source_locator,
                record_id=record_id,
                adapter=_WORKER_ADAPTER.spec.name,
            )
        )
    return values


def _source_batch_worker(
    batch: Sequence[Tuple[int, str, Mapping[str, Any]]]
) -> List[Dict[str, Any]]:
    outcomes: List[Dict[str, Any]] = []
    for sequence, source_id, value in batch:
        locator = str(source_id)
        try:
            if not locator.strip():
                raise ValueError("source_id must be non-empty")
            result = _WORKER_ADAPTER.convert(locator, value, _WORKER_CONTEXT)
            if not isinstance(result, AdapterResult):
                raise TypeError("Source adapter convert() must return AdapterResult")
            diagnostics = _normalize_diagnostics(result.diagnostics, result.warnings, locator)
            records: List[Dict[str, Any]] = []
            for record in result.records:
                if not isinstance(record, dict):
                    raise TypeError("Source adapter records must be objects")
                ensure_valid(
                    record,
                    check_json_schema=_WORKER_CHECK_SCHEMA,
                    base_dir=_WORKER_CONTEXT.source_root,
                )
                # Fail per source item before the main-process writer sees a non-JSON value.
                json_line(record)
                records.append(record)
            if not records:
                raise ValueError("Source adapter emitted no records")
            adapter_errors = [item for item in diagnostics if item.severity == "error"]
            outcomes.append(
                {
                    "sequence": sequence,
                    "source_locator": locator,
                    "values": records,
                    "diagnostics": diagnostics,
                    "error": adapter_errors[0] if adapter_errors else None,
                }
            )
        except MemoryError:
            raise
        except Exception as error:
            outcomes.append(
                {
                    "sequence": sequence,
                    "source_locator": locator,
                    "values": [],
                    "diagnostics": [],
                    "error": _diagnostic_from_exception(error, "source_convert", locator),
                }
            )
    return outcomes


def _target_batch_worker(
    batch: Sequence[Tuple[int, int, Mapping[str, Any]]]
) -> List[Dict[str, Any]]:
    outcomes: List[Dict[str, Any]] = []
    for sequence, line_number, record in batch:
        record_id = record.get("id") if isinstance(record.get("id"), str) else None
        locator = "line:{}".format(line_number)
        try:
            ensure_valid(
                record,
                check_json_schema=_WORKER_CHECK_SCHEMA,
                base_dir=_WORKER_CONTEXT.input_base_dir,
            )
            result = _WORKER_ADAPTER.convert(record, _WORKER_CONTEXT)
            if not isinstance(result, TargetAdapterResult):
                raise TypeError("Target adapter convert() must return TargetAdapterResult")
            diagnostics = _normalize_diagnostics(
                result.diagnostics, result.warnings, locator, record_id=record_id
            )
            values: List[Dict[str, Any]] = []
            for value in result.values:
                if not isinstance(value, dict):
                    raise TypeError("Target adapter values must be objects")
                json_line(value)
                values.append(value)
            if not values:
                raise ValueError("Target adapter emitted no records")
            adapter_errors = [item for item in diagnostics if item.severity == "error"]
            outcomes.append(
                {
                    "sequence": sequence,
                    "source_locator": locator,
                    "record_id": record_id,
                    "values": values,
                    "diagnostics": diagnostics,
                    "error": adapter_errors[0] if adapter_errors else None,
                }
            )
        except MemoryError:
            raise
        except Exception as error:
            outcomes.append(
                {
                    "sequence": sequence,
                    "source_locator": locator,
                    "record_id": record_id,
                    "values": [],
                    "diagnostics": [],
                    "error": _diagnostic_from_exception(
                        error, "target_convert", locator, record_id=record_id
                    ),
                }
            )
    return outcomes


def _batches(values: Iterable[Any], size: int) -> Iterator[List[Any]]:
    batch: List[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _ordered_parallel(
    worker: Any,
    batches: Iterable[List[Any]],
    executor: ProcessPoolExecutor,
    max_pending_batches: int,
) -> Iterator[Dict[str, Any]]:
    iterator = iter(batches)
    pending: Deque[Future] = deque()
    for _ in range(max_pending_batches):
        try:
            pending.append(executor.submit(worker, next(iterator)))
        except StopIteration:
            break
    while pending:
        for outcome in pending.popleft().result():
            yield outcome
        try:
            pending.append(executor.submit(worker, next(iterator)))
        except StopIteration:
            pass


class _ReportBuilder:
    def __init__(
        self,
        direction: str,
        adapter_spec: Mapping[str, Any],
        input_path: Path,
        output_path: Path,
        error_policy: str,
        max_examples: int,
        workers: int,
        batch_size: int,
        max_pending_batches: int,
    ) -> None:
        self.direction = direction
        self.adapter_spec = dict(adapter_spec)
        self.input_path = input_path
        self.output_path = output_path
        self.error_policy = error_policy
        self.max_examples = max_examples
        self.input_items = 0
        self.output_records = 0
        self.skipped_items = 0
        self.severity_counts: Counter = Counter()
        self.code_counts: Counter = Counter()
        self.examples: List[Dict[str, Any]] = []
        self.fatal_error: Optional[str] = None
        self.workers = workers
        self.batch_size = batch_size
        self.max_pending_batches = max_pending_batches

    def add(self, diagnostic: AdapterDiagnostic) -> None:
        self.severity_counts[diagnostic.severity] += 1
        self.code_counts[diagnostic.code] += 1
        if len(self.examples) < self.max_examples:
            self.examples.append(diagnostic.as_dict())

    def result(self) -> Dict[str, Any]:
        if self.fatal_error:
            status = "failed"
        elif self.skipped_items:
            status = "complete_with_skips"
        else:
            status = "complete"
        return {
            "status": status,
            "direction": self.direction,
            "adapter": self.adapter_spec,
            "udf_version": CURRENT_UDF_VERSION,
            "input": str(self.input_path),
            "output": str(self.output_path),
            "input_items": self.input_items,
            "output_records": 0 if self.fatal_error else self.output_records,
            "skipped_items": self.skipped_items,
            "diagnostics": {
                "severity_counts": dict(sorted(self.severity_counts.items())),
                "code_counts": dict(sorted(self.code_counts.items())),
                "examples": self.examples,
                "examples_truncated": sum(self.severity_counts.values()) > len(self.examples),
            },
            "fatal_error": self.fatal_error,
            "policies": {"errors": self.error_policy},
            "parallelism": {
                "workers": self.workers,
                "batch_size": self.batch_size,
                "max_pending_batches": self.max_pending_batches,
            },
        }


def _write_report(
    path: Optional[Path], report: Mapping[str, Any], overwrite: bool = False
) -> None:
    if path is None:
        return
    with atomic_text_writer(path, overwrite=overwrite) as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _validate_conversion_paths(
    input_path: Path,
    output_path: Path,
    report_path: Optional[Path],
    overwrite: bool,
) -> Optional[Path]:
    if input_path == output_path:
        raise ConversionConfigurationError("Conversion output must not overwrite its input")
    if report_path is None:
        return None
    resolved_report = report_path.expanduser().resolve()
    if resolved_report in (input_path, output_path):
        raise ConversionConfigurationError(
            "Conversion report must differ from both input and output paths"
        )
    if resolved_report.exists() and not overwrite:
        raise ConversionConfigurationError(
            "Report already exists: {}. Use --overwrite to replace it.".format(resolved_report)
        )
    return resolved_report


def _validate_runner_options(
    workers: int,
    batch_size: int,
    max_pending_batches: Optional[int],
    error_policy: str,
    max_diagnostic_examples: int,
) -> int:
    if workers <= 0 or batch_size <= 0 or max_diagnostic_examples <= 0:
        raise ConversionConfigurationError(
            "workers, batch_size, and max_diagnostic_examples must be greater than zero"
        )
    if error_policy not in ("error", "skip"):
        raise ConversionConfigurationError("error_policy must be error or skip")
    pending = max_pending_batches if max_pending_batches is not None else max(1, workers * 2)
    if pending <= 0:
        raise ConversionConfigurationError("max_pending_batches must be greater than zero")
    return pending


def run_source_adapter(
    source: Path,
    output: Path,
    adapter_name: str,
    context: AdapterContext,
    plugins: Sequence[str] = (),
    workers: int = 1,
    batch_size: int = 64,
    max_pending_batches: Optional[int] = None,
    error_policy: str = "error",
    max_diagnostic_examples: int = 100,
    report_path: Optional[Path] = None,
    overwrite: bool = False,
    check_json_schema: bool = True,
) -> Dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    report_path = _validate_conversion_paths(source, output, report_path, overwrite)
    pending = _validate_runner_options(
        workers, batch_size, max_pending_batches, error_policy, max_diagnostic_examples
    )
    plugin_values = normalize_plugin_references(plugins)
    _load_adapters(plugin_values)
    adapter = REGISTRY.create_source(adapter_name)
    if not adapter.spec.supports(context.udf_version):
        raise ConversionConfigurationError(
            "Adapter {!r} does not support S1-UDF {}".format(adapter_name, context.udf_version)
        )
    report = _ReportBuilder(
        "source_to_udf", adapter.spec.as_dict(), source, output, error_policy,
        max_diagnostic_examples, workers, batch_size, pending,
    )
    executor: Optional[ProcessPoolExecutor] = None
    interrupted = False
    converted: Iterable[Dict[str, Any]]
    try:
        source_values = (
            (sequence, source_id, value)
            for sequence, (source_id, value) in enumerate(adapter.iter_source(source, context))
        )
        batches = _batches(source_values, batch_size)
        if workers == 1:
            _source_initializer(adapter_name, plugin_values, context, check_json_schema)
            converted = (outcome for batch in batches for outcome in _source_batch_worker(batch))
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_source_initializer,
                initargs=(adapter_name, plugin_values, context, check_json_schema),
            )
            converted = _ordered_parallel(_source_batch_worker, batches, executor, pending)
        seen_ids = set()
        try:
            with atomic_text_writer(output, overwrite=overwrite) as stream:
                for outcome in converted:
                    report.input_items += 1
                    for diagnostic in outcome["diagnostics"]:
                        report.add(diagnostic)
                    error = outcome["error"]
                    if error is not None:
                        if error not in outcome["diagnostics"]:
                            report.add(error)
                        report.skipped_items += 1
                        if error_policy == "error":
                            report.fatal_error = error.message
                            raise _AbortConversion(error.message)
                        continue
                    records = outcome["values"]
                    record_ids = [record.get("id") for record in records]
                    duplicate_id = next(
                        (
                            record_id
                            for position, record_id in enumerate(record_ids)
                            if record_id in seen_ids or record_id in record_ids[:position]
                        ),
                        None,
                    )
                    if duplicate_id is not None:
                        diagnostic = AdapterDiagnostic(
                            stage="dataset_finalize",
                            code="duplicate_record_id",
                            severity="error",
                            message="duplicate output record id {!r}".format(duplicate_id),
                            source_locator=outcome["source_locator"],
                            record_id=str(duplicate_id),
                            adapter=adapter_name,
                        )
                        report.add(diagnostic)
                        report.skipped_items += 1
                        if error_policy == "error":
                            report.fatal_error = diagnostic.message
                            raise _AbortConversion(diagnostic.message)
                        continue
                    seen_ids.update(record_ids)
                    for record in records:
                        stream.write(json_line(record))
                        report.output_records += 1
                if report.input_items == 0:
                    diagnostic = AdapterDiagnostic(
                        stage="source_parse",
                        code="empty_dataset",
                        severity="error",
                        message="source adapter produced no input items",
                        adapter=adapter_name,
                    )
                    report.add(diagnostic)
                    report.fatal_error = diagnostic.message
                    raise _AbortConversion(diagnostic.message)
        except _AbortConversion:
            pass
    except MemoryError:
        interrupted = True
        raise
    except Exception as error:
        if not isinstance(error, _AbortConversion):
            diagnostic = AdapterDiagnostic(
                stage="source_parse",
                code="source_iterator_exception",
                severity="error",
                message="{}: {}".format(type(error).__name__, error),
                adapter=adapter_name,
            )
            report.add(diagnostic)
            report.fatal_error = diagnostic.message
    except BaseException:
        interrupted = True
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=not interrupted, cancel_futures=True)
    result = report.result()
    _write_report(report_path, result, overwrite=overwrite)
    return result


def run_target_adapter(
    input_path: Path,
    output: Path,
    adapter_name: str,
    context: TargetContext,
    plugins: Sequence[str] = (),
    workers: int = 1,
    batch_size: int = 64,
    max_pending_batches: Optional[int] = None,
    error_policy: str = "error",
    max_diagnostic_examples: int = 100,
    report_path: Optional[Path] = None,
    overwrite: bool = False,
    check_json_schema: bool = True,
) -> Dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    output = output.expanduser().resolve()
    report_path = _validate_conversion_paths(input_path, output, report_path, overwrite)
    context = TargetContext(
        output_path=output,
        input_base_dir=context.input_base_dir,
        options=context.options,
        udf_version=context.udf_version,
    )
    pending = _validate_runner_options(
        workers, batch_size, max_pending_batches, error_policy, max_diagnostic_examples
    )
    plugin_values = normalize_plugin_references(plugins)
    _load_adapters(plugin_values)
    adapter = REGISTRY.create_target(adapter_name)
    if not adapter.spec.supports(context.udf_version):
        raise ConversionConfigurationError(
            "Adapter {!r} does not support S1-UDF {}".format(adapter_name, context.udf_version)
        )
    report = _ReportBuilder(
        "udf_to_target", adapter.spec.as_dict(), input_path, output, error_policy,
        max_diagnostic_examples, workers, batch_size, pending,
    )
    executor: Optional[ProcessPoolExecutor] = None
    interrupted = False
    converted: Iterable[Dict[str, Any]]
    try:
        source_values = (
            (sequence, line_number, record)
            for sequence, (line_number, record) in enumerate(iter_jsonl(input_path))
        )
        batches = _batches(source_values, batch_size)
        if workers == 1:
            _target_initializer(adapter_name, plugin_values, context, check_json_schema)
            converted = (outcome for batch in batches for outcome in _target_batch_worker(batch))
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_target_initializer,
                initargs=(adapter_name, plugin_values, context, check_json_schema),
            )
            converted = _ordered_parallel(_target_batch_worker, batches, executor, pending)
        seen_record_ids = set()
        conversion_exhausted = False

        def output_values() -> Iterator[Mapping[str, Any]]:
            nonlocal conversion_exhausted
            for outcome in converted:
                report.input_items += 1
                for diagnostic in outcome["diagnostics"]:
                    report.add(diagnostic)
                error = outcome["error"]
                record_id = outcome.get("record_id")
                if isinstance(record_id, str):
                    if record_id in seen_record_ids:
                        duplicate = AdapterDiagnostic(
                            stage="dataset_validate",
                            code="duplicate_record_id",
                            severity="error",
                            message="duplicate input record id {!r}".format(record_id),
                            source_locator=outcome["source_locator"],
                            record_id=record_id,
                            adapter=adapter_name,
                        )
                        report.add(duplicate)
                        report.skipped_items += 1
                        if error_policy == "error":
                            report.fatal_error = duplicate.message
                            raise _AbortConversion(duplicate.message)
                        continue
                    seen_record_ids.add(record_id)
                if error is not None:
                    if error not in outcome["diagnostics"]:
                        report.add(error)
                    report.skipped_items += 1
                    if error_policy == "error":
                        report.fatal_error = error.message
                        raise _AbortConversion(error.message)
                    continue
                for value in outcome["values"]:
                    report.output_records += 1
                    yield value
            if report.input_items == 0:
                diagnostic = AdapterDiagnostic(
                    stage="input_parse",
                    code="empty_dataset",
                    severity="error",
                    message="input S1-UDF JSONL contains no records",
                    adapter=adapter_name,
                )
                report.add(diagnostic)
                report.fatal_error = diagnostic.message
                raise _AbortConversion(diagnostic.message)
            conversion_exhausted = True

        try:
            written = adapter.write_output(output_values(), context, overwrite=overwrite)
            if not conversion_exhausted:
                raise ValueError(
                    "Target adapter write_output() returned before consuming the complete value stream"
                )
            if not isinstance(written, int) or isinstance(written, bool) or written < 0:
                raise ValueError("Target adapter write_output() must return a non-negative integer")
            if written != report.output_records:
                raise ValueError(
                    "Target writer consumed {}, expected {} projected values".format(
                        written, report.output_records
                    )
                )
        except _AbortConversion:
            pass
    except MemoryError:
        interrupted = True
        raise
    except Exception as error:
        if not isinstance(error, _AbortConversion):
            diagnostic = AdapterDiagnostic(
                stage="input_parse",
                code="input_iterator_exception",
                severity="error",
                message="{}: {}".format(type(error).__name__, error),
                adapter=adapter_name,
            )
            report.add(diagnostic)
            report.fatal_error = diagnostic.message
    except BaseException:
        interrupted = True
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=not interrupted, cancel_futures=True)
    result = report.result()
    _write_report(report_path, result, overwrite=overwrite)
    return result
