from __future__ import annotations

import copy
import json
import sys
import tempfile
import textwrap
import unittest
import uuid
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from universal_dataset.adapter import (
    AdapterRegistry,
    AdapterSpec,
    SourceAdapter,
    provisional_media_group_id,
    stable_record_id,
)
from universal_dataset.conversion import (
    ConversionConfigurationError,
    parse_adapter_options,
    run_source_adapter,
    run_target_adapter,
)
from universal_dataset.adapter import AdapterContext, TargetContext, load_adapter_plugin
from universal_dataset.io import atomic_text_writer, iter_jsonl, json_line
from universal_dataset.manifest import ManifestBuildError, build_manifest
from universal_dataset.validation import validate_manifest


def text_record(identifier: str, group: str, split: str = "train"):
    return {
        "format": "s1-udf",
        "schema_version": "1.0.0",
        "id": identifier,
        "group_id": group,
        "split": split,
        "task_types": ["dialogue"],
        "provenance": {
            "dataset": "toy",
            "source_record_id": identifier,
            "conversion": {"tool": "test-fixture", "version": "1.0.0"},
        },
        "conversations": [
            {
                "id": "conversation:0",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": identifier}],
                    }
                ],
            }
        ],
    }


class AdapterIdentityTest(unittest.TestCase):
    def test_missing_media_identities_use_fallback(self):
        self.assertEqual(provisional_media_group_id([None, "", "  "], "row-1"), "text:row-1")
        with self.assertRaises(ValueError):
            stable_record_id("dataset", None)
        with self.assertRaises(ValueError):
            provisional_media_group_id([], "")

    def test_registry_rejects_incompatible_adapter(self):
        class Incompatible(SourceAdapter):
            spec = AdapterSpec(
                name="incompatible", version="1", direction="source", udf_versions=("9.9.9",)
            )

            def iter_source(self, source, context):
                return iter(())

            def convert(self, source_id, value, context):
                raise AssertionError

        with self.assertRaises(ValueError):
            AdapterRegistry().register_source(Incompatible)

    def test_adapter_options_are_typed_and_unique(self):
        self.assertEqual(
            parse_adapter_options(["flag=true", "count=3", 'name="x"']),
            {"flag": True, "count": 3, "name": "x"},
        )
        with self.assertRaises(ValueError):
            parse_adapter_options(["x=1", "x=2"])


class AdapterRunnerTest(unittest.TestCase):
    def _plugin(self, directory: Path):
        suffix = uuid.uuid4().hex
        source_name = "toy-source-" + suffix
        target_name = "toy-target-" + suffix
        plugin = directory / "toy_plugin.py"
        plugin.write_text(
            textwrap.dedent(
                """
                from pathlib import Path
                from universal_dataset.adapter import (
                    AdapterDiagnostic, AdapterResult, AdapterSpec, SourceAdapter, TargetAdapter,
                    TargetAdapterResult, register_source_adapter, register_target_adapter,
                )
                from universal_dataset.io import iter_jsonl

                @register_source_adapter
                class ToySource(SourceAdapter):
                    spec = AdapterSpec(name=%r, version="1.0", direction="source", capabilities=("dialogue",))
                    def iter_source(self, source: Path, context):
                        for line, value in iter_jsonl(source):
                            yield str(value.get("source_id") or line), value
                    def convert(self, source_id, value, context):
                        if value.get("interrupt"):
                            raise KeyboardInterrupt("requested interrupt")
                        if value.get("invalid"):
                            raise ValueError("requested invalid row")
                        identifier = str(value.get("output_id") or (context.dataset_name + ":" + source_id))
                        diagnostics = []
                        if value.get("diagnostic_error"):
                            diagnostics.append(AdapterDiagnostic(
                                stage="source_convert", code="declared_error", severity="error",
                                message="adapter rejected this item",
                            ))
                        record = {
                            "format": "s1-udf", "schema_version": "1.0.0",
                            "id": identifier, "group_id": "text:" + source_id,
                            "task_types": ["dialogue"],
                            "provenance": {
                                "dataset": "toy", "source_record_id": source_id,
                                "conversion": {"tool": "toy-source", "version": "1.0"},
                            },
                            "conversations": [{"id": "c", "messages": [{"role": "assistant", "content": [{"type": "text", "text": source_id}]}]}]
                        }
                        if value.get("non_json_raw"):
                            record["provenance"]["raw_record"] = b"bytes"
                        return AdapterResult(records=[record], diagnostics=diagnostics)

                @register_target_adapter
                class ToyTarget(TargetAdapter):
                    spec = AdapterSpec(name=%r, version="1.0", direction="target", capabilities=("dialogue",))
                    def convert(self, record, context):
                        return TargetAdapterResult(values=[{"source_record_id": record["id"]}])
                    def write_output(self, values, context, overwrite=False):
                        import json
                        from universal_dataset.io import atomic_text_writer
                        if context.options.get("do_not_consume"):
                            return 0
                        if context.options.get("consume_one"):
                            next(iter(values), None)
                            return 1
                        values = list(values)
                        with atomic_text_writer(context.output_path, overwrite=overwrite) as stream:
                            json.dump(values, stream, sort_keys=True)
                            stream.write("\\n")
                        return len(values)
                """ % (source_name, target_name)
            ),
            encoding="utf-8",
        )
        return plugin, source_name, target_name

    def test_plugin_parallel_output_is_deterministic_and_target_is_symmetric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin, source_name, target_name = self._plugin(root)
            load_adapter_plugin(str(plugin))
            source = root / "source.jsonl"
            source.write_text(
                "".join(json_line({"source_id": str(index)}) for index in range(9)),
                encoding="utf-8",
            )
            outputs = []
            for workers in (1, 2):
                output = root / "out-{}.jsonl".format(workers)
                report = run_source_adapter(
                    source,
                    output,
                    source_name,
                    AdapterContext(dataset_name="toy", source_root=root),
                    plugins=[str(plugin)],
                    workers=workers,
                    batch_size=2,
                    max_pending_batches=2,
                )
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["input_items"], 9)
                outputs.append(output.read_bytes())
            self.assertEqual(outputs[0], outputs[1])

            target = root / "target.jsonl"
            target_report = run_target_adapter(
                root / "out-2.jsonl",
                target,
                target_name,
                TargetContext(output_path=target, input_base_dir=root),
                plugins=[str(plugin)],
                workers=2,
                batch_size=3,
                max_pending_batches=2,
            )
            self.assertEqual(target_report["status"], "complete")
            self.assertEqual(target_report["output_records"], 9)
            self.assertEqual(len(json.loads(target.read_text(encoding="utf-8"))), 9)

    def test_error_report_is_structured_and_output_is_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin, source_name, _ = self._plugin(root)
            load_adapter_plugin(str(plugin))
            source = root / "source.jsonl"
            source.write_text(
                json_line({"source_id": "ok"}) + json_line({"source_id": "bad", "invalid": True}),
                encoding="utf-8",
            )
            output = root / "output.jsonl"
            report_path = root / "report.json"
            report = run_source_adapter(
                source,
                output,
                source_name,
                AdapterContext(dataset_name="toy", source_root=root),
                plugins=[str(plugin)],
                report_path=report_path,
                error_policy="error",
            )
            self.assertEqual(report["status"], "failed")
            self.assertFalse(output.exists())
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["diagnostics"]["code_counts"]["adapter_exception"], 1)

            with self.assertRaisesRegex(ConversionConfigurationError, "report must differ"):
                run_source_adapter(
                    source,
                    output,
                    source_name,
                    AdapterContext(dataset_name="toy", source_root=root),
                    plugins=[str(plugin)],
                    report_path=output,
                )
            with self.assertRaisesRegex(ConversionConfigurationError, "output must not overwrite"):
                run_source_adapter(
                    source,
                    source,
                    source_name,
                    AdapterContext(dataset_name="toy", source_root=root),
                    plugins=[str(plugin)],
                )

    def test_error_diagnostic_obeys_policy_and_interrupt_is_not_swallowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin, source_name, _ = self._plugin(root)
            source = root / "source.jsonl"
            source.write_text(
                json_line({"source_id": "ok"})
                + json_line({"source_id": "bad", "diagnostic_error": True}),
                encoding="utf-8",
            )
            failed_output = root / "failed.jsonl"
            failed = run_source_adapter(
                source,
                failed_output,
                source_name,
                AdapterContext(dataset_name="toy", source_root=root),
                plugins=[str(plugin)],
            )
            self.assertEqual(failed["status"], "failed")
            self.assertFalse(failed_output.exists())

            skipped_output = root / "skipped.jsonl"
            skipped = run_source_adapter(
                source,
                skipped_output,
                source_name,
                AdapterContext(dataset_name="toy", source_root=root),
                plugins=[str(plugin)],
                error_policy="skip",
            )
            self.assertEqual(skipped["status"], "complete_with_skips")
            self.assertEqual(skipped["skipped_items"], 1)
            self.assertEqual(
                [value["id"] for _, value in iter_jsonl(skipped_output)], ["toy:ok"]
            )

            source.write_text(
                json_line({"source_id": "ok"})
                + json_line({"source_id": "bytes", "non_json_raw": True}),
                encoding="utf-8",
            )
            non_json_output = root / "non-json-skipped.jsonl"
            non_json = run_source_adapter(
                source,
                non_json_output,
                source_name,
                AdapterContext(dataset_name="toy", source_root=root),
                plugins=[str(plugin)],
                error_policy="skip",
            )
            self.assertEqual(non_json["status"], "complete_with_skips")
            self.assertEqual(non_json["skipped_items"], 1)
            self.assertEqual(
                [value["id"] for _, value in iter_jsonl(non_json_output)], ["toy:ok"]
            )

            source.write_text(
                json_line({"source_id": "interrupt", "interrupt": True}), encoding="utf-8"
            )
            with self.assertRaises(KeyboardInterrupt):
                run_source_adapter(
                    source,
                    root / "interrupt.jsonl",
                    source_name,
                    AdapterContext(dataset_name="toy", source_root=root),
                    plugins=[str(plugin)],
                )

    def test_target_rejects_duplicate_ids_and_non_consuming_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin, _, target_name = self._plugin(root)
            records = root / "records.jsonl"
            records.write_text(
                json_line(text_record("duplicate", "group-a"))
                + json_line(text_record("duplicate", "group-b")),
                encoding="utf-8",
            )
            duplicate_output = root / "duplicates.json"
            duplicate_report = run_target_adapter(
                records,
                duplicate_output,
                target_name,
                TargetContext(output_path=duplicate_output, input_base_dir=root),
                plugins=[str(plugin)],
            )
            self.assertEqual(duplicate_report["status"], "failed")
            self.assertFalse(duplicate_output.exists())

            unique = root / "unique.jsonl"
            unique.write_text(json_line(text_record("one", "group-one")), encoding="utf-8")
            ignored_output = root / "ignored.json"
            ignored_report = run_target_adapter(
                unique,
                ignored_output,
                target_name,
                TargetContext(
                    output_path=ignored_output,
                    input_base_dir=root,
                    options={"do_not_consume": True},
                ),
                plugins=[str(plugin)],
            )
            self.assertEqual(ignored_report["status"], "failed")
            self.assertIn("complete value stream", ignored_report["fatal_error"])

            two = root / "two.jsonl"
            two.write_text(
                json_line(text_record("one", "group-one"))
                + json_line(text_record("two", "group-two")),
                encoding="utf-8",
            )
            partial_output = root / "partial.json"
            partial_report = run_target_adapter(
                two,
                partial_output,
                target_name,
                TargetContext(
                    output_path=partial_output,
                    input_base_dir=root,
                    options={"consume_one": True},
                ),
                plugins=[str(plugin)],
            )
            self.assertEqual(partial_report["status"], "failed")
            self.assertIn("complete value stream", partial_report["fatal_error"])


class AtomicWriterTest(unittest.TestCase):
    def test_no_overwrite_refuses_target_created_during_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.txt"
            with self.assertRaisesRegex(FileExistsError, "appeared while writing"):
                with atomic_text_writer(path) as stream:
                    stream.write("generated")
                    path.write_text("concurrent", encoding="utf-8")
            self.assertEqual(path.read_text(encoding="utf-8"), "concurrent")


class ManifestBuildTest(unittest.TestCase):
    def test_build_and_validate_manifest_from_record_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.jsonl"
            records.write_text(
                json_line(text_record("a", "group-a", "train"))
                + json_line(text_record("b", "group-b", "val")),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            value = build_manifest([records], manifest_path, "toy")
            self.assertEqual(value["splits"], {"train": 1, "val": 1})
            self.assertEqual(value["task_types"], ["dialogue"])
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                validate_manifest(value, manifest_path=manifest_path, check_files=True), []
            )

            wrong = copy.deepcopy(value)
            wrong["splits"] = {"train": 999, "val": 999}
            codes = {
                issue.code
                for issue in validate_manifest(wrong, manifest_path=manifest_path, check_files=True)
            }
            self.assertIn("count_mismatch", codes)

            wrong_statistics = copy.deepcopy(value)
            wrong_statistics["statistics"] = {
                "records": 999,
                "groups": 999,
                "unique_assets": 999,
                "episodes": 999,
            }
            statistic_issues = validate_manifest(
                wrong_statistics, manifest_path=manifest_path, check_files=True
            )
            self.assertEqual(
                {issue.path for issue in statistic_issues if issue.code == "statistics_mismatch"},
                {
                    "$.statistics.records",
                    "$.statistics.groups",
                    "$.statistics.unique_assets",
                    "$.statistics.episodes",
                },
            )

    def test_manifest_build_rejects_cross_split_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = root / "image.jpg"
            media.write_bytes(b"same")
            left = text_record("a", "group-a", "train")
            right = text_record("b", "group-b", "val")
            for record in (left, right):
                record["assets"] = [{"id": "image", "kind": "image", "uri": str(media)}]
            left["assets"][0]["sha256"] = "a" * 64
            records = root / "records.jsonl"
            records.write_text(json_line(left) + json_line(right), encoding="utf-8")
            with self.assertRaises(ManifestBuildError):
                build_manifest([records], root / "manifest.json", "toy")

    def test_manifest_build_rejects_invalid_metadata_and_output_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.jsonl"
            records.write_text(json_line(text_record("a", "group-a")), encoding="utf-8")
            with self.assertRaisesRegex(ManifestBuildError, "violates the S1-UDF schema"):
                build_manifest(
                    [records],
                    root / "manifest.json",
                    "toy",
                    source_schemas=[{}],
                )
            with self.assertRaisesRegex(ManifestBuildError, "must differ"):
                build_manifest([records], records, "toy")


if __name__ == "__main__":
    unittest.main()
