from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from universal_dataset.io import iter_jsonl, parse_json_strict, write_jsonl
from universal_dataset.cli import main as cli_main
from universal_dataset.manifest import build_manifest
from universal_dataset.ms_swift import MsSwiftConversionError, ms_swift_to_record, record_to_ms_swift
from universal_dataset.profile import profile_path
from universal_dataset.split import canonical_asset_identity, split_jsonl
from universal_dataset.validation import MANIFEST_SCHEMA_PATH, SCHEMA_PATH, validate_manifest, validate_record


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "representative.jsonl"


def example_records():
    return [record for _, record in iter_jsonl(EXAMPLES)]


class UniversalDatasetTest(unittest.TestCase):
    def test_representative_records_pass_semantic_validation(self):
        records = example_records()
        self.assertEqual(len(records), 5)
        for record in records:
            with self.subTest(record=record["id"]):
                self.assertEqual(validate_record(record, check_json_schema=False), [])

    def test_json_schema_when_optional_dependency_is_available(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("optional jsonschema is not installed")
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for record in example_records():
            with self.subTest(record=record["id"]):
                self.assertEqual(list(validator.iter_errors(record)), [])
        manifest_schema = json.loads(MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(manifest_schema)
        manifest = json.loads((EXAMPLES.parent / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(list(Draft202012Validator(manifest_schema).iter_errors(manifest)), [])

    def test_vqa_annotation_exports_to_ms_swift(self):
        value = record_to_ms_swift(example_records()[0])[0]
        self.assertEqual(value["images"], ["/datasets/coco/train2017/000000000001.jpg"])
        self.assertTrue(value["messages"][0]["content"].startswith("<image>"))
        self.assertEqual(value["messages"][1]["content"], "A red cup.")

    def test_vqa_alias_and_structured_caption_export_without_loss(self):
        vqa = copy.deepcopy(example_records()[0])
        vqa["annotations"][0]["type"] = "vqa"
        self.assertEqual(record_to_ms_swift(vqa)[0]["messages"][1]["content"], "A red cup.")

        caption = copy.deepcopy(example_records()[0])
        target = caption["annotations"][0]["target"]
        caption["task_types"] = ["caption"]
        caption["annotations"] = [
            {
                "id": "caption:0",
                "type": "captioning",
                "target": target,
                "content": [{"type": "text", "text": "A structured caption."}],
                "metadata": {"sft_prompt": "Describe the image."},
            }
        ]
        self.assertEqual(validate_record(caption, check_json_schema=False), [])
        value = record_to_ms_swift(caption)[0]
        self.assertEqual(value["messages"][1]["content"], "A structured caption.")

    def test_vqa_alias_references_and_empty_answers_are_validated(self):
        record = copy.deepcopy(example_records()[0])
        annotation = record["annotations"][0]
        annotation["type"] = "document_qa"
        annotation["question"] = [{"type": "asset", "asset_id": "missing"}]
        annotation["answers"][0] = {"text": ""}
        issues = validate_record(record, check_json_schema=False)
        self.assertIn("missing_ref", {issue.code for issue in issues})
        self.assertIn("annotation_contract", {issue.code for issue in issues})
        with self.assertRaisesRegex(MsSwiftConversionError, "empty answer"):
            record_to_ms_swift(record)

    def test_blank_structured_targets_are_rejected(self):
        conversation_record = copy.deepcopy(example_records()[0])
        conversation_record.pop("annotations")
        conversation_record["task_types"] = ["dialogue"]
        conversation_record["conversations"] = [
            {
                "id": "conversation:blank-target",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Answer the question."}]},
                    {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                ],
            }
        ]
        with self.assertRaisesRegex(MsSwiftConversionError, "renders an empty assistant target"):
            record_to_ms_swift(conversation_record)

        annotation_record = copy.deepcopy(example_records()[0])
        annotation_record["annotations"][0]["answers"] = [
            {"content": [{"type": "text", "text": ""}]}
        ]
        annotation_record["annotations"][0]["canonical_answer_index"] = 0
        issues = validate_record(annotation_record, check_json_schema=False)
        self.assertIn("annotation_contract", {issue.code for issue in issues})
        with self.assertRaisesRegex(MsSwiftConversionError, "selected an empty answer"):
            record_to_ms_swift(annotation_record)

    def test_multiple_choice_vqa_choices_are_rendered_deterministically(self):
        record = copy.deepcopy(example_records()[0])
        annotation = record["annotations"][0]
        annotation["answer_mode"] = "multiple_choice"
        annotation["choices"] = [
            {"id": "choice:plate", "text": "A plate."},
            {
                "id": "choice:cup",
                "content": [{"type": "text", "text": "A red cup."}],
            },
        ]
        annotation["correct_choice_indices"] = [1]
        self.assertEqual(validate_record(record, check_json_schema=False), [])
        value = record_to_ms_swift(record)[0]
        self.assertEqual(
            value["messages"][0]["content"],
            "<image>What is on the table?\nChoices:\n1. A plate.\n2. A red cup.",
        )
        self.assertEqual(value["messages"][1]["content"], "A red cup.")

    def test_grounding_round_trip_preserves_ms_swift_contract(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / "grounding.ppm"
            image_path.write_bytes(b"P6\n800 600\n255\n" + b"\0" * (800 * 600 * 3))
            record = copy.deepcopy(example_records()[1])
            record["assets"][0]["uri"] = str(image_path)
            source = record_to_ms_swift(record)[0]
            self.assertEqual(source["messages"][1]["content"].count("<ref-object>"), 2)
            self.assertEqual(source["messages"][1]["content"].count("<bbox>"), 2)
            self.assertEqual(source["objects"]["bbox_type"], "real")
            imported = ms_swift_to_record(source, source_id="roundtrip")
            self.assertEqual(imported["assets"][0]["media"], {"width": 800, "height": 600})
            self.assertEqual(validate_record(imported, check_json_schema=False), [])
            exported = record_to_ms_swift(imported)[0]
            for key in ("messages", "images", "objects"):
                self.assertEqual(exported[key], source[key])

    def test_non_sft_annotation_fails_instead_of_being_dropped(self):
        panoptic = example_records()[3]
        with self.assertRaisesRegex(MsSwiftConversionError, "no implicit ms-swift"):
            record_to_ms_swift(panoptic)

    def test_multi_annotation_export_has_unique_ids(self):
        record = copy.deepcopy(example_records()[0])
        second = copy.deepcopy(record["annotations"][0])
        second["id"] = "qa:1"
        second["question"] = "What color is the cup?"
        record["annotations"].append(second)
        values = record_to_ms_swift(record)
        self.assertEqual(len(values), 2)
        self.assertEqual(len({value["id"] for value in values}), 2)
        self.assertEqual({value["group_id"] for value in values}, {record["group_id"]})

    def test_multiple_answers_require_canonical_or_explicit_policy(self):
        record = copy.deepcopy(example_records()[0])
        del record["annotations"][0]["canonical_answer_index"]
        with self.assertRaisesRegex(MsSwiftConversionError, "explicit answer policy"):
            record_to_ms_swift(record)
        value = record_to_ms_swift(record, answer_policy="highest-count")[0]
        self.assertEqual(value["messages"][1]["content"], "A red cup.")

    def test_invalid_episode_order_is_reported(self):
        episode = copy.deepcopy(example_records()[4])
        episode["episode"]["steps"][1]["index"] = 0
        issues = validate_record(episode, check_json_schema=False)
        self.assertIn("order", {issue.code for issue in issues})

    def test_split_keeps_media_group_atomic(self):
        first = copy.deepcopy(example_records()[0])
        second = copy.deepcopy(first)
        second["id"] = "vqa:0002"
        second["annotations"][0]["id"] = "qa:1"
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.jsonl"
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            write_jsonl(source, [first, second])
            summary = split_jsonl(source, train, val, val_ratio=0.5, seed=42)
            self.assertEqual(summary.total_records, 2)
            self.assertIn((summary.train_records, summary.val_records), ((2, 0), (0, 2)))
            self.assertEqual(summary.leakage_overlap, 0)

    def test_split_merges_shared_asset_with_conflicting_group_ids(self):
        first = copy.deepcopy(example_records()[0])
        second = copy.deepcopy(first)
        second["id"] = "vqa:0002"
        second["group_id"] = "wrong-group"
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.jsonl"
            write_jsonl(source, [first, second])
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            summary = split_jsonl(source, train, val, val_ratio=0.5)
            output = [record for _, record in iter_jsonl(train)] + [record for _, record in iter_jsonl(val)]
            self.assertEqual(summary.total_groups, 2)
            self.assertEqual(summary.total_components, 1)
            self.assertEqual(summary.merged_groups, 2)
            self.assertEqual(summary.rewritten_records, 2)
            self.assertEqual(len({record["group_id"] for record in output}), 1)
            self.assertEqual(len({record["split"] for record in output}), 1)
            self.assertEqual(
                {record["extensions"]["s1_udf.split"]["source_group_id"] for record in output},
                {first["group_id"], second["group_id"]},
            )

    def test_split_connects_sha_and_uri_aliases_and_rejects_hash_conflicts(self):
        first = copy.deepcopy(example_records()[0])
        second = copy.deepcopy(first)
        first["assets"][0]["sha256"] = "a" * 64
        second["id"] = "vqa:sha-uri-alias"
        second["group_id"] = "other-group"
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.jsonl"
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            write_jsonl(source, [first, second])
            summary = split_jsonl(source, train, val, val_ratio=0.5)
            self.assertEqual(summary.total_components, 1)
            self.assertEqual(summary.unique_assets, 1)

            second["assets"][0]["sha256"] = "b" * 64
            write_jsonl(source, [first, second], overwrite=True)
            with self.assertRaisesRegex(ValueError, "conflicting sha256"):
                split_jsonl(source, train, val, val_ratio=0.5, overwrite=True)

    def test_split_connects_namespaced_source_media_identities(self):
        first = copy.deepcopy(example_records()[0])
        second = copy.deepcopy(first)
        identity = {"namespace": "coco:2017", "value": 42, "verified": True}
        first["assets"][0]["sha256"] = "a" * 64
        first["assets"][0]["source_identities"] = [identity]
        second["id"] = "vqa:source-id-alias"
        second["group_id"] = "other-group"
        second["assets"][0]["uri"] = "/replica/coco-42.jpg"
        second["assets"][0]["source_identities"] = [identity]
        self.assertEqual(validate_record(first), [])
        self.assertEqual(validate_record(second), [])
        invalid = copy.deepcopy(first)
        invalid["assets"][0]["source_identities"].append(
            {"namespace": "coco:2017", "value": 42, "verified": False}
        )
        self.assertIn("duplicate_id", {issue.code for issue in validate_record(invalid)})

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.jsonl"
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            write_jsonl(source, [first, second])
            summary = split_jsonl(source, train, val, val_ratio=0.5)
            self.assertEqual(summary.total_components, 1)
            self.assertEqual(summary.unique_assets, 1)

            second["assets"][0]["sha256"] = "b" * 64
            write_jsonl(source, [first, second], overwrite=True)
            with self.assertRaisesRegex(ValueError, "conflicting sha256"):
                split_jsonl(source, train, val, val_ratio=0.5, overwrite=True)

    def test_split_uses_transitive_media_components(self):
        base = copy.deepcopy(example_records()[0])
        records = []
        for index, (group_id, uris) in enumerate(
            (("g1", ["/media/A.jpg", "/media/B.jpg"]), ("g2", ["/media/B.jpg", "/media/C.jpg"]), ("g3", ["/media/C.jpg"]))
        ):
            record = copy.deepcopy(base)
            record["id"] = "row:{}".format(index)
            record["group_id"] = group_id
            record["assets"] = [
                {"id": "image:{}".format(asset_index), "kind": "image", "uri": uri}
                for asset_index, uri in enumerate(uris)
            ]
            record["annotations"][0]["target"] = {"asset_id": "image:0"}
            records.append(record)
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.jsonl"
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            write_jsonl(source, records)
            summary = split_jsonl(source, train, val, val_ratio=0.5)
            output = [record for _, record in iter_jsonl(train)] + [record for _, record in iter_jsonl(val)]
            self.assertEqual(summary.total_components, 1)
            self.assertEqual(len({record["group_id"] for record in output}), 1)

    def test_split_merges_shared_episode_without_shared_assets(self):
        first = copy.deepcopy(example_records()[4])
        second = copy.deepcopy(first)
        second["id"] = "episode:0002"
        second["group_id"] = "wrong-episode-group"
        for asset in second["assets"]:
            asset["uri"] += ".copy"
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.jsonl"
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            write_jsonl(source, [first, second])
            summary = split_jsonl(source, train, val, val_ratio=0.5, audit_assets=False)
            output = [record for _, record in iter_jsonl(train)] + [
                record for _, record in iter_jsonl(val)
            ]
            self.assertEqual(summary.unique_episodes, 1)
            self.assertEqual(summary.total_components, 1)
            self.assertEqual(len({record["group_id"] for record in output}), 1)
            self.assertEqual(len({record["split"] for record in output}), 1)

    def test_validate_reports_episode_group_and_split_leakage(self):
        first = copy.deepcopy(example_records()[4])
        second = copy.deepcopy(first)
        first["split"] = "train"
        second["id"] = "episode:0002"
        second["group_id"] = "wrong-episode-group"
        second["split"] = "val"
        for asset in second["assets"]:
            asset["uri"] += ".copy"
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.jsonl"
            report_path = root / "validation.json"
            write_jsonl(source, [first, second])
            exit_code = cli_main(
                ["validate", str(source), "--no-json-schema", "--report", str(report_path)]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            codes = {error["code"] for error in report["errors"]}
            self.assertIn("episode_group_conflict", codes)
            self.assertIn("episode_split_leakage", codes)

    def test_url_query_is_part_of_asset_identity(self):
        input_path = Path("dataset.jsonl").resolve()
        first = canonical_asset_identity(
            {"id": "a", "uri": "https://drive.google.com/uc?id=A"}, input_path
        )
        second = canonical_asset_identity(
            {"id": "b", "uri": "https://drive.google.com/uc?id=B"}, input_path
        )
        self.assertNotEqual(first, second)

    def test_split_does_not_overwrite_output_created_during_commit(self):
        record = copy.deepcopy(example_records()[0])
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.jsonl"
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            write_jsonl(source, [record])
            real_link = __import__("os").link

            def racing_link(source_path, output_path):
                output = Path(output_path)
                if output == train and not output.exists():
                    output.write_text("concurrent writer\n", encoding="utf-8")
                return real_link(source_path, output_path)

            with patch("universal_dataset.split.os.link", side_effect=racing_link):
                with self.assertRaises(FileExistsError):
                    split_jsonl(source, train, val, val_ratio=0.5)
            self.assertEqual(train.read_text(encoding="utf-8"), "concurrent writer\n")
            self.assertFalse(val.exists())

    def test_relative_ms_swift_media_is_resolved_against_input_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir).resolve()
            value = {
                "messages": [
                    {"role": "user", "content": "<image>Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": ["images/example.jpg"],
            }
            record = ms_swift_to_record(value, source_id="relative", media_base_dir=base_dir)
            self.assertEqual(record["assets"][0]["uri"], str((base_dir / "images/example.jpg").resolve()))

    def test_normalized_xywh_must_fit_after_conversion(self):
        record = copy.deepcopy(example_records()[1])
        geometry = record["regions"][0]["geometry"]
        geometry["format"] = "xywh"
        geometry["coordinates"] = [0.8, 0.8, 0.5, 0.5]
        geometry["coordinate_space"] = {"type": "normalized"}
        issues = validate_record(record, check_json_schema=False)
        self.assertIn("range", {issue.code for issue in issues})

    def test_manifest_checks_record_count_and_split(self):
        record = copy.deepcopy(example_records()[0])
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            records = root / "train.jsonl"
            manifest_path = root / "manifest.json"
            write_jsonl(records, [record])
            manifest = {
                "format": "s1-udf",
                "schema_version": "1.0.0",
                "dataset": {"name": "test"},
                "record_files": [
                    {
                        "path": "train.jsonl",
                        "format": "jsonl",
                        "split": "train",
                        "count": 1,
                        "sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
                    }
                ],
                "task_types": ["qa"],
                "splits": {"train": 1},
                "grouping": {
                    "field": "group_id",
                    "semantics": "Complete media groups are indivisible across splits.",
                    "identity_fields": [
                        "group_id",
                        "assets[].sha256",
                        "assets[].source_identities[]",
                        "assets[].uri",
                        "assets[].source_ref",
                        "episode.id",
                    ],
                },
                "statistics": {
                    "records": 1,
                    "groups": 1,
                    "unique_assets": 1,
                    "episodes": 0,
                },
            }
            issues = validate_manifest(
                manifest, manifest_path=manifest_path, check_json_schema=False, check_files=True
            )
            self.assertEqual(issues, [])

    def test_manifest_cli_records_an_explicit_empty_split_file(self):
        record = copy.deepcopy(example_records()[0])
        record["split"] = "train"
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            manifest_path = root / "manifest.json"
            write_jsonl(train, [record])
            val.write_text("", encoding="utf-8")
            with patch("builtins.print"):
                exit_code = cli_main(
                    [
                        "build-manifest",
                        str(train),
                        str(val),
                        "--output",
                        str(manifest_path),
                        "--dataset-name",
                        "test",
                        "--record-split",
                        "{}=train".format(train),
                        "--record-split",
                        "{}=val".format(val),
                    ]
                )
            self.assertEqual(exit_code, 0)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {
                    entry["path"]: (entry["split"], entry["count"])
                    for entry in manifest["record_files"]
                },
                {"train.jsonl": ("train", 1), "val.jsonl": ("val", 0)},
            )
            self.assertEqual(manifest["splits"], {"train": 1, "val": 0})
            self.assertEqual(manifest["statistics"]["records"], 1)
            self.assertEqual(
                validate_manifest(
                    manifest,
                    manifest_path=manifest_path,
                    check_json_schema=False,
                    check_files=True,
                ),
                [],
            )

    def test_non_finite_json_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            parse_json_strict('{"value": NaN}')

    def test_profiler_reports_nested_fields(self):
        report = profile_path(EXAMPLES, sample_rows=2, max_depth=4)
        file_report = report["files"][0]
        self.assertEqual(file_report["kind"], "jsonl")
        self.assertEqual(file_report["total_records"], 5)
        fields = file_report["schema"]["fields"]
        self.assertIn("$.assets[].uri", fields)
        self.assertIn("$.group_id", fields)


if __name__ == "__main__":
    unittest.main()
