from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from universal_dataset.io import write_jsonl
from universal_dataset.manifest import build_manifest
from universal_dataset.validation import validate_manifest, validate_record


def provenance(source_id: str = "record:1") -> dict:
    return {
        "dataset": "contract-fixture",
        "source_record_id": source_id,
        "conversion": {"tool": "contract-test-adapter", "version": "1.0.0"},
    }


def asset_record(asset: dict | None = None) -> dict:
    return {
        "format": "s1-udf",
        "schema_version": "1.0.0",
        "id": "record:1",
        "group_id": "group:1",
        "assets": [
            asset
            or {
                "id": "image:0",
                "kind": "image",
                "uri": "https://example.test/image.jpg",
                "media": {"width": 100, "height": 80},
            }
        ],
        "provenance": provenance(),
    }


def issue_codes(record: dict, *, schema: bool = False, **kwargs) -> set[str]:
    return {
        issue.code
        for issue in validate_record(record, check_json_schema=schema, **kwargs)
    }


class RecordContractRegressionTest(unittest.TestCase):
    def test_source_identity_requires_literal_verified_true(self) -> None:
        record = asset_record()
        identity = {"namespace": "coco:2017", "value": 42, "verified": True}
        record["assets"][0]["source_identities"] = [identity]
        self.assertEqual(validate_record(record), [])

        for invalid in (
            {"namespace": "coco:2017", "value": 42},
            {"namespace": "coco:2017", "value": 42, "verified": False},
            {"namespace": "coco:2017", "value": 42, "verified": 1},
        ):
            with self.subTest(identity=invalid):
                candidate = copy.deepcopy(record)
                candidate["assets"][0]["source_identities"] = [invalid]
                self.assertIn("json_schema", issue_codes(candidate, schema=True))
                self.assertIn("asset_identity", issue_codes(candidate))

    def test_message_accepts_content_or_candidates_and_rejects_empty_signal(self) -> None:
        record = asset_record()
        record.pop("assets")
        record["conversations"] = [
            {
                "id": "conversation:0",
                "messages": [
                    {
                        "role": "assistant",
                        "candidates": [
                            {"id": "candidate:0", "content": [{"type": "text", "text": "answer"}]}
                        ],
                    }
                ],
            }
        ]
        self.assertEqual(validate_record(record), [])

        no_payload = copy.deepcopy(record)
        no_payload["conversations"][0]["messages"][0].pop("candidates")
        self.assertIn("json_schema", issue_codes(no_payload, schema=True))
        self.assertIn("empty", issue_codes(no_payload))

        blank_content = copy.deepcopy(record)
        blank_content["conversations"][0]["messages"][0] = {
            "role": "assistant",
            "content": [{"type": "text", "text": " \t\n"}],
        }
        self.assertIn("empty", issue_codes(blank_content))

        blank_candidate = copy.deepcopy(record)
        blank_candidate["conversations"][0]["messages"][0]["candidates"][0]["content"] = [
            {"type": "text", "text": " "}
        ]
        self.assertIn("empty", issue_codes(blank_candidate))

    def test_pixel_geometry_requires_a_reliable_image_size(self) -> None:
        record = asset_record(
            {"id": "image:0", "kind": "image", "uri": "https://example.test/image.jpg"}
        )
        record["regions"] = [
            {
                "id": "region:0",
                "asset_id": "image:0",
                "geometry": {
                    "type": "bbox2d",
                    "format": "xyxy",
                    "coordinates": [1, 2, 10, 20],
                    "coordinate_space": {"type": "pixel"},
                },
            }
        ]
        self.assertIn("missing_image_size", issue_codes(record))

        coordinate_sized = copy.deepcopy(record)
        coordinate_sized["regions"][0]["geometry"]["coordinate_space"]["image_size"] = [100, 80]
        self.assertNotIn("missing_image_size", issue_codes(coordinate_sized))

        asset_sized = copy.deepcopy(record)
        asset_sized["assets"][0]["media"] = {"width": 100, "height": 80}
        self.assertNotIn("missing_image_size", issue_codes(asset_sized))

        boolean_sized = copy.deepcopy(record)
        boolean_sized["regions"][0]["geometry"]["coordinate_space"]["image_size"] = [True, 80]
        self.assertIn("missing_image_size", issue_codes(boolean_sized))

    def test_asset_may_be_addressed_only_by_external_source_ref(self) -> None:
        record = asset_record(
            {
                "id": "frame:0",
                "kind": "image",
                "source_ref": {
                    "uri": "https://example.test/frames.tar",
                    "format": "tar",
                    "key": "images/000001.jpg",
                    "sha256": "a" * 64,
                },
            }
        )
        self.assertEqual(validate_record(record, check_assets=True), [])

        missing_location = asset_record({"id": "frame:0", "kind": "image"})
        self.assertIn("json_schema", issue_codes(missing_location, schema=True))
        self.assertIn("asset_location", issue_codes(missing_location))

        with tempfile.TemporaryDirectory() as temporary:
            local_ref = copy.deepcopy(record)
            local_ref["assets"][0]["source_ref"] = {
                "uri": "missing.tar",
                "key": "images/000001.jpg",
            }
            self.assertIn(
                "missing_data_ref",
                issue_codes(local_ref, check_assets=True, base_dir=Path(temporary)),
            )

    def test_bbox3d_requires_axis_frame_and_rotation_conventions(self) -> None:
        geometry = {
            "type": "bbox3d",
            "center": [1, 2, 3],
            "dimensions": [4, 5, 6],
            "dimension_order": ["length", "width", "height"],
            "rotation_convention": "axis_aligned_right_handed",
            "coordinate_space": {"type": "lidar", "frame_id": "lidar_top"},
        }
        record = asset_record()
        record["regions"] = [
            {"id": "region:0", "asset_id": "image:0", "geometry": geometry}
        ]
        self.assertEqual(validate_record(record), [])

        for field in ("dimension_order", "rotation_convention"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(record)
                invalid["regions"][0]["geometry"].pop(field)
                self.assertIn("json_schema", issue_codes(invalid, schema=True))
                self.assertIn("geometry_convention", issue_codes(invalid))
        no_frame = copy.deepcopy(record)
        no_frame["regions"][0]["geometry"]["coordinate_space"].pop("frame_id")
        self.assertIn("json_schema", issue_codes(no_frame, schema=True))
        self.assertIn("geometry_convention", issue_codes(no_frame))

    def test_episode_steps_and_terminal_flags_are_self_consistent(self) -> None:
        record = asset_record()
        record.pop("assets")
        record["episode"] = {
            "id": "episode:1",
            "steps": [
                {"index": 0, "is_first": True, "is_last": True, "is_terminal": True}
            ],
            "step_count": 1,
        }
        self.assertEqual(validate_record(record), [])

        empty_steps = copy.deepcopy(record)
        empty_steps["episode"]["steps"] = []
        empty_steps["episode"]["step_count"] = 0
        self.assertIn("json_schema", issue_codes(empty_steps, schema=True))
        self.assertIn("episode_steps", issue_codes(empty_steps))

        early_terminal = copy.deepcopy(record)
        early_terminal["episode"] = {
            "id": "episode:1",
            "steps": [
                {"index": 0, "is_first": True, "is_terminal": True},
                {"index": 1, "is_last": True, "is_terminal": False},
            ],
            "step_count": 2,
        }
        self.assertIn("episode_flag", issue_codes(early_terminal))

        terminal_without_last = copy.deepcopy(record)
        terminal_without_last["episode"]["steps"][0].pop("is_last")
        self.assertIn("episode_flag", issue_codes(terminal_without_last))

        nonterminal_final = copy.deepcopy(record)
        nonterminal_final["episode"]["steps"][0]["is_terminal"] = False
        self.assertEqual(validate_record(nonterminal_final), [])

    def test_provenance_is_required_and_has_reliable_minimum_fields(self) -> None:
        record = asset_record()
        self.assertEqual(validate_record(record), [])

        missing = copy.deepcopy(record)
        missing.pop("provenance")
        self.assertIn("json_schema", issue_codes(missing, schema=True))
        self.assertIn("provenance", issue_codes(missing))

        invalid_values = (
            {"dataset": " ", "source_record_id": "x", "conversion": {"tool": "a", "version": "1"}},
            {"dataset": "d", "source_record_id": True, "conversion": {"tool": "a", "version": "1"}},
            {"dataset": "d", "source_record_id": "x", "conversion": {"tool": "a"}},
        )
        for invalid in invalid_values:
            with self.subTest(provenance=invalid):
                candidate = copy.deepcopy(record)
                candidate["provenance"] = invalid
                self.assertIn("provenance", issue_codes(candidate))


class ManifestContractRegressionTest(unittest.TestCase):
    def test_builder_and_validator_reconcile_all_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            manifest_path = root / "manifest.json"
            record = asset_record(
                {
                    "id": "image:0",
                    "kind": "image",
                    "source_ref": {
                        "uri": "https://example.test/archive.tar",
                        "key": "images/1.jpg",
                    },
                }
            )
            record["split"] = "train"
            record["task_types"] = ["dialogue"]
            record["conversations"] = [
                {
                    "id": "conversation:0",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "A valid target."}],
                        }
                    ],
                }
            ]
            write_jsonl(train, [record])
            val.write_text("", encoding="utf-8")

            manifest = build_manifest(
                [train, val],
                manifest_path,
                "contract-fixture",
                record_splits={train: "train", val: "val"},
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                manifest["splits"],
                {"train": 1, "val": 0},
            )
            self.assertEqual(
                manifest["statistics"],
                {"records": 1, "groups": 1, "unique_assets": 1, "episodes": 0},
            )
            self.assertIn(
                "assets[].source_ref",
                manifest["grouping"]["identity_fields"],
            )
            self.assertEqual(
                validate_manifest(
                    manifest,
                    manifest_path=manifest_path,
                    check_json_schema=True,
                    check_files=True,
                ),
                [],
            )

            for field in ("task_types", "grouping", "statistics"):
                with self.subTest(field=field):
                    incomplete = copy.deepcopy(manifest)
                    incomplete.pop(field)
                    codes = {
                        issue.code
                        for issue in validate_manifest(
                            incomplete,
                            manifest_path=manifest_path,
                            check_json_schema=False,
                            check_files=False,
                        )
                    }
                    self.assertIn("manifest_contract", codes)

            wrong_statistics = copy.deepcopy(manifest)
            wrong_statistics["statistics"]["records"] = 2
            self.assertIn(
                "statistics_mismatch",
                {
                    issue.code
                    for issue in validate_manifest(
                        wrong_statistics,
                        manifest_path=manifest_path,
                        check_json_schema=False,
                        check_files=False,
                    )
                },
            )

            missing_splits = copy.deepcopy(manifest)
            missing_splits.pop("splits")
            self.assertIn(
                "missing_split",
                {
                    issue.code
                    for issue in validate_manifest(
                        missing_splits,
                        manifest_path=manifest_path,
                        check_json_schema=False,
                        check_files=False,
                    )
                },
            )

            missing_identity_audit = copy.deepcopy(manifest)
            missing_identity_audit["grouping"]["identity_fields"].remove(
                "assets[].source_ref"
            )
            self.assertIn(
                "manifest_contract",
                {
                    issue.code
                    for issue in validate_manifest(
                        missing_identity_audit,
                        manifest_path=manifest_path,
                        check_json_schema=False,
                        check_files=False,
                    )
                },
            )

    def test_checked_in_example_manifest_is_reconciled(self) -> None:
        manifest_path = Path(__file__).resolve().parents[1] / "examples" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            validate_manifest(
                manifest,
                manifest_path=manifest_path,
                check_json_schema=True,
                check_files=True,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
