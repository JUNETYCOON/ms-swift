from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_stage1_data_remediation import (
    SPATIAL_REQUIRED_VALUES,
    VerificationError,
    _sha256,
    verify_grouped_split,
    verify_manifest,
    verify_robo2vlm_choice_contract,
    verify_spatialvlm,
)


class Stage1RemediationVerificationTest(unittest.TestCase):
    def test_robo2vlm_choice_contract_rejects_serialized_option_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "robo.jsonl"
            record = {
                "id": "episode_1_q2",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "<image>\nQuestion: Was it completed?\nChoices:\n"
                            "A. No\nB. Yes"
                        ),
                    },
                    {"role": "assistant", "content": "B. Yes"},
                ],
                "images": ["image.png"],
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(
                verify_robo2vlm_choice_contract(path), {"rows": 1, "choices": 2}
            )

            record["messages"][0]["content"] = (
                "<image>\nQuestion: Was it completed?\nChoices:\n"
                "A. ['No', 'Yes']"
            )
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "serialized choice collection"):
                verify_robo2vlm_choice_contract(path)

    def test_grouped_split_checks_rows_hashes_and_reserved_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "dataset" / "train.jsonl"
            eval_path = root / "dataset" / "eval.jsonl"
            report_path = root / "dataset" / "report.json"
            train.parent.mkdir()
            train.write_text('{"id":"train"}\n', encoding="utf-8")
            eval_path.write_text('{"id":"eval"}\n', encoding="utf-8")
            report = {
                "train_jsonl": str(train),
                "eval_jsonl": str(eval_path),
                "group_key": "SHA-256 of image bytes",
                "groups": {
                    "train_eval_overlap": 0,
                    "eval": 1,
                    "present_in_multiple_source_files": 3,
                },
                "reserved_eval": {
                    "missing_groups": 0,
                    "reserve_only": True,
                    "groups": 1,
                },
                "dataset_policy": {"hash_algorithm": "sha256"},
                "rows": {"train": 1, "eval": 1},
                "output_sha256": {
                    "train": _sha256(train),
                    "eval": _sha256(eval_path),
                },
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            expectation = {
                "report": "dataset/report.json",
                "train": "dataset/train.jsonl",
                "eval": "dataset/eval.jsonl",
                "reserved_groups": 1,
                "source_overlap_groups": 3,
                "group_key_contains": "SHA-256",
                "policy": {"hash_algorithm": "sha256"},
            }

            result = verify_grouped_split("demo", expectation, root)

            self.assertEqual(result["train"]["rows"], 1)
            self.assertEqual(result["eval"]["rows"], 1)
            self.assertEqual(result["source_overlap_groups"], 3)
            invalid = dict(expectation, reserved_groups=2)
            with self.assertRaisesRegex(VerificationError, "reserved media groups"):
                verify_grouped_split("demo", invalid, root)
            invalid = dict(expectation, source_overlap_groups=4)
            with self.assertRaisesRegex(VerificationError, "source-overlap media groups"):
                verify_grouped_split("demo", invalid, root)
            with self.assertRaisesRegex(VerificationError, "trailing _qN"):
                verify_grouped_split("robo2vlm", expectation, root)

    def test_spatialvlm_requires_official_split_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "spatialvlm"
            root.mkdir(parents=True)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            groups = root / "media_groups.tsv"
            report_path = root / "conversion_report.json"
            train.write_text('{"images":["train.jpg"]}\n', encoding="utf-8")
            val.write_text('{"images":["test.jpg"]}\n', encoding="utf-8")
            groups.write_text("split\tmedia_key\ntrain\tsha256:train\nval\tsha256:test\n", encoding="utf-8")
            report = {
                "dataset": "spatialvlm",
                "input_files": ["/source/train-00000.parquet", "/source/test-00000.parquet"],
                "output_files": {
                    "train": str(train),
                    "val": str(val),
                    "media_groups": str(groups),
                },
                "output_sha256": {
                    "train": _sha256(train),
                    "val": _sha256(val),
                    "media_groups": _sha256(groups),
                },
                "configuration": {
                    "spatialvlm_split_policy": (
                        "preserve_official_train_test_and_reserve_test_image_sha256"
                    )
                },
                "counters": {"written_train": 1, "written_val": 1},
                "unique_media": {
                    "total": 2,
                    "train": 1,
                    "val": 1,
                    "cross_split_leakage": 0,
                },
                "spatialvlm_official_split_audit": {
                    "official_test_source_rows": 1,
                    "test_media_hashes_reserved": 1,
                    "official_test_records_written_to_train": 0,
                    "official_train_records_written_to_val": 0,
                    "post_filter_train_test_hash_overlap": 0,
                },
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")

            result = verify_spatialvlm(root.parent)

            self.assertEqual(result["unique_media"], 2)
            report["output_sha256"]["train"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "output SHA-256"):
                verify_spatialvlm(root.parent)
            report["output_sha256"]["train"] = _sha256(train)
            report["spatialvlm_official_split_audit"][
                "official_test_records_written_to_train"
            ] = 1
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "expected 0"):
                verify_spatialvlm(root.parent)

    def test_manifest_forbids_non_global_train_and_requires_spatial_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            spatial_requirements = dict(SPATIAL_REQUIRED_VALUES)
            manifest = {
                "global_dedup": {
                    "required": True,
                    "require_media_identity": True,
                    "training_priority": ["robo2vlm", "spatialvlm"],
                },
                "datasets": {
                    "robo2vlm": {
                        "enabled": True,
                        "source_train": str(root / "robo-source.jsonl"),
                        "train": str(root / "robo_global_train.jsonl"),
                        "eval": str(root / "robo-eval.jsonl"),
                        "task_type": "multiple-choice VQA / embodied state understanding",
                    },
                    "spatialvlm": {
                        "enabled": True,
                        "source_train": str(root / "spatial-source.jsonl"),
                        "train": str(root / "spatial_global_train.jsonl"),
                        "eval": str(root / "spatial-eval.jsonl"),
                        "prerequisite_values": spatial_requirements,
                    },
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = verify_manifest(manifest_path)

            self.assertEqual(result["enabled_datasets"], 2)
            manifest["datasets"]["spatialvlm"]["train"] = str(root / "plain-train.jsonl")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(VerificationError, "not a global_train"):
                verify_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
