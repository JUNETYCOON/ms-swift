from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.validate_sft_entrypoints import (
    canonical,
    file_fingerprint,
    validate_entries,
    validate_global_dedup,
)


class ValidateEntrypointsTest(unittest.TestCase):
    def test_robo2vlm_eval_queue_uses_grouped_eval(self) -> None:
        queue = (
            Path(__file__).resolve().parent
            / "benchmark-tool"
            / "run_custom_eval_queue.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '${DATA_ROOT}/robo2vlm/robo2vlm_sft_eval.jsonl',
            queue,
        )
        self.assertNotIn(
            '${DATA_ROOT}/robo2vlm/robo2vlm_test.jsonl',
            queue,
        )

    def test_stage1_manifest_enables_only_gated_spatialvlm(self) -> None:
        manifest_path = Path(__file__).with_name("curated_dataset_entrypoints.stage1.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        enabled = {
            name
            for name, config in manifest["datasets"].items()
            if config.get("enabled", True)
        }

        self.assertEqual(
            set(manifest["global_dedup"]["training_priority"]), enabled
        )
        spatial = manifest["datasets"]["spatialvlm"]
        self.assertTrue(spatial["enabled"])
        self.assertEqual(
            spatial["prerequisite_values"][
                "spatialvlm_official_split_audit.official_test_records_written_to_train"
            ],
            0,
        )
        self.assertEqual(
            spatial["prerequisite_values"][
                "spatialvlm_official_split_audit.post_filter_train_test_hash_overlap"
            ],
            0,
        )

    def test_requires_verified_global_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            train = root / "clean.jsonl"
            eval_path = root / "eval.jsonl"
            for path in (source, train, eval_path):
                path.write_text("{}\n", encoding="utf-8")
            report_path = root / "dedup.json"
            manifest_path = root / "manifest.json"
            manifest = {
                "datasets": {
                    "demo": {
                        "enabled": True,
                        "source_train": str(source),
                        "train": str(train),
                        "eval": str(eval_path),
                    }
                },
                "global_dedup": {
                    "required": True,
                    "require_media_identity": True,
                    "training_priority": ["demo"],
                    "report": str(report_path),
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "complete",
                        "manifest": str(manifest_path.resolve()),
                        "manifest_fingerprint": file_fingerprint(manifest_path),
                        "training_priority": ["demo"],
                        "policy": {"require_media_identity": True},
                        "verification": {
                            "status": "complete",
                            "train_eval_overlap_rows": 0,
                            "cross_dataset_train_overlap_rows": 0,
                        },
                        "datasets": {
                            "demo": {
                                "source_train": str(source.resolve()),
                                "train": str(train.resolve()),
                                "fingerprints": {
                                    "source_train": file_fingerprint(source),
                                    "train": file_fingerprint(train),
                                    "eval": [file_fingerprint(eval_path)],
                                },
                                "counts": {},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            validate_global_dedup(manifest, manifest_path)
            validate_entries(manifest, [canonical(train)], [canonical(eval_path)])
            with self.assertRaisesRegex(ValueError, "globally deduplicated"):
                validate_entries(manifest, [canonical(source)], [canonical(eval_path)])

            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["datasets"]["demo"]["allow_text_only"] = True
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "allow_text_only does not match"):
                validate_global_dedup(manifest, manifest_path)
            report["datasets"]["demo"]["allow_text_only"] = False
            report_path.write_text(json.dumps(report), encoding="utf-8")

            source_stat = source.stat()
            source.write_text("[]\n", encoding="utf-8")
            os.utime(
                source,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            )
            with self.assertRaisesRegex(ValueError, "fingerprint sha256 changed"):
                validate_global_dedup(manifest, manifest_path)
            source.write_text("{}\n", encoding="utf-8")
            os.utime(
                source,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["datasets"]["demo"]["counts"]["train_rows_without_identity"] = 1
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train_rows_without_identity is not zero"):
                validate_global_dedup(manifest, manifest_path)

    def test_relaxed_policy_allows_reported_cross_dataset_train_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            for path in (source, train, eval_path):
                path.write_text("{}\n", encoding="utf-8")
            report_path = root / "report.json"
            manifest_path = root / "manifest.json"
            manifest = {
                "datasets": {
                    "demo": {
                        "enabled": True,
                        "source_train": str(source),
                        "train": str(train),
                        "eval": str(eval_path),
                    }
                },
                "global_dedup": {
                    "required": True,
                    "require_media_identity": True,
                    "deduplicate_cross_dataset_train": False,
                    "training_priority": ["demo"],
                    "report": str(report_path),
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "complete",
                        "manifest": str(manifest_path.resolve()),
                        "manifest_fingerprint": file_fingerprint(manifest_path),
                        "training_priority": ["demo"],
                        "policy": {
                            "require_media_identity": True,
                            "deduplicate_cross_dataset_train": False,
                        },
                        "verification": {
                            "status": "complete",
                            "train_eval_overlap_rows": 0,
                            "cross_dataset_train_overlap_rows": 3,
                        },
                        "datasets": {
                            "demo": {
                                "source_train": str(source.resolve()),
                                "train": str(train.resolve()),
                                "allow_text_only": False,
                                "prerequisite_report": None,
                                "fingerprints": {
                                    "source_train": file_fingerprint(source),
                                    "train": file_fingerprint(train),
                                    "eval": [file_fingerprint(eval_path)],
                                    "prerequisite_report": None,
                                },
                                "counts": {},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            validate_global_dedup(manifest, manifest_path)


if __name__ == "__main__":
    unittest.main()
