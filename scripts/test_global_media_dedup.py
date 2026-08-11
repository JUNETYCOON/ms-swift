from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.global_media_dedup import parse_args, run
from scripts.validate_sft_entrypoints import validate_global_dedup


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def read_ids(path: Path) -> list[str]:
    return [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]


class GlobalMediaDedupTest(unittest.TestCase):
    def test_video_stem_identity_can_be_disabled_for_generic_segment_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_video = root / "train-video" / "frames-00000000-00000127.mp4"
            eval_video = root / "eval-video" / "frames-00000000-00000127.mp4"
            train_video.parent.mkdir()
            eval_video.parent.mkdir()
            train_video.write_bytes(b"train")
            eval_video.write_bytes(b"eval")
            write_jsonl(root / "source.jsonl", [{"id": "train", "videos": [str(train_video)]}])
            write_jsonl(root / "eval.jsonl", [{"id": "eval", "videos": [str(eval_video)]}])
            manifest = {
                "version": 2,
                "global_dedup": {
                    "required": True,
                    "deduplicate_cross_dataset_train": False,
                    "training_priority": ["video"],
                    "report": str(root / "report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    "video": {
                        "source_train": str(root / "source.jsonl"),
                        "train": str(root / "clean.jsonl"),
                        "eval": str(root / "eval.jsonl"),
                        "use_video_stem_identity": False,
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = run(parse_args(["--manifest", str(manifest_path), "--overwrite"]))

            self.assertEqual(read_ids(root / "clean.jsonl"), ["train"])
            self.assertFalse(report["datasets"]["video"]["use_video_stem_identity"])
            validate_global_dedup(manifest, manifest_path)

    def test_relaxed_policy_retains_cross_dataset_train_but_filters_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared.jpg"
            eval_image = root / "eval.jpg"
            shared.write_bytes(b"shared-train")
            eval_image.write_bytes(b"reserved-eval")
            write_jsonl(root / "a-eval.jsonl", [{"id": "a-eval", "images": [str(eval_image)]}])
            write_jsonl(root / "b-eval.jsonl", [{"id": "b-eval", "images": [str(eval_image)]}])
            write_jsonl(
                root / "a-source.jsonl",
                [
                    {"id": "a-shared", "images": [str(shared)]},
                    {"id": "a-contaminated", "images": [str(eval_image)]},
                ],
            )
            write_jsonl(
                root / "b-source.jsonl",
                [{"id": "b-shared", "images": [str(shared)]}],
            )
            manifest = {
                "version": 2,
                "global_dedup": {
                    "required": True,
                    "deduplicate_cross_dataset_train": False,
                    "training_priority": ["dataset-a", "dataset-b"],
                    "report": str(root / "report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    name: {
                        "enabled": True,
                        "source_train": str(root / f"{suffix}-source.jsonl"),
                        "train": str(root / f"{suffix}-clean.jsonl"),
                        "eval": str(root / f"{suffix}-eval.jsonl"),
                    }
                    for name, suffix in (("dataset-a", "a"), ("dataset-b", "b"))
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = run(parse_args(["--manifest", str(manifest_path), "--overwrite"]))

            self.assertEqual(read_ids(root / "a-clean.jsonl"), ["a-shared"])
            self.assertEqual(read_ids(root / "b-clean.jsonl"), ["b-shared"])
            self.assertEqual(report["verification"]["train_eval_overlap_rows"], 0)
            self.assertEqual(report["verification"]["cross_dataset_train_overlap_rows"], 1)
            self.assertEqual(report["verification"]["status"], "complete")
            self.assertFalse(report["policy"]["deduplicate_cross_dataset_train"])
            self.assertEqual(
                report["datasets"]["dataset-b"]["counts"][
                    "retained_cross_dataset_train_overlap_rows"
                ],
                1,
            )
            validate_global_dedup(manifest, manifest_path)

    def test_eval_reservation_coco_id_hash_and_episode_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_coco = root / "COCO_train2014_000000000123.jpg"
            different_coco_copy = root / "nested" / "COCO_train2014_000000000123.jpg"
            hash_copy = root / "hash-copy.jpg"
            unique = root / "unique.jpg"
            episode_q1 = root / "episode-q1.jpg"
            episode_q2 = root / "episode-q2.jpg"
            different_coco_copy.parent.mkdir()
            eval_coco.write_bytes(b"eval-coco")
            different_coco_copy.write_bytes(b"different-bytes-same-coco-id")
            hash_copy.write_bytes(b"eval-coco")
            unique.write_bytes(b"unique")
            episode_q1.write_bytes(b"episode-one")
            episode_q2.write_bytes(b"episode-two")

            write_jsonl(root / "a-eval.jsonl", [{"id": "eval", "images": [str(eval_coco)]}])
            write_jsonl(
                root / "a-source.jsonl",
                [
                    {"id": "a-coco", "images": [str(different_coco_copy)]},
                    {"id": "a-hash", "images": [str(hash_copy)]},
                    {"id": "a-keep-1", "images": [str(unique)]},
                    {"id": "a-keep-2", "images": [str(unique)]},
                ],
            )
            write_jsonl(
                root / "b-eval.jsonl",
                [{"id": "robot_task_7_q2", "images": [str(episode_q2)]}],
            )
            write_jsonl(
                root / "b-source.jsonl",
                [
                    {"id": "other_task_8_q1", "images": [str(unique)]},
                    {"id": "robot_task_7_q1", "images": [str(episode_q1)]},
                ],
            )
            manifest = {
                "version": 2,
                "global_dedup": {
                    "required": True,
                    "training_priority": ["dataset-a", "robo2vlm"],
                    "report": str(root / "report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    "dataset-a": {
                        "enabled": True,
                        "source_train": str(root / "a-source.jsonl"),
                        "train": str(root / "a-clean.jsonl"),
                        "eval": str(root / "a-eval.jsonl"),
                    },
                    "robo2vlm": {
                        "enabled": True,
                        "source_train": str(root / "b-source.jsonl"),
                        "train": str(root / "b-clean.jsonl"),
                        "eval": str(root / "b-eval.jsonl"),
                    },
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--hash-workers",
                    "2",
                    "--hash-prefetch-rows",
                    "2",
                    "--overwrite",
                ]
            )
            report = run(args)

            self.assertEqual(read_ids(root / "a-clean.jsonl"), ["a-keep-1", "a-keep-2"])
            self.assertEqual(read_ids(root / "b-clean.jsonl"), [])
            self.assertEqual(report["verification"]["status"], "complete")
            self.assertEqual(report["verification"]["train_eval_overlap_rows"], 0)
            self.assertEqual(report["verification"]["cross_dataset_train_overlap_rows"], 0)
            self.assertEqual(report["datasets"]["dataset-a"]["counts"]["excluded_eval_media_overlap_rows"], 2)
            self.assertEqual(report["datasets"]["robo2vlm"]["counts"]["excluded_higher_priority_train_overlap_rows"], 1)
            self.assertEqual(report["datasets"]["robo2vlm"]["counts"]["excluded_eval_media_overlap_rows"], 1)
            validate_global_dedup(manifest, manifest_path)

    def test_rejects_rows_without_a_media_or_lineage_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "eval.jpg"
            image.write_bytes(b"eval")
            write_jsonl(root / "eval.jsonl", [{"id": "eval", "images": [str(image)]}])
            write_jsonl(root / "source.jsonl", [{"id": "text-only"}])
            manifest = {
                "version": 2,
                "global_dedup": {
                    "training_priority": ["demo"],
                    "require_media_identity": True,
                    "report": str(root / "report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    "demo": {
                        "enabled": True,
                        "source_train": str(root / "source.jsonl"),
                        "train": str(root / "clean.jsonl"),
                        "eval": str(root / "eval.jsonl"),
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no image, video, or lineage identity"):
                run(parse_args(["--manifest", str(manifest_path), "--overwrite"]))

            self.assertFalse((root / "clean.jsonl").exists())
            self.assertFalse((root / "report.json").exists())

    def test_opted_in_text_only_rows_are_globally_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared_messages = [
                {"role": "user", "content": "Write a short Python function."},
                {
                    "role": "assistant",
                    "content": "Use the `<video>` element.\ndef answer():\n    return 42",
                },
            ]
            write_jsonl(
                root / "a-source.jsonl",
                [{"id": "a", "messages": shared_messages}],
            )
            write_jsonl(
                root / "b-source.jsonl",
                [
                    {
                        "id": "b",
                        "messages": [
                            {"content": shared_messages[0]["content"], "role": "user"},
                            {
                                "content": shared_messages[1]["content"],
                                "role": "assistant",
                            },
                        ],
                    }
                ],
            )
            write_jsonl(
                root / "a-eval.jsonl",
                [
                    {
                        "id": "a-eval",
                        "messages": [{"role": "user", "content": "Evaluate A."}],
                    }
                ],
            )
            write_jsonl(
                root / "b-eval.jsonl",
                [
                    {
                        "id": "b-eval",
                        "messages": [{"role": "user", "content": "Evaluate B."}],
                    }
                ],
            )
            manifest = {
                "version": 2,
                "global_dedup": {
                    "required": True,
                    "require_media_identity": True,
                    "training_priority": ["dataset-a", "dataset-b"],
                    "report": str(root / "report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    name: {
                        "enabled": True,
                        "allow_text_only": True,
                        "source_train": str(root / f"{suffix}-source.jsonl"),
                        "train": str(root / f"{suffix}-clean.jsonl"),
                        "eval": str(root / f"{suffix}-eval.jsonl"),
                    }
                    for name, suffix in (("dataset-a", "a"), ("dataset-b", "b"))
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = run(parse_args(["--manifest", str(manifest_path), "--overwrite"]))

            self.assertEqual(read_ids(root / "a-clean.jsonl"), ["a"])
            self.assertEqual(read_ids(root / "b-clean.jsonl"), [])
            self.assertEqual(
                report["datasets"]["dataset-b"]["counts"][
                    "excluded_higher_priority_train_overlap_rows"
                ],
                1,
            )
            self.assertEqual(
                report["datasets"]["dataset-a"]["counts"]["train_text_only_rows"],
                1,
            )
            self.assertIn(
                "text:sha256",
                report["datasets"]["dataset-a"]["identity_observations"],
            )
            validate_global_dedup(manifest, manifest_path)

    def test_text_only_opt_in_does_not_mask_a_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_jsonl(
                root / "eval.jsonl",
                [
                    {
                        "id": "eval",
                        "messages": [{"role": "user", "content": "Text-only eval."}],
                    }
                ],
            )
            write_jsonl(
                root / "source.jsonl",
                [
                    {
                        "id": "missing-image",
                        "messages": [
                            {"role": "user", "content": "<image>\nDescribe this image."}
                        ],
                    }
                ],
            )
            manifest = {
                "version": 2,
                "global_dedup": {
                    "training_priority": ["demo"],
                    "require_media_identity": True,
                    "report": str(root / "report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    "demo": {
                        "enabled": True,
                        "allow_text_only": True,
                        "source_train": str(root / "source.jsonl"),
                        "train": str(root / "clean.jsonl"),
                        "eval": str(root / "eval.jsonl"),
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no image, video, or lineage identity"):
                run(parse_args(["--manifest", str(manifest_path), "--overwrite"]))

            self.assertFalse((root / "clean.jsonl").exists())
            self.assertFalse((root / "report.json").exists())

    def test_text_only_opt_in_does_not_mask_a_media_content_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_jsonl(
                root / "eval.jsonl",
                [
                    {
                        "id": "eval",
                        "messages": [{"role": "user", "content": "Text-only eval."}],
                    }
                ],
            )
            write_jsonl(
                root / "source.jsonl",
                [
                    {
                        "id": "missing-image-block",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": ""}},
                                    {"type": "text", "text": "Describe this image."},
                                ],
                            }
                        ],
                    }
                ],
            )
            manifest = {
                "version": 2,
                "global_dedup": {
                    "training_priority": ["demo"],
                    "require_media_identity": True,
                    "report": str(root / "report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    "demo": {
                        "enabled": True,
                        "allow_text_only": True,
                        "source_train": str(root / "source.jsonl"),
                        "train": str(root / "clean.jsonl"),
                        "eval": str(root / "eval.jsonl"),
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no image, video, or lineage identity"):
                run(parse_args(["--manifest", str(manifest_path), "--overwrite"]))

    def test_generic_lineage_ids_require_an_explicit_shared_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for name in ("a", "b", "c", "a-eval", "b-eval", "c-eval"):
                path = root / f"{name}.jpg"
                path.write_bytes(name.encode("ascii"))
                paths[name] = path
            for name in ("a", "b", "c"):
                write_jsonl(
                    root / f"{name}-source.jsonl",
                    [{"id": name, "episode_id": "1", "images": [str(paths[name])]}],
                )
                write_jsonl(
                    root / f"{name}-eval.jsonl",
                    [{"id": f"{name}-eval", "images": [str(paths[f'{name}-eval'])]}],
                )
            manifest = {
                "version": 2,
                "global_dedup": {
                    "training_priority": ["dataset-a", "dataset-b", "dataset-c"],
                    "report": str(root / "report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    "dataset-a": {
                        "enabled": True,
                        "identity_namespace": "shared-robot-source",
                        "source_train": str(root / "a-source.jsonl"),
                        "train": str(root / "a-clean.jsonl"),
                        "eval": str(root / "a-eval.jsonl"),
                    },
                    "dataset-b": {
                        "enabled": True,
                        "source_train": str(root / "b-source.jsonl"),
                        "train": str(root / "b-clean.jsonl"),
                        "eval": str(root / "b-eval.jsonl"),
                    },
                    "dataset-c": {
                        "enabled": True,
                        "identity_namespace": "shared-robot-source",
                        "source_train": str(root / "c-source.jsonl"),
                        "train": str(root / "c-clean.jsonl"),
                        "eval": str(root / "c-eval.jsonl"),
                    },
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = run(parse_args(["--manifest", str(manifest_path), "--overwrite"]))

            self.assertEqual(read_ids(root / "a-clean.jsonl"), ["a"])
            self.assertEqual(read_ids(root / "b-clean.jsonl"), ["b"])
            self.assertEqual(read_ids(root / "c-clean.jsonl"), [])
            self.assertEqual(
                report["datasets"]["dataset-c"]["counts"][
                    "excluded_higher_priority_train_overlap_rows"
                ],
                1,
            )

    def test_remote_media_query_variants_are_one_global_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_jsonl(
                root / "a-source.jsonl",
                [{"id": "a", "images": ["https://EXAMPLE.com/media/image.jpg?sig=one"]}],
            )
            write_jsonl(
                root / "b-source.jsonl",
                [{"id": "b", "images": ["https://example.com/media/image.jpg?sig=two"]}],
            )
            write_jsonl(
                root / "a-eval.jsonl",
                [{"id": "a-eval", "images": ["https://example.com/eval/a.jpg"]}],
            )
            write_jsonl(
                root / "b-eval.jsonl",
                [{"id": "b-eval", "images": ["https://example.com/eval/b.jpg"]}],
            )
            manifest = {
                "version": 2,
                "global_dedup": {
                    "training_priority": ["dataset-a", "dataset-b"],
                    "report": str(root / "report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    name: {
                        "enabled": True,
                        "source_train": str(root / f"{suffix}-source.jsonl"),
                        "train": str(root / f"{suffix}-clean.jsonl"),
                        "eval": str(root / f"{suffix}-eval.jsonl"),
                    }
                    for name, suffix in (("dataset-a", "a"), ("dataset-b", "b"))
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            run(parse_args(["--manifest", str(manifest_path), "--overwrite"]))

            self.assertEqual(read_ids(root / "a-clean.jsonl"), ["a"])
            self.assertEqual(read_ids(root / "b-clean.jsonl"), [])

    def test_rejects_dedup_owner_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            write_jsonl(root / "source.jsonl", [{"id": "source", "images": [str(image)]}])
            write_jsonl(root / "eval.jsonl", [{"id": "eval", "images": [str(image)]}])
            manifest = {
                "version": 2,
                "global_dedup": {
                    "training_priority": ["demo"],
                    "report": str(root / "report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    "demo": {
                        "enabled": True,
                        "dedup_owner": "shared-owner",
                        "source_train": str(root / "source.jsonl"),
                        "train": str(root / "clean.jsonl"),
                        "eval": str(root / "eval.jsonl"),
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dedup_owner is not allowed"):
                run(parse_args(["--manifest", str(manifest_path), "--overwrite"]))

    def test_prerequisite_report_gates_an_enabled_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_image = root / "source.jpg"
            eval_image = root / "eval.jpg"
            source_image.write_bytes(b"source")
            eval_image.write_bytes(b"eval")
            write_jsonl(
                root / "source.jsonl",
                [{"id": "source", "images": [str(source_image)]}],
            )
            write_jsonl(
                root / "eval.jsonl",
                [{"id": "eval", "images": [str(eval_image)]}],
            )
            prerequisite_path = root / "conversion_report.json"
            prerequisite_path.write_text(
                json.dumps({"split_audit": {"test_in_train": 1}}),
                encoding="utf-8",
            )
            manifest = {
                "version": 2,
                "global_dedup": {
                    "required": True,
                    "training_priority": ["spatial"],
                    "report": str(root / "dedup_report.json"),
                    "exclusions": str(root / "exclusions.jsonl"),
                    "cache_db": str(root / "cache.sqlite"),
                },
                "datasets": {
                    "spatial": {
                        "enabled": True,
                        "source_train": str(root / "source.jsonl"),
                        "train": str(root / "clean.jsonl"),
                        "eval": str(root / "eval.jsonl"),
                        "prerequisite_report": str(prerequisite_path),
                        "prerequisite_values": {"split_audit.test_in_train": 0},
                    }
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected 0"):
                run(parse_args(["--manifest", str(manifest_path), "--overwrite"]))

            prerequisite_path.write_text(
                json.dumps({"split_audit": {"test_in_train": 0}}),
                encoding="utf-8",
            )
            report = run(
                parse_args(["--manifest", str(manifest_path), "--overwrite"])
            )

            self.assertEqual(report["status"], "complete")
            self.assertEqual(
                report["datasets"]["spatial"]["prerequisite_report"],
                str(prerequisite_path),
            )
            validate_global_dedup(manifest, manifest_path)


if __name__ == "__main__":
    unittest.main()
