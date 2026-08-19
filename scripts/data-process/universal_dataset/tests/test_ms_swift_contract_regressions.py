"""Regression tests for the S1-UDF to ms-swift SFT contract."""

from __future__ import annotations

import builtins
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from universal_dataset.cli import main as cli_main
from universal_dataset.ms_swift import (
    MsSwiftConversionError,
    ms_swift_to_record,
    record_to_ms_swift,
)
from universal_dataset.validation import validate_record


def _provenance(source_record_id: str) -> dict:
    return {
        "dataset": "contract-regression",
        "source_record_id": source_record_id,
        "conversion": {"tool": "test-fixture", "version": "1.0.0"},
    }


def _write_ppm(path: Path, width: int, height: int) -> None:
    path.write_bytes(
        "P6\n{} {}\n255\n".format(width, height).encode("ascii")
        + b"\0" * (width * height * 3)
    )


def _grounding_source(images: list[str], bbox_type: str = "real") -> dict:
    boxes = (
        [[3, 4, 30, 40], [1, 2, 10, 20]]
        if bbox_type == "real"
        else [[0.1, 0.2, 0.7, 0.8], [0.2, 0.3, 0.6, 0.9]]
    )
    return {
        "messages": [
            {
                "role": "user",
                "content": "{}Locate both regions.".format("<image>" * len(images)),
            },
            {"role": "assistant", "content": "<bbox><bbox>"},
        ],
        "images": images,
        "objects": {
            "ref": [],
            "bbox": boxes,
            "bbox_type": bbox_type,
            "image_id": [1, 0],
        },
    }


def _conversation_record() -> dict:
    return {
        "format": "s1-udf",
        "schema_version": "1.0.0",
        "id": "dialogue:1",
        "group_id": "image:canonical-1",
        "task_types": ["dialogue"],
        "provenance": _provenance("dialogue:1"),
        "conversations": [
            {
                "id": "conversation:0",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Question?"}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Answer."}],
                    },
                ],
            }
        ],
    }


def _vqa_record() -> dict:
    return {
        "format": "s1-udf",
        "schema_version": "1.0.0",
        "id": "vqa:1",
        "group_id": "image:vqa-1",
        "task_types": ["qa"],
        "provenance": _provenance("vqa:1"),
        "assets": [{"id": "image:0", "kind": "image", "uri": "/data/vqa-1.jpg"}],
        "annotations": [
            {
                "id": "qa:0",
                "type": "vqa",
                "target": {"asset_id": "image:0"},
                "question": "What is shown?",
                "answers": [{"text": "A cup."}],
            }
        ],
    }


def _grounding_record() -> dict:
    record = _conversation_record()
    record.update(
        {
            "id": "grounding:1",
            "group_id": "image:grounding-1",
            "task_types": ["dialogue", "grounding"],
            "provenance": _provenance("grounding:1"),
            "assets": [{"id": "image:0", "kind": "image", "uri": "/data/grounding-1.jpg"}],
            "entities": [{"id": "entity:0", "display_name": "cup"}],
            "regions": [
                {
                    "id": "region:0",
                    "asset_id": "image:0",
                    "entity_id": "entity:0",
                    "geometry": {
                        "type": "bbox2d",
                        "format": "xyxy",
                        "coordinates": [1, 2, 10, 20],
                        "coordinate_space": {"type": "pixel"},
                    },
                }
            ],
            "conversations": [
                {
                    "id": "conversation:0",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "asset", "asset_id": "image:0"},
                                {"type": "text", "text": "Locate the cup."},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "entity", "entity_id": "entity:0"},
                                {"type": "region", "region_id": "region:0"},
                            ],
                        },
                    ],
                }
            ],
        }
    )
    return record


class MsSwiftContractRegressionTests(unittest.TestCase):
    def test_include_ids_false_drops_only_record_id(self):
        output = record_to_ms_swift(_conversation_record(), include_ids=False)[0]
        self.assertNotIn("id", output)
        self.assertEqual(output["group_id"], "image:canonical-1")

    def test_cli_drop_ids_retains_canonical_group_id(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.jsonl"
            target = root / "target.jsonl"
            source.write_text(
                json.dumps(_conversation_record(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with patch("builtins.print"):
                exit_code = cli_main(
                    [
                        "export-ms-swift",
                        str(source),
                        "--output",
                        str(target),
                        "--drop-ids",
                    ]
                )
            self.assertEqual(exit_code, 0)
            output = json.loads(target.read_text(encoding="utf-8"))
            self.assertNotIn("id", output)
            self.assertEqual(output["group_id"], "image:canonical-1")

    def test_conversation_without_assistant_target_fails_closed(self):
        record = _conversation_record()
        record["conversations"][0]["messages"] = record["conversations"][0]["messages"][:1]
        with self.assertRaisesRegex(MsSwiftConversionError, "assistant target"):
            record_to_ms_swift(record)

    def test_each_empty_assistant_target_fails_closed(self):
        empty_contents = [
            [],
            [{"type": "text", "text": " \n\t "}],
            [{"type": "tool_result", "result": "  "}],
            [{"type": "tool_result", "result": None}],
            [{"type": "tool_result", "result": []}],
            [{"type": "tool_result", "result": {}}],
        ]
        for content in empty_contents:
            with self.subTest(content=content):
                record = _conversation_record()
                record["conversations"][0]["messages"][1]["content"] = content
                with self.assertRaisesRegex(MsSwiftConversionError, "assistant target"):
                    record_to_ms_swift(record)

    def test_candidate_only_assistant_is_not_silently_selected(self):
        record = _conversation_record()
        assistant = record["conversations"][0]["messages"][1]
        assistant["content"] = []
        assistant["candidates"] = [
            {"content": [{"type": "text", "text": "Candidate answer."}]}
        ]
        with self.assertRaisesRegex(MsSwiftConversionError, "preference candidates"):
            record_to_ms_swift(record)

    def test_external_only_asset_requires_materialization(self):
        record = _conversation_record()
        record["assets"] = [
            {
                "id": "image:0",
                "kind": "image",
                "source_ref": {
                    "uri": "https://example.test/images.h5",
                    "key": "images/0",
                    "offset": 1024,
                    "length": 4096,
                },
            }
        ]
        record["conversations"][0]["messages"][0]["content"].insert(
            0, {"type": "asset", "asset_id": "image:0"}
        )
        with self.assertRaisesRegex(MsSwiftConversionError, "source_ref.*materialize"):
            record_to_ms_swift(record)

    def test_import_preserves_file_authority_in_uri_and_group_identity(self):
        def convert(authority: str) -> dict:
            return ms_swift_to_record(
                {
                    "messages": [
                        {"role": "user", "content": "<image>Question?"},
                        {"role": "assistant", "content": "Answer."},
                    ],
                    "images": ["file://{}/share/image.jpg".format(authority)],
                },
                source_id="same-source-id",
                preserve_raw=False,
            )

        first = convert("SERVER-A")
        second = convert("server-b")
        self.assertEqual(first["assets"][0]["uri"], "file://server-a/share/image.jpg")
        self.assertEqual(second["assets"][0]["uri"], "file://server-b/share/image.jpg")
        self.assertNotEqual(first["assets"][0]["uri"], second["assets"][0]["uri"])
        self.assertNotEqual(first["group_id"], second["group_id"])

        localhost = convert("localhost")
        self.assertEqual(localhost["assets"][0]["uri"], "/share/image.jpg")

    def test_real_grounding_import_probes_dimensions_by_image_id(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first_path = root / "first.ppm"
            second_path = root / "second.ppm"
            _write_ppm(first_path, 20, 30)
            _write_ppm(second_path, 40, 50)
            record = ms_swift_to_record(
                _grounding_source([str(first_path), str(second_path)]),
                source_id="multi-image-real",
                preserve_raw=False,
            )

        self.assertEqual(record["assets"][0]["media"], {"width": 20, "height": 30})
        self.assertEqual(record["assets"][1]["media"], {"width": 40, "height": 50})
        self.assertEqual(
            [region["asset_id"] for region in record["regions"]],
            ["image:1", "image:0"],
        )
        self.assertEqual(validate_record(record), [])

    def test_real_grounding_import_rejects_unmaterialized_or_invalid_images(self):
        with self.assertRaisesRegex(MsSwiftConversionError, "materialize.*locally"):
            ms_swift_to_record(
                _grounding_source(
                    ["https://example.test/first.jpg", "https://example.test/second.jpg"]
                ),
                source_id="remote-real",
            )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            valid_path = root / "valid.ppm"
            missing_path = root / "missing.jpg"
            invalid_path = root / "invalid.jpg"
            _write_ppm(valid_path, 20, 30)
            invalid_path.write_text("not an image", encoding="ascii")
            with self.assertRaisesRegex(MsSwiftConversionError, "not a readable local file"):
                ms_swift_to_record(
                    _grounding_source([str(valid_path), str(missing_path)]),
                    source_id="missing-real",
                )
            with self.assertRaisesRegex(MsSwiftConversionError, "Cannot decode"):
                ms_swift_to_record(
                    _grounding_source([str(valid_path), str(invalid_path)]),
                    source_id="invalid-real",
                )

    def test_real_grounding_import_reports_missing_pillow(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first_path = root / "first.ppm"
            second_path = root / "second.ppm"
            _write_ppm(first_path, 20, 30)
            _write_ppm(second_path, 40, 50)
            real_import = builtins.__import__

            def import_without_pillow(name, *args, **kwargs):
                if name == "PIL" or name.startswith("PIL."):
                    raise ImportError("Pillow intentionally unavailable")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=import_without_pillow):
                with self.assertRaisesRegex(MsSwiftConversionError, "Pillow is required"):
                    ms_swift_to_record(
                        _grounding_source([str(first_path), str(second_path)]),
                        source_id="missing-pillow",
                    )

    def test_non_real_grounding_and_plain_import_do_not_probe_dimensions(self):
        with patch(
            "universal_dataset.ms_swift._probe_local_image_size",
            side_effect=AssertionError("dimension probe must remain lazy"),
        ):
            plain = ms_swift_to_record(
                {
                    "messages": [
                        {"role": "user", "content": "<image>Question?"},
                        {"role": "assistant", "content": "Answer."},
                    ],
                    "images": ["https://example.test/plain.jpg"],
                },
                source_id="plain",
            )
            normalized = ms_swift_to_record(
                _grounding_source(
                    ["https://example.test/first.jpg", "https://example.test/second.jpg"],
                    bbox_type="norm1",
                ),
                source_id="normalized",
            )
        self.assertNotIn("media", plain["assets"][0])
        self.assertTrue(all("media" not in asset for asset in normalized["assets"]))
        self.assertEqual(validate_record(normalized), [])

    def test_multiple_outputs_keep_unique_ids_and_canonical_group(self):
        record = _conversation_record()
        second = copy.deepcopy(record["conversations"][0])
        second["id"] = "conversation:1"
        second["messages"][1]["content"][0]["text"] = "Second answer."
        record["conversations"].append(second)
        outputs = record_to_ms_swift(record)
        self.assertEqual(len({output["id"] for output in outputs}), 2)
        self.assertEqual(
            {output["group_id"] for output in outputs},
            {record["group_id"]},
        )

    def test_vqa_and_grounding_projection_contracts_remain_intact(self):
        vqa = record_to_ms_swift(_vqa_record())[0]
        self.assertEqual(vqa["messages"][0]["content"], "<image>What is shown?")
        self.assertEqual(vqa["messages"][1]["content"], "A cup.")

        grounding = record_to_ms_swift(_grounding_record())[0]
        self.assertEqual(grounding["images"], ["/data/grounding-1.jpg"])
        self.assertEqual(grounding["messages"][1]["content"], "<ref-object><bbox>")
        self.assertEqual(grounding["objects"]["ref"], ["cup"])
        self.assertEqual(grounding["objects"]["bbox"], [[1.0, 2.0, 10.0, 20.0]])


if __name__ == "__main__":
    unittest.main()
