from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_dlc_sft import DEFAULT_MANIFEST, audit_dataset


class AuditDlcSftTest(unittest.TestCase):
    def test_default_manifest_is_ready_entrypoint(self) -> None:
        self.assertEqual(
            DEFAULT_MANIFEST.name,
            "dlc_ready_entrypoints.stage1.json",
        )

    def test_valid_point_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.jpg"
            image.write_bytes(b"media")
            data = root / "train.jsonl"
            record = {
                "messages": [
                    {"role": "user", "content": "<image>Point to <ref-object>."},
                    {"role": "assistant", "content": "<bbox>"},
                ],
                "images": [str(image.resolve())],
                "objects": {
                    "ref": ["target"],
                    "bbox": [[0.25, 0.75]],
                    "bbox_type": "norm1",
                },
            }
            data.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = audit_dataset("demo", str(data), 1, 100, 10)

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["counts"]["task::point_grounding"], 1)
            self.assertEqual(report["counts"]["point_coordinates"], 1)
            self.assertEqual(report["schema_errors"], {})

    def test_rejects_duplicate_placeholder_and_missing_media(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "train.jsonl"
            record = {
                "messages": [
                    {"role": "user", "content": "<image><image>Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": [str((root / "missing.jpg").resolve())],
            }
            line = json.dumps(record) + "\n"
            data.write_text(line + line, encoding="utf-8")

            report = audit_dataset("demo", str(data), 1, 100, 10)

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"]["exact_duplicate_rows"], 1)
            self.assertEqual(report["schema_errors"]["media_placeholder_mismatch"], 2)
            self.assertEqual(report["schema_errors"]["missing_media"], 1)

    def test_rejects_real_bbox_outside_decoded_image(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            Image.new("RGB", (20, 10)).save(image)
            data = root / "train.jsonl"
            record = {
                "messages": [
                    {"role": "user", "content": "<image>Locate <ref-object>."},
                    {"role": "assistant", "content": "<bbox>"},
                ],
                "images": [str(image.resolve())],
                "objects": {
                    "ref": ["target"],
                    "bbox": [[0, 0, 21, 10]],
                    "bbox_type": "real",
                    "image_id": [0],
                },
            }
            data.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = audit_dataset("demo", str(data), 1, 100, 10)

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["schema_errors"]["real_bbox_out_of_bounds"], 1)
            self.assertEqual(report["real_grounding_images_decoded"], 1)

    def test_classifies_grounded_description(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            Image.new("RGB", (20, 10)).save(image)
            data = root / "train.jsonl"
            record = {
                "messages": [
                    {"role": "user", "content": "<image>Describe grounded objects."},
                    {
                        "role": "assistant",
                        "content": "The <ref-object><bbox> is visible.",
                    },
                ],
                "images": [str(image.resolve())],
                "objects": {
                    "ref": ["target"],
                    "bbox": [[0, 0, 20, 10]],
                    "bbox_type": "real",
                },
            }
            data.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = audit_dataset("demo", str(data), 1, 100, 10)

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["counts"]["task::grounded_description"], 1)

    def test_assistant_html_video_tag_is_text_without_video_media(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "train.jsonl"
            record = {
                "messages": [
                    {"role": "user", "content": "Show an HTML video element."},
                    {"role": "assistant", "content": "Use `<video controls>` in HTML."},
                ]
            }
            data.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = audit_dataset("demo", str(data), 0, 100, 10)

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["schema_errors"], {})
            self.assertEqual(report["counts"]["task::text_only"], 1)


if __name__ == "__main__":
    unittest.main()
