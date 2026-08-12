#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

import converted_dataset_audit_collector as collector


class MediaPlaceholderValidationTest(unittest.TestCase):
    jsonl_path = Path("/tmp/example.jsonl")

    def validate(self, messages: list[dict]) -> list[str]:
        errors, _ = collector.validate_row({"messages": messages}, self.jsonl_path)
        return errors

    def test_assistant_html_media_tags_are_plain_text(self) -> None:
        errors = self.validate(
            [
                {"role": "user", "content": "Show an HTML media example."},
                {
                    "role": "assistant",
                    "content": "Use `<video>` and `<image>` elements in the code sample.",
                },
            ]
        )
        self.assertEqual(errors, [])

    def test_user_media_tag_without_media_path_still_fails(self) -> None:
        errors = self.validate(
            [
                {"role": "user", "content": "<video> Describe this clip."},
                {"role": "assistant", "content": "A person walks."},
            ]
        )
        self.assertIn("videos_placeholder_mismatch", errors)

    def test_structured_assistant_media_block_still_requires_media(self) -> None:
        errors = self.validate(
            [
                {"role": "user", "content": "Return a media block."},
                {
                    "role": "assistant",
                    "content": [{"type": "video"}, {"type": "text", "text": "clip"}],
                },
            ]
        )
        self.assertIn("videos_placeholder_mismatch", errors)

    def test_assistant_grounding_placeholders_are_still_counted(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "Locate the object."},
                {"role": "assistant", "content": "<ref-object> <bbox>"},
            ],
            "objects": {
                "ref": ["object"],
                "bbox": [[0.1, 0.2, 0.3, 0.4]],
                "bbox_type": "norm1",
            },
        }
        errors, _ = collector.validate_row(row, self.jsonl_path)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
