from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.sanitize_sft_jsonl import parse_args, run


class SanitizeSftJsonlTest(unittest.TestCase):
    def test_removes_duplicates_and_invalid_real_bbox_but_keeps_html_tag(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.png"
            Image.new("RGB", (20, 10)).save(image)
            valid = {
                "messages": [
                    {"role": "user", "content": "<image>Locate it."},
                    {"role": "assistant", "content": "<bbox>"},
                ],
                "images": [str(image.resolve())],
                "objects": {
                    "ref": [],
                    "bbox": [[0, 0, 20, 10]],
                    "bbox_type": "real",
                },
            }
            invalid = json.loads(json.dumps(valid))
            invalid["objects"]["bbox"] = [[0, 0, 21, 10]]
            html = {
                "messages": [
                    {"role": "user", "content": "Show HTML."},
                    {"role": "assistant", "content": "Use `<video controls>`."},
                ]
            }
            source = root / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps(record, separators=(",", ":")) + "\n"
                    for record in (valid, valid, invalid, html)
                ),
                encoding="utf-8",
            )
            output = root / "clean.jsonl"

            report = run(
                parse_args(
                    [str(source), str(output), "--progress-every", "0", "--overwrite"]
                )
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [valid, html])
            self.assertEqual(report["counts"]["source_rows"], 4)
            self.assertEqual(report["counts"]["retained_rows"], 2)
            self.assertEqual(report["counts"]["rejected::exact_duplicate_row"], 1)
            self.assertEqual(report["counts"]["rejected::real_bbox_out_of_bounds"], 1)


if __name__ == "__main__":
    unittest.main()
