from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
SPLIT = SCRIPTS / "split"
for value in (str(SCRIPTS), str(SPLIT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from resplit_visualgenome_gqa_shared import parse_args, run


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def read_ids(path: Path) -> list[str]:
    return [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]


class ResplitVisualGenomeGqaSharedTest(unittest.TestCase):
    def test_shared_gqa_image_stays_in_each_train_unique_image_can_go_to_val(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_jsonl(
                root / "gqa" / "gqa_train_balanced_sft_msswift.jsonl",
                [{"id": "gqa-train", "images": ["gqa/images/train_balanced/11.jpg"]}],
            )
            write_jsonl(
                root / "gqa" / "gqa_val_balanced_sft_msswift.jsonl",
                [{"id": "gqa-val", "images": ["gqa/images/val_balanced/11.jpg"]}],
            )
            write_jsonl(
                root / "visualgenome" / "visualgenome_qa_train.jsonl",
                [
                    {"id": "vg-shared", "image_id": 11, "images": ["VG_100K/11.jpg"]},
                    {"id": "vg-unique", "image_id": 33, "images": ["VG_100K/33.jpg"]},
                ],
            )
            write_jsonl(
                root / "visualgenome" / "visualgenome_qa_val.jsonl",
                [{"id": "vg-old-val", "image_id": 11, "images": ["VG_100K/11.jpg"]}],
            )
            write_jsonl(
                root / "visualgenome" / "visualgenome_qa_dlc_train.jsonl",
                [{"id": "stale-dlc", "image_id": 99, "images": ["VG_100K/99.jpg"]}],
            )

            summary = run(
                parse_args(
                    [
                        "--data-root",
                        str(root),
                        "--val-ratio",
                        "0.99",
                        "--seed",
                        "42",
                        "--overwrite",
                    ]
                )
            )
            train_ids = set(read_ids(root / "visualgenome" / "visualgenome_qa_train.jsonl"))
            eval_ids = set(read_ids(root / "visualgenome" / "visualgenome_qa_val.jsonl"))
            self.assertEqual(train_ids, {"vg-shared", "vg-old-val"})
            self.assertEqual(eval_ids, {"vg-unique"})
            self.assertEqual(summary["gqa_unique_images"], 1)
            self.assertEqual(
                read_ids(root / "visualgenome" / "visualgenome_qa_dlc_train.jsonl"),
                read_ids(root / "visualgenome" / "visualgenome_qa_train.jsonl"),
            )


if __name__ == "__main__":
    unittest.main()
