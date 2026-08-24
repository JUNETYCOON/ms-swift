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

from grouped_jsonl_split import image_stem_group, media_hash_group_resolver, split_jsonl
from split_robo2vlm import episode_group


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class GroupedSplitTest(unittest.TestCase):
    def test_image_bytes_group_different_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_a = root / "a.png"
            image_b = root / "copy.png"
            image_c = root / "c.png"
            image_a.write_bytes(b"same-image")
            image_b.write_bytes(b"same-image")
            image_c.write_bytes(b"other-image")
            source = root / "source.jsonl"
            reserved = root / "reserved.jsonl"
            rows = [
                {"id": "a", "images": [str(image_a)]},
                {"id": "b", "images": [str(image_b)]},
                {"id": "c", "images": [str(image_c)]},
            ]
            write_jsonl(source, rows)
            write_jsonl(reserved, [rows[0]])
            report = split_jsonl(
                input_paths=[source],
                train_output=root / "train.jsonl",
                eval_output=root / "eval.jsonl",
                report_output=root / "report.json",
                group_resolver=media_hash_group_resolver(),
                group_key_description="image sha256",
                reserved_eval_paths=[reserved],
                reserve_only=True,
            )
            self.assertEqual([row["id"] for row in read_jsonl(root / "eval.jsonl")], ["a", "b"])
            self.assertEqual([row["id"] for row in read_jsonl(root / "train.jsonl")], ["c"])
            self.assertEqual(report["groups"]["train_eval_overlap"], 0)

    def test_robo2vlm_questions_share_episode_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_train = root / "source-train.jsonl"
            source_test = root / "source-test.jsonl"
            rows_train = [{"id": "droid_pick_cup_42_q1"}, {"id": "other_7_q1"}]
            rows_test = [{"id": "droid_pick_cup_42_q2"}, {"id": "third_9_q1"}]
            write_jsonl(source_train, rows_train)
            write_jsonl(source_test, rows_test)
            write_jsonl(root / "reserved.jsonl", [rows_test[0]])
            report = split_jsonl(
                input_paths=[source_train, source_test],
                train_output=root / "train.jsonl",
                eval_output=root / "eval.jsonl",
                report_output=root / "report.json",
                group_resolver=episode_group,
                group_key_description="id - _qN",
                reserved_eval_paths=[root / "reserved.jsonl"],
                reserve_only=True,
            )
            self.assertEqual(
                {row["id"] for row in read_jsonl(root / "eval.jsonl")},
                {"droid_pick_cup_42_q1", "droid_pick_cup_42_q2"},
            )
            self.assertEqual(report["groups"]["present_in_multiple_source_files"], 1)
            self.assertEqual(report["groups"]["train_eval_overlap"], 0)

    def test_force_train_keeps_shared_images_out_of_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            rows = [
                {"id": "shared", "image_id": 11, "images": ["gqa/11.jpg"]},
                {"id": "unique-eval", "image_id": 33, "images": ["vg/33.jpg"]},
            ]
            write_jsonl(source, rows)
            report = split_jsonl(
                input_paths=[source],
                train_output=root / "train.jsonl",
                eval_output=root / "eval.jsonl",
                report_output=root / "report.json",
                group_resolver=image_stem_group,
                group_key_description="image stem",
                eval_ratio=0.99,
                seed=42,
                force_train_groups={"11"},
                overwrite=True,
            )
            self.assertEqual([row["id"] for row in read_jsonl(root / "train.jsonl")], ["shared"])
            self.assertEqual([row["id"] for row in read_jsonl(root / "eval.jsonl")], ["unique-eval"])
            self.assertEqual(report["rows"]["forced_train"], 1)
            self.assertEqual(report["groups"]["train_eval_overlap"], 0)

    def test_image_stem_group_normalizes_numeric_ids(self) -> None:
        record = {"image_id": "0007", "images": ["/data/VG_100K/7.jpg"]}
        self.assertEqual(image_stem_group(record, Path("/tmp/demo.jsonl")), "7")
        self.assertEqual(
            image_stem_group({"images": ["gqa/images/train_balanced/7.jpg"]}, Path("/tmp/demo.jsonl")),
            "7",
        )

    def test_existing_curators_can_import_shared_api(self) -> None:
        import curate_llava_sft  # noqa: F401
        import curate_robovqa_sft  # noqa: F401


if __name__ == "__main__":
    unittest.main()
