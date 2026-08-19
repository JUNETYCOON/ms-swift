from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

try:
    from scripts.prepare_robo2vlm_swift import make_record, normalize_choices
except ModuleNotFoundError:
    from prepare_robo2vlm_swift import make_record, normalize_choices


class Robo2VLMConverterTest(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            dataset_mode="sft",
            pt_template="<image>\nQuestion: {question}\nChoices:\n{choices}\nAnswer: {answer}",
            user_template="<image>\nQuestion: {question}\nChoices:\n{choices}",
            choice_format="{label}. {choice}",
            answer_format="{label}. {choice}",
            relative_paths=False,
        )

    def test_nested_choices_are_flattened_and_answer_is_canonical(self) -> None:
        self.assertEqual(
            normalize_choices([["No", "Yes", "Cannot be determined"]]),
            ["No", "Yes", "Cannot be determined"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = make_record(
                {
                    "id": "episode_1_q2",
                    "question": "Was the task completed?",
                    "choices": [["No", "Yes", "Cannot be determined"]],
                    "correct_answer": 1,
                },
                root / "image.png",
                root / "robo2vlm_train.jsonl",
                self._args(),
            )

        self.assertEqual(
            record["messages"][0]["content"],
            "<image>\nQuestion: Was the task completed?\nChoices:\n"
            "A. No\nB. Yes\nC. Cannot be determined",
        )
        self.assertEqual(record["messages"][1]["content"], "B. Yes")

    def test_single_serialized_list_choice_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "expected 2..26"):
                make_record(
                    {
                        "id": "episode_1_q2",
                        "question": "Was the task completed?",
                        "choices": "not a parseable list",
                        "correct_answer": 0,
                    },
                    root / "image.png",
                    root / "robo2vlm_train.jsonl",
                    self._args(),
                )

    def test_case_only_duplicate_choice_is_collapsed_and_answer_is_remapped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = make_record(
                {
                    "id": "episode_1_q2",
                    "question": "Which instruction matches the trajectory?",
                    "choices": [
                        "Place the t-shirts on the brown box",
                        "place the t-shirts on the brown box",
                        "Pick up the t-shirts from the brown box",
                    ],
                    "correct_answer": 1,
                },
                root / "image.png",
                root / "robo2vlm_train.jsonl",
                self._args(),
            )

        self.assertEqual(
            record["messages"][0]["content"],
            "<image>\nQuestion: Which instruction matches the trajectory?\nChoices:\n"
            "A. Place the t-shirts on the brown box\n"
            "B. Pick up the t-shirts from the brown box",
        )
        self.assertEqual(record["messages"][1]["content"], "A. Place the t-shirts on the brown box")


if __name__ == "__main__":
    unittest.main()
