from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "repair_sample_context.py"
SPEC = importlib.util.spec_from_file_location("repair_sample_context", SCRIPT)
repair = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(repair)


class RepairSampleContextTest(unittest.TestCase):
    def test_grounding_context_replaces_placeholders_and_checks_bbox(self) -> None:
        source = {
            "messages": [
                {"role": "user", "content": "<image>Locate <ref-object>."},
                {"role": "assistant", "content": "<bbox>"},
            ],
            "images": ["/data/image.jpg"],
            "objects": {"ref": ["black luggage"], "bbox": [[1, 2, 3, 4]]},
        }
        sample = {
            "sample_id": "7",
            "media_source_locator": "/data/image.jpg",
            "baseline_result": {"gt_boxes": "[[1,2,3,4]]"},
        }
        repair.enrich_grounding_sample(sample, source, Path("source.jsonl"))
        self.assertEqual(sample["question"], "Locate black luggage.")
        self.assertEqual(sample["ground_truth_boxes"], [[1.0, 2.0, 3.0, 4.0]])
        self.assertIn("black luggage", sample["references"][0])

        bad = dict(sample)
        bad["baseline_result"] = {"gt_boxes": "[[9,9,9,9]]"}
        with self.assertRaisesRegex(ValueError, "GT bbox mismatch"):
            repair.enrich_grounding_sample(bad, source, Path("source.jsonl"))

    def test_robo_choices_are_individual_options(self) -> None:
        source = {
            "id": "task_q1",
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\nQuestion: Done?\nChoices:\nA. No\nB. Yes",
                },
                {"role": "assistant", "content": "B. Yes"},
            ],
            "images": ["/data/robot.jpg"],
        }
        sample = {"sample_id": "task_q1", "media_source_locator": "/data/robot.jpg"}
        repair.enrich_robo_sample(sample, source, Path("source.jsonl"))
        self.assertEqual(sample["question"], "Done?")
        self.assertEqual(sample["options"], {"A": "No", "B": "Yes"})
        self.assertEqual(sample["references"], ["B. Yes"])

    def test_html_update_is_scoped_to_the_matching_article(self) -> None:
        def article(sample_id: str, question: str) -> str:
            return (
                f'<article class="sample" data-search="search-{sample_id}">'
                f'<strong>sample_id: {sample_id}</strong>'
                '<div class="sample-context">'
                f'<div class="question"><p>{question}</p></div>\n'
                '      </div>\n'
                '      <div class="comparison">result</div>'
                '</article>'
            )

        document = article("first", "first question") + article("second", "old second")
        sample = {
            "sample_id": "second",
            "category": "demo",
            "question": "new second",
            "options": {"A": "one", "B": "two"},
            "references": ["B. two"],
        }
        updated = repair.update_article(document, sample)

        first_article, second_article = updated.split("</article>", maxsplit=1)
        self.assertIn('data-search="search-first"', first_article)
        self.assertIn("first question", first_article)
        self.assertNotIn("new second", first_article)
        self.assertIn("new second", second_article)
        self.assertIn("<b>A.</b> one", second_article)

    def test_report_validation_rejects_cross_article_context(self) -> None:
        samples = [
            {
                "sample_id": "first",
                "question": "first question",
                "options": {},
                "references": ["first answer"],
            },
            {
                "sample_id": "second",
                "question": "second question",
                "options": {},
                "references": ["second answer"],
            },
        ]

        def article(sample: dict[str, object]) -> str:
            search = repair.html.escape(repair._search_text(sample), quote=True)
            return (
                f'<article class="sample" data-search="{search}">'
                f'<strong>sample_id: {sample["sample_id"]}</strong>'
                f'{repair._context_html(sample)}'
                '</article>'
            )

        document = "".join(article(sample) for sample in samples)
        repair.validate_report_context(document, samples)

        corrupted = document.replace(
            repair._context_html(samples[1]), repair._context_html(samples[0]), 1
        )
        with self.assertRaisesRegex(ValueError, "HTML context does not match"):
            repair.validate_report_context(corrupted, samples)

    def test_layout_style_is_injected_once(self) -> None:
        document = "<html><head></head><body></body></html>"
        updated = repair.ensure_layout_style(document)
        self.assertIn(f'id="{repair.LAYOUT_STYLE_ID}"', updated)
        self.assertEqual(repair.ensure_layout_style(updated), updated)
        stale = updated.replace(".sample,", ".stale,")
        self.assertEqual(repair.ensure_layout_style(stale), updated)


if __name__ == "__main__":
    unittest.main()
