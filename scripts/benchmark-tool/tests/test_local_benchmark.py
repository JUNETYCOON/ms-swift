import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from score_local_benchmark import _ocr_bbox, _ocr_references, score


class LocalBenchmarkScoreTest(unittest.TestCase):

    def test_ocr_chart_reference_decodes_json_object(self):
        row = {
            'ocr_type': 'chart parsing en',
            'labels': ['{"title": "Quarterly revenue", "values": {"Q1": "10"}}'],
        }

        self.assertEqual(
            _ocr_references(row),
            [{'title': 'Quarterly revenue', 'values': {'Q1': '10'}}],
        )

    def test_ocr_bbox_falls_back_to_spotting_bbox_list(self):
        row = {'ocr_bbox': None, 'ocr_bbox_list': [[1, 2, 3, 2, 3, 4, 1, 4]]}

        self.assertEqual(_ocr_bbox(row), row['ocr_bbox_list'])

    def test_choice_accuracy_extracts_answer_tags(self):
        rows = [
            {'response': '<answer>C</answer>', 'labels': ['C']},
            {'response': 'The answer is A.', 'labels': ['B']},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            report = score('egoplan', rows, Path(temp_dir) / 'result.json')
        self.assertEqual(report['metrics']['accuracy'], 0.5)
        self.assertEqual(report['metrics']['parse_success_rate'], 1.0)

    def test_openeqa_is_explicitly_diagnostic(self):
        rows = [{'response': 'air conditioning unit', 'labels': ['Air conditioning unit']}]
        with tempfile.TemporaryDirectory() as temp_dir:
            report = score('openeqa', rows, Path(temp_dir) / 'result.json')
        self.assertEqual(report['metrics']['chrf'], 1.0)
        self.assertIn('diagnostic', report['metrics']['metric_scope'])
        self.assertEqual(report['metrics']['official_llm_judge_status'], 'not_scored_no_judge_configuration')

    def test_video_mme_reports_duration_and_domain_breakdowns(self):
        rows = [
            {'response': 'A', 'labels': ['A'], 'duration': 'short', 'domain': 'Knowledge',
             'sub_category': 'History', 'task_type': 'Counting'},
            {'response': 'B', 'labels': ['C'], 'duration': 'short', 'domain': 'Knowledge',
             'sub_category': 'History', 'task_type': 'Counting'},
            {'response': 'D', 'labels': ['D'], 'duration': 'long', 'domain': 'Sports',
             'sub_category': 'Football', 'task_type': 'Recognition'},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            report = score('video_mme', rows, Path(temp_dir) / 'result.json')
        metrics = report['metrics']
        self.assertAlmostEqual(metrics['accuracy'], 2 / 3)
        self.assertEqual(metrics['duration_metrics']['short']['accuracy'], 0.5)
        self.assertEqual(metrics['duration_metrics']['long']['accuracy'], 1.0)
        self.assertEqual(metrics['domain_metrics']['Sports']['num_samples'], 1)


if __name__ == '__main__':
    unittest.main()
