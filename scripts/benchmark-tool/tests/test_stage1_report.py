import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from generate_stage1_report import generate


class Stage1ReportTest(unittest.TestCase):

    def test_writes_markdown_and_per_benchmark_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_root = Path(temp_dir)
            official_root = stage_root / 'benchmark-eval-result' / 'stage1-autoeval' / 'official-local'
            for model, accuracy in (('baseline', 0.5), ('ours', 0.75)):
                run_dir = official_root / model / 'video_mme'
                run_dir.mkdir(parents=True)
                report = {
                    'benchmark': 'video_mme',
                    'metrics': {
                        'metric_scope': 'official_exact_choice_without_subtitles',
                        'num_samples': 4,
                        'accuracy': accuracy,
                        'duration_metrics': {
                            'short': {'num_samples': 2, 'accuracy': accuracy - 0.1},
                        },
                    },
                }
                (run_dir / 'official_result.json').write_text(json.dumps(report), encoding='utf-8')

            archive_dir = official_root / 'baseline' / 'video_mme.pre-local-media'
            archive_dir.mkdir(parents=True)
            (archive_dir / 'official_result.json').write_text(json.dumps({
                'benchmark': 'video_mme',
                'metrics': {
                    'metric_scope': 'stale_archive',
                    'num_samples': 4,
                    'accuracy': 1.0,
                },
            }), encoding='utf-8')

            _, _, markdown_path, rows = generate(stage_root, stage_root / 'report')

            self.assertEqual(len(rows), 2)
            self.assertTrue(all('pre-local-media' not in row['result_file'] for row in rows))
            self.assertTrue(markdown_path.is_file())
            markdown = markdown_path.read_text(encoding='utf-8')
            self.assertIn('| Video-MME | accuracy |', markdown)
            self.assertIn('| Video-MME | duration_metrics.short.accuracy |', markdown)
            self.assertTrue((stage_root / 'benchmark-eval-result' / 'benchmark-summary.csv').is_file())
            self.assertTrue((stage_root / 'benchmark-eval-result' / 'benchmark-csv' / 'video_mme.csv').is_file())


if __name__ == '__main__':
    unittest.main()
