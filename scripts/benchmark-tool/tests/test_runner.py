import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from cli import main
from official_results import extract_official_metrics
from registry import canonical_benchmark_name, resolve_metrics


MOCK_RUNNER = r'''import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)
parser.add_argument('--benchmark', required=True)
parser.add_argument('--output-file', required=True)
args = parser.parse_args()
offset = 5 if args.model_path.endswith('model_b') else 0
if args.benchmark == 'flickr30k':
    metrics = {'Recall@1': 70 + offset, 'Recall@5': 85 + offset, 'Recall@10': 90 + offset}
elif args.benchmark == 'robospatial':
    metrics = {'Accuracy': 60 + offset, 'IoU': 0.5 + offset / 100}
else:
    raise ValueError(args.benchmark)
output = Path(args.output_file)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'metrics': metrics}), encoding='utf-8')
print(f'wrote {output}')
'''


class RegistryTest(unittest.TestCase):

    def test_benchmark_aliases_and_default_metrics(self):
        self.assertEqual(canonical_benchmark_name('EgoPlan-Bench'), 'egoplan_bench')
        self.assertEqual(canonical_benchmark_name('Video-MME'), 'video_mme')
        self.assertEqual(
            resolve_metrics(['flickr30k', 'robospatial']),
            ['recall_at_1', 'recall_at_5', 'recall_at_10', 'accuracy', 'iou'])


class OfficialResultTest(unittest.TestCase):

    def test_json_pointer_and_percentage_normalization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = Path(temp_dir) / 'result.json'
            result_file.write_text(json.dumps({'metrics': {'Accuracy': '75%'}}), encoding='utf-8')
            metrics = extract_official_metrics(
                result_file,
                'realworldqa',
                {'format': 'json', 'metrics': {'accuracy': '/metrics/Accuracy'}})
            self.assertEqual(metrics, {'accuracy': 0.75})

    def test_ambiguous_official_values_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = Path(temp_dir) / 'result.json'
            result_file.write_text(
                json.dumps({'short': {'acc': 70}, 'long': {'acc': 60}}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'multiple values'):
                extract_official_metrics(result_file, 'video_mme')

    def test_csv_column_filter_and_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = Path(temp_dir) / 'result.csv'
            result_file.write_text('split,Accuracy\nval,50\ntest,75\n', encoding='utf-8')
            metrics = extract_official_metrics(
                result_file,
                'realworldqa', {
                    'format': 'csv',
                    'row_filter': {
                        'split': 'test'
                    },
                    'metrics': {
                        'accuracy': {
                            'column': 'Accuracy',
                            'scale': 0.01
                        }
                    }
                })
            self.assertEqual(metrics, {'accuracy': 0.75})

    def test_text_regex_selector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = Path(temp_dir) / 'stdout.log'
            result_file.write_text('Final Accuracy: 64.5\n', encoding='utf-8')
            metrics = extract_official_metrics(
                result_file,
                'egoplan_bench', {
                    'format': 'text',
                    'metrics': {
                        'accuracy': {
                            'regex': r'Accuracy:\s*([0-9.]+)'
                        }
                    }
                })
            self.assertEqual(metrics, {'accuracy': 0.645})


class ExternalRunnerTest(unittest.TestCase):

    def test_external_runner_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner_file = root / 'mock_official.py'
            runner_file.write_text(MOCK_RUNNER, encoding='utf-8')
            config_file = root / 'runners.json'
            config_file.write_text(
                json.dumps({
                    'benchmarks': {
                        benchmark: {
                            'root': str(root),
                            'command': [
                                '{python}',
                                str(runner_file),
                                '--model-path',
                                '{model_path}',
                                '--benchmark',
                                '{benchmark}',
                                '--output-file',
                                '{official_result}',
                            ],
                            'result': {
                                'file': '{official_result}',
                                'format': 'json',
                            },
                        }
                        for benchmark in ('flickr30k', 'robospatial')
                    }
                }),
                encoding='utf-8')
            output_dir = root / 'output'

            exit_code = main([
                'run',
                '--model',
                'model_a=/weights/model_a',
                'model_b=/weights/model_b',
                '--benchmark',
                'flickr30k,robospatial',
                '--runner-config',
                str(config_file),
                '--output-dir',
                str(output_dir),
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / 'summary.csv').is_file())
            self.assertTrue((output_dir / 'model_a' / 'flickr30k' / 'score.csv').is_file())
            self.assertTrue((output_dir / 'model_b' / 'robospatial' / 'stdout.log').is_file())
            self.assertTrue(
                (output_dir / 'comparison.png').is_file() or (output_dir / 'comparison.svg').is_file())
            summary = (output_dir / 'summary.csv').read_text(encoding='utf-8')
            self.assertIn('model_a,flickr30k,0.7,0.85,0.9', summary)
            self.assertIn('model_b,robospatial', summary)


if __name__ == '__main__':
    unittest.main()
