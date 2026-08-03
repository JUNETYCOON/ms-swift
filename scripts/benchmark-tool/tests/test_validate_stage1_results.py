import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from validate_stage1_results import validate_model_result


class Stage1ResultValidationTest(unittest.TestCase):

    def _write_result(self, root: Path, failed_samples: int = 0) -> None:
        output = root / 'baseline' / 'sample_benchmark'
        output.mkdir(parents=True)
        (output / 'predictions.jsonl').write_text('{"id": 1}\n{"id": 2}\n', encoding='utf-8')
        (output / 'scores.json').write_text(json.dumps({
            'status': 'complete',
            'processed_samples': 2,
            'model': 'baseline',
            'model_type': 'qwen3_vl',
            'model_weights': '/weights/baseline',
            'scores': [{
                'task': 'vqa',
                'num_samples': 2,
                'failed_samples': failed_samples,
            }],
        }), encoding='utf-8')
        (output / 'official_result.json').write_text(json.dumps({
            'benchmark': 'sample_benchmark',
            'metrics': {'num_samples': 2, 'metric_scope': 'official_test'},
        }), encoding='utf-8')

    def test_accepts_complete_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_result(root)

            self.assertEqual(
                validate_model_result(root, 'sample_benchmark', 'baseline', 2, 'official_test'), [])

    def test_rejects_failed_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_result(root, failed_samples=1)

            errors = validate_model_result(root, 'sample_benchmark', 'baseline', 2, 'official_test')

        self.assertTrue(any('failed_samples=1' in error for error in errors))

    def test_rejects_wrong_model_weights_and_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_result(root)

            errors = validate_model_result(
                root, 'sample_benchmark', 'baseline', 2, 'official_test',
                expected_weights=Path('/weights/other'), expected_task='description')

        self.assertTrue(any('model_weights=' in error for error in errors))
        self.assertTrue(any("tasks=['vqa']" in error for error in errors))


if __name__ == '__main__':
    unittest.main()
