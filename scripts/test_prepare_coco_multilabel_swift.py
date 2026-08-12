from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
from PIL import Image

try:
    from scripts.prepare_coco_multilabel_swift import OUTPUT_NAMES, convert, parse_args, stable_split
except ModuleNotFoundError:
    from prepare_coco_multilabel_swift import OUTPUT_NAMES, convert, parse_args, stable_split


def make_image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new('RGB', (12, 10), color).save(buffer, format='JPEG')
    return buffer.getvalue()


def write_arrow(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(path), 'wb') as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


class CocoMultilabelConverterTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        source = root / 'source'
        source.mkdir()
        (source / 'labels.txt').write_text(
            ''.join(f'label-{index:02d}\n' for index in range(80)), encoding='utf-8')
        image_a = make_image_bytes((255, 0, 0))
        image_b = make_image_bytes((0, 255, 0))
        image_c = make_image_bytes((0, 0, 255))
        image_d = make_image_bytes((255, 255, 0))
        image_test = make_image_bytes((0, 255, 255))
        write_arrow(
            source / 'train' / 'data-00000.arrow',
            [
                {'images': [{'bytes': image_a, 'path': None}], 'labels': [0, 1]},
                {'images': [{'bytes': image_test, 'path': None}], 'labels': [2]},
                {'images': [{'bytes': image_b, 'path': None}], 'labels': [3]},
                {'images': [{'bytes': image_c, 'path': None}], 'labels': [99]},
            ],
        )
        write_arrow(
            source / 'train' / 'data-00001.arrow',
            [
                {'images': [{'bytes': image_a, 'path': None}], 'labels': [1, 0]},
                {'images': [{'bytes': image_b, 'path': None}], 'labels': [4]},
                {'images': [{'bytes': image_d, 'path': None}], 'labels': [5, 5]},
            ],
        )
        write_arrow(
            source / 'test' / 'data-00000.arrow',
            [
                {'images': [{'bytes': image_test, 'path': None}]},
                {'images': [{'bytes': image_test, 'path': None}]},
            ],
        )
        return source

    def run_converter(self, source: Path, output: Path, workers: int) -> dict:
        args = parse_args([
            '--input-dir',
            str(source),
            '--output-dir',
            str(output),
            '--num-workers',
            str(workers),
            '--val-ratio',
            '0.5',
            '--seed',
            '123',
            '--relative-paths',
        ])
        return convert(args)

    def test_stable_split_is_deterministic(self) -> None:
        digest = 'a' * 64
        self.assertEqual(stable_split(digest, 0.2, 42), stable_split(digest, 0.2, 42))

    def test_cleaning_accounting_and_worker_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_fixture(root)
            output_one = root / 'output-one'
            output_two = root / 'output-two'
            report_one = self.run_converter(source, output_one, workers=1)
            report_two = self.run_converter(source, output_two, workers=2)

            self.assertEqual(sum(report_one['output_rows'].values()), 3)
            self.assertEqual(report_one['output_rows']['test'], 1)
            self.assertEqual(report_one['output_rows']['train'] + report_one['output_rows']['val'], 2)
            self.assertEqual(report_one['output_rows'], report_two['output_rows'])
            self.assertEqual(
                report_one['rejected_rows'],
                {
                    'conflicting_duplicate_labels': 2,
                    'duplicate_test_media': 1,
                    'duplicate_train_media': 1,
                    'out_of_range_label_id': 1,
                    'train_test_media_overlap': 1,
                },
            )
            self.assertEqual(report_one['normalized_duplicate_label_rows'], 1)
            self.assertTrue(all(item['balanced'] for item in report_one['accounting'].values()))
            self.assertTrue(report_one['validation']['split_media_sha256_disjoint'])

            for name in [*OUTPUT_NAMES.values(), 'media_groups.tsv', 'rejected.jsonl']:
                self.assertEqual(
                    (output_one / name).read_text(encoding='utf-8'),
                    (output_two / name).read_text(encoding='utf-8'),
                    name,
                )
            with (output_one / 'media_groups.tsv').open(encoding='utf-8', newline='') as handle:
                media_rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(media_rows), 9)
            self.assertEqual(sum(row['status'] == 'retained' for row in media_rows), 3)
            image_files = list((output_one / 'images').rglob('*.*'))
            self.assertEqual(len(image_files), 3)

            records = []
            for split in ('train', 'val'):
                records.extend(
                    json.loads(line)
                    for line in (output_one / OUTPUT_NAMES[split]).read_text(encoding='utf-8').splitlines())
            answers = {record['messages'][1]['content'] for record in records}
            self.assertEqual(answers, {'label-00, label-01', 'label-05'})


if __name__ == '__main__':
    unittest.main()
