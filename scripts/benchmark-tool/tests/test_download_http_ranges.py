import tempfile
import unittest
from pathlib import Path

import sys

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import download_http_ranges as downloader


class RangePartitionTest(unittest.TestCase):

    def test_ranges_are_contiguous_and_cover_the_requested_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            segments = downloader.partition_ranges(11, 1010, 7, Path(temp_dir))

        self.assertEqual(segments[0].start, 11)
        self.assertEqual(segments[-1].end, 1010)
        self.assertEqual(sum(segment.size for segment in segments), 1000)
        self.assertTrue(all(left.end + 1 == right.start for left, right in zip(segments, segments[1:])))

    def test_content_range_is_parsed_strictly(self):
        self.assertEqual(downloader.parse_content_range('bytes 10-19/100'), (10, 19, 100))
        with self.assertRaises(ValueError):
            downloader.parse_content_range('10-19/100')


if __name__ == '__main__':
    unittest.main()
