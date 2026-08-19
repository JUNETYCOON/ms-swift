import io
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from prepare_local_benchmarks import _extract_tar_archives, _extract_zip_archives


class SelectiveMediaExtractionTest(unittest.TestCase):

    def test_zip_extraction_only_writes_requested_members(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / 'first.zip'
            second = root / 'second.zip'
            with zipfile.ZipFile(first, 'w') as stream:
                stream.writestr('nested/a.mp4', b'a')
            with zipfile.ZipFile(second, 'w') as stream:
                stream.writestr('nested/b.mp4', b'bbb')

            output = root / 'videos'
            _extract_zip_archives([first, second], output, {'b.mp4'})

            self.assertFalse((output / 'a.mp4').exists())
            self.assertEqual((output / 'b.mp4').read_bytes(), b'bbb')

    def test_tar_extraction_preserves_dataset_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / 'videos.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                payload = b'video'
                info = tarfile.TarInfo('hm3d-v0/episode.mp4')
                info.size = len(payload)
                stream.addfile(info, io.BytesIO(payload))

            output = root / 'videos'
            _extract_tar_archives([archive], output, {'hm3d-v0/episode.mp4'})

            self.assertEqual((output / 'hm3d-v0' / 'episode.mp4').read_bytes(), b'video')


if __name__ == '__main__':
    unittest.main()
