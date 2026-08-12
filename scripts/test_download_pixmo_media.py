import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parent / "download_pixmo_media.py"
SPEC = importlib.util.spec_from_file_location("download_pixmo_media", SCRIPT_PATH)
downloader = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


class ImageHandler(BaseHTTPRequestHandler):
    payload = png_bytes()

    def do_GET(self):
        if self.path.startswith("/missing"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, format, *args):
        pass


class DownloaderTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ImageHandler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_canonical_url_strips_query_fragment_and_default_port(self):
        self.assertEqual(
            downloader.canonical_url("HTTPS://Example.COM:443/a.png?token=1#part"),
            "https://example.com/a.png",
        )

    def test_download_validates_hash_and_decodes_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = hashlib.sha256(ImageHandler.payload).hexdigest()
            task = downloader.DownloadTask(
                downloader.canonical_url(f"{self.base_url}/image.png"),
                f"{self.base_url}/image.png",
                "a" * 64,
                (expected, ),
            )
            result = downloader.download_one(task, root / "images", root / "parts", 2, 2, 0, 1024 * 1024)
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(result.actual_sha256, expected)
            self.assertEqual((result.width, result.height, result.image_format), (3, 2, "PNG"))
            self.assertTrue(Path(result.local_path).is_file())

    def test_hash_mismatch_and_http_error_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mismatch = downloader.DownloadTask(
                downloader.canonical_url(f"{self.base_url}/image.png"),
                f"{self.base_url}/image.png",
                "b" * 64,
                ("0" * 64, ),
            )
            missing = downloader.DownloadTask(
                downloader.canonical_url(f"{self.base_url}/missing.png"),
                f"{self.base_url}/missing.png",
                "c" * 64,
                (),
            )
            mismatch_result = downloader.download_one(
                mismatch, root / "images", root / "parts", 2, 2, 0, 1024 * 1024)
            missing_result = downloader.download_one(
                missing, root / "images", root / "parts", 2, 2, 0, 1024 * 1024)
            self.assertEqual(mismatch_result.status, "hash_mismatch")
            self.assertEqual(missing_result.status, "http_error")
            self.assertEqual(missing_result.http_status, 404)

    def test_rewrite_keeps_only_verified_local_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "pixmo-cap"
            dataset_dir.mkdir()
            local_image = dataset_dir / "images" / "ok.png"
            local_image.parent.mkdir()
            local_image.write_bytes(ImageHandler.payload)
            good_url = "https://example.com/ok.png?source=one"
            bad_url = "https://example.com/missing.png"
            records = [
                {"messages": [{"role": "user", "content": "<image>"}], "images": [good_url]},
                {"messages": [{"role": "user", "content": "<image>"}], "images": [bad_url]},
            ]
            for split in ("train", "val", "global_train"):
                with (dataset_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
                    for record in records:
                        stream.write(json.dumps(record) + "\n")
            connection = downloader.connect_database(dataset_dir / "media_download.sqlite3")
            now = downloader.utc_now()
            connection.executemany(
                """
                INSERT INTO media(canonical_url, source_url, identity, status, local_path, error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (downloader.canonical_url(good_url), good_url, "d" * 64, "downloaded", str(local_image), None, now),
                    (downloader.canonical_url(bad_url), bad_url, "e" * 64, "http_error", None, "HTTP 404", now),
                ],
            )
            connection.commit()
            report = downloader.rewrite_local_jsonl(
                connection,
                dataset_dir,
                "pixmo-cap",
                True,
                ("train", "val", "global_train"),
            )
            connection.close()
            self.assertEqual(report["splits"]["train"]["written_records"], 1)
            local_record = json.loads((dataset_dir / "local_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(local_record["images"], [str(local_image)])
            rejected_lines = (dataset_dir / "local_train_rejected.jsonl").read_text(encoding="utf-8").splitlines()
            rejected = json.loads(rejected_lines[0])
            self.assertEqual(rejected["status"], "http_error")
            global_record = json.loads(
                (dataset_dir / "local_global_train.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(global_record["images"], [str(local_image)])
            self.assertEqual(report["splits"]["global_train"]["written_records"], 1)

    def test_database_checkpoint_can_restore_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state" / "media.sqlite3"
            checkpoint = root / "remote" / "media.sqlite3.checkpoint"
            checkpoint.parent.mkdir()
            connection = downloader.connect_database(state)
            connection.execute(
                """
                INSERT INTO media(canonical_url, source_url, identity, status, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("https://example.com/a.png", "https://example.com/a.png", "a" * 64, "pending",
                 downloader.utc_now()),
            )
            downloader.checkpoint_database(connection, state, checkpoint)
            connection.close()
            state.unlink()
            downloader.restore_database(state, checkpoint)
            restored = downloader.connect_database(state)
            count = restored.execute("SELECT COUNT(*) FROM media").fetchone()[0]
            restored.close()
            self.assertEqual(count, 1)

    def test_training_manifest_uses_local_deduplicated_entrypoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "pixmo-cap"
            dataset_dir.mkdir()
            (dataset_dir / "local_global_train.jsonl").write_text("{}\n", encoding="utf-8")
            (dataset_dir / "local_val.jsonl").write_text("{}\n", encoding="utf-8")
            (dataset_dir / "localization_report.json").write_text(
                json.dumps(
                    {
                        "media": {"successful_media": 1, "total_media": 2},
                        "splits": {
                            "global_train": {"written_records": 3},
                            "val": {"written_records": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "training.json"
            manifest = downloader.write_training_manifest(root, ("pixmo-cap",), destination)
            self.assertEqual(manifest["status"], "ready_partial_media")
            self.assertEqual(manifest["datasets"]["pixmo-cap"]["train_records"], 3)
            self.assertEqual(
                manifest["datasets"]["pixmo-cap"]["train"],
                str((dataset_dir / "local_global_train.jsonl").resolve()),
            )
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), manifest)


if __name__ == "__main__":
    unittest.main()
