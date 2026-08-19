import importlib.util
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parent / "materialize_molmo2_videotrack_media.py"
SPEC = importlib.util.spec_from_file_location("materialize_molmo2_videotrack_media", SCRIPT_PATH)
materializer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def jpeg_bytes(width=32, height=24, color=(20, 40, 60)):
    output = io.BytesIO()
    Image.new("RGB", (width, height), color).save(output, format="JPEG")
    return output.getvalue()


def tar_bytes(members, mode="w:gz"):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode=mode) as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def write_parquet(path, dataset, video="video-1", clip="clip-1", start=0, end=2, fps=20.0, width=32, height=24):
    row = {
        "video_dataset": dataset,
        "video": video,
        "clip": clip,
        "start_frame": start,
        "end_frame": end,
        "n_frames": end - start + 1,
        "fps": fps,
        "w": width,
        "h": height,
        "id": f"{clip}-row",
        "exp": "the tracked object",
        "points": [
            {
                "object_id": "0",
                "points": [[float(2 + index), float(3 + index)] for index in range(end - start + 1)],
            }
        ],
    }
    pq.write_table(pa.Table.from_pylist([row, dict(row)]), path)


class FakeMediaTool:

    def __init__(self):
        self.encode_calls = []
        self.probe_calls = []
        self._lock = threading.Lock()

    def encode(self, frame_pattern, start_frame, n_frames, fps, pixel_format, destination):
        sizes = []
        colors = []
        pattern = str(frame_pattern)
        for frame in range(start_frame, start_frame + n_frames):
            path = Path(pattern.replace("%08d", f"{frame:08d}"))
            with Image.open(path) as image:
                image.load()
                sizes.append(image.size)
                colors.append(tuple(image.convert("RGB").getpixel((0, 0))))
        if len(set(sizes)) != 1:
            raise RuntimeError("fake encoder received mixed dimensions")
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "codec_name": "h264",
            "pixel_format": pixel_format,
            "width": sizes[0][0],
            "height": sizes[0][1],
            "fps": fps,
            "n_frames": n_frames,
            "colors": colors,
        }
        destination.write_text(json.dumps(metadata), encoding="utf-8")
        with self._lock:
            self.encode_calls.append(
                {
                    "start": start_frame,
                    "count": n_frames,
                    "fps": fps,
                    "pixel_format": pixel_format,
                    "sizes": sizes,
                    "colors": colors,
                    "destination": str(destination),
                }
            )

    def probe(self, path):
        with self._lock:
            self.probe_calls.append(str(Path(path).resolve()))
        value = json.loads(path.read_text(encoding="utf-8"))
        return materializer.VideoProbe(
            value["codec_name"],
            value["pixel_format"],
            value["width"],
            value["height"],
            value["fps"],
            value["n_frames"],
        )

    def decode_frame(self, path, local_frame, destination):
        value = json.loads(path.read_text(encoding="utf-8"))
        if local_frame < 0 or local_frame >= value["n_frames"]:
            raise RuntimeError("fake decode frame is out of range")
        colors = value.get("colors") or [(30, 40, 50)] * value["n_frames"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (value["width"], value["height"]),
            tuple(colors[local_frame]),
        ).save(destination, format="PNG")

    def write_existing(self, path, width, height, fps, n_frames):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "codec_name": "h264",
                    "pixel_format": materializer.pixel_format_for(width, height),
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "n_frames": n_frames,
                }
            ),
            encoding="utf-8",
        )


class MaterializerTests(unittest.TestCase):

    def test_sequential_copy_supports_short_writes_without_seek_or_tell(self):
        payload = (b"sequential-ossfs-upload" * 19) + b"!"

        class NoSeekShortWriter:

            def __init__(self):
                self.payload = bytearray()

            def write(self, value):
                count = min(5, len(value))
                self.payload.extend(value[:count])
                return count

            def seek(self, *_args):
                raise AssertionError("sequential copy must not seek")

            def tell(self):
                raise AssertionError("sequential copy must not tell")

        writer = NoSeekShortWriter()
        receipt = materializer.copy_stream_sequential(
            io.BytesIO(payload), writer, chunk_size=17
        )
        self.assertEqual(bytes(writer.payload), payload)
        self.assertEqual(receipt.size_bytes, len(payload))
        self.assertEqual(receipt.sha256, hashlib.sha256(payload).hexdigest())

    def test_local_media_upload_is_sequential_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = (b"local-mp4-payload" * 101) + b"tail"
            local_path = root / "scratch" / "window.mp4"
            staged_path = root / "staging" / "window.mp4"
            local_path.parent.mkdir()
            local_path.write_bytes(payload)

            receipt = materializer.upload_local_media_to_staging(local_path, staged_path)
            self.assertEqual(staged_path.read_bytes(), payload)
            self.assertEqual(receipt.size_bytes, len(payload))
            self.assertEqual(receipt.sha256, hashlib.sha256(payload).hexdigest())

            existing = root / "staging" / "existing.mp4"
            existing.write_bytes(b"keep-existing")
            with self.assertRaises(FileExistsError):
                materializer.upload_local_media_to_staging(local_path, existing)
            self.assertEqual(existing.read_bytes(), b"keep-existing")

    def test_failed_sequential_upload_removes_partial_staging_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local_path = root / "scratch" / "window.mp4"
            staged_path = root / "staging" / "window.mp4"
            local_path.parent.mkdir()
            local_path.write_bytes(b"complete-local-media")

            def fail_after_partial_copy(reader, writer, chunk_size=8 * 1024 * 1024):
                del chunk_size
                writer.write(reader.read(7))
                raise OSError("injected sequential upload failure")

            with mock.patch.object(
                materializer,
                "copy_stream_sequential",
                side_effect=fail_after_partial_copy,
            ):
                with self.assertRaisesRegex(OSError, "injected sequential upload failure"):
                    materializer.upload_local_media_to_staging(local_path, staged_path)
            self.assertTrue(local_path.is_file())
            self.assertFalse(staged_path.exists())

    def test_strict_archive_path_mappings(self):
        self.assertEqual(
            materializer.parse_dancetrack_member("train1/dancetrack0001/img1/00000001.jpg", "train1"),
            ("dancetrack0001", 0),
        )
        self.assertEqual(
            materializer.parse_soccernet_member("train/SNMOT-001/img1/000042.jpg"),
            ("SNMOT-001", 41),
        )
        self.assertEqual(
            materializer.parse_mose_member("train/JPEGImages/abc123/00000.jpg"),
            ("abc123", 0),
        )
        self.assertEqual(
            materializer.parse_vipseg_member("VIPSeg/imgs/123_video/00000906.jpg"),
            ("123_video", 906),
        )
        self.assertIsNone(materializer.parse_mose_member("train/videos/abc123/00000.jpg"))
        with self.assertRaisesRegex(materializer.MaterializationError, "root mismatch"):
            materializer.parse_dancetrack_member(
                "train2/dancetrack0001/img1/00000001.jpg", "train1"
            )

    def test_cli_default_output_is_existing_dataset_videos_folder(self):
        self.assertEqual(
            materializer.DEFAULT_OUTPUT_DIR.as_posix(),
            "/mnt/luojunkun/stage1/dataset_ms-swift/Molmo2-VideoTrack/videos",
        )
        with tempfile.TemporaryDirectory() as directory:
            args = materializer.parse_args(["--scratch-dir", directory])
        self.assertEqual(args.output_dir.name, "videos")
        self.assertEqual(args.index_path.name, "video_track_media.json")
        self.assertEqual(args.index_path.parent, args.output_dir.parent)

    def test_closed_window_partition_never_exceeds_128(self):
        windows = materializer.partition_window(0, 300)
        self.assertEqual(
            windows,
            (
                materializer.FrameWindow(0, 127),
                materializer.FrameWindow(128, 255),
                materializer.FrameWindow(256, 300),
            ),
        )
        self.assertEqual(sum(window.n_frames for window in windows), 301)
        self.assertTrue(all(window.n_frames <= 128 for window in windows))

    def test_non_vip_frame_validation_reads_header_without_forcing_pillow_load(self):
        clip = materializer.ClipSpec("mose", "video", "clip", 0, 0, 1, 6.0, 32, 24)
        plan = materializer.build_video_plans([clip], 128)["video"]

        class HeaderOnlyImage:
            format = "JPEG"
            size = (32, 24)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def load(self):
                raise AssertionError("non-VIP validation must not force a Pillow pixel decode")

        class HeaderOnlyPillow:

            @staticmethod
            def open(_stream):
                return HeaderOnlyImage()

        payload = b"jpeg-payload"
        with mock.patch.object(materializer, "_pillow_image", return_value=HeaderOnlyPillow):
            self.assertIs(
                materializer.validate_and_transform_frame(payload, plan, "frame.jpg"),
                payload,
            )

    def _directories(self, root):
        output = root / "output" / "media"
        scratch = root / "scratch-run"
        staging = root / "staging"
        output.mkdir(parents=True)
        scratch.mkdir()
        staging.mkdir()
        return output, scratch, staging

    def _process(self, source, root, tool=None, resume=False, max_frames=128):
        output, scratch, staging = self._directories(root)
        tool = tool or FakeMediaTool()
        budget = materializer.ScratchBudget(64 * 1024 * 1024)
        outcome = materializer.process_dataset(
            source,
            output,
            scratch,
            staging,
            max_frames,
            workers=2,
            max_inflight_videos=2,
            resume=resume,
            budget=budget,
            media_tool=tool,
        )
        self.assertEqual(budget.current, 0)
        return outcome, tool, output, scratch, staging

    def test_dancetrack_zip_materializes_once_and_writes_converter_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "tracks.parquet"
            train1 = root / "train1.zip"
            train2 = root / "train2.zip"
            write_parquet(parquet, "dancetrack", video="dancetrack0001")
            with zipfile.ZipFile(train1, "w") as archive:
                for index in range(3):
                    archive.writestr(
                        f"train1/dancetrack0001/img1/{index + 1:08d}.jpg", jpeg_bytes()
                    )
            with zipfile.ZipFile(train2, "w"):
                pass
            source = materializer.SourceDefinition(
                "dancetrack",
                parquet,
                "dancetrack_zip",
                (train1, train2),
                ("train1", "train2"),
            )
            with mock.patch.object(
                materializer.os,
                "fsync",
                side_effect=AssertionError("scratch frames must not be fsynced"),
            ):
                outcome, tool, _, scratch, staging = self._process(source, root)
            self.assertFalse(outcome.fatal)
            self.assertEqual(outcome.accounting["archive_random_member_reads"], 3)
            self.assertEqual(len(tool.encode_calls), 1)
            encode_destination = Path(tool.encode_calls[0]["destination"])
            self.assertTrue(materializer._is_below(encode_destination, scratch))
            self.assertFalse(materializer._is_below(encode_destination, staging))
            self.assertEqual(list(scratch.rglob("*.mp4")), [])
            self.assertEqual(len(outcome.staged_files), 1)
            staged_path, _ = outcome.staged_files[0]
            self.assertTrue(staged_path.is_file())
            self.assertIn(str(encode_destination.resolve()), tool.probe_calls)
            self.assertIn(str(staged_path.resolve()), tool.probe_calls)
            self.assertEqual(tool.probe(staged_path).codec_name, "h264")
            entry = outcome.entries["dancetrack::clip-1"]
            self.assertEqual(entry["source_video_id"], "dancetrack::dancetrack0001")
            self.assertEqual(entry["lineage_key"], "dancetrack::dancetrack0001")
            self.assertEqual((entry["width"], entry["height"], entry["fps"]), (32, 24, 20.0))
            self.assertEqual(
                entry["windows"][0]["source_end_frame"]
                - entry["windows"][0]["source_start_frame"]
                + 1,
                3,
            )

    def test_missing_frame_is_reported_and_clip_is_not_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "tracks.parquet"
            train1 = root / "train1.zip"
            train2 = root / "train2.zip"
            write_parquet(parquet, "dancetrack", video="dancetrack0001")
            with zipfile.ZipFile(train1, "w") as archive:
                for index in (0, 2):
                    archive.writestr(
                        f"train1/dancetrack0001/img1/{index + 1:08d}.jpg", jpeg_bytes()
                    )
            with zipfile.ZipFile(train2, "w"):
                pass
            source = materializer.SourceDefinition(
                "dancetrack",
                parquet,
                "dancetrack_zip",
                (train1, train2),
                ("train1", "train2"),
            )
            outcome, tool, _, _, _ = self._process(source, root)
            self.assertEqual(outcome.entries, {})
            self.assertEqual(tool.encode_calls, [])
            self.assertIn("missing_frame", {failure["reason"] for failure in outcome.failures})

    def test_upload_failure_cleans_local_and_staged_mp4_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "tracks.parquet"
            train1 = root / "train1.zip"
            train2 = root / "train2.zip"
            write_parquet(parquet, "dancetrack", video="dancetrack0001")
            with zipfile.ZipFile(train1, "w") as archive:
                for index in range(3):
                    archive.writestr(
                        f"train1/dancetrack0001/img1/{index + 1:08d}.jpg", jpeg_bytes()
                    )
            with zipfile.ZipFile(train2, "w"):
                pass
            source = materializer.SourceDefinition(
                "dancetrack",
                parquet,
                "dancetrack_zip",
                (train1, train2),
                ("train1", "train2"),
            )

            def fail_upload(_local_path, staged_path):
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                staged_path.write_bytes(b"partial")
                raise OSError("injected staging failure")

            with mock.patch.object(
                materializer,
                "upload_local_media_to_staging",
                side_effect=fail_upload,
            ):
                outcome, tool, _, scratch, staging = self._process(source, root)
            self.assertFalse(outcome.fatal)
            self.assertEqual(outcome.entries, {})
            self.assertEqual(len(tool.encode_calls), 1)
            self.assertEqual(list(scratch.rglob("*.mp4")), [])
            self.assertEqual(list(staging.rglob("*.mp4")), [])
            self.assertIn(
                "media_encode_or_probe_failed",
                {failure["reason"] for failure in outcome.failures},
            )

    def test_dimension_mismatch_is_reported_without_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "tracks.parquet"
            train = root / "train.zip"
            write_parquet(parquet, "soccernet", video="SNMOT-001")
            with zipfile.ZipFile(train, "w") as archive:
                archive.writestr("train/SNMOT-001/img1/000001.jpg", jpeg_bytes())
                archive.writestr("train/SNMOT-001/img1/000002.jpg", jpeg_bytes(31, 24))
                archive.writestr("train/SNMOT-001/img1/000003.jpg", jpeg_bytes())
            source = materializer.SourceDefinition(
                "soccernet", parquet, "soccernet_zip", (train,)
            )
            outcome, tool, _, _, _ = self._process(source, root)
            self.assertEqual(outcome.entries, {})
            self.assertEqual(tool.encode_calls, [])
            self.assertIn(
                "frame_dimension_mismatch", {failure["reason"] for failure in outcome.failures}
            )

    def test_mose_outer_zip_and_mosev2_parts_are_single_stream_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mose_parquet = root / "mose.parquet"
            mose_zip = root / "MOSE_release.zip"
            write_parquet(mose_parquet, "mose", video="shared", fps=6.0)
            inner = tar_bytes(
                [(f"train/JPEGImages/shared/{index:05d}.jpg", jpeg_bytes()) for index in range(3)]
            )
            with zipfile.ZipFile(mose_zip, "w") as archive:
                archive.writestr(materializer.MOSE_INNER_MEMBER, inner)
            mose_source = materializer.SourceDefinition(
                "mose", mose_parquet, "mose_outer_zip", (mose_zip,)
            )
            mose_outcome, _, _, _, _ = self._process(mose_source, root / "mose-run")
            self.assertEqual(mose_outcome.accounting["archive_passes"], 1)
            self.assertEqual(mose_outcome.entries["mose::clip-1"]["lineage_key"], "mose-family::shared")

            v2_root = root / "v2"
            v2_root.mkdir()
            v2_parquet = v2_root / "mosev2.parquet"
            write_parquet(v2_parquet, "mosev2", video="shared", fps=10.0)
            payload = tar_bytes(
                [(f"train/JPEGImages/shared/{index:05d}.jpg", jpeg_bytes()) for index in range(3)]
            )
            cut1, cut2 = len(payload) // 3, 2 * len(payload) // 3
            parts = []
            checksum_lines = []
            for suffix, data in zip(("aa", "ab", "ac"), (payload[:cut1], payload[cut1:cut2], payload[cut2:])):
                path = v2_root / f"train.tar.gz.{suffix}"
                path.write_bytes(data)
                parts.append(path)
                checksum_lines.append(f"{hashlib.sha256(data).hexdigest()}  {path.name}")
            checksums = v2_root / "SHA256SUMS"
            checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
            v2_source = materializer.SourceDefinition(
                "mosev2",
                v2_parquet,
                "mosev2_parts",
                tuple(parts),
                checksum_path=checksums,
            )
            v2_outcome, _, _, _, _ = self._process(v2_source, v2_root / "run")
            self.assertEqual(v2_outcome.accounting["archive_passes"], 1)
            self.assertTrue(
                all(value["matched"] for value in v2_outcome.accounting["multipart_sha256"].values())
            )
            self.assertEqual(v2_outcome.entries["mosev2::clip-1"]["lineage_key"], "mose-family::shared")
            self.assertEqual(v2_outcome.entries["mosev2::clip-1"]["fps"], 10.0)

    def test_mosev2_checksum_mismatch_fails_the_whole_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "mosev2.parquet"
            write_parquet(parquet, "mosev2", video="video-1")
            payload = tar_bytes(
                [(f"train/JPEGImages/video-1/{index:05d}.jpg", jpeg_bytes()) for index in range(3)]
            )
            cut1, cut2 = len(payload) // 3, 2 * len(payload) // 3
            parts = []
            lines = []
            for suffix, data in zip(
                ("aa", "ab", "ac"),
                (payload[:cut1], payload[cut1:cut2], payload[cut2:]),
            ):
                path = root / f"train.tar.gz.{suffix}"
                path.write_bytes(data)
                parts.append(path)
                digest = "0" * 64 if suffix == "ab" else hashlib.sha256(data).hexdigest()
                lines.append(f"{digest}  {path.name}")
            checksums = root / "SHA256SUMS"
            checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
            source = materializer.SourceDefinition(
                "mosev2",
                parquet,
                "mosev2_parts",
                tuple(parts),
                checksum_path=checksums,
            )
            outcome, _, _, _, staging = self._process(source, root / "run")
            self.assertTrue(outcome.fatal)
            self.assertEqual(outcome.entries, {})
            self.assertIn("SHA256 verification failed", outcome.accounting["fatal_error"])
            self.assertFalse((staging / "mosev2").exists())

    def test_scratch_byte_cap_fails_closed_and_cleans_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "tracks.parquet"
            train = root / "train.zip"
            write_parquet(parquet, "soccernet", video="SNMOT-001")
            with zipfile.ZipFile(train, "w") as archive:
                for index in range(3):
                    archive.writestr(
                        f"train/SNMOT-001/img1/{index + 1:06d}.jpg", jpeg_bytes()
                    )
            output, scratch, staging = self._directories(root)
            budget = materializer.ScratchBudget(10)
            outcome = materializer.process_dataset(
                materializer.SourceDefinition(
                    "soccernet", parquet, "soccernet_zip", (train,)
                ),
                output,
                scratch,
                staging,
                max_window_frames=128,
                workers=1,
                max_inflight_videos=1,
                resume=False,
                budget=budget,
                media_tool=FakeMediaTool(),
            )
            self.assertTrue(outcome.fatal)
            self.assertEqual(outcome.entries, {})
            self.assertEqual(budget.current, 0)
            self.assertEqual(list(scratch.rglob("*.jpg")), [])

    def test_vipseg_numeric_sort_and_official_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "vipseg.parquet"
            archive_path = root / "VIPSeg.tar"
            write_parquet(
                parquet,
                "vipseg",
                video="vip-video",
                start=0,
                end=1,
                fps=6.0,
                width=360,
                height=720,
            )
            members = [
                ("VIPSeg/imgs/vip-video/00000009.jpg", jpeg_bytes(10, 20, (200, 0, 0))),
                ("VIPSeg/imgs/vip-video/00000003.jpg", jpeg_bytes(10, 20, (0, 200, 0))),
            ]
            archive_path.write_bytes(tar_bytes(members, mode="w:"))
            source = materializer.SourceDefinition(
                "vipseg", parquet, "vipseg_tar", (archive_path,)
            )
            outcome, tool, _, _, _ = self._process(source, root)
            self.assertIn("vipseg::clip-1", outcome.entries)
            call = tool.encode_calls[0]
            self.assertEqual(call["sizes"], [(360, 720), (360, 720)])
            self.assertGreater(call["colors"][0][1], call["colors"][0][0])
            self.assertGreater(call["colors"][1][0], call["colors"][1][1])
            self.assertEqual(
                outcome.accounting["transformation"]["name"], "VIPSeg/change2_720p.py"
            )

    def test_odd_width_uses_yuv444p_and_resume_probes_existing_output(self):
        clip = materializer.ClipSpec("vipseg", "v", "c", 0, 0, 1, 6.0, 1281, 720)
        plans = materializer.build_video_plans([clip], 128)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            window = plans["v"].windows[0]
            path = materializer.output_path_for(output, "vipseg", "v", window)
            tool = FakeMediaTool()
            tool.write_existing(path, 1281, 720, 6.0, 1)
            states, failures = materializer.prepare_initial_states(plans, output, True, tool)
            self.assertEqual(failures, [])
            self.assertEqual(states["v"][window].state, "reused")
            self.assertEqual(materializer.pixel_format_for(1281, 720), "yuv444p")

    def test_index_generation_is_deterministic(self):
        clips = [
            materializer.ClipSpec("mose", "v", "second", 2, 3, 2, 6.0, 32, 24),
            materializer.ClipSpec("mose", "v", "first", 0, 1, 2, 6.0, 32, 24),
        ]
        plans = materializer.build_video_plans(clips, 128)
        states = {}
        for window in plans["v"].windows:
            path = Path("/deterministic") / f"{window.start_frame}-{window.end_frame}.mp4"
            states[window] = materializer.WindowState("reused", path)
        results = {"v": materializer.VideoEncodingResult("v", states)}
        first, _, _ = materializer.build_index_entries(plans, results)
        second, _, _ = materializer.build_index_entries(plans, results)
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )
        self.assertEqual(sorted(first), ["mose::first", "mose::second"])

    def test_overlay_decodes_exact_local_frames_and_renders_all_gt_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "tracks.parquet"
            write_parquet(parquet, "mose", video="video-1", clip="clip-1", start=10, end=12)
            source = materializer.SourceDefinition(
                "mose", parquet, "mose_outer_zip", (root / "unused.zip",)
            )
            video_path = root / "media" / "window.mp4"
            tool = FakeMediaTool()
            tool.write_existing(video_path, 32, 24, 20.0, 3)
            index = {
                "mose::clip-1": {
                    "source_video_id": "mose::video-1",
                    "lineage_key": "mose-family::video-1",
                    "width": 32,
                    "height": 24,
                    "fps": 20.0,
                    "windows": [
                        {
                            "mode": "cropped",
                            "path": str(video_path),
                            "source_start_frame": 10,
                            "source_end_frame": 12,
                        }
                    ],
                }
            }
            audit = materializer.generate_overlay_audit(
                [source], index, root / "overlays", 1, tool
            )
            self.assertEqual(audit["status"], "complete")
            self.assertEqual(audit["accounting"]["overlays"], 3)
            self.assertEqual(audit["accounting"]["declared_points"], 3)
            self.assertEqual(audit["accounting"]["rendered_points"], 3)
            overlays = audit["sources"][0]["overlays"]
            self.assertEqual([item["local_frame"] for item in overlays], [0, 1, 2])
            self.assertTrue(all(Path(item["overlay_png"]).is_file() for item in overlays))
            self.assertTrue(Path(audit["html_path"]).is_file())

    def test_full_run_atomically_installs_media_index_report_and_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            source_dir = dataset_root / "data" / "dancetrack"
            downloads = source_dir / "downloads"
            downloads.mkdir(parents=True)
            write_parquet(
                source_dir / "dancetrack_point_tracks.parquet",
                "dancetrack",
                video="dancetrack0001",
            )
            with zipfile.ZipFile(downloads / "train1.zip", "w") as archive:
                for index in range(3):
                    archive.writestr(
                        f"train1/dancetrack0001/img1/{index + 1:08d}.jpg",
                        jpeg_bytes(color=(10 + index, 20, 30)),
                    )
            with zipfile.ZipFile(downloads / "train2.zip", "w"):
                pass
            output_dir = root / "converted" / "videos"
            scratch_dir = root / "scratch"
            args = materializer.parse_args(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--datasets",
                    "dancetrack",
                    "--output-dir",
                    str(output_dir),
                    "--scratch-dir",
                    str(scratch_dir),
                    "--overlay-samples-per-source",
                    "1",
                ]
            )
            tool = FakeMediaTool()
            exit_code, report = materializer.run(args, media_tool=tool)
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["status"], "complete")
            self.assertTrue(args.index_path.is_file())
            self.assertTrue(args.report_path.is_file())
            self.assertTrue((args.overlay_dir / "overlay_manifest.json").is_file())
            index = json.loads(args.index_path.read_text(encoding="utf-8"))
            video_path = Path(index["dancetrack::clip-1"]["windows"][0]["path"])
            self.assertTrue(video_path.is_file())
            self.assertEqual(list(scratch_dir.iterdir()), [])

            args.resume = True
            args.overlay_samples_per_source = 2
            unresolved_code, unresolved_report = materializer.run(args, media_tool=tool)
            self.assertEqual(unresolved_code, 2)
            self.assertEqual(unresolved_report["status"], "failed")
            self.assertEqual(
                unresolved_report["ground_truth_overlay_audit"]["status"],
                "ground_truth_overlay_unresolved",
            )
            self.assertTrue(args.index_path.is_file())

    def test_partial_output_is_kept_but_run_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            source_dir = dataset_root / "data" / "dancetrack"
            downloads = source_dir / "downloads"
            downloads.mkdir(parents=True)

            def row(video):
                return {
                    "video_dataset": "dancetrack",
                    "video": video,
                    "clip": video,
                    "start_frame": 0,
                    "end_frame": 2,
                    "n_frames": 3,
                    "fps": 20.0,
                    "w": 32,
                    "h": 24,
                    "id": f"{video}-row",
                    "exp": "the object",
                    "points": [
                        {
                            "object_id": "0",
                            "points": [[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
                        }
                    ],
                }

            pq.write_table(
                pa.Table.from_pylist([row("dancetrack0001"), row("dancetrack0002")]),
                source_dir / "dancetrack_point_tracks.parquet",
            )
            with zipfile.ZipFile(downloads / "train1.zip", "w") as archive:
                for index in range(3):
                    archive.writestr(
                        f"train1/dancetrack0001/img1/{index + 1:08d}.jpg", jpeg_bytes()
                    )
            with zipfile.ZipFile(downloads / "train2.zip", "w"):
                pass
            args = materializer.parse_args(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--datasets",
                    "dancetrack",
                    "--output-dir",
                    str(root / "converted" / "videos"),
                    "--scratch-dir",
                    str(root / "scratch"),
                    "--overlay-samples-per-source",
                    "1",
                ]
            )
            exit_code, report = materializer.run(args, media_tool=FakeMediaTool())
            self.assertEqual(exit_code, 2)
            self.assertEqual(report["status"], "partial")
            index = json.loads(args.index_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(index), ["dancetrack::dancetrack0001"])
            self.assertGreater(report["failure_accounting"]["total"], 0)

    def test_index_write_failure_rolls_back_new_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "dataset"
            source_dir = dataset_root / "data" / "dancetrack"
            downloads = source_dir / "downloads"
            downloads.mkdir(parents=True)
            write_parquet(
                source_dir / "dancetrack_point_tracks.parquet",
                "dancetrack",
                video="dancetrack0001",
            )
            with zipfile.ZipFile(downloads / "train1.zip", "w") as archive:
                for index in range(3):
                    archive.writestr(
                        f"train1/dancetrack0001/img1/{index + 1:08d}.jpg", jpeg_bytes()
                    )
            with zipfile.ZipFile(downloads / "train2.zip", "w"):
                pass
            output = root / "converted" / "videos"
            scratch = root / "scratch"
            args = materializer.parse_args(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--datasets",
                    "dancetrack",
                    "--output-dir",
                    str(output),
                    "--scratch-dir",
                    str(scratch),
                    "--overlay-samples-per-source",
                    "1",
                ]
            )
            real_atomic_write = materializer.atomic_write_json

            def fail_index(path, payload):
                if path.resolve() == args.index_path:
                    raise OSError("injected index write failure")
                return real_atomic_write(path, payload)

            with mock.patch.object(materializer, "atomic_write_json", side_effect=fail_index):
                with self.assertRaisesRegex(OSError, "injected index write failure"):
                    materializer.run(args, media_tool=FakeMediaTool())
            self.assertEqual(list(output.rglob("*.mp4")) if output.exists() else [], [])
            self.assertFalse(args.index_path.exists())
            self.assertEqual(list(scratch.iterdir()), [])

    def test_write_locations_inside_source_root_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = materializer.parse_args(
                [
                    "--dataset-root",
                    str(root),
                    "--output-dir",
                    str(root / "converted"),
                    "--scratch-dir",
                    str(root / "scratch"),
                ]
            )
            with self.assertRaisesRegex(materializer.MaterializationError, "read-only dataset root"):
                materializer.validate_write_locations(args)


if __name__ == "__main__":
    unittest.main()
