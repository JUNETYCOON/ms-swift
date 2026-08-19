import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve().parent / "audit_molmo2_videotrack_swift.py"
SPEC = importlib.util.spec_from_file_location("audit_molmo2_videotrack_swift", SCRIPT_PATH)
auditor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = auditor
SPEC.loader.exec_module(auditor)


def source_row(
    *,
    dataset="demo",
    video="video-1",
    clip="clip-1",
    start=0,
    end=1,
    width=32,
    height=24,
    fps=10.0,
    points=None,
    segments=None,
    expression="the target",
):
    n_frames = end - start + 1
    if points is None:
        points = [{"object_id": "0", "points": [[width / 2, height / 2]] * n_frames}]
    if segments is None:
        segments = [{"object_id": "0", "segments": [[0, n_frames - 1]]}]
    return {
        "id": f"{clip}-annotation",
        "video_dataset": dataset,
        "video": video,
        "clip": clip,
        "exp": expression,
        "points": points,
        "segments": segments,
        "start_frame": start,
        "end_frame": end,
        "n_frames": n_frames,
        "w": width,
        "h": height,
        "fps": fps,
    }


def output_record(path, token="<video>"):
    return {
        "messages": [
            {"role": "user", "content": f"{token}\nTrack the target."},
            {"role": "assistant", "content": "Object 0, frame 0: [500, 500]"},
        ],
        "videos": [str(path)],
    }


def passing_media(expectation, _ffprobe, _ffmpeg):
    return {
        "path": expectation.path,
        "status": "passed",
        "errors": [],
        "checks": {
            "frame_count": {"status": "matched"},
            "width": {
                "status": "matched" if expectation.width is not None else "unavailable"
            },
            "height": {
                "status": "matched" if expectation.height is not None else "unavailable"
            },
            "fps": {
                "status": "matched" if expectation.fps is not None else "unavailable"
            },
            "decode_first": {"status": "passed"},
            "decode_last": {"status": "passed"},
        },
    }


class Fixture:
    def __init__(self, root, rows, entries, train_records, val_records, groups):
        self.root = root
        self.source_root = root / "source"
        self.output_dir = root / "output"
        self.source_root.mkdir()
        self.output_dir.mkdir()
        data = self.source_root / "data" / "annotations"
        data.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), data / "tracks.parquet")
        self.index_path = self.output_dir / "video_track_media.json"
        self.index_path.write_text(json.dumps(entries), encoding="utf-8")
        for split, records in (("train", train_records), ("val", val_records)):
            with (self.output_dir / f"{split}.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
        lines = ["split\tmedia_key", *[f"{split}\t{key}" for split, key in groups]]
        (self.output_dir / "media_groups.tsv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def audit(self, **kwargs):
        return auditor.run_audit(
            self.source_root,
            self.output_dir,
            self.index_path,
            kwargs.pop("media_workers", 2),
            audit_one=kwargs.pop("audit_one", passing_media),
            **kwargs,
        )


def make_entry(path, *, dataset="demo", video="video-1", start=0, end=1, metadata=True):
    entry = {
        "source_video_id": f"{dataset}::{video}",
        "lineage_key": f"{dataset}::{video}",
        "windows": [
            {
                "mode": "cropped",
                "path": str(path.resolve()),
                "source_start_frame": start,
                "source_end_frame": end,
            }
        ],
    }
    if metadata:
        entry.update({"width": 32, "height": 24, "fps": 10.0})
    return entry


class AuditTests(unittest.TestCase):
    def test_source_filter_only_reads_selected_data_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            for group in ("Keep", "skip"):
                data = source_root / "data" / group
                data.mkdir(parents=True)
                pq.write_table(
                    pa.Table.from_pylist(
                        [source_row(dataset=group.casefold(), video=f"{group}-video")]
                    ),
                    data / "tracks.parquet",
                )

            failures = auditor.Failures()
            annotations, rejections, reports = auditor.load_source_annotations(
                source_root, failures, ["keep"]
            )

            self.assertEqual([item.dataset for item in annotations], ["keep"])
            self.assertEqual(rejections, [])
            self.assertEqual(len(reports), 1)
            self.assertIn("Keep", reports[0]["path"])
            self.assertEqual(failures.sorted_items(), [])

    def test_unknown_source_filter_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            data = source_root / "data" / "available"
            data.mkdir(parents=True)
            pq.write_table(
                pa.Table.from_pylist([source_row(dataset="available")]),
                data / "tracks.parquet",
            )

            failures = auditor.Failures()
            annotations, _, _ = auditor.load_source_annotations(
                source_root, failures, ["missing"]
            )

            self.assertEqual(annotations, [])
            self.assertIn(
                "unknown_source_selection",
                {item["code"] for item in failures.sorted_items()},
            )

    def test_happy_path_balances_sources_and_reports_unavailable_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_video = root / "train.mp4"
            val_video = root / "val.mp4"
            train_video.write_bytes(b"mp4")
            val_video.write_bytes(b"mp4")
            rows = [
                source_row(video="train-video", clip="train-clip"),
                source_row(video="val-video", clip="val-clip", start=5, end=5),
            ]
            entries = {
                "demo::train-clip": make_entry(
                    train_video, video="train-video"
                ),
                "demo::val-clip": make_entry(
                    val_video,
                    video="val-video",
                    start=5,
                    end=5,
                    metadata=False,
                ),
            }
            fixture = Fixture(
                root,
                rows,
                entries,
                [output_record(train_video)],
                [output_record(val_video)],
                [
                    ("train", "video:track:demo:train-video"),
                    ("val", "video:track:demo:val-video"),
                ],
            )
            report = fixture.audit()
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["source_accounting"]["source_rows"], 2)
            self.assertEqual(report["source_accounting"]["expected_records"], 2)
            self.assertTrue(report["source_accounting"]["record_count_balanced"])
            self.assertEqual(
                report["index"]["metadata_availability"]["width"],
                {"available": 1, "unavailable": 1},
            )
            self.assertEqual(
                report["assistant_content_distribution"]["train"]["characters"]["count"],
                1,
            )
            self.assertEqual(
                report["assistant_content_distribution"]["total"]["nonempty_lines"],
                {"count": 2, "min": 1, "p50": 1.0, "p95": 1.0, "p99": 1.0, "max": 1},
            )
            self.assertEqual([item["path"] for item in report["media"]["results"]], sorted([str(train_video.resolve()), str(val_video.resolve())]))

    def test_unindexed_source_rows_are_reported_but_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "indexed.mp4"
            video.write_bytes(b"mp4")
            rows = [source_row(), source_row(video="missing", clip="not-materialized")]
            fixture = Fixture(
                root,
                rows,
                {"demo::clip-1": make_entry(video)},
                [output_record(video)],
                [],
                [("train", "video:track:demo:video-1")],
            )
            report = fixture.audit()
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["source_accounting"]["indexed_retained_rows"], 1)
            self.assertEqual(
                report["source_accounting"]["unindexed_or_index_invalid_retained_rows"],
                1,
            )

    def test_cleaning_rejections_do_not_create_false_accounting_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retained_video = root / "retained.mp4"
            empty_video = root / "empty.mp4"
            outside_video = root / "outside.mp4"
            for video in (retained_video, empty_video, outside_video):
                video.write_bytes(b"mp4")
            rows = [
                source_row(video="retained", clip="retained"),
                source_row(
                    video="empty",
                    clip="empty",
                    points=[],
                    segments=[],
                ),
                source_row(
                    video="outside",
                    clip="outside",
                    points=[{"object_id": "0", "points": [[33, 12], [16, 12]]}],
                ),
            ]
            entries = {
                "demo::retained": make_entry(retained_video, video="retained"),
                "demo::empty": make_entry(empty_video, video="empty"),
                "demo::outside": make_entry(outside_video, video="outside"),
            }
            fixture = Fixture(
                root,
                rows,
                entries,
                [output_record(retained_video)],
                [],
                [("train", "video:track:demo:retained")],
            )
            report = fixture.audit()
            self.assertEqual(report["status"], "passed")
            accounting = report["source_accounting"]
            self.assertEqual(accounting["source_rows"], 3)
            self.assertEqual(accounting["retained_rows"], 1)
            self.assertEqual(accounting["expected_rejected_rows"], 2)
            self.assertEqual(accounting["expected_records"], 1)
            self.assertTrue(accounting["record_count_balanced"])
            self.assertEqual(
                accounting["cleaning_rejections"]["by_reason"],
                {"no_point_tracks": 1, "point_out_of_bounds": 1},
            )

    def test_placeholder_and_media_path_schema_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "track.mp4"
            video.write_bytes(b"mp4")
            fixture = Fixture(
                root,
                [source_row()],
                {"demo::clip-1": make_entry(video)},
                [output_record(video, token="no placeholder")],
                [],
                [("train", "video:track:demo:video-1")],
            )
            report = fixture.audit()
            self.assertEqual(report["status"], "failed")
            codes = {item["code"] for item in report["hard_failures"]}
            self.assertIn("placeholder_mismatch", codes)

    def test_expected_actual_media_counter_must_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "track.mp4"
            video.write_bytes(b"mp4")
            fixture = Fixture(
                root,
                [source_row(), source_row()],
                {"demo::clip-1": make_entry(video)},
                [output_record(video)],
                [],
                [("train", "video:track:demo:video-1")],
            )
            report = fixture.audit()
            self.assertEqual(report["status"], "failed")
            self.assertEqual(
                report["source_accounting"]["media_count_mismatches"],
                [{"path": str(video.resolve()), "expected": 2, "actual": 1}],
            )
            self.assertIn(
                "source_accounting_mismatch",
                {item["code"] for item in report["hard_failures"]},
            )

    def test_annotation_and_canonical_lineage_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mp4"
            second = root / "second.mp4"
            first.write_bytes(b"mp4")
            second.write_bytes(b"mp4")
            rows = [
                source_row(clip="clip-a", start=0, end=0),
                source_row(clip="clip-b", start=1, end=1),
            ]
            entries = {
                "demo::clip-a": make_entry(first, start=0, end=0),
                "demo::clip-b": make_entry(second, start=1, end=1),
            }
            fixture = Fixture(
                root,
                rows,
                entries,
                [output_record(first)],
                [output_record(second)],
                [("train", "video:track:demo:video-1")],
            )
            report = fixture.audit()
            codes = {item["code"] for item in report["hard_failures"]}
            self.assertIn("lineage_cross_split", codes)
            self.assertIn("source_video_cross_split", codes)
            self.assertEqual(report["lineage"]["cross_split_lineages"], 1)

    def test_index_lineage_must_match_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "track.mp4"
            video.write_bytes(b"mp4")
            fixture = Fixture(
                root,
                [source_row(video="actual")],
                {
                    "demo::clip-1": make_entry(
                        video, video="different"
                    )
                },
                [output_record(video)],
                [],
                [("train", "video:track:demo:different")],
            )
            report = fixture.audit()
            self.assertIn(
                "annotation_index_lineage_mismatch",
                {item["code"] for item in report["hard_failures"]},
            )

    def test_media_group_header_and_split_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "track.mp4"
            video.write_bytes(b"mp4")
            fixture = Fixture(
                root,
                [source_row()],
                {"demo::clip-1": make_entry(video)},
                [output_record(video)],
                [],
                [("val", "video:track:demo:video-1")],
            )
            (fixture.output_dir / "media_groups.tsv").write_text(
                "bad\theader\nval\tvideo:track:demo:video-1\n", encoding="utf-8"
            )
            report = fixture.audit()
            codes = {item["code"] for item in report["hard_failures"]}
            self.assertIn("invalid_media_groups_header", codes)
            self.assertIn("media_group_split_mismatch", codes)

    def test_media_probe_mismatch_and_last_frame_decode_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.mp4"
            path.write_bytes(b"mp4")
            expectation = auditor.MediaExpectation(
                str(path.resolve()), 2, 32, 24, 10.0
            )
            probe_payload = json.dumps(
                {
                    "streams": [
                        {
                            "width": 64,
                            "height": 24,
                            "avg_frame_rate": "20/1",
                            "nb_read_frames": "3",
                        }
                    ]
                }
            )

            def fake_run(command, **_kwargs):
                if command[0] == "probe":
                    return subprocess.CompletedProcess(command, 0, probe_payload, "")
                frame_filter = command[command.index("-vf") + 1]
                if frame_filter.endswith("2)"):
                    return subprocess.CompletedProcess(command, 1, b"", b"decode failed")
                return subprocess.CompletedProcess(command, 0, auditor.PNG_SIGNATURE + b"data", b"")

            with mock.patch.object(auditor.subprocess, "run", side_effect=fake_run):
                result = auditor.audit_media(expectation, "probe", "mpeg")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["checks"]["frame_count"]["status"], "mismatch")
            self.assertEqual(result["checks"]["width"]["status"], "mismatch")
            self.assertEqual(result["checks"]["fps"]["status"], "mismatch")
            self.assertEqual(result["checks"]["decode_last"]["status"], "failed")

    def test_missing_optional_probe_expectations_are_unavailable_not_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.mp4"
            path.write_bytes(b"mp4")
            expectation = auditor.MediaExpectation(str(path.resolve()), 1, None, None, None)
            probe_payload = json.dumps(
                {
                    "streams": [
                        {
                            "width": 32,
                            "height": 24,
                            "avg_frame_rate": "10/1",
                            "nb_read_frames": "1",
                        }
                    ]
                }
            )

            def fake_run(command, **_kwargs):
                if command[0] == "probe":
                    return subprocess.CompletedProcess(command, 0, probe_payload, "")
                return subprocess.CompletedProcess(command, 0, auditor.PNG_SIGNATURE + b"data", b"")

            with mock.patch.object(auditor.subprocess, "run", side_effect=fake_run):
                result = auditor.audit_media(expectation, "probe", "mpeg")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["checks"]["width"]["status"], "unavailable")
            self.assertEqual(result["checks"]["height"]["status"], "unavailable")
            self.assertEqual(result["checks"]["fps"]["status"], "unavailable")

    def test_parallel_media_results_are_sorted_and_worker_count_is_bounded(self):
        expectations = {
            name: auditor.MediaExpectation(name, 1, None, None, None)
            for name in ("c.mp4", "a.mp4", "b.mp4")
        }
        lock = threading.Lock()
        active = 0
        peak = 0

        def delayed(expectation, _ffprobe, _ffmpeg):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep({"a.mp4": 0.03, "b.mp4": 0.02, "c.mp4": 0.01}[expectation.path])
            with lock:
                active -= 1
            return {"path": expectation.path, "status": "passed", "errors": []}

        results = auditor.audit_media_parallel(
            expectations, 2, "probe", "mpeg", audit_one=delayed
        )
        self.assertEqual([item["path"] for item in results], ["a.mp4", "b.mp4", "c.mp4"])
        self.assertLessEqual(peak, 2)
        self.assertGreaterEqual(peak, 2)

    def test_assistant_distribution_uses_character_and_nonempty_line_counts(self):
        values = [1, 5, 9]
        self.assertEqual(
            auditor.distribution(values),
            {"count": 3, "min": 1, "p50": 5, "p95": 8.6, "p99": 8.92, "max": 9},
        )
        record = {
            "messages": [
                {"role": "assistant", "content": "abc\n\nxy"},
                {"role": "assistant", "content": "z"},
            ]
        }
        self.assertEqual(auditor.assistant_content_metrics(record), (9, 3))

    def test_non_positive_media_workers_are_rejected(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                auditor.parse_args(["--media-workers", "0"])

    def test_source_filter_cli_is_case_insensitive_and_deduplicated(self):
        args = auditor.parse_args(
            ["--video-track-sources", "MOSE", "mose", "VipSeg"]
        )
        self.assertEqual(args.video_track_sources, ["mose", "vipseg"])

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                auditor.parse_args(["--video-track-sources", "../mose"])

    def test_main_returns_nonzero_for_hard_failure(self):
        failed = {"status": "failed", "hard_failures": [{"code": "test"}]}
        with (
            mock.patch.object(auditor, "run_audit", return_value=failed),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            exit_code = auditor.main([])
        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(output.getvalue()), failed)


if __name__ == "__main__":
    unittest.main()
