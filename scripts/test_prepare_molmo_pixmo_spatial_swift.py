import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parent / "prepare_molmo_pixmo_spatial_swift.py"
SPEC = importlib.util.spec_from_file_location("prepare_new_datasets", SCRIPT_PATH)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


class ConverterTests(unittest.TestCase):

    def setUp(self):
        converter._RUNTIME = converter.RuntimeConfig(
            input_root=".",
            output_root=".",
            val_ratio=0.05,
            seed=42,
            include_subtitles=True,
            max_rejected_examples=10,
            allow_unverified_remote_media=True,
        )
        converter._VIDEO_URLS = {}
        converter._GENERATED_VIDEOS = {}
        converter._TRACK_VIDEOS = {}
        converter._PIXMO_POINT_GROUPS = {}
        converter._PIXMO_URL_GROUPS = {}
        converter._SPATIALVLM_TEST_MEDIA_KEYS = set()

    def test_cli_defaults_keep_subtitles_and_bound_track_windows(self):
        args = converter.parse_args([])
        self.assertTrue(args.include_subtitles)
        self.assertEqual(args.max_track_window_frames, 128)
        self.assertIsNone(args.video_track_sources)

        excluded = converter.parse_args(["--exclude-subtitles"])
        self.assertFalse(excluded.include_subtitles)

    def test_video_track_source_argument_normalizes_and_validates(self):
        args = converter.parse_args(
            [
                "--datasets",
                "Molmo2-VideoTrack",
                "--video-track-sources",
                "MOSE",
                "mOsEv2",
                "mose",
            ]
        )
        converter.validate_args(args)
        self.assertEqual(args.video_track_sources, ["mose", "mosev2"])

        invalid = converter.parse_args(
            ["--datasets", "Molmo2-VideoTrack", "--video-track-sources", "../mose"]
        )
        with self.assertRaisesRegex(SystemExit, "Invalid --video-track-sources"):
            converter.validate_args(invalid)

        unrelated = converter.parse_args(
            ["--datasets", "pixmo-cap", "--video-track-sources", "mose"]
        )
        with self.assertRaisesRegex(SystemExit, "requires Molmo2-VideoTrack"):
            converter.validate_args(unrelated)

    def test_video_track_source_allowlist_filters_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source_name in ("MOSE", "mosev2", "vipseg"):
                source_dir = root / "Molmo2-VideoTrack" / "data" / source_name
                source_dir.mkdir(parents=True)
                (source_dir / f"{source_name}_point_tracks.parquet").write_bytes(b"parquet")

            sources = converter.discover_sources(
                "Molmo2-VideoTrack", root, ["mose", "mosev2"]
            )
            self.assertEqual(
                {source.parent.name.casefold() for source in sources}, {"mose", "mosev2"}
            )

            with self.assertRaisesRegex(SystemExit, "Unknown or unavailable.*missing"):
                converter.discover_sources("Molmo2-VideoTrack", root, ["missing"])

    def test_video_track_source_allowlist_is_recorded_in_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fragments = root / "fragments"
            output = root / "output"
            fragments.mkdir()
            output.mkdir()
            train_path = fragments / "000000.train.jsonl"
            val_path = fragments / "000000.val.jsonl"
            rejected_path = fragments / "000000.rejected.jsonl"
            groups_path = fragments / "000000.groups.tsv"
            train_path.write_text("{}\n", encoding="utf-8")
            val_path.write_text("", encoding="utf-8")
            rejected_path.write_text("", encoding="utf-8")
            groups_path.write_text(
                "train\tvideo:track:mose-family:shared-video\n", encoding="utf-8"
            )
            result = converter.WorkerResult(
                ordinal=0,
                source_path=str(root / "mose_point_tracks.parquet"),
                train_path=str(train_path),
                val_path=str(val_path),
                groups_path=str(groups_path),
                rejected_path=str(rejected_path),
                counters={"source_rows": 1, "written_train": 1},
            )
            report = converter.merge_results(
                "Molmo2-VideoTrack",
                [Path(result.source_path)],
                [result],
                output,
                SimpleNamespace(
                    seed=42,
                    val_ratio=0.05,
                    num_workers=1,
                    max_source_rows=None,
                    include_subtitles=True,
                    max_track_window_frames=128,
                    video_track_sources=["mose", "mosev2"],
                    allow_unverified_remote_media=False,
                    max_reject_ratio=1.0,
                    allow_empty_dataset=False,
                ),
                selected_source_rows=1,
            )
            self.assertEqual(
                report["configuration"]["video_track_sources"], ["mose", "mosev2"]
            )

    def test_media_split_is_stable(self):
        key = "image:sha256:" + "a" * 64
        self.assertEqual(
            converter.split_for_media(key, seed=42, val_ratio=0.05),
            converter.split_for_media(key, seed=42, val_ratio=0.05),
        )

    def test_global_audit_scans_existing_manifests_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset in ("first", "second"):
                output = root / dataset
                output.mkdir()
                (output / "media_groups.tsv").write_text(
                    "split\tmedia_key\ntrain\timage:shared\n", encoding="utf-8"
                )
            report = converter.audit_global_media(root)
            self.assertEqual(report["manifests"], 2)
            self.assertEqual(report["dataset_media_memberships"], 2)
            self.assertEqual(report["unique_media"], 1)

            (root / "second" / "media_groups.tsv").write_text(
                "split\tmedia_key\nval\timage:shared\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "cross-dataset media leakage"):
                converter.audit_global_media(root)

    def test_global_audit_can_be_scoped_to_selected_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "selected"
            stale = root / "stale"
            selected.mkdir()
            stale.mkdir()
            (selected / "media_groups.tsv").write_text(
                "split\tmedia_key\ntrain\tvideo:selected\n", encoding="utf-8"
            )
            (stale / "media_groups.tsv").write_text(
                "invalid header\n", encoding="utf-8"
            )

            report = converter.audit_global_media(root, ["selected"])

            self.assertEqual(report["manifests"], 1)
            self.assertEqual(report["unique_media"], 1)

    def test_long_capqa_expands_qa_list_with_one_media_key(self):
        converter._VIDEO_URLS = {"video-1": "https://example.com/video.mp4"}
        samples = converter.convert_capqa_row(
            {
                "video_id": "video-1",
                "qa_list": [
                    {"Question": "What happens first?", "Answer": "The door opens."},
                    {"Question": "What happens next?", "Answer": "A person enters."},
                ],
            },
            Path("LongCapQA-00000-of-00001.parquet"),
        )
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].media_key, samples[1].media_key)
        for sample in samples:
            converter.validate_record(sample.record)

    def test_subtitleqa_can_include_timestamped_subtitles(self):
        converter._VIDEO_URLS = {"video-2": "https://example.com/video.mp4"}
        converter._RUNTIME = converter.RuntimeConfig(
            input_root=".",
            output_root=".",
            val_ratio=0.05,
            seed=42,
            include_subtitles=True,
            max_rejected_examples=10,
            allow_unverified_remote_media=True,
        )
        sample = converter.convert_subtitleqa_row(
            {
                "video_id": "video-2",
                "Question": "What is closed?",
                "Answer": "The lid.",
                "subtitle": [{"start": 1.2, "end": 3.8, "text": "Close the lid."}],
            }
        )[0]
        converter.validate_record(sample.record)
        self.assertIn("[1.200-3.800] Close the lid.", sample.record["messages"][0]["content"])

    def test_pixmo_points_grounding_contract_and_clamping(self):
        image_hash = "a" * 64
        converter._PIXMO_POINT_GROUPS = {image_hash: image_hash}
        counters = Counter()
        sample = converter.convert_pixmo_points_row(
            {
                "image_url": "https://example.com/image.jpg?token=temporary",
                "image_sha256": image_hash,
                "label": "red apple",
                "points": [{"x": -0.25, "y": 50.0}, {"x": 100.2, "y": 80.0}],
                "count": 2,
            },
            counters,
        )[0]
        converter.validate_record(sample.record)
        self.assertEqual(sample.record["objects"]["bbox"], [[0.0, 0.5], [1.0, 0.8]])
        self.assertEqual(counters["clamped_points"], 2)
        self.assertEqual(converter.placeholder_count(sample.record, "<bbox>"), 2)

    def test_pixmo_points_negative_sample_has_nonempty_answer(self):
        image_hash = "b" * 64
        sample = converter.convert_pixmo_points_row(
            {
                "image_url": "https://example.com/image.png",
                "image_sha256": image_hash,
                "label": "person",
                "points": [],
                "count": 0,
            },
            Counter(),
        )[0]
        converter.validate_record(sample.record)
        self.assertEqual(sample.record["objects"]["bbox"], [])
        self.assertEqual(converter.placeholder_count(sample.record, "<bbox>"), 0)
        self.assertTrue(sample.record["messages"][-1]["content"])

    def test_video_point_keeps_time_binding_without_objects(self):
        converter._VIDEO_URLS = {"video-1": "https://example.com/video.mp4"}
        sample = converter.convert_video_point_row(
            {
                "video_source": "youtube",
                "video_id": "video-1",
                "question": "How many eggs are in the jar?",
                "label": "eggs in the jar",
                "count": 2,
                "raw_timestamps": [25.0],
                "raw_frames": [750],
                "points": [[{"x": 50.0, "y": 25.0}, {"x": 10.0, "y": 90.0}]],
            },
            Counter(),
        )[0]
        converter.validate_record(sample.record)
        self.assertNotIn("objects", sample.record)
        answer = sample.record["messages"][-1]["content"]
        self.assertIn("25.000 seconds, frame 750", answer)
        self.assertIn("[500, 250]", answer)

    def test_video_track_normalizes_points_and_preserves_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "track.mp4"
            video.write_bytes(b"test")
            converter._TRACK_VIDEOS = {
                "demo::clip-1": converter.TrackMediaEntry(
                    windows=(
                        {
                            "media": str(video.resolve()),
                            "source_start_frame": 10,
                            "source_end_frame": 11,
                        },
                    )
                )
            }
            sample = converter.convert_video_track_row(
                {
                    "id": "clip-1_0",
                    "video_dataset": "demo",
                    "video": "clip-1",
                    "clip": "clip-1",
                    "exp": "the red car",
                    "w": 1920,
                    "h": 1080,
                    "fps": 20.0,
                    "start_frame": 10,
                    "end_frame": 11,
                    "n_frames": 2,
                    "points": [{"object_id": "0", "points": [[960, 540], None]}],
                    "segments": [{"object_id": "0", "segments": [[0, 0]]}],
                },
                Counter(),
            )[0]
            converter.validate_record(sample.record)
            answer = sample.record["messages"][-1]["content"]
            self.assertIn("frame 0", answer)
            self.assertIn("[500, 500]", answer)
            self.assertNotIn("frame 1", answer)
            self.assertIn("frames 0 through 1", sample.record["messages"][0]["content"])

    def test_mose_and_mosev2_share_video_track_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "track.mp4"
            video.write_bytes(b"test")
            row = {
                "id": "clip-1_0",
                "video": "shared-video",
                "clip": "clip-1",
                "exp": "the red car",
                "w": 100,
                "h": 100,
                "fps": 20.0,
                "start_frame": 0,
                "end_frame": 0,
                "n_frames": 1,
                "points": [{"object_id": "0", "points": [[50, 50]]}],
                "segments": [{"object_id": "0", "segments": [[0, 0]]}],
            }
            samples = []
            for dataset in ("mose", "mosev2"):
                converter._TRACK_VIDEOS = {
                    f"{dataset}::clip-1": converter.TrackMediaEntry(
                        windows=(
                            {
                                "media": str(video.resolve()),
                                "source_start_frame": 0,
                                "source_end_frame": 0,
                            },
                        )
                    )
                }
                samples.append(
                    converter.convert_video_track_row(
                        {**row, "video_dataset": dataset}, Counter()
                    )[0]
                )

            self.assertEqual(samples[0].media_key, "video:track:mose-family:shared-video")
            self.assertEqual(samples[0].media_key, samples[1].media_key)

    def test_video_track_index_lineage_is_preferred_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "track.mp4"
            video.write_bytes(b"test")
            index_path = root / "index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "demo::clip-1": {
                            "source_video_id": "demo::shared-video",
                            "lineage_key": "canonical-family::shared-video",
                            "windows": [
                                {
                                    "mode": "cropped",
                                    "path": str(video),
                                    "source_start_frame": 0,
                                    "source_end_frame": 0,
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            converter._TRACK_VIDEOS = converter.load_track_video_index(index_path)
            row = {
                "id": "clip-1_0",
                "video_dataset": "demo",
                "video": "shared-video",
                "clip": "clip-1",
                "exp": "the red car",
                "w": 100,
                "h": 100,
                "fps": 20.0,
                "start_frame": 0,
                "end_frame": 0,
                "n_frames": 1,
                "points": [{"object_id": "0", "points": [[50, 50]]}],
                "segments": [{"object_id": "0", "segments": [[0, 0]]}],
            }
            sample = converter.convert_video_track_row(row, Counter())[0]
            self.assertEqual(sample.media_key, "video:track:canonical-family:shared-video")

            with self.assertRaisesRegex(ValueError, "does not match annotation"):
                converter.convert_video_track_row(
                    {**row, "video": "different-video"}, Counter()
                )

            invalid_index = json.loads(index_path.read_text(encoding="utf-8"))
            invalid_index["demo::clip-1"]["lineage_key"] = "canonical-family::different-video"
            index_path.write_text(json.dumps(invalid_index), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "does not match source_video_id video"):
                converter.load_track_video_index(index_path)

    def test_video_track_clip_index_takes_precedence_over_video_fallback(self):
        clip_entry = converter.TrackMediaEntry(
            windows=(
                {
                    "media": "https://example.invalid/clip.mp4",
                    "source_start_frame": 10,
                    "source_end_frame": 11,
                },
            )
        )
        video_entry = converter.TrackMediaEntry(
            windows=(
                {
                    "media": "https://example.invalid/video.mp4",
                    "source_start_frame": 0,
                    "source_end_frame": 99,
                },
            )
        )
        converter._TRACK_VIDEOS = {
            "demo::clip-1": clip_entry,
            "demo::video-1": video_entry,
        }
        row = {
            "id": "clip-1_0",
            "video_dataset": "demo",
            "video": "video-1",
            "clip": "clip-1",
            "start_frame": 10,
            "end_frame": 11,
        }

        _, windows = converter.resolve_track_windows(row)

        self.assertEqual(windows, list(clip_entry.windows))

    def test_video_track_conflicting_specific_aliases_still_fail(self):
        converter._TRACK_VIDEOS = {
            "demo::clip-1": converter.TrackMediaEntry(
                windows=(
                    {
                        "media": "/media/clip.mp4",
                        "source_start_frame": 10,
                        "source_end_frame": 11,
                    },
                )
            ),
            "row-1": converter.TrackMediaEntry(
                windows=(
                    {
                        "media": "/media/row.mp4",
                        "source_start_frame": 10,
                        "source_end_frame": 11,
                    },
                )
            ),
        }
        row = {
            "id": "row-1",
            "video_dataset": "demo",
            "video": "video-1",
            "clip": "clip-1",
            "start_frame": 10,
            "end_frame": 11,
        }

        with self.assertRaisesRegex(ValueError, "conflicting VideoTrack media index aliases"):
            converter.resolve_track_windows(row)

    def test_video_track_uses_real_cropped_windows_with_one_media_split_key(self):
        with tempfile.TemporaryDirectory() as directory:
            first_video = Path(directory) / "track-000.mp4"
            second_video = Path(directory) / "track-001.mp4"
            first_video.write_bytes(b"first")
            second_video.write_bytes(b"second")
            converter._RUNTIME = converter.RuntimeConfig(
                input_root=".",
                output_root=".",
                val_ratio=0.05,
                seed=42,
                include_subtitles=True,
                max_rejected_examples=10,
                allow_unverified_remote_media=True,
                max_track_window_frames=2,
            )
            converter._TRACK_VIDEOS = {
                "demo::clip-1": converter.TrackMediaEntry(
                    windows=(
                        {
                            "media": str(first_video.resolve()),
                            "source_start_frame": 10,
                            "source_end_frame": 11,
                        },
                        {
                            "media": str(second_video.resolve()),
                            "source_start_frame": 12,
                            "source_end_frame": 13,
                        },
                    )
                )
            }
            row = {
                "id": "clip-1_0",
                "video_dataset": "demo",
                "video": "source-video",
                "clip": "clip-1",
                "exp": "the red car",
                "w": 1000,
                "h": 1000,
                "fps": 20.0,
                "start_frame": 10,
                "end_frame": 13,
                "n_frames": 4,
                "points": [
                    {
                        "object_id": "0",
                        "points": [[100, 100], [200, 200], [300, 300], [400, 400]],
                    }
                ],
                "segments": [{"object_id": "0", "segments": [[0, 3]]}],
            }
            samples = converter.convert_video_track_row(row, Counter())

            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0].media_key, samples[1].media_key)
            self.assertEqual(samples[0].record["videos"], [str(first_video.resolve())])
            self.assertEqual(samples[1].record["videos"], [str(second_video.resolve())])
            first_answer = samples[0].record["messages"][-1]["content"]
            second_answer = samples[1].record["messages"][-1]["content"]
            self.assertIn("frame 0", first_answer)
            self.assertIn("frame 1", first_answer)
            self.assertIn("[100, 100]", first_answer)
            self.assertNotIn("[300, 300]", first_answer)
            self.assertIn("frame 0", second_answer)
            self.assertIn("frame 1", second_answer)
            self.assertIn("[300, 300]", second_answer)
            self.assertNotIn("[100, 100]", second_answer)

    def test_video_track_rejects_gapped_cropped_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            first_video = Path(directory) / "track-000.mp4"
            second_video = Path(directory) / "track-001.mp4"
            first_video.write_bytes(b"first")
            second_video.write_bytes(b"second")
            converter._TRACK_VIDEOS = {
                "demo::clip-1": converter.TrackMediaEntry(
                    windows=(
                        {
                            "media": str(first_video.resolve()),
                            "source_start_frame": 10,
                            "source_end_frame": 10,
                        },
                        {
                            "media": str(second_video.resolve()),
                            "source_start_frame": 12,
                            "source_end_frame": 13,
                        },
                    )
                )
            }
            row = {
                "id": "clip-1_0",
                "video_dataset": "demo",
                "video": "source-video",
                "clip": "clip-1",
                "exp": "the red car",
                "w": 1000,
                "h": 1000,
                "fps": 20.0,
                "start_frame": 10,
                "end_frame": 13,
                "n_frames": 4,
                "points": [{"object_id": "0", "points": [[1, 1], [2, 2], [3, 3], [4, 4]]}],
                "segments": [{"object_id": "0", "segments": [[0, 3]]}],
            }
            with self.assertRaisesRegex(ValueError, "continuously cover"):
                converter.convert_video_track_row(row, Counter())

    def test_generated_video_index_matches_nested_archive_path(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sora2-1128" / "temporal_and_continuity" / "clip_v1.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"test")
            index = converter.build_generated_video_index(Path(directory))
            candidates = converter.generated_video_id_candidates("sora2-1128/clip_v1")
            self.assertTrue(any(candidate in index for candidate in candidates))

    def test_pixmo_cap_groups_query_variants_by_canonical_url(self):
        first = converter.convert_pixmo_cap_row(
            {"image_url": "https://EXAMPLE.com/a.jpg?token=one", "caption": "First caption."}
        )[0]
        second = converter.convert_pixmo_cap_row(
            {"image_url": "https://example.com/a.jpg?token=two", "caption": "Second caption."}
        )[0]
        self.assertEqual(first.media_key, second.media_key)

    def test_pixmo_cap_uses_cross_dataset_url_sha_component(self):
        canonical = "https://example.com/shared.jpg"
        converter._PIXMO_URL_GROUPS = {canonical: "c" * 64}
        sample = converter.convert_pixmo_cap_row(
            {"image_url": canonical + "?download=1", "caption": "Shared image."}
        )[0]
        self.assertEqual(sample.media_key, "image:sha256:" + "c" * 64)

    def test_remote_media_requires_explicit_opt_in(self):
        converter._RUNTIME = converter.RuntimeConfig(
            input_root=".",
            output_root=".",
            val_ratio=0.05,
            seed=42,
            include_subtitles=False,
            max_rejected_examples=10,
            allow_unverified_remote_media=False,
        )
        record = converter.make_qa_record(
            "video", "https://example.com/video.mp4", "What happens?", "A door opens."
        )
        with self.assertRaisesRegex(ValueError, "allow-unverified-remote-media"):
            converter.validate_record(record)

    def test_spatialvlm_rejects_multi_image_rows(self):
        with self.assertRaisesRegex(ValueError, "multi-image"):
            converter.convert_spatialvlm_row(
                {
                    "messages": [],
                    "images": [
                        {"bytes": b"first", "path": "first.jpg"},
                        {"bytes": b"second", "path": "second.jpg"},
                    ],
                }
            )

    def test_source_row_limit_samples_every_authoritative_file(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cap_path = root / "CapQA-00000-of-00001.parquet"
            long_path = root / "LongCapQA-00000-of-00001.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [{"video_id": "cap", "Question": "Q", "Answer": "A"}]
                ),
                cap_path,
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "video_id": "long",
                            "qa_list": [{"Question": "Q", "Answer": "A"}],
                        }
                    ]
                ),
                long_path,
            )
            tasks, selected = converter.build_tasks(
                "Molmo2-VideoCapQA", [cap_path, long_path], root / "fragments", 2
            )
            self.assertEqual(selected, 2)
            self.assertEqual({Path(task.source_path).name for task in tasks}, {cap_path.name, long_path.name})

    def test_spatialvlm_extracts_image_and_replaces_norm1_box(self):
        with tempfile.TemporaryDirectory() as directory:
            converter._RUNTIME = converter.RuntimeConfig(
                input_root=directory,
                output_root=directory,
                val_ratio=0.05,
                seed=42,
                include_subtitles=False,
                max_rejected_examples=10,
            )
            sample = converter.convert_spatialvlm_row(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "index": 0, "text": None},
                                {
                                    "type": "text",
                                    "index": None,
                                    "text": "Describe the region [0.04, 0.25, 0.43, 0.86].",
                                },
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": "A woman."}]},
                    ],
                    "images": [{"bytes": b"\xff\xd8\xfftest-image", "path": "source.jpg"}],
                }
            )[0]
            converter.validate_record(sample.record)
            self.assertTrue(Path(sample.record["images"][0]).is_file())
            self.assertEqual(sample.record["objects"]["bbox"], [[0.04, 0.25, 0.43, 0.86]])
            self.assertEqual(converter.placeholder_count(sample.record, "<bbox>"), 1)

    def test_spatialvlm_expands_independent_qa_pairs_with_same_media_key(self):
        with tempfile.TemporaryDirectory() as directory:
            converter._RUNTIME = converter.RuntimeConfig(
                input_root=directory,
                output_root=directory,
                val_ratio=0.05,
                seed=42,
                include_subtitles=False,
                max_rejected_examples=10,
            )
            samples = converter.convert_spatialvlm_row(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "index": 0},
                                {"type": "text", "text": "What is shown?"},
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": "A kitchen."}]},
                        {"role": "user", "content": [{"type": "text", "text": "Where is the stove?"}]},
                        {"role": "assistant", "content": [{"type": "text", "text": "On the left."}]},
                    ],
                    "images": [{"bytes": b"\x89PNG\r\n\x1a\nimage", "path": "source.png"}],
                }
            )
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0].media_key, samples[1].media_key)
            for sample in samples:
                converter.validate_record(sample.record)
                self.assertEqual(converter.placeholder_count(sample.record, "<image>"), 1)

    def test_spatialvlm_preserves_official_split_and_excludes_test_hash_from_train(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        def row(payload: bytes, name: str):
            return {
                "messages": [
                    {"role": "user", "content": "<image>What is shown?"},
                    {"role": "assistant", "content": "A room."},
                ],
                "images": [{"bytes": payload, "path": name}],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train-00000-of-00001.parquet"
            test_path = root / "test-00000-of-00001.parquet"
            shared = b"shared-test-image"
            pq.write_table(pa.Table.from_pylist([row(shared, "train.jpg")]), train_path)
            pq.write_table(pa.Table.from_pylist([row(shared, "test.jpg")]), test_path)
            converter._RUNTIME = converter.RuntimeConfig(
                input_root=directory,
                output_root=directory,
                val_ratio=0.05,
                seed=42,
                include_subtitles=False,
                max_rejected_examples=10,
            )
            keys, source_rows = converter.build_spatialvlm_test_media_keys(
                [train_path, test_path]
            )
            converter._SPATIALVLM_TEST_MEDIA_KEYS = keys
            self.assertEqual(source_rows, 1)
            self.assertEqual(converter.spatialvlm_official_split(train_path), "train")
            self.assertEqual(converter.spatialvlm_official_split(test_path), "val")

            train_task = converter.RowGroupTask(
                dataset="spatialvlm",
                source_path=str(train_path),
                row_group=0,
                ordinal=0,
                row_limit=None,
                fragment_dir=str(root / "train-fragments"),
                forced_split="train",
            )
            test_task = converter.RowGroupTask(
                dataset="spatialvlm",
                source_path=str(test_path),
                row_group=0,
                ordinal=1,
                row_limit=None,
                fragment_dir=str(root / "test-fragments"),
                forced_split="val",
            )
            train_result = converter.process_row_group(train_task)
            test_result = converter.process_row_group(test_task)
            self.assertEqual(
                train_result.counters["excluded_train_records_with_test_image_hash"], 1
            )
            self.assertEqual(train_result.counters.get("written_train", 0), 0)
            self.assertEqual(test_result.counters["written_val"], 1)
            self.assertEqual(
                test_result.counters["official_test_records_written_to_val"], 1
            )
            self.assertEqual(
                test_result.counters.get("official_test_records_written_to_train", 0), 0
            )
            self.assertEqual(Path(train_result.train_path).read_text(encoding="utf-8"), "")
            self.assertTrue(Path(test_result.val_path).read_text(encoding="utf-8").strip())

            output_dir = root / "merged"
            output_dir.mkdir()
            report = converter.merge_results(
                "spatialvlm",
                [train_path, test_path],
                [train_result, test_result],
                output_dir,
                SimpleNamespace(
                    seed=42,
                    val_ratio=0.05,
                    num_workers=1,
                    max_source_rows=None,
                    include_subtitles=False,
                    max_track_window_frames=128,
                    allow_unverified_remote_media=False,
                    max_reject_ratio=1.0,
                    allow_empty_dataset=False,
                ),
                selected_source_rows=2,
                spatialvlm_test_source_rows=source_rows,
            )
            audit = report["spatialvlm_official_split_audit"]
            self.assertEqual(audit["official_test_source_rows"], 1)
            self.assertEqual(audit["official_test_records_written_to_val"], 1)
            self.assertEqual(audit["official_test_records_written_to_train"], 0)
            self.assertEqual(audit["official_train_records_written_to_val"], 0)
            self.assertEqual(audit["post_filter_train_test_hash_overlap"], 0)
            for key in ("train", "val", "rejected", "media_groups"):
                self.assertEqual(
                    report["output_sha256"][key],
                    converter.file_sha256(Path(report["output_files"][key])),
                )

    def test_objects_without_images_are_rejected(self):
        record = {
            "messages": [
                {"role": "user", "content": "<video>Locate the target."},
                {"role": "assistant", "content": "<bbox>"},
            ],
            "videos": ["https://example.com/video.mp4"],
            "objects": {"ref": [], "bbox": [[0.5, 0.5]], "bbox_type": "norm1"},
        }
        with self.assertRaisesRegex(ValueError, "requires images"):
            converter.validate_record(record)


if __name__ == "__main__":
    unittest.main()
