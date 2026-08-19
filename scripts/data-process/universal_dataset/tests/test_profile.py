from __future__ import annotations

import io
import json
import sqlite3
import struct
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import universal_dataset.profile as profile_module
from universal_dataset.profile import profile_path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def encode_varint(value: int) -> bytes:
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def protobuf_bytes_field(field_number: int, payload: bytes) -> bytes:
    return encode_varint((field_number << 3) | 2) + encode_varint(len(payload)) + payload


def bytes_feature(*values: bytes) -> bytes:
    bytes_list = b"".join(protobuf_bytes_field(1, value) for value in values)
    return protobuf_bytes_field(1, bytes_list)


def int64_feature(*values: int) -> bytes:
    packed = b"".join(encode_varint(value) for value in values)
    return protobuf_bytes_field(3, protobuf_bytes_field(1, packed))


def features_message(values) -> bytes:
    result = []
    for key, feature in sorted(values.items()):
        entry = protobuf_bytes_field(1, key.encode("utf-8")) + protobuf_bytes_field(2, feature)
        result.append(protobuf_bytes_field(1, entry))
    return b"".join(result)


def example_message(values) -> bytes:
    return protobuf_bytes_field(1, features_message(values))


def sequence_example_message(context, feature_lists) -> bytes:
    encoded_lists = []
    for key, features in sorted(feature_lists.items()):
        feature_list = b"".join(protobuf_bytes_field(1, feature) for feature in features)
        entry = protobuf_bytes_field(1, key.encode("utf-8")) + protobuf_bytes_field(2, feature_list)
        encoded_lists.append(protobuf_bytes_field(1, entry))
    return (
        protobuf_bytes_field(1, features_message(context))
        + protobuf_bytes_field(2, b"".join(encoded_lists))
    )


def tfrecord_frame(payload: bytes) -> bytes:
    length = struct.pack("<Q", len(payload))
    return (
        length
        + struct.pack("<I", profile_module._masked_crc32c(length))
        + payload
        + struct.pack("<I", profile_module._masked_crc32c(payload))
    )


class ProfileSelectionTest(unittest.TestCase):
    def test_selection_is_deterministic_and_covers_top_level_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            for dataset in ("zeta", "alpha", "middle"):
                for index in (3, 1, 2, 0):
                    write_json(first / dataset / "records" / "{:02d}.json".format(index), {"value": index})
            for dataset in ("middle", "alpha", "zeta"):
                for index in (0, 2, 1, 3):
                    write_json(second / dataset / "records" / "{:02d}.json".format(index), {"value": index})

            serial = profile_path(first, max_files=3, relative_paths=True, workers=1)
            parallel = profile_path(second, max_files=3, relative_paths=True, workers=3)
            serial_paths = [item["path"] for item in serial["files"]]
            parallel_paths = [item["path"] for item in parallel["files"]]

            self.assertEqual(serial, parallel)
            self.assertEqual(serial_paths, parallel_paths)
            self.assertEqual({path.split("/")[0] for path in serial_paths}, {"alpha", "middle", "zeta"})
            self.assertEqual(serial["profiled_files"], 3)
            self.assertTrue(serial["truncated"])
            self.assertEqual(
                serial["selection_buckets"],
                [
                    {"dataset": "alpha", "format": "json", "candidates": 4, "selected": 1, "omitted": 3},
                    {"dataset": "middle", "format": "json", "candidates": 4, "selected": 1, "omitted": 3},
                    {"dataset": "zeta", "format": "json", "candidates": 4, "selected": 1, "omitted": 3},
                ],
            )

    def test_columnar_schema_containers_get_remaining_capacity_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(4):
                write_json(root / "dataset_a" / "json" / "{}.json".format(index), {"value": index})
            for index in range(3):
                path = root / "dataset_a" / "parquet" / "{}.parquet".format(index)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not-a-real-parquet-file")
            arrow = root / "dataset_b" / "metadata" / "part.arrow"
            arrow.parent.mkdir(parents=True, exist_ok=True)
            arrow.write_bytes(b"not-a-real-arrow-file")
            csv_path = root / "dataset_b" / "metadata" / "labels.csv"
            csv_path.write_text("id,label\n1,x\n", encoding="utf-8")

            report = profile_path(root, max_files=6, relative_paths=True)
            selected = [item["path"] for item in report["files"]]

            self.assertEqual(sum(path.endswith(".parquet") for path in selected), 3)
            self.assertEqual(sum(path.endswith(".arrow") for path in selected), 1)
            self.assertEqual(sum(path.endswith(".json") for path in selected), 1)
            self.assertEqual(sum(path.endswith(".csv") for path in selected), 1)


class ProfileMetadataTest(unittest.TestCase):
    def test_appledouble_and_known_os_metadata_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "dataset" / "record.json", {"id": 1})
            write_json(root / "dataset" / "._record.json", {"resource_fork": True})
            write_json(root / "dataset" / "__MACOSX" / "copy.json", {"metadata": True})
            write_json(root / "dataset" / ".AppleDouble" / "copy.json", {"metadata": True})
            write_json(root / ".cache" / "download" / "metadata.json", {"cache": True})
            (root / "dataset" / ".DS_Store").write_bytes(b"metadata")

            report = profile_path(root, relative_paths=True)

            self.assertEqual(report["total_files"], 6)
            self.assertEqual(report["eligible_files"], 1)
            self.assertEqual(report["excluded_files"], 5)
            self.assertEqual(
                report["excluded_reasons"],
                {
                    "appledouble_sidecar": 1,
                    "cache_directory": 1,
                    "macos_metadata_directory": 2,
                    "os_metadata_file": 1,
                },
            )
            self.assertEqual([item["path"] for item in report["files"]], ["dataset/record.json"])


class ProfileJsonRootTest(unittest.TestCase):
    def test_mapping_traversal_never_requests_item_past_breadth_cap(self):
        class GuardedMapping(dict):
            def items(self):
                for index, item in enumerate(super().items()):
                    if index >= profile_module.MAX_MAPPING_KEYS:
                        raise AssertionError("mapping traversal exceeded its breadth cap")
                    yield item

        value = GuardedMapping(
            ("field_{:03d}".format(index), {"value": index})
            for index in range(profile_module.MAX_MAPPING_KEYS + 1)
        )
        profiler = profile_module.ShapeProfiler()

        profiler.add(value)

        self.assertTrue(profiler.result()["fields"])

    def test_dynamic_root_object_map_uses_wildcard_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.json"
            write_json(
                path,
                {
                    "frame_alpha": {"question": "q1", "answer": "a1"},
                    "frame_beta": {"question": "q2", "answer": "a2"},
                    "frame_gamma": {"question": "q3", "answer": "a3", "optional": True},
                },
            )

            report = profile_path(path, sample_rows=2, relative_paths=True)
            file_report = report["files"][0]
            fields = file_report["schema"]["fields"]

            self.assertEqual(file_report["root_kind"], "object_map")
            self.assertEqual(file_report["total_records"], 3)
            self.assertEqual(file_report["sampled_records"], 2)
            self.assertIn("$.*.question", fields)
            self.assertFalse(any("frame_alpha" in field or "frame_beta" in field for field in fields))

    def test_semantic_document_root_remains_an_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coco.json"
            write_json(
                path,
                {
                    "images": [{"id": 1, "file_name": "a.jpg"}],
                    "annotations": [{"id": 2, "image_id": 1}],
                    "categories": [{"id": 3, "name": "thing"}],
                },
            )

            file_report = profile_path(path, relative_paths=True)["files"][0]

            self.assertEqual(file_report["root_kind"], "object")
            self.assertIn("$.images[].file_name", file_report["schema"]["fields"])

    def test_nested_dynamic_maps_use_wildcards_and_mapping_breadth_is_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested.json"
            wide_mapping = {"annotations": 0}
            for first in "abcd":
                for second in "abcdefghijklmnopqrstuvwxyz":
                    wide_mapping["key{}{}".format(first, second)] = 1
            write_json(
                path,
                {
                    "records": {
                        "123456": {"question": "q1", "answer": "a1"},
                        "987654": {"question": "q2", "answer": "a2"},
                    },
                    "metadata": {
                        "555555": {"source": "single-id-map"},
                        "wide": wide_mapping,
                    },
                },
            )

            fields = profile_path(path, relative_paths=True)["files"][0]["schema"]["fields"]

            self.assertIn("$.records.*.question", fields)
            self.assertIn("$.metadata.*.source", fields)
            self.assertFalse(
                any(
                    identifier in field
                    for field in fields
                    for identifier in ("123456", "987654", "555555")
                )
            )
            wide_children = [
                field for field in fields
                if field.startswith("$.metadata.wide.") and field != "$.metadata.wide"
            ]
            self.assertLessEqual(len(wide_children), profile_module.MAX_MAPPING_KEYS)


class ProfileSummaryTest(unittest.TestCase):
    def test_status_counts_and_schema_families_exclude_failures_and_schema_less_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "valid.json", {"name": "sample", "count": 1})
            (root / "invalid.json").write_text("{", encoding="utf-8")
            (root / "invalid.jsonl").write_text("not-json\n", encoding="utf-8")
            with zipfile.ZipFile(root / "media.zip", "w") as archive:
                archive.writestr("image.jpg", b"image")

            report = profile_path(root, relative_paths=True)

            self.assertEqual(report["status_counts"], {"error": 1, "invalid": 1, "ok": 2})
            families = {family["kind"]: family for family in report["schema_families"]}
            self.assertEqual(set(families), {"json", "zip"})
            family = families["json"]
            self.assertEqual(family["kind"], "json")
            self.assertEqual(family["files"], 1)
            self.assertTrue(family["schema"])
            self.assertEqual(
                {(issue["status"], issue["count"]) for issue in report["issues"]},
                {("error", 1), ("invalid", 1)},
            )
            error_report = next(item for item in report["files"] if item["path"] == "invalid.json")
            self.assertEqual(error_report["kind"], "json")
            valid_report = next(item for item in report["files"] if item["path"] == "valid.json")
            self.assertNotIn('"examples"', json.dumps(valid_report["schema"], sort_keys=True))
            self.assertTrue(all(not Path(item["path"]).is_absolute() for item in report["files"]))

            compact = profile_path(root, relative_paths=True, workers=3, summary_only=True)
            self.assertNotIn("files", compact)
            self.assertEqual(compact["file_reports_omitted"], 4)
            self.assertEqual(compact["status_counts"], report["status_counts"])
            self.assertEqual(compact["issues"], report["issues"])
            self.assertEqual(compact["schema_families"], report["schema_families"])

    def test_schema_family_fingerprint_ignores_sample_counts_and_array_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_jsonl(root / "short.jsonl", [{"id": 1, "values": [1]}])
            write_jsonl(
                root / "long.jsonl",
                [
                    {"id": index, "values": [1, 2, 3, 4]}
                    for index in range(5)
                ],
            )

            report = profile_path(root, include_examples=True, relative_paths=True)

            self.assertEqual(len(report["schema_families"]), 1)
            self.assertEqual(report["schema_families"][0]["files"], 2)
            self.assertEqual(report["schema_families"][0]["records"], 6)


class ProfileTFRecordTest(unittest.TestCase):
    def test_crc32c_known_vector_and_example_sequence_example_summary(self):
        self.assertEqual(profile_module._crc32c(b"123456789"), 0xE3069283)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.tfrecords"
            example = example_message(
                {"image": bytes_feature(b"jpeg"), "label": int64_feature(3)}
            )
            sequence = sequence_example_message(
                {"unique_id": bytes_feature(b"episode-1")},
                {
                    "images": [bytes_feature(b"frame-1"), bytes_feature(b"frame-2")],
                    "timestamps": [int64_feature(10), int64_feature(20)],
                },
            )
            path.write_bytes(tfrecord_frame(example) + tfrecord_frame(sequence))

            file_report = profile_path(path, relative_paths=True)["files"][0]

            self.assertEqual(file_report["status"], "ok")
            self.assertEqual(file_report["total_records"], 2)
            self.assertEqual(file_report["sampled_records"], 2)
            self.assertEqual(file_report["record_type_counts"], {"Example": 1, "SequenceExample": 1})
            self.assertEqual(
                file_report["schema"]["example"]["features"]["image"]["types"],
                {"bytes": 1},
            )
            self.assertEqual(
                file_report["schema"]["sequence_example"]["feature_lists"]["timestamps"]["types"],
                {"int64": 2},
            )

    def test_tfrecord_shard_name_is_selected_and_profiled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "robovqa" / "train.tfrecord-00000-of-00175"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tfrecord_frame(example_message({"label": int64_feature(1)})))

            report = profile_path(root, relative_paths=True)

            self.assertEqual(report["structured_files"], 1)
            self.assertEqual(report["profiled_files"], 1)
            self.assertFalse(report["truncated"])
            file_report = report["files"][0]
            self.assertEqual(file_report["path"], "robovqa/train.tfrecord-00000-of-00175")
            self.assertEqual(file_report["kind"], "tfrecord")
            self.assertEqual(file_report["status"], "ok")
            self.assertEqual(file_report["total_records"], 1)

    def test_tfrecord_rejects_bad_length_and_data_crc32c(self):
        payload = example_message({"value": int64_feature(1)})
        frame = tfrecord_frame(payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corruptions = {
                "length": frame[:8] + bytes([frame[8] ^ 1]) + frame[9:],
                "data": frame[:-1] + bytes([frame[-1] ^ 1]),
            }
            for name, corrupted in corruptions.items():
                with self.subTest(name=name):
                    path = root / "{}.tfrecord".format(name)
                    path.write_bytes(corrupted)
                    file_report = profile_path(path, relative_paths=True)["files"][0]
                    self.assertEqual(file_report["status"], "invalid")
                    self.assertIn("CRC32C", file_report["error"])

    def test_empty_example_is_a_valid_protobuf_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "empty-example.tfrecord"
            path.write_bytes(tfrecord_frame(b""))

            file_report = profile_path(path, relative_paths=True)["files"][0]

            self.assertEqual(file_report["status"], "ok")
            self.assertEqual(file_report["record_type_counts"], {"Example": 1})


class ProfileArchiveTest(unittest.TestCase):
    def test_zip_and_tar_profile_embedded_json_without_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zip_path = root / "records.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("a.jsonl", '{"id": 1, "values": [1]}\n')
                archive.writestr(
                    "nested/b.jsonl",
                    '{"id": 2, "values": [1, 2]}\n{"id": 3, "values": [3, 4]}\n',
                )

            tar_path = root / "records.tar"
            tar_payload = json.dumps([{"name": "first"}, {"name": "second"}]).encode("utf-8")
            with tarfile.open(tar_path, "w") as archive:
                item = tarfile.TarInfo("../../must-not-be-extracted.json")
                item.size = len(tar_payload)
                archive.addfile(item, io.BytesIO(tar_payload))

            zip_report = profile_path(zip_path, relative_paths=True)["files"][0]
            tar_report = profile_path(tar_path, relative_paths=True)["files"][0]

            self.assertEqual(zip_report["embedded_status_counts"], {"ok": 2})
            self.assertEqual(len(zip_report["embedded_schema_families"]), 1)
            self.assertEqual(zip_report["embedded_schema_families"][0]["files"], 2)
            self.assertEqual(tar_report["embedded_status_counts"], {"ok": 1})
            self.assertEqual(len(tar_report["embedded_schema_families"]), 1)
            self.assertFalse((root.parent / "must-not-be-extracted.json").exists())

    def test_archive_json_sampling_is_byte_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("large.json", json.dumps({"payload": "x" * 256}))

            with mock.patch.object(profile_module, "MAX_ARCHIVE_MEMBER_SAMPLE_BYTES", 32):
                file_report = profile_path(path, relative_paths=True)["files"][0]

            self.assertEqual(file_report["embedded_status_counts"], {"skipped": 1})
            self.assertEqual(file_report["embedded_sampled_bytes"], 0)
            self.assertEqual(file_report["embedded_schema_families"], [])


class ProfileRoboticsContainerTest(unittest.TestCase):
    def test_db3_profiles_schema_and_topics_without_message_blob_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recording.db3"
            marker = b"MESSAGE_BLOB_MUST_NOT_APPEAR"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT)"
                )
                connection.execute(
                    "CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)"
                )
                connection.execute(
                    "INSERT INTO topics VALUES (1, '/camera', 'sensor_msgs/msg/Image', 'cdr')"
                )
                connection.execute("INSERT INTO messages VALUES (1, 1, 123, ?)", (marker,))
                connection.commit()

            file_report = profile_path(path, relative_paths=True)["files"][0]
            table_names = [table["name"] for table in file_report["schema"]["tables"]]

            self.assertEqual(file_report["status"], "ok")
            self.assertEqual(table_names, ["messages", "topics"])
            self.assertEqual(file_report["topics_total"], 1)
            self.assertEqual(file_report["topics"][0]["name"], "/camera")
            self.assertNotIn(marker.decode("ascii"), json.dumps(file_report, sort_keys=True))

    def test_optional_mcap_and_bag_dependencies_report_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("empty.mcap", {"mcap": None, "mcap.reader": None}),
                ("empty.bag", {"rosbags": None, "rosbags.rosbag1": None}),
            )
            for filename, missing_modules in cases:
                with self.subTest(filename=filename):
                    path = root / filename
                    path.write_bytes(b"")
                    with mock.patch.dict(sys.modules, missing_modules):
                        file_report = profile_path(path, relative_paths=True)["files"][0]
                    self.assertEqual(file_report["status"], "skipped")
                    self.assertIn("optional", file_report["error"])

    def test_mcap_and_bag_metadata_paths_do_not_require_message_decoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mcap_path = root / "metadata.mcap"
            bag_path = root / "metadata.bag"
            mcap_path.write_bytes(b"")
            bag_path.write_bytes(b"")

            mcap_package = types.ModuleType("mcap")
            mcap_package.__path__ = []
            mcap_reader_module = types.ModuleType("mcap.reader")
            summary = types.SimpleNamespace(
                schemas={
                    1: types.SimpleNamespace(id=1, name="sensor_msgs/msg/Image", encoding="ros2msg")
                },
                channels={
                    1: types.SimpleNamespace(
                        id=1, topic="/camera", message_encoding="cdr", schema_id=1
                    )
                },
                statistics=types.SimpleNamespace(message_count=12),
            )
            mcap_reader_module.make_reader = lambda stream: types.SimpleNamespace(
                get_summary=lambda: summary
            )
            with mock.patch.dict(
                sys.modules, {"mcap": mcap_package, "mcap.reader": mcap_reader_module}
            ):
                mcap_report = profile_path(mcap_path, relative_paths=True)["files"][0]

            rosbags_package = types.ModuleType("rosbags")
            rosbags_package.__path__ = []
            rosbag1_module = types.ModuleType("rosbags.rosbag1")

            class FakeBagReader:
                def __init__(self, path):
                    self.path = path
                    self.connections = [
                        types.SimpleNamespace(
                            topic="/pose", msgtype="geometry_msgs/msg/Pose", msgcount=7
                        )
                    ]

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    return False

            rosbag1_module.Reader = FakeBagReader
            with mock.patch.dict(
                sys.modules, {"rosbags": rosbags_package, "rosbags.rosbag1": rosbag1_module}
            ):
                bag_report = profile_path(bag_path, relative_paths=True)["files"][0]

            self.assertEqual(mcap_report["status"], "ok")
            self.assertEqual(mcap_report["total_records"], 12)
            self.assertEqual(mcap_report["channels"][0]["topic"], "/camera")
            self.assertEqual(bag_report["status"], "ok")
            self.assertEqual(bag_report["total_records"], 7)
            self.assertEqual(bag_report["connections"][0]["topic"], "/pose")


if __name__ == "__main__":
    unittest.main()
