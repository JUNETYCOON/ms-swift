from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from universal_dataset.validation import SCHEMA_PATH, validate_record


def base_record() -> dict:
    return {
        "format": "s1-udf",
        "schema_version": "1.0.0",
        "id": "extended:1",
        "group_id": "media:1",
        "provenance": {
            "dataset": "extended-schema-fixture",
            "source_record_id": "extended:1",
            "conversion": {"tool": "test-fixture", "version": "1.0.0"},
        },
        "assets": [
            {
                "id": "image:0",
                "kind": "image",
                "uri": "https://example.test/image.png",
                "media": {"width": 100, "height": 80},
            },
            {
                "id": "mask:0",
                "kind": "mask",
                "uri": "https://example.test/mask.png",
                "media": {"width": 100, "height": 80},
            },
        ],
    }


def geometry_record(geometry: dict, asset_id: str = "image:0") -> dict:
    record = base_record()
    record["regions"] = [{"id": "region:0", "asset_id": asset_id, "geometry": geometry}]
    return record


class ExtendedRecordSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from jsonschema import Draft202012Validator
        except ImportError as error:
            raise unittest.SkipTest("jsonschema is required for formal schema tests") from error
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        cls.schema_validator = Draft202012Validator(schema)

    def assert_schema_valid(self, record: dict) -> None:
        errors = list(self.schema_validator.iter_errors(record))
        self.assertEqual(errors, [], [error.message for error in errors])

    def assert_schema_invalid(self, record: dict) -> None:
        self.assertTrue(list(self.schema_validator.iter_errors(record)))

    def issue_codes(self, record: dict, **kwargs) -> set[str]:
        return {issue.code for issue in validate_record(record, check_json_schema=False, **kwargs)}

    def test_geometry_contracts_accept_typed_2d_3d_and_external_point_sets(self):
        geometries = [
            {"type": "point2d", "coordinates": [10, 20], "coordinate_space": {"type": "pixel"}},
            {"type": "bbox2d", "format": "xywh", "coordinates": [10, 20, 30, 40], "coordinate_space": {"type": "pixel"}},
            {"type": "polygon2d", "coordinates": [[1, 1], [20, 1], [20, 20]], "coordinate_space": {"type": "pixel"}},
            {"type": "polyline2d", "coordinates": [1, 1, 20, 20], "coordinate_space": {"type": "pixel"}},
            {"type": "keypoints2d", "coordinates": [[1, 2], [3, 4]], "visibility": [1, 0], "coordinate_space": {"type": "pixel"}},
            {"type": "rle", "size": [80, 100], "counts": [8000], "coordinate_space": {"type": "pixel"}},
            {"type": "mask_ref", "asset_ref": "mask:0", "coordinate_space": {"type": "pixel"}},
            {"type": "point3d", "coordinates": [1, 2, 3], "coordinate_space": {"type": "world", "unit": "meter"}},
            {"type": "keypoints3d", "coordinates": [[1, 2, 3], [4, 5, 6]], "visibility": [1, 1], "coordinate_space": {"type": "world"}},
            {
                "type": "bbox3d",
                "center": [1, 2, 3],
                "dimensions": [4, 5, 6],
                "dimension_order": ["length", "width", "height"],
                "rotation_convention": "axis_aligned_right_handed",
                "coordinate_space": {"type": "world", "frame_id": "world"},
            },
            {
                "type": "cuboid3d",
                "center": [1, 2, 3],
                "dimensions": [4, 5, 6],
                "pose": {"quaternion_xyzw": [0, 0, 0, 1]},
                "coordinate_space": {"type": "world"},
            },
            {"type": "pose3d", "pose": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, "coordinate_space": {"type": "world"}},
            {
                "type": "point_set",
                "data_ref": {"uri": "https://example.test/points.parquet", "format": "parquet"},
                "coordinate_space": {"type": "lidar"},
            },
        ]
        for index, geometry in enumerate(geometries):
            with self.subTest(geometry=geometry["type"]):
                record = geometry_record(geometry)
                record["id"] = "geometry:{}".format(index)
                self.assert_schema_valid(record)
                self.assertEqual(validate_record(record, check_json_schema=False), [])

    def test_geometry_audit_counterexamples_are_rejected(self):
        malformed = [
            {"type": "point2d", "coordinates": [1], "coordinate_space": {"type": "pixel"}},
            {"type": "bbox2d", "coordinates": [1, 2, 3], "coordinate_space": {"type": "pixel"}},
            {"type": "polygon2d", "coordinates": [1, 2, 3, 4], "coordinate_space": {"type": "pixel"}},
            {"type": "polyline2d", "coordinates": [1, "bad", 3, 4], "coordinate_space": {"type": "pixel"}},
            {"type": "keypoints3d", "coordinates": [[1, 2]], "coordinate_space": {"type": "world"}},
            {"type": "rle", "size": [80, 100], "counts": [], "coordinate_space": {"type": "pixel"}},
            {"type": "bbox3d", "center": [1, 2, 3], "coordinate_space": {"type": "world"}},
            {"type": "cuboid3d", "center": [1, 2, 3], "dimensions": [1, 1, 1], "coordinate_space": {"type": "world"}},
            {"type": "pose3d", "pose": {}, "coordinate_space": {"type": "world"}},
            {"type": "point_set", "coordinate_space": {"type": "lidar"}},
        ]
        for geometry in malformed:
            with self.subTest(geometry=geometry):
                record = geometry_record(geometry)
                self.assert_schema_invalid(record)
                self.assertTrue(validate_record(record, check_json_schema=False))

        wrong_mask = geometry_record({"type": "mask_ref", "asset_ref": "image:0", "coordinate_space": {"type": "pixel"}})
        self.assertIn("asset_kind", self.issue_codes(wrong_mask))

    def test_coordinate_ranges_and_pixel_bound_source_precedence(self):
        normalized = geometry_record(
            {"type": "bbox2d", "format": "xywh", "coordinates": [0.8, 0.8, 0.4, 0.4], "coordinate_space": {"type": "normalized"}}
        )
        self.assertIn("range", self.issue_codes(normalized))
        percent = geometry_record(
            {"type": "point2d", "coordinates": [101, 20], "coordinate_space": {"type": "percent"}}
        )
        self.assertIn("range", self.issue_codes(percent))
        explicit_bounds = geometry_record(
            {"type": "point2d", "coordinates": [30, 5], "coordinate_space": {"type": "pixel", "image_size": [20, 20]}}
        )
        self.assertIn("bounds", self.issue_codes(explicit_bounds))
        asset_bounds = geometry_record(
            {"type": "point2d", "coordinates": [101, 5], "coordinate_space": {"type": "pixel"}}
        )
        self.assertIn("bounds", self.issue_codes(asset_bounds))

    def test_tensor_shape_count_and_transform_representation(self):
        record = base_record()
        record["annotations"] = [
            {
                "id": "depth:0",
                "type": "depth",
                "target": {"asset_id": "image:0"},
                "tensor": {"dtype": "float32", "shape": [2, 2], "data": [1, 2, 3]},
            }
        ]
        self.assert_schema_valid(record)
        self.assertIn("count", self.issue_codes(record))

        record["annotations"][0]["tensor"]["data"] = [[1, 2], [3, 4]]
        self.assertEqual(validate_record(record, check_json_schema=False), [])

        record["annotations"][0]["tensor"]["data"] = [[1, 2, 3, 4]]
        self.assertIn("shape", self.issue_codes(record))
        record["annotations"][0]["tensor"]["data"] = [[1, 2, 3], [4]]
        self.assertIn("shape", self.issue_codes(record))

        episode = {
            "format": "s1-udf",
            "schema_version": "1.0.0",
            "id": "transform:bad",
            "group_id": "episode:bad",
            "provenance": {
                "dataset": "extended-schema-fixture",
                "source_record_id": "transform:bad",
                "conversion": {"tool": "test-fixture", "version": "1.0.0"},
            },
            "episode": {
                "id": "episode:bad",
                "coordinate_frames": [{"id": "world"}, {"id": "camera", "parent_id": "world", "transform_to_parent": {"metadata": {}}}],
                "steps": [],
            },
        }
        self.assert_schema_invalid(episode)
        self.assertIn("required", self.issue_codes(episode))

    def test_multi_asset_track_time_points_and_frame_bounds(self):
        record = base_record()
        record["assets"] = [
            {"id": "video:0", "kind": "video", "uri": "https://example.test/a.mp4", "media": {"frame_count": 3}},
            {"id": "video:1", "kind": "video", "uri": "https://example.test/b.mp4", "media": {"frame_count": 5}},
        ]
        record["clocks"] = [{"id": "ros", "type": "ros", "unit": "second"}]
        record["regions"] = [
            {"id": "r0", "asset_id": "video:0", "frame_index": 2, "geometry": {"type": "point2d", "coordinates": [0.5, 0.5], "coordinate_space": {"type": "normalized"}}},
            {"id": "r1", "asset_id": "video:1", "frame_index": 4, "geometry": {"type": "point2d", "coordinates": [0.5, 0.5], "coordinate_space": {"type": "normalized"}}},
        ]
        record["tracks"] = [
            {
                "id": "track:0",
                "asset_ids": ["video:0", "video:1"],
                "observations": [
                    {"asset_id": "video:0", "region_id": "r0", "frame_index": 2, "time_point": {"value": 0.1, "unit": "second", "clock_id": "ros"}},
                    {"asset_id": "video:1", "region_id": "r1", "frame_index": 4, "time_point": {"value": 0.2, "unit": "second", "clock_id": "ros"}},
                ],
            }
        ]
        self.assert_schema_valid(record)
        self.assertEqual(validate_record(record, check_json_schema=False), [])

        record["tracks"][0]["observations"][1]["frame_index"] = 5
        self.assertIn("bounds", self.issue_codes(record))
        record["tracks"][0]["observations"] = []
        self.assert_schema_invalid(record)
        self.assertIn("required", self.issue_codes(record))

    def test_document_anchors_and_multimodal_choices(self):
        record = base_record()
        record["regions"] = [
            {"id": "word:0", "asset_id": "image:0", "geometry": {"type": "bbox2d", "coordinates": [1, 2, 20, 10], "coordinate_space": {"type": "pixel"}}}
        ]
        record["annotations"] = [
            {
                "id": "ocr:0",
                "type": "ocr",
                "text": "TOTAL",
                "document_anchors": [{"asset_id": "image:0", "region_id": "word:0", "page_index": 0, "span": {"start": 0, "end": 5, "unit": "character"}}],
            },
            {
                "id": "qa:0",
                "type": "qa",
                "question": [{"type": "asset", "asset_id": "image:0"}, {"type": "text", "text": "Which crop?"}],
                "answers": [{"text": "A"}],
                "answer_mode": "multiple_choice",
                "choices": [
                    {"id": "a", "content": [{"type": "region", "region_id": "word:0"}, {"type": "text", "text": "A"}]},
                    "None",
                ],
                "correct_choice_indices": [0],
            },
        ]
        self.assert_schema_valid(record)
        self.assertEqual(validate_record(record, check_json_schema=False), [])

        record["annotations"][0]["document_anchors"][0]["region_id"] = "missing"
        self.assertIn("missing_ref", self.issue_codes(record))
        incomplete_ocr = base_record()
        incomplete_ocr["annotations"] = [{"id": "ocr:bad", "type": "ocr", "text": "text"}]
        self.assert_schema_invalid(incomplete_ocr)
        self.assertIn("annotation_contract", self.issue_codes(incomplete_ocr))

    def test_embodied_stream_clock_dynamic_transform_and_external_steps_ref(self):
        record = base_record()
        record["clocks"] = [{"id": "ros", "type": "ros", "unit": "nanosecond"}]
        record["streams"] = [
            {
                "id": "camera-stream",
                "protocol": "mcap",
                "topic": "/camera/image",
                "schema": "sensor_msgs/msg/Image",
                "qos": {"reliability": "best_effort"},
                "clock_id": "ros",
                "records_ref": {"uri": "https://example.test/demo.mcap", "format": "mcap"},
            }
        ]
        record["episode"] = {
            "id": "episode:0",
            "clock_id": "ros",
            "coordinate_frames": [{"id": "world"}, {"id": "camera", "parent_id": "world"}],
            "feature_specs": [{"name": "joint_position", "kind": "tensor", "source": "observation", "dtype": "float32", "shape": [7]}],
            "dynamic_transforms": [
                {
                    "parent_frame_id": "world",
                    "child_frame_id": "camera",
                    "transform": {"translation": [0, 0, 1], "quaternion_xyzw": [0, 0, 0, 1]},
                    "time_point": {"value": 0, "unit": "nanosecond", "clock_id": "ros"},
                }
            ],
            "steps": [
                {
                    "index": 0,
                    "time_point": {"value": 0, "unit": "nanosecond", "clock_id": "ros", "sample": 0},
                    "time_span": {"start": 0, "end": 1, "unit": "step", "clock_id": "ros", "sample": 0},
                    "observations": [
                        {
                            "name": "camera",
                            "asset_id": "image:0",
                            "frame_index": 0,
                            "time_point": {"value": 0, "unit": "nanosecond", "clock_id": "ros"},
                        }
                    ],
                    "actions": [
                        {
                            "name": "move",
                            "data_ref": {"uri": "https://example.test/actions.parquet", "format": "parquet", "key": 0},
                            "frame_id": "world",
                            "clock_id": "ros",
                        }
                    ],
                    "is_first": True,
                    "is_last": True,
                    "is_terminal": True,
                }
            ],
            "step_count": 1,
        }
        self.assert_schema_valid(record)
        self.assertEqual(validate_record(record, check_json_schema=False), [])

        external = {
            "format": "s1-udf",
            "schema_version": "1.0.0",
            "id": "external:steps",
            "group_id": "episode:external",
            "provenance": {
                "dataset": "extended-schema-fixture",
                "source_record_id": "external:steps",
                "conversion": {"tool": "test-fixture", "version": "1.0.0"},
            },
            "episode": {
                "id": "episode:external",
                "steps_ref": {"uri": "missing/steps.parquet", "format": "parquet"},
                "step_count": 10,
            },
        }
        self.assert_schema_valid(external)
        with tempfile.TemporaryDirectory() as temporary_dir:
            self.assertIn("missing_data_ref", self.issue_codes(external, check_assets=True, base_dir=Path(temporary_dir)))

    def test_task_and_annotation_minimum_supervision_contracts(self):
        task_record = base_record()
        task_record["task_types"] = ["tracking"]
        self.assertIn("task_contract", self.issue_codes(task_record))

        bad_segmentation = base_record()
        bad_segmentation["annotations"] = [{"id": "seg:0", "type": "panoptic_segmentation"}]
        self.assert_schema_invalid(bad_segmentation)
        self.assertIn("annotation_contract", self.issue_codes(bad_segmentation))

        custom = base_record()
        custom["annotations"] = [{"id": "custom:0", "type": "domain_specific", "value": {"label": 1}}]
        self.assert_schema_valid(custom)
        self.assertEqual(validate_record(custom, check_json_schema=False), [])


if __name__ == "__main__":
    unittest.main()
