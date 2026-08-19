"""Structural and cross-reference validation for S1-UDF records."""

from __future__ import annotations

import math
import os
import hashlib
import json
import re
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit

from .schema_registry import CURRENT_SCHEMA_VERSION, FORMAT_NAME, SCHEMA_REGISTRY


SCHEMA_VERSION = CURRENT_SCHEMA_VERSION
_CURRENT_SCHEMAS = SCHEMA_REGISTRY.get(SCHEMA_VERSION)
SCHEMA_PATH = _CURRENT_SCHEMAS.record_schema
MANIFEST_SCHEMA_PATH = _CURRENT_SCHEMAS.manifest_schema


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    code: str = "invalid"

    def __str__(self) -> str:
        return "{}: {} ({})".format(self.path, self.message, self.code)


class DatasetValidationError(ValueError):
    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = list(issues)
        preview = "; ".join(str(issue) for issue in self.issues[:8])
        if len(self.issues) > 8:
            preview += "; ... {} more".format(len(self.issues) - 8)
        super().__init__(preview)


def _issue(issues: List[ValidationIssue], path: str, message: str, code: str = "invalid") -> None:
    issues.append(ValidationIssue(path=path, message=message, code=code))


@lru_cache(maxsize=1)
def _schema_validator() -> Any:
    try:
        import json
        from jsonschema import Draft202012Validator
    except ImportError as error:
        return error
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_issues(record: Mapping[str, Any]) -> List[ValidationIssue]:
    validator = _schema_validator()
    if isinstance(validator, ImportError):
        return [
            ValidationIssue(
                "$",
                "jsonschema is required for JSON Schema validation; install it or explicitly disable schema checks",
                "jsonschema_unavailable",
            )
        ]
    result = []
    for error in sorted(validator.iter_errors(record), key=lambda item: list(item.absolute_path)):
        location = "$"
        for part in error.absolute_path:
            location += "[{}]".format(part) if isinstance(part, int) else ".{}".format(part)
        result.append(ValidationIssue(location, error.message, "json_schema"))
    return result


def _index_unique(
    values: Any, field: str, issues: List[ValidationIssue]
) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    if values is None:
        return result
    if not isinstance(values, list):
        _issue(issues, "$.{}".format(field), "must be an array", "type")
        return result
    for index, value in enumerate(values):
        path = "$.{}[{}]".format(field, index)
        if not isinstance(value, Mapping):
            _issue(issues, path, "must be an object", "type")
            continue
        identifier = value.get("id")
        if not isinstance(identifier, str) or not identifier:
            _issue(issues, path + ".id", "must be a non-empty string", "required")
            continue
        if identifier in result:
            _issue(issues, path + ".id", "duplicate ID {!r}".format(identifier), "duplicate_id")
            continue
        result[identifier] = value
    return result


def _finite_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, list):
        for item in value:
            for number in _finite_numbers(item):
                yield number


def _nonfinite_paths(value: Any, path: str = "$") -> Iterable[str]:
    if isinstance(value, float) and not math.isfinite(value):
        yield path
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _nonfinite_paths(item, path + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _nonfinite_paths(item, "{}[{}]".format(path, index))


def _check_time_span(value: Any, path: str, issues: List[ValidationIssue]) -> None:
    if not isinstance(value, Mapping):
        return
    start, end = value.get("start"), value.get("end")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (start, end)):
        return
    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        _issue(issues, path, "start and end must be finite", "non_finite")
    elif end < start:
        _issue(issues, path, "end must be greater than or equal to start", "range")


def _check_time_span_asset(
    value: Any,
    path: str,
    assets: Mapping[str, Mapping[str, Any]],
    default_asset_id: Optional[str],
    issues: List[ValidationIssue],
) -> None:
    if not isinstance(value, Mapping):
        return
    asset_id = value.get("reference_asset_id", default_asset_id)
    if asset_id is None:
        return
    asset = assets.get(asset_id)
    if asset is None:
        _issue(issues, path + ".reference_asset_id", "unknown asset {!r}".format(asset_id), "missing_ref")
        return
    start, end, unit = value.get("start"), value.get("end"), value.get("unit")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (start, end)):
        return
    media = asset.get("media")
    if not isinstance(media, Mapping):
        return
    if unit == "second" and isinstance(media.get("duration"), (int, float)):
        if start < 0 or end > media["duration"]:
            _issue(issues, path, "lies outside asset duration {} seconds".format(media["duration"]), "bounds")
    if unit == "frame" and isinstance(media.get("frame_count"), int):
        if start < 0 or end > media["frame_count"]:
            _issue(issues, path, "lies outside asset frame_count {}".format(media["frame_count"]), "bounds")


def _bbox_xyxy(geometry: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    if geometry.get("type") != "bbox2d":
        return None
    values = geometry.get("coordinates")
    if not isinstance(values, list) or len(values) != 4 or any(isinstance(value, bool) for value in values):
        return None
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        return None
    x0, y0, c2, c3 = (float(value) for value in values)
    bbox_format = geometry.get("format", "xyxy")
    if bbox_format == "xyxy":
        return x0, y0, c2, c3
    if bbox_format == "xywh":
        return x0, y0, x0 + c2, y0 + c3
    if bbox_format == "cxcywh":
        return x0 - c2 / 2, y0 - c3 / 2, x0 + c2 / 2, y0 + c3 / 2
    return None


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _coordinate_tuples(value: Any, dimension: int) -> Tuple[List[Tuple[float, ...]], bool]:
    if not isinstance(value, list) or not value:
        return [], False
    if all(_is_finite_number(item) for item in value):
        if len(value) % dimension:
            return [], False
        return [
            tuple(float(value[offset + axis]) for axis in range(dimension))
            for offset in range(0, len(value), dimension)
        ], True
    points: List[Tuple[float, ...]] = []
    for item in value:
        nested, valid = _coordinate_tuples(item, dimension)
        if not valid:
            return [], False
        points.extend(nested)
    return points, bool(points)


def _coordinate_pairs(value: Any) -> List[Tuple[float, float]]:
    points, valid = _coordinate_tuples(value, 2)
    return [(point[0], point[1]) for point in points] if valid else []


def _check_vector(
    value: Any,
    length: int,
    path: str,
    description: str,
    issues: List[ValidationIssue],
    positive: bool = False,
) -> bool:
    if not isinstance(value, list) or len(value) != length or not all(_is_finite_number(item) for item in value):
        _issue(issues, path, "{} requires exactly {} finite numeric values".format(description, length), "shape")
        return False
    if positive and any(float(item) <= 0 for item in value):
        _issue(issues, path, "{} values must be greater than zero".format(description), "range")
        return False
    return True


def _validate_transform(value: Any, path: str, issues: List[ValidationIssue]) -> None:
    if not isinstance(value, Mapping):
        return
    representations = ("translation", "quaternion_xyzw", "quaternion_wxyz", "matrix")
    if not any(field in value for field in representations):
        _issue(issues, path, "transform requires translation, quaternion, or matrix representation", "required")
    if "translation" in value:
        _check_vector(value.get("translation"), 3, path + ".translation", "translation", issues)
    for field in ("quaternion_xyzw", "quaternion_wxyz"):
        if field in value and _check_vector(value.get(field), 4, path + "." + field, field, issues):
            if math.isclose(sum(float(item) ** 2 for item in value[field]), 0.0):
                _issue(issues, path + "." + field, "quaternion must not be the zero quaternion", "range")
    if "quaternion_xyzw" in value and "quaternion_wxyz" in value:
        _issue(issues, path, "use only one quaternion component ordering", "conflict")
    matrix = value.get("matrix")
    if "matrix" in value:
        valid_matrix = (
            isinstance(matrix, list)
            and len(matrix) in (3, 4)
            and all(isinstance(row, list) for row in matrix)
            and bool(matrix)
            and len({len(row) for row in matrix}) == 1
            and len(matrix[0]) in (3, 4)
            and all(_is_finite_number(item) for row in matrix for item in row)
        )
        if not valid_matrix or (len(matrix), len(matrix[0])) not in ((3, 3), (3, 4), (4, 4)):
            _issue(issues, path + ".matrix", "matrix must have shape 3x3, 3x4, or 4x4", "shape")


def _validate_geometry(
    region: Mapping[str, Any],
    path: str,
    asset: Optional[Mapping[str, Any]],
    assets: Mapping[str, Mapping[str, Any]],
    issues: List[ValidationIssue],
) -> None:
    geometry = region.get("geometry")
    if not isinstance(geometry, Mapping):
        return
    geometry_path = path + ".geometry"
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    points2d: List[Tuple[float, float]] = []

    vector_lengths = {"point2d": 2, "bbox2d": 4, "point3d": 3}
    if geometry_type in vector_lengths:
        _check_vector(
            coordinates,
            vector_lengths[geometry_type],
            geometry_path + ".coordinates",
            str(geometry_type),
            issues,
        )
    if geometry_type in ("polygon2d", "polyline2d", "keypoints2d"):
        points, valid = _coordinate_tuples(coordinates, 2)
        points2d = [(point[0], point[1]) for point in points] if valid else []
        minimum = 3 if geometry_type == "polygon2d" else 2 if geometry_type == "polyline2d" else 1
        if not valid or len(points2d) < minimum:
            _issue(
                issues,
                geometry_path + ".coordinates",
                "{} requires at least {} finite numeric coordinate pairs".format(geometry_type, minimum),
                "shape",
            )
        visibility = geometry.get("visibility")
        if geometry_type == "keypoints2d" and isinstance(visibility, list) and len(visibility) != len(points2d):
            _issue(
                issues,
                geometry_path + ".visibility",
                "length must match keypoint count {}".format(len(points2d)),
                "count",
            )
    if geometry_type in ("keypoints3d", "point_set") and "coordinates" in geometry:
        points3d, valid = _coordinate_tuples(coordinates, 3)
        if not valid or not points3d:
            _issue(issues, geometry_path + ".coordinates", "{} requires finite numeric 3D points".format(geometry_type), "shape")
        visibility = geometry.get("visibility")
        if geometry_type == "keypoints3d" and isinstance(visibility, list) and len(visibility) != len(points3d):
            _issue(
                issues,
                geometry_path + ".visibility",
                "length must match keypoint count {}".format(len(points3d)),
                "count",
            )
    if geometry_type == "rle":
        size = geometry.get("size")
        counts = geometry.get("counts")
        valid_size = (
            isinstance(size, list)
            and len(size) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in size)
        )
        valid_counts = (isinstance(counts, str) and bool(counts)) or (
            isinstance(counts, list)
            and bool(counts)
            and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in counts)
        )
        if not valid_size or not valid_counts:
            _issue(issues, geometry_path, "rle requires size=[height,width] and non-empty valid counts", "required")
        elif isinstance(counts, list) and sum(counts) != size[0] * size[1]:
            _issue(
                issues,
                geometry_path + ".counts",
                "uncompressed rle counts sum {} does not match mask area {}".format(sum(counts), size[0] * size[1]),
                "count",
            )
    if geometry_type == "mask_ref":
        asset_ref = geometry.get("asset_ref")
        if not isinstance(asset_ref, str) or not asset_ref:
            _issue(issues, geometry_path + ".asset_ref", "mask_ref requires an asset_ref", "required")
        elif asset_ref in assets and assets[asset_ref].get("kind") != "mask":
            _issue(issues, geometry_path + ".asset_ref", "mask_ref must reference an asset with kind='mask'", "asset_kind")
        elif asset_ref in assets:
            mask_media = assets[asset_ref].get("media")
            source_media = asset.get("media") if isinstance(asset, Mapping) else None
            if isinstance(mask_media, Mapping) and isinstance(source_media, Mapping):
                mask_size = (mask_media.get("width"), mask_media.get("height"))
                source_size = (source_media.get("width"), source_media.get("height"))
                if all(isinstance(item, int) for item in mask_size + source_size) and mask_size != source_size:
                    _issue(issues, geometry_path + ".asset_ref", "mask dimensions do not match source asset", "bounds")
    if geometry_type in ("bbox3d", "cuboid3d"):
        _check_vector(geometry.get("center"), 3, geometry_path + ".center", "3D center", issues)
        _check_vector(
            geometry.get("dimensions"), 3, geometry_path + ".dimensions", "3D dimensions", issues, positive=True
        )
    if geometry_type == "bbox3d":
        dimension_order = geometry.get("dimension_order")
        if not (
            isinstance(dimension_order, list)
            and len(dimension_order) == 3
            and all(isinstance(item, str) and bool(item.strip()) for item in dimension_order)
            and len(set(dimension_order)) == 3
        ):
            _issue(
                issues,
                geometry_path + ".dimension_order",
                "bbox3d requires three unique non-empty dimension axis names",
                "geometry_convention",
            )
        rotation_convention = geometry.get("rotation_convention")
        if not isinstance(rotation_convention, str) or not rotation_convention.strip():
            _issue(
                issues,
                geometry_path + ".rotation_convention",
                "bbox3d requires an explicit rotation convention",
                "geometry_convention",
            )
        coordinate_space = geometry.get("coordinate_space")
        frame_id = coordinate_space.get("frame_id") if isinstance(coordinate_space, Mapping) else None
        if not isinstance(frame_id, str) or not frame_id.strip():
            _issue(
                issues,
                geometry_path + ".coordinate_space.frame_id",
                "bbox3d requires an explicit coordinate frame",
                "geometry_convention",
            )
    if geometry_type in ("cuboid3d", "pose3d"):
        if not isinstance(geometry.get("pose"), Mapping):
            _issue(issues, geometry_path + ".pose", "{} requires a typed pose transform".format(geometry_type), "required")
        else:
            _validate_transform(geometry["pose"], geometry_path + ".pose", issues)
    if geometry_type == "point_set" and not any(
        field in geometry for field in ("coordinates", "tensor", "asset_ref", "data_ref")
    ):
        _issue(issues, geometry_path, "point_set requires coordinates, tensor, asset_ref, or data_ref", "required")
    if geometry_type == "point_set" and isinstance(geometry.get("asset_ref"), str):
        point_asset = assets.get(geometry["asset_ref"])
        if isinstance(point_asset, Mapping) and point_asset.get("kind") not in ("point_cloud", "tensor"):
            _issue(
                issues,
                geometry_path + ".asset_ref",
                "point_set asset_ref must reference a point_cloud or tensor asset",
                "asset_kind",
            )

    bbox = _bbox_xyxy(geometry)
    if geometry_type == "bbox2d" and bbox is None:
        _issue(issues, geometry_path, "bbox2d format must be xyxy, xywh, or cxcywh with four numbers", "bbox_format")
    elif bbox is not None and (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
        _issue(issues, geometry_path + ".coordinates", "bbox has negative width or height", "range")
    if geometry_type == "point2d" and _check_vector(coordinates, 2, geometry_path + ".coordinates", "point2d", []):
        points2d = [(float(coordinates[0]), float(coordinates[1]))]
    elif geometry_type in ("polygon2d", "polyline2d", "keypoints2d") and not points2d:
        points2d = _coordinate_pairs(coordinates)

    coordinate_space = geometry.get("coordinate_space")
    if not isinstance(coordinate_space, Mapping):
        return
    space_type = coordinate_space.get("type")
    if bbox is not None:
        bounded_numbers: Sequence[float] = bbox
    elif points2d:
        bounded_numbers = [number for point in points2d for number in point]
    elif geometry_type in ("bbox3d", "cuboid3d") and all(
        isinstance(geometry.get(field), list) and len(geometry[field]) == 3 for field in ("center", "dimensions")
    ):
        bounded_numbers = [
            bound
            for center, dimension in zip(geometry["center"], geometry["dimensions"])
            if _is_finite_number(center) and _is_finite_number(dimension)
            for bound in (float(center) - float(dimension) / 2, float(center) + float(dimension) / 2)
        ]
    else:
        bounded_numbers = list(_finite_numbers(coordinates))
    if space_type in ("normalized", "percent"):
        upper = 1.0 if space_type == "normalized" else 100.0
        if any(number < 0 or number > upper for number in bounded_numbers):
            _issue(
                issues,
                geometry_path + ".coordinates",
                "{} coordinates must be in [0, {}]".format(space_type, int(upper)),
                "range",
            )
    if space_type != "pixel" or geometry_type not in (
        "point2d", "bbox2d", "polygon2d", "polyline2d", "keypoints2d", "rle", "mask_ref"
    ):
        return
    image_size = coordinate_space.get("image_size")
    width: Any = image_size[0] if isinstance(image_size, list) and len(image_size) == 2 else None
    height: Any = image_size[1] if isinstance(image_size, list) and len(image_size) == 2 else None
    has_valid_size = (
        isinstance(width, int)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
    )
    if not has_valid_size:
        media = asset.get("media") if isinstance(asset, Mapping) else None
        width = media.get("width") if isinstance(media, Mapping) else None
        height = media.get("height") if isinstance(media, Mapping) else None
        has_valid_size = (
            isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
            and isinstance(height, int)
            and not isinstance(height, bool)
            and height > 0
        )
    if not has_valid_size:
        _issue(
            issues,
            geometry_path + ".coordinate_space.image_size",
            "pixel geometry requires coordinate_space.image_size or source asset media width/height",
            "missing_image_size",
        )
        return
    out_of_bounds = False
    if bbox is not None:
        out_of_bounds = bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height
    elif points2d:
        out_of_bounds = any(x < 0 or y < 0 or x > width or y > height for x, y in points2d)
    if geometry_type == "rle" and isinstance(geometry.get("size"), list) and len(geometry["size"]) == 2:
        if geometry["size"] != [height, width]:
            _issue(issues, geometry_path + ".size", "rle size does not match pixel bounds {}x{}".format(width, height), "bounds")
    if out_of_bounds:
        _issue(
            issues,
            geometry_path + ".coordinates",
            "pixel geometry lies outside bounds {}x{}".format(width, height),
            "bounds",
        )


def _inline_value_count(value: Any) -> int:
    if isinstance(value, list):
        return sum(_inline_value_count(item) for item in value)
    return 1


def _inline_array_shape(value: Any) -> Optional[List[int]]:
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    child_shapes = [_inline_array_shape(item) for item in value]
    if any(shape is None for shape in child_shapes):
        return None
    first = child_shapes[0]
    if any(shape != first for shape in child_shapes[1:]):
        return None
    return [len(value)] + list(first or [])


def _shape_matches_declared(actual: Sequence[int], declared: Sequence[int]) -> bool:
    if len(actual) > len(declared):
        return False
    for observed, expected in zip(actual, declared):
        if expected != -1 and observed != expected:
            return False
    if len(actual) == len(declared):
        return True
    return bool(actual) and actual[-1] == 0


def _validate_tensor(value: Any, path: str, issues: List[ValidationIssue]) -> None:
    if not isinstance(value, Mapping):
        return
    shape = value.get("shape")
    if not isinstance(value.get("dtype"), str) or not value.get("dtype"):
        _issue(issues, path + ".dtype", "tensor dtype must be a non-empty string", "required")
    valid_shape = isinstance(shape, list) and all(
        isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= -1 for dimension in shape
    )
    if not valid_shape:
        _issue(issues, path + ".shape", "tensor shape must contain integer dimensions >= -1", "shape")
        return
    has_data, has_ref = "data" in value, "ref" in value
    if has_data == has_ref:
        _issue(issues, path, "tensor requires exactly one of inline data or ref", "conflict")
        return
    unknown_dimensions = sum(dimension == -1 for dimension in shape)
    if unknown_dimensions > 1:
        _issue(issues, path + ".shape", "tensor shape may contain at most one inferred (-1) dimension", "shape")
    if not has_data or (isinstance(value.get("data"), str) and value.get("encoding")):
        return
    inline_data = value.get("data")
    actual_count = _inline_value_count(inline_data)
    actual_shape = _inline_array_shape(inline_data)
    if actual_shape is None:
        _issue(issues, path + ".data", "inline tensor data must be a rectangular array", "shape")
    elif not _shape_matches_declared(actual_shape, shape):
        _issue(
            issues,
            path + ".data",
            "inline tensor nesting has shape {}, expected {}".format(actual_shape, shape),
            "shape",
        )
    known_product = math.prod(dimension for dimension in shape if dimension != -1)
    if unknown_dimensions == 0 and actual_count != known_product:
        _issue(
            issues,
            path + ".data",
            "inline tensor has {} values but shape {} requires {}".format(actual_count, shape, known_product),
            "count",
        )
    elif unknown_dimensions == 1 and (known_product == 0 and actual_count != 0 or known_product and actual_count % known_product):
        _issue(
            issues,
            path + ".data",
            "inline tensor value count {} cannot satisfy inferred shape {}".format(actual_count, shape),
            "count",
        )


def _iter_tensors(value: Any, path: str = "$") -> Iterable[Tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if "dtype" in value and "shape" in value and ("data" in value or "ref" in value):
            yield path, value
            return
        for key, item in value.items():
            yield from _iter_tensors(item, path + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, "{}[{}]".format(path, index))


def _check_time_point(
    value: Any,
    path: str,
    assets: Mapping[str, Mapping[str, Any]],
    clocks: Mapping[str, Mapping[str, Any]],
    default_asset_id: Optional[str],
    issues: List[ValidationIssue],
) -> None:
    if not isinstance(value, Mapping):
        return
    point_value, unit = value.get("value"), value.get("unit")
    if not _is_finite_number(point_value):
        _issue(issues, path + ".value", "time point value must be finite", "non_finite")
        return
    if unit in ("frame", "step", "sample", "pts") and float(point_value) != int(point_value):
        _issue(issues, path + ".value", "{} time points require an integer value".format(unit), "type")
    if unit in ("frame", "step", "sample") and point_value < 0:
        _issue(issues, path + ".value", "{} time points must be non-negative".format(unit), "range")
    asset_id = value.get("reference_asset_id", default_asset_id)
    if value.get("reference_asset_id") is not None and value.get("reference_asset_id") not in assets:
        _issue(issues, path + ".reference_asset_id", "unknown asset reference", "missing_ref")
    if value.get("clock_id") is not None and value.get("clock_id") not in clocks:
        _issue(issues, path + ".clock_id", "unknown clock reference", "missing_ref")
    asset = assets.get(asset_id) if isinstance(asset_id, str) else None
    media = asset.get("media") if isinstance(asset, Mapping) else None
    if unit == "frame" and isinstance(media, Mapping) and isinstance(media.get("frame_count"), int):
        if point_value < 0 or point_value >= media["frame_count"]:
            _issue(issues, path + ".value", "lies outside asset frame_count {}".format(media["frame_count"]), "bounds")
    if unit == "second" and isinstance(media, Mapping) and isinstance(media.get("duration"), (int, float)):
        if point_value < 0 or point_value > media["duration"]:
            _issue(issues, path + ".value", "lies outside asset duration {} seconds".format(media["duration"]), "bounds")


def _check_clock_ref(
    value: Any, path: str, clocks: Mapping[str, Mapping[str, Any]], issues: List[ValidationIssue]
) -> None:
    if isinstance(value, Mapping) and value.get("clock_id") is not None and value.get("clock_id") not in clocks:
        _issue(issues, path + ".clock_id", "unknown clock reference", "missing_ref")


def _check_node_ref(
    value: Any,
    path: str,
    indexes: Mapping[str, Mapping[str, Any]],
    issues: List[ValidationIssue],
) -> None:
    if not isinstance(value, Mapping):
        return
    mapping = {"asset_id": "assets", "entity_id": "entities", "region_id": "regions", "track_id": "tracks"}
    for key, collection in mapping.items():
        if key in value and value[key] not in indexes[collection]:
            _issue(issues, path + "." + key, "unknown {} {!r}".format(key, value[key]), "missing_ref")


def _check_document_anchor(
    value: Any,
    path: str,
    indexes: Mapping[str, Mapping[str, Any]],
    issues: List[ValidationIssue],
) -> None:
    if not isinstance(value, Mapping):
        return
    asset_id = value.get("asset_id")
    region_id = value.get("region_id")
    asset = indexes["assets"].get(asset_id) if isinstance(asset_id, str) else None
    region = indexes["regions"].get(region_id) if isinstance(region_id, str) else None
    if asset_id is not None and asset is None:
        _issue(issues, path + ".asset_id", "unknown asset reference", "missing_ref")
    if region_id is not None and region is None:
        _issue(issues, path + ".region_id", "unknown region reference", "missing_ref")
    if isinstance(region, Mapping):
        region_asset_id = region.get("asset_id")
        if asset_id is not None and asset_id != region_asset_id:
            _issue(issues, path, "document anchor asset does not match referenced region asset", "asset_mismatch")
        if asset is None:
            asset = indexes["assets"].get(region_asset_id)
    if asset_id is None and region_id is None:
        _issue(issues, path, "document anchor requires asset_id or region_id", "required")
    span = value.get("span")
    if isinstance(span, Mapping):
        start, end = span.get("start"), span.get("end")
        if isinstance(start, int) and isinstance(end, int) and end < start:
            _issue(issues, path + ".span", "document span end must be >= start", "range")
    if isinstance(value.get("geometry"), Mapping):
        _validate_geometry({"geometry": value["geometry"]}, path, asset, indexes["assets"], issues)


def _content_has_signal(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if part_type == "text":
            if isinstance(part.get("text"), str) and bool(part["text"].strip()):
                return True
        elif part_type == "tool_result":
            if part.get("result") not in (None, "", [], {}):
                return True
        elif isinstance(part_type, str) and part_type:
            return True
    return False


def _check_content(
    content: Any,
    path: str,
    indexes: Mapping[str, Mapping[str, Any]],
    issues: List[ValidationIssue],
) -> None:
    if not isinstance(content, list):
        return
    for index, part in enumerate(content):
        part_path = "{}[{}]".format(path, index)
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        collection = {"asset": "assets", "entity": "entities", "region": "regions"}.get(part_type)
        key = {"asset": "asset_id", "entity": "entity_id", "region": "region_id"}.get(part_type)
        if collection and key and part.get(key) not in indexes[collection]:
            _issue(issues, part_path + "." + key, "unknown reference {!r}".format(part.get(key)), "missing_ref")
        if part_type == "document_anchor":
            _check_document_anchor(part.get("anchor"), part_path + ".anchor", indexes, issues)
        if "time_span" in part:
            _check_time_span(part["time_span"], part_path + ".time_span", issues)
            _check_time_span_asset(
                part["time_span"],
                part_path + ".time_span",
                indexes["assets"],
                part.get("asset_id") if part_type == "asset" else None,
                issues,
            )
            _check_clock_ref(part["time_span"], part_path + ".time_span", indexes.get("clocks", {}), issues)
        if part_type == "asset" and isinstance(part.get("frame_index"), int):
            asset = indexes["assets"].get(part.get("asset_id"))
            media = asset.get("media") if isinstance(asset, Mapping) else None
            frame_count = media.get("frame_count") if isinstance(media, Mapping) else None
            if isinstance(frame_count, int) and part["frame_index"] >= frame_count:
                _issue(issues, part_path + ".frame_index", "lies outside asset frame_count {}".format(frame_count), "bounds")


def _local_asset_path(uri: str, base_dir: Optional[Path]) -> Optional[Path]:
    parsed = urlsplit(uri)
    if parsed.scheme and parsed.scheme.lower() != "file":
        return None
    if parsed.scheme.lower() == "file":
        return Path(parsed.path)
    path = Path(uri).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _check_external_data_ref(
    value: Any,
    path: str,
    check_assets: bool,
    base_dir: Optional[Path],
    issues: List[ValidationIssue],
) -> None:
    if not check_assets or not isinstance(value, Mapping) or not isinstance(value.get("uri"), str):
        return
    uri = value["uri"]
    parsed = urlsplit(uri)
    if parsed.scheme and parsed.scheme.lower() != "file":
        return
    raw_path = parsed.path if parsed.scheme.lower() == "file" or parsed.fragment or parsed.query else uri
    local_path = Path(raw_path).expanduser()
    if not local_path.is_absolute() and base_dir is not None:
        local_path = base_dir / local_path
    local_path = local_path.resolve()
    if not local_path.is_file():
        _issue(issues, path + ".uri", "external data reference does not exist: {}".format(local_path), "missing_data_ref")


def _annotation_has_region_target(annotation: Mapping[str, Any]) -> bool:
    target = annotation.get("target")
    if isinstance(target, Mapping) and "region_id" in target:
        return True
    return any(isinstance(item, Mapping) and "region_id" in item for item in annotation.get("targets") or [])


def _validate_annotation_contract(
    annotation: Mapping[str, Any],
    path: str,
    indexes: Mapping[str, Mapping[str, Any]],
    issues: List[ValidationIssue],
) -> None:
    annotation_type = annotation.get("type")
    if annotation_type in ("caption", "captioning", "dense_caption"):
        if not (
            isinstance(annotation.get("text"), str) and bool(annotation["text"].strip())
        ) and not _content_has_signal(annotation.get("content")):
            _issue(issues, path, "{} annotation requires text or structured content".format(annotation_type), "annotation_contract")
    if annotation_type in ("qa", "vqa", "document_qa"):
        question = annotation.get("question")
        if (
            question is None
            or (isinstance(question, str) and not question.strip())
            or (isinstance(question, list) and not _content_has_signal(question))
        ):
            _issue(issues, path + ".question", "qa annotation requires a non-empty question", "annotation_contract")
        if not annotation.get("answers"):
            _issue(issues, path + ".answers", "qa annotation requires at least one answer", "annotation_contract")
        if annotation.get("answer_mode") == "multiple_choice":
            if not annotation.get("choices"):
                _issue(issues, path + ".choices", "multiple-choice QA requires choices", "annotation_contract")
            if not annotation.get("correct_choice_indices"):
                _issue(
                    issues,
                    path + ".correct_choice_indices",
                    "multiple-choice QA requires at least one correct choice index",
                    "annotation_contract",
                )
    if annotation_type in ("classification", "multi_label_classification"):
        if not annotation.get("label_ids") and not annotation.get("entity_ids") and "value" not in annotation:
            _issue(issues, path, "classification annotation requires labels, entities, or value", "annotation_contract")
    if annotation_type in ("detection", "object_detection", "grounding", "referring_expression", "3d_detection"):
        if not annotation.get("region_ids") and not _annotation_has_region_target(annotation):
            _issue(issues, path, "{} annotation requires a region target".format(annotation_type), "annotation_contract")
    if annotation_type in (
        "segmentation", "semantic_segmentation", "instance_segmentation", "panoptic_segmentation", "keypoints", "keypoint_detection", "pose", "pose_estimation"
    ):
        if not annotation.get("region_ids") and not _annotation_has_region_target(annotation):
            _issue(issues, path, "{} annotation requires region supervision".format(annotation_type), "annotation_contract")
    if annotation_type in ("tracking", "track", "object_tracking") and not annotation.get("track_ids"):
        _issue(issues, path + ".track_ids", "tracking annotation requires track_ids", "annotation_contract")
    if annotation_type in ("relation", "scene_graph") and not annotation.get("relation_ids"):
        _issue(issues, path + ".relation_ids", "relation annotation requires relation_ids", "annotation_contract")
    if annotation_type in ("ocr", "document_ocr"):
        has_output = (isinstance(annotation.get("text"), str) and bool(annotation.get("text"))) or bool(
            annotation.get("content")
        ) or "value" in annotation
        has_anchor = bool(annotation.get("document_anchors")) or annotation.get("target") is not None or bool(
            annotation.get("region_ids")
        )
        if not has_output:
            _issue(issues, path, "OCR annotation requires text, structured content, or value", "annotation_contract")
        if not has_anchor:
            _issue(issues, path, "OCR annotation requires a document/media anchor", "annotation_contract")
    if annotation_type in ("depth", "depth_estimation", "optical_flow"):
        if not any(field in annotation for field in ("value", "tensor", "data_ref")):
            target = annotation.get("target")
            target_asset = indexes["assets"].get(target.get("asset_id")) if isinstance(target, Mapping) else None
            if not isinstance(target_asset, Mapping) or target_asset.get("kind") not in ("depth", "tensor"):
                _issue(issues, path, "{} annotation requires value, tensor, data_ref, or a typed target asset".format(annotation_type), "annotation_contract")


def _validate_task_contracts(
    record: Mapping[str, Any], indexes: Mapping[str, Mapping[str, Any]], issues: List[ValidationIssue]
) -> None:
    task_types = record.get("task_types")
    if task_types is None:
        return
    if not isinstance(task_types, list) or not task_types:
        _issue(issues, "$.task_types", "task_types must be a non-empty array when present", "task_contract")
        return
    annotation_types = {
        annotation.get("type") for annotation in indexes["annotations"].values() if isinstance(annotation.get("type"), str)
    }
    geometry_types = {
        region.get("geometry", {}).get("type")
        for region in indexes["regions"].values()
        if isinstance(region.get("geometry"), Mapping)
    }
    episode = record.get("episode")
    episode_steps = episode.get("steps") if isinstance(episode, Mapping) else None
    has_control = isinstance(episode, Mapping) and (
        "steps_ref" in episode
        or any(isinstance(step, Mapping) and bool(step.get("actions")) for step in (episode_steps or []))
    )
    asset_kinds = {asset.get("kind") for asset in indexes["assets"].values()}
    checks = {
        "qa": bool(annotation_types & {"qa", "vqa", "document_qa"}) or bool(indexes["conversations"]),
        "vqa": bool(annotation_types & {"qa", "vqa"}) or bool(indexes["conversations"]),
        "document_qa": bool(annotation_types & {"qa", "document_qa"}) or bool(indexes["conversations"]),
        "caption": bool(annotation_types & {"caption", "captioning", "dense_caption"}) or bool(indexes["conversations"]),
        "captioning": bool(annotation_types & {"caption", "captioning", "dense_caption"}) or bool(indexes["conversations"]),
        "image_captioning": bool(annotation_types & {"caption", "captioning", "dense_caption"}) or bool(indexes["conversations"]),
        "video_captioning": bool(annotation_types & {"caption", "captioning", "dense_caption"}) or bool(indexes["conversations"]),
        "dialogue": bool(indexes["conversations"]),
        "classification": bool(annotation_types & {"classification", "multi_label_classification"}),
        "object_detection": bool(indexes["regions"]),
        "detection": bool(indexes["regions"]),
        "grounding": bool(indexes["regions"]),
        "segmentation": bool(geometry_types & {"polygon2d", "rle", "mask_ref"}),
        "semantic_segmentation": bool(geometry_types & {"polygon2d", "rle", "mask_ref"}),
        "instance_segmentation": bool(geometry_types & {"polygon2d", "rle", "mask_ref"}),
        "panoptic_segmentation": bool(geometry_types & {"polygon2d", "rle", "mask_ref"}),
        "tracking": bool(indexes["tracks"]),
        "object_tracking": bool(indexes["tracks"]),
        "video_tracking": bool(indexes["tracks"]),
        "keypoints": bool(geometry_types & {"keypoints2d", "keypoints3d"}),
        "keypoint_detection": bool(geometry_types & {"keypoints2d", "keypoints3d"}),
        "pose_estimation": bool(geometry_types & {"keypoints2d", "keypoints3d", "pose3d"}),
        "3d_detection": bool(geometry_types & {"bbox3d", "cuboid3d"}),
        "ocr": bool(annotation_types & {"ocr", "document_ocr"}),
        "document_ocr": bool(annotation_types & {"ocr", "document_ocr"}),
        "depth_estimation": "depth" in asset_kinds or bool(annotation_types & {"depth", "depth_estimation"}),
        "optical_flow": bool(annotation_types & {"optical_flow"}),
        "embodied_trajectory": isinstance(record.get("episode"), Mapping),
        "language_conditioned_control": has_control,
        "control": has_control,
    }
    for index, task_type in enumerate(task_types):
        if not isinstance(task_type, str) or not task_type:
            _issue(issues, "$.task_types[{}]".format(index), "task type must be a non-empty string", "task_contract")
        elif task_type in checks and not checks[task_type]:
            _issue(
                issues,
                "$.task_types[{}]".format(index),
                "task {!r} lacks its minimum supervision carrier".format(task_type),
                "task_contract",
            )


def validate_record(
    record: Mapping[str, Any],
    check_json_schema: bool = True,
    check_assets: bool = False,
    base_dir: Optional[Path] = None,
) -> List[ValidationIssue]:
    """Return every detectable issue in one S1-UDF record."""

    issues: List[ValidationIssue] = []
    if not isinstance(record, Mapping):
        return [ValidationIssue("$", "record must be an object", "type")]
    for nonfinite_path in _nonfinite_paths(record):
        _issue(issues, nonfinite_path, "contains NaN or infinity", "non_finite")
    if check_json_schema:
        issues.extend(_schema_issues(record))
    if record.get("format") != FORMAT_NAME:
        _issue(issues, "$.format", "must equal {!r}".format(FORMAT_NAME), "version")
    if record.get("schema_version") != SCHEMA_VERSION:
        _issue(issues, "$.schema_version", "must equal {!r}".format(SCHEMA_VERSION), "version")
    for field in ("id", "group_id"):
        if not isinstance(record.get(field), str) or not record.get(field):
            _issue(issues, "$.{}".format(field), "must be a non-empty string", "required")

    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        _issue(issues, "$.provenance", "record requires provenance", "provenance")
    else:
        if not isinstance(provenance.get("dataset"), str) or not provenance["dataset"].strip():
            _issue(issues, "$.provenance.dataset", "must be a non-empty source dataset name", "provenance")
        source_record_id = provenance.get("source_record_id")
        if (
            isinstance(source_record_id, bool)
            or not isinstance(source_record_id, (str, int))
            or (isinstance(source_record_id, str) and not source_record_id.strip())
        ):
            _issue(
                issues,
                "$.provenance.source_record_id",
                "must be a non-empty string or integer source record ID",
                "provenance",
            )
        conversion = provenance.get("conversion")
        if not isinstance(conversion, Mapping):
            _issue(
                issues,
                "$.provenance.conversion",
                "must identify the conversion tool and version",
                "provenance",
            )
        else:
            for field in ("tool", "version"):
                if not isinstance(conversion.get(field), str) or not conversion[field].strip():
                    _issue(
                        issues,
                        "$.provenance.conversion.{}".format(field),
                        "must be a non-empty string",
                        "provenance",
                    )

    carrier_fields = ("assets", "entities", "regions", "tracks", "conversations", "annotations", "streams")
    has_carrier = any(isinstance(record.get(field), list) and bool(record.get(field)) for field in carrier_fields)
    has_carrier = has_carrier or isinstance(record.get("episode"), Mapping)
    if not has_carrier:
        _issue(
            issues,
            "$",
            "record must contain at least one non-empty data carrier collection or episode",
            "empty_record",
        )

    indexes = {
        "assets": _index_unique(record.get("assets"), "assets", issues),
        "entities": _index_unique(record.get("entities"), "entities", issues),
        "regions": _index_unique(record.get("regions"), "regions", issues),
        "relations": _index_unique(record.get("relations"), "relations", issues),
        "tracks": _index_unique(record.get("tracks"), "tracks", issues),
        "conversations": _index_unique(record.get("conversations"), "conversations", issues),
        "annotations": _index_unique(record.get("annotations"), "annotations", issues),
        "clocks": _index_unique(record.get("clocks"), "clocks", issues),
        "streams": _index_unique(record.get("streams"), "streams", issues),
    }

    for tensor_path, tensor in _iter_tensors(record):
        _validate_tensor(tensor, tensor_path, issues)
        _check_external_data_ref(tensor.get("ref"), tensor_path + ".ref", check_assets, base_dir, issues)
    _validate_task_contracts(record, indexes, issues)

    episode = record.get("episode")
    episode_frame_ids: Set[str] = set()
    episode_sensor_ids: Set[str] = set()
    if isinstance(episode, Mapping):
        episode_frame_ids = {
            frame.get("id")
            for frame in episode.get("coordinate_frames") or []
            if isinstance(frame, Mapping) and isinstance(frame.get("id"), str)
        }
        episode_sensor_ids = {
            sensor.get("id")
            for sensor in episode.get("sensors") or []
            if isinstance(sensor, Mapping) and isinstance(sensor.get("id"), str)
        }

    for stream_index, stream in enumerate(record.get("streams") or []):
        if not isinstance(stream, Mapping):
            continue
        path = "$.streams[{}]".format(stream_index)
        if stream.get("asset_id") is not None and stream.get("asset_id") not in indexes["assets"]:
            _issue(issues, path + ".asset_id", "unknown asset reference", "missing_ref")
        if stream.get("sensor_id") is not None and stream.get("sensor_id") not in episode_sensor_ids:
            _issue(issues, path + ".sensor_id", "unknown episode sensor reference", "missing_ref")
        if stream.get("clock_id") is not None and stream.get("clock_id") not in indexes["clocks"]:
            _issue(issues, path + ".clock_id", "unknown clock reference", "missing_ref")
        if stream.get("protocol") in ("ros1", "ros2"):
            for required_field in ("topic", "schema"):
                if not stream.get(required_field):
                    _issue(issues, path + "." + required_field, "ROS stream requires {}".format(required_field), "required")
        if "records_ref" in stream:
            _check_external_data_ref(stream["records_ref"], path + ".records_ref", check_assets, base_dir, issues)
        if "time_span" in stream:
            _check_time_span(stream["time_span"], path + ".time_span", issues)
            _check_time_span_asset(
                stream["time_span"], path + ".time_span", indexes["assets"], stream.get("asset_id"), issues
            )
            _check_clock_ref(stream["time_span"], path + ".time_span", indexes["clocks"], issues)

    for asset_index, asset in enumerate(record.get("assets") or []):
        if not isinstance(asset, Mapping):
            continue
        path = "$.assets[{}]".format(asset_index)
        source_identities = asset.get("source_identities")
        if source_identities is not None:
            if not isinstance(source_identities, list) or not source_identities:
                _issue(
                    issues,
                    path + ".source_identities",
                    "source_identities must be a non-empty array",
                    "asset_identity",
                )
            else:
                seen_source_identities: Set[Any] = set()
                for identity_index, identity in enumerate(source_identities):
                    identity_path = "{}.source_identities[{}]".format(path, identity_index)
                    if not isinstance(identity, Mapping):
                        _issue(issues, identity_path, "source identity must be an object", "asset_identity")
                        continue
                    namespace = identity.get("namespace")
                    value = identity.get("value")
                    if not isinstance(namespace, str) or not namespace.strip():
                        _issue(
                            issues,
                            identity_path + ".namespace",
                            "source identity namespace must not be empty",
                            "asset_identity",
                        )
                        continue
                    if isinstance(value, bool) or not isinstance(value, (str, int)):
                        _issue(
                            issues,
                            identity_path + ".value",
                            "source identity value must be a string or integer",
                            "asset_identity",
                        )
                        continue
                    if isinstance(value, str) and not value.strip():
                        _issue(
                            issues,
                            identity_path + ".value",
                            "source identity value must not be empty",
                            "asset_identity",
                        )
                        continue
                    if identity.get("verified") is not True:
                        _issue(
                            issues,
                            identity_path + ".verified",
                            "source identities must be explicitly verified=true",
                            "asset_identity",
                        )
                    canonical_identity = (
                        namespace.strip(),
                        "integer" if isinstance(value, int) else "string",
                        value,
                    )
                    if canonical_identity in seen_source_identities:
                        _issue(
                            issues,
                            identity_path,
                            "duplicate canonical source identity",
                            "duplicate_id",
                        )
                    seen_source_identities.add(canonical_identity)
        if "time_span" in asset:
            _check_time_span(asset["time_span"], path + ".time_span", issues)
            _check_time_span_asset(
                asset["time_span"], path + ".time_span", indexes["assets"], asset.get("id"), issues
            )
            _check_clock_ref(asset["time_span"], path + ".time_span", indexes["clocks"], issues)
        if isinstance(episode, Mapping) and asset.get("sensor_id") is not None and asset.get("sensor_id") not in episode_sensor_ids:
            _issue(issues, path + ".sensor_id", "unknown episode sensor {!r}".format(asset.get("sensor_id")), "missing_ref")
        if isinstance(episode, Mapping) and asset.get("frame_id") is not None and asset.get("frame_id") not in episode_frame_ids:
            _issue(issues, path + ".frame_id", "unknown coordinate frame {!r}".format(asset.get("frame_id")), "missing_ref")
        uri = asset.get("uri")
        source_ref = asset.get("source_ref")
        has_uri = isinstance(uri, str) and bool(uri.strip())
        has_source_ref = (
            isinstance(source_ref, Mapping)
            and isinstance(source_ref.get("uri"), str)
            and bool(source_ref["uri"].strip())
        )
        if uri is not None and not has_uri:
            _issue(issues, path + ".uri", "asset uri must be a non-empty string", "asset_location")
        if not has_uri and not has_source_ref:
            _issue(
                issues,
                path,
                "asset requires a non-empty uri or external source_ref",
                "asset_location",
            )
        if source_ref is not None:
            if not isinstance(source_ref, Mapping):
                _issue(issues, path + ".source_ref", "must be an external data reference", "asset_location")
            elif not has_source_ref:
                _issue(
                    issues,
                    path + ".source_ref.uri",
                    "external source_ref requires a non-empty uri",
                    "asset_location",
                )
            else:
                _check_external_data_ref(source_ref, path + ".source_ref", check_assets, base_dir, issues)
        if check_assets and has_uri:
            local_path = _local_asset_path(uri, base_dir)
            if local_path is not None and not local_path.is_file():
                _issue(issues, path + ".uri", "local asset does not exist: {}".format(local_path), "missing_asset")

    for region_index, region in enumerate(record.get("regions") or []):
        if not isinstance(region, Mapping):
            continue
        path = "$.regions[{}]".format(region_index)
        asset_id = region.get("asset_id")
        asset = indexes["assets"].get(asset_id)
        if asset is None:
            _issue(issues, path + ".asset_id", "unknown asset {!r}".format(asset_id), "missing_ref")
        entity_id = region.get("entity_id")
        if entity_id is not None and entity_id not in indexes["entities"]:
            _issue(issues, path + ".entity_id", "unknown entity {!r}".format(entity_id), "missing_ref")
        if "time_span" in region:
            _check_time_span(region["time_span"], path + ".time_span", issues)
            _check_time_span_asset(
                region["time_span"], path + ".time_span", indexes["assets"], asset_id, issues
            )
            _check_clock_ref(region["time_span"], path + ".time_span", indexes["clocks"], issues)
        geometry = region.get("geometry")
        if isinstance(geometry, Mapping):
            asset_ref = geometry.get("asset_ref")
            if asset_ref is not None and asset_ref not in indexes["assets"]:
                _issue(issues, path + ".geometry.asset_ref", "unknown asset {!r}".format(asset_ref), "missing_ref")
            if "data_ref" in geometry:
                _check_external_data_ref(
                    geometry["data_ref"], path + ".geometry.data_ref", check_assets, base_dir, issues
                )
            coordinate_space = geometry.get("coordinate_space")
            if isinstance(coordinate_space, Mapping):
                frame_id = coordinate_space.get("frame_id")
                if isinstance(episode, Mapping) and frame_id is not None and frame_id not in episode_frame_ids:
                    _issue(
                        issues,
                        path + ".geometry.coordinate_space.frame_id",
                        "unknown coordinate frame {!r}".format(frame_id),
                        "missing_ref",
                    )
        frame_index = region.get("frame_index")
        media = asset.get("media") if isinstance(asset, Mapping) else None
        frame_count = media.get("frame_count") if isinstance(media, Mapping) else None
        if isinstance(frame_index, int) and isinstance(frame_count, int) and frame_index >= frame_count:
            _issue(issues, path + ".frame_index", "lies outside asset frame_count {}".format(frame_count), "bounds")
        _validate_geometry(region, path, asset, indexes["assets"], issues)

    for relation_index, relation in enumerate(record.get("relations") or []):
        if not isinstance(relation, Mapping):
            continue
        path = "$.relations[{}]".format(relation_index)
        _check_node_ref(relation.get("subject"), path + ".subject", indexes, issues)
        _check_node_ref(relation.get("object"), path + ".object", indexes, issues)
        if "time_span" in relation:
            _check_time_span(relation["time_span"], path + ".time_span", issues)
            _check_time_span_asset(
                relation["time_span"], path + ".time_span", indexes["assets"], None, issues
            )
            _check_clock_ref(relation["time_span"], path + ".time_span", indexes["clocks"], issues)

    for track_index, track in enumerate(record.get("tracks") or []):
        if not isinstance(track, Mapping):
            continue
        path = "$.tracks[{}]".format(track_index)
        default_asset_id = track.get("asset_id")
        declared_asset_ids = set(track.get("asset_ids") or [])
        if default_asset_id is not None:
            declared_asset_ids.add(default_asset_id)
            if default_asset_id not in indexes["assets"]:
                _issue(issues, path + ".asset_id", "unknown asset {!r}".format(default_asset_id), "missing_ref")
        for asset_position, declared_asset_id in enumerate(track.get("asset_ids") or []):
            if declared_asset_id not in indexes["assets"]:
                _issue(
                    issues,
                    "{}.asset_ids[{}]".format(path, asset_position),
                    "unknown asset {!r}".format(declared_asset_id),
                    "missing_ref",
                )
        if track.get("entity_id") is not None and track.get("entity_id") not in indexes["entities"]:
            _issue(issues, path + ".entity_id", "unknown entity {!r}".format(track.get("entity_id")), "missing_ref")
        observations = track.get("observations")
        if not isinstance(observations, list) or not observations:
            _issue(issues, path + ".observations", "track requires at least one observation", "required")
        previous_frames: Dict[str, int] = {}
        previous_timestamp: Optional[float] = None
        previous_time_points: Dict[Tuple[Any, Any, Any], float] = {}
        for observation_index, observation in enumerate(observations or []):
            if not isinstance(observation, Mapping):
                continue
            observation_path = "{}.observations[{}]".format(path, observation_index)
            region_id = observation.get("region_id")
            region = indexes["regions"].get(region_id) if isinstance(region_id, str) else None
            if region_id is not None and region is None:
                _issue(
                    issues,
                    observation_path + ".region_id",
                    "unknown region {!r}".format(region_id),
                    "missing_ref",
                )
            observation_asset_id = observation.get("asset_id")
            region_asset_id = region.get("asset_id") if isinstance(region, Mapping) else None
            if observation_asset_id is not None and observation_asset_id not in indexes["assets"]:
                _issue(issues, observation_path + ".asset_id", "unknown asset reference", "missing_ref")
            if observation_asset_id is not None and region_asset_id is not None and observation_asset_id != region_asset_id:
                _issue(
                    issues,
                    observation_path + ".asset_id",
                    "observation asset does not match referenced region asset {!r}".format(region_asset_id),
                    "asset_mismatch",
                )
            effective_asset_id = observation_asset_id or region_asset_id or default_asset_id
            if effective_asset_id is None:
                _issue(
                    issues,
                    observation_path,
                    "observation must resolve an asset from asset_id, region_id, or track asset_id",
                    "required",
                )
            elif declared_asset_ids and effective_asset_id not in declared_asset_ids:
                _issue(
                    issues,
                    observation_path + ".asset_id",
                    "asset {!r} is not declared by the track".format(effective_asset_id),
                    "asset_mismatch",
                )
            if region is not None:
                if effective_asset_id is not None and region_asset_id != effective_asset_id:
                    _issue(
                        issues,
                        observation_path + ".region_id",
                        "region asset {!r} does not match observation asset {!r}".format(region_asset_id, effective_asset_id),
                        "asset_mismatch",
                    )
                if (
                    isinstance(observation.get("frame_index"), int)
                    and isinstance(region.get("frame_index"), int)
                    and observation["frame_index"] != region["frame_index"]
                ):
                    _issue(
                        issues,
                        observation_path + ".frame_index",
                        "does not match referenced region frame_index {}".format(region["frame_index"]),
                        "frame_mismatch",
                    )
            frame_index = observation.get("frame_index")
            if frame_index is not None and (not isinstance(frame_index, int) or isinstance(frame_index, bool)):
                _issue(issues, observation_path + ".frame_index", "must be an integer", "type")
            elif isinstance(frame_index, int):
                previous_frame = previous_frames.get(str(effective_asset_id))
                if previous_frame is not None and frame_index < previous_frame:
                    _issue(issues, observation_path + ".frame_index", "track frames must be nondecreasing", "order")
                if effective_asset_id is not None:
                    previous_frames[str(effective_asset_id)] = frame_index
                track_asset = indexes["assets"].get(effective_asset_id)
                media = track_asset.get("media") if isinstance(track_asset, Mapping) else None
                frame_count = media.get("frame_count") if isinstance(media, Mapping) else None
                if isinstance(frame_count, int) and frame_index >= frame_count:
                    _issue(
                        issues,
                        observation_path + ".frame_index",
                        "lies outside asset frame_count {}".format(frame_count),
                        "bounds",
                    )
            timestamp = observation.get("timestamp")
            if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))):
                _issue(issues, observation_path + ".timestamp", "must be a number", "type")
            elif isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                timestamp_value = float(timestamp)
                if previous_timestamp is not None and timestamp_value < previous_timestamp:
                    _issue(issues, observation_path + ".timestamp", "track timestamps must be nondecreasing", "order")
                previous_timestamp = timestamp_value
            if not any(field in observation for field in ("frame_index", "timestamp", "time_point")):
                _issue(issues, observation_path, "track observation requires frame_index, timestamp, or time_point", "required")
            if not any(field in observation for field in ("region_id", "state")):
                _issue(issues, observation_path, "track observation requires region_id or state", "required")
            if "time_point" in observation:
                _check_time_point(
                    observation["time_point"],
                    observation_path + ".time_point",
                    indexes["assets"],
                    indexes["clocks"],
                    effective_asset_id,
                    issues,
                )
                time_point = observation.get("time_point")
                if isinstance(time_point, Mapping) and _is_finite_number(time_point.get("value")):
                    time_key = (
                        time_point.get("clock_id"),
                        time_point.get("unit"),
                        time_point.get("reference_asset_id", effective_asset_id),
                    )
                    time_value = float(time_point["value"])
                    if time_key in previous_time_points and time_value < previous_time_points[time_key]:
                        _issue(issues, observation_path + ".time_point.value", "track time points must be nondecreasing", "order")
                    previous_time_points[time_key] = time_value
            if "time_span" in observation:
                _check_time_span(observation["time_span"], observation_path + ".time_span", issues)
                _check_time_span_asset(
                    observation["time_span"], observation_path + ".time_span", indexes["assets"], effective_asset_id, issues
                )
                _check_clock_ref(observation["time_span"], observation_path + ".time_span", indexes["clocks"], issues)
            if "data_ref" in observation:
                _check_external_data_ref(
                    observation["data_ref"], observation_path + ".data_ref", check_assets, base_dir, issues
                )
            visibility = observation.get("visibility")
            if visibility is not None and _is_finite_number(visibility) and not 0 <= float(visibility) <= 1:
                _issue(issues, observation_path + ".visibility", "visibility must be in [0, 1]", "range")

    for conversation_index, conversation in enumerate(record.get("conversations") or []):
        if not isinstance(conversation, Mapping):
            continue
        path = "$.conversations[{}]".format(conversation_index)
        messages = conversation.get("messages") or []
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                continue
            message_path = "{}.messages[{}]".format(path, message_index)
            content = message.get("content")
            candidates = message.get("candidates")
            content_signal = _content_has_signal(content)
            candidate_signal = False
            if "content" in message and not content_signal:
                _issue(issues, message_path + ".content", "content has no non-empty signal", "empty")
            _check_content(content, message_path + ".content", indexes, issues)
            if "candidates" in message and (not isinstance(candidates, list) or not candidates):
                _issue(issues, message_path + ".candidates", "candidates must be a non-empty array", "empty")
            for candidate_index, candidate in enumerate(candidates if isinstance(candidates, list) else []):
                if isinstance(candidate, Mapping):
                    candidate_content = candidate.get("content")
                    has_signal = _content_has_signal(candidate_content)
                    candidate_signal = candidate_signal or has_signal
                    if not has_signal:
                        _issue(
                            issues,
                            "{}.candidates[{}].content".format(message_path, candidate_index),
                            "candidate content has no non-empty signal",
                            "empty",
                        )
                    _check_content(
                        candidate_content,
                        "{}.candidates[{}].content".format(message_path, candidate_index),
                        indexes,
                        issues,
                    )
            if not content_signal and not candidate_signal:
                _issue(issues, message_path, "message must have non-empty content or candidates", "empty")

    reference_fields = {
        "entity_ids": "entities",
        "region_ids": "regions",
        "relation_ids": "relations",
        "track_ids": "tracks",
    }
    for annotation_index, annotation in enumerate(record.get("annotations") or []):
        if not isinstance(annotation, Mapping):
            continue
        path = "$.annotations[{}]".format(annotation_index)
        _check_node_ref(annotation.get("target"), path + ".target", indexes, issues)
        for target_index, target in enumerate(annotation.get("targets") or []):
            _check_node_ref(target, "{}.targets[{}]".format(path, target_index), indexes, issues)
        for field, collection in reference_fields.items():
            for value_index, identifier in enumerate(annotation.get(field) or []):
                if identifier not in indexes[collection]:
                    _issue(
                        issues,
                        "{}.{}[{}]".format(path, field, value_index),
                        "unknown {} {!r}".format(collection[:-1], identifier),
                        "missing_ref",
                    )
        annotation_type = annotation.get("type")
        if isinstance(annotation.get("content"), list):
            _check_content(annotation["content"], path + ".content", indexes, issues)
        if annotation_type in ("qa", "vqa", "document_qa"):
            if "question" not in annotation:
                _issue(issues, path + ".question", "qa annotation requires question", "required")
            if not annotation.get("answers"):
                _issue(issues, path + ".answers", "qa annotation requires at least one answer", "required")
            if isinstance(annotation.get("question"), list):
                _check_content(annotation["question"], path + ".question", indexes, issues)
            for answer_index, answer in enumerate(annotation.get("answers") or []):
                if not isinstance(answer, Mapping):
                    continue
                answer_path = "{}.answers[{}]".format(path, answer_index)
                has_text = isinstance(answer.get("text"), str) and bool(answer["text"].strip())
                has_content = _content_has_signal(answer.get("content"))
                if not has_text and not has_content:
                    _issue(
                        issues,
                        answer_path,
                        "qa answer requires non-empty text or structured content",
                        "annotation_contract",
                    )
                if isinstance(answer.get("content"), list):
                    _check_content(answer["content"], answer_path + ".content", indexes, issues)
            canonical_index = annotation.get("canonical_answer_index")
            answers = annotation.get("answers")
            if (
                isinstance(canonical_index, int)
                and not isinstance(canonical_index, bool)
                and isinstance(answers, list)
                and (canonical_index < 0 or canonical_index >= len(answers))
            ):
                _issue(
                    issues,
                    path + ".canonical_answer_index",
                    "lies outside answers array of length {}".format(len(answers)),
                    "bounds",
                )
            choices = annotation.get("choices")
            for choice_index, choice in enumerate(choices or []):
                choice_path = "{}.choices[{}]".format(path, choice_index)
                if isinstance(choice, str):
                    if not choice.strip():
                        _issue(issues, choice_path, "choice text must not be empty", "annotation_contract")
                elif isinstance(choice, Mapping):
                    has_choice_text = isinstance(choice.get("text"), str) and bool(
                        choice["text"].strip()
                    )
                    has_choice_content = _content_has_signal(choice.get("content"))
                    if not has_choice_text and not has_choice_content:
                        _issue(
                            issues,
                            choice_path,
                            "choice requires non-empty text or structured content",
                            "annotation_contract",
                        )
                    if isinstance(choice.get("content"), list):
                        _check_content(choice["content"], choice_path + ".content", indexes, issues)
            for choice_position, choice_index in enumerate(annotation.get("correct_choice_indices") or []):
                if (
                    isinstance(choice_index, int)
                    and not isinstance(choice_index, bool)
                    and isinstance(choices, list)
                    and (choice_index < 0 or choice_index >= len(choices))
                ):
                    _issue(
                        issues,
                        "{}.correct_choice_indices[{}]".format(path, choice_position),
                        "lies outside choices array of length {}".format(len(choices)),
                        "bounds",
                    )
        for anchor_index, anchor in enumerate(annotation.get("document_anchors") or []):
            _check_document_anchor(anchor, "{}.document_anchors[{}]".format(path, anchor_index), indexes, issues)
        _validate_annotation_contract(annotation, path, indexes, issues)
        if "data_ref" in annotation:
            _check_external_data_ref(annotation["data_ref"], path + ".data_ref", check_assets, base_dir, issues)
        if "time_span" in annotation:
            _check_time_span(annotation["time_span"], path + ".time_span", issues)
            target = annotation.get("target")
            target_asset_id = target.get("asset_id") if isinstance(target, Mapping) else None
            _check_time_span_asset(
                annotation["time_span"], path + ".time_span", indexes["assets"], target_asset_id, issues
            )
            _check_clock_ref(annotation["time_span"], path + ".time_span", indexes["clocks"], issues)

    if isinstance(episode, Mapping):
        frame_ids: Set[str] = set()
        frame_parents: Dict[str, Optional[str]] = {}
        _check_clock_ref(episode, "$.episode", indexes["clocks"], issues)
        if "steps_ref" in episode:
            _check_external_data_ref(
                episode["steps_ref"], "$.episode.steps_ref", check_assets, base_dir, issues
            )
        feature_names: Set[str] = set()
        for feature_index, feature in enumerate(episode.get("feature_specs") or []):
            if not isinstance(feature, Mapping):
                continue
            feature_name = feature.get("name")
            if isinstance(feature_name, str) and feature_name in feature_names:
                _issue(
                    issues,
                    "$.episode.feature_specs[{}].name".format(feature_index),
                    "duplicate feature specification name {!r}".format(feature_name),
                    "duplicate_id",
                )
            elif isinstance(feature_name, str):
                feature_names.add(feature_name)
        for frame_index, frame in enumerate(episode.get("coordinate_frames") or []):
            if not isinstance(frame, Mapping):
                continue
            identifier = frame.get("id")
            if identifier in frame_ids:
                _issue(issues, "$.episode.coordinate_frames[{}].id".format(frame_index), "duplicate frame ID", "duplicate_id")
            if isinstance(identifier, str):
                frame_ids.add(identifier)
                parent_id = frame.get("parent_id")
                frame_parents[identifier] = parent_id if isinstance(parent_id, str) else None
            if "transform_to_parent" in frame:
                _validate_transform(
                    frame["transform_to_parent"],
                    "$.episode.coordinate_frames[{}].transform_to_parent".format(frame_index),
                    issues,
                )
        for frame_index, frame in enumerate(episode.get("coordinate_frames") or []):
            if not isinstance(frame, Mapping):
                continue
            parent_id = frame.get("parent_id")
            if parent_id is not None and parent_id not in frame_ids:
                _issue(
                    issues,
                    "$.episode.coordinate_frames[{}].parent_id".format(frame_index),
                    "unknown parent coordinate frame {!r}".format(parent_id),
                    "missing_ref",
                )
        reported_cycles: Set[Tuple[str, ...]] = set()
        for identifier in frame_parents:
            chain: List[str] = []
            positions: Dict[str, int] = {}
            current: Optional[str] = identifier
            while current is not None and current in frame_parents:
                if current in positions:
                    cycle = tuple(chain[positions[current] :] + [current])
                    normalized = tuple(sorted(set(cycle)))
                    if normalized not in reported_cycles:
                        _issue(
                            issues,
                            "$.episode.coordinate_frames",
                            "coordinate frame parent cycle: {}".format(" -> ".join(cycle)),
                            "cycle",
                        )
                        reported_cycles.add(normalized)
                    break
                positions[current] = len(chain)
                chain.append(current)
                current = frame_parents[current]
        sensor_ids: Set[str] = set()
        for sensor_index, sensor in enumerate(episode.get("sensors") or []):
            if not isinstance(sensor, Mapping):
                continue
            identifier = sensor.get("id")
            if identifier in sensor_ids:
                _issue(issues, "$.episode.sensors[{}].id".format(sensor_index), "duplicate sensor ID", "duplicate_id")
            if isinstance(identifier, str):
                sensor_ids.add(identifier)
            if sensor.get("frame_id") is not None and sensor.get("frame_id") not in frame_ids:
                _issue(
                    issues,
                    "$.episode.sensors[{}].frame_id".format(sensor_index),
                    "unknown coordinate frame {!r}".format(sensor.get("frame_id")),
                    "missing_ref",
                )
            if "extrinsics" in sensor:
                _validate_transform(sensor["extrinsics"], "$.episode.sensors[{}].extrinsics".format(sensor_index), issues)
        for transform_index, dynamic_transform in enumerate(episode.get("dynamic_transforms") or []):
            if not isinstance(dynamic_transform, Mapping):
                continue
            transform_path = "$.episode.dynamic_transforms[{}]".format(transform_index)
            parent_frame_id = dynamic_transform.get("parent_frame_id")
            child_frame_id = dynamic_transform.get("child_frame_id")
            for field, frame_id in (("parent_frame_id", parent_frame_id), ("child_frame_id", child_frame_id)):
                if frame_id not in frame_ids:
                    _issue(issues, transform_path + "." + field, "unknown coordinate frame reference", "missing_ref")
            if parent_frame_id is not None and parent_frame_id == child_frame_id:
                _issue(issues, transform_path, "dynamic transform parent and child frames must differ", "cycle")
            if "transform" in dynamic_transform:
                _validate_transform(dynamic_transform["transform"], transform_path + ".transform", issues)
            if "time_point" in dynamic_transform:
                _check_time_point(
                    dynamic_transform["time_point"],
                    transform_path + ".time_point",
                    indexes["assets"],
                    indexes["clocks"],
                    None,
                    issues,
                )
            if "time_span" in dynamic_transform:
                _check_time_span(dynamic_transform["time_span"], transform_path + ".time_span", issues)
                _check_time_span_asset(
                    dynamic_transform["time_span"], transform_path + ".time_span", indexes["assets"], None, issues
                )
                _check_clock_ref(dynamic_transform["time_span"], transform_path + ".time_span", indexes["clocks"], issues)
            _check_clock_ref(dynamic_transform, transform_path, indexes["clocks"], issues)
            if "data_ref" in dynamic_transform:
                _check_external_data_ref(
                    dynamic_transform["data_ref"], transform_path + ".data_ref", check_assets, base_dir, issues
                )
        previous_step = -1
        previous_timestamp: Optional[float] = None
        previous_step_time_points: Dict[Tuple[Any, Any, Any], float] = {}
        raw_steps = episode.get("steps")
        if "steps" in episode and (not isinstance(raw_steps, list) or not raw_steps):
            _issue(issues, "$.episode.steps", "inline episode steps must be non-empty", "episode_steps")
        if "steps_ref" in episode and (
            isinstance(episode.get("step_count"), bool)
            or not isinstance(episode.get("step_count"), int)
            or episode["step_count"] < 1
        ):
            _issue(
                issues,
                "$.episode.step_count",
                "external episode steps require a positive step_count",
                "episode_steps",
            )
        steps = raw_steps if isinstance(raw_steps, list) else []
        for step_position, step in enumerate(steps):
            if not isinstance(step, Mapping):
                continue
            path = "$.episode.steps[{}]".format(step_position)
            index = step.get("index")
            if isinstance(index, int):
                if index <= previous_step:
                    _issue(issues, path + ".index", "step indexes must be strictly increasing", "order")
                previous_step = index
            timestamp = step.get("timestamp")
            if _is_finite_number(timestamp):
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    _issue(issues, path + ".timestamp", "step timestamps must be nondecreasing", "order")
                previous_timestamp = float(timestamp)
            if "time_point" in step:
                _check_time_point(
                    step["time_point"], path + ".time_point", indexes["assets"], indexes["clocks"], None, issues
                )
                time_point = step.get("time_point")
                if isinstance(time_point, Mapping) and _is_finite_number(time_point.get("value")):
                    time_key = (
                        time_point.get("clock_id"),
                        time_point.get("unit"),
                        time_point.get("reference_asset_id"),
                    )
                    time_value = float(time_point["value"])
                    if time_key in previous_step_time_points and time_value < previous_step_time_points[time_key]:
                        _issue(issues, path + ".time_point.value", "step time points must be nondecreasing", "order")
                    previous_step_time_points[time_key] = time_value
            if "time_span" in step:
                _check_time_span(step["time_span"], path + ".time_span", issues)
                _check_time_span_asset(step["time_span"], path + ".time_span", indexes["assets"], None, issues)
                _check_clock_ref(step["time_span"], path + ".time_span", indexes["clocks"], issues)
            _check_clock_ref(step, path, indexes["clocks"], issues)
            if "data_ref" in step:
                _check_external_data_ref(step["data_ref"], path + ".data_ref", check_assets, base_dir, issues)
            for observation_index, observation in enumerate(step.get("observations") or []):
                if not isinstance(observation, Mapping):
                    continue
                observation_path = "{}.observations[{}]".format(path, observation_index)
                if observation.get("asset_id") is not None and observation.get("asset_id") not in indexes["assets"]:
                    _issue(issues, observation_path + ".asset_id", "unknown asset reference", "missing_ref")
                if observation.get("sensor_id") is not None and observation.get("sensor_id") not in sensor_ids:
                    _issue(issues, observation_path + ".sensor_id", "unknown sensor reference", "missing_ref")
                if observation.get("frame_id") is not None and observation.get("frame_id") not in frame_ids:
                    _issue(issues, observation_path + ".frame_id", "unknown coordinate frame reference", "missing_ref")
                observation_asset_id = observation.get("asset_id")
                if "frame_index" in observation and isinstance(observation.get("frame_index"), int):
                    observation_asset = indexes["assets"].get(observation_asset_id)
                    media = observation_asset.get("media") if isinstance(observation_asset, Mapping) else None
                    frame_count = media.get("frame_count") if isinstance(media, Mapping) else None
                    if isinstance(frame_count, int) and observation["frame_index"] >= frame_count:
                        _issue(
                            issues,
                            observation_path + ".frame_index",
                            "lies outside asset frame_count {}".format(frame_count),
                            "bounds",
                        )
                if "time_point" in observation:
                    _check_time_point(
                        observation["time_point"],
                        observation_path + ".time_point",
                        indexes["assets"],
                        indexes["clocks"],
                        observation_asset_id,
                        issues,
                    )
                if "time_span" in observation:
                    _check_time_span(observation["time_span"], observation_path + ".time_span", issues)
                    _check_time_span_asset(
                        observation["time_span"],
                        observation_path + ".time_span",
                        indexes["assets"],
                        observation_asset_id,
                        issues,
                    )
                    _check_clock_ref(
                        observation["time_span"], observation_path + ".time_span", indexes["clocks"], issues
                    )
                _check_clock_ref(observation, observation_path, indexes["clocks"], issues)
                if "data_ref" in observation:
                    _check_external_data_ref(
                        observation["data_ref"], observation_path + ".data_ref", check_assets, base_dir, issues
                    )
            for action_index, action in enumerate(step.get("actions") or []):
                if isinstance(action, Mapping):
                    action_path = "{}.actions[{}]".format(path, action_index)
                    _check_node_ref(action.get("target"), action_path + ".target", indexes, issues)
                    if action.get("frame_id") is not None and action.get("frame_id") not in frame_ids:
                        _issue(issues, action_path + ".frame_id", "unknown coordinate frame reference", "missing_ref")
                    target = action.get("target")
                    target_asset_id = target.get("asset_id") if isinstance(target, Mapping) else None
                    if "time_point" in action:
                        _check_time_point(
                            action["time_point"],
                            action_path + ".time_point",
                            indexes["assets"],
                            indexes["clocks"],
                            target_asset_id,
                            issues,
                        )
                    if "time_span" in action:
                        _check_time_span(action["time_span"], action_path + ".time_span", issues)
                        _check_time_span_asset(
                            action["time_span"], action_path + ".time_span", indexes["assets"], target_asset_id, issues
                        )
                        _check_clock_ref(action["time_span"], action_path + ".time_span", indexes["clocks"], issues)
                    _check_clock_ref(action, action_path, indexes["clocks"], issues)
                    if "data_ref" in action:
                        _check_external_data_ref(action["data_ref"], action_path + ".data_ref", check_assets, base_dir, issues)
            if "is_first" in step and bool(step.get("is_first")) != (step_position == 0):
                _issue(issues, path + ".is_first", "must be true only on the first step", "episode_flag")
            if "is_last" in step and bool(step.get("is_last")) != (step_position == len(steps) - 1):
                _issue(issues, path + ".is_last", "must be true only on the last step", "episode_flag")
            if step.get("is_terminal") is True:
                if step_position != len(steps) - 1:
                    _issue(issues, path + ".is_terminal", "only the final step may be terminal", "episode_flag")
                if step.get("is_last") is not True:
                    _issue(
                        issues,
                        path + ".is_terminal",
                        "a terminal final step must also set is_last=true",
                        "episode_flag",
                    )
        steps = episode.get("steps")
        if isinstance(steps, list) and isinstance(episode.get("step_count"), int):
            if episode["step_count"] != len(steps):
                _issue(issues, "$.episode.step_count", "does not match inline steps length", "count")

    unique_issues = []
    seen = set()
    for issue in issues:
        key = (issue.path, issue.message, issue.code)
        if key not in seen:
            unique_issues.append(issue)
            seen.add(key)
    return unique_issues


def ensure_valid(
    record: Mapping[str, Any],
    check_json_schema: bool = True,
    check_assets: bool = False,
    base_dir: Optional[Path] = None,
) -> None:
    issues = validate_record(record, check_json_schema=check_json_schema, check_assets=check_assets, base_dir=base_dir)
    if issues:
        raise DatasetValidationError(issues)


@lru_cache(maxsize=1)
def _manifest_schema_validator() -> Any:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        return error
    schema = json.loads(MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError("non-finite JSON number {!r} is not allowed".format(value))


def validate_manifest(
    manifest: Mapping[str, Any],
    manifest_path: Optional[Path] = None,
    check_json_schema: bool = True,
    check_files: bool = False,
    max_errors: int = 100,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not isinstance(manifest, Mapping):
        return [ValidationIssue("$", "manifest must be an object", "manifest_contract")]
    if check_json_schema:
        validator = _manifest_schema_validator()
        if isinstance(validator, ImportError):
            _issue(
                issues,
                "$",
                "jsonschema is required for manifest validation; install it or explicitly disable schema checks",
                "jsonschema_unavailable",
            )
        else:
            for error in sorted(validator.iter_errors(manifest), key=lambda item: list(item.absolute_path)):
                location = "$"
                for part in error.absolute_path:
                    location += "[{}]".format(part) if isinstance(part, int) else ".{}".format(part)
                _issue(issues, location, error.message, "json_schema")
                if len(issues) >= max_errors:
                    return issues

    record_files = manifest.get("record_files")
    declared_record_count = 0
    complete_record_counts = isinstance(record_files, list) and bool(record_files)
    file_split_counts: Dict[str, int] = {}
    if not isinstance(record_files, list) or not record_files:
        _issue(issues, "$.record_files", "must be a non-empty record file inventory", "manifest_contract")
    else:
        for index, entry in enumerate(record_files):
            location = "$.record_files[{}]".format(index)
            if not isinstance(entry, Mapping):
                complete_record_counts = False
                _issue(issues, location, "record file entry must be an object", "manifest_contract")
                continue
            path_value = entry.get("path")
            if not isinstance(path_value, str) or not path_value.strip():
                _issue(issues, location + ".path", "must be a non-empty path", "manifest_contract")
            if entry.get("format") not in ("jsonl", "json", "parquet"):
                _issue(issues, location + ".format", "must identify a supported record format", "manifest_contract")
            count = entry.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                complete_record_counts = False
                _issue(issues, location + ".count", "must be a non-negative integer", "manifest_contract")
            else:
                declared_record_count += count
                split_name = entry.get("split")
                if isinstance(split_name, str) and split_name:
                    file_split_counts[split_name] = file_split_counts.get(split_name, 0) + count
                elif "split" in entry:
                    _issue(
                        issues,
                        location + ".split",
                        "must be a non-empty split name when present",
                        "manifest_contract",
                    )
            sha256 = entry.get("sha256")
            if not isinstance(sha256, str) or re.fullmatch(r"[A-Fa-f0-9]{64}", sha256) is None:
                _issue(issues, location + ".sha256", "must be a SHA-256 digest", "manifest_contract")
    if complete_record_counts and declared_record_count == 0:
        _issue(issues, "$.record_files", "record files must contain at least one record in total", "empty_dataset")

    task_types = manifest.get("task_types")
    if not isinstance(task_types, list):
        _issue(issues, "$.task_types", "must be an array derived from record contents", "manifest_contract")
    else:
        valid_task_types = [value for value in task_types if isinstance(value, str) and value]
        if len(valid_task_types) != len(task_types) or len(set(valid_task_types)) != len(valid_task_types):
            _issue(
                issues,
                "$.task_types",
                "must contain unique non-empty task names",
                "manifest_contract",
            )

    required_identity_fields = {
        "group_id",
        "assets[].sha256",
        "assets[].source_identities[]",
        "assets[].uri",
        "assets[].source_ref",
        "episode.id",
    }
    grouping = manifest.get("grouping")
    if not isinstance(grouping, Mapping):
        _issue(issues, "$.grouping", "must describe the canonical grouping policy", "manifest_contract")
    else:
        if grouping.get("field") != "group_id":
            _issue(issues, "$.grouping.field", "must be group_id", "manifest_contract")
        if not isinstance(grouping.get("semantics"), str) or not grouping["semantics"].strip():
            _issue(issues, "$.grouping.semantics", "must describe grouping semantics", "manifest_contract")
        identity_fields = grouping.get("identity_fields")
        declared_identity_fields = (
            {value for value in identity_fields if isinstance(value, str) and value}
            if isinstance(identity_fields, list)
            else set()
        )
        if (
            not isinstance(identity_fields, list)
            or len(declared_identity_fields) != len(identity_fields)
        ):
            _issue(
                issues,
                "$.grouping.identity_fields",
                "must contain unique non-empty field names",
                "manifest_contract",
            )
        missing_identity_fields = required_identity_fields - declared_identity_fields
        if missing_identity_fields:
            _issue(
                issues,
                "$.grouping.identity_fields",
                "missing audited identity fields: {}".format(", ".join(sorted(missing_identity_fields))),
                "manifest_contract",
            )

    required_statistics = ("records", "groups", "unique_assets", "episodes")
    declared_statistics = manifest.get("statistics")
    if not isinstance(declared_statistics, Mapping):
        _issue(issues, "$.statistics", "must contain derived production statistics", "manifest_contract")
    else:
        for field in required_statistics:
            value = declared_statistics.get(field)
            minimum = 1 if field in ("records", "groups") else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                _issue(
                    issues,
                    "$.statistics.{}".format(field),
                    "must be an integer >= {} derived from record contents".format(minimum),
                    "manifest_contract",
                )
        records_statistic = declared_statistics.get("records")
        if (
            complete_record_counts
            and isinstance(records_statistic, int)
            and not isinstance(records_statistic, bool)
            and records_statistic != declared_record_count
        ):
            _issue(
                issues,
                "$.statistics.records",
                "declares {}, but record file counts sum to {}".format(records_statistic, declared_record_count),
                "statistics_mismatch",
            )

    declared_splits = manifest.get("splits")
    if declared_splits is not None and not isinstance(declared_splits, Mapping):
        _issue(issues, "$.splits", "must map split names to record counts", "manifest_contract")
    if isinstance(declared_splits, Mapping):
        valid_split_total = 0
        valid_splits = True
        for split_name, count in declared_splits.items():
            if not isinstance(split_name, str) or not split_name:
                valid_splits = False
                _issue(issues, "$.splits", "split names must be non-empty", "manifest_contract")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                valid_splits = False
                _issue(issues, "$.splits.{}".format(split_name), "must be a non-negative integer", "manifest_contract")
            else:
                valid_split_total += count
        if valid_splits and complete_record_counts and valid_split_total != declared_record_count:
            _issue(
                issues,
                "$.splits",
                "split counts sum to {}, but record file counts sum to {}".format(
                    valid_split_total, declared_record_count
                ),
                "count_mismatch",
            )
        for split_name, count in file_split_counts.items():
            if split_name not in declared_splits:
                _issue(
                    issues,
                    "$.splits.{}".format(split_name),
                    "record files declare this split, but the manifest does not",
                    "missing_split",
                )
            elif isinstance(declared_splits.get(split_name), int) and declared_splits[split_name] < count:
                _issue(
                    issues,
                    "$.splits.{}".format(split_name),
                    "declares fewer records than its split-specific record files",
                    "count_mismatch",
                )
    elif file_split_counts:
        _issue(
            issues,
            "$.splits",
            "split-specific record files require top-level split counts",
            "missing_split",
        )

    base_dir = manifest_path.expanduser().resolve().parent if manifest_path is not None else Path.cwd()
    seen_paths: Set[Path] = set()
    observed_splits: Dict[str, int] = {}
    observed_tasks: Set[str] = set()
    unsplit_records = 0
    seen_record_ids: Set[str] = set()
    group_splits: Dict[str, str] = {}
    asset_groups: Dict[str, str] = {}
    asset_splits: Dict[str, str] = {}
    try:
        from .split import AssetIdentityRegistry
    except ImportError:
        from universal_dataset.split import AssetIdentityRegistry
    asset_identities = AssetIdentityRegistry()
    episode_groups: Dict[str, str] = {}
    episode_splits: Dict[str, str] = {}
    observed_groups: Set[str] = set()
    observed_records = 0

    def inspect_record(
        record: Any,
        location: str,
        entry_path: Path,
        declared_split: Any,
        record_locator: str,
    ) -> None:
        nonlocal observed_records, unsplit_records
        observed_records += 1
        if not isinstance(record, Mapping):
            _issue(issues, location + ".path", "{} is not an object".format(record_locator), "record_type")
            return
        if record.get("format") != FORMAT_NAME or record.get("schema_version") != SCHEMA_VERSION:
            _issue(
                issues,
                location + ".path",
                "{} has format/version {!r}/{!r}, expected {!r}/{!r}".format(
                    record_locator,
                    record.get("format"),
                    record.get("schema_version"),
                    FORMAT_NAME,
                    SCHEMA_VERSION,
                ),
                "record_version",
            )
        record_id = record.get("id")
        if isinstance(record_id, str):
            if record_id in seen_record_ids:
                _issue(
                    issues,
                    location + ".path",
                    "duplicate record id {!r} at {}".format(record_id, record_locator),
                    "duplicate_record_id",
                )
            seen_record_ids.add(record_id)
        split_value = record.get("split")
        if isinstance(split_value, str) and split_value:
            observed_splits[split_value] = observed_splits.get(split_value, 0) + 1
        else:
            unsplit_records += 1
        if isinstance(declared_split, str) and split_value != declared_split:
            _issue(
                issues,
                location + ".split",
                "{} has split {!r}, expected {!r}".format(
                    record_locator, split_value, declared_split
                ),
                "split_mismatch",
            )
        for task in record.get("task_types") or []:
            if isinstance(task, str) and task:
                observed_tasks.add(task)
        group_id = record.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            return
        observed_groups.add(group_id)
        if isinstance(split_value, str) and split_value:
            previous_split = group_splits.setdefault(group_id, split_value)
            if previous_split != split_value:
                _issue(
                    issues,
                    location + ".path",
                    "group {!r} appears in splits {!r} and {!r}".format(
                        group_id, previous_split, split_value
                    ),
                    "group_split_leakage",
                )
        for asset in record.get("assets") or []:
            if not isinstance(asset, Mapping):
                continue
            try:
                identities = asset_identities.add(asset, entry_path)
            except ValueError as error:
                _issue(issues, location + ".path", str(error), "asset_identity")
                continue
            for identity in identities:
                previous_group = asset_groups.setdefault(identity, group_id)
                if previous_group != group_id:
                    _issue(
                        issues,
                        location + ".path",
                        "asset {} belongs to groups {!r} and {!r}".format(
                            identity, previous_group, group_id
                        ),
                        "asset_group_conflict",
                    )
                if isinstance(split_value, str) and split_value:
                    previous_split = asset_splits.setdefault(identity, split_value)
                    if previous_split != split_value:
                        _issue(
                            issues,
                            location + ".path",
                            "asset {} appears in splits {!r} and {!r}".format(
                                identity, previous_split, split_value
                            ),
                            "asset_split_leakage",
                        )
        episode = record.get("episode")
        if isinstance(episode, Mapping) and isinstance(episode.get("id"), str):
            episode_id = episode["id"]
            previous_group = episode_groups.setdefault(episode_id, group_id)
            if previous_group != group_id:
                _issue(
                    issues,
                    location + ".path",
                    "episode {!r} belongs to groups {!r} and {!r}".format(
                        episode_id, previous_group, group_id
                    ),
                    "episode_group_conflict",
                )
            if isinstance(split_value, str) and split_value:
                previous_split = episode_splits.setdefault(episode_id, split_value)
                if previous_split != split_value:
                    _issue(
                        issues,
                        location + ".path",
                        "episode {!r} appears in splits {!r} and {!r}".format(
                            episode_id, previous_split, split_value
                        ),
                        "episode_split_leakage",
                    )
    for index, entry in enumerate(manifest.get("record_files") or []):
        if len(issues) >= max_errors:
            break
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            continue
        entry_path = Path(entry["path"]).expanduser()
        if not entry_path.is_absolute():
            entry_path = base_dir / entry_path
        entry_path = entry_path.resolve()
        location = "$.record_files[{}]".format(index)
        if entry_path in seen_paths:
            _issue(issues, location + ".path", "duplicate record file {!r}".format(str(entry_path)), "duplicate_path")
        seen_paths.add(entry_path)
        if not check_files:
            continue
        if not entry_path.is_file():
            _issue(issues, location + ".path", "record file does not exist: {}".format(entry_path), "missing_file")
            continue
        expected_sha256 = entry.get("sha256")
        if isinstance(expected_sha256, str) and _file_sha256(entry_path).lower() != expected_sha256.lower():
            _issue(issues, location + ".sha256", "does not match record file", "checksum")
        actual_count: Optional[int] = None
        declared_split = entry.get("split")
        file_format = entry.get("format")
        if file_format == "jsonl":
            actual_count = 0
            with entry_path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    actual_count += 1
                    try:
                        record = json.loads(line, parse_constant=_reject_nonfinite_json)
                    except (json.JSONDecodeError, ValueError) as error:
                        _issue(
                            issues,
                            location + ".path",
                            "invalid JSONL line {}: {}".format(line_number, error),
                            "invalid_jsonl",
                        )
                        break
                    inspect_record(
                        record,
                        location,
                        entry_path,
                        declared_split,
                        "line {}".format(line_number),
                    )
                    if len(issues) >= max_errors:
                        break
        elif file_format == "json":
            try:
                value = json.loads(
                    entry_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json
                )
            except (json.JSONDecodeError, ValueError) as error:
                _issue(issues, location + ".path", "invalid JSON: {}".format(error), "invalid_json")
            else:
                records = value if isinstance(value, list) else [value]
                actual_count = len(records)
                for record_index, record in enumerate(records, 1):
                    inspect_record(
                        record,
                        location,
                        entry_path,
                        declared_split,
                        "record {}".format(record_index),
                    )
                    if len(issues) >= max_errors:
                        break
        elif file_format == "parquet":
            try:
                import pyarrow.parquet as parquet
            except ImportError:
                _issue(issues, location + ".path", "pyarrow is required to inspect parquet counts", "dependency")
            else:
                parquet_file = parquet.ParquetFile(entry_path)
                actual_count = parquet_file.metadata.num_rows
                row_number = 0
                for batch in parquet_file.iter_batches(batch_size=1024):
                    for record in batch.to_pylist():
                        row_number += 1
                        inspect_record(
                            record,
                            location,
                            entry_path,
                            declared_split,
                            "row {}".format(row_number),
                        )
                        if len(issues) >= max_errors:
                            break
                    if len(issues) >= max_errors:
                        break
        if isinstance(entry.get("count"), int) and actual_count is not None and entry["count"] != actual_count:
            _issue(
                issues,
                location + ".count",
                "declares {}, observed {}".format(entry["count"], actual_count),
                "count_mismatch",
            )
    declared_splits = manifest.get("splits")
    if check_files and isinstance(declared_splits, Mapping):
        if unsplit_records:
            _issue(
                issues,
                "$.splits",
                "manifest declares splits but {} records have no split".format(unsplit_records),
                "unsplit_records",
            )
        declared_split_names = {
            split_name
            for split_name in declared_splits
            if isinstance(split_name, str) and split_name
        }
        for split_name in sorted(declared_split_names | set(observed_splits)):
            expected_count = declared_splits.get(split_name)
            observed_count = observed_splits.get(split_name, 0)
            if not isinstance(expected_count, int):
                _issue(
                    issues,
                    "$.splits.{}".format(split_name),
                    "split was observed but is not declared",
                    "missing_split",
                )
            elif observed_count != expected_count:
                _issue(
                    issues,
                    "$.splits.{}".format(split_name),
                    "declares {}, observed {} from record contents".format(
                        expected_count, observed_count
                    ),
                    "count_mismatch",
                )
    elif check_files and observed_splits:
        _issue(
            issues,
            "$.splits",
            "record contents contain splits, but the manifest does not declare them",
            "missing_split",
        )
    if check_files and isinstance(manifest.get("task_types"), list):
        declared_tasks = {
            value for value in manifest["task_types"] if isinstance(value, str) and value
        }
        for task in sorted(declared_tasks - observed_tasks):
            _issue(
                issues,
                "$.task_types",
                "declared task {!r} is absent from record contents".format(task),
                "extra_task_type",
            )
        for task in sorted(observed_tasks - declared_tasks):
            _issue(
                issues,
                "$.task_types",
                "record task {!r} is missing from manifest".format(task),
                "missing_task_type",
            )
    declared_statistics = manifest.get("statistics")
    if check_files and isinstance(declared_statistics, Mapping):
        observed_statistics = {
            "records": observed_records,
            "groups": len(observed_groups),
            "unique_assets": len(asset_identities),
            "episodes": len(episode_groups),
        }
        for field, observed_value in observed_statistics.items():
            if field not in declared_statistics:
                continue
            declared_value = declared_statistics[field]
            if not isinstance(declared_value, int) or isinstance(declared_value, bool):
                _issue(
                    issues,
                    "$.statistics.{}".format(field),
                    "must be an integer derived from record contents",
                    "statistics_type",
                )
            elif declared_value != observed_value:
                _issue(
                    issues,
                    "$.statistics.{}".format(field),
                    "declares {}, observed {} from record contents".format(
                        declared_value, observed_value
                    ),
                    "statistics_mismatch",
                )
    return issues[:max_errors]
