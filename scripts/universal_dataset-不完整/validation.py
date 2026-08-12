"""Structural and cross-reference validation for S1-UDF records."""

from __future__ import annotations

import math
import os
import hashlib
import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit


FORMAT_NAME = "s1-udf"
SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = Path(__file__).with_name("schemas") / "record-1.0.0.schema.json"
MANIFEST_SCHEMA_PATH = Path(__file__).with_name("schemas") / "manifest-1.0.0.schema.json"


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


def _coordinate_pairs(value: Any) -> List[Tuple[float, float]]:
    if not isinstance(value, list):
        return []
    if value and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        if len(value) % 2 != 0:
            return []
        return [(float(value[index]), float(value[index + 1])) for index in range(0, len(value), 2)]
    pairs: List[Tuple[float, float]] = []
    for item in value:
        pairs.extend(_coordinate_pairs(item))
    return pairs


def _validate_geometry(
    region: Mapping[str, Any], path: str, asset: Optional[Mapping[str, Any]], issues: List[ValidationIssue]
) -> None:
    geometry = region.get("geometry")
    if not isinstance(geometry, Mapping):
        return
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    numbers = list(_finite_numbers(coordinates))
    if any(not math.isfinite(number) for number in numbers):
        _issue(issues, path + ".geometry.coordinates", "contains NaN or infinity", "non_finite")
    expected_lengths = {"point2d": 2, "bbox2d": 4, "point3d": 3}
    if geometry_type in expected_lengths:
        if not isinstance(coordinates, list) or len(coordinates) != expected_lengths[geometry_type]:
            _issue(
                issues,
                path + ".geometry.coordinates",
                "{} requires {} coordinates".format(geometry_type, expected_lengths[geometry_type]),
                "shape",
            )
    if geometry_type in ("polygon2d", "polyline2d", "keypoints2d"):
        pairs = _coordinate_pairs(coordinates)
        minimum = 3 if geometry_type == "polygon2d" else 2 if geometry_type == "polyline2d" else 1
        if len(pairs) < minimum:
            _issue(
                issues,
                path + ".geometry.coordinates",
                "{} requires at least {} coordinate pairs".format(geometry_type, minimum),
                "shape",
            )
        visibility = geometry.get("visibility")
        if geometry_type == "keypoints2d" and isinstance(visibility, list) and len(visibility) != len(pairs):
            _issue(
                issues,
                path + ".geometry.visibility",
                "length must match keypoint count {}".format(len(pairs)),
                "count",
            )
    if geometry_type == "rle":
        size = geometry.get("size")
        if not isinstance(size, list) or len(size) != 2 or geometry.get("counts") is None:
            _issue(issues, path + ".geometry", "rle requires size=[height,width] and counts", "required")
    if geometry_type == "mask_ref" and not isinstance(geometry.get("asset_ref"), str):
        _issue(issues, path + ".geometry.asset_ref", "mask_ref requires an asset_ref", "required")
    bbox = _bbox_xyxy(geometry)
    if geometry_type == "bbox2d" and bbox is None:
        _issue(issues, path + ".geometry", "bbox2d format must be xyxy, xywh, or cxcywh", "bbox_format")
    elif bbox is not None and (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
        _issue(issues, path + ".geometry.coordinates", "bbox has negative width or height", "range")

    coordinate_space = geometry.get("coordinate_space")
    if not isinstance(coordinate_space, Mapping):
        return
    space_type = coordinate_space.get("type")
    if space_type == "normalized":
        bounded_numbers = bbox if bbox is not None else numbers
        if any(number < 0 or number > 1 for number in bounded_numbers):
            _issue(issues, path + ".geometry.coordinates", "normalized coordinates must be in [0, 1]", "range")
    if space_type != "pixel" or not asset:
        return
    media = asset.get("media")
    if not isinstance(media, Mapping):
        return
    width, height = media.get("width"), media.get("height")
    if isinstance(width, int) and isinstance(height, int):
        out_of_bounds = False
        if bbox is not None:
            out_of_bounds = bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height
        elif geometry_type in ("point2d", "polygon2d", "polyline2d", "keypoints2d"):
            out_of_bounds = any(
                x < 0 or y < 0 or x > width or y > height for x, y in _coordinate_pairs(coordinates)
            )
        if out_of_bounds:
            _issue(
                issues,
                path + ".geometry.coordinates",
                "pixel geometry lies outside asset dimensions {}x{}".format(width, height),
                "bounds",
            )


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
        if "time_span" in part:
            _check_time_span(part["time_span"], part_path + ".time_span", issues)
            _check_time_span_asset(
                part["time_span"],
                part_path + ".time_span",
                indexes["assets"],
                part.get("asset_id") if part_type == "asset" else None,
                issues,
            )


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

    carrier_fields = ("assets", "entities", "regions", "tracks", "conversations", "annotations")
    has_carrier = any(isinstance(record.get(field), list) and bool(record.get(field)) for field in carrier_fields)
    has_carrier = has_carrier or isinstance(record.get("episode"), Mapping)
    if not has_carrier:
        _issue(
            issues,
            "$",
            "record must contain at least one non-empty assets/entities/regions/tracks/conversations/annotations collection or episode",
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
    }

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

    for asset_index, asset in enumerate(record.get("assets") or []):
        if not isinstance(asset, Mapping):
            continue
        path = "$.assets[{}]".format(asset_index)
        if "time_span" in asset:
            _check_time_span(asset["time_span"], path + ".time_span", issues)
            _check_time_span_asset(
                asset["time_span"], path + ".time_span", indexes["assets"], asset.get("id"), issues
            )
        if isinstance(episode, Mapping) and asset.get("sensor_id") is not None and asset.get("sensor_id") not in episode_sensor_ids:
            _issue(issues, path + ".sensor_id", "unknown episode sensor {!r}".format(asset.get("sensor_id")), "missing_ref")
        if isinstance(episode, Mapping) and asset.get("frame_id") is not None and asset.get("frame_id") not in episode_frame_ids:
            _issue(issues, path + ".frame_id", "unknown coordinate frame {!r}".format(asset.get("frame_id")), "missing_ref")
        uri = asset.get("uri")
        if check_assets and isinstance(uri, str):
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
        geometry = region.get("geometry")
        if isinstance(geometry, Mapping):
            asset_ref = geometry.get("asset_ref")
            if asset_ref is not None and asset_ref not in indexes["assets"]:
                _issue(issues, path + ".geometry.asset_ref", "unknown asset {!r}".format(asset_ref), "missing_ref")
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
        _validate_geometry(region, path, asset, issues)

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

    for track_index, track in enumerate(record.get("tracks") or []):
        if not isinstance(track, Mapping):
            continue
        path = "$.tracks[{}]".format(track_index)
        if track.get("asset_id") not in indexes["assets"]:
            _issue(issues, path + ".asset_id", "unknown asset {!r}".format(track.get("asset_id")), "missing_ref")
        if track.get("entity_id") is not None and track.get("entity_id") not in indexes["entities"]:
            _issue(issues, path + ".entity_id", "unknown entity {!r}".format(track.get("entity_id")), "missing_ref")
        track_asset_id = track.get("asset_id")
        previous_frame: Optional[int] = None
        previous_timestamp: Optional[float] = None
        for observation_index, observation in enumerate(track.get("observations") or []):
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
            elif region is not None:
                if region.get("asset_id") != track_asset_id:
                    _issue(
                        issues,
                        observation_path + ".region_id",
                        "region asset {!r} does not match track asset {!r}".format(
                            region.get("asset_id"), track_asset_id
                        ),
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
            if frame_index is not None and not isinstance(frame_index, int):
                _issue(issues, observation_path + ".frame_index", "must be an integer", "type")
            elif isinstance(frame_index, int):
                if previous_frame is not None and frame_index < previous_frame:
                    _issue(issues, observation_path + ".frame_index", "track frames must be nondecreasing", "order")
                previous_frame = frame_index
                track_asset = indexes["assets"].get(track_asset_id)
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
            candidates = message.get("candidates") or []
            if not content and not candidates:
                _issue(issues, message_path, "message must have content or candidates", "empty")
            _check_content(content, message_path + ".content", indexes, issues)
            for candidate_index, candidate in enumerate(candidates):
                if isinstance(candidate, Mapping):
                    _check_content(
                        candidate.get("content"),
                        "{}.candidates[{}].content".format(message_path, candidate_index),
                        indexes,
                        issues,
                    )

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
        if annotation_type == "caption" and not isinstance(annotation.get("text"), str):
            _issue(issues, path + ".text", "caption annotation requires text", "required")
        if annotation_type == "qa":
            if "question" not in annotation:
                _issue(issues, path + ".question", "qa annotation requires question", "required")
            if not annotation.get("answers"):
                _issue(issues, path + ".answers", "qa annotation requires at least one answer", "required")
            if isinstance(annotation.get("question"), list):
                _check_content(annotation["question"], path + ".question", indexes, issues)
            for answer_index, answer in enumerate(annotation.get("answers") or []):
                if isinstance(answer, Mapping) and isinstance(answer.get("content"), list):
                    _check_content(answer["content"], "{}.answers[{}].content".format(path, answer_index), indexes, issues)
            canonical_index = annotation.get("canonical_answer_index")
            answers = annotation.get("answers")
            if isinstance(canonical_index, int) and isinstance(answers, list) and canonical_index >= len(answers):
                _issue(
                    issues,
                    path + ".canonical_answer_index",
                    "lies outside answers array of length {}".format(len(answers)),
                    "bounds",
                )
            choices = annotation.get("choices")
            for choice_position, choice_index in enumerate(annotation.get("correct_choice_indices") or []):
                if isinstance(choice_index, int) and isinstance(choices, list) and choice_index >= len(choices):
                    _issue(
                        issues,
                        "{}.correct_choice_indices[{}]".format(path, choice_position),
                        "lies outside choices array of length {}".format(len(choices)),
                        "bounds",
                    )
        if "time_span" in annotation:
            _check_time_span(annotation["time_span"], path + ".time_span", issues)
            target = annotation.get("target")
            target_asset_id = target.get("asset_id") if isinstance(target, Mapping) else None
            _check_time_span_asset(
                annotation["time_span"], path + ".time_span", indexes["assets"], target_asset_id, issues
            )

    if isinstance(episode, Mapping):
        frame_ids: Set[str] = set()
        frame_parents: Dict[str, Optional[str]] = {}
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
        previous_step = -1
        previous_timestamp: Optional[float] = None
        steps = episode.get("steps") or []
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
            if isinstance(timestamp, (int, float)):
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    _issue(issues, path + ".timestamp", "step timestamps must be nondecreasing", "order")
                previous_timestamp = float(timestamp)
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
            for action_index, action in enumerate(step.get("actions") or []):
                if isinstance(action, Mapping):
                    action_path = "{}.actions[{}]".format(path, action_index)
                    _check_node_ref(action.get("target"), action_path + ".target", indexes, issues)
                    if action.get("frame_id") is not None and action.get("frame_id") not in frame_ids:
                        _issue(issues, action_path + ".frame_id", "unknown coordinate frame reference", "missing_ref")
            if "is_first" in step and bool(step.get("is_first")) != (step_position == 0):
                _issue(issues, path + ".is_first", "must be true only on the first step", "episode_flag")
            if "is_last" in step and bool(step.get("is_last")) != (step_position == len(steps) - 1):
                _issue(issues, path + ".is_last", "must be true only on the last step", "episode_flag")
            if step.get("is_terminal") is True and step.get("is_last") is False:
                _issue(issues, path + ".is_terminal", "terminal step cannot explicitly set is_last=false", "episode_flag")
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

    base_dir = manifest_path.expanduser().resolve().parent if manifest_path is not None else Path.cwd()
    seen_paths: Set[Path] = set()
    observed_splits: Dict[str, int] = {}
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
                    if isinstance(declared_split, str):
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
                        if isinstance(record, Mapping) and record.get("split") != declared_split:
                            _issue(
                                issues,
                                location + ".split",
                                "line {} has split {!r}, expected {!r}".format(
                                    line_number, record.get("split"), declared_split
                                ),
                                "split_mismatch",
                            )
                            break
        elif file_format == "json":
            value = json.loads(
                entry_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json
            )
            actual_count = len(value) if isinstance(value, list) else 1
        elif file_format == "parquet":
            try:
                import pyarrow.parquet as parquet
            except ImportError:
                _issue(issues, location + ".path", "pyarrow is required to inspect parquet counts", "dependency")
            else:
                actual_count = parquet.ParquetFile(entry_path).metadata.num_rows
        if isinstance(entry.get("count"), int) and actual_count is not None and entry["count"] != actual_count:
            _issue(
                issues,
                location + ".count",
                "declares {}, observed {}".format(entry["count"], actual_count),
                "count_mismatch",
            )
        if isinstance(declared_split, str) and actual_count is not None:
            observed_splits[declared_split] = observed_splits.get(declared_split, 0) + actual_count

    declared_splits = manifest.get("splits")
    if check_files and isinstance(declared_splits, Mapping):
        for split_name, expected_count in declared_splits.items():
            if isinstance(expected_count, int) and split_name in observed_splits:
                if observed_splits[split_name] != expected_count:
                    _issue(
                        issues,
                        "$.splits.{}".format(split_name),
                        "declares {}, observed {} from record_files".format(
                            expected_count, observed_splits[split_name]
                        ),
                        "count_mismatch",
                    )
    return issues[:max_errors]
