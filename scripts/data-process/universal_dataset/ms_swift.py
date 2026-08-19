"""Loss-aware adapters between S1-UDF records and ms-swift JSONL records."""

from __future__ import annotations

import hashlib
import json
import math
import ntpath
import posixpath
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from .schema_registry import CURRENT_SCHEMA_VERSION, FORMAT_NAME


TOKEN_PATTERN = re.compile(r"<(image|video|audio|ref-object|bbox)>")
MEDIA_KEYS = {"image": "images", "video": "videos", "audio": "audios"}
WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class MsSwiftConversionError(ValueError):
    pass


def _resolve_media(value: str, base_dir: Optional[Path] = None) -> str:
    value = value.strip()
    if WINDOWS_DRIVE_PATH.match(value):
        return ntpath.normpath(value)
    if value.startswith("\\\\"):
        return ntpath.normpath(value)
    parsed = urlsplit(value)
    if parsed.scheme and parsed.scheme.lower() != "file":
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))
    if parsed.scheme.lower() == "file" and parsed.netloc.lower() not in ("", "localhost"):
        return urlunsplit(("file", parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))
    local_value = parsed.path if parsed.scheme.lower() == "file" else value
    if re.match(r"^/[A-Za-z]:/", local_value):
        return ntpath.normpath(local_value[1:])
    if local_value.startswith("/"):
        return posixpath.normpath(local_value)
    path = Path(local_value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return str(path.resolve()) if path.is_absolute() or base_dir is not None else str(path)


def _media_group_id(values: Iterable[str], fallback: str, base_dir: Optional[Path] = None) -> str:
    canonical = sorted(set(_resolve_media(value, base_dir) for value in values if value))
    if not canonical:
        return "text:" + fallback
    digest = hashlib.sha256("\0".join(canonical).encode("utf-8")).hexdigest()
    return "media:" + digest


def _probe_local_image_size(uri: str) -> Tuple[int, int]:
    if not WINDOWS_DRIVE_PATH.match(uri) and not uri.startswith("\\\\"):
        parsed = urlsplit(uri)
        if parsed.scheme:
            raise MsSwiftConversionError(
                "ms-swift bbox_type='real' requires a readable local image to determine pixel dimensions; "
                "materialize {!r} locally before import".format(uri)
            )
    path = Path(uri)
    if not path.is_file():
        raise MsSwiftConversionError(
            "ms-swift bbox_type='real' image {!r} is not a readable local file; "
            "materialize the media before import".format(uri)
        )
    try:
        from PIL import Image
    except ImportError as error:
        raise MsSwiftConversionError(
            "Pillow is required to read image dimensions for ms-swift bbox_type='real'; "
            "install the universal_dataset requirements"
        ) from error
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, SyntaxError, ValueError) as error:
        raise MsSwiftConversionError(
            "Cannot decode ms-swift bbox_type='real' image {!r} with Pillow; "
            "materialize a supported local image before import: {}".format(uri, error)
        ) from error
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise MsSwiftConversionError(
            "Decoded image {!r} has invalid dimensions {!r}x{!r}".format(uri, width, height)
        )
    return width, height


def _entity_name(entity: Mapping[str, Any]) -> str:
    display_name = entity.get("display_name")
    if isinstance(display_name, str) and display_name:
        return display_name
    for label in entity.get("labels") or []:
        if isinstance(label, Mapping) and isinstance(label.get("name"), str) and label["name"]:
            return label["name"]
    return str(entity.get("id", "object"))


def _geometry_box(region: Mapping[str, Any]) -> Tuple[List[float], str]:
    geometry = region.get("geometry")
    if not isinstance(geometry, Mapping):
        raise MsSwiftConversionError("Region {!r} has no geometry".format(region.get("id")))
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "point2d":
        if not isinstance(coordinates, list) or len(coordinates) != 2:
            raise MsSwiftConversionError("point2d region {!r} must have two coordinates".format(region.get("id")))
        result = [float(value) for value in coordinates]
    elif geometry_type == "bbox2d":
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise MsSwiftConversionError("bbox2d region {!r} must have four coordinates".format(region.get("id")))
        x0, y0, c2, c3 = (float(value) for value in coordinates)
        bbox_format = geometry.get("format", "xyxy")
        if bbox_format == "xyxy":
            result = [x0, y0, c2, c3]
        elif bbox_format == "xywh":
            result = [x0, y0, x0 + c2, y0 + c3]
        elif bbox_format == "cxcywh":
            result = [x0 - c2 / 2, y0 - c3 / 2, x0 + c2 / 2, y0 + c3 / 2]
        else:
            raise MsSwiftConversionError("Unsupported bbox2d format {!r}".format(bbox_format))
    else:
        raise MsSwiftConversionError(
            "ms-swift objects cannot represent geometry type {!r}; convert it explicitly or choose a target that can".format(
                geometry_type
            )
        )
    coordinate_space = geometry.get("coordinate_space") or {}
    space_type = coordinate_space.get("type")
    if space_type == "pixel":
        return result, "real"
    if space_type == "normalized":
        return result, "norm1"
    raise MsSwiftConversionError(
        "ms-swift objects support pixel or normalized 2D coordinates, got {!r}".format(space_type)
    )


def _normalize_box(box: Sequence[float], asset: Mapping[str, Any]) -> List[float]:
    media = asset.get("media") or {}
    width, height = media.get("width"), media.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise MsSwiftConversionError(
            "Mixed bbox coordinate spaces require width/height for asset {!r}".format(asset.get("id"))
        )
    if len(box) == 2:
        return [box[0] / width, box[1] / height]
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


def _asset_order(messages: Sequence[Mapping[str, Any]]) -> Tuple[List[str], Dict[str, int], Dict[str, List[str]]]:
    occurrences: List[str] = []
    first_image_index: Dict[str, int] = {}
    by_kind = {"images": [], "videos": [], "audios": []}
    for message in messages:
        for part in message.get("content") or []:
            if isinstance(part, Mapping) and part.get("type") == "asset":
                occurrences.append(str(part.get("asset_id")))
    return occurrences, first_image_index, by_kind


def _render_conversation(
    record: Mapping[str, Any], conversation: Mapping[str, Any], include_ids: bool,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    assets = {asset["id"]: asset for asset in record.get("assets") or [] if isinstance(asset, Mapping) and "id" in asset}
    entities = {
        entity["id"]: entity for entity in record.get("entities") or [] if isinstance(entity, Mapping) and "id" in entity
    }
    regions = {
        region["id"]: region for region in record.get("regions") or [] if isinstance(region, Mapping) and "id" in region
    }
    messages = conversation.get("messages") or []
    asset_occurrences, image_indexes, media_values = _asset_order(messages)
    occurrence_assets: List[Mapping[str, Any]] = []
    for asset_id in asset_occurrences:
        asset = assets.get(asset_id)
        if asset is None:
            raise MsSwiftConversionError("Conversation references unknown asset {!r}".format(asset_id))
        kind = asset.get("kind")
        key = MEDIA_KEYS.get(kind)
        if key is None:
            raise MsSwiftConversionError("ms-swift cannot insert asset kind {!r}".format(kind))
        uri = asset.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            if asset.get("source_ref") is not None:
                raise MsSwiftConversionError(
                    "Asset {!r} has only source_ref; materialize it to a direct media uri or use a target-specific resolver".format(
                        asset_id
                    )
                )
            raise MsSwiftConversionError(
                "Asset {!r} has no non-empty direct media uri".format(asset_id)
            )
        if kind == "image" and asset_id not in image_indexes:
            image_indexes[asset_id] = len(media_values["images"])
        media_values[key].append(_resolve_media(uri, base_dir))
        occurrence_assets.append(asset)

    output_messages: List[Dict[str, str]] = []
    refs: List[str] = []
    boxes: List[List[float]] = []
    box_modes: List[str] = []
    box_assets: List[Mapping[str, Any]] = []
    image_ids: List[int] = []
    asset_cursor = 0
    has_assistant_target = False
    for message_index, message in enumerate(messages):
        if message.get("candidates"):
            raise MsSwiftConversionError(
                "Conversation {!r} message {} contains preference candidates; ms-swift SFT messages need one response".format(
                    conversation.get("id"), message_index
                )
            )
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise MsSwiftConversionError(
                "Conversation {!r} message {} has unsupported ms-swift role {!r}".format(
                    conversation.get("id"), message_index, role
                )
            )
        chunks: List[str] = []
        for part in message.get("content") or []:
            if not isinstance(part, Mapping):
                raise MsSwiftConversionError("Message content parts must be objects")
            part_type = part.get("type")
            if part_type == "text":
                chunks.append(str(part.get("text", "")))
            elif part_type == "asset":
                if asset_cursor >= len(occurrence_assets):
                    raise MsSwiftConversionError("Internal asset occurrence mismatch")
                kind = occurrence_assets[asset_cursor].get("kind")
                chunks.append("<{}>".format(kind))
                asset_cursor += 1
            elif part_type == "entity":
                entity = entities.get(part.get("entity_id"))
                if entity is None:
                    raise MsSwiftConversionError("Unknown entity {!r}".format(part.get("entity_id")))
                chunks.append("<ref-object>")
                surface = part.get("surface")
                refs.append(surface if isinstance(surface, str) and surface else _entity_name(entity))
            elif part_type == "region":
                region = regions.get(part.get("region_id"))
                if region is None:
                    raise MsSwiftConversionError("Unknown region {!r}".format(part.get("region_id")))
                asset_id = region.get("asset_id")
                if asset_id not in image_indexes:
                    raise MsSwiftConversionError(
                        "Region {!r} refers to image {!r} that is not inserted in the conversation".format(
                            region.get("id"), asset_id
                        )
                    )
                box, mode = _geometry_box(region)
                chunks.append("<bbox>")
                boxes.append(box)
                box_modes.append(mode)
                box_assets.append(assets[asset_id])
                image_ids.append(image_indexes[asset_id])
            elif part_type == "tool_call":
                raise MsSwiftConversionError("Structured tool_call content needs a tool-training target adapter")
            elif part_type == "tool_result":
                result_value = part.get("result", "")
                chunks.append(
                    result_value
                    if isinstance(result_value, str)
                    else json.dumps(result_value, ensure_ascii=False, separators=(",", ":"))
                )
            else:
                raise MsSwiftConversionError("Unsupported content part type {!r}".format(part_type))
        rendered_content = "".join(chunks)
        if role == "assistant":
            if not rendered_content.strip():
                raise MsSwiftConversionError(
                    "Conversation {!r} message {} renders an empty assistant target".format(
                        conversation.get("id"), message_index
                    )
                )
            if not _content_has_renderable_signal(message.get("content")):
                raise MsSwiftConversionError(
                    "Conversation {!r} message {} has no valid assistant target signal".format(
                        conversation.get("id"), message_index
                    )
                )
            has_assistant_target = True
        output_messages.append({"role": str(role), "content": rendered_content})

    if not has_assistant_target:
        raise MsSwiftConversionError(
            "Conversation {!r} has no non-empty renderable assistant target".format(
                conversation.get("id")
            )
        )

    bbox_type: Optional[str] = None
    if box_modes:
        unique_modes = set(box_modes)
        if len(unique_modes) == 1:
            bbox_type = box_modes[0]
        else:
            boxes = [
                _normalize_box(box, asset) if mode == "real" else box
                for box, mode, asset in zip(boxes, box_modes, box_assets)
            ]
            bbox_type = "norm1"

    result: Dict[str, Any] = {
        "messages": output_messages,
        "group_id": str(record["group_id"]),
    }
    if include_ids:
        result["id"] = str(record["id"])
    for key in ("images", "videos", "audios"):
        if media_values[key]:
            result[key] = media_values[key]
    if refs or boxes:
        objects: Dict[str, Any] = {"ref": refs, "bbox": boxes}
        if bbox_type is not None:
            objects["bbox_type"] = bbox_type
        if image_ids and (len(media_values["images"]) > 1 or any(index != 0 for index in image_ids)):
            objects["image_id"] = image_ids
        result["objects"] = objects
    return result


def _select_answer(annotation: Mapping[str, Any], answer_policy: str) -> Mapping[str, Any]:
    answers = annotation.get("answers") or []
    if not answers:
        raise MsSwiftConversionError("QA annotation {!r} has no answers".format(annotation.get("id")))
    canonical_index = annotation.get("canonical_answer_index")
    if isinstance(canonical_index, int):
        if canonical_index < 0 or canonical_index >= len(answers):
            raise MsSwiftConversionError(
                "QA annotation {!r} canonical_answer_index lies outside answers".format(annotation.get("id"))
            )
        answer = answers[canonical_index]
    elif len(answers) == 1:
        answer = answers[0]
    elif answer_policy == "first":
        answer = answers[0]
    elif answer_policy == "highest-count":
        ranked = [
            (answer.get("count", 0) if isinstance(answer, Mapping) else 0, -index, answer)
            for index, answer in enumerate(answers)
        ]
        answer = max(ranked, key=lambda value: (value[0], value[1]))[2]
    else:
        raise MsSwiftConversionError(
            "QA annotation {!r} has {} answers and no canonical_answer_index; choose an explicit answer policy".format(
                annotation.get("id"), len(answers)
            )
        )
    if not isinstance(answer, Mapping):
        raise MsSwiftConversionError("QA answer must be an object")
    return answer


def _content_has_renderable_signal(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") == "text":
            if isinstance(part.get("text"), str) and bool(part["text"].strip()):
                return True
        elif part.get("type") == "tool_result":
            result = part.get("result")
            if isinstance(result, str):
                if result.strip():
                    return True
            elif result is not None and not (
                isinstance(result, (Mapping, list, tuple)) and len(result) == 0
            ):
                return True
        elif isinstance(part.get("type"), str) and part.get("type"):
            return True
    return False


def _render_choice_parts(choices: Sequence[Any]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = [{"type": "text", "text": "\nChoices:"}]
    for index, choice in enumerate(choices):
        parts.append({"type": "text", "text": "\n{}. ".format(index + 1)})
        if isinstance(choice, str) and choice.strip():
            parts.append({"type": "text", "text": choice})
            continue
        if not isinstance(choice, Mapping):
            raise MsSwiftConversionError("Choice {} must be text or structured content".format(index))
        text = choice.get("text")
        content = choice.get("content")
        has_text = isinstance(text, str) and bool(text.strip())
        has_content = _content_has_renderable_signal(content)
        if has_text and has_content:
            raise MsSwiftConversionError(
                "Choice {} defines both text and content; projection would be ambiguous".format(index)
            )
        if has_text:
            parts.append({"type": "text", "text": text})
        elif has_content:
            parts.extend(dict(part) for part in content if isinstance(part, Mapping))
        else:
            raise MsSwiftConversionError(
                "Choice {} has no non-empty text or structured content".format(index)
            )
    return parts


def _annotation_conversation(
    record: Mapping[str, Any],
    annotation: Mapping[str, Any],
    answer_policy: str,
    caption_prompt: Optional[str],
) -> Mapping[str, Any]:
    annotation_type = annotation.get("type")
    target = annotation.get("target") or {}
    target_asset = target.get("asset_id") if isinstance(target, Mapping) else None
    prefix = [{"type": "asset", "asset_id": target_asset}] if target_asset else []
    if annotation_type in ("caption", "captioning", "dense_caption"):
        metadata = annotation.get("metadata") or {}
        prompt = metadata.get("sft_prompt") if isinstance(metadata, Mapping) else None
        prompt = prompt if isinstance(prompt, str) and prompt else caption_prompt
        if not prompt:
            raise MsSwiftConversionError(
                "Caption annotation {!r} needs metadata.sft_prompt or an explicit caption prompt".format(
                    annotation.get("id")
                )
            )
        caption_content = annotation.get("content")
        if _content_has_renderable_signal(caption_content):
            assistant_content = list(caption_content)
        elif isinstance(annotation.get("text"), str) and annotation["text"].strip():
            assistant_content = [{"type": "text", "text": annotation["text"]}]
        else:
            raise MsSwiftConversionError(
                "Caption annotation {!r} has no non-empty text or content".format(
                    annotation.get("id")
                )
            )
        messages = [
            {"role": "user", "content": prefix + [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": assistant_content},
        ]
    elif annotation_type in ("qa", "vqa", "document_qa"):
        question = annotation.get("question")
        if isinstance(question, list) and _content_has_renderable_signal(question):
            question_content = list(question)
        elif isinstance(question, str) and question.strip():
            question_content = [{"type": "text", "text": question}]
        else:
            raise MsSwiftConversionError(
                "QA annotation {!r} has no non-empty question".format(annotation.get("id"))
            )
        choices = annotation.get("choices") or []
        if choices:
            if not isinstance(choices, list):
                raise MsSwiftConversionError("QA choices must be an array")
            question_content.extend(_render_choice_parts(choices))
        answer = _select_answer(annotation, answer_policy)
        answer_content = answer.get("content")
        if _content_has_renderable_signal(answer_content):
            answer_content = list(answer_content)
        elif isinstance(answer.get("text"), str) and answer["text"].strip():
            answer_content = [{"type": "text", "text": answer["text"]}]
        else:
            raise MsSwiftConversionError(
                "QA annotation {!r} selected an empty answer".format(annotation.get("id"))
            )
        messages = [
            {"role": "user", "content": prefix + list(question_content)},
            {"role": "assistant", "content": answer_content},
        ]
    else:
        raise MsSwiftConversionError(
            "Annotation type {!r} has no implicit ms-swift SFT mapping; add a conversation or target-specific adapter".format(
                annotation_type
            )
        )
    return {"id": str(annotation.get("id")), "messages": messages}


def record_to_ms_swift(
    record: Mapping[str, Any],
    include_ids: bool = True,
    base_dir: Optional[Path] = None,
    answer_policy: str = "error",
    annotation_policy: str = "error",
    caption_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if answer_policy not in ("error", "first", "highest-count"):
        raise ValueError("answer_policy must be error, first, or highest-count")
    if annotation_policy not in ("error", "prefer-conversations", "prefer-annotations"):
        raise ValueError("annotation_policy must be error, prefer-conversations, or prefer-annotations")
    conversations = record.get("conversations") or []
    annotations = record.get("annotations") or []
    if conversations and annotations:
        if annotation_policy == "error":
            raise MsSwiftConversionError(
                "Record {!r} has both conversations and annotations; choose an explicit projection policy".format(
                    record.get("id")
                )
            )
        if annotation_policy == "prefer-annotations":
            conversations = []
    if conversations:
        result = [
            _render_conversation(record, conversation, include_ids, base_dir=base_dir)
            for conversation in conversations
        ]
        if include_ids and len(result) > 1:
            for index, (conversation, value) in enumerate(zip(conversations, result)):
                value["id"] = "{}::conversation:{}:{}".format(
                    record["id"], index, conversation.get("id", index)
                )
        return result
    if annotations:
        result = [
            _render_conversation(
                record,
                _annotation_conversation(record, annotation, answer_policy, caption_prompt),
                include_ids,
                base_dir=base_dir,
            )
            for annotation in annotations
        ]
        if include_ids and len(result) > 1:
            for index, (annotation, value) in enumerate(zip(annotations, result)):
                value["id"] = "{}::annotation:{}:{}".format(
                    record["id"], index, annotation.get("id", index)
                )
        return result
    raise MsSwiftConversionError("Record {!r} has no conversation or exportable annotation".format(record.get("id")))


def ms_swift_projection_warnings(
    record: Mapping[str, Any], answer_policy: str, annotation_policy: str
) -> List[str]:
    warnings: List[str] = []
    conversations = record.get("conversations") or []
    annotations = record.get("annotations") or []
    if conversations and annotations:
        ignored = "annotations" if annotation_policy == "prefer-conversations" else "conversations"
        warnings.append("ignored_{}".format(ignored))
    selected_annotations = annotations if not conversations or annotation_policy == "prefer-annotations" else []
    for annotation in selected_annotations:
        if not isinstance(annotation, Mapping) or annotation.get("type") not in (
            "qa", "vqa", "document_qa"
        ):
            continue
        answers = annotation.get("answers") or []
        if len(answers) > 1:
            warnings.append(
                "multiple_answers_canonical" if isinstance(annotation.get("canonical_answer_index"), int)
                else "multiple_answers_{}".format(answer_policy.replace("-", "_"))
            )
        if annotation.get("answer_mode") == "multiple_choice" or annotation.get(
            "correct_choice_indices"
        ):
            warnings.append("multiple_choice_metadata_not_represented")
    for field in ("relations", "tracks", "episode"):
        if record.get(field):
            warnings.append("{}_not_represented".format(field))
    for conversation in conversations:
        if not isinstance(conversation, Mapping):
            continue
        for message in conversation.get("messages") or []:
            if not isinstance(message, Mapping):
                continue
            for part in message.get("content") or []:
                if isinstance(part, Mapping) and part.get("type") == "asset" and (
                    part.get("time_span") is not None or part.get("frame_index") is not None
                ):
                    warnings.append("asset_insertion_time_not_represented")
    return warnings


def _split_token_content(
    text: str,
    assets_by_token: Mapping[str, Sequence[str]],
    cursors: Dict[str, int],
    refs: Sequence[str],
    boxes: Sequence[Sequence[float]],
    image_ids: Sequence[int],
    bbox_type: str,
    entities: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    position = 0
    for match in TOKEN_PATTERN.finditer(text):
        if match.start() > position:
            content.append({"type": "text", "text": text[position : match.start()]})
        token = match.group(1)
        if token in MEDIA_KEYS:
            cursor = cursors[token]
            values = assets_by_token[token]
            if cursor >= len(values):
                raise MsSwiftConversionError("More <{}> tokens than media paths".format(token))
            content.append({"type": "asset", "asset_id": values[cursor]})
            cursors[token] += 1
        elif token == "ref-object":
            cursor = cursors[token]
            if cursor >= len(refs):
                raise MsSwiftConversionError("More <ref-object> tokens than objects.ref values")
            identifier = "entity:{}".format(cursor)
            entities.append({"id": identifier, "display_name": str(refs[cursor])})
            content.append({"type": "entity", "entity_id": identifier})
            cursors[token] += 1
        elif token == "bbox":
            cursor = cursors[token]
            if cursor >= len(boxes):
                raise MsSwiftConversionError("More <bbox> tokens than objects.bbox values")
            values = list(boxes[cursor])
            if len(values) not in (2, 4):
                raise MsSwiftConversionError("ms-swift bbox entries must contain two or four coordinates")
            image_index = image_ids[cursor] if cursor < len(image_ids) else 0
            image_assets = assets_by_token["image"]
            if not image_assets or image_index < 0 or image_index >= len(image_assets):
                raise MsSwiftConversionError("objects.image_id points outside images")
            identifier = "region:{}".format(cursor)
            regions.append(
                {
                    "id": identifier,
                    "asset_id": image_assets[image_index],
                    "geometry": {
                        "type": "point2d" if len(values) == 2 else "bbox2d",
                        "format": "xyxy" if len(values) == 4 else "xy",
                        "coordinates": values,
                        "coordinate_space": {"type": "normalized" if bbox_type == "norm1" else "pixel"},
                    },
                }
            )
            content.append({"type": "region", "region_id": identifier})
            cursors[token] += 1
        position = match.end()
    if position < len(text):
        content.append({"type": "text", "text": text[position:]})
    return content


def ms_swift_to_record(
    value: Mapping[str, Any],
    source_id: Optional[str] = None,
    source_dataset: str = "ms-swift",
    preserve_raw: bool = True,
    media_base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages:
        raise MsSwiftConversionError("ms-swift record requires non-empty messages")
    record_id = source_id or str(value.get("id") or value.get("sample_id") or "record")
    assets: List[Dict[str, Any]] = []
    assets_by_token: Dict[str, List[str]] = {"image": [], "video": [], "audio": []}
    media_values: List[str] = []
    for token, key in MEDIA_KEYS.items():
        paths = value.get(key) or []
        if not isinstance(paths, list):
            raise MsSwiftConversionError("{} must be an array".format(key))
        for index, uri in enumerate(paths):
            if not isinstance(uri, str) or not uri.strip():
                raise MsSwiftConversionError("{}[{}] must be a non-empty string".format(key, index))
            resolved_uri = _resolve_media(uri, media_base_dir)
            identifier = "{}:{}".format(token, index)
            assets.append({"id": identifier, "kind": token, "uri": resolved_uri})
            assets_by_token[token].append(identifier)
            media_values.append(resolved_uri)

    objects = value.get("objects") or {}
    if not isinstance(objects, Mapping):
        raise MsSwiftConversionError("objects must be an object")
    refs = objects.get("ref") or []
    boxes = objects.get("bbox") or []
    image_ids = objects.get("image_id") if "image_id" in objects else [0] * len(boxes)
    if not isinstance(refs, list) or any(not isinstance(item, str) for item in refs):
        raise MsSwiftConversionError("objects.ref must be an array of strings")
    if not isinstance(boxes, list):
        raise MsSwiftConversionError("objects.bbox must be an array")
    for index, box in enumerate(boxes):
        if not isinstance(box, list) or len(box) not in (2, 4):
            raise MsSwiftConversionError("objects.bbox[{}] must contain two or four numbers".format(index))
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in box):
            raise MsSwiftConversionError("objects.bbox[{}] contains a non-finite or non-numeric value".format(index))
    if not isinstance(image_ids, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in image_ids):
        raise MsSwiftConversionError("objects.image_id must be an array of integers")
    bbox_type = str(objects.get("bbox_type", "real"))
    if bbox_type not in ("real", "norm1"):
        raise MsSwiftConversionError("objects.bbox_type must be real or norm1")
    if len(image_ids) != len(boxes):
        raise MsSwiftConversionError("objects.image_id length must equal objects.bbox length")

    cursors = {"image": 0, "video": 0, "audio": 0, "ref-object": 0, "bbox": 0}
    entities: List[Dict[str, Any]] = []
    regions: List[Dict[str, Any]] = []
    converted_messages: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise MsSwiftConversionError("messages entries must be objects")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise MsSwiftConversionError("Unsupported ms-swift message role {!r}".format(role))
        if not isinstance(message.get("content"), str):
            raise MsSwiftConversionError("ms-swift message content must be a string")
        converted_messages.append(
            {
                "role": role,
                "content": _split_token_content(
                    message["content"],
                    assets_by_token,
                    cursors,
                    refs,
                    boxes,
                    [int(item) for item in image_ids],
                    bbox_type,
                    entities,
                    regions,
                ),
            }
        )
    expected = {"image": len(assets_by_token["image"]), "video": len(assets_by_token["video"]), "audio": len(assets_by_token["audio"]), "ref-object": len(refs), "bbox": len(boxes)}
    for token, count in expected.items():
        if cursors[token] != count:
            raise MsSwiftConversionError(
                "Unused {} value(s): consumed {}, available {}".format(token, cursors[token], count)
            )

    if bbox_type == "real" and boxes:
        assets_by_id = {asset["id"]: asset for asset in assets}
        for image_index in sorted(set(image_ids)):
            image_asset_ids = assets_by_token["image"]
            if image_index < 0 or image_index >= len(image_asset_ids):
                raise MsSwiftConversionError("objects.image_id points outside images")
            asset = assets_by_id[image_asset_ids[image_index]]
            width, height = _probe_local_image_size(str(asset["uri"]))
            asset["media"] = {"width": width, "height": height}

    group_id = value.get("group_id")
    if not isinstance(group_id, str) or not group_id:
        group_id = _media_group_id(media_values, record_id, media_base_dir)
    result: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "schema_version": CURRENT_SCHEMA_VERSION,
        "id": record_id,
        "group_id": group_id,
        "task_types": ["dialogue"] + (["grounding"] if entities or regions else []),
        "conversations": [{"id": "conversation:0", "messages": converted_messages}],
        "provenance": {
            "dataset": source_dataset,
            "source_record_id": record_id,
            "conversion": {"tool": "s1-udf-ms-swift-adapter", "version": "1.0.0"},
        },
    }
    if assets:
        result["assets"] = assets
    if entities:
        result["entities"] = entities
    if regions:
        result["regions"] = regions
    if preserve_raw:
        result["provenance"]["raw_record"] = dict(value)
    return result
