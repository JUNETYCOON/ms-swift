"""Built-in adapters exposed through the generic adapter registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Tuple

from .adapter import (
    AdapterContext,
    AdapterResult,
    AdapterSpec,
    SourceAdapter,
    TargetAdapter,
    TargetAdapterResult,
    TargetContext,
    register_source_adapter,
    register_target_adapter,
)
from .io import iter_jsonl
from .ms_swift import ms_swift_projection_warnings, ms_swift_to_record, record_to_ms_swift


@register_source_adapter
class MsSwiftSourceAdapter(SourceAdapter):
    spec = AdapterSpec(
        name="ms-swift",
        version="1.0.0",
        direction="source",
        capabilities=("text", "image", "video", "audio", "grounding2d", "multi_turn"),
        loss_contract=(
            "ms-swift does not carry arbitrary S1-UDF relations, tracks, 3D geometry, or episodes",
        ),
        description="Import the documented ms-swift JSONL multimodal/grounding subset.",
    )

    def iter_source(
        self, source: Path, context: AdapterContext
    ) -> Iterator[Tuple[str, Mapping[str, Any]]]:
        for line_number, value in iter_jsonl(source):
            identifier = value.get("id") or value.get("sample_id")
            yield str(identifier or "{}:{}".format(source.name, line_number)), value

    def convert(
        self, source_id: str, value: Mapping[str, Any], context: AdapterContext
    ) -> AdapterResult:
        record = ms_swift_to_record(
            value,
            source_id=source_id,
            source_dataset=context.dataset_name,
            preserve_raw=context.preserve_raw,
            media_base_dir=context.source_root,
        )
        return AdapterResult(records=[record])


@register_target_adapter
class MsSwiftTargetAdapter(TargetAdapter):
    spec = AdapterSpec(
        name="ms-swift",
        version="1.0.0",
        direction="target",
        capabilities=("text", "image", "video", "audio", "grounding2d", "multi_turn"),
        loss_contract=(
            "only selected conversations or caption/qa annotations are rendered",
            "relations, temporal tracks, episodes, masks, and 3D geometry need target-specific renderers",
        ),
        description="Export S1-UDF conversations and the supported annotation subset to ms-swift JSONL.",
    )

    def convert(self, record: Mapping[str, Any], context: TargetContext) -> TargetAdapterResult:
        options = context.options
        answer_policy = str(options.get("answer_policy", "error"))
        annotation_policy = str(options.get("annotation_policy", "error"))
        caption_prompt = options.get("caption_prompt")
        if caption_prompt is not None:
            caption_prompt = str(caption_prompt)
        values = record_to_ms_swift(
            record,
            include_ids=bool(options.get("include_ids", True)),
            base_dir=Path(
                str(options.get("base_dir", context.input_base_dir or context.output_path.parent))
            ),
            answer_policy=answer_policy,
            annotation_policy=annotation_policy,
            caption_prompt=caption_prompt,
        )
        warnings = ms_swift_projection_warnings(record, answer_policy, annotation_policy)
        return TargetAdapterResult(values=values, warnings=warnings)
