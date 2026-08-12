#!/usr/bin/env python3
"""Build navigable, full-record HTML pages for deterministic source samples."""

from __future__ import annotations

import ast
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from source_task_taxonomy import TAXONOMY_VERSION, classify_task


ARTIFACT_FIELDS = [
    "record_archive_path",
    "qa_archive_path",
    "question_archive_path",
    "annotation_archive_path",
    "regions_archive_path",
    "image_data_archive_path",
]

DETAIL_STYLE = """
.detail-media{background:#20282d;display:grid;place-items:center;min-height:260px;padding:12px}
.detail-media img,.detail-media video{display:block;max-width:100%;width:auto;max-height:72vh;object-fit:contain}
.detail-media video{width:min(100%,1080px);background:#000}
.inline-video-actions{display:flex;align-items:center;flex-wrap:wrap;gap:8px}
.inline-video-play{display:inline-flex;align-items:center;min-height:34px;padding:6px 10px;border:1px solid #245a82;border-radius:4px;background:#fff;color:#245a82;cursor:pointer;font:inherit;font-weight:650;letter-spacing:0}
.inline-video-play:hover{background:#eaf2f8}.inline-video-play:disabled{cursor:default;opacity:.6}
.inline-video-status{color:var(--muted,#657078);font-size:12px}.inline-video-status.error{color:var(--red,#963128);font-weight:650}
.sample-nav{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:18px 0}
.sample-nav a{min-width:0;overflow-wrap:anywhere}.sample-nav .next{text-align:right}
.qa-list{display:grid;gap:14px}.qa-item{background:#fff;border:1px solid var(--line);padding:14px;min-width:0}
.qa-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:9px}
.qa-head span{min-width:0;max-width:100%;overflow-wrap:anywhere}
.qa-fields{display:grid;grid-template-columns:110px minmax(0,1fr);gap:7px 12px;margin:0}
.qa-fields dt{color:var(--muted);font-weight:700}.qa-fields dd{margin:0;min-width:0;max-width:100%}
.qa-value,.raw-json{max-width:100%;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;margin:0;font:12px/1.58 Consolas,monospace}
.artifact{min-width:0;max-width:100%;border-top:1px solid var(--line);padding:12px 0}.artifact summary{cursor:pointer;font-weight:700;overflow-wrap:anywhere;word-break:break-word}
.artifact-meta{font-size:12px;color:var(--muted);overflow-wrap:anywhere}
.sample-index{min-width:1240px}.sample-index td:nth-child(1){min-width:150px}.sample-index td:nth-child(4),.sample-index td:nth-child(5){min-width:230px}
.catalog-links{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:6px;margin:14px 0}
.catalog-links a{background:#fff;border:1px solid var(--line);padding:7px 4px;text-align:center;font-size:12px}
@media(max-width:700px){.qa-fields{grid-template-columns:minmax(0,1fr)}.qa-fields dt{margin-top:5px}.catalog-links{grid-template-columns:repeat(5,minmax(0,1fr))}.detail-media{min-height:180px}}
"""

INLINE_SAMPLE_STYLE = DETAIL_STYLE + """
.sample-catalog-summary{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin:14px 0;color:var(--muted)}
.sample-catalog-summary b{color:var(--ink);font-size:20px}.sample-catalog-summary span{display:flex;align-items:baseline;gap:5px}
.sample-jump-panel{margin:14px 0 22px;border-block:1px solid var(--line);background:#fff}
.sample-jump-panel>summary{cursor:pointer;padding:10px;font-weight:700;color:var(--blue)}
.sample-jumps{display:grid;grid-template-columns:repeat(20,minmax(0,1fr));gap:5px;padding:0 10px 12px}
.sample-jumps a{border:1px solid var(--line);background:#fff;padding:6px 2px;text-align:center;font:11px/1.2 Consolas,monospace}
.sample-records{border-top:1px solid var(--line)}
.sample-record{border-bottom:1px solid var(--line);scroll-margin-top:58px}
.sample-record>summary{display:grid;grid-template-columns:64px minmax(180px,1fr) auto auto;align-items:center;gap:12px;padding:13px 10px;cursor:pointer;list-style:none;background:#fff}
.sample-record>summary::-webkit-details-marker{display:none}.sample-record>summary:hover,.sample-record[open]>summary{background:#eef5f2}
.sample-record-number{font:700 12px/1.2 Consolas,monospace;color:var(--accent)}
.sample-record-id{min-width:0;overflow-wrap:anywhere;font-weight:700}.sample-record-chip{font-size:12px;color:var(--muted);white-space:nowrap}
.sample-record-body{display:grid;grid-template-columns:minmax(280px,.85fr) minmax(0,1.35fr);gap:20px;padding:16px 10px 22px;background:#f8faf9}
.sample-record-media,.sample-record-qa{min-width:0}.sample-record-media{align-self:start}
.sample-record-media .detail-media{min-height:190px;padding:8px}.sample-record-media .detail-media img,.sample-record-media .detail-media video{max-height:440px}
.sample-record-meta{font-size:12px;color:var(--muted);overflow-wrap:anywhere;word-break:break-word}.sample-record-meta code{white-space:pre-wrap}
.sample-record .qa-list{gap:0;border-top:1px solid var(--line)}
.sample-record .qa-item{background:transparent;border:0;border-bottom:1px solid var(--line);padding:12px 0}
.sample-record .qa-item:last-child{border-bottom:0}.sample-record .qa-head{margin-bottom:7px}
@media(max-width:1000px){.sample-jumps{grid-template-columns:repeat(10,minmax(0,1fr))}.sample-record-body{grid-template-columns:minmax(0,1fr)}}
@media(max-width:700px){.sample-jumps{grid-template-columns:repeat(5,minmax(0,1fr))}.sample-record>summary{grid-template-columns:48px minmax(0,1fr)}.sample-record-chip{white-space:normal}.sample-record-body{padding:12px 4px 18px}.sample-record-media .detail-media{min-height:160px}}
"""

VIDEO_PLAYBACK_SCRIPT = """<script>
(() => {
  document.addEventListener('click', async event => {
    const button = event.target.closest('.inline-video-play');
    if (!button) return;
    event.preventDefault();
    const record = button.closest('.sample-record') || document;
    const video = record.querySelector('video');
    const status = button.parentElement?.querySelector('.inline-video-status');
    if (!video) {
      if (status) {
        status.textContent = '未找到当前样本的视频播放器';
        status.className = 'inline-video-status error';
      }
      return;
    }
    document.querySelectorAll('video').forEach(other => {
      if (other !== video) other.pause();
    });
    video.preload = 'auto';
    video.scrollIntoView({behavior: 'smooth', block: 'center'});
    if (status) {
      status.textContent = '正在读取本地视频';
      status.className = 'inline-video-status';
    }
    try {
      await video.play();
      if (status) status.textContent = '正在本页播放';
    } catch (error) {
      if (status) {
        const hint = location.protocol === 'file:' ? '；请用 open-local-videos.cmd 打开报告' : '';
        status.textContent = `播放失败：${error?.message || '浏览器不支持该视频'}${hint}`;
        status.className = 'inline-video-status error';
      }
    }
  });
})();
</script>"""


def e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def manifest_for(dataset_dir: Path) -> tuple[Path, list[dict[str, Any]], str]:
    visual = dataset_dir / "sampling-manifest.jsonl"
    rows = load_jsonl(visual)
    if rows:
        return visual, rows, "visual"
    records = dataset_dir / "record-sampling-manifest.jsonl"
    rows = load_jsonl(records)
    if rows:
        return records, rows, "record-only"
    raise RuntimeError(f"no accepted sample manifest: {dataset_dir}")


def safe_artifact(dataset_dir: Path, relative: str) -> Path:
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"unsafe archive path: {relative}")
    path = dataset_dir.joinpath(*relative_path.parts)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def record_artifacts(dataset_dir: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for field in ARTIFACT_FIELDS:
        relative = row.get(field)
        if not relative or relative in seen:
            continue
        seen.add(relative)
        try:
            path = safe_artifact(dataset_dir, str(relative))
        except FileNotFoundError:
            continue
        output.append({
            "field": field,
            "relative_path": str(relative),
            "bytes": path.stat().st_size,
            "value": load_json(path),
        })
    if not output:
        raise RuntimeError(f"{dataset_dir.name}/{row.get('sample_id')}: no archived source record")
    return output


def text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                if item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("type") in {"image", "video", "audio"}:
                    index = item.get("index")
                    parts.append(f"[{item['type']}{'' if index is None else f' #{index}'}]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict) and value.get("text") is not None:
        return str(value["text"])
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def message_rows(messages: Any, task: str = "conversation") -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    output = []
    system = []
    pending_user = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or "").lower()
        content = text_content(message.get("content", message.get("value")))
        if role == "system":
            if content:
                system.append(content)
        elif role in {"user", "human"}:
            pending_user = content
        elif role in {"assistant", "gpt"}:
            output.append({
                "task": task,
                "instruction": "\n\n".join(system),
                "question": pending_user,
                "answer": content or "源记录中的 assistant 内容为空",
                "choices": "",
                "annotation": "",
            })
            pending_user = ""
    if pending_user:
        output.append({
            "task": task,
            "instruction": "\n\n".join(system),
            "question": pending_user,
            "answer": "源记录未提供 assistant/answer",
            "choices": "",
            "annotation": "",
        })
    return output


def first_raw_record(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    for artifact in artifacts:
        value = artifact["value"]
        if isinstance(value, dict) and isinstance(value.get("raw_record"), dict):
            return value["raw_record"]
    for artifact in artifacts:
        if isinstance(artifact["value"], dict):
            return artifact["value"]
    return {}


def parse_choices(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, (list, tuple, dict)) else value
    except (ValueError, SyntaxError):
        return value


def generic_qa(raw: dict[str, Any], task: str) -> list[dict[str, Any]]:
    rows = []
    rows.extend(message_rows(raw.get("messages"), task))
    rows.extend(message_rows(raw.get("conversations"), task))
    if rows:
        return rows
    question = raw.get("Question", raw.get("question", raw.get("query")))
    if question is None:
        return []
    choices = raw.get("NegativeAnswers", raw.get("choices", raw.get("answerTexts", "")))
    choices = parse_choices(choices)
    answer = raw.get("Answer", raw.get("answer", raw.get("multiple_choice_answer")))
    if answer is None and raw.get("answers") is not None:
        answer = raw["answers"]
    if answer is None and raw.get("label") is not None:
        answer = raw["label"]
    if answer is None and raw.get("correct_answer") is not None:
        index = raw["correct_answer"]
        if isinstance(choices, (list, tuple)) and isinstance(index, int) and 0 <= index < len(choices):
            answer = {"index": index, "value": choices[index]}
        else:
            answer = index
    annotation = {
        key: raw[key]
        for key in [
            "question_id", "questionId", "id", "image_id", "imageId",
            "question_type", "answer_type", "Category", "category", "AlignmentType",
        ]
        if key in raw
    }
    if isinstance(raw.get("types"), dict):
        annotation.update({
            "types.structural": raw["types"].get("structural"),
            "types.semantic": raw["types"].get("semantic"),
            "types.detailed": raw["types"].get("detailed"),
        })
    return [{
        "task": task,
        "instruction": raw.get("instruction", raw.get("instruction_processed", "")),
        "question": question,
        "answer": answer if answer is not None else "源记录未提供答案字段",
        "choices": choices,
        "annotation": annotation,
    }]


def supervision_rows(dataset: str, row: dict[str, Any], artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    raw = first_raw_record(artifacts)

    if dataset == "AI2D":
        question_artifact = next((item["value"] for item in artifacts if item["field"] == "question_archive_path"), {})
        for question, payload in (question_artifact.get("questions") or {}).items():
            choices = payload.get("answerTexts") or []
            index = payload.get("correctAnswer")
            answer = choices[index] if isinstance(index, int) and 0 <= index < len(choices) else index
            output.append({
                "task": "diagram multiple-choice QA",
                "instruction": "",
                "question": question,
                "answer": {"index": index, "value": answer},
                "choices": choices,
                "annotation": {key: value for key, value in payload.items() if key not in {"answerTexts", "correctAnswer"}},
            })
    elif dataset == "VisualGenome":
        qa_artifact = next((item["value"] for item in artifacts if item["field"] == "qa_archive_path"), {})
        for qa in qa_artifact.get("qas") or []:
            output.append({
                "task": "Visual Genome QA",
                "instruction": "",
                "question": qa.get("question", ""),
                "answer": qa.get("answer", "源记录未提供答案字段"),
                "choices": "",
                "annotation": {key: value for key, value in qa.items() if key not in {"question", "answer"}},
            })
        if not output:
            regions = next((item["value"] for item in artifacts if item["field"] == "regions_archive_path"), {})
            phrases = [region.get("phrase") for region in regions.get("regions", []) if isinstance(region, dict) and region.get("phrase")]
            output.append({
                "task": "region description (no QA record for this image)",
                "instruction": "",
                "question": "该源图像没有 question_answers 条目",
                "answer": phrases or "源记录没有区域描述",
                "choices": "",
                "annotation": {"region_count": len(regions.get("regions") or [])},
            })
    elif dataset == "RoboVQA":
        source = artifacts[0]["value"] if artifacts else {}
        output.extend(message_rows(source.get("conversations"), "video description conversation"))
        for task_row in (source.get("metadata") or {}).get("task_metadata") or []:
            output.append({
                "task": task_row.get("task", "embodied QA"),
                "instruction": task_row.get("instruction_processed", task_row.get("instruction", "")),
                "question": task_row.get("question_processed", task_row.get("question", "")),
                "answer": task_row.get("answer_processed", task_row.get("answer", "源记录未提供答案字段")),
                "choices": "",
                "annotation": {key: task_row.get(key) for key in ["uid", "split", "video_id"]},
            })
    elif dataset == "COCO":
        output.append({
            "task": "multi-label classification",
            "instruction": "识别图像中的源分类标签",
            "question": "该源记录不是 QA 格式",
            "answer": row.get("label_names", []),
            "choices": "",
            "annotation": {"label_ids": row.get("label_ids", [])},
        })
    elif dataset == "PixMo-Cap":
        output.append({
            "task": "long description",
            "instruction": "根据图像生成详细描述",
            "question": "Describe the image in detail.",
            "answer": raw.get("caption", "源记录未提供 caption"),
            "choices": "",
            "annotation": {"source_transcripts": raw.get("transcripts", [])},
        })
    elif dataset == "PixMo-Points":
        output.append({
            "task": "pointing / counting",
            "instruction": raw.get("collection_method", ""),
            "question": f"Locate: {raw.get('label', '')}",
            "answer": {"label": raw.get("label"), "count": raw.get("count"), "points": raw.get("points")},
            "choices": "",
            "annotation": {
                "image_sha256": raw.get("image_sha256"),
                "collection_method": raw.get("collection_method"),
            },
        })
    elif dataset == "Molmo2-VideoPoint":
        output.append({
            "task": f"video pointing / {raw.get('category', 'unknown')}",
            "instruction": "",
            "question": raw.get("question", ""),
            "answer": {"label": raw.get("label"), "count": raw.get("count"), "points": raw.get("points")},
            "choices": "",
            "annotation": {
                "raw_frames": raw.get("raw_frames"),
                "raw_timestamps": raw.get("raw_timestamps"),
                "two_fps_timestamps": raw.get("two_fps_timestamps"),
                "clip_start": raw.get("clip_start"),
                "clip_end": raw.get("clip_end"),
            },
        })
    elif dataset == "Molmo2-VideoTrack":
        output.append({
            "task": "video tracking",
            "instruction": "Track the referred object through the clip.",
            "question": raw.get("exp", ""),
            "answer": {"object_ids": raw.get("obj_id"), "mask_ids": raw.get("mask_id")},
            "choices": "",
            "annotation": {
                "points": raw.get("points"), "segments": raw.get("segments"),
                "start_frame": raw.get("start_frame"), "end_frame": raw.get("end_frame"),
                "fps": raw.get("fps"), "n_frames": raw.get("n_frames"),
            },
        })
    else:
        output.extend(generic_qa(raw, dataset))

    if not output:
        output.append({
            "task": "source annotation",
            "instruction": "",
            "question": row.get("input_preview", row.get("question", "该源记录不是 QA 格式")),
            "answer": row.get("output_preview", row.get("answer_preview", "完整标注见下方原始 JSON")),
            "choices": "",
            "annotation": raw,
        })

    unique = []
    seen = set()
    for item in output:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def field_html(label: str, value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return f"<dt>{e(label)}</dt><dd><pre class=\"qa-value\">{e(text)}</pre></dd>"


def qa_html(rows: list[dict[str, Any]], id_prefix: str = "qa") -> str:
    cards = []
    for index, row in enumerate(rows, 1):
        taxonomy = row.get("taxonomy") or {}
        fields = "".join([
            field_html("统一任务分类", {
                "simple_task": taxonomy.get("simple_task_label"),
                "task_family": taxonomy.get("task_family"),
                "fine_task": taxonomy.get("fine_task"),
            }),
            field_html("任务详细描述", taxonomy.get("task_description")),
            field_html("Instruction", row.get("instruction")),
            field_html("Question / User", row.get("question")),
            field_html("Answer / Assistant", row.get("answer")),
            field_html("Choices / Negatives", row.get("choices")),
            field_html("Annotation / Metadata", row.get("annotation")),
        ])
        cards.append(
            f'<article class="qa-item" id="{e(id_prefix)}-{index}"><div class="qa-head"><b>#{index}</b>'
            f'<span>{e(taxonomy.get("fine_task", row.get("task", "source supervision")))}</span>'
            f'</div><p class="mini">源任务标签：<code>{e(row.get("task", "source supervision"))}</code></p>'
            f'<dl class="qa-fields">{fields}</dl></article>'
        )
    return "".join(cards)


TASK_UNIT_COLUMNS = [
    "dataset", "sample_id", "qa_index", "source_task_label",
    "simple_task_category", "simple_task_label", "task_family", "fine_task",
    "task_description", "input", "output", "domain", "classification_basis",
    "taxonomy_version", "question_preview",
]


TASK_DISTRIBUTION_COLUMNS = [
    "dataset", "source_task_label", "simple_task_category", "simple_task_label",
    "task_family", "fine_task", "task_description", "input", "output", "domain",
    "classification_basis", "taxonomy_version", "sample_count", "annotation_unit_count",
    "dataset_annotation_units", "within_dataset_percentage", "example_sample_id",
]


def classify_qa_rows(dataset: str, sample_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units = []
    for index, row in enumerate(rows, 1):
        taxonomy = classify_task(
            dataset,
            row.get("task"),
            question=row.get("question"),
            answer=row.get("answer"),
            instruction=row.get("instruction"),
            annotation=row.get("annotation"),
        )
        row["taxonomy"] = taxonomy
        units.append({
            "dataset": dataset,
            "sample_id": sample_id,
            "qa_index": index,
            "source_task_label": row.get("task", "source supervision"),
            **taxonomy,
            "question_preview": short(row.get("question") or row.get("instruction"), 260),
        })
    return units


def aggregate_task_units(units: list[dict[str, Any]], datasets: list[str]) -> list[dict[str, Any]]:
    dataset_totals: dict[str, int] = defaultdict(int)
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    identity_fields = TASK_DISTRIBUTION_COLUMNS[:12]
    for unit in units:
        dataset_totals[unit["dataset"]] += 1
        key = tuple(unit[field] for field in identity_fields)
        if key not in grouped:
            grouped[key] = {
                **{field: unit[field] for field in identity_fields},
                "sample_ids": set(),
                "annotation_unit_count": 0,
                "example_sample_id": unit["sample_id"],
            }
        grouped[key]["sample_ids"].add(unit["sample_id"])
        grouped[key]["annotation_unit_count"] += 1

    order = {dataset: index for index, dataset in enumerate(datasets)}
    rows = []
    for group in grouped.values():
        total = dataset_totals[group["dataset"]]
        rows.append({
            **{field: group[field] for field in identity_fields},
            "sample_count": len(group["sample_ids"]),
            "annotation_unit_count": group["annotation_unit_count"],
            "dataset_annotation_units": total,
            "within_dataset_percentage": group["annotation_unit_count"] / total if total else 0,
            "example_sample_id": group["example_sample_id"],
        })
    rows.sort(key=lambda row: (
        order[row["dataset"]],
        -int(row["annotation_unit_count"]),
        row["task_family"],
        row["fine_task"],
        row["source_task_label"],
    ))
    return rows


def source_locator(row: dict[str, Any]) -> str:
    return str(
        row.get("source_uri")
        or row.get("source_image_path")
        or row.get("source_media_url")
        or row.get("source_member")
        or row.get("member")
        or f"{row.get('shard', row.get('source_shard', ''))}#row={row.get('shard_row', row.get('source_row', ''))}"
    )


def media_html(
    dataset: str,
    row: dict[str, Any],
    base_prefix: str = "../../",
    lazy: bool = False,
) -> tuple[str, str]:
    base = f"{base_prefix}sub-dataset/{quote(dataset)}/"
    video = row.get("source_video_archive_path")
    preview = row.get("preview_archive_path")
    image = row.get("media_archive_path")
    if video:
        video_url = base + quote(str(video), safe="/")
        poster = base + quote(str(preview), safe="/") if preview else ""
        poster_attr = f' poster="{poster}"' if poster else ""
        preload = "none" if lazy else "metadata"
        content = (
            f'<div class="detail-media"><video controls preload="{preload}"{poster_attr}>'
            f'<source src="{video_url}" type="video/mp4">当前浏览器无法播放该视频。</video></div>'
            f'<p class="inline-video-actions"><button type="button" class="inline-video-play" '
            f'data-video-src="{video_url}">在本页播放原视频</button>'
            f'<span class="inline-video-status" aria-live="polite"></span>'
            + (f' · <a href="{poster}">打开派生预览帧</a>' if poster else "")
            + "</p>"
        )
        return content, "archived source video"
    if image:
        image_url = base + quote(str(image), safe="/")
        load_attrs = ' loading="lazy" decoding="async"' if lazy else ""
        return (
            f'<div class="detail-media"><a href="{image_url}"><img src="{image_url}" alt="{e(dataset)} source sample"{load_attrs}></a></div>'
            f'<p><a href="{image_url}">打开归档源图片</a></p>',
            "archived source image",
        )
    if preview:
        preview_url = base + quote(str(preview), safe="/")
        load_attrs = ' loading="lazy" decoding="async"' if lazy else ""
        return (
            f'<div class="detail-media"><a href="{preview_url}"><img src="{preview_url}" alt="{e(dataset)} derived preview"{load_attrs}></a></div>'
            f'<p class="warning"><strong>仅派生预览：</strong>{e(row.get("preview_generation", "源视频未归档"))}。'
            f'<a href="{preview_url}">打开预览帧</a></p>',
            "derived preview only",
        )
    return (
        '<p class="danger"><strong>媒体不可查看：</strong>当前原始源包没有可验证的媒体字节。该页只展示真实源记录，不从转换目录或其他数据集补媒体。</p>',
        "source media unavailable",
    )


def inline_sample_card(
    dataset: str,
    row: dict[str, Any],
    qa_rows: list[dict[str, Any]],
    index: int,
) -> str:
    sample_id = str(row["sample_id"])
    media, media_status = media_html(dataset, row, base_prefix="", lazy=True)
    status_label = {
        "archived source video": "源视频",
        "archived source image": "源图片",
        "derived preview only": "仅派生预览",
        "source media unavailable": "媒体不可用",
    }[media_status]
    split = row.get("split", row.get("task_splits", "n/a"))
    digest = row.get("sha256") or row.get("source_video_sha256") or row.get("preview_sha256") or "not available"
    return (
        f'<details class="sample-record" id="sample-{index:03d}" data-sample-id="{e(sample_id)}">'
        f'<summary><span class="sample-record-number">#{index:03d}</span>'
        f'<span class="sample-record-id">{e(sample_id)}</span>'
        f'<span class="sample-record-chip">{len(qa_rows)} 个 QA / 标注</span>'
        f'<span class="sample-record-chip">{e(status_label)}</span></summary>'
        f'<div class="sample-record-body"><div class="sample-record-media">{media}'
        f'<p class="sample-record-meta"><strong>源 split：</strong>{e(split)}<br>'
        f'<strong>源定位：</strong><code>{e(source_locator(row))}</code><br>'
        f'<strong>媒体 hash：</strong><code>{e(digest)}</code></p></div>'
        f'<div class="sample-record-qa"><h4>QA / 标注详细内容</h4>'
        f'<div class="qa-list">{qa_html(qa_rows, f"sample-{index:03d}-qa")}</div></div></div></details>'
    )


def inline_dataset_samples(
    rows: list[dict[str, Any]],
    cards: list[str],
    qa_count: int,
    source_images: int,
    source_videos: int,
    media_unavailable: int,
) -> str:
    links = "".join(
        f'<a href="#sample-{index:03d}" title="{e(row["sample_id"])}">{index:03d}</a>'
        for index, row in enumerate(rows, 1)
    )
    return (
        '<div class="sample-catalog-summary">'
        f'<span><b>{len(rows)}</b> 抽样记录</span><span><b>{qa_count}</b> QA / 标注单元</span>'
        f'<span><b>{source_images}</b> 图片</span><span><b>{source_videos}</b> 视频</span>'
        f'<span><b>{media_unavailable}</b> 媒体不可用</span></div>'
        f'<details class="sample-jump-panel"><summary>样本编号跳转 · 001-200</summary>'
        f'<div class="sample-jumps" aria-label="抽样记录跳转">{links}</div></details>'
        f'<div class="sample-records">{"".join(cards)}</div>{VIDEO_PLAYBACK_SCRIPT}'
    )


def sample_page(
    dataset: str,
    row: dict[str, Any],
    artifacts: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    previous_id: str | None,
    next_id: str | None,
    style: str,
    analysis_date: str,
) -> str:
    sample_id = str(row["sample_id"])
    media, media_status = media_html(dataset, row)
    previous = f'<a href="{quote(previous_id)}.html">← {e(previous_id)}</a>' if previous_id else "<span>第一条</span>"
    following = f'<a class="next" href="{quote(next_id)}.html">{e(next_id)} →</a>' if next_id else '<span class="next">最后一条</span>'
    artifact_blocks = []
    for index, artifact in enumerate(artifacts):
        raw_url = "../../sub-dataset/" + quote(dataset) + "/" + quote(artifact["relative_path"], safe="/")
        payload = json.dumps(artifact["value"], ensure_ascii=False, indent=2, sort_keys=True)
        open_attr = " open" if index == 0 else ""
        artifact_blocks.append(
            f'<details class="artifact"{open_attr}><summary>{e(artifact["field"])} · {e(artifact["relative_path"])}</summary>'
            f'<p class="artifact-meta">{artifact["bytes"]:,} bytes · <a href="{raw_url}">打开原始归档 JSON</a></p>'
            f'<pre class="raw-json">{e(payload)}</pre></details>'
        )
    manifest_text = json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True)
    digest = row.get("sha256") or row.get("source_video_sha256") or row.get("preview_sha256") or "not available"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(dataset)} · {e(sample_id)} QA 抽检详情</title><style>{style}{DETAIL_STYLE}</style></head>
<body><main data-dataset="{e(dataset)}" data-sample-id="{e(sample_id)}"><header><div class="eyebrow">DETERMINISTIC SOURCE SAMPLE</div><h1>{e(sample_id)}</h1><p class="lede">{e(dataset)} 第 {int(row.get('draw_order', 0)) + 1} 个固定 seed 抽检候选的完整媒体、QA/标注与原始记录。</p></header>
<nav class="nav"><a href="../../index.html">总览</a><a href="../../{quote(dataset)}.html#sample-{quote(sample_id)}">数据集报告</a><a href="index.html">200 条目录</a><a href="#media">媒体</a><a href="#qa">QA/标注</a><a href="#raw">原始 JSON</a><a href="../../source-schemas/{quote(dataset)}.json">Schema</a></nav>
<div class="sample-nav">{previous}{following}</div>
<section class="facts"><div class="fact"><b>{int(row.get('draw_order', 0)) + 1}</b><span>draw order</span></div><div class="fact"><b>{e(row.get('split', row.get('task_splits', 'n/a')))}</b><span>源 split</span></div><div class="fact"><b>{len(qa_rows)}</b><span>结构化 QA/标注单元</span></div><div class="fact"><b>{e(media_status)}</b><span>媒体状态</span></div></section>
<section id="media"><h2>一、图片或视频</h2>{media}<p class="provenance"><strong>源定位：</strong><code>{e(source_locator(row))}</code><br><strong>媒体 hash：</strong><code>{e(digest)}</code></p></section>
<section id="qa"><h2>二、QA / 标注详细内容</h2><div class="qa-list">{qa_html(qa_rows)}</div></section>
<section id="raw"><h2>三、完整原始记录</h2><p>以下内容按归档文件完整打印，没有页面摘要截断；媒体字节只以路径、大小和 hash 表示。</p>{''.join(artifact_blocks)}</section>
<section id="manifest"><h2>四、抽样 manifest</h2><pre class="schema">{e(manifest_text)}</pre><p><a href="../../source-schemas/{quote(dataset)}.json">打开 {e(dataset)} 完整 source schema</a></p></section>
<div class="sample-nav">{previous}{following}</div><footer>生成时间 {e(analysis_date)}；证据仅来自服务器原始源目录。</footer></main>{VIDEO_PLAYBACK_SCRIPT}</body></html>'''


def short(value: Any, limit: int = 180) -> str:
    text = text_content(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def input_output_preview(row: dict[str, Any], qa_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if qa_rows:
        return short(qa_rows[0].get("question") or qa_rows[0].get("instruction")), short(qa_rows[0].get("answer"))
    return short(row.get("input_preview", row.get("question", ""))), short(row.get("output_preview", row.get("answer_preview", "")))


def dataset_catalog(
    dataset: str,
    rows: list[dict[str, Any]],
    qa_counts: list[int],
    media_statuses: list[str],
    summary: dict[str, Any],
    schema: Any,
    style: str,
    analysis_date: str,
) -> str:
    table_rows = []
    links = []
    for index, (row, qa_count, media_status) in enumerate(zip(rows, qa_counts, media_statuses), 1):
        sample_id = str(row["sample_id"])
        detail = f"{quote(sample_id)}.html"
        links.append(f'<a href="{detail}" title="{e(sample_id)}">{index:03d}</a>')
        input_text = row.get("input_preview", row.get("question", row.get("first_user", row.get("first_question", ""))))
        output_text = row.get("output_preview", row.get("answer_preview", row.get("first_assistant", row.get("first_answer", ""))))
        table_rows.append(
            f'<tr><td><a href="{detail}">{e(sample_id)}</a></td><td>{int(row.get("draw_order", index - 1))}</td>'
            f'<td>{qa_count}</td><td>{e(short(input_text))}</td><td>{e(short(output_text))}</td>'
            f'<td>{e(media_status)}</td><td>{e(short(source_locator(row), 220))}</td></tr>'
        )
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(dataset)} · 200 条随机抽检目录</title><style>{style}{DETAIL_STYLE}</style></head><body><main>
<header><div class="eyebrow">FIXED-SEED RANDOM SOURCE INSPECTION</div><h1>{e(dataset)} · 200 条抽检</h1><p class="lede">从完整源总体按固定 seed 无放回随机排列得到；每条链接进入完整 QA、标注、媒体和原始 JSON。</p></header>
<nav class="nav"><a href="../index.html">全部数据集</a><a href="../../index.html">总报告</a><a href="../../{quote(dataset)}.html">数据集报告</a><a href="#samples">200 条目录</a><a href="#schema">Schema</a></nav>
<section class="facts"><div class="fact"><b>{len(rows)}</b><span>抽检记录</span></div><div class="fact"><b>{e(summary.get('seed', 'n/a'))}</b><span>固定 seed</span></div><div class="fact"><b>{e(summary.get('sampling_unit', 'n/a'))}</b><span>抽样单位</span></div><div class="fact"><b>{e(summary.get('population_total', 'n/a'))}</b><span>完整总体</span></div></section>
<section id="samples"><h2>一、样本跳转</h2><div class="catalog-links">{''.join(links)}</div><div class="table-wrap"><table class="sample-index"><thead><tr><th>样本</th><th>draw</th><th>QA/标注数</th><th>输入</th><th>输出</th><th>媒体</th><th>源定位</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></section>
<section id="schema"><h2>二、完整 source schema</h2><p><a href="../../source-schemas/{quote(dataset)}.json">打开独立 schema JSON</a>。下方完整打印实际字段、类型、结构变体和代表源记录。</p><pre class="schema">{e(schema_text)}</pre></section>
<footer>生成时间 {e(analysis_date)}；原始服务器源只读。</footer></main></body></html>'''


def global_catalog(dataset_summaries: list[dict[str, Any]], style: str, analysis_date: str) -> str:
    rows = []
    for item in dataset_summaries:
        dataset = item["dataset"]
        rows.append(
            f'<tr><td><a href="{quote(dataset)}/index.html">{e(dataset)}</a></td><td>{item["sample_count"]}</td>'
            f'<td>{e(item["sample_mode"])}</td><td>{item["qa_annotation_units"]}</td><td>{item["source_images"]}</td>'
            f'<td>{item["source_videos"]}</td><td>{item["media_unavailable"]}</td>'
            f'<td><a href="../source-schemas/{quote(dataset)}.json">schema</a> · <a href="../{quote(dataset)}.html">report</a></td></tr>'
        )
    total = sum(item["sample_count"] for item in dataset_summaries)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLM 原始数据 3600 条随机抽检</title><style>{style}{DETAIL_STYLE}</style></head><body><main>
<header><div class="eyebrow">SOURCE-ONLY SAMPLE CATALOG</div><h1>VLM 原始数据随机抽检</h1><p class="lede">18 个数据集，每集 200 条；逐条查看完整 QA/标注、schema、图片或视频及源定位。</p></header>
<nav class="nav"><a href="../index.html">分析总报告</a><a href="#datasets">数据集目录</a><a href="../methodology.md">方法</a></nav>
<section class="facts"><div class="fact"><b>{len(dataset_summaries)}</b><span>数据集</span></div><div class="fact"><b>{total}</b><span>详情页</span></div><div class="fact"><b>{sum(item['source_images'] for item in dataset_summaries)}</b><span>源图片</span></div><div class="fact"><b>{sum(item['source_videos'] for item in dataset_summaries)}</b><span>归档源视频</span></div></section>
<section id="datasets"><h2>数据集目录</h2><div class="table-wrap"><table><thead><tr><th>数据集</th><th>抽检</th><th>模式</th><th>QA/标注单元</th><th>图片</th><th>视频</th><th>缺媒体</th><th>其他</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<footer>生成时间 {e(analysis_date)}；不使用转换后数据作为证据。</footer></main></body></html>'''


def build_sample_details(
    root: Path,
    sub_root: Path,
    datasets: list[str],
    style: str,
    analysis_date: str,
    include_report_sections: bool = False,
) -> dict[str, Any]:
    summaries = []
    task_units: list[dict[str, Any]] = []
    report_sections: dict[str, str] = {}
    for dataset in datasets:
        dataset_dir = sub_root / dataset
        _, rows, sample_mode = manifest_for(dataset_dir)
        if len(rows) != 200:
            raise RuntimeError(f"{dataset}: expected 200 sample records, got {len(rows)}")
        qa_counts = []
        media_statuses = []
        source_images = 0
        source_videos = 0
        media_unavailable = 0
        report_cards = []
        for index, row in enumerate(rows, 1):
            artifacts = record_artifacts(dataset_dir, row)
            qa_rows = supervision_rows(dataset, row, artifacts)
            task_units.extend(classify_qa_rows(dataset, str(row["sample_id"]), qa_rows))
            qa_counts.append(len(qa_rows))
            if row.get("source_video_archive_path"):
                media_statuses.append("source video")
                source_videos += 1
            elif row.get("media_archive_path"):
                media_statuses.append("source image")
                source_images += 1
            elif row.get("preview_archive_path"):
                media_statuses.append("derived preview only")
            else:
                media_statuses.append("source media unavailable")
                media_unavailable += 1
            if include_report_sections:
                report_cards.append(inline_sample_card(dataset, row, qa_rows, index))
        if include_report_sections:
            report_sections[dataset] = inline_dataset_samples(
                rows,
                report_cards,
                sum(qa_counts),
                source_images,
                source_videos,
                media_unavailable,
            )
        summaries.append({
            "dataset": dataset,
            "sample_count": len(rows),
            "sample_mode": sample_mode,
            "qa_annotation_units": sum(qa_counts),
            "source_images": source_images,
            "source_videos": source_videos,
            "derived_preview_only": sum(status == "derived preview only" for status in media_statuses),
            "media_unavailable": media_unavailable,
        })
    task_distribution = aggregate_task_units(task_units, datasets)
    atomic_csv(root / "sample-task-annotations.csv", task_units, TASK_UNIT_COLUMNS)
    atomic_csv(root / "sample-task-distribution.csv", task_distribution, TASK_DISTRIBUTION_COLUMNS)
    delivery_digest = hashlib.sha256()
    for path in (root / "sample-task-annotations.csv", root / "sample-task-distribution.csv"):
        delivery_digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes())
    result = {
        "version": 1,
        "analysis_date": analysis_date,
        "source_only": True,
        "datasets": summaries,
        "dataset_count": len(summaries),
        "sample_records": sum(item["sample_count"] for item in summaries),
        "sample_detail_pages": 0,
        "embedded_sample_records": sum(item["sample_count"] for item in summaries),
        "qa_annotation_units": sum(item["qa_annotation_units"] for item in summaries),
        "classified_task_units": len(task_units),
        "task_distribution_rows": len(task_distribution),
        "task_taxonomy_version": TAXONOMY_VERSION,
        "delivery_digest": delivery_digest.hexdigest(),
    }
    if include_report_sections:
        result["_report_sections"] = report_sections
    return result


def main() -> int:
    from build_source_analysis_report import ANALYSIS_DATE, DATASET_ORDER, ROOT, STYLE, SUB

    result = build_sample_details(ROOT, SUB, DATASET_ORDER, STYLE, ANALYSIS_DATE)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
