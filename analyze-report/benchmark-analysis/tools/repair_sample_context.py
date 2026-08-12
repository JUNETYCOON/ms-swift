#!/usr/bin/env python3
"""Restore source questions, choices, labels, and GT boxes in sample reports."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/mnt/luojunkun/stage1/dataset_ms-swift")
REF_TOKEN = "<ref-object>"
IMAGE_TOKEN_RE = re.compile(r"<\s*image\s*>", re.IGNORECASE)
CHOICE_RE = re.compile(r"^\s*([A-Z])\.\s*(.+?)\s*$")
LAYOUT_STYLE_ID = "repaired-context-layout"
LAYOUT_STYLE = f"""<style id="{LAYOUT_STYLE_ID}">
.sample,.sample-context,.comparison,.model-output{{min-width:0;max-width:100%}}
.sample-header>div:first-child{{display:grid;min-width:0;max-width:100%;grid-template-columns:auto minmax(0,1fr);gap:10px}}
.sample-header strong{{display:block;min-width:0;max-width:100%;white-space:normal;overflow-wrap:anywhere;word-break:break-all}}
figcaption,.question,.options li,.references,.diagnostic,pre{{max-width:100%;overflow-wrap:anywhere;word-break:break-word}}
.compact-meta,.compact-meta div{{min-width:0;max-width:100%}}
.compact-meta div{{overflow:hidden}}
.compact-meta dd{{display:block;width:100%;max-width:100%;white-space:normal!important;overflow-wrap:anywhere;word-break:break-all}}
@media(max-width:520px){{
  .sample-header>div:first-child{{width:100%}}
  .compact-meta{{width:100%}}
  .compact-meta div{{width:100%}}
  .sample-index{{width:auto;margin:0}}
}}
</style>"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=ROOT)
    parser.add_argument(
        "--robo2vlm-source",
        type=Path,
        default=DEFAULT_DATA / "robo2vlm/robo2vlm_test.jsonl",
    )
    parser.add_argument(
        "--robo2vlm-source-label",
        help="Source path recorded in report provenance; defaults to --robo2vlm-source.",
    )
    parser.add_argument(
        "--vlmr1-source",
        type=Path,
        default=DEFAULT_DATA / "vlm-r1/vlm_r1_sft_grounding_msswift_eval.jsonl",
    )
    parser.add_argument(
        "--vlmr1-source-label",
        help="Source path recorded in report provenance; defaults to --vlmr1-source.",
    )
    parser.add_argument(
        "--visualgenome-source",
        type=Path,
        default=DEFAULT_DATA / "visualgenome_grounding/visualgenome_regions_grounding_val.jsonl",
    )
    parser.add_argument(
        "--visualgenome-source-label",
        help="Source path recorded in report provenance; defaults to --visualgenome-source.",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _source_image(record: Mapping[str, Any]) -> str:
    images = [str(value) for value in _values(record.get("images")) if str(value).strip()]
    if len(images) != 1:
        raise ValueError(f"source row must contain exactly one image, found {len(images)}")
    return images[0]


def _normalize_path(value: str) -> str:
    return os.path.normcase(os.path.normpath(value))


def _boxes(record: Mapping[str, Any]) -> list[list[float]]:
    objects = record.get("objects")
    if not isinstance(objects, Mapping):
        return []
    result: list[list[float]] = []
    for value in _values(objects.get("bbox")):
        if not isinstance(value, (list, tuple)) or len(value) not in {2, 4}:
            raise ValueError(f"invalid source bbox: {value!r}")
        result.append([float(item) for item in value])
    return result


def _labels(record: Mapping[str, Any]) -> list[str]:
    objects = record.get("objects")
    if not isinstance(objects, Mapping):
        return []
    return [str(value).strip() for value in _values(objects.get("ref")) if str(value).strip()]


def _result_boxes(sample: Mapping[str, Any]) -> list[list[float]]:
    for side in ("baseline_result", "ours_result"):
        result = sample.get(side)
        if not isinstance(result, Mapping):
            continue
        value = result.get("gt_boxes")
        if not value:
            continue
        parsed = json.loads(value) if isinstance(value, str) else value
        return [[float(item) for item in box] for box in parsed]
    return []


def _boxes_equal(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> bool:
    return len(first) == len(second) and all(
        len(left) == len(right)
        and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(left, right))
        for left, right in zip(first, second)
    )


def _user_content(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("source row has no messages list")
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise ValueError("source row has no user message")


def _assistant_content(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("source row has no messages list")
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise ValueError("source row has no assistant answer")


def source_question(record: Mapping[str, Any]) -> str:
    content = IMAGE_TOKEN_RE.sub("", _user_content(record)).strip()
    for label in _labels(record):
        content = content.replace(REF_TOKEN, label, 1)
    if REF_TOKEN in content:
        raise ValueError("not every <ref-object> token has a source label")
    return content


def parse_multiple_choice(record: Mapping[str, Any]) -> tuple[str, dict[str, str], str]:
    content = IMAGE_TOKEN_RE.sub("", _user_content(record)).strip()
    lines = content.splitlines()
    choice_index = next(
        (index for index, line in enumerate(lines) if line.strip().casefold() == "choices:"),
        None,
    )
    if choice_index is None:
        raise ValueError("Robo2VLM source question has no Choices section")
    question = "\n".join(lines[:choice_index]).strip()
    if question.casefold().startswith("question:"):
        question = question.split(":", 1)[1].strip()
    options: dict[str, str] = {}
    for line in lines[choice_index + 1 :]:
        match = CHOICE_RE.fullmatch(line)
        if not match:
            raise ValueError(f"invalid Robo2VLM choice line: {line!r}")
        options[match.group(1)] = match.group(2)
    if len(options) < 2:
        raise ValueError("Robo2VLM source row has fewer than two choices")
    answer = _assistant_content(record)
    answer_match = CHOICE_RE.fullmatch(answer)
    if not answer_match or answer_match.group(1) not in options:
        raise ValueError(f"Robo2VLM answer does not select an available choice: {answer!r}")
    if answer_match.group(2) != options[answer_match.group(1)]:
        raise ValueError("Robo2VLM answer text does not match the selected choice")
    return question, options, answer


def validate_source(sample: Mapping[str, Any], source: Mapping[str, Any], require_boxes: bool) -> None:
    report_image = str(sample.get("media_source_locator") or "")
    source_image = _source_image(source)
    if _normalize_path(report_image) != _normalize_path(source_image):
        raise ValueError(
            f"sample {sample.get('sample_id')}: image mismatch: {report_image!r} != {source_image!r}"
        )
    if require_boxes:
        source_boxes = _boxes(source)
        result_boxes = _result_boxes(sample)
        if not source_boxes or not result_boxes or not _boxes_equal(source_boxes, result_boxes):
            raise ValueError(
                f"sample {sample.get('sample_id')}: GT bbox mismatch: "
                f"source={source_boxes!r} result={result_boxes!r}"
            )


def load_robo_sources(path: Path, sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        sample_id = str(record.get("id") or "")
        if sample_id in sample_ids:
            if sample_id in result:
                raise ValueError(f"duplicate Robo2VLM source id: {sample_id}")
            result[sample_id] = record
    return result


def load_line_sources(path: Path, line_numbers: set[int]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number not in line_numbers:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            result[str(line_number)] = record
    return result


def load_region_sources(path: Path, region_ids: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        region_id = str(record.get("region_id") or "")
        if region_id in region_ids:
            if region_id in result:
                raise ValueError(f"duplicate Visual Genome region_id: {region_id}")
            result[region_id] = record
    return result


def _grounding_reference(labels: list[str], boxes: list[list[float]]) -> list[str]:
    label_text = "; ".join(labels) if labels else "unlabeled target"
    return [f"{label_text} | GT bbox: {json.dumps(boxes, ensure_ascii=False)}"]


def enrich_grounding_sample(
    sample: dict[str, Any], source: Mapping[str, Any], source_label: str | Path
) -> None:
    validate_source(sample, source, require_boxes=True)
    labels = _labels(source)
    boxes = _boxes(source)
    sample["question"] = source_question(source)
    sample["references"] = _grounding_reference(labels, boxes)
    sample["ground_truth_labels"] = labels
    sample["ground_truth_boxes"] = boxes
    sample["context_source"] = str(source_label)
    sample["context_repair_status"] = "verified"


def enrich_robo_sample(
    sample: dict[str, Any], source: Mapping[str, Any], source_label: str | Path
) -> None:
    validate_source(sample, source, require_boxes=False)
    question, options, answer = parse_multiple_choice(source)
    sample["question"] = question
    sample["options"] = options
    sample["references"] = [answer]
    sample["context_source"] = str(source_label)
    sample["context_repair_status"] = "verified"


def _context_html(sample: Mapping[str, Any]) -> str:
    question = html.escape(str(sample["question"]))
    options = sample.get("options")
    option_html = ""
    if isinstance(options, Mapping) and options:
        option_html = '<ul class="options">' + "".join(
            f"<li><b>{html.escape(str(label))}.</b> {html.escape(str(value))}</li>"
            for label, value in options.items()
        ) + "</ul>"
    references = sample.get("references") or []
    references_html = "\n".join(
        f"[{index}] {html.escape(str(value))}" for index, value in enumerate(references, start=1)
    )
    return (
        '<div class="question"><div class="field-label">问题</div>'
        f"<p>{question}</p>{option_html}"
        '<div class="references"><strong>参考答案：</strong>\n'
        f"{references_html}</div></div>"
    )


def _search_text(sample: Mapping[str, Any]) -> str:
    values = [
        sample.get("sample_id"),
        sample.get("category"),
        sample.get("question"),
        " ".join(str(value) for value in (sample.get("options") or {}).values()),
        " ".join(str(value) for value in sample.get("references") or []),
        sample.get("baseline_response"),
        sample.get("ours_response"),
        " ".join(sample.get("baseline_error_tags") or []),
        " ".join(sample.get("ours_error_tags") or []),
    ]
    return " ".join(str(value) for value in values if value is not None).casefold()


def _article_bounds(document: str, sample_id: str) -> tuple[int, int]:
    header = f"sample_id: {html.escape(sample_id)}</strong>"
    header_positions = [match.start() for match in re.finditer(re.escape(header), document)]
    if not header_positions:
        raise ValueError(f"HTML article not found for sample_id={sample_id}")
    if len(header_positions) != 1:
        raise ValueError(f"HTML contains duplicate articles for sample_id={sample_id}")
    header_position = header_positions[0]
    article_start = document.rfind('<article class="sample"', 0, header_position)
    article_end = document.find("</article>", header_position)
    if article_start < 0 or article_end < 0:
        raise ValueError(f"HTML article bounds not found for sample_id={sample_id}")
    article_end += len("</article>")
    return article_start, article_end


def update_article(document: str, sample: Mapping[str, Any]) -> str:
    sample_id = str(sample["sample_id"])
    article_start, article_end = _article_bounds(document, sample_id)
    article = document[article_start:article_end]
    article, count = re.subn(
        r'data-search=".*?"',
        f'data-search="{html.escape(_search_text(sample), quote=True)}"',
        article,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError(f"data-search not found for sample_id={sample_id}")
    start = article.find('<div class="question">')
    marker = "</div>\n      </div>\n      <div class=\"comparison\">"
    end = article.find(marker, start)
    if start < 0 or end < 0:
        raise ValueError(f"question block not found for sample_id={sample_id}")
    end += len("</div>")
    article = article[:start] + _context_html(sample) + article[end:]
    return document[:article_start] + article + document[article_end:]


def validate_report_context(document: str, samples: Sequence[Mapping[str, Any]]) -> None:
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("paired samples contain duplicate sample_id values")
    article_count = document.count('<article class="sample"')
    if article_count != len(samples):
        raise ValueError(
            f"HTML article count does not match paired samples: {article_count} != {len(samples)}"
        )
    for sample in samples:
        sample_id = str(sample["sample_id"])
        article_start, article_end = _article_bounds(document, sample_id)
        article = document[article_start:article_end]
        if article.count(_context_html(sample)) != 1:
            raise ValueError(f"HTML context does not match paired sample_id={sample_id}")
        expected_search = f'data-search="{html.escape(_search_text(sample), quote=True)}"'
        if article.count(expected_search) != 1:
            raise ValueError(f"HTML search data does not match paired sample_id={sample_id}")


def ensure_layout_style(document: str) -> str:
    pattern = re.compile(
        rf'<style id="{re.escape(LAYOUT_STYLE_ID)}">.*?</style>', re.DOTALL
    )
    matches = list(pattern.finditer(document))
    if len(matches) > 1:
        raise ValueError(f"HTML contains duplicate {LAYOUT_STYLE_ID} styles")
    if matches:
        match = matches[0]
        return document[: match.start()] + LAYOUT_STYLE + document[match.end() :]
    if "</head>" not in document:
        raise ValueError("HTML document has no closing head element")
    return document.replace("</head>", LAYOUT_STYLE + "\n</head>", 1)


def write_report_context(report_dir: Path, samples: list[dict[str, Any]], check_only: bool) -> None:
    report_path = report_dir / "report.html"
    document = report_path.read_text(encoding="utf-8")
    for sample in samples:
        document = update_article(document, sample)
    document = ensure_layout_style(document)
    if "样本上下文已回查" not in document:
        document = document.replace(
            "<main>",
            '<main><div class="provenance"><strong>样本上下文已回查：</strong>'
            "问题、选项、目标标签和 GT bbox 来自源 JSONL，并已核对媒体与结果中的 GT bbox。"
            "模型输出及历史分数未被改写。</div>",
            1,
        )
    validate_report_context(document, samples)
    if not check_only:
        write_jsonl_atomic(report_dir / "paired_samples.jsonl", samples)
        temporary = report_path.with_name(f".{report_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(document, encoding="utf-8")
            os.replace(temporary, report_path)
        finally:
            temporary.unlink(missing_ok=True)


def repair(args: argparse.Namespace) -> dict[str, int]:
    report_root = args.report_root.expanduser().resolve()
    sources = {
        "robo2vlm": args.robo2vlm_source.expanduser().resolve(),
        "vlm-r1-grounding": args.vlmr1_source.expanduser().resolve(),
        "visualgenome-grounding": args.visualgenome_source.expanduser().resolve(),
    }
    source_labels = {
        "robo2vlm": args.robo2vlm_source_label or str(sources["robo2vlm"]),
        "vlm-r1-grounding": args.vlmr1_source_label or str(sources["vlm-r1-grounding"]),
        "visualgenome-grounding": args.visualgenome_source_label
        or str(sources["visualgenome-grounding"]),
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(f"source JSONL does not exist: {path}")

    report_dirs = {slug: report_root / "self-val" / slug for slug in sources}
    samples = {
        slug: load_jsonl(report_dir / "paired_samples.jsonl")
        for slug, report_dir in report_dirs.items()
    }
    robo_ids = {str(sample["sample_id"]) for sample in samples["robo2vlm"]}
    vlmr1_lines = {int(sample["sample_id"]) for sample in samples["vlm-r1-grounding"]}
    region_ids = {
        str(sample["sample_id"]) for sample in samples["visualgenome-grounding"]
    }
    source_rows = {
        "robo2vlm": load_robo_sources(sources["robo2vlm"], robo_ids),
        "vlm-r1-grounding": load_line_sources(sources["vlm-r1-grounding"], vlmr1_lines),
        "visualgenome-grounding": load_region_sources(
            sources["visualgenome-grounding"], region_ids
        ),
    }
    expected = {
        "robo2vlm": robo_ids,
        "vlm-r1-grounding": {str(value) for value in vlmr1_lines},
        "visualgenome-grounding": region_ids,
    }
    for slug, ids in expected.items():
        missing = ids - set(source_rows[slug])
        if missing:
            raise ValueError(f"{slug}: source rows missing for {sorted(missing)[:10]}")

    for sample in samples["robo2vlm"]:
        enrich_robo_sample(
            sample,
            source_rows["robo2vlm"][str(sample["sample_id"])],
            source_labels["robo2vlm"],
        )
    for slug in ("vlm-r1-grounding", "visualgenome-grounding"):
        for sample in samples[slug]:
            enrich_grounding_sample(
                sample, source_rows[slug][str(sample["sample_id"])], source_labels[slug]
            )
    for slug, rows in samples.items():
        write_report_context(report_dirs[slug], rows, args.check_only)
    return {slug: len(rows) for slug, rows in samples.items()}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    counts = repair(args)
    mode = "validated" if args.check_only else "repaired"
    print(f"[{mode}] " + " ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
