#!/usr/bin/env python3
"""Mark leakage-affected benchmark results invalid throughout the static report."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = "audit/data-leakage-findings.json"
INVALID = {
    "ai2d": {
        "display": "AI2D 自建验证集",
        "verdict": "invalid_data_leakage",
        "note": "无效历史结果：eval 79/79 图像及完整答案行进入实际训练入口；须改用严格 _train 后重跑。",
        "task": "DESCRIPTION",
    },
    "robo2vlm": {
        "display": "Robo2VLM 自建验证集",
        "verdict": "invalid_episode_leakage",
        "note": "无效历史结果：去掉 _qN 后 eval 5,239/5,239 个底层 episode 均与 train 重叠；任务为多选 VQA，不是 caption。",
        "task": "MULTIPLE-CHOICE VQA / EMBODIED STATE",
    },
    "vlm-r1-grounding": {
        "display": "VLM-R1 Grounding 自建验证集",
        "verdict": "invalid_image_leakage",
        "note": "不可作为泛化证据：eval 2,825/2,825 张图均在 VQAv2 train，且 2,548 张也在 LLaVA train。",
        "task": "GROUNDING",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_note(existing: str, note: str) -> str:
    existing = existing.strip()
    return note if not existing else existing if note in existing else f"{existing} {note}"


def update_structured() -> None:
    presented_path = ROOT / "presented-results.json"
    presented = load_json(presented_path)
    presented.setdefault("invalid_results", {})
    for slug, correction in INVALID.items():
        item = presented["benchmarks"][slug]
        item["result_status"] = correction["verdict"]
        item["note"] = append_note(item.get("note", ""), correction["note"])
        presented["invalid_results"][slug] = {
            "verdict": correction["verdict"],
            "evidence": EVIDENCE,
            "clean_rerun_required": True,
        }
    write_json(presented_path, presented)

    for filename in ("benchmark-registry.json", "generation-summary.json"):
        path = ROOT / filename
        payload = load_json(path)
        by_slug = {entry["slug"]: entry for entry in payload["benchmarks"]}
        for slug, correction in INVALID.items():
            entry = by_slug[slug]
            entry["status"] = "invalid"
            entry["task_type"] = correction["task"]
            entry["diagnostic_note"] = correction["note"]
            entry["presented_result"] = presented["benchmarks"][slug]
            entry["validity"] = {
                "verdict": correction["verdict"],
                "evidence": EVIDENCE,
                "clean_rerun_required": True,
            }
        write_json(path, payload)

    for slug, correction in INVALID.items():
        path = ROOT / "self-val" / slug / "summary.json"
        summary = load_json(path)
        summary["status"] = "invalid"
        summary["task_type"] = correction["task"]
        summary["data_risk"] = correction["note"]
        summary["presented_result"] = presented["benchmarks"][slug]
        summary["validity"] = {
            "verdict": correction["verdict"],
            "evidence": EVIDENCE,
            "clean_rerun_required": True,
        }
        write_json(path, summary)

    overall_path = ROOT / "overall-summary.json"
    overall = load_json(overall_path)
    registry = load_json(ROOT / "benchmark-registry.json")["benchmarks"]
    counts: dict[str, int] = {}
    for entry in registry:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    overall["complete_count"] = counts.get("complete", 0)
    overall["partial_count"] = counts.get("partial", 0)
    overall["invalid_count"] = counts.get("invalid", 0)
    overall["overall_findings"] = [
        "当前自建 val 中 AI2D、Robo2VLM 与 VLM-R1 Grounding 受训练/评测媒体泄漏影响，历史分数无效或不能作为泛化证据。",
        "其余自建 val 的已提供指标多数提升；Visual Genome Grounding Precision 是局部回归项。",
        "外部 benchmark 中 RealWorldQA、OpenEQA、Flickr30k 与 EgoPlan 提升，Video-MME、OCRBench v2 与 RoboSpatial 回归。",
        "不同 benchmark 的指标和尺度不可平均；只有无泄漏、同口径结果可用于能力结论。",
    ]
    limitation = (
        "AI2D、Robo2VLM、VLM-R1 Grounding 历史结果受泄漏影响；详见 "
        f"{EVIDENCE}，完成干净重跑前不得计入能力结论。"
    )
    overall["important_limitations"] = [
        value for value in overall["important_limitations"] if "AI2D、Robo2VLM、VLM-R1" not in value
    ]
    overall["important_limitations"].append(limitation)
    write_json(overall_path, overall)


def update_table_row(html: str, slug: str, correction: dict[str, str]) -> str:
    pattern = re.compile(r'<tr data-group="self-val".*?</tr>', re.DOTALL)
    match = next(
        (
            candidate
            for candidate in pattern.finditer(html)
            if f"{slug}/report.html" in candidate.group(0)
        ),
        None,
    )
    if not match:
        raise ValueError(f"table row not found: {slug}")
    row = match.group(0)
    row = re.sub(r'data-status="[^"]+"', 'data-status="invalid"', row, count=1)
    row = re.sub(
        r'<span class="pill [^"]+">[^<]+</span>',
        '<span class="pill invalid">invalid</span>',
        row,
        count=1,
    )
    task_cell = re.search(r'</th>\s*<td>.*?</td>', row, re.DOTALL)
    if task_cell:
        row = row[: task_cell.start()] + re.sub(
            r'<td>.*?</td>', f'<td>{correction["task"]}</td>', task_cell.group(0), count=1
        ) + row[task_cell.end() :]
    marker = f"数据有效性：{correction['note']}"
    if marker not in row:
        row = row.replace(
            "</td>\n<td class=\"num\">",
            f"<br><span class='page-note invalid-note'>{marker}</span></td>\n<td class=\"num\">",
            1,
        )
    return html[: match.start()] + row + html[match.end() :]


def update_index(path: Path) -> None:
    html = path.read_text(encoding="utf-8")
    if ".pill.invalid" not in html:
        html = html.replace(
            ".pill.regression{color:var(--red);border-color:#dfaaa4;background:var(--red-bg)}",
            ".pill.regression,.pill.invalid{color:var(--red);border-color:#dfaaa4;background:var(--red-bg)}",
            1,
        )
    for slug, correction in INVALID.items():
        html = update_table_row(html, slug, correction)
    html = html.replace(
        "自建 val 除 Visual Genome Grounding 的 Precision 外，已提供的可比指标整体明显提升；RoboVQA 未提供数值。",
        "AI2D、Robo2VLM 与 VLM-R1 Grounding 历史结果受泄漏影响，不纳入能力结论；其余自建 val 多数指标提升。",
    )
    html = html.replace(
        "GQA、TextVQA、ChartQA、AI2D、VQAv2、LLaVA-Instruct 与 Robo2VLM 的指定主指标均明显提升。",
        "GQA、TextVQA、ChartQA、VQAv2 与 LLaVA-Instruct 的可用指标明显提升。",
    )
    path.write_text(html, encoding="utf-8")


def update_detail(slug: str, correction: dict[str, str]) -> None:
    path = ROOT / "self-val" / slug / "report.html"
    html = path.read_text(encoding="utf-8")
    status_pattern = re.compile(
        r'<section class="status-band"><div class="status-inner"><div class="status-note(?: warning)?">.*?</div></div></section>',
        re.DOTALL,
    )
    warning = (
        '<section class="status-band"><div class="status-inner"><div class="status-note warning">'
        f'<strong>结果无效：</strong>{correction["note"]} '
        f'审计证据：<code>{EVIDENCE}</code></div></div></section>'
    )
    html, count = status_pattern.subn(warning, html, count=1)
    if count != 1:
        raise ValueError(f"status band not found: {path}")
    if slug == "robo2vlm":
        html = html.replace(
            "Robo2VLM 自建验证集 配对评测报告",
            "Robo2VLM 多选 VQA 历史评测报告",
        )
        html = html.replace(
            "Baseline 与 Ours | 用户指定主结果 + 100 条已抽取定性样本",
            "多选 VQA / 具身状态理解 | 泄漏历史结果，仅供排查格式问题",
        )
    path.write_text(html, encoding="utf-8")


def main() -> None:
    evidence = load_json(ROOT / EVIDENCE)
    if set(evidence["findings"]) != {"ai2d", "vlm-r1-grounding", "robo2vlm", "spatialvlm"}:
        raise ValueError("unexpected leakage evidence set")
    update_structured()
    update_index(ROOT / "index.html")
    update_index(ROOT / "self-val" / "index.html")
    for slug, correction in INVALID.items():
        update_detail(slug, correction)
    print("synchronized leakage validity status")


if __name__ == "__main__":
    main()
