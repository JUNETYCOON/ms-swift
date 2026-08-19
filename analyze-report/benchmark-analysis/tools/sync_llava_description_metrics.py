#!/usr/bin/env python3
"""Synchronize the audited LLaVA description metrics into report metadata/pages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "audit" / "llava-description-paired-metrics.json"
SLUG = "llava-instruct"
LABELS = (("ROUGE-L", "rouge_l"), ("BLEU-4", "bleu_4"), ("ChrF", "chrf"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def metric_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, key in LABELS:
        metric = audit["metrics"][key]
        baseline = metric["baseline"]
        ours = metric["ours"]
        delta = metric["delta"]
        rows.append(
            {
                "label": label,
                "baseline": baseline,
                "ours": ours,
                "baseline_display": f"{baseline:.6f}",
                "ours_display": f"{ours:.6f}",
                "delta": delta,
                "delta_display": f"{delta:+.6f}",
            }
        )
    return rows


def presented_result(audit: dict[str, Any]) -> dict[str, Any]:
    count = audit["shared_task_rows"]
    return {
        "group": "self-val",
        "display_name": "LLaVA-Instruct 自建验证集",
        "source": "strict_shared_description_subset",
        "source_display": f"严格共同 Description 子集（{count} 条）",
        "metrics": metric_rows(audit),
        "result_status": "provided",
        "note": (
            "Exact Match/VQA Accuracy 不适用于 description，已移除；"
            "CIDEr 当前评测器未产出。"
        ),
    }


def update_registry_entry(entry: dict[str, Any], audit: dict[str, Any]) -> None:
    metrics = metric_rows(audit)
    entry.update(
        {
            "task_type": "DESCRIPTION",
            "metric_label": "\n".join(item["label"] for item in metrics),
            "baseline": [item["baseline"] for item in metrics],
            "ours": [item["ours"] for item in metrics],
            "delta": [item["delta"] for item in metrics],
            "baseline_display": "\n".join(item["baseline_display"] for item in metrics),
            "ours_display": "\n".join(item["ours_display"] for item in metrics),
            "delta_display": "\n".join(item["delta_display"] for item in metrics),
            "metric_scope": "strict_shared_description_subset",
            "scope_tier": "strict_shared_description_subset",
            "diagnostic_note": (
                "仅比较共同 1103 条 description；两个 partial run 的其余行不进入主结果。"
            ),
            "presented_result": presented_result(audit),
        }
    )


def update_structured_files(audit: dict[str, Any]) -> None:
    presented_path = ROOT / "presented-results.json"
    presented = load_json(presented_path)
    presented["computed_aggregates_rendered"] = True
    presented.setdefault("corrections", {})[SLUG] = {
        "reason": "description tasks require generation metrics, not exact_match/vqa_accuracy",
        "evidence": "audit/llava-description-paired-metrics.json",
    }
    presented["benchmarks"][SLUG] = presented_result(audit)
    write_json(presented_path, presented)

    for name in ("benchmark-registry.json", "generation-summary.json"):
        path = ROOT / name
        payload = load_json(path)
        matches = [entry for entry in payload["benchmarks"] if entry.get("slug") == SLUG]
        if len(matches) != 1:
            raise ValueError(f"expected one {SLUG} entry in {path}, found {len(matches)}")
        update_registry_entry(matches[0], audit)
        write_json(path, payload)

    overall_path = ROOT / "overall-summary.json"
    overall = load_json(overall_path)
    overall["presentation_policy"]["headline_source"] = "provided_or_strictly_paired_task_metrics"
    overall["presentation_policy"]["computed_aggregates_rendered"] = True
    limitation = (
        "LLaVA-Instruct 仅报告共同 1103 条 description 的 ROUGE-L、BLEU-4、ChrF；"
        "CIDEr 未产出，Exact Match/VQA Accuracy 不用于 description。"
    )
    if limitation not in overall["important_limitations"]:
        overall["important_limitations"].append(limitation)
    write_json(overall_path, overall)


def table_row(href: str, audit: dict[str, Any]) -> str:
    metrics = metric_rows(audit)
    labels = "\n".join(item["label"] for item in metrics)
    baseline = "\n".join(item["baseline_display"] for item in metrics)
    ours = "\n".join(item["ours_display"] for item in metrics)
    delta = "\n".join(item["delta_display"] for item in metrics)
    search = (
        "llava-instruct 自建验证集 description "
        f"{labels} {baseline} {ours} strict_shared_description_subset partial"
    )
    return f'''<tr data-group="self-val" data-status="partial" data-search="{search}">
<th><a href="{href}">LLaVA-Instruct 自建验证集</a><br><span class="pill partial">partial</span></th>
<td>DESCRIPTION</td><td class="scope"><strong>{labels}</strong><br><span class="muted">严格共同 Description 子集（1103 条） | strict_shared_description_subset</span><br><span class='page-note'>Exact Match/VQA Accuracy 已移除；CIDEr 当前评测器未产出。</span></td>
<td class="num">100</td><td class="num">{baseline}</td><td class="num">{ours}</td>
<td class="num delta pos">{delta}</td></tr>'''


def replace_llava_row(path: Path, audit: dict[str, Any]) -> None:
    html = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'<tr data-group="self-val" data-status="partial" data-search="llava-instruct.*?</tr>',
        re.DOTALL,
    )
    match = pattern.search(html)
    if not match:
        raise ValueError(f"LLaVA table row not found in {path}")
    href_match = re.search(r'<a href="([^"]+)">LLaVA-Instruct', match.group(0))
    if not href_match:
        raise ValueError(f"LLaVA href not found in {path}")
    updated, count = pattern.subn(table_row(href_match.group(1), audit), html, count=1)
    if count != 1:
        raise ValueError(f"expected one LLaVA row replacement in {path}, got {count}")
    path.write_text(updated, encoding="utf-8")


def update_detail_page(audit: dict[str, Any]) -> None:
    path = ROOT / "self-val" / SLUG / "report.html"
    html = path.read_text(encoding="utf-8")
    metrics = metric_rows(audit)
    body = "".join(
        f"<tr><th>{item['label']}</th><td class='num'>{item['baseline_display']}</td>"
        f"<td class='num'>{item['ours_display']}</td>"
        f"<td class='num delta'>{item['delta_display']}</td></tr>"
        for item in metrics
    )
    summary = (
        '<section class="summary-band"><div class="summary-inner">'
        '<h2>严格共同 Description 子集（1103 条）</h2>'
        '<div class="table-wrap"><table class="presented-results"><thead><tr>'
        '<th>描述指标</th><th class="num">Baseline</th><th class="num">Ours</th>'
        '<th class="num">变化（Ours - Baseline）</th></tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
        '<div class="provenance"><strong>展示口径：</strong><br>'
        '仅统计 Baseline 与 Ours 共同 sample_id、且问题和参考答案一致的 1103 条 description。<br>'
        'Exact Match/VQA Accuracy 不适用于描述任务，已从主结果移除；CIDEr 当前评测器未产出。<br>'
        '下方 100 条混合任务样本只用于定性浏览，不产生主分数。'
        '</div></div></section>'
    )
    pattern = re.compile(r'<section class="summary-band">.*?</section>', re.DOTALL)
    html, count = pattern.subn(summary, html, count=1)
    if count != 1:
        raise ValueError(f"expected one summary band in {path}, got {count}")
    html = html.replace(
        "Baseline 与 Ours | 用户指定主结果 + 100 条已抽取定性样本",
        "Baseline 与 Ours | 严格共同 description 指标 + 100 条混合任务定性样本",
        1,
    )
    html = html.replace(
        "<strong>页面用途：</strong>主结果采用用户指定结果表；下方 100 条已抽取样本仅用于定性分析。",
        "<strong>页面用途：</strong>主结果采用严格共同 description 子集；下方混合任务样本仅用于定性分析。",
        1,
    )
    path.write_text(html, encoding="utf-8")


def main() -> None:
    audit = load_json(AUDIT_PATH)
    if audit.get("shared_task_rows") != 1103 or audit.get("task") != "description":
        raise ValueError("unexpected LLaVA description audit population")
    update_structured_files(audit)
    replace_llava_row(ROOT / "index.html", audit)
    replace_llava_row(ROOT / "self-val" / "index.html", audit)
    update_detail_page(audit)
    print("synchronized LLaVA description metrics")


if __name__ == "__main__":
    main()
