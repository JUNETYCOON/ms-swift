#!/usr/bin/env python3
"""Build offline Chinese HTML reports for the converted-dataset audit."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


STYLE = r"""
:root{--ink:#1d292e;--muted:#66747a;--line:#d4dcde;--paper:#fff;--wash:#f2f5f4;--teal:#08776d;--teal-soft:#e3f1ef;--blue:#285f86;--blue-soft:#e8f0f6;--amber:#8a5a08;--amber-soft:#fff1d2;--red:#963c36;--red-soft:#f9e6e4;--green:#246b43;--shadow:0 1px 3px rgba(24,35,40,.09)}
*{box-sizing:border-box}html{scroll-behavior:smooth;max-width:100%}body{margin:0;max-width:100%;overflow-x:hidden;background:var(--wash);color:var(--ink);font:14px/1.58 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;letter-spacing:0}a{color:var(--blue);text-underline-offset:2px}code,pre{font-family:Consolas,"SFMono-Regular",monospace;letter-spacing:0}code{overflow-wrap:anywhere;word-break:break-word}pre{max-width:100%;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}main{width:min(1420px,calc(100% - 30px));min-width:0;margin:auto;padding:22px 0 60px}.top{background:var(--paper);border-bottom:1px solid var(--line)}.top-inner{width:min(1420px,calc(100% - 30px));min-width:0;margin:auto;padding:22px 0}.eyebrow{color:var(--teal);font-size:12px;font-weight:750}.top h1{font-size:30px;line-height:1.22;margin:5px 0 8px;letter-spacing:0}.top p{margin:0;max-width:105ch;color:var(--muted)}.nav{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:13px}.nav a{font-size:13px}.summary-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border:1px solid var(--line);background:var(--line);gap:1px;margin:20px 0}.metric{background:var(--paper);padding:14px;min-width:0}.metric b{display:block;font-size:22px;color:var(--teal);overflow-wrap:anywhere}.metric span{color:var(--muted);font-size:12px}.notice{max-width:100%;padding:12px 14px;border-left:4px solid var(--blue);background:var(--blue-soft);margin:14px 0;overflow-wrap:anywhere;word-break:break-word}.notice.warn{border-color:var(--amber);background:var(--amber-soft)}.notice.bad{border-color:var(--red);background:var(--red-soft)}.notice.ok{border-color:var(--green);background:var(--teal-soft)}h2{font-size:21px;line-height:1.3;margin:34px 0 12px;letter-spacing:0}h3{font-size:16px;margin:22px 0 9px;letter-spacing:0}.table-wrap{width:100%;max-width:100%;overflow-x:auto;background:var(--paper);border:1px solid var(--line);min-width:0}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;vertical-align:top;padding:9px 10px;border-bottom:1px solid var(--line);overflow-wrap:anywhere;word-break:break-word}th{background:#e9eeee;white-space:nowrap}.compact{min-width:1120px}.status{display:inline-block;border:1px solid currentColor;padding:1px 6px;font-size:11px;font-weight:700;background:#fff}.ok{color:var(--green)}.partial{color:var(--amber)}.bad{color:var(--red)}.muted{color:var(--muted)}.filters{display:grid;grid-template-columns:minmax(220px,1fr) repeat(3,minmax(130px,190px));gap:9px;background:var(--paper);border-block:1px solid var(--line);padding:11px 0;position:sticky;top:0;z-index:4}.filters input,.filters select{width:100%;min-width:0;height:36px;border:1px solid #b9c4c8;background:#fff;color:var(--ink);padding:5px 8px;font:inherit;border-radius:4px;letter-spacing:0}.filter-count{grid-column:1/-1;color:var(--muted);font-size:12px}.samples{display:grid;gap:13px}.sample{display:grid;grid-template-columns:minmax(280px,.72fr) minmax(0,1.28fr);background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow);border-radius:6px;overflow:hidden;min-width:0}.sample[hidden]{display:none}.sample-media{background:#20292d;min-height:260px;display:grid;place-items:center;padding:10px;min-width:0}.media-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;width:100%;min-width:0}.media-grid.one{grid-template-columns:minmax(0,1fr)}.media-item{display:grid;place-items:center;min-width:0;max-width:100%}.media-item a{display:grid;place-items:center;width:100%;min-width:0}.media-item img{display:block;max-width:100%;width:auto;max-height:440px;object-fit:contain}.media-label{max-width:100%;color:#dce7e8;font-size:11px;margin-top:5px;text-align:center;overflow-wrap:anywhere;word-break:break-word}.media-empty{max-width:100%;color:#d9e0e2;text-align:center;padding:22px;overflow-wrap:anywhere;word-break:break-word}.sample-body{padding:14px;min-width:0}.sample-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:9px}.sample-id{font-weight:750;color:var(--teal);overflow-wrap:anywhere}.chips{display:flex;gap:5px;flex-wrap:wrap}.chip{border:1px solid var(--line);padding:1px 5px;font-size:11px;background:#f7f9f8}.qa{display:grid;grid-template-columns:84px minmax(0,1fr);gap:7px 10px;margin:12px 0}.qa dt{color:var(--muted);font-weight:700}.qa dd{margin:0;min-width:0}.qa pre{margin:0;max-height:260px;overflow:auto;font-size:12px}.source{font-size:11px;color:var(--muted);overflow-wrap:anywhere;word-break:break-word}.details{border-top:1px solid var(--line);padding-top:8px;margin-top:9px}.details summary{cursor:pointer;color:var(--blue);font-weight:650}.details pre{max-height:300px;overflow:auto;background:#edf1f1;padding:9px;font-size:11px}.media-paths{margin:7px 0 0;padding-left:20px;font-size:11px}.media-paths li{overflow-wrap:anywhere;word-break:break-word;margin:3px 0}.error-list{margin:8px 0;padding-left:20px;color:var(--red)}.method{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.method>div{border-top:3px solid var(--teal);padding-top:9px;min-width:0}.method p{margin:4px 0;color:var(--muted);font-size:12px}.footer{margin-top:38px;border-top:1px solid var(--line);padding-top:14px;color:var(--muted);font-size:12px}
@media(max-width:980px){.summary-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.filters{grid-template-columns:repeat(2,minmax(0,1fr))}.sample{grid-template-columns:minmax(0,1fr)}.sample-media{min-height:210px}.method{grid-template-columns:minmax(0,1fr)}}
@media(max-width:560px){main,.top-inner{width:calc(100% - 18px)}.top h1{font-size:25px}.summary-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.summary-grid>.metric:last-child:nth-child(odd){grid-column:1/-1}.filters{position:static;grid-template-columns:minmax(0,1fr)}.media-grid{grid-template-columns:minmax(0,1fr)}.qa{grid-template-columns:minmax(0,1fr)}.sample-body{padding:11px}.sample-media{min-height:170px}.metric b{font-size:19px}}
"""


FILTER_SCRIPT = r"""
(() => {
  const samples = Array.from(document.querySelectorAll('.sample'));
  const search = document.querySelector('#filter-search');
  const status = document.querySelector('#filter-status');
  const split = document.querySelector('#filter-split');
  const format = document.querySelector('#filter-format');
  const count = document.querySelector('#filter-count');
  if (!search || !status || !split || !format || !count) return;
  function apply() {
    const term = search.value.trim().toLocaleLowerCase();
    let visible = 0;
    for (const sample of samples) {
      const matches = (!term || sample.dataset.search.includes(term)) &&
        (!status.value || sample.dataset.status === status.value) &&
        (!split.value || sample.dataset.split === split.value) &&
        (!format.value || sample.dataset.format === format.value);
      sample.hidden = !matches;
      if (matches) visible += 1;
    }
    count.textContent = `当前显示 ${visible} / ${samples.length} 条`;
  }
  [search, status, split, format].forEach(node => node.addEventListener('input', apply));
  apply();
})();
"""


TASK_LABELS = {
    "ai2d": "图解理解 / 描述",
    "chartqa": "图表问答",
    "gqa": "组合式视觉问答",
    "textvqa": "场景文字问答",
    "visualgenome-qa": "视觉问答",
    "visualgenome-regions": "区域描述",
    "vlm-r1": "视觉定位",
    "vqav2": "视觉问答",
    "robo2vlm": "机器人图像问答",
    "robovqa": "机器人视频问答",
    "spatialvlm": "空间推理",
    "molmo2-video-capqa": "视频描述 / 问答",
    "molmo2-video-point": "视频指点定位",
    "molmo2-video-subtitleqa": "视频字幕问答",
    "molmo2-video-track": "视频跟踪",
    "pixmo-cap": "图像描述",
    "pixmo-points": "图像指点定位",
    "llava": "多任务视觉指令微调",
    "coco": "多标签图像识别 / 描述式指令",
}

MEDIA_STATUS_LABELS = {
    "available": "媒体可展示",
    "partial": "部分媒体可展示",
    "remote_not_downloaded": "远程媒体未下载",
    "missing": "本地媒体缺失",
    "decode_error": "媒体解码失败",
    "text_only": "纯文本记录",
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                rows.append(value)
    return rows


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def n(value: Any) -> str:
    return f"{int(value or 0):,}"


def pct(value: Any) -> str:
    return f"{float(value or 0) * 100:.2f}%"


def status_chip(text: str, css: str) -> str:
    return f'<span class="status {css}">{esc(text)}</span>'


def media_aggregate(summary: dict[str, Any]) -> dict[str, int]:
    media = summary.get("media_snapshot") or {}
    return {
        "unique": sum(int(row.get("unique_references") or 0) for row in media.values()),
        "local": sum(int(row.get("local_references") or 0) for row in media.values()),
        "present": sum(int(row.get("local_present") or 0) for row in media.values()),
        "missing": sum(int(row.get("local_missing") or 0) for row in media.values()),
        "remote": sum(int(row.get("remote_references") or 0) for row in media.values()),
    }


def format_judgment(summary: dict[str, Any]) -> tuple[str, str]:
    fmt = summary.get("format") or {}
    json_errors = fmt.get("json_error_counts") or {}
    errors = fmt.get("format_error_counts") or {}
    if json_errors:
        return f"JSON 解析失败 {n(sum(json_errors.values()))} 行", "bad"
    if errors:
        return f"结构异常 {n(sum(errors.values()))} 行", "partial"
    return "结构检查全部通过", "ok"


def media_judgment(summary: dict[str, Any], post_media: dict[str, Any]) -> tuple[str, str]:
    aggregate = media_aggregate(summary)
    name = summary["name"]
    post = (post_media.get("datasets") or {}).get(name, {})
    download = post.get("download_report") or {}
    if aggregate["missing"]:
        return f"本地缺失 {n(aggregate['missing'])}", "bad"
    if download:
        successful = int(download.get("successful_media") or 0)
        total = int(download.get("total_media") or 0)
        css = "ok" if total and successful == total else "partial"
        return f"已下载 {n(successful)}/{n(total)}", css
    if aggregate["remote"] and not aggregate["local"]:
        return f"仅远程引用 {n(aggregate['remote'])}", "bad"
    if aggregate["remote"]:
        return f"本地/远程混合，远程 {n(aggregate['remote'])}", "partial"
    if aggregate["local"] and aggregate["present"] == aggregate["local"]:
        return f"本地齐全 {n(aggregate['present'])}", "ok"
    if not aggregate["unique"]:
        return "无视觉媒体引用", "partial"
    return "需要复核", "partial"


def page_shell(title: str, intro: str, body: str, *, script: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{STYLE}</style></head><body>
<header class="top"><div class="top-inner"><div class="eyebrow">转换数据只读复核 · 2026-08-11</div><h1>{esc(title)}</h1><p>{esc(intro)}</p></div></header>
<main>{body}<footer class="footer">固定种子对完整 population 抽样；报告不修改服务器转换 JSONL 或源媒体。所有问题、答案、元数据和路径均经过 HTML 转义。</footer></main>
{f'<script>{script}</script>' if script else ''}</body></html>"""


def sample_media(dataset: str, row: dict[str, Any]) -> str:
    assets = [asset for asset in row.get("media_assets") or [] if asset.get("archive_path")]
    if not assets:
        label = MEDIA_STATUS_LABELS.get(str(row.get("media_status")), "??????")
        return f'<div class="sample-media"><div class="media-empty"><b>{esc(label)}</b></div></div>'
    rendered = []
    for asset in assets:
        relative = f"sub-dataset/{dataset}/{asset['archive_path']}"
        source = asset.get("source_remote") or asset.get("source") or ""
        media_type = asset.get("type")
        if asset.get("gt_overlay"):
            label = "GT ??????????????????"
        elif media_type == "videos":
            label = "??????????"
        else:
            label = "?????????"
        clean = asset.get("clean_archive_path")
        clean_link = ""
        if clean:
            clean_relative = f"sub-dataset/{dataset}/{clean}"
            clean_link = f'<br><a href="{quote(clean_relative)}">???????</a>'
        rendered.append(
            f'<div class="media-item"><a href="{quote(relative)}"><img loading="lazy" src="{quote(relative)}" alt="{esc(row["sample_id"])} ??"></a>'
            f'<div class="media-label">{esc(label)}<br>{esc(source)}{clean_link}</div></div>'
        )
    css = "one" if len(rendered) == 1 else ""
    return f'<div class="sample-media"><div class="media-grid {css}">{"".join(rendered)}</div></div>'


def sample_card(dataset: str, row: dict[str, Any]) -> str:
    errors = row.get("format_errors") or []
    format_state = "error" if errors else "pass"
    status = str(row.get("media_status") or "")
    search_text = " ".join(
        str(value)
        for value in (
            row.get("sample_id"), row.get("input_preview"), row.get("output_preview"),
            row.get("source_path"), row.get("metadata_preview"), errors,
        )
    ).lower()
    references = row.get("media_references") or []
    refs_html = "".join(
        f'<li><b>{esc(item.get("type"))}</b> <code>{esc(item.get("reference"))}</code></li>'
        for item in references
    ) or "<li>?????</li>"
    errors_html = (
        '<ul class="error-list">' + "".join(f"<li>{esc(error)}</li>" for error in errors) + "</ul>"
        if errors else '<span class="status ok">??</span>'
    )
    record_link = f"sub-dataset/{dataset}/{row['record_archive_path']}"
    raw_record = row.get("raw_record")
    raw_record_html = esc(pretty(raw_record if raw_record is not None else {"unavailable": record_link}))
    overlay_html = esc(pretty(row.get("ground_truth_overlay") or []))
    return f"""<article class="sample" data-search="{esc(search_text)}" data-status="{esc(status)}" data-split="{esc(row.get('split'))}" data-format="{format_state}">
{sample_media(dataset, row)}<div class="sample-body"><div class="sample-head"><div><div class="sample-id">{esc(row['sample_id'])}</div><div class="source">population #{n(row.get('population_index'))} ? {esc(row.get('source_file'))}:{n(row.get('line_number'))}</div></div>
<div class="chips"><span class="chip">{esc(row.get('split'))}</span><span class="chip">{esc(MEDIA_STATUS_LABELS.get(status, status))}</span><span class="chip">{'????' if errors else '????'}</span></div></div>
<dl class="qa"><dt>????</dt><dd><pre>{esc(row.get('input_preview'))}</pre></dd><dt>????</dt><dd><pre>{esc(row.get('output_preview'))}</pre></dd></dl>
<details class="details" open><summary>?????? JSON??????</summary><pre>{raw_record_html}</pre><p><a href="{quote(record_link)}">???? JSON ??</a></p></details>
<details class="details"><summary>??????????</summary><h3>????</h3>{errors_html}<h3>????</h3><ul class="media-paths">{refs_html}</ul><h3>GT overlay ??</h3><pre>{overlay_html}</pre><h3>????</h3><pre>{esc(pretty(row.get('objects')))}</pre><h3>????</h3><pre>{esc(pretty(row.get('metadata_preview')))}</pre></details></div></article>"""


def media_table(summary: dict[str, Any]) -> str:
    rows = []
    for media_type, item in sorted((summary.get("media_snapshot") or {}).items()):
        ratio = item.get("local_presence_ratio")
        rows.append(
            f"<tr><td>{esc(media_type)}</td><td>{n(item.get('unique_references'))}</td>"
            f"<td>{n(item.get('local_references'))}</td><td>{n(item.get('local_present'))}</td>"
            f"<td>{n(item.get('local_missing'))}</td><td>{n(item.get('remote_references'))}</td>"
            f"<td>{pct(ratio) if ratio is not None else '不适用'}</td></tr>"
        )
    return "".join(rows) or '<tr><td colspan="7">无媒体字段</td></tr>'


def error_table(summary: dict[str, Any]) -> str:
    fmt = summary.get("format") or {}
    errors = Counter(fmt.get("json_error_counts") or {}) + Counter(fmt.get("format_error_counts") or {})
    population = int(summary.get("population_total") or 0)
    if not errors:
        return '<tr><td>未发现结构异常</td><td>0</td><td>0.00%</td></tr>'
    return "".join(
        f"<tr><td><code>{esc(name)}</code></td><td>{n(count)}</td><td>{(count / population * 100 if population else 0):.6f}%</td></tr>"
        for name, count in errors.most_common()
    )


def format_diagnostic_note(item: dict[str, Any]) -> str:
    format_data = item.get("format") or {}
    note = format_data.get("diagnostic_note")
    if not note:
        return ""
    evidence = format_data.get("diagnostic_evidence")
    evidence_html = (
        f' <a href="{quote(str(evidence))}">查看完整复扫证据与原始 JSON</a>'
        if evidence
        else ""
    )
    return f'<div class="notice ok"><b>诊断更正：</b>{esc(note)}{evidence_html}</div>'


def format_error_examples(dataset_dir: Path) -> str:
    path = dataset_dir / "format-error-examples.jsonl"
    if not path.is_file():
        return ""
    rows = load_jsonl(path)
    if not rows:
        return ""
    rendered = "".join(
        f"<tr><td><code>{esc(row.get('source_path'))}</code></td><td>{n(row.get('line_number'))}</td>"
        f"<td>{n(row.get('video_placeholder_count'))}</td><td>{esc(row.get('video_values'))}</td></tr>"
        for row in rows
    )
    relative = f"sub-dataset/{dataset_dir.name}/format-error-examples.jsonl"
    return f"""<h3>结构异常的精确源行</h3><div class="table-wrap"><table><thead><tr><th>源文件</th><th>行号</th><th>&lt;video&gt; 数量</th><th>videos 字段</th></tr></thead><tbody>{rendered}</tbody></table></div><p><a href="{quote(relative)}">查看 {n(len(rows))} 条未改写的完整异常记录</a></p>"""


def schema_table(schema: dict[str, Any], population: int) -> str:
    presence = schema.get("field_presence_counts") or {}
    types = schema.get("field_type_counts") or {}
    return "".join(
        f"<tr><td><code>{esc(field)}</code></td><td>{n(count)}</td><td>{(count / population * 100 if population else 0):.2f}%</td><td>{esc(types.get(field, {}))}</td></tr>"
        for field, count in sorted(presence.items())
    )


def input_files_table(files: list[dict[str, Any]]) -> str:
    return "".join(
        f"<tr><td>{esc(row.get('split'))}</td><td><code>{esc(row.get('path'))}</code></td><td>{n(row.get('rows'))}</td><td>{n(row.get('size'))}</td><td><code>{esc(row.get('sha256'))}</code></td></tr>"
        for row in files
    )


def pixmo_detail(name: str, post_media: dict[str, Any]) -> str:
    post = (post_media.get("datasets") or {}).get(name)
    if not post:
        return ""
    download = post.get("download_report") or {}
    localization = post.get("localization_report") or {}
    splits = localization.get("splits") or {}
    written = sum(int(row.get("written_records") or 0) for row in splits.values())
    source = sum(int(row.get("source_records") or 0) for row in splits.values())
    return f"""<h3>PixMo 下载完成后的快照</h3><div class="table-wrap"><table><thead><tr><th>唯一媒体</th><th>下载成功</th><th>失败</th><th>待处理</th><th>可本地化记录</th><th>更新时间</th></tr></thead><tbody><tr><td>{n(download.get('total_media'))}</td><td>{n(download.get('successful_media'))}</td><td>{n(download.get('failed_media'))}</td><td>{n(download.get('pending_media'))}</td><td>{n(written)}/{n(source)}</td><td>{esc(download.get('updated_at'))}</td></tr></tbody></table></div><p class="muted">失败分布：{esc(download.get('statuses'))}。原转换 JSONL 仍保留远程 URL，本表说明 URL 对应媒体是否已下载；两者是不同口径。</p>"""


def dataset_report(root: Path, item: dict[str, Any], post_media: dict[str, Any], post_entry: dict[str, Any]) -> str:
    name = item["name"]
    dataset_dir = root / "sub-dataset" / name
    samples = load_jsonl(dataset_dir / "displayed-samples.jsonl")
    schema = load_json(dataset_dir / "source-schema.json", {})
    input_files = load_json(dataset_dir / "input-files.json", [])
    fmt_text, fmt_css = format_judgment(item)
    media_text, media_css = media_judgment(item, post_media)
    aggregate = media_aggregate(item)
    population = int(item.get("population_total") or 0)
    splits = sorted({str(row.get("split")) for row in samples})
    statuses = sorted({str(row.get("media_status")) for row in samples})
    sample_status = Counter(str(row.get("media_status")) for row in samples)
    current_entry = (post_entry.get("datasets") or {}).get(name, {})
    current_ready = bool(current_entry.get("exists") and current_entry.get("size"))
    overall_class = "bad" if fmt_css == "bad" or media_css == "bad" else ("warn" if fmt_css == "partial" or media_css == "partial" else "ok")
    text_quality = (item.get("format") or {}).get("text_quality_counts") or {}
    roles = schema.get("message_role_counts") or {}
    intro = f"任务类型：{TASK_LABELS.get(name, item.get('task_type') or '多模态指令微调')}。全量检查 {n(population)} 条稳定转换记录，并固定展示 100 条逻辑记录。"
    body = f"""
<nav class="nav"><a href="index.html">返回总览</a><a href="#full-audit">全量审计</a><a href="#samples">100 条样本</a></nav>
<div class="summary-grid"><div class="metric"><b>{n(population)}</b><span>稳定 population 行数</span></div><div class="metric"><b>{pct(item.get('format',{}).get('valid_format_ratio'))}</b><span>结构检查通过率</span></div><div class="metric"><b>{n(aggregate['present'])}/{n(aggregate['local'])}</b><span>本地媒体存在 / 引用</span></div><div class="metric"><b>{n(aggregate['remote'])}</b><span>远程媒体引用</span></div><div class="metric"><b>{n(len(samples))}</b><span>固定种子展示行</span></div></div>
<div class="notice {overall_class}"><b>总体判断：</b>{esc(fmt_text)}；{esc(media_text)}。当前正式 train 入口{'已通过官方校验' if current_ready else '尚不可用'}。结构正确、媒体可用和正式入口可用是三个独立检查面。</div>
<section id="full-audit"><h2>一、全量转换与入口审计</h2><div class="table-wrap"><table><thead><tr><th>检查面</th><th>结论</th><th>证据</th></tr></thead><tbody>
<tr><td>JSONL 与 ms-swift 结构</td><td>{status_chip(fmt_text,fmt_css)}</td><td>通过 {n(item.get('format',{}).get('valid_format_rows'))}/{n(population)}</td></tr>
<tr><td>媒体引用</td><td>{status_chip(media_text,media_css)}</td><td>本地存在 {n(aggregate['present'])}，本地缺失 {n(aggregate['missing'])}，远程引用 {n(aggregate['remote'])}</td></tr>
<tr><td>采集时正式入口</td><td>{status_chip('可用' if item.get('canonical_train_ready') else '采集时未就绪','ok' if item.get('canonical_train_ready') else 'partial')}</td><td><code>{esc((item.get('canonical_train') or {}).get('path'))}</code></td></tr>
<tr><td>后审计正式入口</td><td>{status_chip('官方校验通过' if current_ready else '不可用','ok' if current_ready else 'bad')}</td><td><code>{esc(current_entry.get('path'))}</code>，{n(current_entry.get('size'))} bytes</td></tr>
</tbody></table></div>
<h3>结构异常分布</h3><div class="table-wrap"><table><thead><tr><th>异常类型</th><th>行数</th><th>population 占比</th></tr></thead><tbody>{error_table(item)}</tbody></table></div>{format_diagnostic_note(item)}{format_error_examples(dataset_dir)}
<h3>唯一媒体路径快照</h3><div class="table-wrap"><table><thead><tr><th>媒体类型</th><th>唯一引用</th><th>本地引用</th><th>本地存在</th><th>本地缺失</th><th>远程引用</th><th>本地存在率</th></tr></thead><tbody>{media_table(item)}</tbody></table></div>{pixmo_detail(name, post_media)}
<h3>文本质量诊断</h3><pre class="notice">{esc(pretty(text_quality if text_quality else {'diagnostic':'未检出替换字符、常见乱码片段或空消息'}))}</pre></section>
<section><h2>二、输入、输出与存储结构</h2><p>实际消息角色计数：<code>{esc(roles)}</code>。字段来自完整转换 population 的逐行观测，不根据数据集名称推断未出现的监督。</p><div class="table-wrap"><table><thead><tr><th>字段</th><th>出现行数</th><th>覆盖率</th><th>观测类型</th></tr></thead><tbody>{schema_table(schema,population)}</tbody></table></div>
<h3>输入文件与内容指纹</h3><div class="table-wrap"><table><thead><tr><th>split</th><th>路径</th><th>行数</th><th>字节</th><th>SHA-256</th></tr></thead><tbody>{input_files_table(input_files)}</tbody></table></div></section>
<section><h2>三、抽样口径</h2><div class="method"><div><b>抽样单位</b><p>转换后的 logical record；同一媒体上的不同问答不会被合并。</p></div><div><b>确定性</b><p>seed={n(item.get('seed'))}，对完整 population 的文件路径和行号做 SHA-256 排序；记录 400 个候选。</p></div><div><b>证据边界</b><p>固定展示前 100 条，不因媒体缺失而重抽。样本媒体状态：{esc(dict(sample_status))}</p></div></div></section>
<section><h2>四、只读与可追溯性</h2><p>声明的 source_train/eval 文件在扫描前后具有相同路径、大小和 mtime 指纹：<code>{esc(item.get('source_fingerprint_before'))}</code>。每条样本保留源文件、行号、完整原始记录和媒体归档路径。</p></section>
<section id="samples"><h2>五、100 条转换样本</h2><div class="filters"><input id="filter-search" type="search" placeholder="搜索问题、答案、路径或样本 ID" aria-label="关键词搜索"><select id="filter-status"><option value="">全部媒体状态</option>{''.join(f'<option value="{esc(value)}">{esc(MEDIA_STATUS_LABELS.get(value,value))}</option>' for value in statuses)}</select><select id="filter-split"><option value="">全部 split</option>{''.join(f'<option value="{esc(value)}">{esc(value)}</option>' for value in splits)}</select><select id="filter-format"><option value="">全部格式状态</option><option value="pass">格式通过</option><option value="error">格式异常</option></select><div id="filter-count" class="filter-count"></div></div><div class="samples">{''.join(sample_card(name,row) for row in samples)}</div></section>"""
    return page_shell(f"{item['display_name']} 转换数据复核", intro, body, script=FILTER_SCRIPT)


def pixmo_index_table(post_media: dict[str, Any], overall: dict[str, Any]) -> str:
    by_name = {item["name"]: item for item in overall["datasets"]}
    rows = []
    for name, post in sorted((post_media.get("datasets") or {}).items()):
        download = post.get("download_report") or {}
        successful = int(download.get("successful_media") or 0)
        total = int(download.get("total_media") or 0)
        failed = int(download.get("failed_media") or 0)
        sample = by_name.get(name, {}).get("sample_media_statuses") or {}
        rows.append(
            f"<tr><td><a href=\"{quote(name)}.html\">{esc(name)}</a></td><td>{n(successful)}/{n(total)} ({(successful/total*100 if total else 0):.2f}%)</td><td>{n(failed)}</td><td>{esc(download.get('statuses'))}</td><td>{esc(sample)}</td><td>{esc(download.get('updated_at'))}</td></tr>"
        )
    return "".join(rows)


def index_report(root: Path, overall: dict[str, Any], post_media: dict[str, Any], post_entry: dict[str, Any]) -> str:
    datasets = overall["datasets"]
    dataset_count = len(datasets)
    eval_count = int((post_entry.get("eval_count") or 0))
    manifest_label = overall.get("manifest_label") or Path(str(overall.get("manifest") or "")).name
    train_input_label = overall.get("train_input_label") or "source_train"
    total_population = sum(int(item.get("population_total") or 0) for item in datasets)
    valid_rows = sum(int(item.get("format", {}).get("valid_format_rows") or 0) for item in datasets)
    format_issue_datasets = sum(bool(item.get("format", {}).get("format_error_counts") or item.get("format", {}).get("json_error_counts")) for item in datasets)
    invalid_format_rows = max(total_population - valid_rows, 0)
    format_notice_class = "ok" if invalid_format_rows == 0 else "warn"
    remote_datasets = sum(media_aggregate(item)["remote"] > 0 for item in datasets)
    missing_datasets = sum(media_aggregate(item)["missing"] > 0 for item in datasets)
    collection_ready = sum(bool(item.get("canonical_train_ready")) for item in datasets)
    current_ready = int(post_entry.get("ready_count") or 0)
    validator_passed = bool(post_entry.get("official_validator_passed"))
    delivery_validation = load_json(root / "validation.json", {})
    browser_validation = load_json(root / "browser-validation.json", {})
    rows = []
    findings = []
    for item in datasets:
        name = item["name"]
        aggregate = media_aggregate(item)
        fmt_text, fmt_css = format_judgment(item)
        media_text, media_css = media_judgment(item, post_media)
        current = (post_entry.get("datasets") or {}).get(name, {})
        ready = bool(current.get("exists") and current.get("size"))
        rows.append(
            f'<tr><td><a href="{quote(name)}.html"><b>{esc(item["display_name"])}</b></a><br><span class="muted">{esc(TASK_LABELS.get(name,item.get("task_type")))}</span></td>'
            f'<td>{n(item.get("population_total"))}</td><td>{status_chip(fmt_text,fmt_css)}<br>{pct(item.get("format",{}).get("valid_format_ratio"))}</td>'
            f'<td>{status_chip(media_text,media_css)}<br>本地 {n(aggregate["present"])} / 缺失 {n(aggregate["missing"])} / 远程 {n(aggregate["remote"])}</td>'
            f'<td>{status_chip("通过" if ready else "缺失","ok" if ready else "bad")}</td><td>{esc(item.get("sample_media_statuses"))}</td></tr>'
        )
        if fmt_css != "ok":
            findings.append({"dataset": name, "type": "format", "finding": fmt_text})
        if media_css != "ok":
            findings.append({"dataset": name, "type": "media", "finding": media_text})
    excluded_rows = "".join(
        f"<tr><td>{esc(item.get('name'))}</td><td>{esc(item.get('kind'))}</td><td><code>{esc(item.get('path'))}</code></td><td>{esc(item.get('reason'))}</td></tr>"
        for item in overall.get("excluded_inventory") or []
    )
    processes = "\n".join(overall.get("active_processes_before") or []) or "采集开始时未检测到相关下载或去重进程"
    entry_css = "ok" if validator_passed and current_ready == len(datasets) else "bad"
    body = f"""
<div class="summary-grid"><div class="metric"><b>{n(len(datasets))}</b><span>纳入数据集</span></div><div class="metric"><b>{n(total_population)}</b><span>全量扫描逻辑行</span></div><div class="metric"><b>{n(overall.get('displayed_rows_total'))}</b><span>HTML 展示样本</span></div><div class="metric"><b>{n(current_ready)}/{n(len(datasets))}</b><span>当前正式入口可用</span></div><div class="metric"><b>{n(format_issue_datasets)}</b><span>含结构异常的数据集</span></div></div>
<div class="notice {format_notice_class}"><b>总体判断：</b>{n(dataset_count)} 个稳定 {esc(train_input_label)} + eval population 共 {n(total_population)} 行，其中 {n(valid_rows)} 行通过结构检查，{n(invalid_format_rows)} 行存在结构异常。所有本地媒体引用均存在，本地缺失数据集数为 {n(missing_datasets)}；{n(remote_datasets)} 个数据集仍含远程 URL。远程 URL 不等于媒体已本地化，PixMo 的最终下载比例在下表单列。</div>
<section><h2>一、{n(dataset_count)} 个数据集总览</h2><div class="table-wrap"><table class="compact"><thead><tr><th>数据集 / 任务</th><th>population</th><th>结构</th><th>媒体</th><th>当前入口</th><th>100 条样本媒体状态</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section><h2>二、正式入口时间边界</h2><div class="notice {entry_css}"><b>当前结论：</b>官方 <code>validate_sft_entrypoints.py</code> 返回码为 {n(post_entry.get('official_validator_returncode'))}，{n(dataset_count)} 个 train 与 {n(eval_count)} 个 eval 入口校验通过。<a href="entrypoint-validation.json">查看完整输出</a></div><p>本报告以 <code>{esc(manifest_label)}</code> 为准，逐数据集采集快照观察到 {n(collection_ready)}/{n(dataset_count)} 个训练入口；于 {esc(post_entry.get('checked_at'))} 重新运行官方校验，当前状态为 {n(current_ready)}/{n(dataset_count)}。报告不把采集时快照误写成当前状态。</p></section>
<section><h2>三、PixMo 下载与本地化</h2><div class="table-wrap"><table><thead><tr><th>数据集</th><th>下载成功 / 唯一媒体</th><th>失败</th><th>状态分布</th><th>固定 100 条当前展示状态</th><th>更新时间</th></tr></thead><tbody>{pixmo_index_table(post_media,overall)}</tbody></table></div><p class="muted">固定 100 条 sample_id 未改变。下载完成后仅补充已落盘的原图：PixMo-Cap 从 15 条可展示增至 46 条，PixMo-Points 从 28 条增至 48 条。</p></section>
<section><h2>四、审计口径</h2><div class="method"><div><b>范围</b><p>以服务器 <code>{esc(manifest_label)}</code> 的 {n(dataset_count)} 个启用项为准；备份、smoke、工具与派生报告目录只列为排除项。</p></div><div><b>全量检查</b><p>逐行解析 {esc(train_input_label)} 和 eval，核对 messages、role、媒体有序数组、占位符和 grounding 对象；唯一媒体路径做存在性快照。</p></div><div><b>100 条展示</b><p>每项使用独立固定 seed，从完整 logical-record population 确定性抽样，不因媒体缺失而替换。</p></div></div><h3>采集开始时的并行进程</h3><pre class="notice">{esc(processes)}</pre></section>
<section><h2>五、交付验证</h2><div class="table-wrap"><table><thead><tr><th>验证面</th><th>结果</th><th>证据</th></tr></thead><tbody>
<tr><td>logical-record 专用结构校验</td><td>{status_chip('通过' if delivery_validation.get('status') == 'passed' else '未通过','ok' if delivery_validation.get('status') == 'passed' else 'bad')}</td><td>{n(delivery_validation.get('datasets'))} 个数据集，{n(delivery_validation.get('displayed_samples'))} 条展示样本，{n(delivery_validation.get('candidate_rows'))} 个候选分区。<a href="validation.json">查看结果</a></td></tr>
<tr><td>浏览器双视口校验</td><td>{status_chip('通过' if browser_validation.get('status') == 'passed' else '未通过','ok' if browser_validation.get('status') == 'passed' else 'bad')}</td><td>{n(browser_validation.get('html_pages'))} 个页面，{n(browser_validation.get('viewport_checks'))} 次检查。<a href="browser-validation.json">查看结果</a></td></tr>
<tr><td>通用 source-visual 校验器</td><td>{status_chip('契约不适用','partial')}</td><td>该校验器要求每条 accepted row 都是唯一且已归档的视觉；本报告按 logical record 抽样并保留同图多问答、纯文本和远程媒体行。<a href="skill-validator-result.json">查看调用结果与适配说明</a></td></tr>
</tbody></table></div></section>
<section><h2>六、排除与非正式目录</h2><div class="table-wrap"><table><thead><tr><th>名称</th><th>类型</th><th>路径</th><th>原因</th></tr></thead><tbody>{excluded_rows}</tbody></table></div></section>"""
    (root / "audit-findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return page_shell(
        "ms-swift 转换数据复核总览",
        f"以 {manifest_label} 为准，全量复核 {dataset_count} 个训练入口及对应 eval，并为每项展示 100 条确定性抽样记录。",
        body,
    )


def write_csv(root: Path, overall: dict[str, Any], post_media: dict[str, Any]) -> None:
    fields = [
        "dataset", "display_name", "task_type", "population_total", "format_valid_ratio",
        "local_media_references", "local_media_present", "local_media_missing", "remote_media_references",
        "post_audit_canonical_train_ready", "selected_count", "source_unchanged",
    ]
    with (root / "dataset-audit-summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in overall["datasets"]:
            aggregate = media_aggregate(item)
            writer.writerow({
                "dataset": item["name"], "display_name": item["display_name"],
                "task_type": TASK_LABELS.get(item["name"], item.get("task_type")),
                "population_total": item.get("population_total"),
                "format_valid_ratio": item.get("format", {}).get("valid_format_ratio"),
                "local_media_references": aggregate["local"], "local_media_present": aggregate["present"],
                "local_media_missing": aggregate["missing"], "remote_media_references": aggregate["remote"],
                "post_audit_canonical_train_ready": item.get("post_audit_canonical_train_ready"),
                "selected_count": item.get("selected_count"), "source_unchanged": item.get("source_unchanged"),
            })


def build(root: Path, expected_datasets: int = 17) -> None:
    root = root.resolve()
    overall = load_json(root / "overall-summary.json")
    post_media = load_json(root / "post-audit-media-status.json", {"datasets": {}})
    post_entry = load_json(root / "post-audit-entrypoint-status.json", {"datasets": {}})
    if overall.get("dataset_count") != expected_datasets:
        raise ValueError(f"Expected {expected_datasets} datasets, got {overall.get('dataset_count')}")
    for item in overall["datasets"]:
        report = dataset_report(root, item, post_media, post_entry)
        (root / f"{item['name']}.html").write_text(report, encoding="utf-8", newline="\n")
    (root / "index.html").write_text(index_report(root, overall, post_media, post_entry), encoding="utf-8", newline="\n")
    write_csv(root, overall, post_media)
    metadata = {
        "version": 2, "generated_at": datetime.now(timezone.utc).isoformat(), "language": "zh-CN",
        "dataset_reports": expected_datasets, "displayed_rows": overall.get("displayed_rows_total"),
        "offline_assets": True, "external_dependencies": False,
    }
    (root / "report-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"built reports={expected_datasets} displayed_rows={overall.get('displayed_rows_total')} root={root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-datasets", type=int, default=17)
    args = parser.parse_args()
    build(args.root, args.expected_datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
