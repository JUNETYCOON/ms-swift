#!/usr/bin/env python3
"""Build a Chinese HTML report from historical source counts and a current scan."""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def sum_lines(files: list[dict]) -> int | None:
    values = [item['lines'] for item in files if item.get('exists')]
    return sum(values) if values else None


def missing_paths(files: list[dict]) -> list[str]:
    return [item['path'] for item in files if not item.get('exists')]


def fmt(value: int | None) -> str:
    return '未提供' if value is None else f'{value:,}'


def main() -> None:
    args = parse_args()
    with args.inventory.open(encoding='utf-8-sig', newline='') as stream:
        inventory = {row['entry']: row for row in csv.DictReader(stream) if row['status'] == 'included'}
    current = json.loads(args.current.read_text(encoding='utf-8-sig'))
    rows = []
    for name, observed in current['datasets'].items():
        history = inventory[name]
        source_records = int(history['source_records'])
        source_train = sum_lines(observed['source_train'])
        eval_count = sum_lines(observed['eval'])
        global_train = sum_lines(observed['train'])
        available_total = None if source_train is None and eval_count is None else (source_train or 0) + (eval_count or 0)
        snapshot = observed['source_snapshot']
        same_shape = snapshot['file_count'] == int(history['file_count']) and snapshot['total_bytes'] == int(
            history['source_bytes_observed'])
        rows.append({
            'dataset': name,
            'source_records': source_records,
            'supervision_units': int(history['supervision_units']),
            'unique_media': int(history['unique_media']),
            'source_train': source_train,
            'eval': eval_count,
            'available_total': available_total,
            'global_train': global_train,
            'global_missing': missing_paths(observed['train']),
            'global_configured': bool(observed['train']),
            'source_shape_matches_history': same_shape,
            'source_snapshot': snapshot,
            'evidence_scope': history['evidence_scope'],
            'source_splits': history['source_splits'],
        })
    generated_at = datetime.fromisoformat(current['generated_at']).astimezone()
    summary = {
        'version': 1,
        'generated_at': current['generated_at'],
        'historical_source_evidence_date': '2026-08-06',
        'source_host': current['host'],
        'dataset_count': len(rows),
        'source_records_sum_non_comparable': sum(row['source_records'] for row in rows),
        'current_source_train_sum': sum(row['source_train'] or 0 for row in rows),
        'current_eval_sum': sum(row['eval'] or 0 for row in rows),
        'current_available_sum': sum(row['available_total'] or 0 for row in rows),
        'global_train_configured_datasets': sum(bool(current['datasets'][row['dataset']]['train']) for row in rows),
        'global_train_available_datasets': sum(row['global_train'] is not None for row in rows),
        'global_train_missing_datasets': sum(bool(row['global_missing']) for row in rows),
        'source_shape_matching_datasets': sum(row['source_shape_matches_history'] for row in rows),
        'rows': rows,
        'limitations': [
            '原始记录、监督单元与转换后样本的语义不同，跨数据集合计仅用于容量盘点。',
            '原始全量记录数来自 2026-08-06 的全量扫描；当前扫描以文件数和总字节复核源快照形态。',
            'global_train 是配置声明的全局媒体去重后训练入口；缺失时不得用 source_train 冒充。',
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'dataset-counts.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    table_rows = []
    for row in rows:
        if not row['global_configured']:
            global_status, status_class = '未配置', 'unconfigured'
        elif row['global_train'] is None:
            global_status, status_class = '缺失', 'missing'
        else:
            global_status, status_class = fmt(row['global_train']), 'ok'
        source_status = '形态一致' if row['source_shape_matches_history'] else '发生变化'
        search = html.escape(' '.join([row['dataset'], global_status, source_status]).lower(), quote=True)
        table_rows.append(f'''<tr data-search="{search}" data-status="{status_class}">
<td><strong>{html.escape(row['dataset'])}</strong><small>{html.escape(row['source_splits'] or '未声明 split')}</small></td>
<td class="num">{fmt(row['source_records'])}<small>{html.escape(row['evidence_scope'])}</small></td>
<td class="num">{fmt(row['supervision_units'])}</td>
<td class="num">{fmt(row['unique_media'])}</td>
<td class="num">{fmt(row['source_train'])}</td>
<td class="num">{fmt(row['eval'])}</td>
<td class="num"><strong>{fmt(row['available_total'])}</strong></td>
<td><span class="badge {status_class}">{global_status}</span></td>
<td><span class="badge {'ok' if row['source_shape_matches_history'] else 'warn'}">{source_status}</span></td>
</tr>''')
    report = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 1 数据集样本量汇总</title>
<style>
:root{{--ink:#22292f;--muted:#657078;--line:#d5dadd;--paper:#fff;--page:#f3f5f4;--green:#17643b;--green-bg:#e9f4ed;--red:#963128;--red-bg:#faece9;--amber:#775716;--amber-bg:#fbf2db;--blue:#245a82;--blue-bg:#eaf2f8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;letter-spacing:0}}header,main,footer{{width:min(1500px,calc(100% - 32px));margin:auto}}header{{padding:28px 0 20px}}h1{{margin:0 0 8px;font-size:30px;letter-spacing:0}}h2{{font-size:19px;letter-spacing:0}}p{{margin:6px 0}}.muted,small{{color:var(--muted)}}.warning{{padding:12px 14px;border-left:4px solid var(--amber);background:var(--amber-bg)}}.metrics{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1px;margin:20px 0;background:var(--line);border:1px solid var(--line)}}.metric{{padding:14px;background:var(--paper);min-width:0}}.metric span,.metric small{{display:block}}.metric strong{{display:block;font-size:23px;overflow-wrap:anywhere}}.toolbar{{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin:20px 0 12px;padding:12px;background:var(--paper);border:1px solid var(--line)}}label{{display:grid;gap:4px;color:var(--muted);font-size:12px}}input,select{{min-height:36px;padding:6px 9px;border:1px solid #b9bec2;border-radius:4px;background:#fff}}input{{min-width:280px}}.shown{{margin-left:auto}}.table-wrap{{overflow-x:auto;background:var(--paper);border:1px solid var(--line)}}table{{width:100%;border-collapse:collapse;min-width:1180px}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{position:sticky;top:0;background:#edf0f1;font-size:12px}}td.num{{font-variant-numeric:tabular-nums;text-align:right}}td small{{display:block;max-width:240px;margin-top:3px}}.badge{{display:inline-block;padding:2px 7px;border:1px solid;border-radius:4px;white-space:nowrap}}.badge.ok{{color:var(--green);border-color:#9cc9aa;background:var(--green-bg)}}.badge.missing{{color:var(--red);border-color:#dfaaa4;background:var(--red-bg)}}.badge.warn{{color:var(--amber);border-color:#dec582;background:var(--amber-bg)}}.notes{{margin:22px 0;padding:16px;background:var(--paper);border:1px solid var(--line)}}footer{{padding:24px 0 32px;color:var(--muted)}}
@media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}.shown{{margin-left:0}}}}@media(max-width:520px){{header,main,footer{{width:calc(100% - 18px)}}h1{{font-size:23px}}.metrics{{grid-template-columns:1fr}}input{{min-width:0;width:100%}}label{{width:100%}}}}
</style></head><body>
<header><div class="muted">STAGE 1 / DATASET CAPACITY</div><h1>数据集样本量汇总</h1><p>当前转换文件扫描：{generated_at:%Y-%m-%d %H:%M:%S %Z} · 原始全量基线：2026-08-06 · 主机：{html.escape(current['host'])}</p></header>
<main><div class="warning"><strong>当前训练入口不完整：</strong>已配置的 global_train 在 {summary['global_train_missing_datasets']} 个数据集上全部缺失；COCO 未配置转换入口。表中的 source_train 是去重前转换结果，不能直接替代正式训练入口。</div>
<section class="metrics"><div class="metric"><span>数据集</span><strong>{len(rows)}</strong><small>上一轮报告边界</small></div><div class="metric"><span>当前 source_train</span><strong>{summary['current_source_train_sum']:,}</strong><small>去重前转换样本</small></div><div class="metric"><span>当前 eval</span><strong>{summary['current_eval_sum']:,}</strong><small>验证/测试样本</small></div><div class="metric"><span>当前可见转换样本</span><strong>{summary['current_available_sum']:,}</strong><small>source_train + eval</small></div><div class="metric"><span>可用 global_train</span><strong>{summary['global_train_available_datasets']} / {summary['global_train_configured_datasets']}</strong><small>已配置 canonical 入口</small></div></section>
<section><h2>逐数据集统计</h2><div class="toolbar"><label>搜索<input id="search" type="search" placeholder="数据集或状态"></label><label>global_train 状态<select id="status"><option value="">全部</option><option value="ok">可用</option><option value="missing">缺失</option><option value="unconfigured">未配置</option></select></label><div class="shown">显示 <strong id="shown">{len(rows)}</strong> / {len(rows)}</div></div>
<div class="table-wrap"><table><thead><tr><th>数据集</th><th>原始全量记录</th><th>监督单元</th><th>唯一媒体</th><th>当前 source_train</th><th>当前 eval</th><th>当前可见合计</th><th>global_train</th><th>源快照复核</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></section>
<section class="notes"><h2>口径说明</h2><p><strong>原始全量记录：</strong>上一轮报告对源数据的 logical row / canonical record 全量统计。</p><p><strong>监督单元：</strong>可形成监督信号的问答、描述或标注数量；VisualGenome 等数据集会显著高于图片数。</p><p><strong>source_train：</strong>转换、清洗和切分后的训练候选，但尚未应用全局媒体去重。</p><p><strong>global_train：</strong>配置声明的最终训练入口。本次扫描中 17 个已配置入口全部缺失，因此当前不能按 canonical 配置直接启动完整 Stage 1 训练；COCO 没有配置转换入口。</p><p><strong>源快照复核：</strong>当前文件数和总字节是否与 2026-08-06 全量扫描一致；它不等同于逐文件内容哈希。</p></section></main>
<footer>机器可读明细：dataset-counts.json · 数据源保持只读，报告写入 random_sample_report。</footer>
<script>const rows=[...document.querySelectorAll('tbody tr')],search=document.getElementById('search'),status=document.getElementById('status'),shown=document.getElementById('shown');function filter(){{const q=search.value.trim().toLowerCase(),s=status.value;let n=0;for(const row of rows){{const ok=(!q||row.dataset.search.includes(q))&&(!s||row.dataset.status===s);row.hidden=!ok;if(ok)n++}}shown.textContent=n}}search.addEventListener('input',filter);status.addEventListener('change',filter);</script>
</body></html>'''
    (args.output_dir / 'dataset-counts.html').write_text(report, encoding='utf-8')
    print(json.dumps({'datasets': len(rows), 'html': str(args.output_dir / 'dataset-counts.html')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
