import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / 'index.html'
COUNTS_PATH = ROOT / 'current-dataset-counts' / 'dataset-counts.json'


def fmt(value):
    return f'{value:,}' if value is not None else '未配置'


def main():
    counts = json.loads(COUNTS_PATH.read_text(encoding='utf-8'))
    rows_by_dataset = {row['dataset']: row for row in counts['rows']}
    html = INDEX_PATH.read_text(encoding='utf-8')

    section_match = re.search(r'<section id="inventory">.*?</section>', html)
    if not section_match:
        raise RuntimeError('index.html 中未找到 inventory 区块')
    section = section_match.group(0)
    table_match = re.search(r'<table.*?</table>', section)
    if not table_match:
        raise RuntimeError('inventory 区块中未找到数据集表格')
    table = table_match.group(0)

    table = re.sub(r'<tfoot>.*?</tfoot>', '', table)
    header = re.search(r'<thead><tr>(.*?)</tr></thead>', table)
    if not header:
        raise RuntimeError('数据集表头结构不符合预期')
    original_headers = re.findall(r'<th>.*?</th>', header.group(1))
    if len(original_headers) < 11:
        raise RuntimeError('数据集表头列数不足')
    original_headers = original_headers[:11]
    current_headers = [
        '<th>当前训练样本</th>',
        '<th>当前评测样本</th>',
        '<th>当前可见转换样本</th>',
    ]
    table = table[:header.start()] + '<thead><tr>' + ''.join(original_headers + current_headers) + '</tr></thead>' + table[
        header.end():]

    body_match = re.search(r'<tbody>(.*?)</tbody>', table)
    if not body_match:
        raise RuntimeError('数据集表体结构不符合预期')
    merged_rows = []
    for row_html in re.findall(r'<tr>.*?</tr>', body_match.group(1)):
        dataset_match = re.search(r'<td><a href="[^"]+">(.*?)</a></td>', row_html)
        if not dataset_match:
            raise RuntimeError('无法从数据集行提取名称')
        dataset = dataset_match.group(1)
        row = rows_by_dataset.pop(dataset, None)
        if row is None:
            raise RuntimeError(f'当前统计中缺少数据集：{dataset}')
        cells = re.findall(r'<td>.*?</td>', row_html)
        cells = cells[:11]
        cells.extend([
            f'<td>{fmt(row["source_train"])}</td>',
            f'<td>{fmt(row["eval"])}</td>',
            f'<td>{fmt(row["available_total"])}</td>',
        ])
        merged_rows.append('<tr>' + ''.join(cells) + '</tr>')
    if rows_by_dataset:
        raise RuntimeError(f'首页表格中缺少数据集：{", ".join(rows_by_dataset)}')

    numeric_columns = [3, 4, 6, 7, 8, 9]
    totals = {}
    for column in numeric_columns:
        total = 0
        complete = True
        for row_html in merged_rows:
            value = re.findall(r'<td>(.*?)</td>', row_html)[column]
            value = re.sub(r'<.*?>', '', value).replace(',', '')
            if not value.isdigit():
                complete = False
                break
            total += int(value)
        totals[column] = fmt(total) if complete else '不适用'

    total_cells = ['<th>总计（跨源求和）</th>', '<td>—</td>', '<td>—</td>']
    for column in range(3, 11):
        total_cells.append(f'<td>{totals.get(column, "—")}</td>')
    total_cells.extend([
        f'<td>{fmt(counts["current_source_train_sum"])}</td>',
        f'<td>{fmt(counts["current_eval_sum"])}</td>',
        f'<td>{fmt(counts["current_available_sum"])}</td>',
    ])
    new_table = table[:body_match.start()] + '<tbody>' + ''.join(merged_rows) + '</tbody>' + table[body_match.end():]
    new_table = new_table[:-8] + '<tfoot><tr>' + ''.join(total_cells) + '</tr></tfoot></table>'

    note = (
        '<p id="current-counts-note" class="provenance"><strong>当前转换样本（2026-08-10）：</strong>'
        f'训练 {fmt(counts["current_source_train_sum"])}，评测/测试 {fmt(counts["current_eval_sum"])}，'
        f'当前可见合计 {fmt(counts["current_available_sum"])}。转换样本按 canonical JSONL 实际行数统计；'
        '原始记录、监督单元和转换样本语义不同，不应相互加总。'
        '<a href="current-dataset-counts/dataset-counts.html">查看完整当前统计与证据</a>。</p>'
    )
    section = re.sub(
        r'<p(?: id="current-counts-note")? class="provenance"><strong>当前转换样本.*?</p>', '', section)
    table_match = re.search(r'<table.*?</table>', section)
    section = section[:table_match.start()] + new_table + section[table_match.end():]
    section = section.replace('<div class="table-wrap">', note + '<div class="table-wrap">', 1)
    html = html[:section_match.start()] + section + html[section_match.end():]

    style_marker = 'th{background:#edf1f1;position:sticky;top:0;z-index:1}'
    total_style = 'tfoot th,tfoot td{background:#e4efed;font-weight:750;border-top:2px solid var(--teal);white-space:nowrap}'
    if total_style not in html:
        html = html.replace(style_marker, style_marker + total_style, 1)
    INDEX_PATH.write_text(html, encoding='utf-8')


if __name__ == '__main__':
    main()
