import csv
import html
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    from .records import EvaluationRecord
    from .registry import metric_spec_for, normalize_name
except ImportError:
    from records import EvaluationRecord
    from registry import metric_spec_for, normalize_name


def records_to_matrix(records: Sequence[EvaluationRecord]) -> Dict[Tuple[str, str], Mapping[str, float]]:
    return {(record.model, record.benchmark): record.metrics for record in records}


def write_summary_csv(records: Sequence[EvaluationRecord], metrics: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'benchmark', *metrics])
        writer.writeheader()
        for record in records:
            row = {'model': record.model, 'benchmark': record.benchmark}
            row.update({metric: record.metrics.get(normalize_name(metric), '') for metric in metrics})
            writer.writerow(row)


def _display_values(matrix: Dict[Tuple[str, str], Mapping[str, float]], models: Sequence[str], benchmark: str,
                    metric: str) -> Tuple[List[float], bool, bool]:
    spec = metric_spec_for(benchmark, metric)
    metric_name = spec.name if spec else normalize_name(metric)
    values: List[float] = []
    has_value = False
    for model in models:
        raw_value = matrix.get((model, benchmark), {}).get(metric_name)
        if raw_value is None:
            values.append(0.0)
        else:
            has_value = True
            values.append(raw_value * 100.0 if spec and spec.percentage else raw_value)
    return values, has_value, bool(spec and spec.percentage)


def _panel_pairs(benchmarks: Sequence[str], metrics: Sequence[str]) -> List[Tuple[str, str]]:
    return [(benchmark, metric) for benchmark in benchmarks for metric in metrics
            if metric_spec_for(benchmark, metric) is not None]


def _write_svg_bars(records: Sequence[EvaluationRecord], models: Sequence[str], benchmarks: Sequence[str],
                    metrics: Sequence[str], output_path: Path, title: str) -> Path:
    matrix = records_to_matrix(records)
    output_path = output_path if output_path.suffix.lower() == '.svg' else output_path.with_suffix('.svg')
    panels = _panel_pairs(benchmarks, metrics)
    if not panels:
        raise ValueError('No registered benchmark/metric pairs are available for visualization.')
    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    cell_width = 380
    cell_height = 270
    title_height = 56
    width = max(720, cell_width * ncols)
    height = title_height + cell_height * nrows
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2', '#B279A2', '#FF9DA6', '#9D755D']
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20" '
        f'font-family="Arial, sans-serif" font-weight="700">{html.escape(title)}</text>',
    ]

    for panel_idx, (benchmark, metric) in enumerate(panels):
        row_idx, col_idx = divmod(panel_idx, ncols)
        x0 = col_idx * cell_width
        y0 = title_height + row_idx * cell_height
        plot_x = x0 + 58
        plot_y = y0 + 50
        plot_w = cell_width - 90
        plot_h = cell_height - 105
        spec = metric_spec_for(benchmark, metric)
        values, has_value, is_percentage = _display_values(matrix, models, benchmark, metric)
        max_value = max(values) if values else 0.0
        y_max = max(100.0, max_value * 1.12) if is_percentage else (max_value * 1.15 if max_value > 0 else 1.0)
        bar_gap = 12
        bar_w = max(12, (plot_w - bar_gap * (len(models) + 1)) / max(1, len(models)))

        lines.append(
            f'<text x="{x0 + cell_width / 2:.1f}" y="{y0 + 24}" text-anchor="middle" font-size="14" '
            f'font-family="Arial, sans-serif" font-weight="700">{html.escape(benchmark)} / '
            f'{html.escape(metric)}</text>')
        lines.append(
            f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" '
            'stroke="#333" stroke-width="1"/>')
        lines.append(
            f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" '
            'stroke="#333" stroke-width="1"/>')
        for tick in range(5):
            y = plot_y + plot_h - plot_h * tick / 4
            label_value = y_max * tick / 4
            label = f'{label_value:.0f}%' if is_percentage else f'{label_value:.2g}'
            lines.append(
                f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" '
                'stroke="#dddddd" stroke-width="1"/>')
            lines.append(
                f'<text x="{plot_x - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
                f'font-family="Arial, sans-serif" fill="#555">{label}</text>')

        if spec and not spec.higher_is_better:
            lines.append(
                f'<text x="{plot_x + plot_w}" y="{plot_y - 8}" text-anchor="end" font-size="10" '
                'font-family="Arial, sans-serif" fill="#666">lower is better</text>')
        if not has_value:
            lines.append(
                f'<text x="{plot_x + plot_w / 2:.1f}" y="{plot_y + plot_h / 2:.1f}" text-anchor="middle" '
                'font-size="14" font-family="Arial, sans-serif" fill="#888">missing</text>')

        for index, (model, value) in enumerate(zip(models, values)):
            bar_h = 0 if y_max == 0 else plot_h * value / y_max
            x = plot_x + bar_gap + index * (bar_w + bar_gap)
            y = plot_y + plot_h - bar_h
            label = f'{value:.1f}%' if is_percentage else f'{value:.3g}'
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'fill="{colors[index % len(colors)]}"/>')
            if value != 0.0:
                lines.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-size="10" '
                    f'font-family="Arial, sans-serif" fill="#333">{label}</text>')
            lines.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{plot_y + plot_h + 18}" text-anchor="middle" font-size="10" '
                f'font-family="Arial, sans-serif" fill="#333">{html.escape(model)}</text>')

    lines.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path


def plot_grouped_bars(records: Sequence[EvaluationRecord], models: Sequence[str], benchmarks: Sequence[str],
                      metrics: Sequence[str], output_path: Path, title: str = 'VLM evaluation comparison') -> Path:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return _write_svg_bars(records, models, benchmarks, metrics, output_path, title)

    matrix = records_to_matrix(records)
    panels = _panel_pairs(benchmarks, metrics)
    if not panels:
        raise ValueError('No registered benchmark/metric pairs are available for visualization.')
    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig_width = max(7.0, 4.2 * ncols)
    fig_height = max(4.0, 3.2 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2', '#B279A2', '#FF9DA6', '#9D755D']

    for panel_idx, (benchmark, metric) in enumerate(panels):
        row_idx, col_idx = divmod(panel_idx, ncols)
        ax = axes[row_idx][col_idx]
        spec = metric_spec_for(benchmark, metric)
        values, has_value, _ = _display_values(matrix, models, benchmark, metric)
        bars = ax.bar(range(len(models)), values, color=[colors[i % len(colors)] for i in range(len(models))])
        ax.set_title(f'{benchmark} / {metric}')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.45)
        ax.set_axisbelow(True)
        ax.set_ylabel('Percent' if spec and spec.percentage else 'Value')
        if spec and spec.percentage:
            ax.set_ylim(0, max(100.0, max(values) * 1.12 if values else 100.0))
        elif values:
            upper = max(values) * 1.15 if max(values) > 0 else 1.0
            ax.set_ylim(0, upper)
        if spec and not spec.higher_is_better:
            ax.text(0.98, 0.94, 'lower is better', transform=ax.transAxes, ha='right', va='top', fontsize=8)
        if not has_value:
            ax.text(0.5, 0.5, 'missing', transform=ax.transAxes, ha='center', va='center', fontsize=11)
        for bar, value in zip(bars, values):
            if value == 0.0:
                continue
            label = f'{value:.1f}%' if spec and spec.percentage else f'{value:.3g}'
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8)

    for panel_idx in range(len(panels), nrows * ncols):
        row_idx, col_idx = divmod(panel_idx, ncols)
        axes[row_idx][col_idx].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
