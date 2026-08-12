#!/usr/bin/env python3
"""Collect, render, and verify the 200-image COCO source analysis archive."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import random
import shlex
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


HERE = Path(__file__).resolve().parent
REPORT_ROOT = HERE.parent
DATASET_DIR = REPORT_ROOT / "sub-dataset" / "COCO"
REPORT_PATH = REPORT_ROOT / "COCO.html"
REMOTE_HOST = "pudu_lrs_workspace"
REMOTE_PYTHON = "/tmp/vlm_dataset_src_anal_venv/bin/python"
REMOTE_ROOT = "/mnt/oss-data/luojunkun/stage1/dataset/COCO/COCO-MODELSCOPE"
SEED = 2026080401
TARGET_COUNT = 200
CANDIDATE_COUNT = 400

SHARDS = {
    "train": [(f"data-{index:05d}-of-00039.arrow", 3006 if index < 23 else 3005) for index in range(39)],
    "test": [(f"data-{index:05d}-of-00002.arrow", 2500) for index in range(2)],
}


REMOTE_EXTRACTOR = r'''
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image

ROOT = Path("/mnt/oss-data/luojunkun/stage1/dataset/COCO/COCO-MODELSCOPE")
spec = json.loads(base64.urlsafe_b64decode(sys.argv[1].encode("ascii")))


def add_bytes(tar, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(payload))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


label_names = [line.strip() for line in (ROOT / "labels.txt").read_text().splitlines() if line.strip()]
candidates_by_file = defaultdict(list)
for item in spec["candidates"]:
    candidates_by_file[(item["split"], item["shard"])].append(item)

loaded = {}
schemas = {}
for (split, shard), targets in candidates_by_file.items():
    path = ROOT / split / shard
    wanted = {item["shard_row"]: item for item in targets}
    with path.open("rb") as stream:
        reader = ipc.RecordBatchStreamReader(stream)
        schemas[split] = reader.schema
        offset = 0
        for batch in reader:
            batch_end = offset + batch.num_rows
            for row_index in sorted(index for index in wanted if offset <= index < batch_end):
                loaded[wanted[row_index]["draw_order"]] = (wanted[row_index], batch.slice(row_index - offset, 1))
            offset = batch_end

selected = []
rejected = []
seen_hashes = set()
for draw_order in sorted(loaded):
    item, batch = loaded[draw_order]
    row = batch.to_pylist()[0]
    images = row.get("images") or []
    if len(images) != 1 or not images[0].get("bytes"):
        rejected.append({**item, "reason": "missing_or_non_single_image"})
        continue
    image_bytes = images[0]["bytes"]
    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest in seen_hashes:
        rejected.append({**item, "reason": "duplicate_media_sha256", "sha256": digest})
        continue
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            width, height = image.size
            image_format = (image.format or "bin").lower()
            image_mode = image.mode
    except Exception as exc:
        rejected.append({**item, "reason": "image_decode_error", "detail": str(exc)[:240]})
        continue
    if len(selected) >= spec["target_count"]:
        rejected.append({**item, "reason": "reserve_candidate_not_needed"})
        continue
    seen_hashes.add(digest)
    sample_number = len(selected) + 1
    sample_id = f"coco-{sample_number:03d}"
    extension = "jpg" if image_format in {"jpeg", "jpg"} else image_format
    label_ids = row.get("labels") or []
    selected.append({
        **item,
        "sample_id": sample_id,
        "media_archive_path": f"media/{sample_id}.{extension}",
        "record_archive_path": f"records/{sample_id}.json",
        "sha256": digest,
        "byte_length": len(image_bytes),
        "image_format": image_format,
        "image_mode": image_mode,
        "width": width,
        "height": height,
        "label_ids": label_ids,
        "label_names": [label_names[value] if 0 <= value < len(label_names) else f"unknown:{value}" for value in label_ids],
        "_image_bytes": image_bytes,
        "_batch": batch,
    })

if len(selected) != spec["target_count"]:
    raise RuntimeError(f"expected {spec['target_count']} valid unique images, got {len(selected)}")

split_counts = Counter(item["split"] for item in selected)
label_counts = Counter(name for item in selected for name in item["label_names"])
label_count_per_image = Counter(str(len(item["label_ids"])) for item in selected)

summary = {
    "dataset": "COCO",
    "source_root": str(ROOT),
    "seed": spec["seed"],
    "sampling_method": "uniform random candidates over all 122218 source rows; deterministic rejection of invalid or duplicate media",
    "population": spec["population"],
    "population_total": sum(spec["population"].values()),
    "candidate_count": len(spec["candidates"]),
    "selected_count": len(selected),
    "invalid_or_duplicate_count": sum(item["reason"] != "reserve_candidate_not_needed" for item in rejected),
    "reserve_candidate_count": sum(item["reason"] == "reserve_candidate_not_needed" for item in rejected),
    "selected_split_counts": dict(sorted(split_counts.items())),
    "label_count_per_image": dict(sorted(label_count_per_image.items(), key=lambda item: int(item[0]))),
    "top_labels": label_counts.most_common(30),
    "image_width": {"min": min(item["width"] for item in selected), "max": max(item["width"] for item in selected)},
    "image_height": {"min": min(item["height"] for item in selected), "max": max(item["height"] for item in selected)},
    "read_only_source": True,
}

layout = []
for path in sorted(ROOT.rglob("*")):
    relative = path.relative_to(ROOT)
    if len(relative.parts) > 2:
        continue
    kind = "directory" if path.is_dir() else "file"
    size = 0 if path.is_dir() else path.stat().st_size
    layout.append(f"{kind}\t{relative.as_posix()}\t{size}")

schema_payload = {
    "train": str(schemas.get("train", "not sampled")),
    "test": str(schemas.get("test", "not sampled")),
    "semantic_boundary": {
        "input": "images: list<struct<bytes: binary, path: null>>",
        "output_train": "labels: list<int64>",
        "output_test": "no labels field",
    },
}

with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
    add_bytes(tar, "source-layout.txt", ("\n".join(layout) + "\n").encode("utf-8"))
    add_bytes(tar, "source-schema.json", json_bytes(schema_payload))
    add_bytes(tar, "sampling-summary.json", json_bytes(summary))
    public_manifest = [{key: value for key, value in item.items() if not key.startswith("_")} for item in selected]
    add_bytes(tar, "sampling-manifest.jsonl", b"".join(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for item in public_manifest))
    add_bytes(tar, "sampling-candidate-decisions.jsonl", b"".join(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for item in rejected))
    for relative in ["README.md", "labels.txt", "train/dataset_info.json", "train/state.json", "test/dataset_info.json", "test/state.json"]:
        add_bytes(tar, f"source-metadata/{relative}", (ROOT / relative).read_bytes())
    split_batches = defaultdict(list)
    for item in selected:
        add_bytes(tar, item["media_archive_path"], item["_image_bytes"])
        record = {key: value for key, value in item.items() if not key.startswith("_")}
        record["raw_record_view"] = {
            "images": [{"bytes": f"@{item['media_archive_path']}", "path": None, "byte_length": item["byte_length"]}],
            **({"labels": item["label_ids"]} if item["split"] == "train" else {}),
        }
        add_bytes(tar, item["record_archive_path"], json_bytes(record))
        split_batches[item["split"]].append(item["_batch"])
    for split, batches in sorted(split_batches.items()):
        sink = io.BytesIO()
        with ipc.new_stream(sink, schemas[split]) as writer:
            for batch in batches:
                writer.write_batch(batch)
        add_bytes(tar, f"records/coco_{split}_sampled.arrow", sink.getvalue())
'''


def build_candidates() -> dict:
    population = {split: sum(count for _, count in shards) for split, shards in SHARDS.items()}
    total = sum(population.values())
    rng = random.Random(SEED)
    global_indices = rng.sample(range(total), CANDIDATE_COUNT)
    candidates = []
    for draw_order, global_index in enumerate(global_indices):
        remaining = global_index
        for split in ("train", "test"):
            split_count = population[split]
            if remaining >= split_count:
                remaining -= split_count
                continue
            for shard, shard_count in SHARDS[split]:
                if remaining >= shard_count:
                    remaining -= shard_count
                    continue
                candidates.append({
                    "draw_order": draw_order,
                    "population_index": global_index,
                    "split": split,
                    "shard": shard,
                    "shard_row": remaining,
                    "source_uri": f"{REMOTE_ROOT}/{split}/{shard}#row={remaining}",
                })
                break
            break
    assert len(candidates) == CANDIDATE_COUNT
    return {
        "dataset": "COCO",
        "seed": SEED,
        "target_count": TARGET_COUNT,
        "population": population,
        "candidates": candidates,
    }


def safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(destination, filter="data")


def collect() -> None:
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty archive: {DATASET_DIR}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    encoded_spec = base64.urlsafe_b64encode(json.dumps(build_candidates()).encode("utf-8")).decode("ascii")
    remote_command = f"ls -la /mnt/oss-data >/dev/null 2>&1 && PYTHONDONTWRITEBYTECODE=1 {REMOTE_PYTHON} - {shlex.quote(encoded_spec)}"
    with tempfile.NamedTemporaryFile(prefix="coco-samples-", suffix=".tar", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        process = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", REMOTE_HOST, remote_command],
            input=REMOTE_EXTRACTOR.encode("utf-8"),
            stdout=temporary,
            stderr=subprocess.PIPE,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
    try:
        safe_extract(temporary_path, DATASET_DIR)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_manifest() -> list[dict]:
    return [json.loads(line) for line in (DATASET_DIR / "sampling-manifest.jsonl").read_text().splitlines() if line]


def render() -> None:
    summary = json.loads((DATASET_DIR / "sampling-summary.json").read_text())
    manifest = load_manifest()
    top_labels = summary["top_labels"][:12]
    max_label_count = max((count for _, count in top_labels), default=1)
    bars = "".join(
        f'<div class="bar"><span>{html.escape(name)}</span><i style="width:{count / max_label_count * 100:.2f}%"></i><b>{count}</b></div>'
        for name, count in top_labels
    )
    cards = []
    for item in manifest:
        label_text = ", ".join(item["label_names"]) if item["label_names"] else "无标签（test）"
        record_json = html.escape(json.dumps({
            "split": item["split"],
            "shard": item["shard"],
            "row": item["shard_row"],
            "labels": item["label_ids"],
            "label_names": item["label_names"],
            "width": item["width"],
            "height": item["height"],
            "sha256": item["sha256"],
        }, ensure_ascii=False, indent=2))
        cards.append(f'''
        <article class="sample" id="{item['sample_id']}">
          <a href="sub-dataset/COCO/{item['media_archive_path']}"><img loading="lazy" src="sub-dataset/COCO/{item['media_archive_path']}" alt="{item['sample_id']} COCO source sample"></a>
          <div class="sample-body">
            <div class="sample-head"><strong>{item['sample_id']}</strong><span>{item['split']}</span></div>
            <p>{html.escape(label_text)}</p>
            <dl><dt>尺寸</dt><dd>{item['width']} x {item['height']}</dd><dt>源定位</dt><dd><code>{html.escape(item['shard'])}#row={item['shard_row']}</code></dd></dl>
            <details><summary>原始字段与追溯信息</summary><pre>{record_json}</pre></details>
          </div>
        </article>''')
    split_rows = "".join(f"<tr><td>{html.escape(split)}</td><td>{count:,}</td></tr>" for split, count in summary["selected_split_counts"].items())
    label_count_rows = "".join(f"<tr><td>{count}</td><td>{images}</td></tr>" for count, images in summary["label_count_per_image"].items())
    document = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>COCO 源数据集 200 张随机样本分析</title>
  <style>
    :root{{--ink:#1e2934;--muted:#64717d;--line:#d8dee3;--paper:#fff;--wash:#f4f6f7;--accent:#007f78;--accent-soft:#dff3f0;--warn:#a14b00}}
    *{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}
    main{{width:min(1180px,calc(100% - 32px));margin:0 auto;padding:34px 0 64px;min-width:0}} header{{padding:30px 0 26px;border-bottom:1px solid var(--line)}}
    .eyebrow{{color:var(--accent);font-weight:700}} h1{{font-size:clamp(30px,5vw,54px);line-height:1.12;margin:8px 0 14px;letter-spacing:0}} h2{{font-size:25px;margin:42px 0 14px}} h3{{font-size:18px;margin:24px 0 10px}}
    p{{max-width:86ch}} code,pre{{font-family:"SFMono-Regular",Consolas,monospace}} code{{overflow-wrap:anywhere}} pre{{white-space:pre-wrap;word-break:break-word;margin:10px 0 0;font-size:12px}}
    .verdict{{font-size:18px;max-width:78ch}} .facts{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:24px 0}}
    .fact{{background:var(--paper);padding:18px;min-width:0}} .fact b{{display:block;font-size:25px;color:var(--accent)}} .fact span{{color:var(--muted)}}
    .band{{background:var(--paper);border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:22px;margin:22px 0}} .flow{{display:flex;flex-wrap:wrap;gap:8px;align-items:center}} .flow span{{background:var(--accent-soft);padding:8px 11px;border-radius:4px}} .flow b{{color:var(--muted)}}
    .two{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}} .table-wrap{{overflow-x:auto;min-width:0}} table{{width:100%;border-collapse:collapse;background:var(--paper)}} th,td{{text-align:left;border-bottom:1px solid var(--line);padding:9px 11px}}
    .bar{{display:grid;grid-template-columns:minmax(100px,1.4fr) minmax(90px,3fr) 38px;gap:10px;align-items:center;margin:8px 0}} .bar i{{display:block;height:10px;background:var(--accent)}} .bar b{{text-align:right}}
    .note{{border-left:4px solid var(--warn);padding:12px 16px;background:#fff7ed}} .gallery{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
    .sample{{background:var(--paper);border:1px solid var(--line);min-width:0}} .sample>a{{display:block;aspect-ratio:4/3;background:#e8ecee;overflow:hidden}} .sample img{{width:100%;height:100%;object-fit:contain;display:block}} .sample-body{{padding:12px;min-width:0}}
    .sample-head{{display:flex;justify-content:space-between;gap:8px}} .sample-head span{{color:var(--accent);font-weight:700}} .sample p{{font-size:13px;min-height:42px;margin:7px 0;color:#384652}}
    dl{{display:grid;grid-template-columns:48px minmax(0,1fr);gap:3px 8px;margin:0;font-size:12px}} dt{{color:var(--muted)}} dd{{margin:0;min-width:0;overflow-wrap:anywhere}} details{{margin-top:9px}} summary{{cursor:pointer;color:var(--accent)}}
    footer{{margin-top:42px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted)}}
    @media(max-width:880px){{.facts{{grid-template-columns:repeat(2,minmax(0,1fr))}}.gallery{{grid-template-columns:repeat(2,minmax(0,1fr))}}.two{{grid-template-columns:minmax(0,1fr)}}}}
    @media(max-width:520px){{main{{width:min(100% - 20px,1180px);padding-top:18px}}.facts{{grid-template-columns:minmax(0,1fr)}}.gallery{{grid-template-columns:minmax(0,1fr)}}.sample p{{min-height:0}}}}
  </style>
</head>
<body><main>
  <header><div class="eyebrow">VLM SOURCE DATASET / SAMPLE ANALYSIS 01</div><h1>COCO：200 张全量随机样本分析</h1><p class="verdict"><strong>总判断：</strong>本地源是 Hugging Face Arrow 封装的类别监督数据。训练记录以图像字节为输入、类别 ID 列表为输出；test 只有图像，没有标签。它可以提供对象类别与多标签视觉语义监督，但原始字段不含 caption、bbox、image_id、segmentation、area 或 iscrowd，不能直接当作检测、grounding 或 caption SFT 数据。</p></header>
  <section class="facts"><div class="fact"><b>{summary['population_total']:,}</b><span>全量源记录</span></div><div class="fact"><b>{summary['selected_count']}</b><span>有效唯一随机图片</span></div><div class="fact"><b>{summary['seed']}</b><span>固定随机种子</span></div><div class="fact"><b>{summary['candidate_count']}</b><span>确定性候选队列</span></div></section>
  <section><h2>一、总体：这个源数据如何存储</h2><p>源路径为 <code>{html.escape(summary['source_root'])}</code>。train 使用 39 个 Arrow shard，test 使用 2 个 Arrow shard；图像以二进制字段嵌入记录，而不是单独保存为带原始文件名的 JPEG。归档中的媒体文件是从 Arrow 字节字段无损导出，SHA256 对应原始字节。</p>
    <div class="band"><div class="flow"><span>Arrow 记录</span><b>→</b><span>images[0].bytes</span><b>+</b><span>labels（仅 train）</span><b>→</b><span>图像到类别集合</span></div></div>
    <div class="two"><div><h3>输入</h3><p><code>images: list&lt;struct&lt;bytes: binary, path: null&gt;&gt;</code>。本轮 200 条均包含一张可解码图片。</p></div><div><h3>输出</h3><p>train 的 <code>labels: list&lt;int64&gt;</code> 表示一张图关联的 COCO 类别集合；test schema 没有 <code>labels</code> 字段。</p></div></div>
    <p class="note"><strong>证据边界：</strong>报告描述原始源记录可直接观察到的字段与可派生监督方向，不把类别列表补写成 bbox、caption 或计数答案。</p>
  </section>
  <section><h2>二、抽样：如何从全量得到 200 张</h2><p>对 train 117,218 行与 test 5,000 行组成的 122,218 行总体进行等概率、无放回随机候选抽取。候选按随机顺序解码，遇到坏图或媒体 SHA256 重复时记录拒绝原因并顺延，最终固定为 200 张。所有定位信息保存在 <a href="sub-dataset/COCO/sampling-manifest.jsonl">sampling-manifest.jsonl</a>。</p>
    <div class="two"><div class="table-wrap"><table><thead><tr><th>Split</th><th>入选图片</th></tr></thead><tbody>{split_rows}</tbody></table></div><div class="table-wrap"><table><thead><tr><th>每图标签数</th><th>图片数</th></tr></thead><tbody>{label_count_rows}</tbody></table></div></div>
  </section>
  <section><h2>三、分布：200 张样本呈现什么</h2><h3>高频类别</h3><div>{bars}</div><p>图片宽度范围 {summary['image_width']['min']}–{summary['image_width']['max']} px，高度范围 {summary['image_height']['min']}–{summary['image_height']['max']} px。这里的类别频次仅描述固定 200 张样本，不外推为全量类别分布。</p></section>
  <section><h2>四、逐样本：完整 200 张归档</h2><p>每张图都链接到无损归档媒体；展开卡片可查看 split、shard、row、类别 ID、尺寸和 SHA256。对应的单条 JSON 检视记录及按源 schema 保存的 Arrow 子集位于 <a href="sub-dataset/COCO/records/">records/</a>。</p><div class="gallery">{''.join(cards)}</div></section>
  <footer>只读源审计：源目录仅执行枚举、Arrow 读取与媒体字节读取；归档和报告只写入本地 docs 目录。</footer>
</main></body></html>'''
    REPORT_PATH.write_text(document, encoding="utf-8")


def verify() -> None:
    manifest = load_manifest()
    if len(manifest) != TARGET_COUNT:
        raise RuntimeError(f"manifest rows: {len(manifest)}")
    sample_ids = {item["sample_id"] for item in manifest}
    hashes = {item["sha256"] for item in manifest}
    if len(sample_ids) != TARGET_COUNT or len(hashes) != TARGET_COUNT:
        raise RuntimeError("sample IDs or media hashes are not unique")
    media_files = sorted((DATASET_DIR / "media").iterdir())
    if len(media_files) != TARGET_COUNT:
        raise RuntimeError(f"media files: {len(media_files)}")
    for item in manifest:
        path = DATASET_DIR / item["media_archive_path"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise RuntimeError(f"digest mismatch: {path}")
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    report = REPORT_PATH.read_text(encoding="utf-8")
    for item in manifest:
        relative = f"sub-dataset/COCO/{item['media_archive_path']}"
        if relative not in report:
            raise RuntimeError(f"report missing media reference: {relative}")
    print(json.dumps({
        "dataset": "COCO",
        "manifest_rows": len(manifest),
        "unique_media_sha256": len(hashes),
        "readable_media": len(media_files),
        "report": str(REPORT_PATH),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "render", "verify", "all"))
    args = parser.parse_args()
    if args.command in {"collect", "all"}:
        collect()
    if args.command in {"render", "all"}:
        render()
    if args.command in {"verify", "all"}:
        verify()


if __name__ == "__main__":
    main()
