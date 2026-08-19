#!/usr/bin/env python3
"""Collect, render, and verify the AI2D 200-image source analysis."""

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
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


HERE = Path(__file__).resolve().parent
REPORT_ROOT = HERE.parent
DATASET_DIR = REPORT_ROOT / "sub-dataset" / "AI2D"
REPORT_PATH = REPORT_ROOT / "AI2D.html"
REMOTE_HOST = "pudu_lrs_workspace"
REMOTE_PYTHON = "/tmp/vlm_dataset_src_anal_venv/bin/python"
REMOTE_ROOT = "/mnt/oss-data/luojunkun/stage1/dataset/ai2d/ai2d"
SEED = 2026080407
TARGET_COUNT = 200
CANDIDATE_COUNT = 400


REMOTE_EXTRACTOR = r'''
from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import sys
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image

cfg = json.loads(base64.urlsafe_b64decode(sys.argv[1].encode("ascii")))
ROOT = Path(cfg["root"])


def add_bytes(tar, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(payload))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


images = sorted(path for path in (ROOT / "images").iterdir() if path.is_file())
questions = {path.name.removesuffix(".json"): path for path in (ROOT / "questions").glob("*.json")}
annotations = {path.name.removesuffix(".json"): path for path in (ROOT / "annotations").glob("*.json")}
categories = json.loads((ROOT / "categories.json").read_text())
rng = random.Random(cfg["seed"])
candidate_indices = rng.sample(range(len(images)), cfg["candidate_count"])

selected = []
decisions = []
seen_hashes = set()
for draw_order, image_index in enumerate(candidate_indices):
    image_path = images[image_index]
    image_bytes = image_path.read_bytes()
    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest in seen_hashes:
        decisions.append({"draw_order": draw_order, "image_index": image_index, "source_path": f"images/{image_path.name}", "sha256": digest, "reason": "duplicate_media_sha256"})
        continue
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            width, height = image.size
            image_format = (image.format or "bin").lower()
            image_mode = image.mode
    except Exception as exc:
        decisions.append({"draw_order": draw_order, "image_index": image_index, "source_path": f"images/{image_path.name}", "reason": "image_decode_error", "detail": str(exc)[:240]})
        continue
    if len(selected) >= cfg["target_count"]:
        decisions.append({"draw_order": draw_order, "image_index": image_index, "source_path": f"images/{image_path.name}", "reason": "reserve_candidate_not_needed"})
        continue
    seen_hashes.add(digest)
    sample_id = f"ai2d-{len(selected) + 1:03d}"
    question_path = questions.get(image_path.name)
    annotation_path = annotations.get(image_path.name)
    question_data = json.loads(question_path.read_text()) if question_path else None
    annotation_data = json.loads(annotation_path.read_text()) if annotation_path else None
    question_items = list((question_data or {}).get("questions", {}).items())
    first_question = question_items[0] if question_items else None
    answer_text = None
    if first_question:
        answer_index = first_question[1].get("correctAnswer")
        answer_choices = first_question[1].get("answerTexts") or []
        if isinstance(answer_index, int) and 0 <= answer_index < len(answer_choices):
            answer_text = answer_choices[answer_index]
    selected.append({
        "sample_id": sample_id,
        "draw_order": draw_order,
        "population_index": image_index,
        "source_image_path": f"images/{image_path.name}",
        "media_archive_path": f"source-sample/ai2d/images/{image_path.name}",
        "question_archive_path": f"source-sample/ai2d/questions/{question_path.name}" if question_path else None,
        "annotation_archive_path": f"source-sample/ai2d/annotations/{annotation_path.name}" if annotation_path else None,
        "category": categories.get(image_path.name),
        "question_count": len(question_items),
        "first_question": first_question[0] if first_question else None,
        "first_answer_index": first_question[1].get("correctAnswer") if first_question else None,
        "first_answer_text": answer_text,
        "annotation_counts": {
            key: len((annotation_data or {}).get(key, {}))
            for key in ("arrows", "arrowHeads", "blobs", "text")
        },
        "sha256": digest,
        "byte_length": len(image_bytes),
        "image_format": image_format,
        "image_mode": image_mode,
        "width": width,
        "height": height,
        "_image_bytes": image_bytes,
        "_question_bytes": question_path.read_bytes() if question_path else None,
        "_annotation_bytes": annotation_path.read_bytes() if annotation_path else None,
    })

if len(selected) != cfg["target_count"]:
    raise RuntimeError(f"expected {cfg['target_count']} valid unique images, got {len(selected)}")

category_counts = Counter(item["category"] or "<missing>" for item in selected)
question_count_distribution = Counter(str(item["question_count"]) for item in selected)
summary = {
    "dataset": "AI2D",
    "source_root": str(ROOT),
    "seed": cfg["seed"],
    "sampling_method": "uniform random candidates over all source image files; deterministic rejection of invalid or duplicate media",
    "population_total": len(images),
    "source_question_files": len(questions),
    "source_annotation_files": len(annotations),
    "source_images_without_questions": len(images) - len(questions),
    "candidate_count": cfg["candidate_count"],
    "selected_count": len(selected),
    "invalid_or_duplicate_count": sum(item["reason"] != "reserve_candidate_not_needed" for item in decisions),
    "reserve_candidate_count": sum(item["reason"] == "reserve_candidate_not_needed" for item in decisions),
    "selected_without_questions": sum(item["question_count"] == 0 for item in selected),
    "category_counts": category_counts.most_common(),
    "question_count_distribution": sorted(question_count_distribution.items(), key=lambda value: int(value[0])),
    "image_width": {"min": min(item["width"] for item in selected), "max": max(item["width"] for item in selected)},
    "image_height": {"min": min(item["height"] for item in selected), "max": max(item["height"] for item in selected)},
    "read_only_source": True,
}
layout = [
    f"directory\t{subdir}\t0" for subdir in ("images", "questions", "annotations")
] + [
    f"file\t{name}\t{(ROOT / name).stat().st_size}" for name in ("README.txt", "license.txt", "categories.json")
] + [
    f"file-count\timages\t{len(images)}",
    f"file-count\tquestions\t{len(questions)}",
    f"file-count\tannotations\t{len(annotations)}",
]
schema = {
    "input": "images/{imageName}: PNG",
    "question_join": "questions/{imageName}.json (optional)",
    "annotation_join": "annotations/{imageName}.json",
    "category_join": "categories.json[imageName]",
    "question_fields": ["imageName", "questions.{question}.abcLabel", "answerTexts", "correctAnswer", "questionId"],
    "annotation_fields": ["arrows", "arrowHeads", "blobs", "text", "relationships"],
}

with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
    add_bytes(tar, "source-layout.txt", ("\n".join(layout) + "\n").encode("utf-8"))
    add_bytes(tar, "source-schema.json", json_bytes(schema))
    add_bytes(tar, "sampling-summary.json", json_bytes(summary))
    public = [{key: value for key, value in item.items() if not key.startswith("_")} for item in selected]
    add_bytes(tar, "sampling-manifest.jsonl", b"".join(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for item in public))
    add_bytes(tar, "sampling-candidate-decisions.jsonl", b"".join(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for item in decisions))
    for name in ("README.txt", "license.txt", "categories.json"):
        add_bytes(tar, f"source-sample/ai2d/{name}", (ROOT / name).read_bytes())
    for item in selected:
        add_bytes(tar, item["media_archive_path"], item["_image_bytes"])
        if item["_question_bytes"] is not None:
            add_bytes(tar, item["question_archive_path"], item["_question_bytes"])
        if item["_annotation_bytes"] is not None:
            add_bytes(tar, item["annotation_archive_path"], item["_annotation_bytes"])
'''


STYLE = """
:root{--ink:#202a33;--muted:#65717b;--line:#d7dde1;--paper:#fff;--wash:#f3f5f6;--accent:#08786f;--soft:#dff2ef;--warn:#9a4b06}*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}main{width:min(1180px,calc(100% - 32px));margin:auto;padding:34px 0 64px;min-width:0}header{padding:30px 0 26px;border-bottom:1px solid var(--line)}.eyebrow{color:var(--accent);font-weight:700}h1{font-size:clamp(30px,5vw,54px);line-height:1.12;margin:8px 0 14px;letter-spacing:0}h2{font-size:25px;margin:42px 0 14px}h3{font-size:18px;margin:22px 0 10px}p{max-width:86ch}.verdict{font-size:18px;max-width:80ch}code,pre{font-family:"SFMono-Regular",Consolas,monospace}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;word-break:break-word;font-size:12px}.facts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:24px 0}.fact{background:#fff;padding:18px}.fact b{display:block;font-size:25px;color:var(--accent)}.fact span{color:var(--muted)}.band{background:#fff;border-block:1px solid var(--line);padding:22px;margin:22px 0}.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.flow span{background:var(--soft);padding:8px 11px;border-radius:4px}.flow b{color:var(--muted)}.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;background:#fff}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 11px}.note{border-left:4px solid var(--warn);padding:12px 16px;background:#fff7ed}.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.sample{background:#fff;border:1px solid var(--line);min-width:0}.sample>a{display:block;aspect-ratio:4/3;background:#e8ecee;overflow:hidden}.sample img{width:100%;height:100%;object-fit:contain;display:block}.sample-body{padding:12px}.sample-head{display:flex;justify-content:space-between;gap:8px}.sample-head span{color:var(--accent);font-weight:700}.sample p{font-size:13px;min-height:64px;margin:7px 0;overflow-wrap:anywhere}dl{display:grid;grid-template-columns:48px minmax(0,1fr);gap:3px 8px;margin:0;font-size:12px}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}details{margin-top:9px}summary{cursor:pointer;color:var(--accent)}footer{margin-top:42px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted)}@media(max-width:880px){.facts,.gallery,.two{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){main{width:calc(100% - 20px);padding-top:18px}.facts,.gallery,.two{grid-template-columns:1fr}.sample p{min-height:0}}
"""


def safe_extract(path: Path) -> None:
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            candidate = PurePosixPath(member.name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(DATASET_DIR, filter="data")


def collect() -> None:
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty archive: {DATASET_DIR}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {"root": REMOTE_ROOT, "seed": SEED, "target_count": TARGET_COUNT, "candidate_count": CANDIDATE_COUNT}
    encoded = base64.urlsafe_b64encode(json.dumps(cfg).encode()).decode()
    command = f"ls -la /mnt/oss-data >/dev/null 2>&1 && PYTHONDONTWRITEBYTECODE=1 {REMOTE_PYTHON} - {shlex.quote(encoded)}"
    with tempfile.NamedTemporaryFile(prefix="ai2d-samples-", suffix=".tar", delete=False) as temporary:
        path = Path(temporary.name)
        process = subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE_HOST, command], input=REMOTE_EXTRACTOR.encode(), stdout=temporary, stderr=subprocess.PIPE, check=False)
    if process.returncode:
        DATASET_DIR.rmdir()
        raise RuntimeError(process.stderr.decode(errors="replace"))
    try:
        safe_extract(path)
    finally:
        path.unlink(missing_ok=True)


def manifest() -> list[dict]:
    return [json.loads(line) for line in (DATASET_DIR / "sampling-manifest.jsonl").read_text().splitlines() if line]


def render() -> None:
    summary = json.loads((DATASET_DIR / "sampling-summary.json").read_text())
    items = manifest()
    category_rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>" for name, count in summary["category_counts"])
    question_rows = "".join(f"<tr><td>{count}</td><td>{images}</td></tr>" for count, images in summary["question_count_distribution"])
    cards = []
    for item in items:
        media = f"sub-dataset/AI2D/{item['media_archive_path']}"
        details = html.escape(json.dumps({key: value for key, value in item.items() if key not in {"media_archive_path", "question_archive_path", "annotation_archive_path"}}, ensure_ascii=False, indent=2))
        q = item["first_question"] or "该图片没有 questions 文件"
        a = item["first_answer_text"] if item["first_answer_text"] is not None else "无问答输出"
        cards.append(f'''<article class="sample"><a href="{media}"><img loading="lazy" src="{media}" alt="{item['sample_id']} AI2D source sample"></a><div class="sample-body"><div class="sample-head"><strong>{item['sample_id']}</strong><span>{html.escape(item['category'] or '<missing>')}</span></div><p><strong>问：</strong>{html.escape(q)}<br><strong>答：</strong>{html.escape(str(a))}</p><dl><dt>尺寸</dt><dd>{item['width']} x {item['height']}</dd><dt>源文件</dt><dd><code>{html.escape(item['source_image_path'])}</code></dd></dl><details><summary>字段与追溯信息</summary><pre>{details}</pre></details></div></article>''')
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI2D 源数据集 200 张随机样本分析</title><style>{STYLE}</style></head><body><main>
<header><div class="eyebrow">VLM SOURCE DATASET / SAMPLE ANALYSIS 09</div><h1>AI2D：200 张全量随机样本分析</h1><p class="verdict"><strong>总判断：</strong>AI2D 以科学示意图为输入，同时提供图形元素多边形标注、图像类别，以及部分图片对应的多项选择问题。输出用整数 <code>correctAnswer</code> 指向 <code>answerTexts</code>；4,903 张图片中只有 4,563 张有问题文件，所以“无问答”是源数据真实状态，不是抽样错误。</p></header>
<section class="facts"><div class="fact"><b>{summary['population_total']:,}</b><span>全量图片</span></div><div class="fact"><b>200</b><span>有效唯一随机图片</span></div><div class="fact"><b>{summary['source_question_files']:,}</b><span>全量问题文件</span></div><div class="fact"><b>{summary['selected_without_questions']}</b><span>样本中无问题图片</span></div></section>
<section><h2>一、总体：输入、输出与原始组织</h2><p>源路径为 <code>{REMOTE_ROOT}</code>。目录以同名文件连接：<code>images/X.png</code> 对应 <code>annotations/X.png.json</code>，若有问答则再对应 <code>questions/X.png.json</code>；<code>categories.json</code> 按图片名给类别。归档在 <code>source-sample/ai2d/</code> 下保留这套目录关系。</p><div class="band"><div class="flow"><span>科学示意图 PNG</span><b>+</b><span>问题与四个选项</span><b>→</b><span>correctAnswer 整数</span><b>+</b><span>图形元素多边形</span></div></div><div class="two"><div><h3>输入</h3><p>图片与自然语言问题；问题记录还给出 <code>abcLabel</code> 和 <code>questionId</code>。</p></div><div><h3>输出</h3><p><code>correctAnswer</code> 是选项下标，答案文本需用它索引 <code>answerTexts</code>；annotations 另含 arrows、arrowHeads、blobs、text 等图形结构。</p></div></div><p class="note"><strong>证据边界：</strong>本报告仅按源 JSON 的整数索引展示答案；没有问题文件的图片不补写问题。源内 license.txt 明确限制非商业使用和再分发，商用前必须单独完成合规确认。</p></section>
<section><h2>二、抽样方法</h2><p>对 <code>images/</code> 中全部 4,903 张图片按文件名排序建立总体，以固定种子 {SEED} 等概率无放回抽取 400 个候选，再按候选顺序检查解码与 SHA256 唯一性，固定前 200 张。问题与标注只在图片入选后按同名键关联，因此不会改变图片总体。</p></section>
<section><h2>三、200 张样本分布</h2><p>宽度 {summary['image_width']['min']}–{summary['image_width']['max']} px，高度 {summary['image_height']['min']}–{summary['image_height']['max']} px。下表只描述固定样本。</p><div class="two"><div><h3>图像类别</h3><div class="table-wrap"><table><thead><tr><th>类别</th><th>图片</th></tr></thead><tbody>{category_rows}</tbody></table></div></div><div><h3>每图问题数</h3><div class="table-wrap"><table><thead><tr><th>问题数</th><th>图片</th></tr></thead><tbody>{question_rows}</tbody></table></div></div></div></section>
<section><h2>四、逐样本：完整 200 张归档</h2><p>图片、问题 JSON 和图形标注 JSON 均保持原始字节；展开卡片可查看类别、问题数、首题答案索引和标注元素计数。</p><div class="gallery">{''.join(cards)}</div></section><footer>只读源审计：远端仅执行目录枚举与文件读取；tar 从 stdout 回传，源目录未写入。</footer></main></body></html>'''
    REPORT_PATH.write_text(document, encoding="utf-8")


def verify() -> None:
    items = manifest()
    if len(items) != TARGET_COUNT or len({item["sha256"] for item in items}) != TARGET_COUNT:
        raise RuntimeError("manifest count or unique hash count is not 200")
    for item in items:
        payload = (DATASET_DIR / item["media_archive_path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise RuntimeError(f"digest mismatch: {item['sample_id']}")
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    report = REPORT_PATH.read_text()
    if report.count('<article class="sample"') != TARGET_COUNT:
        raise RuntimeError("report sample-card count is not 200")
    print(json.dumps({"dataset": "AI2D", "manifest_rows": 200, "unique_media_sha256": 200, "readable_media": 200, "report_sample_cards": 200}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "render", "verify", "all"))
    args = parser.parse_args()
    if args.command in {"collect", "all"}: collect()
    if args.command in {"render", "all"}: render()
    if args.command in {"verify", "all"}: verify()


if __name__ == "__main__":
    main()
