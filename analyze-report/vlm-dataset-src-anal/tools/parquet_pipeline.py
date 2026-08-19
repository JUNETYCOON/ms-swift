#!/usr/bin/env python3
"""Collect, render, and verify 200-image analyses for Parquet VLM sources."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import shutil
import shlex
import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


HERE = Path(__file__).resolve().parent
REPORT_ROOT = HERE.parent
REMOTE_HOST = "pudu_lrs_workspace"
REMOTE_PYTHON = "/tmp/vlm_dataset_src_anal_venv/bin/python"
TARGET_COUNT = 200
CANDIDATE_COUNT = 400

CONFIGS = {
    "VQAv2": {
        "root": "/mnt/oss-data/luojunkun/stage1/dataset/VQAv2",
        "seed": 2026080402,
        "report_no": "02",
        "mode": "embedded",
        "question_field": "question",
        "answer_fields": ["multiple_choice_answer", "answers"],
        "category_fields": ["answer_type", "question_type"],
        "verdict": "本地源是以问答行为样本单位的视觉问答数据：输入为 COCO 图像与自然语言问题，输出在 train/validation 中是多人答案及聚合答案，test/testdev 只有问题、没有答案。它适合开放式 VQA 监督，但同一图片可对应多条问题，因此本报告的随机总体是全量问答记录，不是去重图片集合。",
        "storage": "全部 split 以 Hugging Face Parquet shard 保存，图像字节嵌入每条问答记录。相同 COCO 图片会随不同问题重复出现；归档阶段按媒体 SHA256 去重并顺延候选，最终保留 200 张不同图片及各自命中的问答记录。",
        "input_desc": "<code>image.bytes</code> 与 <code>question</code>，并带 <code>image_id</code>、<code>question_id</code> 等定位字段。",
        "output_desc": "train/validation 提供 <code>answers</code> 的多人答案列表和 <code>multiple_choice_answer</code> 聚合答案；test/testdev 对应字段为空。",
        "flow": ["Parquet 问答记录", "image.bytes + question", "多人答案 / 聚合答案"],
        "boundary": "test 与 testdev 没有可用真值；报告不会把空值解释成负样本，也不会从图片内容自行补答案。",
    },
    "GQA": {
        "root": "/mnt/oss-data/luojunkun/stage1/dataset/GQA",
        "seed": 2026080403,
        "report_no": "03",
        "mode": "gqa_join",
        "question_field": "question",
        "answer_fields": ["answer", "fullAnswer"],
        "category_fields": ["isBalanced", "types.structural", "types.semantic"],
        "verdict": "本地源把视觉与语言监督拆成两类 Parquet 表：图像表以 id 保存图片字节，指令表以 imageId 保存问题、答案、语义程序和对象标注。本报告从全量指令行随机抽样，再按 imageId 只读连接图像表；因此每张归档图片都带一条真实问答记录，而不是孤立图片。",
        "storage": "目录按 train、val、test、testdev、challenge、submission 及 all/balanced 变体拆分。<code>*_images</code> 是图像表，<code>*_instructions</code> 是问答表；抽样总体仅由问答表行构成，随后通过 <code>imageId → id</code> 找回媒体。",
        "input_desc": "图像表中的 <code>image.bytes</code> 与指令表中的 <code>question</code>，两表通过 <code>imageId</code> / <code>id</code> 连接。",
        "output_desc": "有标注 split 提供 <code>answer</code>、<code>fullAnswer</code>、结构/语义类型、对象对齐标注与 semantic program；challenge/submission 部分记录不含答案。",
        "flow": ["instructions 问答行", "imageId 连接 images 表", "question + answer / semantic program"],
        "boundary": "all 与 balanced 是重叠视图，不应把 24,206,801 个物理行直接理解为同等数量的独立图像或独立问题。",
    },
    "TextVQA": {
        "root": "/mnt/oss-data/luojunkun/stage1/dataset/textvqa",
        "seed": 2026080404,
        "report_no": "04",
        "mode": "embedded",
        "question_field": "question",
        "answer_fields": ["answers"],
        "category_fields": ["set_name"],
        "verdict": "这是面向场景文字理解的视觉问答源：输入是自然图像、问题和随记录提供的 OCR token，输出是多个人工答案。它能训练模型把可见文字与问题结合，但 OCR token 属于辅助元信息，不等于逐字框标注。",
        "storage": "train、validation、test 共 27 个 Parquet shard；每条记录内嵌图片字节，并保存图片尺寸、问题 token、OCR token、图像类别和答案列表。",
        "input_desc": "<code>image.bytes</code>、<code>question</code>、<code>question_tokens</code> 和 <code>ocr_tokens</code>。",
        "output_desc": "<code>answers: list&lt;string&gt;</code> 保存多人答案；test 记录的答案可能为空，应按 split 判断监督可用性。",
        "flow": ["场景图像", "question + OCR tokens", "多人文本答案"],
        "boundary": "OCR token 只有文本序列，本地 schema 没有与 token 对应的 bbox 或置信度，不能据此声称拥有文字定位监督。",
    },
    "ChartQA": {
        "root": "/mnt/oss-data/luojunkun/stage1/dataset/Chartqa",
        "seed": 2026080405,
        "report_no": "05",
        "mode": "embedded",
        "question_field": "query",
        "answer_fields": ["label"],
        "category_fields": ["human_or_machine"],
        "verdict": "这是图表问答数据：输入为图表图片与查询，输出为字符串答案列表，并用 human_or_machine 标记问题来源。它覆盖读数、比较、计数和简单算术，但答案只是最终字符串，本地字段不包含推理步骤或图表元素框。",
        "storage": "全量 32,719 条记录分布在 5 个 Parquet 文件中，图像字节直接嵌入记录；query、label 与 human_or_machine 和图片在同一行。",
        "input_desc": "<code>image.bytes</code> 与自然语言 <code>query</code>。",
        "output_desc": "<code>label: list&lt;string&gt;</code>，通常是一项最终答案；<code>human_or_machine</code> 是问题来源标记，不是答案质量分数。",
        "flow": ["图表图片", "query", "字符串答案 label"],
        "boundary": "仅从最终答案无法恢复可靠的计算过程；报告不把数值答案反推成未提供的 chain-of-thought。",
    },
    "Robo2VLM": {
        "root": "/mnt/oss-data/luojunkun/stage1/dataset/robo2vlm",
        "seed": 2026080406,
        "report_no": "06",
        "mode": "embedded",
        "question_field": "question",
        "answer_fields": ["choices", "correct_answer"],
        "category_fields": ["correct_answer"],
        "verdict": "这是机器人场景的多项选择视觉问答源：输入为机器人视角图片、问题和字符串化选项列表，输出字段是整数 correct_answer。整数究竟表示零基还是一基索引必须由数据约定确认；报告保留原始数字，并把按零基索引得到的文本仅标记为派生解释。",
        "storage": "全量 810,797 条记录分布在 314 个 Parquet shard；每行包含 id、question、choices 字符串、correct_answer 整数和内嵌图片字节。",
        "input_desc": "<code>image.bytes</code>、<code>question</code> 与字符串字段 <code>choices</code>。",
        "output_desc": "原始输出为 <code>correct_answer: int64</code>。归档额外给出按 Python 列表零基索引解析的候选文本，但不改写源字段。",
        "flow": ["机器人视角图像", "question + choices", "correct_answer 整数"],
        "boundary": "未找到本地字段直接声明索引基准；任何选项文本映射都是派生解释，不能替代原始 correct_answer。",
    },
}


REMOTE_INDEXER = r'''
from __future__ import annotations

import base64
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

cfg = json.loads(base64.urlsafe_b64decode(sys.argv[1].encode("ascii")))
root = Path(cfg["root"])


def open_parquet(path):
    last_error = None
    for attempt in range(12):
        try:
            return pq.ParquetFile(path)
        except FileNotFoundError as exc:
            last_error = exc
            time.sleep(min(0.5 * (attempt + 1), 2.0))
    raise last_error


all_parquet = sorted(root.rglob("*.parquet"))
if cfg["mode"] == "gqa_join":
    record_files = [path for path in all_parquet if "_instructions" in path.parent.name]
    image_files = [path for path in all_parquet if "_images" in path.parent.name]
else:
    record_files = all_parquet
    image_files = []
schema_counts = Counter()
shards = []
for path in record_files:
    parquet = open_parquet(path)
    schema_counts[str(parquet.schema_arrow)] += 1
    shards.append({"relative": path.relative_to(root).as_posix(), "rows": parquet.metadata.num_rows})
    parquet.close()
print(json.dumps({
    "all_parquet": [path.relative_to(root).as_posix() for path in all_parquet],
    "image_files": [path.relative_to(root).as_posix() for path in image_files],
    "record_shards": shards,
    "schema_counts": dict(schema_counts),
}, ensure_ascii=False))
'''


REMOTE_EXTRACTOR = r'''
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import random
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

cfg = json.loads(base64.urlsafe_b64decode(sys.argv[1].encode("ascii")))
ROOT = Path(cfg["root"])


def open_parquet(path):
    last_error = None
    for attempt in range(12):
        try:
            return pq.ParquetFile(path)
        except FileNotFoundError as exc:
            last_error = exc
            time.sleep(min(0.5 * (attempt + 1), 2.0))
    raise last_error


def add_bytes(tar, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(payload))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def split_name(path):
    parent = path.parent.name
    if "_instructions" in parent:
        return parent.split("_instructions", 1)[0]
    return path.name.split("-", 1)[0]


def nested_get(row, dotted):
    value = row
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def archive_view(value, media_path=None):
    if isinstance(value, bytes):
        return {"media_reference": media_path, "byte_length": len(value)}
    if isinstance(value, dict):
        return {key: archive_view(item, media_path) for key, item in value.items()}
    if isinstance(value, list):
        return [archive_view(item, media_path) for item in value]
    return value


def answer_preview(row):
    if cfg["dataset"] == "VQAv2":
        return row.get("multiple_choice_answer") or "无真值"
    if cfg["dataset"] == "GQA":
        return row.get("answer") or row.get("fullAnswer") or "无真值"
    if cfg["dataset"] == "TextVQA":
        answers = row.get("answers") or []
        return ", ".join(str(value) for value in answers[:6]) or "无真值"
    if cfg["dataset"] == "ChartQA":
        return ", ".join(str(value) for value in (row.get("label") or [])) or "无真值"
    if cfg["dataset"] == "Robo2VLM":
        raw_choices = row.get("choices")
        choices = None
        try:
            parsed = ast.literal_eval(raw_choices) if isinstance(raw_choices, str) else raw_choices
            if isinstance(parsed, list):
                choices = parsed
        except (SyntaxError, ValueError):
            pass
        index = row.get("correct_answer")
        derived = choices[index] if choices and isinstance(index, int) and 0 <= index < len(choices) else None
        return f"raw={index}; zero-based interpretation={derived!r}"
    return ""


all_parquet = [ROOT / relative for relative in cfg["index"]["all_parquet"]]
image_files = [ROOT / relative for relative in cfg["index"]["image_files"]]
record_files = [ROOT / item["relative"] for item in cfg["index"]["record_shards"]]
shards = [(ROOT / item["relative"], item["relative"], item["rows"]) for item in cfg["index"]["record_shards"]]
schema_counts = Counter(cfg["index"]["schema_counts"])
population_by_split = Counter()
for path, relative, rows in shards:
    population_by_split[split_name(path)] += rows

population_total = sum(rows for _, _, rows in shards)
if population_total < cfg["candidate_count"]:
    raise RuntimeError(f"population too small: {population_total}")
rng = random.Random(cfg["seed"])
global_indices = rng.sample(range(population_total), cfg["candidate_count"])

candidates_by_file = defaultdict(list)
for draw_order, global_index in enumerate(global_indices):
    if cfg.get("batch_mode") and not (cfg["candidate_start"] <= draw_order < cfg["candidate_end"]):
        continue
    remaining = global_index
    for path, relative, rows in shards:
        if remaining >= rows:
            remaining -= rows
            continue
        candidates_by_file[path].append({
            "draw_order": draw_order,
            "population_index": global_index,
            "split": split_name(path),
            "shard": relative,
            "shard_row": remaining,
        })
        break

loaded = {}
for path, targets in candidates_by_file.items():
    parquet = open_parquet(path)
    by_row = {item["shard_row"]: item for item in targets}
    offset = 0
    for group_index in range(parquet.num_row_groups):
        group_rows = parquet.metadata.row_group(group_index).num_rows
        wanted = [index for index in by_row if offset <= index < offset + group_rows]
        if wanted:
            table = parquet.read_row_group(group_index)
            for index in wanted:
                item = by_row[index]
                one = table.slice(index - offset, 1)
                loaded[item["draw_order"]] = {"candidate": item, "table": one, "row": one.to_pylist()[0]}
        offset += group_rows
    parquet.close()

image_rows = {}
if cfg["mode"] == "gqa_join":
    wanted_ids = {str(item["row"].get("imageId")) for item in loaded.values() if item["row"].get("imageId") is not None}
    for path in image_files:
        parquet = open_parquet(path)
        for group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(group_index, columns=["id", "image"])
            ids = table.column("id").to_pylist()
            for row_index, image_id in enumerate(ids):
                key = str(image_id)
                if key in wanted_ids and key not in image_rows:
                    image_rows[key] = {
                        "table": table.slice(row_index, 1),
                        "row": table.slice(row_index, 1).to_pylist()[0],
                        "shard": path.relative_to(ROOT).as_posix(),
                        "shard_row_group": group_index,
                        "row_in_group": row_index,
                    }
        parquet.close()
        if wanted_ids.issubset(image_rows):
            break

selected = []
decisions = []
seen_hashes = set()
for draw_order in sorted(loaded):
    loaded_item = loaded[draw_order]
    candidate = loaded_item["candidate"]
    row = loaded_item["row"]
    image_lookup = None
    if cfg["mode"] == "gqa_join":
        image_lookup = image_rows.get(str(row.get("imageId")))
        image = image_lookup["row"].get("image") if image_lookup else None
    else:
        image = row.get("image")
    image_bytes = image.get("bytes") if isinstance(image, dict) else None
    if not image_bytes:
        decisions.append({**candidate, "reason": "missing_image_bytes", "image_id": row.get("imageId")})
        continue
    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest in seen_hashes:
        decisions.append({**candidate, "reason": "duplicate_media_sha256", "sha256": digest})
        continue
    try:
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            decoded.load()
            width, height = decoded.size
            image_format = (decoded.format or "bin").lower()
            image_mode = decoded.mode
    except Exception as exc:
        decisions.append({**candidate, "reason": "image_decode_error", "detail": str(exc)[:240]})
        continue
    if not cfg.get("batch_mode") and len(selected) >= cfg["target_count"]:
        decisions.append({**candidate, "reason": "reserve_candidate_not_needed"})
        continue
    seen_hashes.add(digest)
    sample_id = f"{cfg['slug']}-draw-{draw_order + 1:03d}" if cfg.get("batch_mode") else f"{cfg['slug']}-{len(selected) + 1:03d}"
    extension = "jpg" if image_format in {"jpeg", "jpg"} else image_format
    media_path = f"media/{sample_id}.{extension}"
    public = {
        **candidate,
        "sample_id": sample_id,
        "source_uri": f"{ROOT}/{candidate['shard']}#row={candidate['shard_row']}",
        "media_archive_path": media_path,
        "record_archive_path": f"records/{sample_id}.json",
        "sha256": digest,
        "byte_length": len(image_bytes),
        "image_format": image_format,
        "image_mode": image_mode,
        "width": width,
        "height": height,
        "question": row.get(cfg["question_field"]),
        "answer_preview": answer_preview(row),
        "category_values": {field: nested_get(row, field) for field in cfg["category_fields"]},
    }
    if image_lookup:
        public["joined_image_source"] = {
            "image_id": row.get("imageId"),
            "shard": image_lookup["shard"],
            "row_group": image_lookup["shard_row_group"],
            "row_in_group": image_lookup["row_in_group"],
        }
    selected.append({
        "public": public,
        "image_bytes": image_bytes,
        "logical_table": loaded_item["table"],
        "image_table": image_lookup["table"] if image_lookup else None,
        "raw_record": row,
        "raw_image_record": image_lookup["row"] if image_lookup else None,
    })

if not cfg.get("batch_mode") and len(selected) != cfg["target_count"]:
    raise RuntimeError(f"expected {cfg['target_count']} valid unique images, got {len(selected)}")

split_counts = Counter(item["public"]["split"] for item in selected)
category_counts = {}
for field in cfg["category_fields"]:
    counts = Counter()
    for item in selected:
        value = item["public"]["category_values"].get(field)
        if isinstance(value, list):
            counts.update(str(part) for part in value)
        else:
            counts[str(value) if value is not None else "<missing>"] += 1
    category_counts[field] = counts.most_common(20)

question_lengths = [len(str(item["public"].get("question") or "")) for item in selected]
answer_available = sum(item["public"]["answer_preview"] not in {"", "无真值"} for item in selected)
summary = {
    "dataset": cfg["dataset"],
    "source_root": str(ROOT),
    "seed": cfg["seed"],
    "sampling_method": "uniform random candidates over all logical source records; deterministic rejection of invalid or duplicate media",
    "population_by_split": dict(sorted(population_by_split.items())),
    "population_total": population_total,
    "source_parquet_files": len(all_parquet),
    "logical_record_files": len(record_files),
    "image_table_files": len(image_files),
    "candidate_count": cfg["candidate_count"],
    "selected_count": len(selected),
    "invalid_or_duplicate_count": sum(item["reason"] != "reserve_candidate_not_needed" for item in decisions),
    "reserve_candidate_count": sum(item["reason"] == "reserve_candidate_not_needed" for item in decisions),
    "selected_split_counts": dict(sorted(split_counts.items())),
    "category_counts": category_counts,
    "question_length_chars": {"min": min(question_lengths), "max": max(question_lengths), "mean": round(sum(question_lengths) / len(question_lengths), 2)},
    "answer_available_count": answer_available,
    "image_width": {"min": min(item["public"]["width"] for item in selected), "max": max(item["public"]["width"] for item in selected)},
    "image_height": {"min": min(item["public"]["height"] for item in selected), "max": max(item["public"]["height"] for item in selected)},
    "read_only_source": True,
}

layout_lines = []
for path in all_parquet:
    stat = path.stat()
    layout_lines.append(f"file\t{path.relative_to(ROOT).as_posix()}\t{stat.st_size}")
schema_payload = {
    "logical_record_schema_variants": [{"file_count": count, "schema": schema} for schema, count in schema_counts.items()],
    "join_contract": "instructions.imageId -> images.id" if cfg["mode"] == "gqa_join" else "image bytes embedded in each logical record",
}

with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
    add_bytes(tar, "source-layout.txt", ("\n".join(layout_lines) + "\n").encode("utf-8"))
    add_bytes(tar, "source-schema.json", json_bytes(schema_payload))
    add_bytes(tar, "sampling-summary.json", json_bytes(summary))
    add_bytes(tar, "sampling-manifest.jsonl", b"".join(json.dumps(item["public"], ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for item in selected))
    add_bytes(tar, "sampling-candidate-decisions.jsonl", b"".join(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for item in decisions))
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix == ".parquet" or path.stat().st_size > 2_000_000:
            continue
        if path.name.lower() not in {"readme.md", ".gitattributes", "dataset_info.json", "state.json"}:
            continue
        add_bytes(tar, f"source-metadata/{path.relative_to(ROOT).as_posix()}", path.read_bytes())
    logical_groups = defaultdict(list)
    image_groups = defaultdict(list)
    for item in selected:
        public = item["public"]
        add_bytes(tar, public["media_archive_path"], item["image_bytes"])
        record_view = {
            **public,
            "raw_record": archive_view(item["raw_record"], public["media_archive_path"]),
        }
        if item["raw_image_record"] is not None:
            record_view["joined_image_record"] = archive_view(item["raw_image_record"], public["media_archive_path"])
        add_bytes(tar, public["record_archive_path"], json_bytes(record_view))
        if cfg.get("batch_mode"):
            sink = io.BytesIO()
            pq.write_table(item["logical_table"], sink)
            add_bytes(tar, f"records/{public['sample_id']}.parquet", sink.getvalue())
            if item["image_table"] is not None:
                sink = io.BytesIO()
                pq.write_table(item["image_table"], sink)
                add_bytes(tar, f"records/{public['sample_id']}.image.parquet", sink.getvalue())
        logical_groups[str(item["logical_table"].schema)].append(item["logical_table"])
        if item["image_table"] is not None:
            image_groups[str(item["image_table"].schema)].append(item["image_table"])
    if not cfg.get("batch_mode"):
        for index, tables in enumerate(logical_groups.values(), 1):
            sink = io.BytesIO()
            pq.write_table(pa.concat_tables(tables), sink)
            add_bytes(tar, f"records/logical-records-schema-{index}.parquet", sink.getvalue())
        for index, tables in enumerate(image_groups.values(), 1):
            sink = io.BytesIO()
            pq.write_table(pa.concat_tables(tables), sink)
            add_bytes(tar, f"records/image-records-schema-{index}.parquet", sink.getvalue())
'''


STYLE = """
:root{--ink:#202a33;--muted:#65717b;--line:#d7dde1;--paper:#fff;--wash:#f3f5f6;--accent:#08786f;--accent-soft:#dff2ef;--warn:#9a4b06}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}
main{width:min(1180px,calc(100% - 32px));margin:0 auto;padding:34px 0 64px;min-width:0}header{padding:30px 0 26px;border-bottom:1px solid var(--line)}
.eyebrow{color:var(--accent);font-weight:700}h1{font-size:clamp(30px,5vw,54px);line-height:1.12;margin:8px 0 14px;letter-spacing:0}h2{font-size:25px;margin:42px 0 14px}h3{font-size:18px;margin:24px 0 10px}p{max-width:86ch}
code,pre{font-family:"SFMono-Regular",Consolas,monospace}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;word-break:break-word;margin:10px 0 0;font-size:12px}.verdict{font-size:18px;max-width:80ch}
.facts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:24px 0}.fact{background:var(--paper);padding:18px;min-width:0}.fact b{display:block;font-size:25px;color:var(--accent)}.fact span{color:var(--muted)}
.band{background:var(--paper);border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:22px;margin:22px 0}.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.flow span{background:var(--accent-soft);padding:8px 11px;border-radius:4px}.flow b{color:var(--muted)}
.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}.table-wrap{overflow-x:auto;min-width:0}table{width:100%;border-collapse:collapse;background:var(--paper)}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px 11px}.note{border-left:4px solid var(--warn);padding:12px 16px;background:#fff7ed}
.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.sample{background:var(--paper);border:1px solid var(--line);min-width:0}.sample>a{display:block;aspect-ratio:4/3;background:#e8ecee;overflow:hidden}.sample img{width:100%;height:100%;object-fit:contain;display:block}.sample-body{padding:12px;min-width:0}.sample-head{display:flex;justify-content:space-between;gap:8px}.sample-head span{color:var(--accent);font-weight:700}.sample p{font-size:13px;min-height:64px;margin:7px 0;color:#384652;overflow-wrap:anywhere}dl{display:grid;grid-template-columns:48px minmax(0,1fr);gap:3px 8px;margin:0;font-size:12px}dt{color:var(--muted)}dd{margin:0;min-width:0;overflow-wrap:anywhere}details{margin-top:9px}summary{cursor:pointer;color:var(--accent)}footer{margin-top:42px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted)}
@media(max-width:880px){.facts{grid-template-columns:repeat(2,minmax(0,1fr))}.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}.two{grid-template-columns:minmax(0,1fr)}}@media(max-width:520px){main{width:min(100% - 20px,1180px);padding-top:18px}.facts,.gallery{grid-template-columns:minmax(0,1fr)}.sample p{min-height:0}}
"""


def dataset_dir(name: str) -> Path:
    return REPORT_ROOT / "sub-dataset" / name


def report_path(name: str) -> Path:
    return REPORT_ROOT / f"{name}.html"


def safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(destination, filter="data")


def collect(name: str) -> None:
    cfg = {**CONFIGS[name], "dataset": name, "slug": name.lower().replace("v2", "v2"), "target_count": TARGET_COUNT, "candidate_count": CANDIDATE_COUNT}
    destination = dataset_dir(name)
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty archive: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    encoded_cfg = base64.urlsafe_b64encode(json.dumps(cfg, ensure_ascii=False).encode("utf-8")).decode("ascii")
    index_command = f"ls -la /mnt/oss-data >/dev/null 2>&1 && PYTHONDONTWRITEBYTECODE=1 {REMOTE_PYTHON} - {shlex.quote(encoded_cfg)}"
    index_process = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", REMOTE_HOST, index_command],
        input=REMOTE_INDEXER.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if index_process.returncode:
        destination.rmdir()
        raise RuntimeError(index_process.stderr.decode("utf-8", errors="replace"))
    cfg["index"] = json.loads(index_process.stdout.decode("utf-8"))
    with tempfile.TemporaryDirectory(prefix=f"{name}-batches-") as temporary_root_value:
        temporary_root = Path(temporary_root_value)
        batch_dirs = []
        for start in range(0, CANDIDATE_COUNT, 100):
            batch_cfg = {**cfg, "batch_mode": True, "candidate_start": start, "candidate_end": start + 100}
            encoded_cfg = base64.urlsafe_b64encode(json.dumps(batch_cfg, ensure_ascii=False).encode("utf-8")).decode("ascii")
            remote_command = f"ls -la /mnt/oss-data >/dev/null 2>&1 && PYTHONDONTWRITEBYTECODE=1 {REMOTE_PYTHON} - {shlex.quote(encoded_cfg)}"
            archive_path = temporary_root / f"batch-{start:03d}.tar"
            with archive_path.open("wb") as archive:
                process = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", REMOTE_HOST, remote_command],
                    input=REMOTE_EXTRACTOR.encode("utf-8"),
                    stdout=archive,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            if process.returncode:
                destination.rmdir()
                raise RuntimeError(process.stderr.decode("utf-8", errors="replace"))
            batch_dir = temporary_root / f"batch-{start:03d}"
            batch_dir.mkdir()
            safe_extract(archive_path, batch_dir)
            archive_path.unlink()
            batch_dirs.append(batch_dir)

        valid_candidates = []
        decisions = []
        for batch_dir in batch_dirs:
            valid_candidates.extend(
                (json.loads(line), batch_dir)
                for line in (batch_dir / "sampling-manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            )
            decisions.extend(
                json.loads(line)
                for line in (batch_dir / "sampling-candidate-decisions.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            )
        selected = []
        seen_hashes = set()
        for item, batch_dir in sorted(valid_candidates, key=lambda value: value[0]["draw_order"]):
            if item["sha256"] in seen_hashes:
                decisions.append({"draw_order": item["draw_order"], "shard": item["shard"], "shard_row": item["shard_row"], "sha256": item["sha256"], "reason": "duplicate_media_sha256_across_batches"})
            elif len(selected) < TARGET_COUNT:
                seen_hashes.add(item["sha256"])
                selected.append((item, batch_dir))
            else:
                decisions.append({"draw_order": item["draw_order"], "shard": item["shard"], "shard_row": item["shard_row"], "reason": "reserve_candidate_not_needed"})
        if len(selected) != TARGET_COUNT:
            destination.rmdir()
            raise RuntimeError(f"expected {TARGET_COUNT} valid unique images, got {len(selected)}")

        first_batch = batch_dirs[0]
        shutil.copy2(first_batch / "source-layout.txt", destination / "source-layout.txt")
        shutil.copy2(first_batch / "source-schema.json", destination / "source-schema.json")
        if (first_batch / "source-metadata").exists():
            shutil.copytree(first_batch / "source-metadata", destination / "source-metadata")
        (destination / "media").mkdir()
        (destination / "records").mkdir()
        for item, batch_dir in selected:
            shutil.copy2(batch_dir / item["media_archive_path"], destination / item["media_archive_path"])
            shutil.copy2(batch_dir / item["record_archive_path"], destination / item["record_archive_path"])
            sample_id = item["sample_id"]
            shutil.copy2(batch_dir / "records" / f"{sample_id}.parquet", destination / "records" / f"{sample_id}.parquet")
            image_record = batch_dir / "records" / f"{sample_id}.image.parquet"
            if image_record.exists():
                shutil.copy2(image_record, destination / "records" / image_record.name)

        manifest = [item for item, _ in selected]
        (destination / "sampling-manifest.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in manifest), encoding="utf-8")
        decisions.sort(key=lambda item: item["draw_order"])
        (destination / "sampling-candidate-decisions.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in decisions), encoding="utf-8")

        base_summary = json.loads((first_batch / "sampling-summary.json").read_text(encoding="utf-8"))
        split_counts = Counter(item["split"] for item in manifest)
        category_counts = {}
        for field in cfg["category_fields"]:
            counts = Counter()
            for item in manifest:
                value = item["category_values"].get(field)
                if isinstance(value, list):
                    counts.update(str(part) for part in value)
                else:
                    counts[str(value) if value is not None else "<missing>"] += 1
            category_counts[field] = counts.most_common(20)
        question_lengths = [len(str(item.get("question") or "")) for item in manifest]
        summary = {
            **base_summary,
            "candidate_count": CANDIDATE_COUNT,
            "selected_count": TARGET_COUNT,
            "invalid_or_duplicate_count": sum(item["reason"] != "reserve_candidate_not_needed" for item in decisions),
            "reserve_candidate_count": sum(item["reason"] == "reserve_candidate_not_needed" for item in decisions),
            "selected_split_counts": dict(sorted(split_counts.items())),
            "category_counts": category_counts,
            "question_length_chars": {"min": min(question_lengths), "max": max(question_lengths), "mean": round(sum(question_lengths) / len(question_lengths), 2)},
            "answer_available_count": sum(item["answer_preview"] not in {"", "无真值"} for item in manifest),
            "image_width": {"min": min(item["width"] for item in manifest), "max": max(item["width"] for item in manifest)},
            "image_height": {"min": min(item["height"] for item in manifest), "max": max(item["height"] for item in manifest)},
            "remote_batch_count": len(batch_dirs),
        }
        (destination / "sampling-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(name: str) -> list[dict]:
    path = dataset_dir(name) / "sampling-manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render(name: str) -> None:
    cfg = CONFIGS[name]
    summary = json.loads((dataset_dir(name) / "sampling-summary.json").read_text(encoding="utf-8"))
    manifest = load_manifest(name)
    split_rows = "".join(f"<tr><td>{html.escape(split)}</td><td>{count:,}</td></tr>" for split, count in summary["selected_split_counts"].items())
    population_rows = "".join(f"<tr><td>{html.escape(split)}</td><td>{count:,}</td></tr>" for split, count in summary["population_by_split"].items())
    category_sections = []
    for field, values in summary["category_counts"].items():
        rows = "".join(f"<tr><td>{html.escape(str(value))}</td><td>{count}</td></tr>" for value, count in values)
        category_sections.append(f'<div><h3>{html.escape(field)}</h3><div class="table-wrap"><table><thead><tr><th>值</th><th>样本数</th></tr></thead><tbody>{rows}</tbody></table></div></div>')
    cards = []
    for item in manifest:
        details = {
            "split": item["split"],
            "logical_record": f"{item['shard']}#row={item['shard_row']}",
            "joined_image_source": item.get("joined_image_source"),
            "question": item.get("question"),
            "answer_preview": item.get("answer_preview"),
            "category_values": item.get("category_values"),
            "width": item["width"],
            "height": item["height"],
            "sha256": item["sha256"],
        }
        media = f"sub-dataset/{name}/{item['media_archive_path']}"
        cards.append(f'''<article class="sample" id="{html.escape(item['sample_id'])}">
          <a href="{html.escape(media)}"><img loading="lazy" src="{html.escape(media)}" alt="{html.escape(item['sample_id'])} {html.escape(name)} source sample"></a>
          <div class="sample-body"><div class="sample-head"><strong>{html.escape(item['sample_id'])}</strong><span>{html.escape(item['split'])}</span></div>
          <p><strong>问：</strong>{html.escape(str(item.get('question') or '<missing>'))}<br><strong>答：</strong>{html.escape(str(item.get('answer_preview') or '<missing>'))}</p>
          <dl><dt>尺寸</dt><dd>{item['width']} x {item['height']}</dd><dt>源定位</dt><dd><code>{html.escape(item['shard'])}#row={item['shard_row']}</code></dd></dl>
          <details><summary>字段与追溯信息</summary><pre>{html.escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre></details></div></article>''')
    flow = "<b>→</b>".join(f"<span>{html.escape(part)}</span>" for part in cfg["flow"])
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(name)} 源数据集 200 张随机样本分析</title><style>{STYLE}</style></head>
<body><main><header><div class="eyebrow">VLM SOURCE DATASET / SAMPLE ANALYSIS {cfg['report_no']}</div><h1>{html.escape(name)}：200 张全量随机样本分析</h1><p class="verdict"><strong>总判断：</strong>{cfg['verdict']}</p></header>
<section class="facts"><div class="fact"><b>{summary['population_total']:,}</b><span>全量逻辑记录</span></div><div class="fact"><b>{summary['selected_count']}</b><span>有效唯一随机图片</span></div><div class="fact"><b>{summary['seed']}</b><span>固定随机种子</span></div><div class="fact"><b>{summary['candidate_count']}</b><span>确定性候选队列</span></div></section>
<section><h2>一、总体：输入、输出与存储</h2><p>源路径为 <code>{html.escape(summary['source_root'])}</code>。{cfg['storage']}</p><div class="band"><div class="flow">{flow}</div></div><div class="two"><div><h3>输入</h3><p>{cfg['input_desc']}</p></div><div><h3>输出</h3><p>{cfg['output_desc']}</p></div></div><p class="note"><strong>证据边界：</strong>{cfg['boundary']}</p></section>
<section><h2>二、全量与抽样方法</h2><p>先读取全部 Parquet footer 建立 {summary['population_total']:,} 条逻辑记录的全量索引，再以固定种子等概率、无放回抽取 {summary['candidate_count']} 个候选。候选按抽取顺序读取；坏图、缺图或媒体 SHA256 重复时记录原因并顺延，直至得到 200 张不同且可解码的图片。抽样定位与拒绝记录分别见 <a href="sub-dataset/{html.escape(name)}/sampling-manifest.jsonl">manifest</a> 和 <a href="sub-dataset/{html.escape(name)}/sampling-candidate-decisions.jsonl">candidate decisions</a>。</p><div class="two"><div><h3>全量逻辑记录</h3><div class="table-wrap"><table><thead><tr><th>Split/视图</th><th>行数</th></tr></thead><tbody>{population_rows}</tbody></table></div></div><div><h3>200 张入选样本</h3><div class="table-wrap"><table><thead><tr><th>Split/视图</th><th>图片数</th></tr></thead><tbody>{split_rows}</tbody></table></div></div></div></section>
<section><h2>三、200 张样本分布</h2><p>图片宽度 {summary['image_width']['min']}–{summary['image_width']['max']} px，高度 {summary['image_height']['min']}–{summary['image_height']['max']} px；问题长度 {summary['question_length_chars']['min']}–{summary['question_length_chars']['max']} 字符，均值 {summary['question_length_chars']['mean']}。有可展示答案的样本为 {summary['answer_available_count']}/200。以下频次仅描述固定样本，不外推为全量分布。</p><div class="two">{''.join(category_sections)}</div></section>
<section><h2>四、逐样本：完整 200 张归档</h2><p>每张图片均为源 Parquet 内嵌字节的无损导出；单条 JSON 保留原字段检视视图，<code>records/</code> 中的 Parquet 子集保持源 schema。点击图片可打开归档原图。</p><div class="gallery">{''.join(cards)}</div></section>
<footer>只读源审计：远端只执行目录枚举、Parquet footer/row-group 读取和媒体字节读取；tar 从 stdout 回传，报告与归档仅写入本地 docs 目录。</footer></main></body></html>'''
    report_path(name).write_text(document, encoding="utf-8")


def verify(name: str) -> None:
    manifest = load_manifest(name)
    if len(manifest) != TARGET_COUNT:
        raise RuntimeError(f"manifest rows: {len(manifest)}")
    hashes = {item["sha256"] for item in manifest}
    sample_ids = {item["sample_id"] for item in manifest}
    if len(hashes) != TARGET_COUNT or len(sample_ids) != TARGET_COUNT:
        raise RuntimeError("sample IDs or media hashes are not unique")
    media_files = sorted((dataset_dir(name) / "media").iterdir())
    if len(media_files) != TARGET_COUNT:
        raise RuntimeError(f"media files: {len(media_files)}")
    for item in manifest:
        path = dataset_dir(name) / item["media_archive_path"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise RuntimeError(f"digest mismatch: {path}")
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
    report = report_path(name).read_text(encoding="utf-8")
    if report.count('<article class="sample"') != TARGET_COUNT:
        raise RuntimeError("report sample-card count is not 200")
    for item in manifest:
        relative = f"sub-dataset/{name}/{item['media_archive_path']}"
        if relative not in report:
            raise RuntimeError(f"report missing media reference: {relative}")
    print(json.dumps({"dataset": name, "manifest_rows": len(manifest), "unique_media_sha256": len(hashes), "readable_media": len(media_files), "report_sample_cards": TARGET_COUNT}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=tuple(CONFIGS))
    parser.add_argument("command", choices=("collect", "render", "verify", "all"))
    args = parser.parse_args()
    if args.command in {"collect", "all"}:
        collect(args.dataset)
    if args.command in {"render", "all"}:
        render(args.dataset)
    if args.command in {"verify", "all"}:
        verify(args.dataset)


if __name__ == "__main__":
    main()
