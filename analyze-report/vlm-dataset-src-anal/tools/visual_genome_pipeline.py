#!/usr/bin/env python3
"""Collect, render, and verify Visual Genome image, QA, and region samples."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import random
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image
from ai2d_pipeline import STYLE


HERE = Path(__file__).resolve().parent
REPORT_ROOT = HERE.parent
DATASET_DIR = REPORT_ROOT / "sub-dataset" / "VisualGenome"
REPORT_PATH = REPORT_ROOT / "VisualGenome.html"
REMOTE_HOST = "pudu_lrs_workspace"
REMOTE_PYTHON = "/tmp/vlm_dataset_src_anal_venv/bin/python"
REMOTE_ROOT = "/mnt/oss-data/luojunkun/stage1/dataset/VisualGenome"
SEED = 2026080408
TARGET_COUNT = 200
CANDIDATE_COUNT = 400


REMOTE_INDEXER = r'''
import base64, json, sys, zipfile
from pathlib import Path
cfg=json.loads(base64.urlsafe_b64decode(sys.argv[1]))
root=Path(cfg["root"])
members=[]
for archive_name in ("images.zip","images2.zip"):
    with zipfile.ZipFile(root/archive_name) as archive:
        for info in archive.infolist():
            if not info.is_dir() and info.filename.lower().endswith((".jpg",".jpeg",".png")):
                members.append({"archive":archive_name,"member":info.filename,"file_size":info.file_size,"image_id":int(Path(info.filename).stem)})
files=[]
for path in sorted(root.iterdir()):
    if path.is_file(): files.append({"name":path.name,"size":path.stat().st_size})
print(json.dumps({"members":members,"source_files":files},ensure_ascii=False))
'''


REMOTE_IMAGE_BATCH = r'''
import base64, hashlib, io, json, sys, tarfile, zipfile
from pathlib import Path
from PIL import Image
cfg=json.loads(base64.urlsafe_b64decode(sys.argv[1]))
root=Path(cfg["root"])
def add(tar,name,payload):
    info=tarfile.TarInfo(name);info.size=len(payload);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(payload))
opened={}
records=[]
try:
    for item in cfg["candidates"]:
        archive=opened.setdefault(item["archive"],zipfile.ZipFile(root/item["archive"]))
        try:
            payload=archive.read(item["member"])
            with Image.open(io.BytesIO(payload)) as image:
                image.load(); width,height=image.size; image_format=(image.format or "bin").lower(); image_mode=image.mode
            sample_id=f"visualgenome-draw-{item['draw_order']+1:03d}"
            extension="jpg" if image_format in {"jpg","jpeg"} else image_format
            media_path=f"source-sample/archive-contents/{item['archive']}/{item['member']}"
            records.append({**item,"sample_id":sample_id,"media_archive_path":media_path,"sha256":hashlib.sha256(payload).hexdigest(),"byte_length":len(payload),"width":width,"height":height,"image_format":image_format,"image_mode":image_mode,"_payload":payload})
        except Exception as exc:
            records.append({**item,"reason":"image_read_or_decode_error","detail":str(exc)[:240]})
    with tarfile.open(fileobj=sys.stdout.buffer,mode="w|") as tar:
        public=[]
        for item in records:
            clean={k:v for k,v in item.items() if not k.startswith("_")};public.append(clean)
            if "_payload" in item:add(tar,item["media_archive_path"],item["_payload"])
        add(tar,"batch-results.jsonl",b"".join(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()+b"\n" for x in public))
finally:
    for archive in opened.values(): archive.close()
'''


REMOTE_METADATA = r'''
import base64, io, json, sys, tarfile, zipfile
from pathlib import Path
import ijson
cfg=json.loads(base64.urlsafe_b64decode(sys.argv[1]))
root=Path(cfg["root"]);wanted=set(cfg["image_ids"])
def add(tar,name,payload):
    info=tarfile.TarInfo(name);info.size=len(payload);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(payload))
def key_for(kind,item):
    if kind=="image_data": return item.get("image_id")
    if kind=="question_answers": return item.get("id")
    regions=item.get("regions") or []
    return regions[0].get("image_id") if regions else item.get("id")
found={kind:{} for kind in ("image_data","question_answers","region_descriptions")}
specs=[("image_data","image_data.json.zip"),("question_answers","question_answers.json.zip"),("region_descriptions","region_descriptions.json.zip")]
for kind,archive_name in specs:
    with zipfile.ZipFile(root/archive_name) as archive:
        member=next(info for info in archive.infolist() if not info.is_dir())
        with archive.open(member) as stream:
            for item in ijson.items(stream,"item"):
                key=key_for(kind,item)
                if key in wanted: found[kind][key]=item
                if len(found[kind])==len(wanted): break
index={}
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|") as tar:
    for image_id in sorted(wanted):
        image_data=found["image_data"].get(image_id)
        qa=found["question_answers"].get(image_id)
        regions=found["region_descriptions"].get(image_id)
        if image_data is not None:add(tar,f"source-sample/metadata/image_data/{image_id}.json",(json.dumps(image_data,ensure_ascii=False,indent=2,default=str)+"\n").encode())
        if qa is not None:add(tar,f"source-sample/metadata/question_answers/{image_id}.json",(json.dumps(qa,ensure_ascii=False,indent=2,default=str)+"\n").encode())
        if regions is not None:add(tar,f"source-sample/metadata/region_descriptions/{image_id}.json",(json.dumps(regions,ensure_ascii=False,indent=2,default=str)+"\n").encode())
        qas=(qa or {}).get("qas") or []; region_items=(regions or {}).get("regions") or []
        index[image_id]={"has_image_data":image_data is not None,"qa_count":len(qas),"region_count":len(region_items),"first_question":qas[0].get("question") if qas else None,"first_answer":qas[0].get("answer") if qas else None,"first_region_phrase":region_items[0].get("phrase") if region_items else None,"source_width":image_data.get("width") if image_data else None,"source_height":image_data.get("height") if image_data else None}
    add(tar,"metadata-index.json",(json.dumps(index,ensure_ascii=False,indent=2,sort_keys=True,default=str)+"\n").encode())
'''


def encoded(value: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(value, ensure_ascii=False).encode()).decode()


def remote(command_script: str, cfg: dict, stdout) -> subprocess.CompletedProcess:
    command = f"ls -la /mnt/oss-data >/dev/null 2>&1 && PYTHONDONTWRITEBYTECODE=1 {REMOTE_PYTHON} - {shlex.quote(encoded(cfg))}"
    return subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE_HOST, command], input=command_script.encode(), stdout=stdout, stderr=subprocess.PIPE, check=False)


def safe_extract(path: Path, destination: Path) -> None:
    with tarfile.open(path, "r") as archive:
        for member in archive.getmembers():
            candidate = PurePosixPath(member.name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        archive.extractall(destination, filter="data")


def collect() -> None:
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty archive: {DATASET_DIR}")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    index_process = remote(REMOTE_INDEXER, {"root": REMOTE_ROOT}, subprocess.PIPE)
    if index_process.returncode:
        DATASET_DIR.rmdir(); raise RuntimeError(index_process.stderr.decode(errors="replace"))
    index = json.loads(index_process.stdout)
    rng = random.Random(SEED)
    positions = rng.sample(range(len(index["members"])), CANDIDATE_COUNT)
    candidates = [{**index["members"][position], "draw_order": draw_order, "population_index": position} for draw_order, position in enumerate(positions)]

    with tempfile.TemporaryDirectory(prefix="visual-genome-") as temporary_value:
        temporary = Path(temporary_value)
        valid = []
        decisions = []
        for start in range(0, CANDIDATE_COUNT, 100):
            batch_dir = temporary / f"batch-{start:03d}"; batch_dir.mkdir()
            archive_path = temporary / f"batch-{start:03d}.tar"
            with archive_path.open("wb") as stream:
                process = remote(REMOTE_IMAGE_BATCH, {"root": REMOTE_ROOT, "candidates": candidates[start:start + 100]}, stream)
            if process.returncode:
                DATASET_DIR.rmdir(); raise RuntimeError(process.stderr.decode(errors="replace"))
            safe_extract(archive_path, batch_dir); archive_path.unlink()
            for line in (batch_dir / "batch-results.jsonl").read_text().splitlines():
                item = json.loads(line)
                (decisions if "reason" in item else valid).append((item, batch_dir))
        selected=[];seen=set()
        for item,batch_dir in sorted(valid,key=lambda value:value[0]["draw_order"]):
            if item["sha256"] in seen:
                decisions.append(({"draw_order":item["draw_order"],"archive":item["archive"],"member":item["member"],"sha256":item["sha256"],"reason":"duplicate_media_sha256"},batch_dir))
            elif len(selected)<TARGET_COUNT:
                seen.add(item["sha256"]);selected.append((item,batch_dir))
            else:
                decisions.append(({"draw_order":item["draw_order"],"archive":item["archive"],"member":item["member"],"reason":"reserve_candidate_not_needed"},batch_dir))
        if len(selected)!=TARGET_COUNT:
            DATASET_DIR.rmdir(); raise RuntimeError(f"only {len(selected)} valid unique images")
        for item,batch_dir in selected:
            target=DATASET_DIR/item["media_archive_path"];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(batch_dir/item["media_archive_path"],target)

        metadata_archive=temporary/"metadata.tar"
        with metadata_archive.open("wb") as stream:
            process=remote(REMOTE_METADATA,{"root":REMOTE_ROOT,"image_ids":[item["image_id"] for item,_ in selected]},stream)
        if process.returncode:
            raise RuntimeError(process.stderr.decode(errors="replace"))
        safe_extract(metadata_archive,DATASET_DIR)

    metadata_index=json.loads((DATASET_DIR/"metadata-index.json").read_text())
    manifest=[]
    for item,_ in selected:
        meta=metadata_index[str(item["image_id"])]
        manifest.append({**item,**meta,
            "image_data_archive_path":f"source-sample/metadata/image_data/{item['image_id']}.json" if meta["has_image_data"] else None,
            "qa_archive_path":f"source-sample/metadata/question_answers/{item['image_id']}.json" if meta["qa_count"] else None,
            "regions_archive_path":f"source-sample/metadata/region_descriptions/{item['image_id']}.json" if meta["region_count"] else None,
        })
    decisions_public=[item for item,_ in decisions]
    archive_counts=Counter(item["archive"] for item in manifest)
    summary={"dataset":"VisualGenome","source_root":REMOTE_ROOT,"seed":SEED,"population_total":len(index["members"]),"population_by_archive":dict(Counter(item["archive"] for item in index["members"])),"candidate_count":CANDIDATE_COUNT,"selected_count":TARGET_COUNT,"selected_archive_counts":dict(archive_counts),"invalid_or_duplicate_count":sum(item["reason"]!="reserve_candidate_not_needed" for item in decisions_public),"reserve_candidate_count":sum(item["reason"]=="reserve_candidate_not_needed" for item in decisions_public),"qa_count":{"min":min(x["qa_count"] for x in manifest),"max":max(x["qa_count"] for x in manifest),"mean":round(sum(x["qa_count"] for x in manifest)/TARGET_COUNT,2)},"region_count":{"min":min(x["region_count"] for x in manifest),"max":max(x["region_count"] for x in manifest),"mean":round(sum(x["region_count"] for x in manifest)/TARGET_COUNT,2)},"image_width":{"min":min(x["width"] for x in manifest),"max":max(x["width"] for x in manifest)},"image_height":{"min":min(x["height"] for x in manifest),"max":max(x["height"] for x in manifest)},"invalid_source_artifact":"images2.zip.1 is not a valid ZIP and is excluded from the image population","read_only_source":True}
    (DATASET_DIR/"sampling-manifest.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in manifest))
    (DATASET_DIR/"sampling-candidate-decisions.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in sorted(decisions_public,key=lambda x:x["draw_order"])))
    (DATASET_DIR/"sampling-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    layout="".join(f"file\t{x['name']}\t{x['size']}\n" for x in index["source_files"])
    (DATASET_DIR/"source-layout.txt").write_text(layout)
    schema={"image_join":"Path(image member).stem == image_id","image_archives":["images.zip","images2.zip"],"analyzed_metadata":{"image_data.json.zip":"[{image_id,width,height,url,coco_id,flickr_id}]","question_answers.json.zip":"[{id,qas:[{question,answer,image_id,qa_id,q_objects,a_objects}]}]","region_descriptions.json.zip":"[{regions:[{image_id,region_id,x,y,width,height,phrase}]}]"},"listed_not_sample_parsed":["attributes.json.zip","objects_v1_2.json.zip","relationships_v1_2.json.zip","region_graphs.json.zip","scene_graphs.json.zip","synsets.json.zip","qa_to_region_mapping.json.zip"]}
    (DATASET_DIR/"source-schema.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2)+"\n")


def manifest() -> list[dict]:
    return [json.loads(line) for line in (DATASET_DIR/"sampling-manifest.jsonl").read_text().splitlines() if line]


def render() -> None:
    summary=json.loads((DATASET_DIR/"sampling-summary.json").read_text());items=manifest()
    archive_rows="".join(f"<tr><td>{html.escape(name)}</td><td>{count:,}</td><td>{summary['selected_archive_counts'].get(name,0)}</td></tr>" for name,count in summary["population_by_archive"].items())
    cards=[]
    for item in items:
        media=f"sub-dataset/VisualGenome/{item['media_archive_path']}"
        detail=html.escape(json.dumps({k:item[k] for k in ("image_id","archive","member","draw_order","qa_count","region_count","first_question","first_answer","first_region_phrase","width","height","sha256")},ensure_ascii=False,indent=2))
        cards.append(f'''<article class="sample"><a href="{media}"><img loading="lazy" src="{media}" alt="{item['sample_id']} Visual Genome source sample"></a><div class="sample-body"><div class="sample-head"><strong>{item['sample_id']}</strong><span>ID {item['image_id']}</span></div><p><strong>QA：</strong>{html.escape(item['first_question'] or '无 QA')} → {html.escape(item['first_answer'] or '无答案')}<br><strong>Region：</strong>{html.escape(item['first_region_phrase'] or '无区域描述')}</p><dl><dt>数量</dt><dd>QA {item['qa_count']} / Regions {item['region_count']}</dd><dt>源成员</dt><dd><code>{html.escape(item['archive'])}/{html.escape(item['member'])}</code></dd></dl><details><summary>字段与追溯信息</summary><pre>{detail}</pre></details></div></article>''')
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual Genome 源数据集 200 张随机样本分析</title><style>{STYLE}</style></head><body><main><header><div class="eyebrow">VLM SOURCE DATASET / SAMPLE ANALYSIS 07</div><h1>Visual Genome：200 张全量随机样本分析</h1><p class="verdict"><strong>总判断：</strong>Visual Genome 不是单一问答表，而是以 image_id 为中心的多层视觉语义资产。本报告在同一份源报告中区分两条输出：QA 提供问题—短答案，Regions 提供矩形区域与自然语言短语；两者都连接到同一张图片，但监督目标不同，不能混成一种标签。</p></header><section class="facts"><div class="fact"><b>{summary['population_total']:,}</b><span>两个有效 ZIP 中图片</span></div><div class="fact"><b>200</b><span>有效唯一随机图片</span></div><div class="fact"><b>{summary['qa_count']['mean']}</b><span>样本每图平均 QA</span></div><div class="fact"><b>{summary['region_count']['mean']}</b><span>样本每图平均 Regions</span></div></section>
<section><h2>一、总体：输入与两类输出</h2><p>源目录为 <code>{REMOTE_ROOT}</code>。图片分别存于 <code>images.zip</code> 和 <code>images2.zip</code>；文件名 stem 与元数据 <code>image_id</code> 连接。归档把 ZIP 内部路径保留在 <code>source-sample/archive-contents/{'{archive}/{member}'}</code> 下，并为每张图保存对应 QA、Regions 和 image_data JSON。</p><div class="band"><div class="flow"><span>JPEG 图片</span><b>+</b><span>image_id</span><b>→</b><span>QA：question → answer</span><b>或</b><span>Regions：bbox → phrase</span></div></div><div class="two"><div><h3>QA 输出</h3><p><code>qas[]</code> 含 question、answer、qa_id 及可选对象映射，适合视觉问答监督。</p></div><div><h3>Regions 输出</h3><p><code>regions[]</code> 含 x、y、width、height、phrase，适合区域描述与 grounding 监督。</p></div></div><p class="note"><strong>源异常与边界：</strong><code>images2.zip.1</code> 只有约 3.7 MB 且不是合法 ZIP，未纳入图片总体。对象、关系、属性等 ZIP 已列入源结构，但本轮逐样本只强制解析 QA 与 Regions，不宣称其他标注已完成样本级审计。</p></section>
<section><h2>二、全量随机抽样</h2><p>从两个有效图片 ZIP 的中央目录枚举全部 {summary['population_total']:,} 个图像成员，以固定种子 {SEED} 等概率无放回产生 400 个候选，再分 4 个只读 SSH 批次抽取，按全局 draw order 去重并固定前 200 张。元数据在图片确定后按 image_id 从完整 JSON 数组流式连接。</p><div class="table-wrap"><table><thead><tr><th>源归档</th><th>全量图片</th><th>入选</th></tr></thead><tbody>{archive_rows}</tbody></table></div></section>
<section><h2>三、200 张样本分布</h2><p>图片宽度 {summary['image_width']['min']}–{summary['image_width']['max']} px，高度 {summary['image_height']['min']}–{summary['image_height']['max']} px；每图 QA {summary['qa_count']['min']}–{summary['qa_count']['max']} 条，每图 Regions {summary['region_count']['min']}–{summary['region_count']['max']} 条。这里是固定样本描述，不外推为全量分布。</p></section>
<section><h2>四、逐样本：QA 与 Regions 并列展示</h2><p>卡片同时显示首条 QA 和首条区域短语；完整的该图 QA/Regions 记录位于相邻归档 JSON。</p><div class="gallery">{''.join(cards)}</div></section><footer>只读源审计：ZIP 中央目录、成员和 JSON 流均只读；所有样本从 stdout 回传并写入本地 docs。</footer></main></body></html>'''
    REPORT_PATH.write_text(doc)


def verify() -> None:
    items=manifest()
    if len(items)!=200 or len({x["sha256"] for x in items})!=200:raise RuntimeError("manifest or hash count mismatch")
    for item in items:
        payload=(DATASET_DIR/item["media_archive_path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=item["sha256"]:raise RuntimeError("digest mismatch")
        with Image.open(io.BytesIO(payload)) as image:image.verify()
    if REPORT_PATH.read_text().count('<article class="sample"')!=200:raise RuntimeError("report card count mismatch")
    print(json.dumps({"dataset":"VisualGenome","manifest_rows":200,"unique_media_sha256":200,"readable_media":200,"report_sample_cards":200},ensure_ascii=False))


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("collect","render","verify","all"));args=parser.parse_args()
    if args.command in {"collect","all"}:collect()
    if args.command in {"render","all"}:render()
    if args.command in {"verify","all"}:verify()
if __name__=="__main__":main()
