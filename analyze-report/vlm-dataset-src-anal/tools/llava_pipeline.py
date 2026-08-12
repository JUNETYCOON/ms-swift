#!/usr/bin/env python3
"""Analyze 200 visual samples from llava_v1_5_mix665k.json."""

from __future__ import annotations

import argparse, base64, hashlib, html, io, json, random, shlex, shutil, subprocess, tarfile, tempfile, zlib
from collections import Counter
from pathlib import Path, PurePosixPath
from PIL import Image
from ai2d_pipeline import STYLE

HERE=Path(__file__).resolve().parent; REPORT_ROOT=HERE.parent
DATASET_DIR=REPORT_ROOT/"sub-dataset"/"LLaVA-style"; REPORT_PATH=REPORT_ROOT/"LLaVA-style.html"
REMOTE_HOST="pudu_lrs_workspace"; REMOTE_PYTHON="/tmp/vlm_dataset_src_anal_venv/bin/python"
REMOTE_ROOT="/mnt/oss-data/luojunkun/stage1/dataset/llava-instruct"; SEED=2026080409; TARGET=200; CANDIDATES=400

REMOTE_SCAN=r'''
import base64, heapq, ijson, json, random, sys, zlib
from collections import Counter
from pathlib import Path
cfg=json.loads(zlib.decompress(base64.urlsafe_b64decode(sys.argv[1])));root=Path(cfg["root"]);path=root/"llava_v1_5_mix665k.json"
rng=random.Random(cfg["seed"]);heap=[];prefix=Counter();turns=Counter();total=0;text_only=0
for row,record in enumerate(ijson.items(path.open("rb"),"item")):
    total+=1;image=record.get("image")
    if image: prefix[image.split("/",1)[0]]+=1
    else: prefix["<text-only>"]+=1;text_only+=1
    conversations=record.get("conversations") or [];turns[len(conversations)]+=1
    priority=rng.random();entry=(-priority,row,record)
    if len(heap)<cfg["candidate_count"]:heapq.heappush(heap,entry)
    elif priority < -heap[0][0]:heapq.heapreplace(heap,entry)
candidates=[]
for draw_order,(negative,row,record) in enumerate(sorted(heap,key=lambda value:-value[0])):
    candidates.append({"draw_order":draw_order,"priority":-negative,"source_row":row,"record":record})
other={p.name:p.stat().st_size for p in root.glob("*.json")}
print(json.dumps({"population_total":total,"text_only":text_only,"source_prefix_counts":dict(prefix),"conversation_turn_counts":dict(turns),"json_files":other,"candidates":candidates},ensure_ascii=False,default=str))
'''

REMOTE_BATCH=r'''
import base64, hashlib, io, json, sys, tarfile, time, zipfile, zlib
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
cfg=json.loads(zlib.decompress(base64.urlsafe_b64decode(sys.argv[1])));stage=Path(cfg["stage_root"]);root=stage/"llava-instruct"
def open_parquet(path):
 last=None
 for attempt in range(12):
  try:return pq.ParquetFile(path)
  except FileNotFoundError as exc:last=exc;time.sleep(min(.5*(attempt+1),2))
 raise last
def add(tar,name,payload):
 info=tarfile.TarInfo(name);info.size=len(payload);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(payload))
def image_key(path):return Path(path).stem
gqa_ids={image_key(x["record"].get("image")) for x in cfg["candidates"] if (x["record"].get("image") or "").startswith("gqa/")}
text_ids={image_key(x["record"].get("image")) for x in cfg["candidates"] if (x["record"].get("image") or "").startswith("textvqa/")}
gqa={};textvqa={}
if gqa_ids:
 for path in sorted((stage/"GQA").rglob("*.parquet")):
  if "_images" not in path.parent.name:continue
  pf=open_parquet(path)
  for group in range(pf.num_row_groups):
   table=pf.read_row_group(group,columns=["id","image"])
   for idx,value in enumerate(table.column("id").to_pylist()):
    key=str(value)
    if key in gqa_ids and key not in gqa:gqa[key]=table.column("image")[idx].as_py().get("bytes")
  pf.close()
  if gqa_ids.issubset(gqa):break
if text_ids:
 for path in sorted((stage/"textvqa").rglob("*.parquet")):
  pf=open_parquet(path)
  for group in range(pf.num_row_groups):
   table=pf.read_row_group(group,columns=["image_id","image"])
   for idx,value in enumerate(table.column("image_id").to_pylist()):
    key=str(value)
    if key in text_ids and key not in textvqa:textvqa[key]=table.column("image")[idx].as_py().get("bytes")
  pf.close()
  if text_ids.issubset(textvqa):break
archives={}
results=[]
try:
 for candidate in cfg["candidates"]:
  record=candidate["record"];image_path=record.get("image")
  if not image_path:
   results.append({"draw_order":candidate["draw_order"],"source_row":candidate["source_row"],"reason":"text_only_record_without_image"});continue
  payload=None;resolver=None
  try:
   if image_path.startswith("coco/train2017/"):
    archive=archives.setdefault("coco",zipfile.ZipFile(root/"datasets/coco2017/train2017.zip"));payload=archive.read(image_path.removeprefix("coco/"));resolver="datasets/coco2017/train2017.zip"
   elif image_path.startswith("ocr_vqa/images/"):
    source=root/"datasets/ocr-vqa/images"/Path(image_path).name;payload=source.read_bytes();resolver="datasets/ocr-vqa/images"
   elif image_path.startswith("vg/"):
    relative=image_path.removeprefix("vg/");archive_name="images2.zip" if relative.startswith("VG_100K_2/") else "images.zip";archive=archives.setdefault(archive_name,zipfile.ZipFile(stage/"VisualGenome"/archive_name));payload=archive.read(relative);resolver=f"../VisualGenome/{archive_name}"
   elif image_path.startswith("gqa/"):
    payload=gqa.get(image_key(image_path));resolver="../GQA/*_images parquet by id"
   elif image_path.startswith("textvqa/"):
    payload=textvqa.get(image_key(image_path));resolver="../textvqa parquet by image_id"
   if not payload:raise FileNotFoundError(f"unresolved image path: {image_path}")
   with Image.open(io.BytesIO(payload)) as image:image.load();width,height=image.size;fmt=(image.format or "bin").lower();mode=image.mode
   sample_id=f"llava-draw-{candidate['draw_order']+1:03d}";media=f"source-sample/resolved-media/{image_path}";record_path=f"source-sample/records/llava_v1_5_mix665k/row-{candidate['source_row']}.json"
   conversations=record.get("conversations") or [];user=next((x.get("value") for x in conversations if x.get("from")=="human"),None);assistant=next((x.get("value") for x in conversations if x.get("from")=="gpt"),None)
   item={"draw_order":candidate["draw_order"],"source_row":candidate["source_row"],"sample_id":sample_id,"logical_image_path":image_path,"media_archive_path":media,"record_archive_path":record_path,"resolver":resolver,"conversation_turn_count":len(conversations),"first_user":user,"first_assistant":assistant,"sha256":hashlib.sha256(payload).hexdigest(),"byte_length":len(payload),"width":width,"height":height,"image_format":fmt,"image_mode":mode}
   results.append(item);candidate["_payload"]=payload;candidate["_public"]=item
  except Exception as exc:results.append({"draw_order":candidate["draw_order"],"source_row":candidate["source_row"],"logical_image_path":image_path,"reason":"media_resolution_or_decode_error","detail":str(exc)[:300]})
 with tarfile.open(fileobj=sys.stdout.buffer,mode="w|") as tar:
  add(tar,"batch-results.jsonl",b"".join(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()+b"\n" for x in results))
  for candidate in cfg["candidates"]:
   if "_public" not in candidate:continue
   add(tar,candidate["_public"]["media_archive_path"],candidate["_payload"])
   add(tar,candidate["_public"]["record_archive_path"],(json.dumps(candidate["record"],ensure_ascii=False,indent=2,default=str)+"\n").encode())
finally:
 for archive in archives.values():archive.close()
'''

def enc(v):return base64.urlsafe_b64encode(zlib.compress(json.dumps(v,ensure_ascii=False).encode(),9)).decode()
def run(script,cfg,stdout):
 cmd=f"ls -la /mnt/oss-data >/dev/null 2>&1 && PYTHONDONTWRITEBYTECODE=1 {REMOTE_PYTHON} - {shlex.quote(enc(cfg))}"
 return subprocess.run(["ssh","-o","BatchMode=yes",REMOTE_HOST,cmd],input=script.encode(),stdout=stdout,stderr=subprocess.PIPE,check=False)
def extract(path,dest):
 with tarfile.open(path) as tar:
  for m in tar.getmembers():
   p=PurePosixPath(m.name)
   if p.is_absolute() or ".." in p.parts:raise RuntimeError("unsafe tar member")
  tar.extractall(dest,filter="data")

def collect():
 if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):raise RuntimeError(f"refusing to overwrite {DATASET_DIR}")
 DATASET_DIR.mkdir(parents=True,exist_ok=True)
 cache=Path(f"/tmp/vlm_llava_scan_{SEED}.json")
 if cache.exists():index=json.loads(cache.read_text())
 else:
  scan=run(REMOTE_SCAN,{"root":REMOTE_ROOT,"seed":SEED,"candidate_count":CANDIDATES},subprocess.PIPE)
  if scan.returncode:DATASET_DIR.rmdir();raise RuntimeError(scan.stderr.decode(errors="replace"))
  index=json.loads(scan.stdout);cache.write_text(json.dumps(index,ensure_ascii=False))
 valid=[];decisions=[]
 with tempfile.TemporaryDirectory(prefix="llava-") as temp_value:
  temp=Path(temp_value);batch_dirs=[]
  families={}
  for candidate in index["candidates"]:
   image_path=candidate["record"].get("image") or "<text-only>"
   families.setdefault(image_path.split("/",1)[0],[]).append(candidate)
  batch_number=0
  for family,candidates in sorted(families.items()):
   for start in range(0,len(candidates),50):
    batch=temp/f"batch-{batch_number:02d}-{family.replace('<','').replace('>','')}";batch.mkdir();archive=temp/f"batch-{batch_number:02d}.tar";batch_number+=1
    with archive.open("wb") as out:process=run(REMOTE_BATCH,{"stage_root":"/mnt/oss-data/luojunkun/stage1/dataset","candidates":candidates[start:start+50]},out)
    if process.returncode:DATASET_DIR.rmdir();raise RuntimeError(process.stderr.decode(errors="replace"))
    extract(archive,batch);batch_dirs.append(batch)
    for line in (batch/"batch-results.jsonl").read_text().splitlines():
     item=json.loads(line);(decisions if "reason" in item else valid).append((item,batch))
  selected=[];seen=set()
  for item,batch in sorted(valid,key=lambda x:x[0]["draw_order"]):
   if item["sha256"] in seen:decisions.append(({"draw_order":item["draw_order"],"source_row":item["source_row"],"reason":"duplicate_media_sha256","sha256":item["sha256"]},batch))
   elif len(selected)<TARGET:seen.add(item["sha256"]);selected.append((item,batch))
   else:decisions.append(({"draw_order":item["draw_order"],"source_row":item["source_row"],"reason":"reserve_candidate_not_needed"},batch))
  if len(selected)!=TARGET:DATASET_DIR.rmdir();raise RuntimeError(f"only {len(selected)} resolved unique images")
  for item,batch in selected:
   for key in ("media_archive_path","record_archive_path"):
    target=DATASET_DIR/item[key];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(batch/item[key],target)
 manifest=[x for x,_ in selected];decision_items=sorted([x for x,_ in decisions],key=lambda x:x["draw_order"])
 selected_prefix=Counter(x["logical_image_path"].split("/",1)[0] for x in manifest);turn_counts=Counter(str(x["conversation_turn_count"]) for x in manifest)
 summary={"dataset":"LLaVA-style","source_root":REMOTE_ROOT,"source_file":"llava_v1_5_mix665k.json","seed":SEED,"population_total":index["population_total"],"text_only_records":index["text_only"],"source_prefix_counts":index["source_prefix_counts"],"selected_prefix_counts":dict(selected_prefix),"selected_turn_counts":dict(sorted(turn_counts.items(),key=lambda x:int(x[0]))),"candidate_count":CANDIDATES,"selected_count":TARGET,"invalid_or_duplicate_count":sum(x["reason"]!="reserve_candidate_not_needed" for x in decision_items),"reserve_candidate_count":sum(x["reason"]=="reserve_candidate_not_needed" for x in decision_items),"image_width":{"min":min(x["width"] for x in manifest),"max":max(x["width"] for x in manifest)},"image_height":{"min":min(x["height"] for x in manifest),"max":max(x["height"] for x in manifest)},"read_only_source":True}
 (DATASET_DIR/"sampling-manifest.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in manifest));(DATASET_DIR/"sampling-candidate-decisions.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in decision_items));(DATASET_DIR/"sampling-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
 (DATASET_DIR/"source-layout.txt").write_text("".join(f"file\t{name}\t{size}\n" for name,size in sorted(index["json_files"].items()))+"directory\tdatasets/coco2017\n"+"directory\tdatasets/ocr-vqa\n")
 schema={"logical_record":{"id":"string","image":"optional string","conversations":"list<{from,value}>"},"analyzed_file":"llava_v1_5_mix665k.json","media_resolution":{"coco":"datasets/coco2017/train2017.zip","ocr_vqa":"datasets/ocr-vqa/images","gqa":"sibling GQA image parquet by id","textvqa":"sibling textvqa parquet by image_id","vg":"sibling VisualGenome image ZIP by member"}}
 (DATASET_DIR/"source-schema.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2)+"\n")
 cache.unlink(missing_ok=True)

def manifest():return [json.loads(x) for x in (DATASET_DIR/"sampling-manifest.jsonl").read_text().splitlines() if x]
def render():
 s=json.loads((DATASET_DIR/"sampling-summary.json").read_text());items=manifest();prefix_rows="".join(f"<tr><td>{html.escape(k)}</td><td>{v:,}</td><td>{s['selected_prefix_counts'].get(k,0)}</td></tr>" for k,v in sorted(s["source_prefix_counts"].items()))
 cards=[]
 for x in items:
  media=f"sub-dataset/LLaVA-style/{x['media_archive_path']}";question=(x["first_user"] or "").replace("<image>","").strip();answer=x["first_assistant"] or "<missing>";detail=html.escape(json.dumps({k:x[k] for k in ("source_row","logical_image_path","resolver","conversation_turn_count","width","height","sha256")},ensure_ascii=False,indent=2))
  cards.append(f'''<article class="sample"><a href="{media}"><img loading="lazy" src="{media}" alt="{x['sample_id']} LLaVA source sample"></a><div class="sample-body"><div class="sample-head"><strong>{x['sample_id']}</strong><span>{html.escape(x['logical_image_path'].split('/',1)[0])}</span></div><p><strong>用户：</strong>{html.escape(question[:240])}<br><strong>助手：</strong>{html.escape(answer[:300])}</p><dl><dt>轮次</dt><dd>{x['conversation_turn_count']} 条 message</dd><dt>源行</dt><dd>{x['source_row']}</dd></dl><details><summary>字段与追溯信息</summary><pre>{detail}</pre></details></div></article>''')
 doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLaVA-style 源数据集 200 张随机样本分析</title><style>{STYLE}</style></head><body><main><header><div class="eyebrow">VLM SOURCE DATASET / SAMPLE ANALYSIS 08</div><h1>LLaVA-style：200 张全量随机样本分析</h1><p class="verdict"><strong>总判断：</strong>本地 665k 主文件是 LLaVA 对话 schema 的混合指令集：输入由可选图片、当前用户文本和历史消息组成，输出是下一条 assistant 文本。它混合 COCO、GQA、OCR-VQA、TextVQA、Visual Genome 和纯文本记录；逻辑记录可统一，媒体存储却不统一，训练前必须先完成路径解析和纯文本分流。</p></header><section class="facts"><div class="fact"><b>{s['population_total']:,}</b><span>全量源记录</span></div><div class="fact"><b>200</b><span>解析成功且唯一图片</span></div><div class="fact"><b>{s['text_only_records']:,}</b><span>纯文本源记录</span></div><div class="fact"><b>{SEED}</b><span>固定随机种子</span></div></section>
<section><h2>一、总体：对话是逻辑层，图片是外部资产</h2><p>分析文件为 <code>{REMOTE_ROOT}/llava_v1_5_mix665k.json</code>。每条记录含 id、可选 image 和 conversations；conversation 的 <code>from=human</code> 是输入消息，<code>from=gpt</code> 是监督输出。图片路径前缀决定实际媒体源，不能把字符串存在误当成媒体已验证。</p><div class="band"><div class="flow"><span>外部图片路径</span><b>+</b><span>历史 human/gpt 消息</span><b>→</b><span>下一轮 gpt 文本</span></div></div><p class="note"><strong>证据边界：</strong>源中 {s['text_only_records']:,} 条记录没有 image；本报告把它们保留在全量总体和拒绝记录中，但 200 张视觉归档只接纳实际可解码媒体。未发现结构化 bbox 字段，不能因自然语言提到位置就计为 grounding。</p></section>
<section><h2>二、完整扫描与路径解析</h2><p>对 1 GB JSON 完整流式扫描，为每行生成固定种子随机优先级，取全量中优先级最低的 400 个候选；候选按优先级排序后分 4 个 SSH 批次解析媒体，遇到纯文本、路径无法解析、坏图或重复 SHA256 时记录并顺延。</p><div class="table-wrap"><table><thead><tr><th>image 前缀</th><th>全量记录</th><th>入选图片</th></tr></thead><tbody>{prefix_rows}</tbody></table></div></section>
<section><h2>三、200 张样本分布</h2><p>图片宽度 {s['image_width']['min']}–{s['image_width']['max']} px，高度 {s['image_height']['min']}–{s['image_height']['max']} px。归档保留每条完整 conversation JSON；卡片只展示首个 user/assistant 对，避免用截断文本替代原记录。</p></section><section><h2>四、逐样本：完整 200 张归档</h2><div class="gallery">{''.join(cards)}</div></section><footer>只读源审计：1 GB JSON、ZIP、目录图片和 sibling Parquet 均只读；样本经 stdout 回传到本地 docs。</footer></main></body></html>''';REPORT_PATH.write_text(doc)
def verify():
 items=manifest()
 if len(items)!=TARGET or len({x["sha256"] for x in items})!=TARGET:raise RuntimeError("manifest/hash mismatch")
 for x in items:
  p=(DATASET_DIR/x["media_archive_path"]).read_bytes()
  if hashlib.sha256(p).hexdigest()!=x["sha256"]:raise RuntimeError("digest mismatch")
  with Image.open(io.BytesIO(p)) as image:image.verify()
 if REPORT_PATH.read_text().count('<article class="sample"')!=TARGET:raise RuntimeError("card count mismatch")
 print(json.dumps({"dataset":"LLaVA-style","manifest_rows":200,"unique_media_sha256":200,"readable_media":200,"report_sample_cards":200},ensure_ascii=False))
def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=("collect","render","verify","all"));a=p.parse_args()
 if a.command in {"collect","all"}:collect()
 if a.command in {"render","all"}:render()
 if a.command in {"verify","all"}:verify()
if __name__=="__main__":main()
