#!/usr/bin/env python3
"""Analyze 200 random VLM-R1 grounding records and their COCO media."""

from __future__ import annotations
import argparse,base64,hashlib,heapq,html,io,json,random,shlex,shutil,subprocess,tarfile,tempfile,zlib
from collections import Counter
from pathlib import Path,PurePosixPath
from PIL import Image
from ai2d_pipeline import STYLE

HERE=Path(__file__).resolve().parent;REPORT_ROOT=HERE.parent;DATASET_DIR=REPORT_ROOT/"sub-dataset"/"VLM-R1";REPORT_PATH=REPORT_ROOT/"VLM-R1.html"
REMOTE_HOST="pudu_lrs_workspace";REMOTE_PYTHON="/tmp/vlm_dataset_src_anal_venv/bin/python";REMOTE_ROOT="/mnt/oss-data/luojunkun/stage1/dataset/vlm-r1"
SEED=2026080410;TARGET=200;CANDIDATES=400

REMOTE_SCAN=r'''
import base64,heapq,ijson,json,random,sys,zlib
from collections import Counter
from pathlib import Path
cfg=json.loads(zlib.decompress(base64.urlsafe_b64decode(sys.argv[1])));root=Path(cfg["root"]);path=root/"sft_related/mllm_rec_json.json";rng=random.Random(cfg["seed"]);heap=[];total=0;image_counts=Counter();format_counts=Counter()
for row,record in enumerate(ijson.items(path.open("rb"),"item")):
 total+=1;images=record.get("images") or [];image_counts[len(images)]+=1;messages=record.get("messages") or [];assistant=next((m.get("content","") for m in messages if m.get("role")=="assistant"),"");format_counts["json_fence" if assistant.lstrip().startswith("```json") else "other"]+=1
 priority=rng.random();entry=(-priority,row,record)
 if len(heap)<cfg["candidate_count"]:heapq.heappush(heap,entry)
 elif priority < -heap[0][0]:heapq.heapreplace(heap,entry)
candidates=[{"draw_order":order,"priority":-negative,"source_row":row,"record":record} for order,(negative,row,record) in enumerate(sorted(heap,key=lambda x:-x[0]))]
print(json.dumps({"population_total":total,"image_count_distribution":dict(image_counts),"assistant_format_counts":dict(format_counts),"source_json_size":path.stat().st_size,"candidates":candidates},ensure_ascii=False))
'''

REMOTE_BATCH=r'''
import base64,hashlib,io,json,sys,tarfile,zipfile,zlib
from pathlib import Path
from PIL import Image
cfg=json.loads(zlib.decompress(base64.urlsafe_b64decode(sys.argv[1])));root=Path(cfg["root"]);results=[]
def add(tar,name,payload):info=tarfile.TarInfo(name);info.size=len(payload);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(payload))
with zipfile.ZipFile(root/"train2014.zip") as archive:
 for candidate in cfg["candidates"]:
  record=candidate["record"];old_path=(record.get("images") or [None])[0]
  try:
   filename=Path(old_path).name;member=f"train2014/{filename}";payload=archive.read(member)
   with Image.open(io.BytesIO(payload)) as image:image.load();width,height=image.size;fmt=(image.format or "bin").lower();mode=image.mode
   messages=record.get("messages") or [];user=next((m.get("content") for m in messages if m.get("role")=="user"),None);assistant=next((m.get("content") for m in messages if m.get("role")=="assistant"),None)
   parsed=None
   if assistant:
    body=assistant.strip();body=body[7:] if body.startswith("```json") else body;body=body[:-3] if body.endswith("```") else body
    try:parsed=json.loads(body)
    except Exception:pass
   boxes=[item.get("bbox_2d") for item in (parsed or []) if isinstance(item,dict) and isinstance(item.get("bbox_2d"),list)]
   out_of_bounds=sum(not(len(b)==4 and 0<=b[0]<=b[2]<=width and 0<=b[1]<=b[3]<=height) for b in boxes)
   sample_id=f"vlm-r1-draw-{candidate['draw_order']+1:03d}";media=f"source-sample/train2014.zip.contents/{member}";record_path=f"source-sample/sft_related/mllm_rec_json/row-{candidate['source_row']}.json"
   public={"draw_order":candidate["draw_order"],"source_row":candidate["source_row"],"sample_id":sample_id,"old_image_path":old_path,"source_archive":"train2014.zip","source_member":member,"media_archive_path":media,"record_archive_path":record_path,"user_prompt":user,"assistant_raw":assistant,"parsed_output":parsed,"bbox_count":len(boxes),"out_of_bounds_bbox_count":out_of_bounds,"sha256":hashlib.sha256(payload).hexdigest(),"byte_length":len(payload),"width":width,"height":height,"image_format":fmt,"image_mode":mode}
   candidate["_payload"]=payload;candidate["_public"]=public;results.append(public)
  except Exception as exc:results.append({"draw_order":candidate["draw_order"],"source_row":candidate["source_row"],"old_image_path":old_path,"reason":"media_or_output_error","detail":str(exc)[:300]})
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|") as tar:
 add(tar,"batch-results.jsonl",b"".join(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()+b"\n" for x in results))
 for candidate in cfg["candidates"]:
  if "_public" not in candidate:continue
  add(tar,candidate["_public"]["media_archive_path"],candidate["_payload"]);add(tar,candidate["_public"]["record_archive_path"],(json.dumps(candidate["record"],ensure_ascii=False,indent=2)+"\n").encode())
'''

def enc(v):return base64.urlsafe_b64encode(zlib.compress(json.dumps(v,ensure_ascii=False).encode(),9)).decode()
def run(script,cfg,stdout):
 cmd=f"ls -la /mnt/oss-data >/dev/null 2>&1 && PYTHONDONTWRITEBYTECODE=1 {REMOTE_PYTHON} - {shlex.quote(enc(cfg))}";return subprocess.run(["ssh","-o","BatchMode=yes",REMOTE_HOST,cmd],input=script.encode(),stdout=stdout,stderr=subprocess.PIPE,check=False)
def extract(path,dest):
 with tarfile.open(path) as tar:
  for m in tar.getmembers():
   p=PurePosixPath(m.name)
   if p.is_absolute() or ".." in p.parts:raise RuntimeError("unsafe tar")
  tar.extractall(dest,filter="data")
def collect():
 if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):raise RuntimeError(f"refusing to overwrite {DATASET_DIR}")
 DATASET_DIR.mkdir(parents=True,exist_ok=True);scan=run(REMOTE_SCAN,{"root":REMOTE_ROOT,"seed":SEED,"candidate_count":CANDIDATES},subprocess.PIPE)
 if scan.returncode:DATASET_DIR.rmdir();raise RuntimeError(scan.stderr.decode(errors="replace"))
 index=json.loads(scan.stdout);valid=[];decisions=[]
 with tempfile.TemporaryDirectory(prefix="vlm-r1-") as tv:
  temp=Path(tv)
  for start in range(0,CANDIDATES,50):
   batch=temp/f"batch-{start}";batch.mkdir();archive=temp/f"batch-{start}.tar"
   with archive.open("wb") as out:process=run(REMOTE_BATCH,{"root":REMOTE_ROOT,"candidates":index["candidates"][start:start+50]},out)
   if process.returncode:DATASET_DIR.rmdir();raise RuntimeError(process.stderr.decode(errors="replace"))
   extract(archive,batch)
   for line in (batch/"batch-results.jsonl").read_text().splitlines():
    item=json.loads(line);(decisions if "reason" in item else valid).append((item,batch))
  selected=[];seen=set()
  for item,batch in sorted(valid,key=lambda x:x[0]["draw_order"]):
   if item["sha256"] in seen:decisions.append(({"draw_order":item["draw_order"],"source_row":item["source_row"],"reason":"duplicate_media_sha256","sha256":item["sha256"]},batch))
   elif len(selected)<TARGET:seen.add(item["sha256"]);selected.append((item,batch))
   else:decisions.append(({"draw_order":item["draw_order"],"source_row":item["source_row"],"reason":"reserve_candidate_not_needed"},batch))
  if len(selected)!=TARGET:DATASET_DIR.rmdir();raise RuntimeError(f"only {len(selected)} valid unique")
  for item,batch in selected:
   for key in ("media_archive_path","record_archive_path"):
    target=DATASET_DIR/item[key];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(batch/item[key],target)
 manifest=[x for x,_ in selected];decision_items=sorted([x for x,_ in decisions],key=lambda x:x["draw_order"]);bbox_counts=Counter(str(x["bbox_count"]) for x in manifest)
 summary={"dataset":"VLM-R1","source_root":REMOTE_ROOT,"source_file":"sft_related/mllm_rec_json.json","seed":SEED,"population_total":index["population_total"],"image_count_distribution":index["image_count_distribution"],"assistant_format_counts":index["assistant_format_counts"],"candidate_count":CANDIDATES,"selected_count":TARGET,"invalid_or_duplicate_count":sum(x["reason"]!="reserve_candidate_not_needed" for x in decision_items),"reserve_candidate_count":sum(x["reason"]=="reserve_candidate_not_needed" for x in decision_items),"bbox_count_distribution":dict(sorted(bbox_counts.items(),key=lambda x:int(x[0]))),"out_of_bounds_bbox_count":sum(x["out_of_bounds_bbox_count"] for x in manifest),"image_width":{"min":min(x["width"] for x in manifest),"max":max(x["width"] for x in manifest)},"image_height":{"min":min(x["height"] for x in manifest),"max":max(x["height"] for x in manifest)},"read_only_source":True}
 (DATASET_DIR/"sampling-manifest.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in manifest));(DATASET_DIR/"sampling-candidate-decisions.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in decision_items));(DATASET_DIR/"sampling-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
 (DATASET_DIR/"source-layout.txt").write_text(f"file\tsft_related/mllm_rec_json.json\t{index['source_json_size']}\nfile\ttrain2014.zip\t{(13510573713)}\nfile\tsft_related/dataset_info.json\nfile\tsft_related/qwen2_5_vl_full_sft.yaml\n")
 (DATASET_DIR/"source-schema.json").write_text(json.dumps({"input":{"images":"list<string>, current file always length 1","messages.user":"<image> + referring expression"},"output":{"messages.assistant":"JSON fence containing list<{bbox_2d:[x1,y1,x2,y2],label:string}>"},"media_resolution":"Path(images[0]).name -> train2014.zip/train2014/{filename}"},ensure_ascii=False,indent=2)+"\n")
def manifest():return [json.loads(x) for x in (DATASET_DIR/"sampling-manifest.jsonl").read_text().splitlines() if x]
def render():
 s=json.loads((DATASET_DIR/"sampling-summary.json").read_text());items=manifest();bbox_rows="".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in s["bbox_count_distribution"].items());cards=[]
 for x in items:
  media=f"sub-dataset/VLM-R1/{x['media_archive_path']}";prompt=(x["user_prompt"] or "").replace("<image>","").strip();output=json.dumps(x["parsed_output"],ensure_ascii=False);detail=html.escape(json.dumps({k:x[k] for k in ("source_row","old_image_path","source_archive","source_member","bbox_count","out_of_bounds_bbox_count","width","height","sha256")},ensure_ascii=False,indent=2))
  cards.append(f'''<article class="sample"><a href="{media}"><img loading="lazy" src="{media}" alt="{x['sample_id']} VLM-R1 source sample"></a><div class="sample-body"><div class="sample-head"><strong>{x['sample_id']}</strong><span>{x['bbox_count']} bbox</span></div><p><strong>输入：</strong>{html.escape(prompt[:260])}<br><strong>输出：</strong>{html.escape(output[:300])}</p><dl><dt>尺寸</dt><dd>{x['width']} x {x['height']}</dd><dt>源行</dt><dd>{x['source_row']}</dd></dl><details><summary>字段与追溯信息</summary><pre>{detail}</pre></details></div></article>''')
 doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VLM-R1 源数据集 200 张随机样本分析</title><style>{STYLE}</style></head><body><main><header><div class="eyebrow">VLM SOURCE DATASET / SAMPLE ANALYSIS 10</div><h1>VLM-R1：200 张全量随机样本分析</h1><p class="verdict"><strong>总判断：</strong>当前训练主文件是纯指代表达定位数据：每条记录输入一张 COCO train2014 图片和一句“定位该描述区域”的指令，输出是 JSON bbox 列表。321,308 条记录全部单图、全部输出 JSON fence；同图可对应多个 referring expression，所以随机总体是逻辑记录而非 unique image。</p></header><section class="facts"><div class="fact"><b>{s['population_total']:,}</b><span>全量逻辑记录</span></div><div class="fact"><b>200</b><span>有效唯一随机图片</span></div><div class="fact"><b>{s['assistant_format_counts'].get('json_fence',0):,}</b><span>JSON fence 输出</span></div><div class="fact"><b>{s['out_of_bounds_bbox_count']}</b><span>样本越界 bbox</span></div></section><section><h2>一、总体：输入与输出</h2><p>源逻辑记录为 <code>{REMOTE_ROOT}/sft_related/mllm_rec_json.json</code>。<code>images[0]</code> 保存旧绝对路径，messages.user 给 referring expression，messages.assistant 给 <code>bbox_2d</code> 与 label。</p><div class="band"><div class="flow"><span>COCO 图片</span><b>+</b><span>referring expression</span><b>→</b><span>[x1,y1,x2,y2] + label</span></div></div><p class="note"><strong>路径证据：</strong>JSON 中的 <code>/data/shz/dataset/coco/train2014/...</code> 不是当前机器可用路径。本报告按文件名解析到同一源目录的 <code>train2014.zip</code>，同时保留旧路径和 ZIP member；不修改原记录。</p></section><section><h2>二、全量随机抽样</h2><p>完整扫描 321,308 行并用固定种子 {SEED} 生成随机优先级，取 400 个候选；候选按全局顺序分 8 个独立 SSH 批次从 13.5 GB ZIP 只读抽取，坏图或重复媒体顺延至 200 张。</p></section><section><h2>三、200 张样本分布</h2><p>图片宽度 {s['image_width']['min']}–{s['image_width']['max']} px，高度 {s['image_height']['min']}–{s['image_height']['max']} px。bbox 数量分布如下；越界检查按解码后原图尺寸执行。</p><div class="table-wrap"><table><thead><tr><th>每记录 bbox 数</th><th>样本</th></tr></thead><tbody>{bbox_rows}</tbody></table></div></section><section><h2>四、逐样本：完整 200 张归档</h2><div class="gallery">{''.join(cards)}</div></section><footer>只读源审计：JSON 与 train2014.zip 仅被读取，源中旧路径未改写。</footer></main></body></html>''';REPORT_PATH.write_text(doc)
def verify():
 items=manifest()
 if len(items)!=200 or len({x["sha256"] for x in items})!=200:raise RuntimeError("manifest/hash mismatch")
 for x in items:
  p=(DATASET_DIR/x["media_archive_path"]).read_bytes()
  if hashlib.sha256(p).hexdigest()!=x["sha256"]:raise RuntimeError("digest mismatch")
  with Image.open(io.BytesIO(p)) as image:image.verify()
 if REPORT_PATH.read_text().count('<article class="sample"')!=200:raise RuntimeError("card mismatch")
 print(json.dumps({"dataset":"VLM-R1","manifest_rows":200,"unique_media_sha256":200,"readable_media":200,"report_sample_cards":200},ensure_ascii=False))
def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=("collect","render","verify","all"));a=p.parse_args()
 if a.command in {"collect","all"}:collect()
 if a.command in {"render","all"}:render()
 if a.command in {"verify","all"}:verify()
if __name__=="__main__":main()
