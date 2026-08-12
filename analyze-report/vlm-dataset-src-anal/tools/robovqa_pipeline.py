#!/usr/bin/env python3
"""Sample 200 RoboVQA videos and archive deterministic representative frames."""

from __future__ import annotations
import argparse,base64,hashlib,html,io,json,random,re,shlex,shutil,subprocess,tarfile,tempfile,zlib
from collections import Counter
from pathlib import Path,PurePosixPath
from PIL import Image
from ai2d_pipeline import STYLE

HERE=Path(__file__).resolve().parent;REPORT_ROOT=HERE.parent;DATASET_DIR=REPORT_ROOT/"sub-dataset"/"RoboVQA";REPORT_PATH=REPORT_ROOT/"RoboVQA.html"
REMOTE_HOST="pudu_lrs_workspace";REMOTE_PYTHON="/tmp/vlm_dataset_src_anal_venv/bin/python";REMOTE_ROOT="/mnt/oss-data/luojunkun/stage1/dataset/robovqa"
SEED=2026080411;TARGET=200;CANDIDATES=400

REMOTE_TAR_INDEX=r'''
import base64,json,sys,tarfile,zlib
from pathlib import Path
cfg=json.loads(zlib.decompress(base64.urlsafe_b64decode(sys.argv[1])));root=Path(cfg["root"]);items=[]
for part in cfg["parts"]:
 archive_name=f"clips/clips_part_{part:03d}.tar.gz"
 with tarfile.open(root/archive_name,"r:gz") as archive:
  for member in archive:
   if member.isfile() and member.name.lower().endswith(".mp4"):
    normalized=member.name.removeprefix("./");items.append({"archive":archive_name,"member":member.name,"normalized_member":normalized,"video_id":Path(normalized).stem,"video_size":member.size})
print(json.dumps(items,ensure_ascii=False))
'''

REMOTE_VIDEO_BATCH=r'''
import base64,hashlib,io,json,os,re,subprocess,sys,tarfile,zlib
from collections import defaultdict
from pathlib import Path
import imageio_ffmpeg
from PIL import Image
cfg=json.loads(zlib.decompress(base64.urlsafe_b64decode(sys.argv[1])));root=Path(cfg["root"]);ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();by_archive=defaultdict(list)
for item in cfg["candidates"]:by_archive[item["archive"]].append(item)
def add(tar,name,payload):info=tarfile.TarInfo(name);info.size=len(payload);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(payload))
results=[];payloads={}
for archive_name,candidates in by_archive.items():
 wanted={item["member"]:item for item in candidates};wanted.update({"./"+item["normalized_member"]:item for item in candidates});found=set()
 with tarfile.open(root/archive_name,"r:gz") as archive:
  for member in archive:
   item=wanted.get(member.name)
   if item is None or item["draw_order"] in found:continue
   found.add(item["draw_order"])
   try:
    stream=archive.extractfile(member);video=stream.read() if stream else None
    if not video:raise RuntimeError("empty video member")
    fd=os.memfd_create("robovqa-preview");os.write(fd,video);os.lseek(fd,0,os.SEEK_SET)
    try:
     command=[ffmpeg,"-hide_banner","-loglevel","info","-i",f"/proc/self/fd/{fd}","-vf","thumbnail=100,showinfo","-frames:v","1","-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
     process=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(fd,),check=False)
    finally:os.close(fd)
    if process.returncode or not process.stdout:raise RuntimeError(process.stderr.decode(errors="replace")[-500:])
    size_match=re.search(rb" s:(\d+)x(\d+) ",process.stderr)
    if not size_match:raise RuntimeError("showinfo did not report frame dimensions")
    width,height=map(int,size_match.groups())
    if len(process.stdout)!=width*height*3:raise RuntimeError(f"unexpected raw frame size: {len(process.stdout)}")
    image=Image.frombytes("RGB",(width,height),process.stdout);sink=io.BytesIO();image.save(sink,format="PNG");frame=sink.getvalue();fmt="png";mode=image.mode
    match=re.search(rb"pts_time:([0-9.]+)",process.stderr);pts=float(match.group(1)) if match else None
    sample_id=f"robovqa-draw-{item['draw_order']+1:03d}";preview=f"derived-preview/{sample_id}.png"
    public={**item,"sample_id":sample_id,"preview_archive_path":preview,"preview_sha256":hashlib.sha256(frame).hexdigest(),"source_video_sha256":hashlib.sha256(video).hexdigest(),"source_video_byte_length":len(video),"preview_generation":"ffmpeg thumbnail=100, first selected frame, PNG output","preview_pts_time_seconds":pts,"width":width,"height":height,"image_format":fmt,"image_mode":mode}
    results.append(public);payloads[item["draw_order"]]=(preview,frame)
   except Exception as exc:results.append({**item,"reason":"video_read_or_frame_decode_error","detail":str(exc)[:500]})
   if len(found)==len(candidates):break
 for item in candidates:
  if item["draw_order"] not in found:results.append({**item,"reason":"tar_member_not_found"})
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|") as tar:
 add(tar,"batch-results.jsonl",b"".join(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()+b"\n" for x in results))
 for _,(path,payload) in payloads.items():add(tar,path,payload)
'''

REMOTE_QA=r'''
import base64,ijson,io,json,sys,tarfile,zlib
from pathlib import Path
cfg=json.loads(zlib.decompress(base64.urlsafe_b64decode(sys.argv[1])));root=Path(cfg["root"]);wanted=set(cfg["video_ids"]);found={}
with (root/"robovqa_understanding.json").open("rb") as stream:
 for row,record in enumerate(ijson.items(stream,"item")):
  video_id=Path(record.get("video") or "").stem
  if video_id in wanted and video_id not in found:found[video_id]=(row,record)
  if len(found)==len(wanted):break
def add(tar,name,payload):info=tarfile.TarInfo(name);info.size=len(payload);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(payload))
index={}
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|") as tar:
 for video_id in sorted(wanted):
  pair=found.get(video_id)
  if pair is None:index[video_id]={"found":False};continue
  row,record=pair;messages=record.get("conversations") or [];metadata=record.get("metadata") or {};tasks=metadata.get("task_metadata") or [];user=next((m.get("content") for m in messages if m.get("role")=="user"),None);assistant=next((m.get("content") for m in messages if m.get("role")=="assistant"),None)
  path=f"source-sample/robovqa_understanding/row-{row}-{video_id}.json";add(tar,path,(json.dumps(record,ensure_ascii=False,indent=2,default=str)+"\n").encode())
  index[video_id]={"found":True,"source_row":row,"record_archive_path":path,"first_user":user,"first_assistant":assistant,"width":metadata.get("width"),"height":metadata.get("height"),"num_frames":metadata.get("num_frames"),"framerate":str(metadata.get("framerate")) if metadata.get("framerate") is not None else None,"task_count":len(tasks),"task_types":[task.get("task") for task in tasks],"task_splits":[task.get("split") for task in tasks],"first_task":tasks[0] if tasks else None}
 add(tar,"qa-index.json",(json.dumps(index,ensure_ascii=False,indent=2,sort_keys=True,default=str)+"\n").encode())
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
 DATASET_DIR.mkdir(parents=True,exist_ok=True);cache=Path(f"/tmp/vlm_robovqa_tar_index_{SEED}.json");all_videos=[];archive_population=Counter()
 if cache.exists():all_videos=json.loads(cache.read_text());archive_population.update(item["archive"] for item in all_videos)
 else:
  for start in range(0,100,10):
   process=run(REMOTE_TAR_INDEX,{"root":REMOTE_ROOT,"parts":list(range(start,start+10))},subprocess.PIPE)
   if process.returncode:DATASET_DIR.rmdir();raise RuntimeError(process.stderr.decode(errors="replace"))
   items=json.loads(process.stdout);all_videos.extend(items);archive_population.update(item["archive"] for item in items)
  cache.write_text(json.dumps(all_videos,ensure_ascii=False))
 rng=random.Random(SEED);positions=rng.sample(range(len(all_videos)),CANDIDATES);candidates=[{**all_videos[position],"draw_order":order,"population_index":position} for order,position in enumerate(positions)]
 groups={}
 for item in candidates:groups.setdefault(int(Path(item["archive"]).stem.split("_")[-1].split(".")[0])//10,[]).append(item)
 valid=[];decisions=[]
 with tempfile.TemporaryDirectory(prefix="robovqa-") as tv:
  temp=Path(tv)
  for group,group_items in sorted(groups.items()):
   batch=temp/f"batch-{group}";batch.mkdir();archive=temp/f"batch-{group}.tar"
   with archive.open("wb") as out:process=run(REMOTE_VIDEO_BATCH,{"root":REMOTE_ROOT,"candidates":group_items},out)
   if process.returncode:DATASET_DIR.rmdir();raise RuntimeError(process.stderr.decode(errors="replace"))
   extract(archive,batch)
   for line in (batch/"batch-results.jsonl").read_text().splitlines():
    item=json.loads(line);(decisions if "reason" in item else valid).append((item,batch))
  selected=[];seen_frames=set();seen_videos=set()
  for item,batch in sorted(valid,key=lambda x:x[0]["draw_order"]):
   if item["video_id"] in seen_videos or item["preview_sha256"] in seen_frames:decisions.append(({"draw_order":item["draw_order"],"archive":item["archive"],"member":item["member"],"reason":"duplicate_video_or_preview"},batch))
   elif len(selected)<TARGET:seen_videos.add(item["video_id"]);seen_frames.add(item["preview_sha256"]);selected.append((item,batch))
   else:decisions.append(({"draw_order":item["draw_order"],"archive":item["archive"],"member":item["member"],"reason":"reserve_candidate_not_needed"},batch))
  if len(selected)!=TARGET:DATASET_DIR.rmdir();raise RuntimeError(f"only {len(selected)} valid unique previews")
  for item,batch in selected:
   target=DATASET_DIR/item["preview_archive_path"];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(batch/item["preview_archive_path"],target)
  qa_archive=temp/"qa.tar"
  with qa_archive.open("wb") as out:process=run(REMOTE_QA,{"root":REMOTE_ROOT,"video_ids":[item["video_id"] for item,_ in selected]},out)
  if process.returncode:raise RuntimeError(process.stderr.decode(errors="replace"))
  extract(qa_archive,DATASET_DIR)
 qa=json.loads((DATASET_DIR/"qa-index.json").read_text());missing=[key for key,value in qa.items() if not value["found"]]
 if missing:raise RuntimeError(f"missing understanding records for {len(missing)} videos")
 manifest=[]
 for item,_ in selected:manifest.append({**item,**qa[item["video_id"]]})
 decision_items=sorted([x for x,_ in decisions],key=lambda x:x["draw_order"]);selected_archives=Counter(x["archive"] for x in manifest);task_types=Counter(task for x in manifest for task in x["task_types"] if task);splits=Counter(split for x in manifest for split in x["task_splits"] if split)
 source_files={}
 for i in range(5):source_files[f"robovqa_reasoning_{i}.json"]=[1701256261,1708592315,1696071456,1703905930,1704408593][i]
 source_files["robovqa_understanding.json"]=1302648980
 summary={"dataset":"RoboVQA","source_root":REMOTE_ROOT,"seed":SEED,"video_population_total":len(all_videos),"tar_archive_count":len(archive_population),"source_json_files":source_files,"candidate_count":CANDIDATES,"selected_count":TARGET,"selected_archive_counts":dict(selected_archives),"invalid_or_duplicate_count":sum(x["reason"]!="reserve_candidate_not_needed" for x in decision_items),"reserve_candidate_count":sum(x["reason"]=="reserve_candidate_not_needed" for x in decision_items),"sample_task_type_counts":task_types.most_common(),"sample_task_split_counts":dict(splits),"num_frames":{"min":min(x["num_frames"] for x in manifest),"max":max(x["num_frames"] for x in manifest)},"framerate_values":Counter(x["framerate"] for x in manifest).most_common(20),"preview_width":{"min":min(x["width"] for x in manifest),"max":max(x["width"] for x in manifest)},"preview_height":{"min":min(x["height"] for x in manifest),"max":max(x["height"] for x in manifest)},"read_only_source":True}
 (DATASET_DIR/"sampling-manifest.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True,default=str)+"\n" for x in manifest));(DATASET_DIR/"sampling-candidate-decisions.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in decision_items));(DATASET_DIR/"sampling-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
 layout="".join(f"file\t{name}\t{size}\n" for name,size in source_files.items())+"directory\tclips\t100 tar.gz archives\n"+"".join(f"archive-member-count\t{name}\t{count}\n" for name,count in sorted(archive_population.items()))
 (DATASET_DIR/"source-layout.txt").write_text(layout);(DATASET_DIR/"source-schema.json").write_text(json.dumps({"logical_record":{"video":"clips/{video_id}.mp4","conversations":"system,user,assistant messages","metadata":"width,height,num_frames,num_bytes,framerate,video_location,task_metadata"},"video_storage":"clips/clips_part_000.tar.gz .. clips_part_099.tar.gz, members ./video_id.mp4","preview":"derived locally for inspection with ffmpeg thumbnail=100; not a source training frame","analyzed_qa_join":"robovqa_understanding.json by Path(video).stem"},ensure_ascii=False,indent=2)+"\n")
 cache.unlink(missing_ok=True)
def manifest():return [json.loads(x) for x in (DATASET_DIR/"sampling-manifest.jsonl").read_text().splitlines() if x]
def render():
 s=json.loads((DATASET_DIR/"sampling-summary.json").read_text());items=manifest();task_rows="".join(f"<tr><td>{html.escape(task)}</td><td>{count}</td></tr>" for task,count in s["sample_task_type_counts"][:20]);cards=[]
 for x in items:
  media=f"sub-dataset/RoboVQA/{x['preview_archive_path']}";question=x["first_user"] or "<missing>";answer=x["first_assistant"] or "<missing>";detail=html.escape(json.dumps({k:x[k] for k in ("video_id","archive","member","source_video_sha256","preview_generation","preview_pts_time_seconds","num_frames","framerate","task_count","source_row")},ensure_ascii=False,indent=2))
  cards.append(f'''<article class="sample"><a href="{media}"><img loading="lazy" src="{media}" alt="{x['sample_id']} RoboVQA derived preview"></a><div class="sample-body"><div class="sample-head"><strong>{x['sample_id']}</strong><span>{x['num_frames']} frames</span></div><p><strong>问：</strong>{html.escape(question[:260])}<br><strong>答：</strong>{html.escape(answer[:300])}</p><dl><dt>预览时刻</dt><dd>{x['preview_pts_time_seconds']} s（派生）</dd><dt>源归档</dt><dd><code>{html.escape(Path(x['archive']).name)}</code></dd></dl><details><summary>字段与追溯信息</summary><pre>{detail}</pre></details></div></article>''')
 doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RoboVQA 源数据集 200 视频随机样本分析</title><style>{STYLE}</style></head><body><main><header><div class="eyebrow">VLM SOURCE DATASET / SAMPLE ANALYSIS 11</div><h1>RoboVQA：200 个视频的派生帧样本分析</h1><p class="verdict"><strong>总判断：</strong>这是当前资产中真正的视频具身问答源：模型输入是短机器人视频与问题，输出分为通用场景/动作描述，以及带 <code>&lt;think&gt;</code>、<code>&lt;answer&gt;</code> 的任务推理。图片卡片只是为人工审阅生成的代表帧，不是源训练样本，也不能替代时序解码。</p></header><section class="facts"><div class="fact"><b>{s['video_population_total']:,}</b><span>100 个 tar 中视频成员</span></div><div class="fact"><b>200</b><span>随机不同视频</span></div><div class="fact"><b>6</b><span>源 conversation JSON</span></div><div class="fact"><b>{SEED}</b><span>固定随机种子</span></div></section><section><h2>一、总体：视频输入与两层问答</h2><p>源路径为 <code>{REMOTE_ROOT}</code>。视频分装在 100 个 <code>clips_part_*.tar.gz</code> 中，JSON 的 <code>video</code> 和 <code>metadata.video_location</code> 保存逻辑路径；理解文件提供对象/动作描述，五个 reasoning 文件提供任务问题、思考文本和最终答案。<code>metadata.task_metadata</code> 还保留上游任务类型、指令、问题和答案。</p><div class="band"><div class="flow"><span>机器人 MP4 视频</span><b>+</b><span>user question</span><b>→</b><span>assistant description</span><b>或</b><span>&lt;think&gt; + &lt;answer&gt;</span></div></div><p class="note"><strong>代表帧边界：</strong>归档图片由 ffmpeg <code>thumbnail=100</code> 从每个入选 MP4 派生，manifest 保存源视频 SHA256 和输出 pts_time。它只用于查看画面，任何训练、时序、成功判断或规划分析都必须回到原 MP4 与 conversation。</p></section><section><h2>二、全量视频随机抽样</h2><p>完整解压扫描 100 个 tar.gz 的成员表，以全部 {s['video_population_total']:,} 个 MP4 为总体，固定种子等概率无放回抽取 400 个候选；按归档分组只读解码并固定前 200 个不同视频、不同预览哈希。随后完整扫描 understanding JSON，按 video_id 连接原始 conversation 和 task_metadata。</p></section><section><h2>三、200 个视频的样本分布</h2><p>源视频帧数 {s['num_frames']['min']}–{s['num_frames']['max']}；预览尺寸宽 {s['preview_width']['min']}–{s['preview_width']['max']} px、高 {s['preview_height']['min']}–{s['preview_height']['max']} px。下表按 200 个视频携带的全部 task_metadata 计数，同一视频可贡献多个任务。</p><div class="table-wrap"><table><thead><tr><th>task type</th><th>条目数</th></tr></thead><tbody>{task_rows}</tbody></table></div></section><section><h2>四、逐视频：200 张派生预览与原始 QA</h2><div class="gallery">{''.join(cards)}</div></section><footer>只读源审计：tar.gz 与 JSON 仅顺序读取；视频经内存管道送入 ffmpeg，远端不落 MP4 或帧文件。</footer></main></body></html>''';REPORT_PATH.write_text(doc)
def verify():
 items=manifest()
 if len(items)!=200 or len({x["video_id"] for x in items})!=200 or len({x["preview_sha256"] for x in items})!=200:raise RuntimeError("manifest/video/frame uniqueness mismatch")
 for x in items:
  payload=(DATASET_DIR/x["preview_archive_path"]).read_bytes()
  if hashlib.sha256(payload).hexdigest()!=x["preview_sha256"]:raise RuntimeError("digest mismatch")
  with Image.open(io.BytesIO(payload)) as image:image.verify()
 if REPORT_PATH.read_text().count('<article class="sample"')!=200:raise RuntimeError("card mismatch")
 print(json.dumps({"dataset":"RoboVQA","manifest_rows":200,"unique_video_ids":200,"unique_preview_sha256":200,"readable_previews":200,"report_sample_cards":200},ensure_ascii=False))
def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=("collect","render","verify","all"));a=p.parse_args()
 if a.command in {"collect","all"}:collect()
 if a.command in {"render","all"}:render()
 if a.command in {"verify","all"}:verify()
if __name__=="__main__":main()
