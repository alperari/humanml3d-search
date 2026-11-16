"""Local video galleries for lexical search, semantic search and their overlap."""

import json
from fractions import Fraction
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .common import atomic_json, base_id, fingerprint, now, read_json, require


def visualize(config, run_directory, copy_videos=False, method=None):
    directory = config.protect(run_directory)
    report, report_facts = read_json(directory / "run_manifest.json")
    require(report["status"] == "passed", "Search must finish successfully before visualization")
    require(report.get("schema_version") == 2, "Rerun search to generate separate lexical and semantic result files")
    require(report["dataset_root"] == str(config.dataset_root), "Search used a different dataset root")
    require(method in (None, "lexical", "semantic", "overlap"), "Unknown gallery selection")
    for name, facts in report["outputs"].items():
        require(Path(name).name == name, "Invalid search output name")
        require(fingerprint(directory / name)["sha256"] == facts["sha256"], f"Search output changed: {name}")
    return {name: build_gallery(config, directory, name, report, report_facts, copy_videos)
            for name in ((method,) if method else ("lexical", "semantic", "overlap"))}


def build_gallery(config, directory, method, report, report_facts, copy_videos):
    results, _ = read_json(directory / f"{method}_results.json")
    require(results["method"] == method, "Result file belongs to a different search method")
    media = directory / "media" / method
    media.mkdir(parents=True, exist_ok=True)
    report_path = directory / f"{method}_gallery_manifest.json"
    html_path = directory / f"{method}_review.html"
    review = {"schema_version": 2, "stage": "visual_review_gallery", "method": method, "status": "running", "created_at_utc": now(),
              "run_signature": report["run_signature"], "search_manifest": report_facts,
              "video_mode": "copy" if copy_videos else "symlink", "fps": 20, "videos": {},
              "review_status": "Candidate retrieval only; human relevance review remains required"}
    atomic_json(report_path, review)
    try:
        ffprobe = shutil.which("ffprobe")
        require(ffprobe is not None, "Gallery creation needs ffprobe from the active project environment")
        for record in results["records"]:
            motion_id = record["motion_id"]
            base_id(motion_id)
            source = config.dataset_root / "animations" / f"{motion_id}.mp4"
            target = media / f"{motion_id}.mp4"
            if source.is_file():
                source_facts = fingerprint(source)
                probe = json.loads(subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                                   "stream=codec_name,pix_fmt,avg_frame_rate,nb_frames,duration,width,height", "-of", "json", str(source)],
                                                  capture_output=True, text=True, check=True).stdout)["streams"]
                require(len(probe) == 1 and probe[0]["codec_name"] == "h264" and probe[0]["pix_fmt"] == "yuv420p"
                        and Fraction(probe[0]["avg_frame_rate"]) == 20, f"Preview needs browser-compatible H.264 at 20 fps: {source}")
                require(fingerprint(source)["sha256"] == source_facts["sha256"], f"Video changed while probing: {source}")
                if copy_videos:
                    with tempfile.NamedTemporaryFile(dir=media, prefix=".video-", suffix=".mp4", delete=False) as stream:
                        temporary = Path(stream.name)
                    try:
                        shutil.copyfile(source, temporary)
                        require(fingerprint(temporary)["sha256"] == source_facts["sha256"], f"Video changed while copying: {source}")
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
                else:
                    temporary = media / f".{motion_id}.link"
                    temporary.unlink(missing_ok=True)
                    temporary.symlink_to(source)
                    os.replace(temporary, target)
                review["videos"][motion_id] = {"source": "original_animation", "input": source_facts, "probe": probe[0]}
            else:
                from .rendering import render_joints

                # Never write through an old preview symlink into the dataset.
                target.unlink(missing_ok=True)
                review["videos"][motion_id] = render_joints(config.dataset_root / "new_joints" / f"{motion_id}.npy", target, motion_id)
            record["video"] = f"media/{method}/{motion_id}.mp4"
        data = dict(results, run_signature=report["run_signature"], split=report["settings"]["split"], fps=20)
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        html = HTML.replace("__RESULT_DATA__", encoded)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".review-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(html)
        os.replace(temporary, html_path)
        review["outputs"] = {html_path.name: fingerprint(html_path)}
        review["status"] = "passed"
    except Exception as exc:
        review["status"], review["error"] = "failed", str(exc)
        raise
    finally:
        atomic_json(report_path, review)
    print(f"{method.capitalize()} review gallery: {html_path}", flush=True)
    return html_path


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HumanML3D motion review</title>
<style>
:root{font:16px system-ui,sans-serif;color:#192c3c;background:#eef3f6}body{margin:0 auto;max-width:1480px;padding:24px}
h1{font-size:1.8rem;margin:0 0 8px}p{line-height:1.5}.controls{position:sticky;top:0;z-index:2;background:#eef3f6f5;padding:16px 0;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
input,button{font:inherit;padding:8px;border:1px solid #9aabb7;border-radius:5px;background:white}button{cursor:pointer}button:hover{background:#e7f1fb}input[type=checkbox]{width:18px;height:18px;vertical-align:middle}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:20px}article{background:white;padding:16px;border-radius:10px;border:1px solid #d6e0e7;min-width:0}
article h2{font-size:1.15rem;margin:0 0 8px}video{width:100%;aspect-ratio:1;background:#f9f9f9}.tag{display:inline-block;padding:3px 7px;background:#e4eef8;border-radius:4px;margin:0 5px 8px 0;font-size:.85rem}
.caption{border-top:1px solid #e3e9ee;padding-top:10px;margin-top:12px}.caption p{margin:6px 0}.detail{font-size:.85rem;color:#486071;overflow-wrap:anywhere}#summary{font-size:.95rem}
</style>
<h1 id="title"></h1><p id="summary"></p>
<p>These are caption-search candidates. Review the motion before assigning an activity label. Mirrored pairs share one base ID; a base ID does not identify a distinct person or source recording.</p>
<p><a id="results-link">Download results</a> <a id="evidence">Caption evidence</a></p>
<div class="controls">
<label><input type="checkbox" id="mirrors"> Show mirrored counterparts</label><label>Find <input id="find" placeholder="Motion ID or caption"></label><span id="shown" aria-live="polite"></span></div>
<main id="grid"></main><script id="results" type="application/json">__RESULT_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('results').textContent), grid=document.getElementById('grid');
const methodLabel=data.method.charAt(0).toUpperCase()+data.method.slice(1);
document.getElementById('title').textContent=data.label+' — '+methodLabel+' motion review';
document.title=data.label+' — '+methodLabel+' motion review';
document.getElementById('results-link').href=data.method+'_results.'+(data.method==='overlap'?'json':'csv');
if(data.method==='overlap')document.getElementById('evidence').remove();else document.getElementById('evidence').href=data.method+'_matches.jsonl';
document.getElementById('summary').textContent=`Split: ${data.split}. ${data.counts.motion_ids} motion IDs / ${data.counts.base_motion_ids} base IDs. Run ${data.run_signature.slice(0,16)}. Playback: 20 fps.`;
if(data.method==='overlap')document.getElementById('summary').textContent+=` Exact-ID gallery: ${data.exact_match_base_motion_ids} base groups. Mirror-adjusted overlap is listed in the results JSON.`;
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function card(r){
 const a=el('article'), title=el('h2',r.motion_id);a.append(title);
 a.append(el('span',methodLabel,'tag'),el('span',r.split,'tag'),el('span','Base '+r.base_motion_id,'tag'));
 const semantic=data.method==='overlap'?r.semantic:data.method==='semantic'?r:null;
 if(semantic)a.append(el('p',`Semantic rank ${semantic.rank} · cosine ${semantic.cosine_similarity.toFixed(4)}`,'detail'));
 const v=document.createElement('video');v.src=r.video;v.controls=true;v.preload='none';v.playsInline=true;v.loop=true;a.append(v);
 let bounds=null;
 function play(start,end){bounds=start===null?null:[start,end];v.loop=!bounds;const seek=()=>{v.currentTime=bounds?bounds[0]:0;v.play().catch(()=>{});};if(v.readyState>=1)seek();else{v.addEventListener('loadedmetadata',seek,{once:true});v.load();}}
 v.addEventListener('timeupdate',()=>{if(bounds&&v.currentTime>=bounds[1])v.currentTime=bounds[0];});
 v.addEventListener('ended',()=>{if(bounds){v.currentTime=bounds[0];v.play().catch(()=>{});}});
 const full=el('button','Play full motion');full.onclick=()=>play(null,null);a.append(full);
 const captions=data.method==='overlap'?[['lexical',r.lexical],['semantic',r.semantic]]:[[data.method,r]];
 for(const [method,cap] of captions){
  const d=el('div',undefined,'caption');d.append(el('strong',method+' · '+cap.caption_id),el('p',cap.caption));
  const details=method==='lexical'?`${cap.match_type}; rules: ${cap.matched_rules.join(', ')}; matching captions: ${cap.matched_caption_count}`:`Query ${cap.query_id}: ${cap.query}`;
  d.append(el('p',details,'detail'));
  const valid=Number.isInteger(cap.start_frame)&&Number.isInteger(cap.end_frame)&&cap.end_frame>cap.start_frame;
  d.append(el('p',valid?`${cap.segment_status}: frames [${cap.start_frame}, ${cap.end_frame}) / ${(cap.start_frame/data.fps).toFixed(2)}–${(cap.end_frame/data.fps).toFixed(2)} s`:`${cap.segment_status}: no valid caption interval; use full motion`,'detail'));
  if(valid){const button=el('button','Play this caption interval');button.onclick=()=>play(cap.start_frame/data.fps,cap.end_frame/data.fps);d.append(button);}a.append(d);
 }
 return a;
}
function render(){
 const expand=document.getElementById('mirrors').checked,q=document.getElementById('find').value.trim().toLowerCase();
 let rows=data.records.filter(r=>(r.motion_id+' '+(r.caption||'')+' '+(r.lexical?.caption||'')+' '+(r.semantic?.caption||'')).toLowerCase().includes(q));
 if(!expand){const reps=new Map();for(const r of rows){if(!reps.has(r.base_motion_id)||!r.is_mirrored)reps.set(r.base_motion_id,r.motion_id);}rows=rows.filter(r=>reps.get(r.base_motion_id)===r.motion_id);}
 for(const v of grid.querySelectorAll('video'))v.pause();grid.replaceChildren(...rows.map(card));document.getElementById('shown').textContent=rows.length+' shown';
}
document.getElementById('mirrors').addEventListener('change',render);document.getElementById('find').addEventListener('input',render);render();
</script></html>'''
