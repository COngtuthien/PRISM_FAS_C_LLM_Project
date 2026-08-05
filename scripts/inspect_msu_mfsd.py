"""Read-only MSU-MFSD inventory. It never opens media payloads or writes below raw_root."""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

def relative(path: Path, root: Path) -> str: return path.relative_to(root).as_posix()
def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--raw-root", type=Path, required=True); parser.add_argument("--reports-root", type=Path, required=True); args=parser.parse_args()
    root=args.raw_root.resolve(); output=(args.reports_root.resolve()/"raw_audit"); output.mkdir(parents=True, exist_ok=True)
    if not root.is_dir(): raise SystemExit(f"raw root is not a directory: {root}")
    files=sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p:p.as_posix())
    ext=Counter(p.suffix.lower() or "<none>" for p in files); sizes=Counter()
    for file in files: sizes[file.suffix.lower() or "<none>"] += file.stat().st_size
    named=[p for p in files if re.search(r"readme|protocol|train|test|dev|split|label|ground[ _-]?truth|client|attack|real|fake", p.name, re.I) or p.suffix.lower() in {".txt",".csv",".json",".xml",".mat"}]
    text=[]
    for p in named:
        if p.suffix.lower() not in {".txt",".csv",".json",".xml"} or p.stat().st_size > 1_048_576: continue
        try: text.append({"path":relative(p,root),"content":p.read_text(encoding="utf-8",errors="replace")})
        except OSError as exc: text.append({"path":relative(p,root),"read_error":str(exc)})
    inventory={"raw_root":str(root),"generated_at":datetime.now(timezone.utc).isoformat(),"file_count":len(files),"top_level":sorted(p.name for p in root.iterdir()),"extensions":[{"extension":k,"file_count":ext[k],"total_bytes":sizes[k]} for k in sorted(ext)],"sample_paths":[relative(p,root) for p in files[:100]],"named_metadata_paths":[relative(p,root) for p in named],"text_metadata":text,"video_metadata":[{"path":relative(p,root),"bytes":p.stat().st_size,"probe_note":"ffprobe unavailable on host; no decoding/frame extraction performed."} for p in files if p.suffix.lower() in {".avi",".mov",".mp4"}][:5],"archive_paths":[relative(p,root) for p in files if p.suffix.lower() in {".zip",".rar",".7z",".001"}]}
    (output/"msu_mfsd_file_inventory.json").write_text(json.dumps(inventory,indent=2),encoding="utf-8")
    (output/"msu_mfsd_sample_paths.txt").write_text("\n".join(inventory["sample_paths"])+"\n",encoding="utf-8")
    lines=["# MSU-MFSD read-only deep inspection","",f"- Raw root: `{root}`",f"- File count: {len(files)}",f"- Top level: {', '.join(inventory['top_level'])}","","## Extension distribution",""]
    lines += [f"- `{entry['extension']}`: {entry['file_count']} files; {entry['total_bytes']} bytes" for entry in inventory["extensions"]]
    lines += ["","## Metadata/protocol files","",* [f"- `{p}`" for p in inventory["named_metadata_paths"]],"","## Video metadata","","ffprobe is unavailable; no media decoding or frame extraction was performed."]
    (output/"msu_mfsd_deep_inspection.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__ == "__main__": main()
