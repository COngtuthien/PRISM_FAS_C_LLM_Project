from pathlib import Path
import argparse, hashlib, json, time
import cv2, pandas as pd, yaml
from prism_fas.config.models import DatasetDefinition,load_paths
from prism_fas.data.adapters import adapter_for
from prism_fas.data.preprocess_m2 import SCRFDDetector,select_largest
from prism_fas.utils.core import sha256_file,stable_json_hash,utc_timestamp
def frames(record):
 p=record.source_path
 if record.dataset=='casia_fasd':
  import re
  m=re.match(r'(?P<prefix>[bf]?s\d+v[A-Za-z0-9_]+)f\d+\.png$',p.name); fs=sorted(p.parent.glob(m['prefix']+'f*.png'),key=lambda q:int(__import__('re').search(r'f(\d+)\.png$',q.name)[1])); return [(i,fs[i]) for i in sorted(set([0,len(fs)//2,len(fs)-1]))]
 cap=cv2.VideoCapture(str(p)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out=[]
 for i in sorted(set([0,max(0,n//2),max(0,n-1)])):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read()
  if ok: out.append((i,img))
 cap.release(); return out
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True);a.add_argument('--preprocess-config',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--seed',type=int,default=2026);a.add_argument('--records-per-dataset',type=int,default=20);a.add_argument('--frames-per-record',type=int,default=3);args=a.parse_args(); paths=load_paths(args.config); conf=yaml.safe_load(args.preprocess_config.read_text()); out=args.output_dir;out.mkdir(parents=True,exist_ok=True); model=Path(conf['scrfd_model_path']); mh=sha256_file(model); selected=[]; results=[]; frame_rows=[]
 for name in ['casia_fasd','msu_mfsd']:
  d=DatasetDefinition.model_validate(yaml.safe_load((Path('configs/data')/(name+'.yaml')).read_text())); rs=adapter_for(d,getattr(paths.raw_datasets,name)).records(); rs=sorted(rs,key=lambda r:hashlib.sha256(f'{name}|{r.video_id}|{args.seed}'.encode()).hexdigest())[:args.records_per_dataset]
  for rank,r in enumerate(rs):
   rid=f'{name}:{r.video_id}';selected.append({'dataset':name,'canonical_record_id':rid,'video_id':r.video_id,'media_type':'image_sequence' if name=='casia_fasd' else 'video_file','selection_hash':hashlib.sha256(f'{name}|{r.video_id}|{args.seed}'.encode()).hexdigest(),'selection_rank':rank,'seed':args.seed,'adapter_version':r.adapter_version,'source_fingerprint':r.source_fingerprint})
   for slot,(idx,x) in enumerate(frames(r)):
    img=cv2.imread(str(x)) if isinstance(x,Path) else x
    if img is None:continue
    fp=hashlib.sha256(img.tobytes()).hexdigest();frame_rows.append({'dataset':name,'canonical_record_id':rid,'frame_slot':slot,'requested_frame_index':idx,'actual_frame_index':idx,'decode_status':'ok','image_width':img.shape[1],'image_height':img.shape[0],'frame_fingerprint':fp})
    for size in [256,320]:
     det=SCRFDDetector(model,size);t=time.perf_counter();ds=det.detect(img);ms=(time.perf_counter()-t)*1000;sel=select_largest(ds,.5,conf['min_face_size']); valid=sel is not None and len(sel.landmarks)==5
     results.append({'dataset':name,'canonical_record_id':rid,'frame_slot':slot,'requested_frame_index':idx,'actual_frame_index':idx,'frame_fingerprint':fp,'input_size':size,'threshold':.5,'detector_model_sha256':mh,'provider':'CPUExecutionProvider','output_shape_valid':True,'decode_success':True,'detection_count_above_threshold':len(ds),'valid_detection_count':int(valid),'valid_detection':valid,'selected_score':sel.score if sel else None,'selected_bbox_valid':valid,'landmark_valid':valid,'inference_time_ms':ms,'failure_code':None if valid else 'no_face'})
 pd.DataFrame(selected).to_parquet(out/'selected_records.parquet',index=False);pd.DataFrame(frame_rows).to_parquet(out/'validation_frames.parquet',index=False);pd.DataFrame(results).to_parquet(out/'size_results.parquet',index=False)
 df=pd.DataFrame(results); agg=df.groupby(['dataset','input_size']).agg(frames=('valid_detection','size'),valid=('valid_detection','sum'),rate=('valid_detection','mean'),time_ms=('inference_time_ms','mean')).reset_index();agg.to_csv(out/'comparison.csv',index=False); combined=agg.groupby('input_size')[['frames','valid']].sum();rates={str(i):float(combined.loc[i,'valid']/combined.loc[i,'frames']) for i in combined.index}; chosen=256 if rates['256']-rates['320']>.03 else 320; decision={'status':'SELECTED','selected_size':chosen,'threshold':.5,'candidate_sizes':[256,320],'rates':rates,'seed':args.seed,'model_sha256':mh,'timestamp':utc_timestamp(),'reason':'approved fixed rule'};(out/'aggregate_results.json').write_text(agg.to_json(orient='records'),encoding='utf-8');(out/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8');print(json.dumps(decision))
if __name__=='__main__':main()
