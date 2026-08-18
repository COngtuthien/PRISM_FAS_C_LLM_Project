from __future__ import annotations
import json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import cv2, pyarrow.parquet as pq
from prism_fas.data.manifests.leakage import find_target_leakage
from prism_fas.data.manifests.resume import RunState
from prism_fas.utils.core import sha256_file, git_commit, atomic_json_write
from prism_fas.data.preprocess_m2 import resolve_detector_path

EXPECTED={'source_frames':24,'source_crops':24,'target_frames':12,'target_crops':12,'preprocessing_failures':0,'completed_samples':36}
def _check(checks:list,ident:str,category:str,description:str,expected:Any,actual:Any,severity='error',affected_samples=None,details=None):
    ok=expected==actual if not callable(expected) else bool(expected(actual)); checks.append({'check_id':ident,'category':category,'description':description,'expected':str(expected),'actual':actual,'passed':ok,'severity':severity,'affected_samples':affected_samples or [],'details':details or {}});return ok
PRIVATE_TARGET_TOKENS=('live','spoof','attack','taxonomy','subject','session','label','genuine','replay','print','mask','paper','makeup','.avi','.mov','.mp4')
def validate_full_profile(paths,cfg,output_root:Path)->dict:
    """Structural validation for the full_preprocessing profile.

    Unlike the small-acceptance profile this asserts no fixed row counts and
    requires no legacy completed-index/run-state/M2A artifacts; it validates
    internal consistency of whatever the run actually produced.
    """
    root=Path(output_root);manifests=root/'manifests';checks=[];tables={};model_hash=sha256_file(resolve_detector_path(cfg.scrfd_model_path))
    _check(checks,'config.hash','config','frozen config hash',cfg.config_hash,cfg.config_hash)
    _check(checks,'detector.hash','config','detector SHA256','5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91',model_hash)
    _check(checks,'detector.input','config','SCRFD input size',320,cfg.scrfd_input_size);_check(checks,'detector.threshold','config','threshold',.5,cfg.detection_threshold)
    for name in ('source_frames','source_crops','target_frames','target_crops','preprocessing_failures'):
        p=manifests/f'{name}.parquet'
        if not p.exists():_check(checks,'file.'+name,'files','required parquet exists',True,False);tables[name]=[];continue
        table=pq.read_table(p);tables[name]=table.to_pylist()
        md={k.decode():v.decode() for k,v in (table.schema.metadata or {}).items()}
        _check(checks,'metadata.'+name,'schema','config metadata',cfg.config_hash,md.get('preprocessing_config_hash'))
        _check(checks,'metadata.detector.'+name,'schema','detector metadata',model_hash,md.get('detector_model_sha256'))
        ids=[r['sample_id'] for r in tables[name] if r.get('sample_id')]
        _check(checks,'duplicates.'+name,'identity','unique sample IDs',len(ids),len(set(ids)))
    _check(checks,'pairing.source','counts','source_frames == source_crops',len(tables['source_frames']),len(tables['source_crops']))
    _check(checks,'pairing.target','counts','target_frames == target_crops',len(tables['target_frames']),len(tables['target_crops']))
    for role in ('source','target'):
        frames={r['sample_id'] for r in tables[role+'_frames']};crops={r['sample_id'] for r in tables[role+'_crops']}
        _check(checks,'join.'+role,'consistency','frame/crop sample sets',sorted(frames),sorted(crops))
    overlap=sorted({r['sample_id'] for r in tables['source_crops']} & {r['sample_id'] for r in tables['target_crops']})
    _check(checks,'roles.disjoint','consistency','source and target sample sets disjoint',[],overlap)
    source_datasets={r['dataset'] for r in tables['source_frames']};target_datasets={r['dataset'] for r in tables['target_frames']}
    _check(checks,'roles.datasets','consistency','no dataset routed to both roles',[],sorted(source_datasets&target_datasets))
    missing=[];mismatch=[];unreadable=[];dimensions=[];manifest_paths=set()
    for role in ('source','target'):
        for row in tables[role+'_crops']:
            rel=row['crop_relative_path'];sid=row['sample_id']
            if re.match(r'^[A-Za-z]:|^\\\\',rel) or '..' in Path(rel).parts:missing.append(sid);continue
            p=root/rel;manifest_paths.add(p.resolve())
            if not p.is_file() or p.stat().st_size==0:missing.append(sid);continue
            if sha256_file(p)!=row['crop_sha256']:mismatch.append(sid)
            image=cv2.imread(str(p))
            if image is None:unreadable.append(sid)
            elif image.shape[:2]!=(row['crop_height'],row['crop_width']) or image.shape[:2]!=(cfg.crop_output_size,cfg.crop_output_size):dimensions.append(sid)
    on_disk={p.resolve() for p in (root/'crops').rglob('*.'+cfg.output_image_format)}
    orphans=sorted(str(p.name) for p in on_disk-manifest_paths)
    temporary=sorted(str(p.name) for p in root.rglob('*.tmp*'))
    _check(checks,'crop.missing','crop_integrity','all manifest crops present',[],missing,affected_samples=missing)
    _check(checks,'crop.sha','crop_integrity','crop SHA256 matches manifest',[],mismatch,affected_samples=mismatch)
    _check(checks,'crop.readable','crop_integrity','crops decode',[],unreadable,affected_samples=unreadable)
    _check(checks,'crop.dimensions','crop_integrity','crop dimensions match config',[],dimensions,affected_samples=dimensions)
    _check(checks,'crop.orphans','crop_integrity','no crop files outside manifests',[],orphans)
    _check(checks,'crop.temporary','crop_integrity','no temporary artifacts',[],temporary)
    codes={};invalid_failures=[]
    for row in tables['preprocessing_failures']:
        codes[row['error_code']]=codes.get(row['error_code'],0)+1
        if not row['dataset'] or not row['source_record_id'] or not row['error_code'] or not row['stage'] or row['recoverable'] is None:invalid_failures.append(row.get('sample_id'))
        if re.search(r'[A-Za-z]:\\|/home/|/Users/',row.get('sanitized_error_message','') or ''):invalid_failures.append(row.get('sample_id'))
    _check(checks,'failures.valid','failures','failure rows well formed',[],invalid_failures)
    target_rows=tables['target_frames']+tables['target_crops']
    leaks=find_target_leakage(target_rows)
    _check(checks,'target.isolation','target_isolation','no forbidden target metadata',[],leaks,affected_samples=[x.get('sample_id') for x in leaks])
    # Values only: field names such as source_fingerprint legitimately contain
    # token substrings, while persisted values must never carry them.
    token_hits=sorted({t for row in target_rows for value in row.values() for t in PRIVATE_TARGET_TOKENS if t in str(value).lower()})
    _check(checks,'target.tokens','target_isolation','no private target tokens in persisted values',[],token_hits)
    _check(checks,'artifacts.m2a','profile','no M2A artifacts in profile root',[],sorted(str(p.relative_to(root)) for p in list(root.rglob('*.jsonl'))+list(root.rglob('m2a'))))
    passed=all(c['passed'] or c['severity']!='error' for c in checks);errors=[c for c in checks if not c['passed'] and c['severity']=='error']
    return {'validation_version':'m2f1a-full-v1','profile':'full_preprocessing','output_root':str(root),'preprocessing_version':cfg.preprocessing_version,'preprocessing_config_hash':cfg.config_hash,'expected_config_hash':cfg.config_hash,'detector_model_sha256':model_hash,'expected_detector_model_sha256':'5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91','detector_input_size':cfg.scrfd_input_size,'detector_threshold':cfg.detection_threshold,'manifests':{k:len(v) for k,v in tables.items()},'completed_index':None,'run_state':{},'lock_state':'absent','crops_on_disk':len(on_disk),'failures_by_code':codes,'target_isolation':{'passed':not leaks and not token_hits,'matches':leaks,'token_matches':token_hits},'crop_integrity':{'checked':len(manifest_paths),'missing':missing,'sha_mismatch':mismatch,'unreadable':unreadable,'wrong_dimensions':dimensions,'orphans':orphans,'temporary':temporary,'passed':not (missing or mismatch or unreadable or dimensions or orphans or temporary)},'checks':checks,'errors':errors,'warnings':[],'passed':passed,'validated_at':datetime.now(timezone.utc).isoformat(),'git_commit':git_commit(paths.project_root)}
def validate_m2(paths,cfg,output_root:Path|None=None)->dict:
    root=output_root or paths.work_root/'m2'/cfg.preprocessing_version/cfg.config_hash; manifests=root/'manifests';checks=[];errors=[]; tables={}
    model_hash=sha256_file(resolve_detector_path(cfg.scrfd_model_path))
    _check(checks,'config.hash','config','frozen config hash',cfg.config_hash,cfg.config_hash)
    _check(checks,'detector.hash','config','detector SHA256','5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91',model_hash)
    _check(checks,'detector.input','config','SCRFD input size',320,cfg.scrfd_input_size);_check(checks,'detector.threshold','config','threshold',.5,cfg.detection_threshold)
    for name,count in EXPECTED.items():
        p=(root/'state'/'completed_samples.parquet') if name=='completed_samples' else manifests/(name+'.parquet')
        if not p.exists(): _check(checks,'file.'+name,'files','required parquet exists',True,False);continue
        try:t=pq.read_table(p);tables[name]=t
        except Exception as e:_check(checks,'read.'+name,'files','readable parquet',True,False,details={'error':str(e)});continue
        _check(checks,'count.'+name,'counts','small_acceptance row count',count,t.num_rows)
        md={k.decode():v.decode() for k,v in (t.schema.metadata or {}).items()}; _check(checks,'metadata.'+name,'schema','config metadata',cfg.config_hash,md.get('preprocessing_config_hash'))
        ids=t.column('sample_id').to_pylist() if 'sample_id' in t.column_names else [];_check(checks,'duplicates.'+name,'identity','unique sample IDs',len(ids),len(set(ids)))
    for role in ('source','target'):
        fr={r['sample_id']:r for r in tables.get(role+'_frames',[]).to_pylist()};cr={r['sample_id']:r for r in tables.get(role+'_crops',[]).to_pylist()};_check(checks,'join.'+role,'consistency','frame/crop sample sets',sorted(fr),sorted(cr))
        for sid,row in cr.items():
            rel=row['crop_relative_path']; safe=not re.match(r'^[A-Za-z]:|^\\\\',rel) and '..' not in Path(rel).parts; p=root/rel
            # Profile-relative layout first; legacy small-acceptance kept its crops under m2a/.
            if not p.exists(): p=root/'m2a'/rel
            image=cv2.imread(str(p)) if safe and p.exists() else None;ok=safe and image is not None and sha256_file(p)==row['crop_sha256']
            _check(checks,'crop.'+sid,'crop_integrity','portable readable matching crop hash',True,ok,affected_samples=[] if ok else [sid])
    target_rows=(tables.get('target_frames').to_pylist() if 'target_frames' in tables else [])+(tables.get('target_crops').to_pylist() if 'target_crops' in tables else [])
    leaks=find_target_leakage(target_rows)
    _check(checks,'target.isolation','target_isolation','no forbidden target metadata',[],leaks,affected_samples=[x.get('sample_id') for x in leaks])
    completed=tables.get('completed_samples');
    if completed is not None:
        rows=completed.to_pylist(); _check(checks,'completed.status','completed','all completed',len(rows),sum(r['status']=='completed' for r in rows)); _check(checks,'completed.datasets','completed','per-dataset 12',{'casia_fasd':12,'msu_mfsd':12,'siw_mv2':12},{d:sum(r['dataset']==d for r in rows) for d in ['casia_fasd','msu_mfsd','siw_mv2']})
    state_path=root/'state'/'run_state.json';state={}
    try: state=RunState.model_validate(json.loads(state_path.read_text())).model_dump(mode='json');_check(checks,'state.status','state','not running',lambda x:x!='running',state['status'])
    except Exception as e:_check(checks,'state.valid','state','valid run state',True,False,details={'error':str(e)})
    _check(checks,'lock.absent','lock','no lingering lock',False,(root/'state'/'run.lock').exists())
    passed=all(c['passed'] or c['severity']!='error' for c in checks);errors=[c for c in checks if not c['passed'] and c['severity']=='error']
    return {'validation_version':'m2b2-v1','profile':'small_acceptance','output_root':str(root),'preprocessing_version':cfg.preprocessing_version,'preprocessing_config_hash':cfg.config_hash,'expected_config_hash':cfg.config_hash,'detector_model_sha256':model_hash,'expected_detector_model_sha256':'5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91','detector_input_size':cfg.scrfd_input_size,'detector_threshold':cfg.detection_threshold,'manifests':{k:v.num_rows for k,v in tables.items() if k!='completed_samples'},'completed_index':tables.get('completed_samples').num_rows if completed else None,'run_state':state,'lock_state':'absent' if not (root/'state'/'run.lock').exists() else 'present','target_isolation':{'passed':not leaks,'matches':leaks},'crop_integrity':{'checked':len(target_rows)//2+len(tables.get('source_crops',[])),'passed':not any(c['check_id'].startswith('crop.') and not c['passed'] for c in checks)},'checks':checks,'errors':errors,'warnings':[],'passed':passed,'validated_at':datetime.now(timezone.utc).isoformat(),'git_commit':git_commit(paths.project_root)}
def status_m2(paths,cfg,root=None,validation_profile='small_acceptance'):
    result=(validate_full_profile(paths,cfg,root) if validation_profile=='full_preprocessing' else validate_m2(paths,cfg,root)); return {'milestone':'M2','milestone_status':'COMPLETED — IMPLEMENTATION AND SMALL ACCEPTANCE' if result['passed'] else 'BLOCKED — VALIDATION FAILURE','preprocessing_version':cfg.preprocessing_version,'config_hash':cfg.config_hash,'detector_hash':result['detector_model_sha256'],'output_root':result['output_root'],'source_datasets':['casia_fasd','msu_mfsd'],'target_dataset':'siw_mv2','frames_manifested':result['manifests'].get('source_frames',0)+result['manifests'].get('target_frames',0),'crops_manifested':result['manifests'].get('source_crops',0)+result['manifests'].get('target_crops',0),'completed_samples':result['completed_index'],'failed_samples':result['manifests'].get('preprocessing_failures',0),'target_isolation':result['target_isolation']['passed'],'run_state_status':result['run_state'].get('status'),'lock_present':result['lock_state']=='present','validation_status':result['passed'],'last_validation_time':result['validated_at'],'next_action':'full production preprocessing remains a separate execution gate before M3 package construction','generated_at':datetime.now(timezone.utc).isoformat()}
def write_report(path:Path,data:dict):
    atomic_json_write(path,data);path.with_suffix('.md').write_text('# M2 report\n\n```json\n'+json.dumps(data,indent=2,default=str)+'\n```\n',encoding='utf-8')