from __future__ import annotations
import json, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from prism_fas.utils.core import atomic_json_write
from .config import INFERENCE_SPLIT, LoaderConfig, TRAINING_SPLIT, VALIDATION_SPLIT
from .collate import collate_source_batch, collate_target_batch
from .contracts import CanonicalTargetSample, FORBIDDEN_TARGET_FIELDS
from .loose_dataset import CanonicalPackageDataset
from .package_index import open_package, package_summary
from .sampler import BalancedDomainClassBatchSampler, batch_fingerprint
from .shard_dataset import CanonicalShardDataset

MODES={TRAINING_SPLIT:"training",VALIDATION_SPLIT:"validation",INFERENCE_SPLIT:"inference"}
def audit_loader(package_root:Path,config:LoaderConfig,*,parity_samples:int=3,progress=None)->dict:
    """Full loose and shard scans over every split of a real package."""
    root=Path(package_root); started=time.time()
    report={"package":package_summary(root),"generated_at":datetime.now(timezone.utc).isoformat(),
            "label_mapping":dict(config.label_mapping),"loose":{},"shard":{},"parity":{},"errors":[]}
    loose_ids={};shard_ids={}
    for split,mode in MODES.items():
        dataset=CanonicalPackageDataset(root,split,config,mode=mode)
        seen=[];identity=0;target_labels=0
        for position in range(len(dataset)):
            sample=dataset[position]; seen.append(sample.sample_id)
            if getattr(sample,"identity_available",False): identity+=1
            if isinstance(sample,CanonicalTargetSample) and (FORBIDDEN_TARGET_FIELDS & set(vars(sample))): target_labels+=1
            if progress and position%500==0: progress({"stage":"loose","split":split,"done":position,"total":len(dataset)})
        loose_ids[split]=set(seen)
        report["loose"][split]={"count":len(seen),"unique":len(set(seen)),"identity_available":identity,
                                "forbidden_target_fields":target_labels}
    for split,mode in MODES.items():
        dataset=CanonicalShardDataset(root,split,config,mode=mode)
        seen=[];identity=0
        for position,sample in enumerate(dataset):
            seen.append(sample.sample_id)
            if getattr(sample,"identity_available",False): identity+=1
            if progress and position%500==0: progress({"stage":"shard","split":split,"done":position,"total":len(dataset)})
        shard_ids[split]=set(seen)
        report["shard"][split]={"count":len(seen),"unique":len(set(seen)),"identity_available":identity,
                                "declared_rows":len(dataset),"shards":len(dataset.shards)}
        report["shard"][split]["ids_match_loose"]=shard_ids[split]==loose_ids[split]
    report["loose"]["total"]=sum(report["loose"][s]["count"] for s in MODES)
    report["shard"]["total"]=sum(report["shard"][s]["count"] for s in MODES)
    for split,mode in MODES.items():
        loose=CanonicalPackageDataset(root,split,config,mode=mode)
        shard_lookup={}
        for sample in CanonicalShardDataset(root,split,config,mode=mode):
            shard_lookup[sample.sample_id]=sample
            if len(shard_lookup)>=parity_samples*40: break
        checked=[];mismatch=[]
        for sample_id in list(loose.sample_ids)[:parity_samples*40]:
            if sample_id not in shard_lookup or len(checked)>=parity_samples: continue
            a=loose[loose.index_of(sample_id)]; b=shard_lookup[sample_id]
            same=(np.array_equal(a.geometry.parsing_labels,b.geometry.parsing_labels)
                  and np.allclose(a.image,b.image,atol=1e-6) and np.allclose(a.geometry.bbox,b.geometry.bbox,atol=1e-6)
                  and np.allclose(a.geometry.landmarks,b.geometry.landmarks,atol=1e-6)
                  and np.allclose(a.geometry.pose_ypr,b.geometry.pose_ypr,atol=1e-6)
                  and np.array_equal(a.geometry.visibility,b.geometry.visibility)
                  and np.allclose(a.geometry.quality_vector,b.geometry.quality_vector,atol=1e-6)
                  and a.dataset==b.dataset and a.project_split==b.project_split
                  and getattr(a,"label",None)==getattr(b,"label",None)
                  and a.identity_available==b.identity_available and a.crop_sha256==b.crop_sha256)
            (checked if same else mismatch).append(sample_id)
        report["parity"][split]={"checked":len(checked),"mismatched":mismatch}
    report["elapsed_seconds"]=round(time.time()-started,1)
    return report
def audit_sampler(package_root:Path,config:LoaderConfig,*,epochs:int=2,batches:int=50)->dict:
    """Real balanced-sampler audit over source_train."""
    root=Path(package_root); index=open_package(root,TRAINING_SPLIT,config,mode="training")
    dataset=CanonicalPackageDataset(root,TRAINING_SPLIT,config,mode="training",index=index)
    sampler=BalancedDomainClassBatchSampler(index,config)
    report={"package":package_summary(root),"generated_at":datetime.now(timezone.utc).isoformat(),
            "sampler_state":sampler.state(),"epochs":[],"determinism":{},"errors":[]}
    fingerprints={}
    for epoch in range(epochs):
        sampler.set_epoch(epoch); composition=Counter(); duplicates=0; record_collisions=0
        batch_ids=[];records=Counter();drawn=[]
        for position,batch in enumerate(sampler):
            if position>=batches: break
            rows=[index.rows[i] for i in batch]
            ids=[row["sample_id"] for row in rows]
            if len(set(ids))!=len(ids): duplicates+=1
            batch_records=[row["source_record_id"] for row in rows]
            if len(set(batch_records))!=len(batch_records): record_collisions+=1
            for row in rows:
                composition[(row["dataset"],row["label_live_spoof"])]+=1
                records[row["source_record_id"]]+=1
            if any(row["project_split"]!=TRAINING_SPLIT for row in rows):
                report["errors"].append("non-training sample in a training batch")
            batch_ids.append(ids); drawn.extend(ids)
        pools=Counter(f"{a}/{b}" for a,b in composition)
        per_batch=[dict(Counter(f"{index.rows[i]['dataset']}/{index.rows[i]['label_live_spoof']}" for i in batch)) for batch in []]
        fingerprints[epoch]=batch_fingerprint(batch_ids)
        report["epochs"].append({"epoch":epoch,"batches":len(batch_ids),
            "composition_per_pool":{f"{a}/{b}":count for (a,b),count in sorted(composition.items())},
            "draws":len(drawn),"unique_samples":len(set(drawn)),"reuse":len(drawn)-len(set(drawn)),
            "distinct_source_records":len(records),"batches_with_duplicate_sample_id":duplicates,
            "batches_with_repeated_record":record_collisions,"fingerprint":fingerprints[epoch],
            "pool_stats":[{"pool":f"{s.key[0]}/{s.key[1]}","size":s.size,"draws":s.draws,"unique":s.unique,"reuse":s.reuse}
                          for s in sampler.last_epoch_stats]})
    sampler.set_epoch(0); replay=[]
    for position,batch in enumerate(sampler):
        if position>=batches: break
        replay.append([index.rows[i]["sample_id"] for i in batch])
    report["determinism"]={"epoch0_fingerprint":fingerprints.get(0),"replay_fingerprint":batch_fingerprint(replay),
        "replay_matches":batch_fingerprint(replay)==fingerprints.get(0),
        "epoch0_differs_from_epoch1":fingerprints.get(0)!=fingerprints.get(1)}
    sample=dataset[0]
    report["batch_contract"]={"source_keys":sorted(collate_source_batch([dataset[i] for i in range(4)]).keys())}
    return report
def write_report(path:Path,data:dict,title:str)->Path:
    atomic_json_write(path,data)
    lines=[f"# {title}","",f"- Package: `{data['package']['package_id']}`",
           f"- Content identity: `{data['package']['content_identity_sha256']}`",
           f"- Parent: `{data['package'].get('parent_package_id')}`","","```json",json.dumps(data,indent=1)[:20000],"```",""]
    path.with_suffix(".md").write_text("\n".join(lines),encoding="utf-8")
    return path
