from __future__ import annotations
import json, time
from pathlib import Path
import typer, yaml
from prism_fas.config.models import DatasetDefinition, load_paths, resolve_config
from prism_fas.data.audit.audit import audit_dataset, write_audits
from prism_fas.utils.core import atomic_json_write
from prism_fas.data.preprocess_m2 import load_m2_config, SCRFDDetector
from prism_fas.data.m2_runner import run as run_m2a, run_preprocessing
from prism_fas.data.manifests.repository import ManifestRepository
from prism_fas.data.package import M3APackageConfig, build_package, build_priors, finalize_lock, load_m2_samples, load_package_config, validate_package, validate_source_m2_hashes
from prism_fas.data.package.audit import build_audit_report, build_model_prior_report
from prism_fas.data.package.m3b import build_m3b_package
from prism_fas.data.loader import CanonicalPackageDataset, CanonicalShardDataset, load_loader_config, package_summary
from prism_fas.data.loader.audit import audit_loader, audit_sampler, write_report as write_loader_report
from prism_fas.train.config import load_b00_config
from prism_fas.train.b00_pipeline import run_b00_calibrate, run_b00_predict_target, run_b00_report, run_b00_smoke, run_b00_train
from prism_fas.data.manifests.migrate_m2a import migrate_m2a
from prism_fas.data.manifests.resume import build_completed_index
from prism_fas.utils.core import sha256_file
from prism_fas.data.m2_validation import validate_m2, validate_full_profile, status_m2, write_report
from prism_fas.data.run_profiles import load_profiles, profile_root
from prism_fas.data.adapters import adapter_for
from prism_fas.data.run_context import PreprocessingRunContext, M2OutputLayout
from prism_fas.utils.core import sha256_file

def build_preprocessing_run_context(paths, cfg, profile, dataset, run_id, *, all_records=False, limit_records=None, limit_samples=None, resume=False, dry_run=False, partial=False, root=None):
    root=root or profile_root(paths.work_root,cfg.preprocessing_version,cfg.config_hash,profile); layout=M2OutputLayout.from_root(root); role='target' if dataset=='siw_mv2' else 'source'
    return PreprocessingRunContext(project_root=paths.project_root,work_root=paths.work_root,run_profile=profile.name,output_namespace=profile.output_namespace,output_root=layout.output_root,crops_root=layout.crops_root,frames_root=layout.frames_root,manifests_root=layout.manifests_root,state_root=layout.state_root,reports_root=layout.reports_root,logs_root=layout.logs_root,run_id=run_id or f'{profile.name}-{dataset}',dataset=dataset,dataset_role=role,preprocessing_version=cfg.preprocessing_version,preprocessing_config_hash=cfg.config_hash,detector_model_path=cfg.scrfd_model_path,detector_model_sha256=sha256_file(cfg.scrfd_model_path),detector_input_size=cfg.scrfd_input_size,detector_threshold=cfg.detection_threshold,all_records=all_records,record_limit=limit_records,sample_limit=limit_samples,resume=resume,dry_run=dry_run,partial_full_profile=partial,command='prism data preprocess run')
class ProgressRecords(list):
    """Canonical records that log run progress as the runner consumes them.

    Only aggregate, non-private progress is emitted: never a record path,
    filename or canonical metadata field.
    """
    def __init__(self,records,dataset,repository_holder,every=25):
        super().__init__(records);self.dataset=dataset;self.holder=repository_holder;self.every=every;self.started=time.monotonic()
    def __iter__(self):
        total=len(self)
        for position,record in enumerate(super().__iter__(),1):
            if position==1 or position%self.every==0 or position==total:
                repo=self.holder.get('repository');counts=repo.counts() if repo is not None else {}
                typer.echo(json.dumps({'progress':{'dataset':self.dataset,'record':position,'records_total':total,'elapsed_seconds':round(time.monotonic()-self.started,1),'manifest_counts':counts}}))
            yield record
app = typer.Typer(help="PRISM-FAS-B local data-factory CLI", no_args_is_help=True)
config_app = typer.Typer(help="Validate and resolve configuration")
app.add_typer(config_app, name="config")
@config_app.command("validate")
def validate(config: Path = typer.Option(..., exists=True, dir_okay=False)) -> None:
    paths=load_paths(config); typer.echo(f"Valid configuration: {paths.project_root}")
@config_app.command("resolve")
def resolve(config: Path = typer.Option(..., exists=True, dir_okay=False), output: Path = typer.Option(Path("resolved_config.json"))) -> None:
    atomic_json_write(output, resolve_config(config).model_dump(mode="json")); typer.echo(str(output))
@app.command("status")
def status() -> None: typer.echo("PRISM-FAS-B: M0/M1 only; M2 not implemented")
@app.command("data")
def data() -> None: """Data subcommands."""
data_app=typer.Typer(); app.add_typer(data_app, name="data")
preprocess_app=typer.Typer(help="M2 deterministic preprocessing")
data_app.add_typer(preprocess_app,name="preprocess")
@preprocess_app.command("inspect")
@preprocess_app.command("sample")
@preprocess_app.command("detect")
@preprocess_app.command("run")
def preprocess_run(dataset:str=typer.Option(...), config:Path=typer.Option(...,exists=True), preprocess_config:Path=typer.Option(...,exists=True), limit_records:int|None=typer.Option(None), resume:bool=False,dry_run:bool=False,force:bool=False,workers:int|None=None,device:str|None=None,run_profile:str=typer.Option('small_acceptance'),all_records:bool=False,confirm_full_run:bool=False,run_id:str|None=None,limit_samples:int|None=None,allow_partial_full_profile:bool=False,output_dir_name:str|None=typer.Option(None,'--output-dir-name',help='alternate output directory under the config-hash root (smoke runs)'))->None:
    if dataset not in {"casia_fasd","msu_mfsd","siw_mv2","all"}: raise typer.BadParameter("unknown dataset")
    if dataset == "all": raise typer.BadParameter("one dataset per run")
    paths=load_paths(config);cfg=load_m2_config(preprocess_config);profiles=load_profiles(Path(__file__).parents[3]/'configs'/'data'/'m2_run_profiles.yaml')
    if run_profile not in profiles: raise typer.BadParameter('unknown run profile')
    profile=profiles[run_profile];root=profile_root(paths.work_root,cfg.preprocessing_version,cfg.config_hash,profile)
    if run_profile=='full_preprocessing':
        if all_records and limit_records is not None: raise typer.BadParameter('--all-records conflicts with --limit-records')
        if not all_records and not (allow_partial_full_profile and limit_records is not None): raise typer.BadParameter('full profile requires --all-records --confirm-full-run')
        if not confirm_full_run: raise typer.BadParameter('full profile requires --confirm-full-run')
        definition=DatasetDefinition.model_validate(yaml.safe_load((Path(__file__).parents[3]/'configs'/'data'/f'{dataset}.yaml').read_text()))
        records=adapter_for(definition,getattr(paths.raw_datasets,dataset)).records();count=len(records);selected=count if all_records else limit_records
        if output_dir_name: root=profile_root(paths.work_root,cfg.preprocessing_version,cfg.config_hash,profile,paths.work_root/'m2'/cfg.preprocessing_version/cfg.config_hash/output_dir_name)
        summary={'run_profile':run_profile,'output_root':str(root),'dataset':dataset,'canonical_records_total':count,'canonical_records_selected':selected,'estimated_selected_samples':selected*cfg.frames_per_video,'all_records':all_records,'confirm_full_run':confirm_full_run,'execution':'dry_run' if dry_run else 'context_aware_runner'}
        if dry_run: typer.echo(json.dumps(summary));return
        typer.echo(json.dumps(summary))
        context=build_preprocessing_run_context(paths,cfg,profile,dataset,run_id,all_records=all_records,limit_records=limit_records,limit_samples=limit_samples,resume=resume,dry_run=False,partial=not all_records,root=root)
        holder={}
        def repository_factory(manifests_root,metadata):
            holder['repository']=ManifestRepository(manifests_root,metadata);return holder['repository']
        detector=SCRFDDetector(cfg.scrfd_model_path,cfg.scrfd_input_size,cfg.detector.get('provider','CPUExecutionProvider'))
        result=run_preprocessing(context,ProgressRecords(records if all_records else records[:limit_records],dataset,holder),detector=detector,repository_factory=repository_factory)
        typer.echo(result.model_dump_json());return
    typer.echo(json.dumps(run_m2a(dataset,config,preprocess_config,limit_records or profile.default_record_limit,dry_run,resume,force),default=str))
@preprocess_app.command("build-completed-index")
def preprocess_build_completed_index(config:Path=typer.Option(...,exists=True),preprocess_config:Path=typer.Option(...,exists=True))->None:
    paths=load_paths(config); cfg=load_m2_config(preprocess_config); root=paths.work_root/'m2'/cfg.preprocessing_version/cfg.config_hash
    typer.echo(json.dumps(build_completed_index(root/'manifests',root,cfg,sha256_file(cfg.scrfd_model_path),paths.reports_root/'m2b1b'),default=str))
@preprocess_app.command("validate")
def preprocess_validate(config:Path=typer.Option(...,exists=True),preprocess_config:Path=typer.Option(...,exists=True),output_root:Path|None=None,report_json:Path|None=None,report_md:Path|None=None,strict:bool=True,validation_profile:str=typer.Option('small_acceptance','--validation-profile',help='small_acceptance or full_preprocessing'))->None:
    if validation_profile not in {'small_acceptance','full_preprocessing'}: raise typer.BadParameter('unknown validation profile')
    if validation_profile=='full_preprocessing' and output_root is None: raise typer.BadParameter('full-profile validation requires --output-root')
    paths=load_paths(config); cfg=load_m2_config(preprocess_config)
    data=validate_full_profile(paths,cfg,output_root) if validation_profile=='full_preprocessing' else validate_m2(paths,cfg,output_root); target=report_json or paths.reports_root/'m2b2'/'validation_report.json';write_report(target,data)
    if report_md: report_md.write_text((target.with_suffix('.md')).read_text(encoding='utf-8'),encoding='utf-8')
    typer.echo(json.dumps({'passed':data['passed'],'errors':len(data['errors']),'report':str(target)}))
    if not data['passed']: raise typer.Exit(1)
@preprocess_app.command("status")
def preprocess_status(config:Path=typer.Option(...,exists=True),preprocess_config:Path=typer.Option(...,exists=True),output_root:Path|None=None,json_output:bool=typer.Option(False,'--json'),report_json:Path|None=None,report_md:Path|None=None,validation_profile:str=typer.Option('small_acceptance','--validation-profile'))->None:
    paths=load_paths(config); data=status_m2(paths,load_m2_config(preprocess_config),output_root,validation_profile); target=report_json or paths.reports_root/'m2b2'/'status_report.json';write_report(target,data)
    if report_md: report_md.write_text((target.with_suffix('.md')).read_text(encoding='utf-8'),encoding='utf-8')
    typer.echo(json.dumps(data) if json_output else f"M2 {data['milestone_status']}: completed={data['completed_samples']}, failures={data['failed_samples']}")
@preprocess_app.command("migrate-m2a")
def preprocess_migrate_m2a(config: Path=typer.Option(...,exists=True), preprocess_config: Path=typer.Option(...,exists=True), m2a_root: Path=typer.Option(...), output_root: Path=typer.Option(...), force: bool=False)->None:
    typer.echo(json.dumps(migrate_m2a(config,preprocess_config,m2a_root,output_root,force),default=str))
package_app=typer.Typer(help="M3A package foundation")
data_app.add_typer(package_app,name="package")
priors_app=typer.Typer(help="M3A deterministic priors")
data_app.add_typer(priors_app,name="priors")
def _package_config(path:Path|None)->'M3APackageConfig':
    return load_package_config(path or Path(__file__).parents[3]/'configs'/'data'/'package_m3a.yaml')
def _resolve_package_root(paths,package_root:Path|None,name:str|None)->Path:
    return Path(package_root) if package_root else (paths.processed_root/(name or 'prism_data_v1_m3a'))
@priors_app.command("build")
def priors_build(config:Path=typer.Option(...,exists=True),input_root:Path=typer.Option(...,exists=True),package_root:Path|None=typer.Option(None),package_name:str|None=typer.Option(None),package_config:Path|None=typer.Option(None),resume:bool=typer.Option(True,'--resume/--no-resume'))->None:
    paths=load_paths(config); cfg=_package_config(package_config); root=_resolve_package_root(paths,package_root,package_name)
    samples=load_m2_samples(input_root)
    rows,stats=build_priors(samples,input_root=input_root,package_root=root,config=cfg,resume=resume,
                            progress=lambda done,total: typer.echo(json.dumps({"progress":{"stage":"priors","done":done,"total":total}})))
    typer.echo(json.dumps({"package_root":str(root),"samples":stats.samples,"priors_built":stats.priors_built,"priors_reused":stats.priors_reused,"priors_total":len(rows)}))
@package_app.command("build")
def package_build(config:Path=typer.Option(...,exists=True),input_root:Path=typer.Option(...,exists=True),package_root:Path|None=typer.Option(None),package_name:str|None=typer.Option(None),package_config:Path|None=typer.Option(None),resume:bool=typer.Option(True,'--resume/--no-resume'),limit_per_dataset:int|None=typer.Option(None,help='smoke builds only'))->None:
    paths=load_paths(config); cfg=_package_config(package_config); root=_resolve_package_root(paths,package_root,package_name)
    result=build_package(input_root,root,cfg,resume=resume,limit_per_dataset=limit_per_dataset,
                         progress=lambda stage,done,total: typer.echo(json.dumps({"progress":{"stage":stage,"done":done,"total":total}})))
    stats=result["stats"]
    pre=validate_package(root,require_validated_status=False)
    lock=finalize_lock(root,pre) if pre["passed"] else result["lock"]
    report=validate_package(root) if pre["passed"] else pre
    audit=build_audit_report(root,lock=lock,resume={"priors_built":stats.priors_built,"priors_reused":stats.priors_reused},
                             m2_failures=result.get("m2_failures",[]))
    typer.echo(json.dumps({"package_root":str(root),"status":lock["status"],"samples":stats.samples,
        "priors_built":stats.priors_built,"priors_reused":stats.priors_reused,"images":stats.images_linked,
        "shards":stats.shards,"per_dataset":stats.per_dataset,"per_split":stats.per_split,
        "validation_passed":report["passed"],"validation_errors":len(report["errors"]),
        "content_identity_sha256":lock.get("content_identity_sha256"),"audit_report":str(root/'audit'/'data_report.json')}))
    if not report["passed"]: raise typer.Exit(1)
@priors_app.command("model-build")
def priors_model_build(input_package:Path=typer.Option(...,exists=True),output_package:Path=typer.Option(...),model_config:Path=typer.Option(...,exists=True),config:Path|None=typer.Option(None,exists=True),weight_root:Path|None=typer.Option(None),resume:bool=typer.Option(True,'--resume/--no-resume'),limit_samples:int|None=typer.Option(None),split:str|None=typer.Option(None),device:str|None=typer.Option(None),batch_size:int|None=typer.Option(None),dry_run:bool=typer.Option(False,'--dry-run'))->None:
    root=weight_root or (load_paths(config).model_cache if config else Path(os.environ.get('PRISM_MODEL_CACHE','model_cache')))
    result=build_m3b_package(input_package,output_package,model_config,weight_root=root,resume=resume,limit_samples=limit_samples,
        split=split,device=device,batch_size=batch_size,dry_run=dry_run,progress=lambda payload: typer.echo(json.dumps({"progress":payload})))
    if dry_run: typer.echo(json.dumps(result)); return
    pre=validate_package(output_package,require_validated_status=False,parent_package=input_package)
    lock=finalize_lock(output_package,pre) if pre["passed"] else result["lock"]
    report=validate_package(output_package,parent_package=input_package) if pre["passed"] else pre
    audit=build_model_prior_report(output_package,lock=lock,counts=result["counts"],failures=result["failures"],device=result["device"])
    typer.echo(json.dumps({"output_package":str(output_package),"status":lock["status"],"samples":lock["total_samples"],
        "prior_counts":lock["prior_counts"],"per_split":lock["per_split_counts"],"failures":lock["model_prior_failures"],
        "shards":len(lock["shards"]),"validation_passed":report["passed"],"validation_errors":len(report["errors"]),
        "content_identity_sha256":lock["content_identity_sha256"],"device":result["device"],
        "reused":result["counts"]["reused"],"rebuilt":result["counts"]["rebuilt"],"audit_report":str(audit)}))
    if not report["passed"]: raise typer.Exit(1)
@package_app.command("validate")
def package_validate(package_root:Path=typer.Option(...,exists=True),input_root:Path|None=typer.Option(None),parent_package:Path|None=typer.Option(None),report_json:Path|None=typer.Option(None))->None:
    report=validate_package(package_root,parent_package=parent_package)
    if input_root is not None:
        report["source_m2_hashes_match"]=validate_source_m2_hashes(package_root,input_root)
        if not report["source_m2_hashes_match"]:
            report["passed"]=False; report["errors"].append({"check_id":"lock.source_m2","passed":False,"description":"frozen M2 input hashes match the lock"})
    if report_json: atomic_json_write(report_json,report)
    typer.echo(json.dumps({"passed":report["passed"],"errors":len(report["errors"]),"checks":len(report["checks"]),
        "counts":report["counts"],"package_id":report["package_id"],"target_isolation":report["target_isolation"]["passed"]}))
    if not report["passed"]: raise typer.Exit(1)
loader_app=typer.Typer(help="M4 canonical loader")
data_app.add_typer(loader_app,name="loader")
sampler_app=typer.Typer(help="M4 balanced sampler")
data_app.add_typer(sampler_app,name="sampler")
def _loader_config(path:Path|None):
    return load_loader_config(path or Path(__file__).parents[3]/'configs'/'data'/'loader_m4.yaml')
@loader_app.command("inspect")
def loader_inspect(package_root:Path=typer.Option(...,exists=True),split:str=typer.Option('source_train'),backend:str=typer.Option('loose'),config:Path|None=typer.Option(None),limit:int=typer.Option(3))->None:
    cfg=_loader_config(config); mode={'source_train':'training','source_dev':'validation','target_test':'inference'}[split]
    dataset=(CanonicalPackageDataset if backend=='loose' else CanonicalShardDataset)(package_root,split,cfg,mode=mode)
    summary=package_summary(package_root); samples=[]
    iterator=(dataset[i] for i in range(min(limit,len(dataset)))) if backend=='loose' else dataset
    for position,sample in enumerate(iterator):
        if position>=limit: break
        entry={"sample_id":sample.sample_id,"dataset":sample.dataset,"project_split":sample.project_split,
               "image_shape":list(sample.image.shape),"image_range":[round(float(sample.image.min()),4),round(float(sample.image.max()),4)],
               "parsing_classes":int(len(set(sample.geometry.parsing_labels.flatten().tolist()))),
               "pose":[round(float(v),4) for v in sample.geometry.pose_ypr],
               "visibility_len":int(sample.geometry.visibility.shape[0]),
               "identity_available":bool(getattr(sample,"identity_available",False))}
        if hasattr(sample,"label"): entry.update(label=sample.label,class_target=sample.class_target)
        samples.append(entry)
    typer.echo(json.dumps({"package":summary,"split":split,"backend":backend,"count":len(dataset),"samples":samples}))
@loader_app.command("audit")
def loader_audit(package_root:Path=typer.Option(...,exists=True),config:Path|None=typer.Option(None),report_json:Path|None=typer.Option(None))->None:
    cfg=_loader_config(config)
    report=audit_loader(package_root,cfg,progress=lambda payload: typer.echo(json.dumps({"progress":payload})))
    target=report_json or Path('reports/m4/loader_audit.json'); write_loader_report(target,report,"M4 loader audit")
    typer.echo(json.dumps({"package_id":report["package"]["package_id"],"content_identity":report["package"]["content_identity_sha256"],
        "loose":{k:v["count"] for k,v in report["loose"].items() if isinstance(v,dict)},"loose_total":report["loose"]["total"],
        "shard":{k:v["count"] for k,v in report["shard"].items() if isinstance(v,dict)},"shard_total":report["shard"]["total"],
        "parity":{k:v["checked"] for k,v in report["parity"].items()},"errors":report["errors"],"report":str(target)}))
    if report["errors"]: raise typer.Exit(1)
@sampler_app.command("audit")
def sampler_audit(package_root:Path=typer.Option(...,exists=True),config:Path|None=typer.Option(None),epochs:int=typer.Option(2),batches:int=typer.Option(50),report_json:Path|None=typer.Option(None))->None:
    cfg=_loader_config(config); report=audit_sampler(package_root,cfg,epochs=epochs,batches=batches)
    target=report_json or Path('reports/m4/sampler_audit.json'); write_loader_report(target,report,"M4 sampler audit")
    typer.echo(json.dumps({"package_id":report["package"]["package_id"],"sampler":report["sampler_state"],
        "epochs":[{k:e[k] for k in ("epoch","batches","composition_per_pool","unique_samples","reuse","distinct_source_records","batches_with_duplicate_sample_id","batches_with_repeated_record")} for e in report["epochs"]],
        "determinism":report["determinism"],"errors":report["errors"],"report":str(target)}))
    if report["errors"]: raise typer.Exit(1)
train_app=typer.Typer(help="M5 local training")
app.add_typer(train_app,name="train")
b00_app=typer.Typer(help="B00 ConvNeXt V2 local baseline")
train_app.add_typer(b00_app,name="b00")
@b00_app.command("smoke")
def b00_smoke(package_root:Path=typer.Option(...,exists=True),config:Path=typer.Option(...,exists=True),run_id:str=typer.Option(...),device:str|None=typer.Option(None),num_workers:int=typer.Option(0),limit_steps:int=typer.Option(5),limit_dev_samples:int=typer.Option(128),limit_target_samples:int=typer.Option(64),dry_run:bool=typer.Option(False,'--dry-run'))->None:
    typer.echo(json.dumps(run_b00_smoke(package_root,load_b00_config(config),run_id,device=device,workers=num_workers,
        limit_steps=limit_steps,limit_dev_samples=limit_dev_samples,limit_target_samples=limit_target_samples,dry_run=dry_run)))
@b00_app.command("run")
def b00_run(package_root:Path=typer.Option(...,exists=True),config:Path=typer.Option(...,exists=True),run_id:str=typer.Option(...),device:str|None=typer.Option(None),num_workers:int=typer.Option(0),resume:bool=typer.Option(False,'--resume'),limit_steps:int|None=typer.Option(None),dry_run:bool=typer.Option(False,'--dry-run'))->None:
    typer.echo(json.dumps(run_b00_train(package_root,load_b00_config(config),run_id,device=device,workers=num_workers,
        resume=resume,limit_steps=limit_steps,dry_run=dry_run,
        progress=lambda payload: typer.echo(json.dumps({"progress":payload})))))
@b00_app.command("calibrate")
def b00_calibrate(run_root:Path=typer.Option(...,exists=True),package_root:Path=typer.Option(...,exists=True),config:Path=typer.Option(...,exists=True),device:str|None=typer.Option(None),num_workers:int=typer.Option(0),limit_dev_samples:int|None=typer.Option(None))->None:
    typer.echo(json.dumps(run_b00_calibrate(package_root,run_root,load_b00_config(config),device=device,workers=num_workers,limit=limit_dev_samples)))
@b00_app.command("predict-target")
def b00_predict_target(run_root:Path=typer.Option(...,exists=True),package_root:Path=typer.Option(...,exists=True),config:Path=typer.Option(...,exists=True),device:str|None=typer.Option(None),num_workers:int=typer.Option(0),limit_target_samples:int|None=typer.Option(None))->None:
    typer.echo(json.dumps(run_b00_predict_target(package_root,run_root,load_b00_config(config),device=device,workers=num_workers,limit=limit_target_samples)))
@b00_app.command("report")
def b00_report(run_root:Path=typer.Option(...,exists=True))->None:
    typer.echo(json.dumps(run_b00_report(run_root)))
recipe_app=typer.Typer(help="M7 source-only recipe schema, compiler and frozen bank")
app.add_typer(recipe_app,name="recipe")
synthesis_app=typer.Typer(help="M7 CPU physics engine and preview audit")
app.add_typer(synthesis_app,name="synthesis")
def _m7_defaults()->tuple[Path,Path,Path]:
    base=Path(__file__).parents[3]
    return base/'configs'/'recipes'/'ontology_m7.yaml',base/'configs'/'recipes'/'bank_m7.yaml',base/'configs'/'synthesis'/'physics_m7.yaml'
@recipe_app.command("build-bank")
def recipe_build_bank(ontology:Path|None=typer.Option(None,'--ontology',exists=True,dir_okay=False),config:Path|None=typer.Option(None,'--config',exists=True,dir_okay=False),output:Path=typer.Option(...,'--output'),dry_run:bool=typer.Option(False,'--dry-run'))->None:
    from prism_fas.recipes.bank import build_bank
    default_ontology,default_config,_=_m7_defaults()
    result=build_bank(output,ontology or default_ontology,config or default_config,dry_run=dry_run)
    typer.echo(json.dumps({"status":result["status"],"bank_id":result["bank_id"],"recipe_count":result["recipe_count"],
        "bank_seed":result["bank_seed"],"output_root":result["output_root"],"written":result["written"],
        "ontology_version":result["ontology_version"],"ontology_sha256":result["ontology_sha256"],
        "bank_content_identity_sha256":result["bank_content_identity_sha256"],
        "compiler_version":result["compiler_version"],"conditioning_dimension":result["conditioning_dimension"],
        "external_llm_invoked":result["external_llm_invoked"],
        "coverage_required_all_met":result["audit"]["coverage"]["required_all_met"],
        "diversity_passed":result["audit"]["diversity"]["passed"]}))
@recipe_app.command("validate-bank")
def recipe_validate_bank(bank:Path=typer.Option(...,'--bank',exists=True,file_okay=False),report_json:Path|None=typer.Option(None))->None:
    from prism_fas.recipes.bank import validate_bank
    report=validate_bank(bank)
    if report_json: atomic_json_write(report_json,report)
    typer.echo(json.dumps({"passed":report["passed"],"errors":report["errors"],"bank_id":report["bank_id"],
        "recipe_count":report["recipe_count"],"bank_content_identity_sha256":report["bank_content_identity_sha256"],
        "ontology_version":report["ontology_version"],"external_llm_invoked":report["external_llm_invoked"],
        "validation_issues":report["validation"]["issue_count"],
        "coverage_required_all_met":report["coverage"]["required_all_met"],
        "diversity":{k:report["diversity"][k] for k in ("method","max_cosine") if k in report["diversity"]} or report["diversity"]["method"]}))
    if not report["passed"]: raise typer.Exit(1)
@recipe_app.command("compile-bank")
def recipe_compile_bank(bank:Path=typer.Option(...,'--bank',exists=True,file_okay=False),report_json:Path|None=typer.Option(None),limit:int|None=typer.Option(None,'--limit'))->None:
    from prism_fas.recipes.bank import compiled_graphs
    result=compiled_graphs(bank); graphs=result["graphs"][:limit] if limit else result["graphs"]
    summary=result["summary"]
    if report_json: atomic_json_write(report_json,{"bank_id":result["bank_id"],**summary})
    typer.echo(json.dumps({"bank_id":result["bank_id"],"compiled":len(graphs),"unique_graph_hashes":summary["unique_graph_hashes"],
        "duplicate_graph_hashes":summary["duplicate_graph_hashes"],"compiler_version":summary["compiler_version"],
        "conditioning_version":summary["conditioning_version"],"conditioning_dimension":summary["conditioning_dimension"],
        "operator_application_order":summary["operator_application_order"],
        "graph_hash_set_sha256":summary["graph_hash_set_sha256"]}))
@synthesis_app.command("physics-audit")
def synthesis_physics_audit(package_root:Path=typer.Option(...,'--package-root',exists=True,file_okay=False),bank:Path=typer.Option(...,'--bank',exists=True,file_okay=False),config:Path|None=typer.Option(None,'--config',exists=True,dir_okay=False),output:Path=typer.Option(Path('reports/m7'),'--output'),limit:int|None=typer.Option(None,'--limit',help='development smoke only; a limited run does not satisfy the M7 contract'),dry_run:bool=typer.Option(False,'--dry-run'))->None:
    from prism_fas.synthesis.audit import run_audit
    _,_,default_config=_m7_defaults()
    result=run_audit(package_root,bank,config or default_config,output,limit=limit,dry_run=dry_run,
        progress=lambda payload: typer.echo(json.dumps({"progress":payload})))
    if result["status"]=="dry_run":
        typer.echo(json.dumps({"status":"dry_run","plan":result["plan"],"selected_samples":result["selected_samples"],
            "planned_pairs":result["planned_pairs"],"bank_validation_passed":result["bank_validation"]["passed"],
            "compiled":result["compile"]["compiled"],"written":[]}));return
    typer.echo(json.dumps({"status":result["status"],"preview_rows":result["preview_rows"],
        "physics_passed":result["physics"]["passed"],"outside_mask_max_abs_error":result["physics"]["outside_mask_max_abs_error"],
        "operators_exercised":result["physics"]["operators_exercised"],"regions_exercised":result["physics"]["regions_exercised"],
        "input_samples_per_dataset":result["physics"]["input_samples_per_dataset"],
        "source_dev_inputs":result["physics"]["source_dev_inputs"],"target_inputs":result["physics"]["target_inputs"],
        "determinism_passed":result["determinism"]["passed"],"determinism_mismatches":result["determinism"]["mismatch_count"],
        "source_isolation_passed":result["source_isolation"]["passed"],"written":result["written"]}))
    if not (result["physics"]["passed"] and result["determinism"]["passed"] and result["source_isolation"]["passed"]):
        raise typer.Exit(1)
@data_app.command("audit")
def audit(dataset: str=typer.Option(..., help="casia_fasd, msu_mfsd, siw_mv2, or all"), config: Path=typer.Option(..., exists=True, dir_okay=False)) -> None:
    paths=load_paths(config); base=Path(__file__).parents[3] / "configs" / "data"; names=[dataset] if dataset != "all" else ["casia_fasd","msu_mfsd","siw_mv2"]
    invalid=set(names)-{"casia_fasd","msu_mfsd","siw_mv2"}
    if invalid: raise typer.BadParameter(f"unknown datasets: {sorted(invalid)}")
    reports=[]
    for name in names:
        with (base / f"{name}.yaml").open(encoding="utf-8") as handle: definition=DatasetDefinition.model_validate(yaml.safe_load(handle))
        reports.append(audit_dataset(definition, getattr(paths.raw_datasets, name)))
    target=write_audits(paths.reports_root, reports); typer.echo(str(target))
if __name__ == "__main__": app()
