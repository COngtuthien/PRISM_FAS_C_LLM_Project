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
synthesis_app=typer.Typer(help="M7 CPU physics engine and preview audit; M8 synthetic bank generation, validation and export")
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
# --- M8 identity calibration v2 -----------------------------------------------
@synthesis_app.command("identity-audit")
def synthesis_identity_audit(package_root:Path|None=typer.Option(None,'--package-root'),
        output:Path=typer.Option(Path('reports/m8/identity_structure_v2.json'),'--output'),
        dry_run:bool=typer.Option(False,'--dry-run'))->None:
    """Audit the source_train live identity structure before any v2 threshold logic exists."""
    from prism_fas.synthesis.identity_calibration import audit_identity_structure
    report=audit_identity_structure(package_root or _m8_defaults()["package"])
    if not dry_run: atomic_json_write(output,report)
    typer.echo(json.dumps({"supports_cross_record_identity_calibration":report["supports_cross_record_identity_calibration"],
        "requirements":report["requirements"],"live_count":report["live_count"],
        "identity_count":report["identity_count"],"identity_count_by_dataset":report["identity_count_by_dataset"],
        "identities_with_at_least_two_source_records":report["identities_with_at_least_two_source_records"],
        "potential_genuine_pairs":report["potential_genuine_pairs"],
        "potential_impostor_pairs_by_dataset":report["potential_impostor_pairs_by_dataset"],
        "written":[] if dry_run else [str(output.name)]}))
    if not report["supports_cross_record_identity_calibration"]: raise typer.Exit(1)
@synthesis_app.command("identity-calibrate-v2")
def synthesis_identity_calibrate_v2(weight_root:Path|None=typer.Option(None,'--weight-root'),
        package_root:Path|None=typer.Option(None,'--package-root'),
        config:Path|None=typer.Option(None,'--config'),v1_calibration:Path|None=typer.Option(None,'--v1-calibration'),
        output:Path=typer.Option(Path('reports/m8'),'--output'),device:str=typer.Option('cpu','--device'),
        repeat:int=typer.Option(2,'--repeat',help='the protocol requires two agreeing runs'),
        dry_run:bool=typer.Option(False,'--dry-run'))->None:
    """Fit tau_id_v2 from real source_train cross-observation pairs. Never reads a candidate."""
    from prism_fas.synthesis.identity_calibration import (IdentityBackend,build_genuine_pairs,build_impostor_pairs,
        calibrate_identity,compare_calibrations,live_rows,load_identity_config,write_calibration_artifacts)
    from prism_fas.synthesis.synthetic_bank import FrozenCalibration
    d=_m8_defaults()
    package=package_root or d["package"]
    payload=load_identity_config(config or (Path(__file__).parents[3]/'configs'/'synthesis'/'quality_gate_m8_v2.yaml'))
    block=payload["identity_calibration"]
    if dry_run:
        rows=live_rows(package)
        genuine=build_genuine_pairs(rows,seed=block["seed"],maximum_per_identity=block["maximum_genuine_pairs_per_identity"])
        impostor=build_impostor_pairs(rows,seed=block["seed"],maximum_total=block["maximum_impostor_pairs_total"])
        typer.echo(json.dumps({"status":"dry_run","live_samples":len(rows),"genuine_pairs":len(genuine),
            "impostor_pairs":len(impostor),"threshold_rule":"tau_id_v2 = max(tau_genuine, tau_impostor)",
            "output_root":str(output),"written":[]}));return
    if weight_root is None: raise typer.BadParameter("--weight-root is required to embed with the pinned AdaFace")
    v1=FrozenCalibration.load(v1_calibration or d["calibration"])
    backends=IdentityBackend(weight_root,device=device)
    package_identity=json.loads((Path(package)/'PACKAGE_LOCK.json').read_text(encoding='utf-8'))["content_identity_sha256"]
    runs=[calibrate_identity(package,payload,backends,v1_thresholds=v1.thresholds,
        progress=lambda item: typer.echo(json.dumps({"progress":item}))) for _ in range(max(1,repeat))]
    comparison=compare_calibrations(runs[0],runs[-1]) if len(runs)>1 else {"identical":True,"mismatch_count":0}
    if not comparison["identical"]:
        typer.echo(json.dumps({"passed":False,"determinism":comparison,"written":[]}));raise typer.Exit(1)
    written=write_calibration_artifacts(output,runs[0],package_identity=package_identity,config=payload)
    summary=written["calibration"]
    typer.echo(json.dumps({"passed":True,"tau_genuine":summary["tau_genuine"],"tau_impostor":summary["tau_impostor"],
        "tau_id_v2":summary["tau_id_v2"],"threshold_rule":summary["threshold_rule"],
        "distributions_are_well_separated":summary["distributions_are_well_separated"],
        "genuine_pairs":summary["populations"]["genuine_pairs"],"impostor_pairs":summary["populations"]["impostor_pairs"],
        "genuine_acceptance_rate_at_tau":summary["genuine_acceptance_rate_at_tau"],
        "impostor_false_match_rate_at_tau":summary["impostor_false_match_rate_at_tau"],
        "calibration_content_identity_sha256":written["lock"]["calibration_content_identity_sha256"],
        "determinism_mismatches":comparison["mismatch_count"],"written":written["written"]}))
@synthesis_app.command("validate-identity-calibration-v2")
def synthesis_validate_identity_calibration_v2(root:Path=typer.Option(Path('reports/m8'),'--root'),
        package_root:Path|None=typer.Option(None,'--package-root'),
        config:Path|None=typer.Option(None,'--config'))->None:
    """Re-derive the v2 pair plans and the lock identity from the manifest alone."""
    from prism_fas.synthesis.identity_calibration import (GENUINE_FIELDS,IMPOSTOR_FIELDS,build_genuine_pairs,
        build_impostor_pairs,build_lock,live_rows,load_identity_config,pairs_digest)
    d=_m8_defaults()
    package=package_root or d["package"]
    payload=load_identity_config(config or (Path(__file__).parents[3]/'configs'/'synthesis'/'quality_gate_m8_v2.yaml'))
    block=payload["identity_calibration"]
    lock=json.loads((Path(root)/'IDENTITY_CALIBRATION_V2_LOCK.json').read_text(encoding='utf-8'))
    calibration=json.loads((Path(root)/'quality_calibration_v2.json').read_text(encoding='utf-8'))
    rows=live_rows(package)
    genuine=build_genuine_pairs(rows,seed=block["seed"],maximum_per_identity=block["maximum_genuine_pairs_per_identity"])
    impostor=build_impostor_pairs(rows,seed=block["seed"],maximum_total=block["maximum_impostor_pairs_total"])
    package_identity=json.loads((Path(package)/'PACKAGE_LOCK.json').read_text(encoding='utf-8'))["content_identity_sha256"]
    rebuilt=build_lock({**calibration,"_genuine_pairs":genuine,"_impostor_pairs":impostor},
        package_identity=package_identity,config=payload)
    checks={"genuine_pair_plan_identity":pairs_digest(genuine,GENUINE_FIELDS)==lock["genuine_pair_plan_identity_sha256"],
        "impostor_pair_plan_identity":pairs_digest(impostor,IMPOSTOR_FIELDS)==lock["impostor_pair_plan_identity_sha256"],
        "package_identity":package_identity==lock["package_identity"],
        "config_sha256":rebuilt["config_sha256"]==lock["config_sha256"],
        "threshold_rule":lock["threshold_rule"]=="tau_id_v2 = max(tau_genuine, tau_impostor)",
        "tau_id_v2_follows_the_rule":lock["tau_id_v2"]==max(lock["tau_genuine"],lock["tau_impostor"]),
        "calibration_content_identity":rebuilt["calibration_content_identity_sha256"]==lock["calibration_content_identity_sha256"],
        "used_generated_candidates_false":lock["used_generated_candidates"] is False,
        "used_source_dev_false":lock["used_source_dev"] is False,"used_target_false":lock["used_target"] is False}
    typer.echo(json.dumps({"passed":all(checks.values()),"checks":checks,"tau_id_v2":lock["tau_id_v2"],
        "calibration_content_identity_sha256":lock["calibration_content_identity_sha256"]}))
    if not all(checks.values()): raise typer.Exit(1)
# --- M8 synthetic bank -------------------------------------------------------
def _m8_defaults()->dict[str,Path]:
    base=Path(__file__).parents[3]
    return {"package":base/'data'/'processed'/'prism_data_v1_m3b',"bank":base/'assets'/'recipe_banks'/'prism_recipe_bank_m7_v1',
        "plan":base/'reports'/'m8',"calibration":base/'reports'/'m8'/'quality_calibration.json',
        "pairs":base/'reports'/'m8'/'pairs',"bank_config":base/'configs'/'synthesis'/'synthetic_bank_m8.yaml'}
def _m8_generator(package_root:Path,bank:Path,work:Path,plan:Path,calibration:Path,weight_root:Path|None,
                  gpat_checkpoint:Path|None,device:str,with_backends:bool=True):
    from prism_fas.synthesis.m8_pipeline import build_generator
    backends=None
    if with_backends:
        from prism_fas.synthesis.quality_calibration import QualityBackends
        if weight_root is None: raise typer.BadParameter("--weight-root is required to quality-gate candidates")
        backends=QualityBackends(weight_root,device=device)
    return build_generator(package_root=package_root,bank_root=bank,work_root=work,plan_root=plan,
        calibration_path=calibration,gpat_checkpoint_path=gpat_checkpoint,backends=backends,device=device)
@synthesis_app.command("generation-pilot")
def synthesis_generation_pilot(work:Path=typer.Option(...,'--work'),package_root:Path|None=typer.Option(None,'--package-root'),
        bank:Path|None=typer.Option(None,'--bank'),plan:Path|None=typer.Option(None,'--plan'),
        calibration:Path|None=typer.Option(None,'--calibration'),weight_root:Path|None=typer.Option(None,'--weight-root'),
        gpat_checkpoint:Path|None=typer.Option(None,'--gpat-checkpoint'),device:str=typer.Option('cpu','--device'),
        count:int=typer.Option(32,'--count'),output:Path=typer.Option(Path('reports/m8'),'--output'),
        determinism:bool=typer.Option(True,'--determinism/--no-determinism'),
        dry_run:bool=typer.Option(False,'--dry-run'))->None:
    """Deterministic 32-candidate correctness pilot. Its acceptance rate never changes a threshold."""
    from prism_fas.synthesis.m8_pipeline import run_pilot,run_pilot_determinism,select_pilot_rows
    d=_m8_defaults()
    generator=_m8_generator(package_root or d["package"],bank or d["bank"],Path(work),plan or d["plan"],
        calibration or d["calibration"],weight_root,gpat_checkpoint,device,with_backends=not dry_run)
    if dry_run:
        rows=select_pilot_rows(generator.plan_rows,generator.bank,count=count)
        typer.echo(json.dumps({"status":"dry_run","planned":len(rows),"identity":generator.identity(),"written":[]}));return
    pilot=run_pilot(generator,pilot_root=Path(work)/'pilot',count=count,
        progress=lambda payload: typer.echo(json.dumps({"progress":payload})))
    atomic_json_write(Path(output)/'generation_pilot.json',pilot)
    written=['generation_pilot.json']
    audit={"passed":True,"mismatch_count":0}
    if determinism:
        audit=run_pilot_determinism(generator,first=pilot,rerun_root=Path(work)/'pilot_rerun',count=count)
        atomic_json_write(Path(output)/'generation_pilot_determinism.json',audit);written.append('generation_pilot_determinism.json')
    typer.echo(json.dumps({"status":"completed","passed":pilot["passed"],"terminal_counts":pilot["terminal_counts"],
        "coverage":pilot["coverage"],"payload_errors":pilot["payload_errors"],
        "determinism_passed":audit["passed"],"determinism_mismatches":audit["mismatch_count"],"written":written}))
    if not (pilot["passed"] and audit["passed"]): raise typer.Exit(1)
@synthesis_app.command("generate-bank")
def synthesis_generate_bank(work:Path=typer.Option(...,'--work'),package_root:Path|None=typer.Option(None,'--package-root'),
        bank:Path|None=typer.Option(None,'--bank'),plan:Path|None=typer.Option(None,'--plan'),
        calibration:Path|None=typer.Option(None,'--calibration'),pairs:Path|None=typer.Option(None,'--pairs'),
        weight_root:Path|None=typer.Option(None,'--weight-root'),gpat_checkpoint:Path|None=typer.Option(None,'--gpat-checkpoint'),
        device:str=typer.Option('cpu','--device'),resume:bool=typer.Option(True,'--resume/--no-resume'),
        interrupt_after:int|None=typer.Option(None,'--interrupt-after',help='simulated interruption for the resume audit'),
        limit:int|None=typer.Option(None,'--limit',help='development only; a limited run does not satisfy the M8 contract'),
        output:Path=typer.Option(Path('reports/m8'),'--output'),dry_run:bool=typer.Option(False,'--dry-run'))->None:
    """Generate every row of the frozen candidate plan, then assemble the bank."""
    from prism_fas.synthesis.m8_pipeline import run_full_generation
    from prism_fas.synthesis.synthetic_bank import assemble_bank
    d=_m8_defaults()
    generator=_m8_generator(package_root or d["package"],bank or d["bank"],Path(work),plan or d["plan"],
        calibration or d["calibration"],weight_root,gpat_checkpoint,device,with_backends=not dry_run)
    if dry_run:
        typer.echo(json.dumps({"status":"dry_run","planned":len(generator.plan_rows),"identity":generator.identity(),
            "resume":resume,"written":[]}));return
    if limit is not None:
        result=generator.run(rows=generator.plan_rows[:limit],resume=resume,
            progress=lambda payload: typer.echo(json.dumps({"progress":payload})))
        typer.echo(json.dumps({"status":"limited","examined":result["examined"],"reused":result["reused"],
            "rebuilt":result["rebuilt"],"contract_satisfied":False,"written":[]}));return
    audit=run_full_generation(generator,interrupt_after=interrupt_after,
        progress=lambda payload: typer.echo(json.dumps({"progress":payload})))
    assembled=assemble_bank(generator,audit["records"],pairs_root=pairs or d["pairs"])
    atomic_json_write(Path(output)/'resume_audit.json',{k:v for k,v in audit.items() if k!="records"})
    typer.echo(json.dumps({"status":assembled["status"],"bank_id":assembled["bank_id"],
        "terminal_counts":audit["terminal_counts"],"resume_passed":audit["passed"],
        "operational_minimums_passed":assembled["operational_minimums"]["passed"],
        "operational_minimum_checks":assembled["operational_minimums"]["checks"],
        "written":['resume_audit.json']}))
    if not (audit["passed"] and assembled["operational_minimums"]["passed"]): raise typer.Exit(1)
@synthesis_app.command("build-shards")
def synthesis_build_shards(bank_root:Path=typer.Option(...,'--bank-root',exists=True,file_okay=False),
        dry_run:bool=typer.Option(False,'--dry-run'))->None:
    """Rebuild the deterministic tar shards of an assembled bank and re-verify parity."""
    from prism_fas.synthesis.synthetic_bank import MANIFEST_SCHEMAS,load_manifest
    from prism_fas.synthesis.synthetic_shards import build_shards,index_digest,plan_shards,validate_shards,write_shards_index
    accepted=load_manifest(Path(bank_root)/'manifests'/'manifest.parquet',MANIFEST_SCHEMAS["manifest"])
    if dry_run:
        typer.echo(json.dumps({"status":"dry_run","accepted":len(accepted),
            "planned_shards":len(plan_shards(accepted)),"written":[]}));return
    index=build_shards(Path(bank_root),accepted); write_shards_index(Path(bank_root),index)
    report=validate_shards(Path(bank_root),index,accepted)
    typer.echo(json.dumps({"status":"created","shards":len(index),"accepted":len(accepted),
        "shards_index_sha256":index_digest(index),"parity_passed":report["passed"],
        "written":['shards_index.parquet']+[row["shard_name"] for row in index]}))
    if not report["passed"]: raise typer.Exit(1)
@synthesis_app.command("validate-bank")
def synthesis_validate_bank(bank_root:Path=typer.Option(...,'--bank-root',exists=True,file_okay=False),
        package_root:Path|None=typer.Option(None,'--package-root'),bank:Path|None=typer.Option(None,'--bank'),
        gpat_checkpoint:Path|None=typer.Option(None,'--gpat-checkpoint'),
        sample_limit:int|None=typer.Option(None,'--sample-limit',help='development only; a limited check does not satisfy the M8 contract'),
        report_json:Path|None=typer.Option(None,'--report-json'))->None:
    """Full re-derivation and validation of a built synthetic bank."""
    from prism_fas.synthesis.synthetic_validation import validate_bank,write_validation_report
    d=_m8_defaults()
    report=validate_bank(Path(bank_root),package_root=package_root or d["package"],recipe_bank_root=bank or d["bank"],
        gpat_checkpoint_path=gpat_checkpoint,sample_limit=sample_limit)
    if report_json: write_validation_report(report_json,report)
    typer.echo(json.dumps({"passed":report["passed"],"bank_id":report["bank_id"],
        "bank_content_identity_sha256":report["bank_content_identity_sha256"],"counts":report["counts"],
        "error_count":report["error_count"],"errors":report["errors"][:10],
        "operational_minimums_passed":report["operational_minimums"]["passed"],
        "shards_passed":report["shard_report"]["passed"]}))
    if not report["passed"]: raise typer.Exit(1)
@synthesis_app.command("export-bank")
def synthesis_export_bank(bank_root:Path=typer.Option(...,'--bank-root',exists=True,file_okay=False),
        frozen_root:Path|None=typer.Option(None,'--frozen-root'),export_root:Path|None=typer.Option(None,'--export-root'),
        allow_unvalidated:bool=typer.Option(False,'--allow-unvalidated',help='move a RETAINED FAILED run for inspection; freezing still requires a validated bank'),
        dry_run:bool=typer.Option(False,'--dry-run'))->None:
    """Freeze a validated bank under its versioned path and write one transport archive."""
    from prism_fas.synthesis.synthetic_export import export_archive,freeze_bank
    frozen={"status":"skipped"}
    if frozen_root is not None: frozen=freeze_bank(Path(bank_root),Path(frozen_root),dry_run=dry_run)
    archive={"status":"skipped"}
    if export_root is not None:
        Path(export_root).mkdir(parents=True,exist_ok=True)
        archive=export_archive(Path(bank_root),Path(export_root),dry_run=dry_run,require_validated=not allow_unvalidated)
    typer.echo(json.dumps({"frozen":frozen,"archive":archive,
        "written":[] if dry_run else archive.get("written",[])}))
@synthesis_app.command("validate-downloaded-bank")
def synthesis_validate_downloaded_bank(archive:Path|None=typer.Option(None,'--archive',exists=True,dir_okay=False),
        destination:Path=typer.Option(Path('data/processed'),'--destination'),
        bank_root:Path|None=typer.Option(None,'--bank-root'),package_root:Path|None=typer.Option(None,'--package-root'),
        bank:Path|None=typer.Option(None,'--bank'),expected_identity:str|None=typer.Option(None,'--expected-identity'),
        report_json:Path|None=typer.Option(Path('reports/m8/local_downloaded_bank_validation.json'),'--report-json'),
        dry_run:bool=typer.Option(False,'--dry-run'))->None:
    """Extract one transport archive locally and run the full validator on it."""
    from prism_fas.synthesis.synthetic_export import extract_archive
    from prism_fas.synthesis.synthetic_validation import validate_bank,write_validation_report
    d=_m8_defaults()
    extraction={"status":"skipped"}
    root=bank_root
    if archive is not None:
        if dry_run:
            typer.echo(json.dumps({"status":"dry_run","archive":archive.name,"destination":str(destination),"written":[]}));return
        extraction=extract_archive(archive,Path(destination))
        root=Path(extraction["bank_root"])
    if root is None: raise typer.BadParameter("either --archive or --bank-root is required")
    report=validate_bank(Path(root),package_root=package_root or d["package"],recipe_bank_root=bank or d["bank"])
    matches=expected_identity is None or report["bank_content_identity_sha256"]==expected_identity
    payload={**report,"extraction":extraction,"expected_identity":expected_identity,
        "local_identity_equals_remote":bool(matches)}
    if report_json and not dry_run: write_validation_report(report_json,payload)
    typer.echo(json.dumps({"passed":bool(report["passed"] and matches),"bank_id":report["bank_id"],
        "bank_content_identity_sha256":report["bank_content_identity_sha256"],
        "local_identity_equals_remote":bool(matches),"counts":report["counts"],
        "error_count":report["error_count"],"errors":report["errors"][:10]}))
    if not (report["passed"] and matches): raise typer.Exit(1)
# --- M10 experiment matrix and blind target evaluation -----------------------
from prism_fas.cli.m10 import app as m10_app
app.add_typer(m10_app, name="m10")
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
