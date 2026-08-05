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
