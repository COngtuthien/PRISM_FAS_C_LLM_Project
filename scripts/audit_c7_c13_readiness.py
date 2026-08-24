"""Build the C7-C13 production-path readiness inventory.

An engineering audit artifact, not scientific evidence. It answers one question
per stage — *could a `--profile full` run of this stage produce a scientific
result today, and if not, exactly what stops it* — and it answers it by INSPECTING
the code rather than by transcribing a belief about it.

Three facts per stage come from the source itself, so the report cannot drift
from the repository the way a hand-written matrix does:

* whether the stage has a `_scientific_workflow` at all;
* whether `workflow` dispatches on the execution context;
* which fixture producers the module still names, and whether each is guarded.

Everything else — required inputs, the lock a stage produces and the verifier
that reads it — is read off the adapter's own declarations.

    python scripts/audit_c7_c13_readiness.py
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

ADAPTERS = REPO / "src" / "prism_fas" / "pipeline" / "adapters"
STAGES = ("c7", "c8", "c9", "c10", "c11", "c12", "c13")
OUT_MD = "reports/readiness/C7_C13_PRODUCTION_READINESS.md"
OUT_JSON = "reports/readiness/C7_C13_PRODUCTION_READINESS.json"

FIXTURE_PRODUCERS = ("_fixture_batch", "_fixture_roots", "_fixture_rows",
                     "prediction_rows", "evaluation_labels", "audit_batch",
                     "build_audit_detector", "_complete_evidence")

#: Everything the code cannot tell us: purpose, hard acceptance, what a stage
#: still owes and whether it may be run. Transcribed from the v1.5 stage table
#: and from the decision records in `configs/` and `docs/PROJECT_STATE.md`.
NARRATIVE: dict[str, dict[str, Any]] = {
    "c7": {
        "purpose": ("prove every typed primary row is executable on a CPU fixture, "
                    "enforce the §13.5 decision-dependency guards, and close the "
                    "§15.2.2 detector/loss envelope in ONE bounded coordinate pass "
                    "before C8"),
        "hard_acceptance": [
            "Track G and Track R typed variants instantiate, forward, produce a "
            "finite loss, backward, step, checkpoint and resume on a CPU fixture",
            "no experiment-id branching; both tracks are configurations of one "
            "implementation",
            "Track-R ConvNeXt and RegionFusion have non-zero autograd dependency on "
            "the fused logit, and the branch-intervention audit passes",
            "Track G decides on global_logit_G / p_G; Track R on fused_logit_R / p_R",
            "decision_graph_hash serialized into run identity",
            "manifold=OFF primary Track R executes no L_real / L_out / L_clean",
            "K=4 is an explicit typed secondary variant",
            "calibration fits and thresholds the SAME decision quantity",
            "one deterministic coordinate pass in the frozen §15.2.2 order, "
            "candidates anchor x {0.5,1,2} (warm-up x {0.5,1,1.5} clipped to "
            "[0,0.20]), inactive terms skipped, every trial retained, canonical "
            "SHA-256 tie-break",
            "DETECTOR_CONFIG_LOCK.json written only after the envelope is terminal",
        ],
        "required_target_capability": "none",
        "forbidden_target_capability": "all",
        "scientific_path_ever_executed": False,
        "unresolved_decisions": [],
        "resolved_decisions": [
            "C7_SOURCE_SEARCH_SYNTHETIC_ARM = DET, FROZEN 2026-08-24 before any C7 "
            "scientific metric existed. One bounded pass per TRACK, both anchored on "
            "the C6 DET bank; every primary generator arm of a track trains at that "
            "track's single frozen configuration in C8. Record: "
            "configs/search/c7_source_search_decision.yaml, identity "
            "ed4f6b777d9f95f089a76191b863e2fb2df0b9e13434470ffd736d6e511b474e",
            "the learning-rate anchor interpretation (B_common_multiplier), "
            "APPROVED in configs/search/lr_anchor_decision.yaml",
            "the optimizer family, uniquely inherited as AdamW from "
            "configs/train/m9_reference.yaml",
            "the protocol (P3), the ranking tuple (P3_READY) and the per-trial "
            "schedule (frozen_m9_schedule), all in the same decision record",
        ],
        "safe_to_run": ("NO on this laptop — the scientific path requires CUDA, the "
                        "M3B package, the pinned weights, the C5 candidate tree and "
                        "the frozen C6 closure, none of which is here. Every "
                        "result-affecting DECISION is now frozen, so what remains is "
                        "inputs and hardware. The engineering readiness path runs on "
                        "CPU."),
        "blockers": [
            "reports/full/c6 absent on this host (the C6 GPU evidence lives on the "
            "execution backend)",
            "data/packages/prism_data_v1_m3b absent",
            "runs/full/c5/scientific/candidates absent",
            "weights/ pinned SigLIP2 + ConvNeXt absent",
            "no CUDA device",
        ],
        "code_paths_never_exercised": [
            "the real M9Trainer flow inside a C7 trial (rehearsed with a stub "
            "trainer; the trainer itself is covered by tests/test_m9_regional_detector)",
            "C6MatchedBankReader over 1024 real candidates (rehearsed at 4)",
            "the CUDA branch of _scientific_device",
        ],
    },
    "c8": {
        "purpose": ("execute the frozen §18 source-only matrix — 42 atomic runs over "
                    "protocol x method x config x seed — at the configuration C7 "
                    "froze, and emit durable per-run evidence for every one"),
        "hard_acceptance": [
            "Track-G P1/P2: RND/DET/LLM at 3 seeds each",
            "Track-G P3-ready: RND/DET/LLM at 5 seeds each",
            "Track-R P3-ready: DET/LLM at 3 seeds each",
            "PromptHead LLM ON/OFF ablation at 3 seeds",
            "42 atomic rows, each with durable evidence; no hidden row, no omitted row",
            "P1/P2 checkpoint selection on that protocol's source_dev only; the "
            "cross-domain side is diagnostic and never a selection signal",
            "P3-ready selection on equal-weight CASIA-dev + MSU-dev, never SiW",
            "calibration on source_dev only, fitting and thresholding the row's own "
            "decision logit and score",
            "cross-source diagnostics and calibration stability exist before C9",
            "no SiW P3 scoring",
        ],
        "required_target_capability": "none",
        "forbidden_target_capability": "all",
        "scientific_path_ever_executed": False,
        "unresolved_decisions": [],
        "resolved_decisions": [
            "the §18 matrix composition and the fixed seed family 20260806-20260810, "
            "materialized by prism_fas.evaluation.source_matrix.build_plan",
        ],
        "safe_to_run": ("NO on this laptop — blocked on the same missing inputs as "
                        "C7 plus a verifying DETECTOR_CONFIG_LOCK. The rehearsal "
                        "path runs on CPU over a bounded sample."),
        "blockers": [
            "reports/full/c7/DETECTOR_CONFIG_LOCK.json absent — C7's decisions are "
            "all frozen but C7 has not RUN scientifically, so the lock does not "
            "exist yet",
            "the same absent C6/C5/package/weight inputs as C7",
            "no CUDA device",
        ],
        "code_paths_never_exercised": [
            "the real M9Trainer flow inside a row (rehearsed with a stub trainer)",
            "_cross_source_evaluation over a real second M9ValidationDataset",
            "the full 42-row schedule (4 representative rows rehearsed)",
        ],
    },
    "c9": {
        "purpose": ("freeze SOURCE_MATRIX_LOCK_C over the completed source matrix, "
                    "or refuse and name every reason"),
        "hard_acceptance": [
            "all mandatory C8 rows complete and terminal",
            "no failed and no hidden row",
            "every checkpoint, calibration and run identity frozen",
            "DETECTOR_RELIABILITY_LOCK_C valid",
        ],
        "required_target_capability": "none",
        "forbidden_target_capability": "all",
        "scientific_path_ever_executed": False,
        "unresolved_decisions": [
            "DETECTOR_BA_SEP_PROBE_PROTOCOL — NEEDS_SCIENTIFIC_DECISION",
            "DETECTOR_BA_SEP_EVIDENCE_VECTOR — NEEDS_SCIENTIFIC_DECISION",
            "DETECTOR_BA_SEP_PROBE_SEEDS — NEEDS_SCIENTIFIC_DECISION",
        ],
        "resolved_decisions": [
            "the staging of the synthetic-vs-real barrier: after C8, before C9 "
            "(SYNTHETIC_VS_REAL_RELIABILITY_STAGE, frozen 2026-08-23)",
        ],
        "safe_to_run": ("NO — and correctly so. C9 BLOCKS on "
                        "DETECTOR_RELIABILITY_LOCK_C, whose protocol is unfrozen. "
                        "The barrier must be decided separately; it may NOT be "
                        "chosen from C8 outcomes."),
        "blockers": [
            "DETECTOR_RELIABILITY_LOCK_C absent and its protocol unfrozen",
            "C8 has not run scientifically",
        ],
        "code_paths_never_exercised": [
            "the freeze over 42 real C8 manifests (rehearsed over 42 synthetic ones "
            "written in C8's own manifest schema)",
        ],
    },
    "c10": {
        "purpose": ("build and lock the sealed SiW-Mv2 target package and the label "
                    "firewall, without opening a label"),
        "hard_acceptance": [
            "the target feature package is mounted READ-ONLY for C11",
            "no training/LLM/synthesis stage may resolve a label root",
            "package identity verified and tamper-detectable",
        ],
        "required_target_capability": "sealed_real (scientific only)",
        "forbidden_target_capability": ("labels, under every profile; features under "
                                        "any non-eligible profile"),
        "scientific_path_ever_executed": False,
        "unresolved_decisions": [],
        "resolved_decisions": [],
        "safe_to_run": ("NOT in this task. Dry structural validation only; no target "
                        "bytes were opened."),
        "blockers": [
            "reports/full/c9/SOURCE_MATRIX_LOCK_C.json absent",
            "data/processed/prism_target_eval_v2 absent",
            "no scientific workflow is written; the stage is rehearsal-only",
        ],
        "code_paths_never_exercised": ["every scientific target-package path"],
        "fixed_this_task": (
            "`_build_fixture` called `sources.target_roots(...)` and then `mkdir`-ed "
            "every returned root and wrote a `labels.json` of INVENTED labels into "
            "it. Under a scientific context that call returns the REAL sealed "
            "package roots, so the stage would have written fabricated labels inside "
            "the artifact the firewall exists to protect. Now guarded by "
            "`assert_fixture_permitted` before any path is created."),
    },
    "c11": {
        "purpose": "run label-isolated P3 prediction over the sealed target package",
        "hard_acceptance": [
            "no prediction row carries a ground-truth label, attack family, raw "
            "path, subject/session taxonomy or hidden target metadata",
            "each row's PREDICTION_LOCK binds its checkpoint, calibration, inference "
            "config and package identity",
            "the TARGET_PREDICTION_LOCKSET is validated twice before label "
            "capability is granted",
        ],
        "required_target_capability": "features, read-only",
        "forbidden_target_capability": "labels",
        "scientific_path_ever_executed": False,
        "unresolved_decisions": [],
        "resolved_decisions": [],
        "safe_to_run": "NOT in this task. No target inference was executed.",
        "blockers": [
            "C9 and C10 have not run scientifically",
            "no scientific workflow is written; prediction rows come from "
            "adapters.tiny under rehearsal",
        ],
        "code_paths_never_exercised": ["every scientific target-inference path"],
        "fixed_this_task": (
            "`_build` constructed prediction rows from `tiny.prediction_rows` with no "
            "context guard, so a scientific context would have written invented "
            "scores behind a PREDICTION_LOCK. Now guarded."),
    },
    "c12": {
        "purpose": ("unlock the sealed labels inside the isolated C-G8 scorer and "
                    "compute the P3 statistics"),
        "hard_acceptance": [
            "the scorer's import closure contains no training capability",
            "a dry run validates preconditions without opening label bytes",
            "label capability is refused before a lockset exists",
            "nothing C12 writes can mutate a C0-C11 artifact",
            "a single-seed comparison is refused rather than reported",
        ],
        "required_target_capability": "labels, read-only, scorer-scoped",
        "forbidden_target_capability": ("any label access outside the scorer; any "
                                        "write to model state or calibration"),
        "scientific_path_ever_executed": False,
        "unresolved_decisions": [],
        "resolved_decisions": [],
        "safe_to_run": ("NOT in this task. No label capability was granted and no "
                        "label byte was opened."),
        "blockers": [
            "C11 has not run scientifically",
            "no scientific workflow is written; labels are fabricated from video ids "
            "by adapters.tiny under rehearsal",
        ],
        "code_paths_never_exercised": ["the real sealed-label resolution path"],
        "fixed_this_task": (
            "`workflow` scored fabricated labels with no context guard. Now guarded, "
            "so a scientific profile cannot score invented labels."),
    },
    "c13": {
        "purpose": ("final acceptance, the evidence package, and the refusal to "
                    "declare completion while upstream milestones are incomplete"),
        "hard_acceptance": [
            "the acceptance matrix assembles over real upstream status",
            "negative and blocked results survive into the evidence package",
            "artifact integrity is checked by re-hashing, not by trusting a manifest",
            "a superiority claim with no statistical support is rejected",
            "C13 proposes a tag; it never creates the scientific tag",
        ],
        "required_target_capability": "none",
        "forbidden_target_capability": "all",
        "scientific_path_ever_executed": False,
        "unresolved_decisions": [],
        "resolved_decisions": [],
        "safe_to_run": ("NOT as science. C13's honest verdict today is a refusal "
                        "naming C4-C12 as scientifically incomplete."),
        "blockers": ["C4-C12 have not run scientifically"],
        "code_paths_never_exercised": ["C_ACCEPTANCE under a complete pipeline"],
    },
}


def _module(stage: str) -> tuple[str, ast.Module]:
    source = (ADAPTERS / f"{stage}.py").read_text(encoding="utf-8")
    return source, ast.parse(source)


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)}


def _calls(node: ast.AST) -> list[str]:
    found: list[str] = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        name = (item.func.id if isinstance(item.func, ast.Name)
                else item.func.attr if isinstance(item.func, ast.Attribute) else "")
        if name:
            found.append(name)
    return found


def inspect_stage(stage: str) -> dict[str, Any]:
    """What the SOURCE says about this stage. No belief, no transcription."""
    source, tree = _module(stage)
    functions = _functions(tree)
    adapter = _adapter(stage)

    scientific = functions.get("_scientific_workflow")
    workflow = functions.get("workflow")
    workflow_body = ast.get_source_segment(source, workflow) if workflow else ""

    fixture_sites: list[dict[str, Any]] = []
    for name, node in sorted(functions.items()):
        body = ast.get_source_segment(source, node) or ""
        producers = sorted({call for call in _calls(node) if call in FIXTURE_PRODUCERS})
        if not producers:
            continue
        fixture_sites.append({
            "function": name, "producers": producers,
            "guarded": ("assert_fixture_permitted" in body
                        or "fixtures_permitted" in body or "is_scientific" in body)})

    return {
        "stage": stage.upper(),
        "module": f"src/prism_fas/pipeline/adapters/{stage}.py",
        "title": adapter.title,
        "modes": list(adapter.modes),
        "scientific_modes": list(getattr(_stage_module(stage), "SCIENTIFIC_MODES", ())),
        "requires_gpu": bool(adapter.requires_gpu),
        "required_inputs": [
            {"name": item.name, "path": item.relative_path,
             "description": item.description,
             "present_on_this_host": (REPO / item.relative_path).exists()}
            for item in adapter.required_inputs()],
        "semantic_preconditions_declared": "semantic_preconditions" in functions,
        "engineering_workflow": "_engineering_workflow" in functions,
        "scientific_workflow": scientific is not None,
        "workflow_dispatches_on_context": bool(
            workflow_body and "context.is_scientific" in workflow_body),
        "claims_scientific_evidence": "scientific_evidence" in source,
        "fixture_call_sites": fixture_sites,
        "unguarded_fixture_sites": [item["function"] for item in fixture_sites
                                    if not item["guarded"]],
        "produces_lock": _lock_for(stage),
        "lock_verifier": _verifier_for(stage),
    }


def _stage_module(stage: str) -> Any:
    import importlib

    return importlib.import_module(f"prism_fas.pipeline.adapters.{stage}")


def _adapter(stage: str) -> Any:
    module = _stage_module(stage)
    return getattr(module, f"{stage.upper()}Adapter")()


#: Which governing lock each stage produces, and who verifies it. Read from the
#: producing module's own constant where one exists, so a renamed path moves here
#: automatically.
def _lock_for(stage: str) -> str | None:
    module = _stage_module(stage)
    for name in ("SCIENTIFIC_CONFIG_LOCK_PATH",):
        if hasattr(module, name):
            return str(getattr(module, name))
    if stage == "c9":
        return f"{module.SCIENTIFIC_REPORTS}/{module.SOURCE_MATRIX_LOCK}"
    return {"c8": "reports/full/c8/C8_ACCEPTANCE.json",
            "c10": "reports/full/c10/TARGET_PACKAGE_LOCK.json (rehearsal-only today)",
            "c11": "reports/full/c11/TARGET_PREDICTION_LOCKSET.json "
                   "(rehearsal-only today)",
            "c13": "reports/full/c13/C_ACCEPTANCE.json (never produced)"}.get(stage)


def _verifier_for(stage: str) -> str | None:
    return {
        "c7": "prism_fas.pipeline.adapters.c7.verify_detector_config_lock "
              "(module level; shared with C8)",
        "c8": "prism_fas.pipeline.adapters.c8.C8Adapter._scientific_acceptance, over "
              "the canonical plan",
        "c9": "prism_fas.evaluation.source_lock.validate",
        "c10": "prism_fas.evaluation.firewall.TargetLabelFirewall",
        "c11": "prism_fas.evaluation.target_prediction.validate_predictions",
        "c12": "prism_fas.evaluation.scoring (isolated C-G8 scorer)",
        "c13": None,
    }.get(stage)


def build() -> dict[str, Any]:
    stages = []
    for stage in STAGES:
        row = inspect_stage(stage)
        row.update(NARRATIVE[stage])
        # Two axes, deliberately not one. `code_path_ready` is a statement about
        # the IMPLEMENTATION: a scientific workflow exists, `workflow` dispatches
        # on the context, no fixture producer is reachable from it, and no
        # result-affecting decision is outstanding. `blocked_on_inputs` is a
        # statement about THIS HOST. A stage can be code-path ready and still
        # unrunnable here, and collapsing the two into one "ready" field is how a
        # reader concludes a stage may be run when it may not.
        row["code_path_ready"] = bool(
            row["scientific_workflow"] and row["workflow_dispatches_on_context"]
            and not row["unguarded_fixture_sites"] and not row["unresolved_decisions"])
        row["blocked_on_inputs"] = [item["path"] for item in row["required_inputs"]
                                    if not item["present_on_this_host"]]
        row["runnable_on_this_host"] = bool(
            row["code_path_ready"] and not row["blocked_on_inputs"]
            and not row["requires_gpu"])
        stages.append(row)

    return {
        "schema_version": "prism-c7-c13-readiness-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "artifact_kind": "ENGINEERING_AUDIT_NOT_SCIENTIFIC_EVIDENCE",
        "scientific_eligible": False,
        "execution_profile": "none",
        "host": "development laptop; no CUDA, no source package, no target package",
        "target_access": 0,
        "stages": stages,
        "summary": {
            "code_path_ready": [row["stage"] for row in stages
                                if row["code_path_ready"]],
            "runnable_on_this_host": [row["stage"] for row in stages
                                      if row["runnable_on_this_host"]],
            "with_scientific_workflow": [row["stage"] for row in stages
                                         if row["scientific_workflow"]],
            "without_scientific_workflow": [row["stage"] for row in stages
                                            if not row["scientific_workflow"]],
            "with_unguarded_fixture_sites": [row["stage"] for row in stages
                                             if row["unguarded_fixture_sites"]],
            "with_unresolved_decisions": [row["stage"] for row in stages
                                          if row["unresolved_decisions"]],
            "scientific_path_ever_executed": [],
        },
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# C7-C13 production-path readiness",
        "",
        "**Engineering audit artifact. Not scientific evidence.** Nothing here is a",
        "measurement, and nothing here may support a claim. It answers one question",
        "per stage: could `--profile full` produce a scientific result today, and if",
        "not, exactly what stops it.",
        "",
        f"Generated `{report['generated_at_utc']}` on {report['host']}.",
        f"`target_access = {report['target_access']}` — no target path, label or",
        "metric was resolved while producing it.",
        "",
        "The three structural facts per stage — does a `_scientific_workflow` exist,",
        "does `workflow` dispatch on the execution context, and is every fixture",
        "producer guarded — are read out of the source by",
        "`scripts/audit_c7_c13_readiness.py`, so this table cannot drift from the",
        "repository the way a hand-written one does.",
        "",
        "## Summary",
        "",
        "| | stages |",
        "| --- | --- |",
    ]
    for label, key in (("scientific workflow present", "with_scientific_workflow"),
                       ("no scientific workflow", "without_scientific_workflow"),
                       ("code path ready", "code_path_ready"),
                       ("runnable on THIS host", "runnable_on_this_host"),
                       ("unguarded fixture call sites", "with_unguarded_fixture_sites"),
                       ("unresolved scientific decisions", "with_unresolved_decisions"),
                       ("scientific path ever executed",
                        "scientific_path_ever_executed")):
        value = ", ".join(report["summary"][key]) or "—"
        lines.append(f"| {label} | {value} |")

    for row in report["stages"]:
        lines += [
            "", f"## {row['stage']} — {row['title']}", "",
            f"**Purpose.** {row['purpose']}", "",
            f"- module: `{row['module']}`",
            f"- modes: {', '.join(f'`{m}`' for m in row['modes'])}",
            f"- scientific substages: "
            + (", ".join(f"`{m}`" for m in row["scientific_modes"]) or "none"),
            f"- requires an accelerator for scientific execution: "
            f"**{row['requires_gpu']}**",
            f"- engineering/rehearsal path: **{row['engineering_workflow'] or 'single workflow'}**",
            f"- scientific path implemented: **{row['scientific_workflow']}**",
            f"- `workflow` dispatches on the context: **{row['workflow_dispatches_on_context']}**",
            f"- declares semantic preconditions beyond existence: "
            f"**{row['semantic_preconditions_declared']}**",
            f"- may claim scientific evidence: **{row['claims_scientific_evidence']}**",
            f"- produces lock: `{row['produces_lock'] or 'none'}`",
            f"- lock verifier: `{row['lock_verifier'] or 'none'}`",
            f"- target capability required: {row['required_target_capability']}",
            f"- target capability forbidden: {row['forbidden_target_capability']}",
            f"- scientific path ever executed: **{row['scientific_path_ever_executed']}**",
            f"- **code path ready: {row['code_path_ready']}** — a statement about the "
            "implementation, not about this host",
            f"- **runnable on this host: {row['runnable_on_this_host']}**"
            + (f" (blocked on {len(row['blocked_on_inputs'])} absent input(s)"
               + (", and needs an accelerator" if row["requires_gpu"] else "") + ")"
               if not row["runnable_on_this_host"] else ""),
            "",
            "### Required inputs", "",
            "| input | path | present here |", "| --- | --- | --- |",
        ]
        for item in row["required_inputs"]:
            mark = "yes" if item["present_on_this_host"] else "**no**"
            lines.append(f"| {item['name']} | `{item['path']}` | {mark} |")

        lines += ["", "### Hard acceptance", ""]
        lines += [f"- {item}" for item in row["hard_acceptance"]]

        lines += ["", "### Fixture call sites", ""]
        if row["fixture_call_sites"]:
            lines += ["| function | producers | guarded |", "| --- | --- | --- |"]
            for item in row["fixture_call_sites"]:
                producers = ", ".join(f"`{p}`" for p in item["producers"])
                lines.append(f"| `{item['function']}` | {producers} | "
                             f"{'yes' if item['guarded'] else '**NO**'} |")
        else:
            lines.append("none.")

        if row["resolved_decisions"]:
            lines += ["", "### Resolved scientific decisions", ""]
            lines += [f"- {item}" for item in row["resolved_decisions"]]

        lines += ["", "### Unresolved result-affecting decisions", ""]
        lines += ([f"- {item}" for item in row["unresolved_decisions"]]
                  or ["none."])

        lines += ["", "### Blockers", ""]
        lines += [f"- {item}" for item in row["blockers"]] or ["none."]

        lines += ["", "### Code paths still unexercised", ""]
        lines += [f"- {item}" for item in row["code_paths_never_exercised"]]

        if row.get("fixed_this_task"):
            lines += ["", "### Fail-closed defect fixed in this task", "",
                      row["fixed_this_task"]]

        lines += ["", f"**Safe to run:** {row['safe_to_run']}"]

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    from prism_fas.pipeline.state import atomic_write_json

    report = build()
    atomic_write_json(REPO / OUT_JSON, report)
    path = REPO / OUT_MD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(report), encoding="utf-8")
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    for row in report["stages"]:
        print(f"  {row['stage']}: scientific_workflow={row['scientific_workflow']} "
              f"dispatches={row['workflow_dispatches_on_context']} "
              f"unguarded={row['unguarded_fixture_sites']} "
              f"code_path_ready={row['code_path_ready']} "
              f"runnable_here={row['runnable_on_this_host']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
