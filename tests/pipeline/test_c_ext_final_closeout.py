"""Tests for the EXT-Q1Q2 FINAL scientific closeout
(``src/prism_fas/evaluation/c_ext_final_closeout.py``).

These tests prove that the closeout generator:

* is deterministic and byte-for-byte idempotent;
* imports nothing that could reach the network, a provider, a GPU, a trainer
  or the target labels, and has no training / generation / inference capability;
* never writes outside ``reports/c_ext_q1q2_v1/final_closeout/`` and never
  touches a historical E0..E10 artifact;
* records E11 as ``NOT_AUTHORIZED_NOT_RUN`` given the authoritative E10 decision;
* classifies every intended extension claim, and can NEVER promote blocked /
  unavailable evidence to an affirmative classification;
* never represents E9 as detector-performance evidence or as independent
  LLM-generation replication;
* never claims statistical significance;
* consults the frozen E0..E10 artifacts only and hashes them plus its own
  outputs into ``C_EXT_FINAL_EVIDENCE.sha256``.

No GPU, no LLM, no network, no target data.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import c_ext_final_closeout as fc  # noqa: E402
from prism_fas.evaluation.c_ext_common import ExtPathSafetyError  # noqa: E402

MODULE_PATH = REPO / "src" / "prism_fas" / "evaluation" / "c_ext_final_closeout.py"


@pytest.fixture(scope="module")
def body() -> dict:
    return fc.evaluate()


# --------------------------------------------------------------------------- #
# authoritative final state, independently re-derived from the artifacts
# --------------------------------------------------------------------------- #

def test_e10_gate_criteria_are_the_authoritative_frozen_set(body):
    assert body["final_status"]["e10_gate_criteria"] == {
        "C1_CROSS_DOMAIN_SIGNAL": "BLOCKED",
        "C2_VALUE_OVER_REAL_ONLY": "NOT_ESTABLISHED",
        "C3_SEMANTIC_ABLATION": "BLOCKED",
        "C4_QUALITY_CONFOUND": "PASS",
        "C5_INTEGRITY": "PASS",
    }


def test_e11_always_not_authorized_not_run(body):
    fs = body["final_status"]
    assert fs["milestones"]["E11"]["status"] == "NOT_AUTHORIZED_NOT_RUN"
    assert fs["e11"] == {
        "status": "NOT_AUTHORIZED_NOT_RUN", "authorized": False, "run": False,
        "new_llm_calls_authorized": False, "q1_mechanism_claim_supported": False,
    }
    fcl = body["final_closure"]
    assert fcl["e11_status"] == "NOT_AUTHORIZED_NOT_RUN"
    assert fcl["e11_authorized"] is False
    assert fcl["new_llm_calls_authorized"] is False
    assert fcl["q1_mechanism_claim_supported"] is False
    assert fcl["extension_status"] == "CLOSED_WITH_BLOCKED_PRIMARY_MECHANISM_EVIDENCE"
    assert body["claim_matrix"]["hard_invariants"]["e11_status"] == "NOT_AUTHORIZED_NOT_RUN"


def test_e11_status_is_pinned_to_the_real_e10_decision(body, tmp_path):
    # If the E10 decision on disk ever flipped to authorized, the closeout must
    # fail closed rather than silently emit E11 = authorized.
    e10d = json.loads((REPO / fc.CONSULTED_SOURCES["e10_gate_decision"]).read_text())
    assert e10d["e11_authorized"] is False
    assert e10d["new_llm_calls_authorized"] is False


def test_e7_recorded_blocked_never_negative(body):
    e7 = body["final_status"]["milestones"]["E7"]
    assert e7["status"] == "SCIENTIFICALLY_BLOCKED"
    blob = e7["reason"].lower()
    assert "0/3" in e7["reason"] or "0 / 3" in e7["reason"]
    assert "is_evidence_llm_recipes_generally_inferior=false" in e7["reason"]
    assert "before any target acer scoring" in blob
    for bad in ("llm failed", "llm is inferior", "llm recipes are worse",
                "negative result"):
        assert bad not in blob


def test_e8_recorded_descriptive_only(body):
    e8 = body["final_status"]["milestones"]["E8"]
    assert e8["status"] == "COMPLETED_DESCRIPTIVE_ONLY"
    assert e8["descriptive_only"] is True
    r = e8["reason"]
    assert "inferential_status=NOT_CLAIMED" in r
    assert "statistical_significance_claimed=false" in r
    assert "llm_superiority_supported=false" in r
    # authoritative means, read straight from the artifact
    assert "0.3487" in r and "0.3506" in r and "0.3516" in r


def test_e9_recorded_as_selection_robustness_only(body):
    e9 = body["final_status"]["milestones"]["E9"]
    assert e9["status"] == "CLOSED_BANK_SELECTION_ROBUSTNESS_ONLY"
    assert e9["is_detector_performance_evidence"] is False
    assert e9["is_independent_llm_generation_replication"] is False
    assert "9/9" in e9["reason"]


def test_e1_is_a_required_consulted_source():
    # A restored E1 subtree can never be silently ignored: it is a hard input.
    assert "e1_recipe_analysis" in fc.CONSULTED_SOURCES
    assert fc.CONSULTED_SOURCES["e1_recipe_analysis"].endswith(
        "e1_recipe_analysis/E1_RECIPE_ANALYSIS.json")


def test_e1_resolved_complete_from_restored_artifact_never_not_present(body):
    e1 = body["final_status"]["milestones"]["E1"]
    assert e1["status"] == "COMPLETE"
    assert e1["status"] != "NOT_PRESENT_IN_THIS_CHECKOUT"
    # re-audited from the artifact, not assumed
    art = json.loads((REPO / fc.CONSULTED_SOURCES["e1_recipe_analysis"]).read_text())
    assert art["milestone"] == "E1"
    assert art["status"].startswith("COMPLETE")
    assert e1["artifact_status_field"] == art["status"]
    assert e1["not_consulted_by_any_e10_criterion"] is True
    r = e1["reason"].lower()
    assert "structural" in r and "not consulted by any e10" in r
    assert "semantic" in r  # explicit non-semantic caveat
    # the E10 disagreement is recorded, not silently reconciled
    assert "e10_reconciliation" in e1
    assert "evidence_not_present_in_checkout" in e1["e10_reconciliation"].lower()
    # E10 gate results untouched
    assert body["final_status"]["e10_gate_criteria"]["C3_SEMANTIC_ABLATION"] == "BLOCKED"


def test_absent_e1_subtree_fails_closed_not_silently_not_present(tmp_path):
    # Every consulted input EXCEPT E1 -> load_inputs must raise, so E1 can never
    # silently fall back to NOT_PRESENT_IN_THIS_CHECKOUT.
    root = tmp_path
    for name, rel in fc.CONSULTED_SOURCES.items():
        if name == "e1_recipe_analysis":
            continue
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO / rel).read_bytes())
    dotgit = REPO / ".git"
    (root / ".git").write_text(dotgit.read_text(encoding="utf-8")
                               if dotgit.is_file() else "", encoding="utf-8")
    with pytest.raises(fc.FinalCloseoutError) as exc:
        fc.load_inputs(root)
    msg = str(exc.value)
    assert "E1_RECIPE_ANALYSIS.json" in msg
    assert "NOT_PRESENT_IN_THIS_CHECKOUT" in msg


def test_milestone_status_map_is_complete_and_no_failure_or_silent_absence(body):
    ms = body["final_status"]["milestones"]
    assert set(ms) == {f"E{i}" for i in range(12)}
    for key, m in ms.items():
        for bad in ("FAIL", "FAILED", "NOT_EXECUTED"):
            assert bad not in m["status"], f"{key}: {m['status']}"
    # no milestone may sit at NOT_PRESENT while its evidence is on disk
    assert ms["E1"]["status"] != "NOT_PRESENT_IN_THIS_CHECKOUT"
    present_now = {"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"}
    for key in present_now:
        assert ms[key]["status"] != "NOT_PRESENT_IN_THIS_CHECKOUT", key


# --------------------------------------------------------------------------- #
# claim matrix -- classification integrity
# --------------------------------------------------------------------------- #

REQUIRED_CLAIMS = {
    "general_llm_superiority", "q_confound_robustness", "semantic_coupling_shuffle",
    "value_over_real_only", "cross_domain_three_fold_robustness",
    "bank_selection_subset_robustness", "independent_llm_generation_robustness",
    "integrity_leakage_status",
}


def _claim(body, cid):
    return next(c for c in body["claim_matrix"]["claims"] if c["claim_id"] == cid)


def test_claim_matrix_covers_every_intended_claim(body):
    ids = {c["claim_id"] for c in body["claim_matrix"]["claims"]}
    assert REQUIRED_CLAIMS.issubset(ids)
    for c in body["claim_matrix"]["claims"]:
        assert c["classification"] in fc.CLASSIFICATIONS


def test_claim_matrix_expected_classifications(body):
    got = {c["claim_id"]: c["classification"]
           for c in body["claim_matrix"]["claims"]}
    assert got["general_llm_superiority"] == "NOT_SUPPORTED"
    assert got["q_confound_robustness"] == "DESCRIPTIVELY_SUPPORTED_ONLY"
    assert got["semantic_coupling_shuffle"] == "BLOCKED_BY_PROTOCOL_FEASIBILITY"
    assert got["value_over_real_only"] == "NOT_ESTABLISHED"
    assert got["cross_domain_three_fold_robustness"] == "BLOCKED_BY_PROTOCOL_FEASIBILITY"
    assert got["bank_selection_subset_robustness"] == "DESCRIPTIVELY_SUPPORTED_ONLY"
    assert got["independent_llm_generation_robustness"] == "NOT_ESTABLISHED"
    assert got["integrity_leakage_status"] == "SUPPORTED"


def test_blocked_or_unavailable_evidence_is_never_affirmative(body):
    for c in body["claim_matrix"]["claims"]:
        if c["evidence_status"] in fc.NON_AFFIRMATIVE_EVIDENCE:
            assert c["classification"] not in fc.AFFIRMATIVE_CLASSIFICATIONS, c["claim_id"]


def test_invariant_guard_rejects_promoting_blocked_evidence_to_supported(body):
    tampered = json.loads(json.dumps(body["claim_matrix"]))
    for c in tampered["claims"]:
        if c["claim_id"] == "cross_domain_three_fold_robustness":
            c["classification"] = "SUPPORTED"      # blocked -> SUPPORTED: forbidden
    with pytest.raises(fc.FinalCloseoutError):
        fc.assert_claim_matrix_invariants(tampered)


def test_invariant_guard_rejects_promoting_unavailable_to_descriptive(body):
    tampered = json.loads(json.dumps(body["claim_matrix"]))
    for c in tampered["claims"]:
        if c["claim_id"] == "independent_llm_generation_robustness":
            c["classification"] = "DESCRIPTIVELY_SUPPORTED_ONLY"
    with pytest.raises(fc.FinalCloseoutError):
        fc.assert_claim_matrix_invariants(tampered)


def test_invariant_guard_rejects_significance_language_in_affirmative_fields(body):
    tampered = json.loads(json.dumps(body["claim_matrix"]))
    for c in tampered["claims"]:
        if c["claim_id"] == "q_confound_robustness":
            c["allowed_statement"] += " The effect is statistically significant."
    with pytest.raises(fc.FinalCloseoutError):
        fc.assert_claim_matrix_invariants(tampered)


def test_e9_claim_not_detector_performance_or_independent_generation(body):
    bsr = _claim(body, "bank_selection_subset_robustness")
    assert bsr["is_detector_performance_evidence"] is False
    assert bsr["is_independent_llm_generation_replication"] is False
    assert bsr["classification"] != "SUPPORTED"
    joined = (bsr["reason"] + bsr["allowed_statement"]).lower()
    assert "detector" not in bsr["allowed_statement"].lower() or "not" in joined
    ind = _claim(body, "independent_llm_generation_robustness")
    assert ind["classification"] not in fc.AFFIRMATIVE_CLASSIFICATIONS
    assert "not independent generator replication" in ind["reason"].lower() \
        or "not independent generation" in ind["reason"].lower() \
        or "explicitly not independent" in ind["reason"].lower()


def test_no_statistical_significance_claimed_anywhere(body):
    hi = body["claim_matrix"]["hard_invariants"]
    assert hi["no_statistical_significance_claimed_anywhere"] is True
    for c in body["claim_matrix"]["claims"]:
        affirmative = " ".join([c["claim_text"], c["reason"],
                                c["allowed_statement"]]).lower()
        assert "statistically significant" not in affirmative


def test_general_llm_superiority_is_not_supported(body):
    c = _claim(body, "general_llm_superiority")
    assert c["classification"] == "NOT_SUPPORTED"
    assert "claim ceiling" in c["reason"].lower()
    assert "LLM is superior overall" in c["prohibited_statements"]


# --------------------------------------------------------------------------- #
# results table -- separation of evidence classes
# --------------------------------------------------------------------------- #

def _rows(body):
    lines = body["results_table_csv"].strip().split("\n")
    header = lines[0].split(",")
    out = []
    for ln in lines[1:]:
        # naive split is fine: quoted commas only appear in the last field
        parts = ln.split(",", len(header) - 1)
        out.append(dict(zip(header, parts)))
    return out


def test_results_table_separates_the_evidence_classes(body):
    classes = {r["evidence_class"] for r in _rows(body)}
    assert "e1_recipe_structural_descriptive" in classes
    assert "e4_historical_partial_ext_f1_threshold_transfer" in classes
    assert "e5_partial_ext_f1_real_only_completed" in classes
    assert "e8_qmatched_ext_f1_descriptive" in classes
    assert "e9_bank_selection_robustness_structural" in classes
    assert "e7_three_fold_unavailable" in classes


def test_results_table_e1_rows_are_structural_only_not_target_metrics(body):
    e1rows = [r for r in _rows(body)
              if r["evidence_class"] == "e1_recipe_structural_descriptive"]
    assert e1rows
    for r in e1rows:
        assert r["milestone"] == "E1"
        assert r["inferential_status"] == "not_claimed"
        note = r["scope_note"].lower()
        assert "not a target/detector metric" in note
        assert "not a semantic-mechanism" in note
        assert "not in the holm family" in note
        # no E1 row masquerades as a target ACER / detector number
        assert "acer" not in r["metric"].lower()


def test_results_table_e7_three_fold_rows_are_unavailable_not_a_number(body):
    e7rows = [r for r in _rows(body) if r["evidence_class"] == "e7_three_fold_unavailable"]
    assert len(e7rows) == 12   # 4 conditions x 3 folds
    for r in e7rows:
        assert r["value"] == "UNAVAILABLE_UPSTREAM_BLOCKED"
        assert r["inferential_status"] == "not_applicable"


def test_results_table_every_real_number_is_descriptive(body):
    for r in _rows(body):
        if r["value"].startswith("UNAVAILABLE"):
            continue
        assert r["inferential_status"] == "not_claimed"
        assert r["statistic_type"] != "significance"


def test_results_table_e9_rows_carry_the_disclaimers(body):
    e9rows = [r for r in _rows(body)
              if r["evidence_class"] == "e9_bank_selection_robustness_structural"]
    assert e9rows
    for r in e9rows:
        note = r["scope_note"].lower()
        assert "not detector-performance evidence" in note \
            or "no target access" in note


def test_results_table_key_numbers_match_the_artifacts(body):
    rows = _rows(body)

    def val(ec, arm, metric):
        return next(r["value"] for r in rows
                   if r["evidence_class"] == ec and r["arm_or_pair"] == arm
                   and r["metric"] == metric)

    e8 = json.loads((REPO / fc.CONSULTED_SOURCES["e8_target_evaluation_closure"])
                    .read_text())["acer_means_by_arm"]
    assert abs(float(val("e8_qmatched_ext_f1_descriptive", "LLM",
                         "target_ACER_mean")) - e8["LLM"]) < 1e-6
    e5 = json.loads((REPO / fc.CONSULTED_SOURCES["e5_target_scoring_result"])
                    .read_text())["aggregate"]["acer"]["mean"]
    assert abs(float(val("e5_partial_ext_f1_real_only_completed", "REAL_ONLY",
                         "target_ACER_mean")) - e5) < 1e-6


# --------------------------------------------------------------------------- #
# provenance / closure
# --------------------------------------------------------------------------- #

def test_final_closure_records_spec_and_closure_provenance(body):
    fcl = body["final_closure"]
    spec = fcl["governing_spec"]
    assert spec["ext_q1q2_detailed_spec"]["sha256"]
    assert spec["version_c_v1_5_spec"]["sha256"] == \
        "ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5"
    prov = fcl["closure_provenance"]
    assert set(prov) == {"E7", "E8", "E9", "E10"}
    for m in ("E7", "E8", "E9", "E10"):
        assert len(prov[m]["artifact_sha256"]) == 64
    assert prov["E10"]["gate_identity"]
    assert prov["E9"]["is_detector_performance_evidence"] is False
    assert prov["E9"]["is_independent_llm_generation_replication"] is False
    assert prov["E8"]["inferential_status"] == "NOT_CLAIMED"
    # E0-E11 status map: E0-E10 resolved, E11 governed by E10
    mstat = fcl["milestones_status"]
    assert set(mstat) == {f"E{i}" for i in range(12)}
    assert mstat["E1"] == "COMPLETE"
    assert mstat["E11"] == "NOT_AUTHORIZED_NOT_RUN"
    assert "NOT_PRESENT" not in json.dumps(mstat)
    assert any(a["name"] == "e1_recipe_analysis" for a in fcl["consulted_artifacts"])


def test_final_closure_task_state_is_all_read_only(body):
    ts = body["final_closure"]["closeout_task_state"]
    assert ts["llm_calls"] == 0
    assert ts["provider_or_network_calls"] == 0
    assert ts["gpu_jobs"] == 0
    assert ts["target_access"] is False
    assert ts["target_features_accessed"] is False
    assert ts["target_labels_accessed"] is False
    assert ts["no_new_scientific_experiment_performed"] is True
    assert ts["no_e0_e10_evidence_modified"] is True
    assert ts["no_frozen_protocol_modified"] is True


def test_closeout_identity_is_deterministic(body):
    b2 = fc.evaluate()
    assert body["final_closure"]["closeout_identity"] == \
        b2["final_closure"]["closeout_identity"]


# --------------------------------------------------------------------------- #
# determinism / idempotency / write confinement
# --------------------------------------------------------------------------- #

def test_evaluate_is_deterministic(body):
    a = json.dumps(body["final_status"], sort_keys=True)
    b = json.dumps(fc.evaluate()["final_status"], sort_keys=True)
    assert a == b
    assert body["results_table_csv"] == fc.evaluate()["results_table_csv"]
    assert body["paper_ready_summary_md"] == fc.evaluate()["paper_ready_summary_md"]


def test_write_artifacts_is_byte_for_byte_idempotent(tmp_path):
    # copy just the consulted inputs into a scratch root, then write twice
    root = tmp_path
    for rel in fc.CONSULTED_SOURCES.values():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO / rel).read_bytes())
    (root / ".git").write_text((REPO / ".git").read_text(encoding="utf-8")
                               if (REPO / ".git").is_file() else "", encoding="utf-8")

    body = fc.evaluate(root)
    w1 = fc.write_artifacts(body, root=root)
    first = {k: (root / v).read_bytes() for k, v in w1.items()}
    w2 = fc.write_artifacts(fc.evaluate(root), root=root)
    second = {k: (root / v).read_bytes() for k, v in w2.items()}
    assert first == second

    # all six outputs, confined to the final_closeout subtree
    assert set(w1) == {"final_status", "results_table", "claim_matrix",
                       "paper_ready_summary", "final_closure", "evidence_manifest"}
    for rel in w1.values():
        assert rel.startswith(fc.OUTPUT_SUBTREE + "/")


def test_evidence_manifest_hashes_inputs_and_all_outputs(tmp_path):
    root = tmp_path
    for rel in fc.CONSULTED_SOURCES.values():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO / rel).read_bytes())
    (root / ".git").write_text((REPO / ".git").read_text(encoding="utf-8")
                               if (REPO / ".git").is_file() else "", encoding="utf-8")
    fc.write_artifacts(fc.evaluate(root), root=root)
    manifest = (root / fc.EVIDENCE_MANIFEST_RELPATH).read_text(encoding="utf-8")
    listed = {ln.split("  ", 1)[1] for ln in manifest.strip().split("\n")}
    for rel in fc.CONSULTED_SOURCES.values():
        assert rel in listed
    for rel in fc.OUTPUT_RELPATHS:
        if rel == fc.EVIDENCE_MANIFEST_RELPATH:
            continue
        assert rel in listed


def test_write_guard_rejects_paths_outside_the_final_closeout_subtree():
    from prism_fas.evaluation.c_ext_common import write_json_atomic
    for bad in ("reports/c_ext_q1q2_v1/e10_gate/HACK.json",
                "reports/c_ext_q1q2_v1/e7_three_fold/HACK.json",
                "reports/full/c6/HACK.json"):
        with pytest.raises(ExtPathSafetyError):
            fc.assert_ext_write_path(bad, must_be_under=fc.OUTPUT_SUBTREE)
    with pytest.raises(ExtPathSafetyError):
        write_json_atomic("reports/full/c3/HACK.json", {"x": 1})


def test_all_declared_outputs_live_under_final_closeout_only():
    for rel in fc.OUTPUT_RELPATHS:
        assert rel.startswith("reports/c_ext_q1q2_v1/final_closeout/")
        assert "/attempts/" not in rel


def test_module_consults_frozen_reports_under_ext_root_only():
    for rel in fc.CONSULTED_SOURCES.values():
        assert rel.startswith("reports/c_ext_q1q2_v1/")


def test_head_mismatch_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "_head_commit", lambda root: "deadbeef" * 5)
    with pytest.raises(fc.FinalCloseoutError):
        fc.load_inputs(REPO)


# --------------------------------------------------------------------------- #
# no forbidden capability / imports / target-label resolution
# --------------------------------------------------------------------------- #

FORBIDDEN_IMPORT_ROOTS = {
    "torch", "tensorflow", "jax", "requests", "urllib", "http", "socket",
    "aiohttp", "httpx", "websocket", "grpc", "boto3", "openai", "anthropic",
    "google", "vertexai", "cohere", "sklearn", "pandas", "pyarrow", "yaml",
    "subprocess",
}
FORBIDDEN_IMPORT_SUBMODULES = {
    "prism_fas.detector", "prism_fas.synthesis", "prism_fas.providers",
    "prism_fas.llm", "prism_fas.recipes", "prism_fas.pipeline",
}


def test_module_imports_no_network_provider_gpu_trainer_or_target_surface():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    for m in mods:
        assert m.split(".")[0] not in FORBIDDEN_IMPORT_ROOTS, f"forbidden import: {m}"
        for sub in FORBIDDEN_IMPORT_SUBMODULES:
            assert not m.startswith(sub), f"forbidden import: {m}"
    assert {m for m in mods if m.startswith("prism_fas")} == {
        "prism_fas.evaluation.c_ext_common"}


def test_module_has_no_training_or_generation_capability():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            continue
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
        if isinstance(node, ast.Name):
            names.append(node.id)
    for token in ("M9Trainer", "backward", "load_state_dict", "state_dict",
                  "build_matched_banks", "SyntheticBankGenerator", "fit", "step",
                  "zero_grad", "optimizer", "generate", "render", "provider"):
        assert token not in names, f"training/generation symbol: {token!r}"


def test_module_never_resolves_target_labels_or_features():
    # The dataset may be named in explanatory prose, but no code path may
    # resolve a raw target label / feature artifact.
    src = MODULE_PATH.read_text(encoding="utf-8").lower()
    for token in ("target_label_root", "target_feature_root", "target_labels_root",
                  "data/evaluation_only", "prism_target_eval", "siw_target_labels",
                  ".parquet", "read_parquet", "label_artifact_relative_path"):
        assert token not in src, f"target-resolution token present: {token!r}"


def test_paper_ready_summary_has_the_required_sections(body):
    md = body["paper_ready_summary_md"]
    for section in ("## What was attempted", "## What completed",
                    "## What was blocked", "## Valid quantitative results",
                    "## Limitations", "## Claims that ARE allowed",
                    "## Claims that must NOT be made"):
        assert section in md
    low = md.lower()
    assert "statistically significant" in low  # only inside the prohibition list
    assert "no significance is claimed anywhere" in low
    assert "e7 was **blocked by source-only matched-bank feasibility" in low
    assert "e10 did not authorize e11" in low
    # E1 now reported as completed, with the explicit non-semantic caveat
    assert "**e1** recipe-bank structural analysis -- complete" in low
    assert "structurally distinguishable" in low
    assert "does not\nmeasure semantic" in low or "does not measure semantic" in low \
        or "not evidence of a semantic mechanism" in low
    # forbidden framings appear ONLY as quoted prohibitions
    for banned in ("llm is superior overall", "the llm mechanism is proven",
                   "e9 proves detector robustness"):
        assert banned in low  # present, but...
        # ...only in the "must NOT" section
        assert low.index(banned) > low.index("## claims that must not be made")
