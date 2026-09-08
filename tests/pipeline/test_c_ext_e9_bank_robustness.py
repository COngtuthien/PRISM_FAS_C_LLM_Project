"""Tests for E9 -- conditional bank-selection robustness from frozen candidate
pools (src/prism_fas/evaluation/c_ext_e9_bank_robustness.py).

No GPU, no LLM, no network, no target labels/predictions/metrics. The one test
that actually runs the frozen MILP selector uses the deterministically
reconstructed RND/DET pool and is skipped when it has not been materialized.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import c_ext_e9_bank_robustness as e9  # noqa: E402
from prism_fas.evaluation.c_ext_common import ExtPathSafetyError  # noqa: E402

ARMS = ("RND", "DET", "LLM")


# --------------------------------------------------------------------------- #
# helpers for the execution-binding provenance tests (synthetic git repos)
# --------------------------------------------------------------------------- #

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, text=True,
                          capture_output=True, check=True).stdout.strip()


def _seed_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "commit.gpgsign", "false")
    # a .gitignore that does NOT hide the E9 report subtree, so the test can
    # commit the binding inputs
    (root / ".gitignore").write_text("", encoding="utf-8")


def _copy_real(root: Path, relpath: str) -> None:
    src = REPO / relpath
    dst = root / relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write(root: Path, relpath: str, obj) -> None:
    p = root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _committed_repo(tmp_path: Path, *, with_execution_inputs: bool = True):
    """A throwaway git repo modelling the corrected E9 lifecycle:

      commit #1  -> base_commit
      commit #2  -> implementation_source_commit  (E9 module + tests + protocol
                    lock + protected historical files)
      commit #3  -> execution_commit              (adds the deterministic E9
                    input artifacts: E9_INPUT_BINDING.json + frozen_pools/*)

    Returns (root, base_commit, implementation_source_commit, execution_commit).
    """
    root = tmp_path / "repo"
    _seed_repo(root)
    (root / "README").write_text("base\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    base_commit = _git(root, "rev-parse", "HEAD")

    _copy_real(root, e9.E9_MODULE_RELPATH)
    _copy_real(root, e9.E9_TEST_RELPATH)
    for rel in e9.PROTECTED_HISTORICAL_RELPATHS:
        _copy_real(root, rel)
    _write(root, f"{e9.OUTPUT_SUBTREE}/E9_PROTOCOL_LOCK.json",
           {"e9_protocol_amendment_identity": "a" * 64, "retained_count": 320})
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "e9 implementation + tests + protocol")
    implementation_source_commit = _git(root, "rev-parse", "HEAD")

    if not with_execution_inputs:
        return root, base_commit, implementation_source_commit, implementation_source_commit

    _write(root, e9.POOL_BINDING_RELPATH,
           {"all_pools_materialized": True,
            "arms": {a: {"status": "MATERIALIZED",
                         "pool_identity": e9.C3_ELIGIBLE_POOL_IDENTITY[a]}
                     for a in ARMS}})
    for a in ARMS:
        p = root / e9.POOL_INPUT_RELPATHS[a]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'{{"arm": "{a}", "n": 384}}\n', encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "e9 deterministic input artifacts")
    execution_commit = _git(root, "rev-parse", "HEAD")

    return root, base_commit, implementation_source_commit, execution_commit


# --------------------------------------------------------------------------- #
# 1. exact frozen seeds
# --------------------------------------------------------------------------- #

def test_frozen_perturbation_seeds_exact():
    assert e9.PERTURBATIONS == {"P1": 20260921, "P2": 20260922, "P3": 20260923}


def test_frozen_arms_exact():
    assert e9.ARMS == ("RND", "DET", "LLM")


# --------------------------------------------------------------------------- #
# 2. exact 384 / 320 / 64 / 256 contracts
# --------------------------------------------------------------------------- #

def test_fixed_counts_and_slack():
    assert e9.ORIGINAL_POOL_COUNT == 384
    assert e9.RETAINED_COUNT == 320
    assert e9.MASKED_COUNT == 64
    assert e9.SELECTED_COUNT == 256
    assert e9.RETAINED_COUNT - e9.SELECTED_COUNT == 64
    assert (e9.RETAINED_FRACTION_NUM, e9.RETAINED_FRACTION_DEN) == (320, 384)


def _ids(n: int, *, prefix: str = "cand", salt: str = "") -> list[str]:
    # 64-hex canonical-looking ids
    return [hashlib.sha256(f"{prefix}|{salt}|{i}".encode()).hexdigest() for i in range(n)]


def test_partition_exact_320_64_and_full_384_manifest():
    ids = _ids(384)
    part = e9.partition_pool("RND", 20260921, ids)
    assert len(part["retained"]) == 320
    assert len(part["masked"]) == 64
    assert len(part["rows"]) == 384
    assert set(part["retained"]).isdisjoint(part["masked"])
    assert set(part["retained"]) | set(part["masked"]) == set(ids)
    assert sum(1 for r in part["rows"] if r["membership"] == "retained") == 320
    assert sum(1 for r in part["rows"] if r["membership"] == "masked") == 64


# --------------------------------------------------------------------------- #
# 3. deterministic hash selection, independent of input row order
# --------------------------------------------------------------------------- #

def test_partition_is_row_order_independent():
    ids = _ids(384, salt="order")
    a = e9.partition_pool("DET", 20260922, ids)
    b = e9.partition_pool("DET", 20260922, list(reversed(ids)))
    import random

    shuffled = list(ids)
    random.Random(99).shuffle(shuffled)
    c = e9.partition_pool("DET", 20260922, shuffled)
    assert a["retained"] == b["retained"] == c["retained"]
    assert a["masked"] == b["masked"] == c["masked"]
    assert a["membership_identity"] == b["membership_identity"] == c["membership_identity"]


def test_perturbation_string_and_digest_are_the_versioned_contract():
    s = e9.perturbation_string("LLM", 20260923, "abc123")
    assert s == "E9-POOL-PERTURB-v1|LLM|20260923|abc123"
    assert e9.perturbation_digest("LLM", 20260923, "abc123") == \
        hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 4. same perturbation algorithm for all arms
# --------------------------------------------------------------------------- #

def test_same_algorithm_all_arms_only_arm_token_differs():
    ids = _ids(384, salt="arm")
    parts = {arm: e9.partition_pool(arm, 20260921, ids) for arm in ARMS}
    # Each arm uses the identical rule; only the arm token enters the hash, so
    # the partitions differ but every one is a clean 320/64 split of the SAME pool.
    for arm in ARMS:
        assert len(parts[arm]["retained"]) == 320 and len(parts[arm]["masked"]) == 64
        assert set(parts[arm]["retained"]) | set(parts[arm]["masked"]) == set(ids)
    assert parts["RND"]["retained"] != parts["DET"]["retained"]
    assert parts["DET"]["retained"] != parts["LLM"]["retained"]


# --------------------------------------------------------------------------- #
# 5. a different seed changes the mask deterministically
# --------------------------------------------------------------------------- #

def test_different_seed_changes_mask_deterministically():
    ids = _ids(384, salt="seed")
    p1 = e9.partition_pool("RND", 20260921, ids)
    p2 = e9.partition_pool("RND", 20260922, ids)
    p3 = e9.partition_pool("RND", 20260923, ids)
    assert p1["masked"] != p2["masked"]
    assert p2["masked"] != p3["masked"]
    # deterministic: same seed -> same mask, byte for byte
    assert e9.partition_pool("RND", 20260921, ids)["masked"] == p1["masked"]


# --------------------------------------------------------------------------- #
# 6. no duplicate candidate ids / wrong pool count fails closed
# --------------------------------------------------------------------------- #

def test_wrong_pool_count_fails_closed():
    with pytest.raises(e9.E9Blocked):
        e9.partition_pool("RND", 20260921, _ids(383))
    with pytest.raises(e9.E9Blocked):
        e9.partition_pool("RND", 20260921, _ids(385))


def test_duplicate_candidate_ids_fail_closed():
    ids = _ids(383) + [_ids(1)[0]]  # 384 entries, one duplicate
    ids[-1] = ids[0]
    with pytest.raises(e9.E9Blocked):
        e9.partition_pool("RND", 20260921, ids)


# --------------------------------------------------------------------------- #
# 7. selector is resolved and reused, never reimplemented
# --------------------------------------------------------------------------- #

def test_selector_provenance_points_at_frozen_selector():
    prov = e9.selector_provenance()
    assert prov["selector_callable"] == "prism_fas.recipes.selection.select"
    assert prov["selection_version"] == "prism_c3_selection_v1"
    assert prov["final_bank_size_per_arm"] == 256
    assert prov["minimum_eligible_pool_per_arm"] == 320
    assert prov["raw_candidate_slots_per_arm"] == 384
    assert prov["reused_not_reimplemented"] is True
    assert len(prov["selector_module_sha256"]) == 64


def test_module_has_exactly_one_selector_call_site():
    src = (REPO / "src/prism_fas/evaluation/c_ext_e9_bank_robustness.py").read_text()
    tree = ast.parse(src)
    imported_select = any(
        isinstance(n, ast.ImportFrom) and n.module == "prism_fas.recipes.selection"
        and any(a.name == "select" for a in n.names)
        for n in ast.walk(tree)
    )
    assert imported_select
    # exactly one call to the bare name `select(...)`
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "select"]
    assert len(calls) == 1
    # and no home-grown MILP / objective symbols
    assert "scipy" not in src and "milp(" not in src
    assert "S_pref" not in src or "s_pref" not in src.lower().split("objective")[0]


# --------------------------------------------------------------------------- #
# 8. selector infeasible -> terminal BLOCKED, never retried, never relaxed
# --------------------------------------------------------------------------- #

def _fake_pool(n=384):
    return [{"candidate_id": cid, "recipe_id": f"R-{i:06d}", "recipe": object()}
            for i, cid in enumerate(_ids(n, salt="fake"))]


def test_infeasible_realization_is_terminal_blocked(monkeypatch):
    from prism_fas.recipes.selection import SelectionInfeasible

    def boom(recipes, ontology, *, arm="LLM"):
        raise SelectionInfeasible("no 256-subset satisfies the hard coverage constraints")

    monkeypatch.setattr(e9, "partition_pool",
                        lambda a, s, ids: {"retained": list(ids)[:320],
                                           "masked": list(ids)[320:],
                                           "membership_identity": "x"})
    monkeypatch.setattr("prism_fas.recipes.selection.select", boom)
    row = e9.run_realization("LLM", "P1", 20260921, _fake_pool(), ontology=None)
    assert row["status"] == "BLOCKED"
    assert row["solver_status"] == "INFEASIBLE"
    assert row["selected_count"] == 0
    assert row["selected_bank_identity"] is None
    assert "infeasible" in row["blocked_reason"].lower()
    assert row["selector_trace"]["infeasible_reason"]


def test_selector_returning_not_256_is_blocked_no_relaxation(monkeypatch):
    class FakeResult:
        selected_shas = [f"h{i}" for i in range(255)]  # 255, not 256
        rejected_shas = []
        s_pref = s_single = s_multi = 0
        counts = {}
        eligible_count = 320
        tie_break_trace = []
        selected_set_identity = "deadbeef"

    monkeypatch.setattr(e9, "partition_pool",
                        lambda a, s, ids: {"retained": list(ids)[:320],
                                           "masked": list(ids)[320:],
                                           "membership_identity": "x"})
    monkeypatch.setattr("prism_fas.recipes.selection.select",
                        lambda *a, **k: FakeResult())
    row = e9.run_realization("RND", "P2", 20260922, _fake_pool(), ontology=None)
    assert row["status"] == "BLOCKED"
    assert row["selected_count"] == 255
    assert "never lowered" in row["blocked_reason"]


# --------------------------------------------------------------------------- #
# 9. descriptive metrics
# --------------------------------------------------------------------------- #

def test_stability_metrics_are_descriptive_only():
    ref = [f"r{i}" for i in range(256)]
    realized = {
        "P1": ref[:200] + [f"x{i}" for i in range(56)],   # 200 overlap
        "P2": list(ref),                                   # identical
        "P3": [],                                          # blocked realization
    }
    s = e9.stability_for_arm("RND", ref, realized)
    assert s["per_realization_vs_reference"]["P1"]["intersection"] == 200
    assert s["per_realization_vs_reference"]["P1"]["reference_retention_fraction"] == 200 / 256
    assert s["per_realization_vs_reference"]["P2"]["jaccard"] == 1.0
    assert s["per_realization_vs_reference"]["P3"]["selected_count"] == 0
    assert s["descriptive_jaccard_vs_reference"]["n_non_blocked"] == 2
    assert set(s["pairwise_between_realizations"]) == {"P1|P2", "P1|P3", "P2|P3"}
    # no p-value / significance key anywhere
    blob = json.dumps(s).lower()
    assert "p_value" not in blob and "pvalue" not in blob and "significan" not in blob


# --------------------------------------------------------------------------- #
# 10. idempotent lock semantics
# --------------------------------------------------------------------------- #

def test_protocol_amendment_identity_is_stable():
    a1 = e9.build_protocol_amendment()
    a2 = e9.build_protocol_amendment()
    assert e9._amendment_identity(a1) == e9._amendment_identity(a2)
    lock = e9.build_protocol_lock(a1)
    assert lock["e9_protocol_amendment_identity"] == e9._amendment_identity(a1)
    assert lock["retained_count"] == 320 and lock["masked_count"] == 64
    assert lock["selected_count"] == 256 and lock["original_pool_count"] == 384


def test_write_idempotent_noop_then_fail_closed_on_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(e9, "write_text_atomic",
                        lambda p, t, **k: Path(p).write_text(t, encoding="utf-8"))
    rel = "reports/c_ext_q1q2_v1/e9_bank_robustness/E9_PROTOCOL_LOCK.json"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    obj = {"a": 1, "b": [1, 2, 3]}
    r1 = e9._write_idempotent(tmp_path, rel, obj)
    assert r1["action"] == "written"
    r2 = e9._write_idempotent(tmp_path, rel, obj)
    assert r2["action"] == "idempotent_noop"
    with pytest.raises(e9.E9Error):
        e9._write_idempotent(tmp_path, rel, {"a": 2})


def test_protocol_amendment_discloses_no_target_or_e8_input():
    a = e9.build_protocol_amendment()
    d = a["disclosure"]
    assert d["not_numerically_specified_in_v1_0"] is True
    assert d["frozen_before_any_e9_perturbation_or_selection_outcome"] is True
    assert d["no_e8_acer_used"] is True
    assert d["no_target_labels_used"] is True
    assert d["no_target_predictions_used"] is True
    assert d["no_target_metrics_used"] is True
    assert d["no_e9_result_used"] is True
    assert d["selected_from_e8_target_performance"] is False


# --------------------------------------------------------------------------- #
# 11. no target-label / target-feature / LLM / network dependency
# --------------------------------------------------------------------------- #

def test_module_imports_no_forbidden_surface():
    imported = e9._module_imports(REPO / "src/prism_fas/evaluation/c_ext_e9_bank_robustness.py")
    forbidden = ("torch", "google.genai", "google.generativeai", "genai",
                 "requests", "urllib.request", "socket", "modal",
                 "prism_fas.detector", "prism_fas.data",
                 "prism_fas.evaluation.firewall",
                 "prism_fas.evaluation.target_prediction")
    for name in imported:
        assert not any(name == f or name.startswith(f + ".") for f in forbidden), name


def test_module_code_references_no_target_or_siw_identifier():
    """Prose/disclosure strings may say 'no E8 ACER / target metric' etc.; the
    executable code must not name any such identifier or attribute."""
    tree = ast.parse((REPO / "src/prism_fas/evaluation/c_ext_e9_bank_robustness.py").read_text())
    banned = ("acer", "siw", "target_label", "target_prediction", "target_metric",
              "target_score", "resolve_target", "label_reveal")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            low = node.id.lower()
            assert not any(b in low for b in banned), node.id
        if isinstance(node, ast.Attribute):
            low = node.attr.lower()
            assert not any(b in low for b in banned), node.attr


def test_run_realization_carries_no_target_no_llm_flags():
    from prism_fas.recipes.selection import SelectionInfeasible

    def boom(*a, **k):
        raise SelectionInfeasible("infeasible")

    import types
    # monkeypatch-free: patch attribute then restore
    orig = sys.modules["prism_fas.recipes.selection"].select
    sys.modules["prism_fas.recipes.selection"].select = boom
    try:
        e9.partition_pool_backup = e9.partition_pool
        e9.partition_pool = lambda a, s, ids: {"retained": list(ids)[:320],
                                               "masked": list(ids)[320:],
                                               "membership_identity": "x"}
        row = e9.run_realization("LLM", "P1", 20260921, _fake_pool(), ontology=None)
    finally:
        sys.modules["prism_fas.recipes.selection"].select = orig
        e9.partition_pool = e9.partition_pool_backup
    assert row["target_access"] is False
    assert row["target_labels_accessed"] is False
    assert row["target_features_accessed"] is False
    assert row["llm_calls"] == 0


# --------------------------------------------------------------------------- #
# 12. writes are confined; protected Flow1/Flow2 + C3 artifacts unchanged
# --------------------------------------------------------------------------- #

PROTECTED = [
    "assets/recipe_banks/c3/rnd/recipes.jsonl",
    "assets/recipe_banks/c3/det/recipes.jsonl",
    "assets/recipe_banks/c3/llm/recipes.jsonl",
    "assets/recipe_banks/c3/rnd/C3_BANK.json",
    "assets/recipe_banks/c3/det/C3_BANK.json",
    "assets/recipe_banks/c3/llm/C3_BANK.json",
    "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json",
    "src/prism_fas/recipes/selection.py",
]


def _hashes():
    out = {}
    for rel in PROTECTED:
        p = REPO / rel
        if p.exists():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_preflight_does_not_mutate_protected_artifacts():
    before = _hashes()
    e9.preflight(REPO)
    assert _hashes() == before


def test_materialize_does_not_mutate_protected_artifacts():
    before = _hashes()
    e9.materialize_frozen_pools(REPO)
    assert _hashes() == before


def test_e9_write_paths_are_confined_to_the_e9_subtree():
    from prism_fas.evaluation.c_ext_common import assert_ext_write_path

    ok = assert_ext_write_path(
        REPO / "reports/c_ext_q1q2_v1/e9_bank_robustness/E9_PREFLIGHT.json",
        must_be_under="reports/c_ext_q1q2_v1/e9_bank_robustness",
    )
    assert ok.startswith("reports/c_ext_q1q2_v1/e9_bank_robustness/")
    for bad in ("assets/recipe_banks/c3/rnd/recipes.jsonl",
                "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json",
                "reports/full/c3/C3_FULL.json"):
        with pytest.raises(ExtPathSafetyError):
            assert_ext_write_path(REPO / bad)


# --------------------------------------------------------------------------- #
# 13. objective / selector provenance preserved on a successful realization
# --------------------------------------------------------------------------- #

def test_successful_realization_preserves_objective_and_trace(monkeypatch):
    class FakeResult:
        selected_shas = [f"h{i:064d}" for i in range(256)]
        rejected_shas = [f"j{i:063d}" for i in range(64)]
        s_pref, s_single, s_multi = 3, 7, 11
        counts = {"medium": {"paper-like": 51}}
        eligible_count = 320
        tie_break_trace = [{"index": 0, "decision": "include", "sha": "h" * 64}]
        selected_set_identity = "f" * 64

    monkeypatch.setattr(e9, "partition_pool",
                        lambda a, s, ids: {"retained": list(ids)[:320],
                                           "masked": list(ids)[320:],
                                           "membership_identity": "mid"})
    monkeypatch.setattr("prism_fas.recipes.selection.select",
                        lambda *a, **k: FakeResult())
    row = e9.run_realization("DET", "P3", 20260923, _fake_pool(), ontology=None)
    assert row["status"] == "OK"
    assert row["solver_status"] == "SELECTED_256_VERIFIED"
    assert row["selected_count"] == 256
    assert row["selected_bank_identity"] == "f" * 64
    assert row["objective"]["S_pref"] == 3
    assert row["objective"]["S_single"] == 7
    assert row["objective"]["S_multi"] == 11
    assert row["objective"]["counts"] == {"medium": {"paper-like": 51}}
    assert row["selector_trace"]["tie_break_trace"]
    assert row["selector_trace"]["rejected_count"] == 64
    assert row["selector"]["selection_version"] == "prism_c3_selection_v1"
    assert row["perturbation_seed"] == 20260923


# --------------------------------------------------------------------------- #
# 14. RND/DET reconstruction fidelity + a real frozen-selector realization
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ["RND", "DET"])
def test_reconstructed_pool_identity_equals_c3_eligible_pool_identity(arm):
    recon = e9._reconstruct_control_pool(REPO, arm)
    from prism_fas.recipes.canonical import recipe_hash
    from prism_fas.evaluation.c_ext_common import sha256_json

    ids = sorted(recipe_hash(r) for r in recon["recipes"])
    assert len(ids) == 384
    bank = json.loads((REPO / e9.C3_BANK_RELPATHS[arm]).read_text())
    assert sha256_json(ids) == bank["eligible_pool_identity"]
    assert recon["provenance"]["schedule_identity"] == \
        recon["provenance"]["frozen_schedule_identity"]


def test_llm_reconstruction_blocks_without_raw_archives():
    raw = REPO / "reports/c3/live/raw_responses"
    if raw.exists() and len(list(raw.glob("c3-llm-req-*.json"))) == 12:
        pytest.skip("LLM raw archives present -- the BLOCKED path is not exercised here")
    with pytest.raises(e9.E9Blocked):
        e9._reconstruct_llm_pool(REPO)


@pytest.mark.slow
def test_real_frozen_selector_realization_rnd():
    # OPT-IN ONLY. Running this executes one of the nine E9 scientific
    # perturbation-selections through the real MILP; the E9 milestone forbids
    # that until the operator has reviewed the preflight and authorized it.
    import os

    if os.environ.get("PRISM_E9_ALLOW_REAL_REALIZATION") != "1":
        pytest.skip("set PRISM_E9_ALLOW_REAL_REALIZATION=1 to run one real E9 realization")
    pool_file = REPO / e9.POOL_INPUT_RELPATHS["RND"]
    if not pool_file.exists():
        pytest.skip("run --materialize-frozen-pools first")
    from prism_fas.recipes.ontology import load_ontology

    ontology = load_ontology(REPO / e9.ONTOLOGY_RELPATH)
    pool = e9.load_frozen_pool(pool_file)
    assert len(pool) == 384
    row = e9.run_realization("RND", "P1", 20260921, pool, ontology)
    # Either outcome is scientifically valid for E9; the invariants are not.
    assert row["retained_count"] == 320 and row["masked_count"] == 64
    assert row["status"] in ("OK", "BLOCKED")
    if row["status"] == "OK":
        assert row["selected_count"] == 256
        assert set(row["selected_candidate_ids"]) <= {c["candidate_id"] for c in pool}
        row2 = e9.run_realization("RND", "P1", 20260921, pool, ontology)
        assert row2["selected_candidate_ids"] == row["selected_candidate_ids"]
    else:
        assert "never lowered" in (row["blocked_reason"] or "") or \
               "infeasible" in (row["blocked_reason"] or "").lower()


# --------------------------------------------------------------------------- #
# 15. preflight reports BLOCKED (not PASS) until every frozen pool is proven
# --------------------------------------------------------------------------- #

def test_preflight_status_and_no_science_side_effects():
    body = e9.preflight(REPO)
    assert body["status"] in ("PASS", "BLOCKED")
    assert body["runs_milp"] is False
    assert body["creates_perturbation_selection_outcomes"] is False
    assert body["writes_scientific_selected_banks"] is False
    assert body["llm_calls"] == 0
    assert body["target_access"] is False
    ids = {c["check_id"] for c in body["checks"]}
    assert {"e9_base_commit_binding", "e9_code_provenance", "e9_perturbation_seeds",
            "e9_fixed_counts", "e9_selector_resolved", "e9_ontology_identity",
            "e9_reference_256_bank_identities", "e9_frozen_384_pools",
            "e9_no_target_or_llm_dependency", "e9_protocol_lock"} <= ids
    # no E9 selection-results / stability / closure artifact is produced by preflight
    for never in ("E9_SELECTION_RESULTS.jsonl", "E9_BANK_STABILITY.json",
                  "E9_CLOSURE.json"):
        assert not (REPO / "reports/c_ext_q1q2_v1/e9_bank_robustness" / never).exists()


# --------------------------------------------------------------------------- #
# 16. base_commit / implementation_source_commit / execution_commit are three
#     distinct, non-conflated facts
# --------------------------------------------------------------------------- #

def test_base_commit_constant_is_permanently_pinned():
    assert e9.E9_BASE_COMMIT == "6f0642a1d05c35e4c1778d329fb547f1b22a4115"


def test_preflight_no_longer_requires_head_to_equal_base_commit():
    body = e9.preflight(REPO)
    ids = {c["check_id"] for c in body["checks"]}
    assert "e9_base_commit" not in ids                      # retired check gone
    bc = next(c for c in body["checks"] if c["check_id"] == "e9_base_commit_binding")
    assert bc["detail"]["base_commit"] == "6f0642a1d05c35e4c1778d329fb547f1b22a4115"
    assert bc["ok"] is True                                 # pinned + in history, NOT HEAD==base
    assert "base_commit" in body and "execution_commit" in body
    assert "implementation_source_commit" in body


def test_code_provenance_reports_three_commits_and_uncommitted_state_here():
    prov = e9.code_provenance(REPO)
    assert prov["base_commit"] == "6f0642a1d05c35e4c1778d329fb547f1b22a4115"
    assert set(("base_commit", "implementation_source_commit", "execution_commit")) <= set(prov)
    assert "implementation_commit" not in prov              # the conflated field is gone
    assert prov["implementation_commit_status"] == e9.IMPL_UNCOMMITTED_REVIEW_STATE
    assert prov["implementation_source_commit"] is None     # module untracked here
    assert prov["scientific_code_tracked_and_clean"] is False
    assert prov["scientific_execution_allowed_from_this_state"] is False
    # selector identity independently unchanged
    assert prov["selector_module"]["matches_frozen_selector_sha256"] is True
    assert prov["selector_module"]["clean_vs_head"] is True


def test_selector_module_sha256_is_the_frozen_value():
    assert e9.SELECTOR_MODULE_SHA256 == \
        hashlib.sha256((REPO / e9.SELECTOR_MODULE_RELPATH).read_bytes()).hexdigest()
    assert e9.selector_provenance()["selector_module_sha256"] == e9.SELECTOR_MODULE_SHA256


def test_frozen_c3_eligible_pool_identities_are_the_locked_values():
    assert e9.C3_ELIGIBLE_POOL_IDENTITY["LLM"] == \
        "4032a7f8708a27d1545a84277d2b439767ae253c04113f3812ac30c16255c978"
    lock = json.loads((REPO / e9.C3_SCIENTIFIC_BANK_LOCK_RELPATH).read_text())
    for a in ARMS:
        assert e9.C3_ELIGIBLE_POOL_IDENTITY[a] == lock["arms"][a]["eligible_pool_identity"]


# --------------------------------------------------------------------------- #
# 17. execution-binding lifecycle (corrected: no self-reference)
# --------------------------------------------------------------------------- #

def test_untracked_e9_implementation_cannot_execute_scientific_run():
    # REPO here has the E9 module untracked (UNCOMMITTED_REVIEW_STATE).
    with pytest.raises(e9.E9Blocked) as exc:
        e9.run_bank_perturbation(REPO, authorized=True)
    assert "UNTRACKED" in str(exc.value) or "absent" in str(exc.value)
    for never in ("E9_SELECTION_RESULTS.jsonl", "E9_CLOSURE.json",
                  "E9_PERTURBATION_MEMBERSHIP.jsonl"):
        assert not (REPO / e9.OUTPUT_SUBTREE / never).exists()


def test_freeze_execution_binding_refuses_uncommitted_review_state():
    with pytest.raises(e9.E9Blocked) as exc:
        e9.freeze_execution_binding(REPO)
    assert "COMMITTED_CLEAN" in str(exc.value)


def test_1_execution_binding_at_final_execution_commit_validates(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)

    prov = e9.code_provenance(root)
    assert prov["implementation_commit_status"] == e9.IMPL_COMMITTED_CLEAN
    assert prov["implementation_source_commit"] == impl_src
    assert prov["execution_commit"] == exec_c
    assert impl_src != exec_c                                # execution is a child
    assert prov["base_commit_is_ancestor_of_head"] is True
    assert prov["implementation_source_commit_is_ancestor_of_head"] is True
    assert prov["head_is_base_commit"] is False
    assert prov["scientific_execution_allowed_from_this_state"] is True

    frozen = e9.freeze_execution_binding(root)
    assert frozen["execution_commit"] == exec_c
    assert frozen["implementation_source_commit"] == impl_src
    binding = json.loads((root / e9.EXECUTION_BINDING_RELPATH).read_text())
    assert binding["schema_version"] == "e9-execution-binding-v2"
    assert binding["base_commit"] == base
    assert binding["implementation_source_commit"] == impl_src
    assert binding["execution_commit"] == exec_c
    assert binding["selector_module_sha256"] == e9.SELECTOR_MODULE_SHA256
    assert binding["base_commit_is_ancestor_of_execution_commit"] is True
    assert binding["pool_identity_by_arm"] == dict(e9.C3_ELIGIBLE_POOL_IDENTITY)

    gate = e9.verify_execution_binding(root)
    assert gate["ok"] is True, gate["failures"]


def test_2_binding_file_need_not_be_tracked_before_run(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    e9.freeze_execution_binding(root)
    # the binding file is deliberately NOT committed
    porcelain = _git(root, "status", "--porcelain", "--", e9.EXECUTION_BINDING_RELPATH)
    assert porcelain.strip().startswith("??")               # untracked
    assert _git(root, "rev-parse", "HEAD") == exec_c         # HEAD did not move
    gate = e9.verify_execution_binding(root)
    assert gate["ok"] is True, gate["failures"]              # still valid


def test_3_modifying_binding_bytes_after_freeze_fails_closed(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    e9.freeze_execution_binding(root)
    bpath = root / e9.EXECUTION_BINDING_RELPATH
    obj = json.loads(bpath.read_text())
    obj["execution_binding_identity"] = "f" * 64
    bpath.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    gate = e9.verify_execution_binding(root)
    assert gate["ok"] is False
    assert any("recomputation" in f or "execution_binding" in f for f in gate["failures"])
    # a re-freeze with different bytes is refused, never silently rewritten
    with pytest.raises(e9.E9Error):
        e9.freeze_execution_binding(root)


def test_4_moving_head_after_binding_freeze_fails_closed(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    e9.freeze_execution_binding(root)
    (root / "NOTES").write_text("later\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "later commit moves HEAD")
    gate = e9.verify_execution_binding(root)
    assert gate["ok"] is False
    assert any("execution_commit" in f and "HEAD" in f for f in gate["failures"])
    # the frozen binding bytes are unchanged: it still records the original commit
    binding = json.loads((root / e9.EXECUTION_BINDING_RELPATH).read_text())
    assert binding["execution_commit"] == exec_c
    with pytest.raises(e9.E9Blocked):
        e9.run_bank_perturbation(root, authorized=True)


def test_5_full_three_arm_input_binding_mandatory_before_freeze(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    ib = root / e9.POOL_BINDING_RELPATH
    obj = json.loads(ib.read_text())
    obj["arms"].pop("LLM")                                  # drop one arm
    ib.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    with pytest.raises(e9.E9Blocked) as exc:
        e9.freeze_execution_binding(root)
    assert "three-arm" in str(exc.value) or "MATERIALIZED" in str(exc.value)


def test_6_missing_llm_pool_prevents_freeze(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    (root / e9.POOL_INPUT_RELPATHS["LLM"]).unlink()         # remove the LLM pool file
    with pytest.raises(e9.E9Blocked) as exc:
        e9.freeze_execution_binding(root)
    assert "input artifacts" in str(exc.value) or "LLM_POOL_384" in str(exc.value)


def test_7_input_binding_drift_after_freeze_fails_closed(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    e9.freeze_execution_binding(root)
    ib = root / e9.POOL_BINDING_RELPATH
    obj = json.loads(ib.read_text())
    obj["arms"]["RND"]["pool_identity"] = "e" * 64          # drift a pool_identity
    ib.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    gate = e9.verify_execution_binding(root)
    assert gate["ok"] is False
    assert any("INPUT_BINDING" in f or "pool_identity" in f or "eligible_pool_identity" in f
               for f in gate["failures"])


def test_8_scientific_code_must_be_tracked_and_clean(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    e9.freeze_execution_binding(root)
    mod = root / e9.E9_MODULE_RELPATH
    mod.write_text(mod.read_text() + "\n# tampered scientific code\n", encoding="utf-8")
    prov = e9.code_provenance(root)
    assert prov["implementation_commit_status"] == e9.IMPL_DIRTY_TRACKED
    assert prov["scientific_code_tracked_and_clean"] is False
    gate = e9.verify_execution_binding(root)
    assert gate["ok"] is False
    assert any("differs from HEAD" in f or "tracked+clean" in f for f in gate["failures"])
    with pytest.raises(e9.E9Blocked):
        e9.run_bank_perturbation(root, authorized=True)


def test_8b_selector_sha_mismatch_fails_closed(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    e9.freeze_execution_binding(root)
    sel = root / e9.SELECTOR_MODULE_RELPATH
    sel.write_text(sel.read_text() + "\n# selector tamper\n", encoding="utf-8")
    gate = e9.verify_execution_binding(root)
    assert gate["ok"] is False
    assert any("selector" in f.lower() for f in gate["failures"])


def test_8c_e9_module_sha_mismatch_vs_binding_fails_closed(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    e9.freeze_execution_binding(root)
    bpath = root / e9.EXECUTION_BINDING_RELPATH
    obj = json.loads(bpath.read_text())
    obj["e9_module_sha256"] = "f" * 64
    bpath.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    gate = e9.verify_execution_binding(root)
    assert gate["ok"] is False
    assert any("e9_module_sha256" in f or "recomputation" in f for f in gate["failures"])


def test_8d_base_commit_binding_mismatch_fails_closed(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    e9.freeze_execution_binding(root)
    bpath = root / e9.EXECUTION_BINDING_RELPATH
    obj = json.loads(bpath.read_text())
    obj["base_commit"] = "0" * 40
    bpath.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    gate = e9.verify_execution_binding(root)
    assert gate["ok"] is False
    assert any("base_commit" in f for f in gate["failures"])


def test_9_input_artifacts_must_be_tracked_and_clean_at_execution_commit(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    # untracked case: build a repo whose input artifacts were never committed
    root2, base2, impl2, _ = _committed_repo(tmp_path / "b", with_execution_inputs=False)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base2)
    (root2 / e9.POOL_BINDING_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    _write(root2, e9.POOL_BINDING_RELPATH,
           {"all_pools_materialized": True,
            "arms": {a: {"status": "MATERIALIZED",
                         "pool_identity": e9.C3_ELIGIBLE_POOL_IDENTITY[a]} for a in ARMS}})
    for a in ARMS:
        p = root2 / e9.POOL_INPUT_RELPATHS[a]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n", encoding="utf-8")
    with pytest.raises(e9.E9Blocked) as exc:
        e9.freeze_execution_binding(root2)                  # inputs untracked -> refuse
    assert "input artifacts" in str(exc.value)

    # dirty case: freeze cleanly, then modify a committed frozen pool file
    e9.freeze_execution_binding(root)
    victim = root / e9.POOL_INPUT_RELPATHS["DET"]
    victim.write_text(victim.read_text() + "\n", encoding="utf-8")
    gate = e9.verify_execution_binding(root)
    assert gate["ok"] is False
    assert any("input artifacts" in f for f in gate["failures"])


def _wire_fake_run(monkeypatch, root):
    """Wire deterministic, NON-scientific stand-ins so run_bank_perturbation
    exercises the archive/idempotency/closure lifecycle with NO real MILP and no
    real pool. The real verify_execution_binding gate still runs against `root`."""
    fake_pool = [{"candidate_id": f"{i:064x}", "recipe_id": f"R-{i:06d}", "recipe": object()}
                 for i in range(384)]
    monkeypatch.setattr(e9, "load_frozen_pool", lambda p: list(fake_pool))
    monkeypatch.setattr(e9, "load_reference_identities", lambda repo: {
        "arms": {a: {"reference_selected_set_identity": "s" * 64,
                     "reference_eligible": 384, "reference_selected": 256,
                     "reference_selected_recipe_identities": [f"{i:064x}" for i in range(256)]}
                 for a in ARMS}})
    monkeypatch.setattr("prism_fas.recipes.ontology.load_ontology", lambda p: object())
    # bypass the engineering preflight (its frozen-pool checks need real 384
    # recipe files); the provenance gate under test is verify_execution_binding.
    monkeypatch.setattr(e9, "preflight", lambda repo: {
        "status": "PASS", "scientific_execution_ready": True,
        "scientific_execution_blockers": [],
        "protocol": {"e9_protocol_amendment_identity": "a" * 64}})

    def _fake_realization(arm, pid, seed, pool, ontology):
        part = e9.partition_pool(arm, seed, [c["candidate_id"] for c in pool])
        return {"arm": arm, "perturbation_id": pid, "perturbation_seed": seed,
                "status": "OK", "solver_status": "SELECTED_256_VERIFIED",
                "retained_count": 320, "masked_count": 64, "selected_count": 256,
                "selected_candidate_ids": part["retained"][:256],
                "selected_bank_identity": "b" * 64,
                "objective": {"S_pref": 0, "S_single": 0, "S_multi": 0, "counts": {}},
                "selector_trace": {"tie_break_trace": []},
                "selector": e9.selector_provenance(), "blocked_reason": None,
                "target_access": False, "target_labels_accessed": False,
                "target_features_accessed": False, "llm_calls": 0}
    monkeypatch.setattr(e9, "run_realization", _fake_realization)


def test_10_post_run_archive_commit_does_not_redefine_execution_commit(tmp_path, monkeypatch):
    """A synthetic run (fake selector, fake pools) writes a closure pinned to
    execution_commit; a later evidence-archive commit must not move it or allow
    a rerun."""
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    e9.freeze_execution_binding(root)
    _wire_fake_run(monkeypatch, root)

    e9.run_bank_perturbation(root, authorized=True)
    closure = json.loads((root / e9.CLOSURE_RELPATH).read_text())
    assert closure["execution_commit"] == exec_c
    assert closure["schema_version"] == "e9-closure-v3"
    for k in ("base_commit", "implementation_source_commit", "execution_commit",
              "e9_module_sha256", "selector_module_sha256", "protocol_lock_identity",
              "input_binding_identity", "execution_binding_identity",
              "execution_binding_file_sha256"):
        assert k in closure
    assert closure["target_access"] is False and closure["llm_calls"] == 0

    # archive the evidence in a LATER additive commit
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "additive E9 evidence archive")
    assert _git(root, "rev-parse", "HEAD") != exec_c

    # closure still records the ORIGINAL execution_commit
    closure2 = json.loads((root / e9.CLOSURE_RELPATH).read_text())
    assert closure2["execution_commit"] == exec_c
    # and a rerun is refused because HEAD != execution_commit
    with pytest.raises(e9.E9Blocked):
        e9.run_bank_perturbation(root, authorized=True)


def test_10b_idempotent_rerun_at_same_execution_commit_is_noop(tmp_path, monkeypatch):
    root, base, impl_src, exec_c = _committed_repo(tmp_path)
    monkeypatch.setattr(e9, "E9_BASE_COMMIT", base)
    e9.freeze_execution_binding(root)
    _wire_fake_run(monkeypatch, root)
    e9.run_bank_perturbation(root, authorized=True)
    again = e9.run_bank_perturbation(root, authorized=True)
    assert again.get("idempotent_noop") is True


def test_11_no_real_scientific_realization_ran():
    # real working tree: no scientific-outcome artifact was produced by any test
    for never in ("E9_SELECTION_RESULTS.jsonl", "E9_BANK_STABILITY.json",
                  "E9_BANK_STABILITY.csv", "E9_CLOSURE.json", "E9_EVIDENCE.sha256",
                  "E9_PERTURBATION_MEMBERSHIP.jsonl", "E9_EXECUTION_BINDING.json"):
        assert not (REPO / e9.OUTPUT_SUBTREE / never).exists()
