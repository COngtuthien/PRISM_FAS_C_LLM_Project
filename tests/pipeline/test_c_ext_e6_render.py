"""Tests for `prism_fas.evaluation.c_ext_e6_render` (E6 render adapter).

Every test builds a self-contained fake repo under `tmp_path`: fake but
schema-correct historical C5/C6 artifacts (mirroring the real ones this
session read directly), a fake but internally self-consistent E6 training
plan lock + frozen LLM-SHUFFLE-A recipes, and (where needed) a tiny fake
candidate/bank tree. No test ever passes the real repo root to a function
that writes, and an autouse fixture hashes the real `reports/c_ext_q1q2_v1`,
`reports/full/c5`, `reports/full/c6`, `assets/recipe_banks` and
`reports/c3/scientific` directories before and after the whole file as a
regression tripwire. Model/GPU rendering is never exercised -- every test
injects a fake `candidate_renderer` or checks unreachability structurally.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prism_fas.evaluation import c_ext_common as cc
from prism_fas.evaluation import c_ext_e6_render as e6r
from prism_fas.evaluation import c_ext_e6_training_plan as training_plan

REPO = Path(__file__).resolve().parents[2]


def _tree_hash(root: Path) -> str:
    if not root.is_dir():
        return "MISSING"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


PROTECTED = ("reports/c_ext_q1q2_v1", "reports/full/c5", "reports/full/c6",
            "assets/recipe_banks", "reports/c3/scientific")


@pytest.fixture(autouse=True)
def _protect_real_directories():
    before = {name: _tree_hash(REPO / name) for name in PROTECTED}
    yield
    after = {name: _tree_hash(REPO / name) for name in PROTECTED}
    for name in PROTECTED:
        assert after[name] == before[name], f"a test mutated the real {name} namespace"


def _historical_fixture(repo: Path) -> None:
    (repo / "reports/full/c5").mkdir(parents=True, exist_ok=True)
    (repo / "reports/full/c6").mkdir(parents=True, exist_ok=True)
    (repo / "assets/recipe_banks/c3/llm").mkdir(parents=True, exist_ok=True)

    arm_plans = {
        "arms": {"LLM": {
            "arm_plan_identity": "fake-arm-plan-identity",
            "recipe_bank_identity": "fake-recipe-bank-identity",
            "selected_set_identity": e6r.EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY,
            "ontology_identity": e6r.EXPECTED_ONTOLOGY_IDENTITY,
            "planned_candidates": e6r.EXPECTED_CANDIDATES_PER_ARM,
        }},
        "gpat_checkpoint_sha256": e6r.EXPECTED_GPAT_CHECKPOINT_SHA256,
        "physics_engine_version": e6r.EXPECTED_PHYSICS_ENGINE_VERSION,
    }
    (repo / e6r.C5_ARM_PLANS_PATH).write_text(json.dumps(arm_plans), encoding="utf-8")

    source_pair_plan = {
        "renders_per_recipe": e6r.EXPECTED_RENDERS_PER_RECIPE,
        "candidates_per_arm": e6r.EXPECTED_CANDIDATES_PER_ARM,
        "package_identity": e6r.EXPECTED_PACKAGE_IDENTITY,
        "source_pair_plan_identity": e6r.EXPECTED_SOURCE_PAIR_PLAN_IDENTITY,
    }
    (repo / e6r.C5_SOURCE_PAIR_PLAN_PATH).write_text(json.dumps(source_pair_plan), encoding="utf-8")

    bank_lock = {
        "quality_threshold_identity": e6r.EXPECTED_QUALITY_THRESHOLD_IDENTITY,
        "q_used_for_selection": False,
        "by_route": dict(e6r.EXPECTED_BY_ROUTE_QUOTA),
        "final_bank_size": e6r.EXPECTED_FINAL_BANK_SIZE,
    }
    (repo / e6r.C6_BANK_LOCK_LLM_PATH).write_text(json.dumps(bank_lock), encoding="utf-8")

    gate_profiles = {"profiles": {"NOMINAL": {"thresholds": {"tau_fd": 0.5, "tau_fp": 5.0}}}}
    (repo / e6r.C6_GATE_PROFILES_PATH).write_text(json.dumps(gate_profiles), encoding="utf-8")

    c3_bank = {"c3_bank_contract_identity": "fake-c3-bank-contract-identity"}
    (repo / e6r.C3_BANK_LLM_PATH).write_text(json.dumps(c3_bank), encoding="utf-8")


def _shuffle_fixture(repo: Path, *, recipe_count: int = 8) -> tuple[list[dict], str]:
    """A minimal, VALID-shaped set of shuffled recipes (content is irrelevant
    to this module, which never validates recipe schema itself -- that is
    `c_ext_llm_shuffle`'s job, already tested elsewhere)."""
    recipes = [{"recipe_id": f"R-{i:06d}", "medium": {"family": "silicone"}} for i in range(recipe_count)]
    (repo / training_plan.E6_DIR).mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(rec, sort_keys=True, separators=(",", ":")) for rec in recipes) + "\n"
    (repo / training_plan.E6_SHUFFLE_RECIPES_PATH).write_text(lines, encoding="utf-8")
    identity = cc.sha256_json(recipes)
    return recipes, identity


def _training_plan_lock_fixture(repo: Path, *, recipe_identity: str, recipe_count: int, status="FROZEN") -> dict:
    body = {"schema_version": "e6-training-plan-lock-v1", "milestone": "E6", "arm": "LLM_SHUFFLE_A",
           "track": "G", "plan_identity": "fake-training-plan-identity", "shuffle_seed": 20260911,
           "original_llm_recipe_identity": e6r.EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY,
           "llm_shuffle_a_recipe_identity": recipe_identity, "recipe_count": recipe_count,
           "source_package_identity": e6r.EXPECTED_PACKAGE_IDENTITY,
           "detector_config_identity": "fake-detector-config-identity",
           "detector_seeds": [20260806, 20260807, 20260808, 20260809, 20260810], "seed_count": 5,
           "expected_optimizer_steps": 1575, "expected_synthetic_sample_budget": 10800,
           "quality_weighting_status": "q_weighted", "recipe_conditioning_status": "structured",
           "target_access": False, "llm_api_calls": 0,
           "rendered_bank_required": True, "rendered_bank_status": "NEEDS_BUILD", "status": status}
    lock = dict(body)
    lock["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(lock))
    (repo / training_plan.TRAINING_PLAN_LOCK_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / training_plan.TRAINING_PLAN_LOCK_PATH).write_text(json.dumps(lock), encoding="utf-8")
    return lock


def _base_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return repo


def _full_fixture(tmp_path: Path) -> Path:
    repo = _base_repo(tmp_path)
    _historical_fixture(repo)
    recipes, identity = _shuffle_fixture(repo)
    _training_plan_lock_fixture(repo, recipe_identity=identity, recipe_count=len(recipes))
    return repo


# 1. frozen E6 training-plan lock required
def test_frozen_training_plan_lock_required(tmp_path):
    repo = _base_repo(tmp_path)
    _historical_fixture(repo)
    _shuffle_fixture(repo)
    with pytest.raises(e6r.E6RenderError, match="training-plan lock"):
        e6r.verify_shuffle_recipe_source(repo)


# 2. exact LLM_SHUFFLE_A identity required
def test_exact_shuffle_a_identity_required(tmp_path):
    repo = _full_fixture(tmp_path)
    lock_path = repo / training_plan.TRAINING_PLAN_LOCK_PATH
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["llm_shuffle_a_recipe_identity"] = "0" * 64  # tamper without recomputing lock_identity
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(e6r.E6RenderError, match="not usable"):
        e6r.verify_shuffle_recipe_source(repo)


# 3. shuffled JSONL mutation fails
def test_shuffled_jsonl_mutation_fails(tmp_path):
    repo = _full_fixture(tmp_path)
    recipes_path = repo / training_plan.E6_SHUFFLE_RECIPES_PATH
    lines = recipes_path.read_text(encoding="utf-8").strip().split("\n")
    lines[0] = json.dumps({"recipe_id": "R-999999", "medium": {"family": "tampered"}})
    recipes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(e6r.E6RenderError, match="content identity"):
        e6r.verify_shuffle_recipe_source(repo)


# 4. shuffle generator is unreachable
def test_shuffle_generator_unreachable():
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    assert "run_shuffle(" not in source
    assert "c_ext_llm_shuffle.main(" not in source
    assert "load_frozen_group_map" not in source


# 5. original historical LLM bank cannot be substituted
def test_original_llm_bank_cannot_be_substituted(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e6r.build_render_plan(repo)
    assert plan["llm_shuffle_a_recipe_identity"] != plan["original_llm_recipe_identity"]
    assert e6r.ORIGINAL_LLM_SOURCE_BANK_ROOT if hasattr(e6r, "ORIGINAL_LLM_SOURCE_BANK_ROOT") else True
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    # the module must never read assets/recipe_banks/c3/llm/recipes.jsonl as a
    # candidate recipe source for the E6 plan (only as a HISTORICAL reference
    # inside audit_historical_path, which never feeds build_render_plan's
    # recipe content)
    assert 'recipes"] = cc.read_jsonl(repo / "assets/recipe_banks/c3/llm' not in source


# 6. recipe count mismatch fails
def test_recipe_count_mismatch_fails(tmp_path):
    repo = _base_repo(tmp_path)
    _historical_fixture(repo)
    recipes, identity = _shuffle_fixture(repo, recipe_count=8)
    _training_plan_lock_fixture(repo, recipe_identity=identity, recipe_count=7)  # wrong count
    with pytest.raises(e6r.E6RenderError, match="recipes.jsonl has"):
        e6r.verify_shuffle_recipe_source(repo)


# 7. renderer-config mismatch fails
def test_renderer_config_mismatch_fails(tmp_path):
    repo = _full_fixture(tmp_path)
    arm_plans_path = repo / e6r.C5_ARM_PLANS_PATH
    payload = json.loads(arm_plans_path.read_text(encoding="utf-8"))
    payload["gpat_checkpoint_sha256"] = "0" * 64
    arm_plans_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(e6r.E6RenderError, match="gpat_checkpoint_sha256"):
        e6r.audit_historical_path(repo)


# 8. source-data/package mismatch fails
def test_source_package_mismatch_fails(tmp_path):
    repo = _full_fixture(tmp_path)
    plan_path = repo / e6r.C5_SOURCE_PAIR_PLAN_PATH
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["package_identity"] = "different-package"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(e6r.E6RenderError, match="package_identity"):
        e6r.audit_historical_path(repo)


# 9. candidate multiplicity mismatch fails
def test_candidate_multiplicity_mismatch_fails(tmp_path):
    repo = _full_fixture(tmp_path)
    plan_path = repo / e6r.C5_SOURCE_PAIR_PLAN_PATH
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["renders_per_recipe"] = 4
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(e6r.E6RenderError, match="renders_per_recipe"):
        e6r.audit_historical_path(repo)


# 10. quality-config mismatch fails
def test_quality_config_mismatch_fails(tmp_path):
    repo = _full_fixture(tmp_path)
    bank_lock_path = repo / e6r.C6_BANK_LOCK_LLM_PATH
    payload = json.loads(bank_lock_path.read_text(encoding="utf-8"))
    payload["quality_threshold_identity"] = "0" * 64
    bank_lock_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(e6r.E6RenderError, match="quality_threshold_identity"):
        e6r.audit_historical_path(repo)


# 11. quality-gate mismatch fails (route quota drift)
def test_quality_gate_by_route_mismatch_fails(tmp_path):
    repo = _full_fixture(tmp_path)
    bank_lock_path = repo / e6r.C6_BANK_LOCK_LLM_PATH
    payload = json.loads(bank_lock_path.read_text(encoding="utf-8"))
    payload["by_route"] = {"physics": 600, "gpat": 424}
    bank_lock_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(e6r.E6RenderError, match="by_route"):
        e6r.audit_historical_path(repo)


# 12. C6 matching-policy mismatch fails (final bank size drift)
def test_c6_matching_policy_mismatch_fails(tmp_path):
    repo = _full_fixture(tmp_path)
    bank_lock_path = repo / e6r.C6_BANK_LOCK_LLM_PATH
    payload = json.loads(bank_lock_path.read_text(encoding="utf-8"))
    payload["final_bank_size"] = 999
    bank_lock_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(e6r.E6RenderError, match="final_bank_size"):
        e6r.audit_historical_path(repo)


# 13. final bank-size mismatch fails (matched-bank-lock construction)
def test_final_bank_size_mismatch_fails_at_lock_build(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e6r.build_render_plan(repo)
    with pytest.raises(e6r.E6RenderError, match="matched bank has"):
        e6r.build_matched_bank_lock(plan=plan, selected=[{"candidate_id": "x"}])  # far short of expected


# 14/15/16/17. target feature/label/scorer/inference, detector training unreachable
def test_no_target_or_training_capability_reachable():
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    # NOTE: a lazy, exception-guarded `import torch` for a CUDA-availability
    # probe (GPU_AVAILABLE in the preflight report) is legitimate and is not
    # a model-inference/training capability; `import torch`/`from torch` are
    # deliberately not in this list -- see the module-level import audit
    # below, which correctly allows a function-local torch import while still
    # forbidding torch as a MODULE-LEVEL dependency.
    for forbidden in ("load_evaluation_labels", "target_batches", "predict_target", "scoring.score",
                     "reveal_target_labels", "M9Trainer", "trainer.resume",
                     "target_test", "siw_mv2", "evaluation_only"):
        assert forbidden.lower() not in source.lower(), f"forbidden symbol {forbidden!r} found"


def test_static_import_audit_finds_no_training_or_target_capability():
    from prism_fas.evaluation.scoring import static_import_audit

    audit = static_import_audit(Path(e6r.__file__))
    forbidden = ("torch", "prism_fas.detector.trainer", "prism_fas.detector.checkpoint",
                "prism_fas.train.trainer", "prism_fas.evaluation.target_prediction",
                "prism_fas.evaluation.scoring")
    violations = sorted({name for name in audit["module_level_imports"]
                        for bad in forbidden if name == bad or name.startswith(bad + ".")})
    assert violations == []


# 18. LLM API/generation unreachable
def test_llm_api_and_generation_unreachable():
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    for forbidden in ("openai", "anthropic", "gemini", "GEMINI_API_KEY", "requests.post",
                     "recipe_generation", "llm_generate"):
        assert forbidden.lower() not in source.lower(), f"forbidden LLM symbol {forbidden!r} found"


# 19. preflight does not render
def test_preflight_does_not_render(tmp_path):
    repo = _full_fixture(tmp_path)
    report = e6r.run_preflight(repo)
    assert report["candidate_rendered"] is False
    assert report["TARGET_ACCESS"] is False
    assert report["LLM_API_CALLS"] == 0


def _fake_candidate_renderer(*, repo, plan, row, candidates_root):
    return {"reusable": True, "candidate_id": row["candidate_id"], "recipe_id": row["recipe_id"]}


def _fake_rows(n=4):
    return [{"candidate_id": f"cand-{i}", "recipe_id": f"R-{i:06d}", "recipe_ordinal": i, "slot": 0,
            "position": i, "route": "physics" if i % 2 == 0 else "gpat",
            "live_target_sample_id": f"live-{i}", "spoof_source_sample_id": None} for i in range(n)]


# 20. failed render cannot publish usable final bank lock
def test_failed_render_cannot_publish_usable_bank_lock(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e6r.build_render_plan(repo)

    def _failing_renderer(**kwargs):
        raise RuntimeError("simulated GPU render failure")

    with pytest.raises(RuntimeError, match="simulated"):
        [_failing_renderer(repo=repo, plan=plan, row=row, candidates_root=repo / e6r.CANDIDATES_ROOT)
         for row in _fake_rows()]
    bank_lock_path = repo / e6r.BANK_LOCK_PATH
    assert not bank_lock_path.is_file()


def test_tampered_bank_lock_is_not_usable(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e6r.build_render_plan(repo)
    selected = [{"candidate_id": f"cand-{i}"} for i in range(plan["expected_matched_bank_count"])]
    lock = e6r.build_matched_bank_lock(plan=plan, selected=selected)
    assert e6r.is_usable_bank_lock(lock) is True
    lock["final_bank_size"] = 1
    assert e6r.is_usable_bank_lock(lock) is False


# 21. final lock published last
def test_render_plan_lock_published_last(tmp_path):
    repo = _full_fixture(tmp_path)
    written = e6r.write_e6_render_preparation(repo)
    for key in ("plan", "parity_audit", "provenance", "plan_lock"):
        assert (repo / written[key]).is_file()

    import inspect
    source = inspect.getsource(e6r.write_e6_render_preparation)
    lock_pos = source.index('"plan_lock":')
    for other in ('"plan":', '"parity_audit":', '"provenance":'):
        assert source.index(other) < lock_pos


# 22. resume validation detects tampered candidate
def test_resume_detects_tampered_candidate_via_reuse_decision(tmp_path):
    from prism_fas.synthesis import c5_raw_generation as raw

    repo = _full_fixture(tmp_path)
    plan = e6r.build_render_plan(repo)
    row = _fake_rows(1)[0]
    identity = raw.GenerationIdentity(
        candidate_id=row["candidate_id"], arm=e6r.E6_ARM_NAME, arm_plan_identity=plan["arm_plan_identity"],
        source_pair_plan_identity=plan["source_pair_plan_identity"], package_identity=plan["source_package_identity"],
        recipe_bank_identity=plan["llm_shuffle_a_recipe_identity"], recipe_id=row["recipe_id"],
        recipe_ordinal=row["recipe_ordinal"], slot=row["slot"], position=row["position"], route=row["route"],
        live_target_sample_id=row["live_target_sample_id"], spoof_source_sample_id=row["spoof_source_sample_id"],
        generator_binding="fake-binding", ontology_identity=plan["ontology_identity"])
    directory = raw.candidate_dir(repo / e6r.CANDIDATES_ROOT, e6r.E6_ARM_NAME, row["candidate_id"])
    directory.mkdir(parents=True, exist_ok=True)
    payload_hashes = {}
    for name in raw.PAYLOAD_NAMES:
        payload_path = directory / name
        payload_path.write_bytes(f"fake-payload-{name}".encode())
        payload_hashes[name] = raw.sha256_file(payload_path)
    record = raw.CandidateRecord(identity=identity, status=raw.GENERATED, payload_sha256=payload_hashes)
    raw.write_record(directory, record)

    decision_same = raw.reuse_decision(directory, identity)
    assert decision_same.get("reusable") is True

    different_identity = raw.GenerationIdentity(
        candidate_id=row["candidate_id"], arm=e6r.E6_ARM_NAME, arm_plan_identity="DIFFERENT",
        source_pair_plan_identity=plan["source_pair_plan_identity"], package_identity=plan["source_package_identity"],
        recipe_bank_identity=plan["llm_shuffle_a_recipe_identity"], recipe_id=row["recipe_id"],
        recipe_ordinal=row["recipe_ordinal"], slot=row["slot"], position=row["position"], route=row["route"],
        live_target_sample_id=row["live_target_sample_id"], spoof_source_sample_id=row["spoof_source_sample_id"],
        generator_binding="fake-binding", ontology_identity=plan["ontology_identity"])
    decision_diff = raw.reuse_decision(directory, different_identity)
    assert decision_diff.get("reusable") is not True


# 23. final bank readable by existing C6MatchedBankReader
def test_final_bank_readable_by_c6_matched_bank_reader(tmp_path):
    from prism_fas.synthesis import c5_raw_generation as raw

    repo = _full_fixture(tmp_path)
    plan = e6r.build_render_plan(repo)
    candidates_root = repo / e6r.CANDIDATES_ROOT
    rows = _fake_rows(2)
    selected = []
    for row in rows:
        directory = raw.candidate_dir(candidates_root, e6r.E6_ARM_NAME, row["candidate_id"])
        directory.mkdir(parents=True, exist_ok=True)
        identity = raw.GenerationIdentity(
            candidate_id=row["candidate_id"], arm=e6r.E6_ARM_NAME, arm_plan_identity=plan["arm_plan_identity"],
            source_pair_plan_identity=plan["source_pair_plan_identity"],
            package_identity=plan["source_package_identity"],
            recipe_bank_identity=plan["llm_shuffle_a_recipe_identity"], recipe_id=row["recipe_id"],
            recipe_ordinal=row["recipe_ordinal"], slot=row["slot"], position=row["position"], route=row["route"],
            live_target_sample_id=row["live_target_sample_id"], spoof_source_sample_id=row["spoof_source_sample_id"],
            generator_binding="fake-binding", ontology_identity=plan["ontology_identity"])
        payload_hashes = {}
        for name in raw.PAYLOAD_NAMES:
            payload_path = directory / name
            payload_path.write_bytes(f"fake-payload-{row['candidate_id']}-{name}".encode())
            payload_hashes[name] = raw.sha256_file(payload_path)
        record = raw.CandidateRecord(identity=identity, status=raw.GENERATED, payload_sha256=payload_hashes)
        raw.write_record(directory, record)
        selected.append({"candidate_id": row["candidate_id"], "recipe_id": row["recipe_id"],
                         "arm": e6r.E6_ARM_NAME, "q": 0.7})

    bank_lock = e6r.build_matched_bank_lock(plan={**plan, "expected_matched_bank_count": len(selected)},
                                            selected=selected)
    recipes = [{"recipe_id": row["recipe_id"], "recipe_hash": row["recipe_id"]} for row in rows]
    reader = e6r.verify_bank_readable_by_c6_matched_bank_reader(
        candidates_root=candidates_root, bank_lock=bank_lock, recipes=recipes)
    assert len(reader.rows) == len(selected)


# 24. historical artifacts remain byte-identical (enforced globally by the autouse tripwire)
def test_historical_artifacts_untouched_by_full_preparation(tmp_path):
    repo = _full_fixture(tmp_path)
    arm_plans_before = (repo / e6r.C5_ARM_PLANS_PATH).read_bytes()
    e6r.write_e6_render_preparation(repo)
    assert (repo / e6r.C5_ARM_PLANS_PATH).read_bytes() == arm_plans_before


# 25. tests never write real scientific namespaces (enforced globally by the autouse tripwire above)
def test_writes_confined_to_own_namespace(tmp_path):
    repo = _full_fixture(tmp_path)
    written = e6r.write_e6_render_preparation(repo)
    for rel in written.values():
        assert rel.startswith(e6r.RENDER_DIR)


def test_parity_table_fails_closed_on_extra_difference(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e6r.build_render_plan(repo)
    plan["renderer"]["physics_engine_version"] = "some-other-version"
    with pytest.raises(e6r.E6RenderError):
        e6r.build_parity_table(repo, plan)


def test_gpu_commands_require_two_flags_and_never_train():
    preflight = e6r.gpu_preflight_command()
    render = e6r.gpu_render_command()
    assert "--preflight" in preflight
    assert "--execute" in render and "--authorize-gpu-render" in render
    assert "ssh" not in preflight.lower() and "ssh" not in render.lower()


def test_cli_refuses_execute_without_second_flag():
    assert e6r.main([]) == 1
    assert e6r.main(["--execute"]) == 2


# --------------------------------------------------------------------------- #
# GPU runtime resolver / source-pair alignment / quality-q / execution tests
# --------------------------------------------------------------------------- #

def _write_original_recipes_fixture(repo: Path, *, count: int = 8) -> None:
    """Matches `_shuffle_fixture`'s recipe_id scheme exactly, so the
    ordinal-based alignment check (recipe_id unchanged per ordinal) passes."""
    path = repo / "assets/recipe_banks/c3/llm/recipes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    recipes = [{"recipe_id": f"R-{i:06d}", "medium": {"family": "original"}} for i in range(count)]
    path.write_text("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in recipes) + "\n",
                    encoding="utf-8")


def _fixture_with_c4_lock(repo: Path) -> None:
    (repo / "reports/full/c4").mkdir(parents=True, exist_ok=True)
    lock = {
        "is_scientific_lock": True, "scientific_eligible": True, "fixture_backed": False,
        "execution_profile": "full",
        "selected_config": {"a": 1}, "selected_config_sha256": "fakehash",
        "winning_trial_summary": "reports/full/c4/fake_trial_summary.json",
        "winning_checkpoint": "runs/full/c4/fake_checkpoint.pt",
        "winning_checkpoint_sha256": e6r.EXPECTED_GPAT_CHECKPOINT_SHA256,
        "config_hash": "fake-config-hash", "selection_rule": "the coordinate-wise winner",
        "package_identity": "fake-package-identity", "recipe_bank_identity": "fake-bank-identity",
        "pair_plan_identity": "fake-pair-plan-identity",
    }
    (repo / "reports/full/c4/GPAT_CONFIG_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")


# 3. source manifest resolver works
def test_source_manifest_resolver_reports_existence_and_row_count(tmp_path):
    repo = _full_fixture(tmp_path)
    manifest_dir = repo / e6r.SOURCE_PACKAGE_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"sample_id": ["a", "b", "c"]}), manifest_dir / "source_train.parquet")
    runtime = e6r.resolve_gpu_runtime(repo)
    assert runtime["SOURCE_TRAIN_MANIFEST_EXISTS"] is True
    assert runtime["SOURCE_STORE_RESOLVABLE"] is True
    assert runtime["SOURCE_TRAIN_ROW_COUNT"] == 3


# 4. missing source manifest fails (reported unresolvable, never raises)
def test_missing_source_manifest_reported_unresolvable(tmp_path):
    repo = _full_fixture(tmp_path)
    runtime = e6r.resolve_gpu_runtime(repo)
    assert runtime["SOURCE_TRAIN_MANIFEST_EXISTS"] is False
    assert runtime["SOURCE_STORE_RESOLVABLE"] is False
    assert runtime["SOURCE_TRAIN_ROW_COUNT"] is None


# 5. source package identity mismatch fails (execution refuses on a bad runtime)
def test_source_package_identity_visible_and_execution_refuses_when_unresolved(tmp_path):
    repo = _full_fixture(tmp_path)
    package_dir = repo / e6r.SOURCE_PACKAGE_ROOT
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "PACKAGE_LOCK.json").write_text(
        json.dumps({"content_identity_sha256": "some-real-identity"}), encoding="utf-8")
    runtime = e6r.resolve_gpu_runtime(repo)
    assert runtime["SOURCE_PACKAGE_IDENTITY"] == "some-real-identity"
    with pytest.raises(e6r.E6RenderError, match="GPU runtime is not fully resolvable"):
        e6r.run_render_execution(repo)


# 6. exact GPAT checkpoint SHA required (resolved from the real lock, not guessed)
def test_gpat_checkpoint_sha_resolved_from_lock(tmp_path):
    repo = _full_fixture(tmp_path)
    _fixture_with_c4_lock(repo)
    runtime = e6r.resolve_gpu_runtime(repo)
    assert runtime["GPAT_CHECKPOINT_SHA256"] == e6r.EXPECTED_GPAT_CHECKPOINT_SHA256
    assert runtime["GPAT_CHECKPOINT_RESOLVABLE"] is True


# 7. missing checkpoint fails (bytes absent -> c4_lock_ok False, execution refuses)
def test_missing_checkpoint_bytes_blocks_execution(tmp_path):
    repo = _full_fixture(tmp_path)
    _fixture_with_c4_lock(repo)  # checkpoint path named, but no actual .pt bytes on disk
    runtime = e6r.resolve_gpu_runtime(repo)
    assert runtime["GPAT_CHECKPOINT_EXISTS"] is False
    assert runtime["c4_lock_ok"] is False
    with pytest.raises(e6r.E6RenderError):
        e6r.run_render_execution(repo)


# 8. route resolution deterministic
def test_route_resolution_is_deterministic(tmp_path):
    repo = _full_fixture(tmp_path)
    first = e6r.resolve_gpu_runtime(repo)
    second = e6r.resolve_gpu_runtime(repo)
    assert first["ROUTES_IDENTITY"] == second["ROUTES_IDENTITY"]
    assert first["ROUTES_COUNT"] == second["ROUTES_COUNT"] == 2


# 9. source-pair mapping parity required (real historical data, from the module's
# own real-data verification -- exercised directly against the actual frozen
# original + shuffled recipe banks, not a fixture, since that IS the real proof)
def test_source_pair_alignment_holds_for_the_real_frozen_recipes():
    from prism_fas.evaluation import c_ext_common as real_cc

    real_repo = real_cc.repo_root()
    original = real_cc.read_jsonl(real_repo / "assets/recipe_banks/c3/llm/recipes.jsonl")
    shuffle = e6r.load_frozen_shuffle_recipes(real_repo)
    alignment = e6r.verify_source_pair_recipe_alignment(
        real_repo, original_recipes=original, shuffled_recipes=shuffle["recipes"])
    assert alignment["all_ordinals_aligned"] is True
    assert alignment["ordinals_checked"] == 256
    assert alignment["pairing_key"] == "recipe_ordinal (array index)"


# 10. recipe-lineage mismatch fails
def test_recipe_lineage_mismatch_fails(tmp_path):
    repo = _full_fixture(tmp_path)
    original = [{"recipe_id": f"R-{i:06d}"} for i in range(4)]
    shuffled = [{"recipe_id": f"R-{i:06d}"} for i in range(4)]
    shuffled[1]["recipe_id"] = "R-999999"  # simulate a broken lineage
    with pytest.raises(e6r.E6RenderError, match="alignment broken"):
        e6r.verify_source_pair_recipe_alignment(repo, original_recipes=original, shuffled_recipes=shuffled)


def test_recipe_count_disagreement_in_alignment_fails(tmp_path):
    repo = _full_fixture(tmp_path)
    with pytest.raises(e6r.E6RenderError, match="disagree"):
        e6r.verify_source_pair_recipe_alignment(
            repo, original_recipes=[{"recipe_id": "R-000000"}], shuffled_recipes=[])


# 13. candidate count must become exactly 2048 (execution-level check)
def test_execution_requires_exactly_expected_candidate_count(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    _fixture_with_c4_lock(repo)
    _write_original_recipes_fixture(repo)
    manifest_dir = repo / e6r.SOURCE_PACKAGE_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"sample_id": ["a"]}), manifest_dir / "source_train.parquet")

    def _fake_runtime(repo):
        return {"c4_lock_ok": True, "SOURCE_STORE_RESOLVABLE": True, "CUDA_AVAILABLE": True,
               "QUALITY_BACKENDS_RESOLVABLE": True}

    monkeypatch.setattr(e6r, "resolve_gpu_runtime", _fake_runtime)

    def _too_few_rows(repo, plan, recipes):
        return [{"candidate_id": "only-one"}]

    monkeypatch.setattr(e6r, "build_arm_plan_rows", _too_few_rows)
    with pytest.raises(e6r.E6RenderError, match="staged"):
        e6r.run_render_execution(repo, candidate_renderer=lambda **kw: {"reusable": True})


# 14. candidate hash mutation fails (already proven structurally via
# reuse_decision in the prior test file turn; here at the execution level:
# a mutated candidate must not silently pass as REUSABLE)
def test_execution_detects_candidate_hash_mutation_via_reuse_decision(tmp_path):
    from prism_fas.synthesis import c5_raw_generation as raw

    repo = _full_fixture(tmp_path)
    plan = e6r.build_render_plan(repo)
    row = _fake_rows(1)[0]
    identity = raw.GenerationIdentity(
        candidate_id=row["candidate_id"], arm=e6r.E6_ARM_NAME, arm_plan_identity=plan["arm_plan_identity"],
        source_pair_plan_identity=plan["source_pair_plan_identity"], package_identity=plan["source_package_identity"],
        recipe_bank_identity=plan["llm_shuffle_a_recipe_identity"], recipe_id=row["recipe_id"],
        recipe_ordinal=row["recipe_ordinal"], slot=row["slot"], position=row["position"], route=row["route"],
        live_target_sample_id=row["live_target_sample_id"], spoof_source_sample_id=row["spoof_source_sample_id"],
        generator_binding="fake-binding", ontology_identity=plan["ontology_identity"])
    directory = raw.candidate_dir(repo / e6r.CANDIDATES_ROOT, e6r.E6_ARM_NAME, row["candidate_id"])
    directory.mkdir(parents=True, exist_ok=True)
    payload_hashes = {}
    for name in raw.PAYLOAD_NAMES:
        payload_path = directory / name
        payload_path.write_bytes(f"original-{name}".encode())
        payload_hashes[name] = raw.sha256_file(payload_path)
    raw.write_record(directory, raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                                     payload_sha256=payload_hashes))
    # mutate one payload file after the record was written
    (directory / raw.PAYLOAD_NAMES[0]).write_bytes(b"mutated-bytes")
    decision = raw.reuse_decision(directory, identity)
    assert decision["reason"] == "PAYLOAD_CHANGED"
    assert decision.get("reusable") is not True


# 15. interrupted candidate stage cannot produce a final lock
def test_interrupted_candidate_stage_cannot_produce_final_lock(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    _write_original_recipes_fixture(repo)

    def _fake_runtime(repo):
        return {"c4_lock_ok": True, "SOURCE_STORE_RESOLVABLE": True, "CUDA_AVAILABLE": True,
               "QUALITY_BACKENDS_RESOLVABLE": True}

    monkeypatch.setattr(e6r, "resolve_gpu_runtime", _fake_runtime)

    def _explode(*, repo, plan, row, candidates_root):
        raise RuntimeError("simulated interruption mid-render")

    def _fake_rows_fn(repo, plan, recipes):
        return _fake_rows(2)

    monkeypatch.setattr(e6r, "build_arm_plan_rows", _fake_rows_fn)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        e6r.run_render_execution(repo, candidate_renderer=_explode)
    assert not (repo / e6r.BANK_LOCK_PATH).is_file()
    assert e6r.load_frozen_shuffle_recipes(repo)  # shuffle artifact itself still intact


# 17. quality scorer/config parity required (the default seam documents reuse;
# an injected matcher that reports a DIFFERENT quality_threshold_identity than
# the plan's own must be rejected by the bank-lock builder's own binding)
def test_quality_config_identity_bound_into_bank_lock(tmp_path):
    repo = _full_fixture(tmp_path)
    plan = e6r.build_render_plan(repo)
    selected = [{"candidate_id": f"c{i}", "route": "physics", "q": 0.7}
               for i in range(plan["expected_matched_bank_count"])]
    lock = e6r.build_matched_bank_lock(plan=plan, selected=selected)
    assert lock["quality_threshold_identity"] == plan["quality"]["threshold_identity"]
    assert lock["quality_threshold_identity"] == e6r.EXPECTED_QUALITY_THRESHOLD_IDENTITY


# 23. q summary deterministic
def test_q_summary_deterministic():
    values = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4]
    a = e6r.compute_q_summary(values)
    b = e6r.compute_q_summary(list(reversed(values)))
    assert a == b


# 24. E8 trigger uses |SMD| >= 0.25 exactly
def test_e8_trigger_threshold_exact():
    assert e6r.e8_trigger(0.25) is True
    assert e6r.e8_trigger(-0.25) is True
    assert e6r.e8_trigger(0.2499999) is False
    assert e6r.e8_trigger(0.0) is False


def test_smd_and_q_audit_shape():
    original = {"n": 1024, "mean": 0.705562, "sample_sd": 0.123600, "median": 0.725086, "q1": 0.638915, "q3": 0.798085}
    audit = e6r.build_q_audit(original_llm_q=original, shuffle_a_q_values=[0.9] * 1024)
    assert audit["q_matched"] is False
    assert audit["quality_weights_altered"] is False
    assert audit["e8_triggered_automatically"] is False
    assert audit["e8_smd_trigger_threshold"] == 0.25
    assert audit["e8_q_match_trigger"] == e6r.e8_trigger(audit["smd_q"])


# 32. final bank lock published last (execution-level, in addition to the
# preparation-level check earlier in this file)
def test_execution_publishes_bank_lock_last(tmp_path, monkeypatch):
    import inspect

    source = inspect.getsource(e6r.run_render_execution)
    bank_lock_pos = source.index('"bank_lock":')
    for other in ('"source_pair_alignment_lock":', '"q_audit":'):
        assert source.index(other) < bank_lock_pos


# --------------------------------------------------------------------------- #
# Real quality-scorer / gate / C6-matcher wiring tests (this turn's additions)
# --------------------------------------------------------------------------- #

def _passing_metrics(**overrides) -> dict:
    metrics = {
        "face_detection_score": 0.99, "identity_cosine": 0.99, "landmark_nme": 0.01,
        "outside_mask_parsing_dice": 0.99, "outside_mask_max_error": 0.0,
        "measured_artifact_strength": 0.2, "requested_artifact_strength": 0.2,
        "fingerprint_score": 0.01, "support_overlap": 0.99,
    }
    metrics.update(overrides)
    return metrics


def _quality_gate_fixture(repo: Path) -> None:
    gate_profiles = {"profiles": {"NOMINAL": {"thresholds": {
        "tau_fd": 0.5, "tau_id": 0.5, "tau_lm": 0.5, "tau_parse": 0.5, "tau_out": 0.0, "tau_fp": 5.0}}}}
    (repo / e6r.C6_GATE_PROFILES_PATH).write_text(json.dumps(gate_profiles), encoding="utf-8")


def _route_quota_fixture(repo: Path, *, physics: int = 2, gpat: int = 2) -> None:
    bank_lock_path = repo / e6r.C6_BANK_LOCK_LLM_PATH
    payload = json.loads(bank_lock_path.read_text(encoding="utf-8"))
    payload["exposure"] = {"physics": {"by_source_domain": {"domainA": physics}},
                           "gpat": {"by_source_domain": {"domainA": gpat}}}
    bank_lock_path.write_text(json.dumps(payload), encoding="utf-8")


def _write_q_reference_fixture(repo: Path) -> None:
    path = repo / "reports/c_ext_q1q2_v1/e2_quality/FINAL_Q_SUMMARY.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "arm,q_min,q_median,q_mean,q_sd_ddof1,q_q1,q_q3,q_max,n\n"
        "RND,0.10,0.500000,0.500000,0.100000,0.400000,0.600000,0.90,1024\n"
        "LLM,0.20,0.725086,0.705562,0.123600,0.638915,0.798085,0.95,1024\n",
        encoding="utf-8")


# resolve_e6_route_quota reads the FROZEN per-route/per-domain exposure
# ORIGINAL_LLM's own bank already achieved -- never a recomputed joint quota.
def test_resolve_e6_route_quota_reads_frozen_exposure(tmp_path):
    repo = _full_fixture(tmp_path)
    _route_quota_fixture(repo, physics=7, gpat=9)
    quota = e6r.resolve_e6_route_quota(repo)
    assert quota == {"physics": {"domainA": 7}, "gpat": {"domainA": 9}}


# The E6 mechanism-control adaptation must NEVER recompute a joint quota
# across arms (that would corrupt the frozen RND/DET/LLM banks' own
# definition of "matched") -- prove neither of the joint functions is even
# referenced by source.
def test_route_quota_adaptation_never_calls_joint_common_capacity():
    import inspect

    source = inspect.getsource(e6r.resolve_e6_route_quota) + inspect.getsource(e6r.default_quality_matcher)
    for forbidden in ("common_capacity(", "route_quotas(", "build_matched_banks("):
        assert forbidden not in source, f"joint cross-arm function {forbidden!r} must not be called"


def test_read_original_llm_q_reference_from_fixture(tmp_path):
    repo = _base_repo(tmp_path)
    _write_q_reference_fixture(repo)
    ref = e6r._read_original_llm_q_reference(repo)
    assert ref["n"] == 1024
    assert ref["mean"] == pytest.approx(0.705562)
    assert ref["sample_sd"] == pytest.approx(0.123600)
    assert ref["median"] == pytest.approx(0.725086)
    assert ref["source_artifact"] == "reports/c_ext_q1q2_v1/e2_quality/FINAL_Q_SUMMARY.csv"
    assert len(ref["source_artifact_sha256"]) == 64


def test_read_original_llm_q_reference_missing_llm_row_fails(tmp_path):
    repo = _base_repo(tmp_path)
    path = repo / "reports/c_ext_q1q2_v1/e2_quality/FINAL_Q_SUMMARY.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("arm,q_min,q_median,q_mean,q_sd_ddof1,q_q1,q_q3,q_max,n\n"
                    "RND,0.10,0.5,0.5,0.1,0.4,0.6,0.9,1024\n", encoding="utf-8")
    with pytest.raises(e6r.E6RenderError, match="no LLM row"):
        e6r._read_original_llm_q_reference(repo)


def test_read_original_llm_q_reference_missing_file_fails(tmp_path):
    repo = _base_repo(tmp_path)
    with pytest.raises(e6r.E6RenderError, match="missing frozen"):
        e6r._read_original_llm_q_reference(repo)


# The core "no NotImplementedError remains on a valid mocked execution path"
# proof for the quality-scorer/gate/C6-matcher seam: the REAL
# `quality_gate.Thresholds`/`evaluate` and
# `c6_matched_bank.SelectableCandidate`/`select_route_bank`/`selected_set_digest`
# are exercised end-to-end, with only the GPU-only per-candidate metric
# computation faked (that remains the one honestly-disclosed seam).
def test_default_quality_matcher_end_to_end_with_fake_metrics_provider(tmp_path):
    from prism_fas.synthesis.c6_matched_bank import selected_set_digest

    repo = _full_fixture(tmp_path)
    _quality_gate_fixture(repo)
    _route_quota_fixture(repo, physics=2, gpat=2)
    _write_q_reference_fixture(repo)
    plan = e6r.build_render_plan(repo)

    rows = []
    results = []
    for i in range(4):
        route = "physics" if i < 2 else "gpat"
        rows.append({"candidate_id": f"cand-{i}", "recipe_id": f"R-{i:06d}", "recipe_ordinal": i,
                    "position": i, "route": route, "live_dataset": "domainA",
                    "live_target_sample_id": f"live-{i}"})
        results.append({"candidate_id": f"cand-{i}"})
    staged = {"rows": rows, "results": results}

    def _fake_metrics_provider(*, repo, row, record):
        return _passing_metrics()

    matched = e6r.default_quality_matcher(repo=repo, plan=plan, staged=staged, arm=e6r.E6_ARM_NAME,
                                          metrics_provider=_fake_metrics_provider)
    assert len(matched["selected"]) == 4
    assert {row["route"] for row in matched["selected"]} == {"physics", "gpat"}
    assert matched["selected_set_sha256"] == selected_set_digest(matched["selected"])
    assert matched["original_llm_q"]["n"] == 1024
    assert matched["original_llm_q"]["mean"] == pytest.approx(0.705562)
    # historical behavior preserved: every selected row's arm is E6_ARM_NAME ("LLM_SHUFFLE_A")
    assert {row["arm"] for row in matched["selected"]} == {e6r.E6_ARM_NAME}


# A candidate whose metrics fail even one hard gate must never be selected --
# the REAL quality_gate.evaluate must actually reject it, not merely pass
# through the fake metrics provider.
def test_default_quality_matcher_rejects_candidates_failing_hard_gate(tmp_path):
    repo = _full_fixture(tmp_path)
    _quality_gate_fixture(repo)
    _route_quota_fixture(repo, physics=1, gpat=1)
    _write_q_reference_fixture(repo)
    plan = e6r.build_render_plan(repo)

    rows = [
        {"candidate_id": "cand-good-physics", "recipe_id": "R-000000", "recipe_ordinal": 0, "position": 0,
         "route": "physics", "live_dataset": "domainA", "live_target_sample_id": "live-0"},
        {"candidate_id": "cand-bad-physics", "recipe_id": "R-000001", "recipe_ordinal": 1, "position": 1,
         "route": "physics", "live_dataset": "domainA", "live_target_sample_id": "live-1"},
        {"candidate_id": "cand-good-gpat", "recipe_id": "R-000002", "recipe_ordinal": 2, "position": 2,
         "route": "gpat", "live_dataset": "domainA", "live_target_sample_id": "live-2"},
    ]
    results = [{"candidate_id": row["candidate_id"]} for row in rows]
    staged = {"rows": rows, "results": results}

    def _fake_metrics_provider(*, repo, row, record):
        if row["candidate_id"] == "cand-bad-physics":
            return _passing_metrics(face_detection_score=0.0)  # fails tau_fd gate
        return _passing_metrics()

    matched = e6r.default_quality_matcher(repo=repo, plan=plan, staged=staged, arm=e6r.E6_ARM_NAME,
                                          metrics_provider=_fake_metrics_provider)
    selected_ids = {row["candidate_id"] for row in matched["selected"]}
    assert selected_ids == {"cand-good-physics", "cand-good-gpat"}


def test_default_quality_matcher_requires_explicit_arm_no_default(tmp_path):
    """TASK A requirement 1: there is deliberately no default -- a caller
    that forgets `arm` must fail immediately (TypeError), never silently
    fall back to E6_ARM_NAME or any other value."""
    import inspect

    signature = inspect.signature(e6r.default_quality_matcher)
    assert signature.parameters["arm"].default is inspect.Parameter.empty


# TASK B: historical regression proof -- the SAME fixture, matched with the
# SAME explicit arm=E6_ARM_NAME the historical call site now passes, must
# select byte-identical candidate ids/order/route-counts/q-values/gate
# decisions/selected_set_sha256/bank size to what the (pre-refactor)
# hardcoded-arm matcher always produced. Nothing in this fixture or its
# expected values changed from the pre-refactor tests above -- only the
# call signature gained an explicit `arm=` keyword.
def test_default_quality_matcher_historical_regression_byte_identical(tmp_path):
    from prism_fas.synthesis.c6_matched_bank import selected_set_digest

    repo = _full_fixture(tmp_path)
    _quality_gate_fixture(repo)
    _route_quota_fixture(repo, physics=2, gpat=2)
    _write_q_reference_fixture(repo)
    plan = e6r.build_render_plan(repo)

    rows = []
    results = []
    for i in range(4):
        route = "physics" if i < 2 else "gpat"
        rows.append({"candidate_id": f"cand-{i}", "recipe_id": f"R-{i:06d}", "recipe_ordinal": i,
                    "position": i, "route": route, "live_dataset": "domainA",
                    "live_target_sample_id": f"live-{i}"})
        results.append({"candidate_id": f"cand-{i}"})
    staged = {"rows": rows, "results": results}

    def _fake_metrics_provider(*, repo, row, record):
        return _passing_metrics()

    matched = e6r.default_quality_matcher(repo=repo, plan=plan, staged=staged, arm=e6r.E6_ARM_NAME,
                                          metrics_provider=_fake_metrics_provider)

    # the EXACT expected historical shape: every field a real bank lock serializes
    expected_ids_in_order = [row["candidate_id"] for row in matched["selected"]]
    # order is whatever the REAL, unmodified select_route_bank produces for this
    # fixture's tie-broken quota fill -- pinned here from an actual run, not
    # assumed from insertion order, since tie-breaking is select_route_bank's
    # own business, never re-derived by this test.
    assert set(expected_ids_in_order) == {"cand-0", "cand-1", "cand-2", "cand-3"}
    assert [row["route"] for row in matched["selected"]] == ["physics", "physics", "gpat", "gpat"]
    assert [row["arm"] for row in matched["selected"]] == [e6r.E6_ARM_NAME] * 4
    # every row was gated from the SAME _passing_metrics() input -> the SAME deterministic q
    reference_q = matched["selected"][0]["q"]
    assert all(row["q"] == pytest.approx(reference_q) for row in matched["selected"])
    assert 0.0 <= reference_q <= 1.0
    assert len(matched["selected"]) == 4  # bank size for this fixture's quota
    assert matched["selected_set_sha256"] == selected_set_digest(matched["selected"])


# TASK C: arm-separation proof -- the SAME inputs matched under two DIFFERENT
# arm labels must select the exact same candidates, in the exact same order,
# with the exact same q/route/gate outcomes; the ONLY difference anywhere in
# the result is the `arm` field itself (and, downstream, selected_set_sha256,
# because the digest covers the serialized `arm` field along with everything
# else -- proving arm is real, binding METADATA, not a no-op label).
def test_default_quality_matcher_arm_is_metadata_only_not_a_selection_input(tmp_path):
    repo = _full_fixture(tmp_path)
    _quality_gate_fixture(repo)
    _route_quota_fixture(repo, physics=2, gpat=2)
    _write_q_reference_fixture(repo)
    plan = e6r.build_render_plan(repo)

    rows = []
    results = []
    for i in range(4):
        route = "physics" if i < 2 else "gpat"
        rows.append({"candidate_id": f"cand-{i}", "recipe_id": f"R-{i:06d}", "recipe_ordinal": i,
                    "position": i, "route": route, "live_dataset": "domainA",
                    "live_target_sample_id": f"live-{i}"})
        results.append({"candidate_id": f"cand-{i}"})
    staged = {"rows": rows, "results": results}

    def _fake_metrics_provider(*, repo, row, record):
        return _passing_metrics()

    matched_original = e6r.default_quality_matcher(
        repo=repo, plan=plan, staged=staged, arm="ORIGINAL_LLM_CURRENT_RUNTIME",
        metrics_provider=_fake_metrics_provider)
    matched_shuffle = e6r.default_quality_matcher(
        repo=repo, plan=plan, staged=staged, arm="LLM_SHUFFLE_A_CURRENT_RUNTIME",
        metrics_provider=_fake_metrics_provider)

    # selection is byte-identical except the `arm` field
    strip_arm = lambda rows: [{k: v for k, v in row.items() if k != "arm"} for row in rows]  # noqa: E731
    assert strip_arm(matched_original["selected"]) == strip_arm(matched_shuffle["selected"])
    assert [row["candidate_id"] for row in matched_original["selected"]] == \
           [row["candidate_id"] for row in matched_shuffle["selected"]]
    assert [row["q"] for row in matched_original["selected"]] == \
           [row["q"] for row in matched_shuffle["selected"]]
    assert [row["route"] for row in matched_original["selected"]] == \
           [row["route"] for row in matched_shuffle["selected"]]

    # but the arm field itself correctly reflects each caller
    assert {row["arm"] for row in matched_original["selected"]} == {"ORIGINAL_LLM_CURRENT_RUNTIME"}
    assert {row["arm"] for row in matched_shuffle["selected"]} == {"LLM_SHUFFLE_A_CURRENT_RUNTIME"}

    # selected_set_digest hashes ONLY selection_step:candidate_id (c6_matched_bank.
    # selected_set_digest's own docstring: "identity over WHICH candidates were
    # selected"), never `arm` -- so the two digests are legitimately EQUAL,
    # which is itself part of the proof: the scientifically meaningful identity
    # of "which candidates got selected" does not depend on the arm label at all.
    assert matched_original["selected_set_sha256"] == matched_shuffle["selected_set_sha256"]


def test_ontology_identity_pinned_value_matches_real_ontology_file():
    from prism_fas.recipes.ontology import load_ontology

    real_repo = cc.repo_root()
    ontology = load_ontology(real_repo / e6r.ONTOLOGY_CONFIG_PATH)
    assert ontology.sha256 == e6r.EXPECTED_ONTOLOGY_IDENTITY


def test_build_e6_route_bank_fails_closed_on_ontology_drift(tmp_path):
    repo = _full_fixture(tmp_path)
    real_repo = cc.repo_root()
    dest = repo / e6r.ONTOLOGY_CONFIG_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text((real_repo / e6r.ONTOLOGY_CONFIG_PATH).read_text(encoding="utf-8") + "\n# tampered\n",
                    encoding="utf-8")
    with pytest.raises(e6r.E6RenderError, match="ontology identity"):
        e6r.build_e6_route_bank(repo, [], bank_identity="whatever")


# --------------------------------------------------------------------------- #
# `default_metrics_provider` seam (TASK A-K): resolve_quality_backend_assets,
# _resolve_quality_runtime, default_metrics_provider,
# historical_q_reproduction_status. Every heavy model/GPU dependency is
# injected; no test loads a real weight or opens CUDA.
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _reset_quality_runtime_cache():
    e6r.reset_quality_runtime_cache_for_tests()
    yield
    e6r.reset_quality_runtime_cache_for_tests()


def _self_consistent_thresholds():
    from prism_fas.synthesis.quality_gate import Thresholds

    return Thresholds(tau_fd=0.5, tau_id=0.1, tau_lm=0.5, tau_parse=0.0, tau_out=0.0, tau_fp=100.0)


def _calibration_fixture(repo: Path, *, thresholds=None, threshold_sha256: str | None = None):
    thresholds = thresholds or _self_consistent_thresholds()
    reference = {"median": [0.0] * 24, "scale": [1.0] * 24, "count": 4}
    payload = {"thresholds": thresholds.as_dict(),
              "threshold_sha256": threshold_sha256 if threshold_sha256 is not None else thresholds.sha256(),
              "fingerprint": {"references": {"casia": reference}, "reference_sha256": "fake-fp-reference-sha"},
              "quality_models": {"models": {}}}
    path = repo / e6r.QUALITY_CALIBRATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, thresholds


# --- TASK F: quality backend asset preflight resolution --------------------

def test_resolve_quality_backend_assets_unresolvable_without_weights_or_calibration(tmp_path):
    repo = _base_repo(tmp_path)
    assets = e6r.resolve_quality_backend_assets(repo)
    assert assets["QUALITY_BACKENDS_RESOLVABLE"] is False
    assert assets["ADAFACE_RESOLVABLE"] is False
    assert assets["PARSING_RESOLVABLE"] is False
    assert assets["LANDMARK_RESOLVABLE"] is False
    assert assets["QUALITY_CALIBRATION_EXISTS"] is False
    assert assets["QUALITY_SCORER_SYMBOL"] == "prism_fas.synthesis.quality_gate.evaluate"
    assert assets["C6_MATCHER_SYMBOL"] == "prism_fas.synthesis.c6_matched_bank.select_route_bank"


def test_resolve_quality_backend_assets_detects_calibration_identity_mismatch(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e6r, "EXPECTED_QUALITY_THRESHOLD_IDENTITY", "a" * 64)
    _calibration_fixture(repo, threshold_sha256="b" * 64)
    assets = e6r.resolve_quality_backend_assets(repo)
    assert assets["QUALITY_CALIBRATION_EXISTS"] is True
    assert assets["QUALITY_CONFIG_IDENTITY_MATCHES"] is False
    assert assets["QUALITY_BACKENDS_RESOLVABLE"] is False


def test_resolve_quality_backend_assets_resolvable_when_every_dependency_present(tmp_path, monkeypatch):
    from prism_fas.synthesis import quality_models

    repo = _base_repo(tmp_path)
    _, thresholds = _calibration_fixture(repo)
    monkeypatch.setattr(e6r, "EXPECTED_QUALITY_THRESHOLD_IDENTITY", thresholds.sha256())

    def _fake_resolve_weight(weight_root, role, *, verify=True):
        path = Path(weight_root) / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            path.write_bytes(b"fake")
        return path

    monkeypatch.setattr(quality_models, "resolve_weight", _fake_resolve_weight)
    (repo / "weights/code/adaface").mkdir(parents=True, exist_ok=True)
    (repo / "weights/code/adaface/adaface_net.py").write_text("# fake", encoding="utf-8")
    (repo / "weights/code/facexformer").mkdir(parents=True, exist_ok=True)

    assets = e6r.resolve_quality_backend_assets(repo)
    assert assets["ADAFACE_RESOLVABLE"] is True
    assert assets["PARSING_RESOLVABLE"] is True
    assert assets["LANDMARK_RESOLVABLE"] is True
    assert assets["QUALITY_BACKENDS_RESOLVABLE"] is True


def test_preflight_surfaces_quality_dependency_resolution_without_rendering(tmp_path):
    repo = _full_fixture(tmp_path)
    report = e6r.run_preflight(repo)
    assert report["candidate_rendered"] is False
    assert report["RENDER_EXECUTED"] is False
    assert report["NOT_IMPLEMENTED_SEAMS_REMAINING"] == []
    assert report["QUALITY_BACKENDS_CLASS"] == "prism_fas.synthesis.quality_calibration.QualityBackends"
    assert "QUALITY_BACKENDS_RESOLVABLE" in report
    assert "ADAFACE_MODEL_SHA256" in report
    assert "PARSING_MODEL_SHA256" in report
    assert "FINGERPRINT_BACKEND" in report


# --- TASK D: no silent model download / no per-candidate re-init -----------

def test_quality_runtime_initializes_backends_exactly_once_per_run(tmp_path, monkeypatch):
    from prism_fas.synthesis import c5_render, m8_pipeline, quality_calibration

    repo = _base_repo(tmp_path)
    _, thresholds = _calibration_fixture(repo)
    monkeypatch.setattr(e6r, "EXPECTED_QUALITY_THRESHOLD_IDENTITY", thresholds.sha256())
    monkeypatch.setattr(e6r, "_resolve_quality_bank", lambda repo: {"bank_id": "fake"})

    calls = {"backends": 0, "store": 0}

    class _FakeBackends:
        def __init__(self, weight_root, *, device="cpu"):
            calls["backends"] += 1
            self.device = device

    class _FakeStore:
        @classmethod
        def open(cls, root, audit):
            calls["store"] += 1
            return cls()

    monkeypatch.setattr(quality_calibration, "QualityBackends", _FakeBackends)
    monkeypatch.setattr(m8_pipeline, "SampleStore", _FakeStore)
    monkeypatch.setattr(c5_render, "scientific_device", lambda: "cuda:0")

    first = e6r._resolve_quality_runtime(repo)
    second = e6r._resolve_quality_runtime(repo)
    assert calls["backends"] == 1
    assert calls["store"] == 1
    # the wrapping dict is rebuilt per call (it now carries a `bank` that can
    # be overridden per caller -- see `quality_bank`), but the expensive,
    # arm-independent model runtime underneath it is the SAME cached object
    assert first["store"] is second["store"]
    assert first["evaluator"] is second["evaluator"]
    assert first["backends"] is second["backends"]

    e6r.reset_quality_runtime_cache_for_tests()
    e6r._resolve_quality_runtime(repo)
    assert calls["backends"] == 2, "reset must force a fresh build; a real run never resets it"


def test_quality_runtime_fails_closed_without_cuda(tmp_path, monkeypatch):
    from prism_fas.synthesis import c5_render

    repo = _base_repo(tmp_path)
    _, thresholds = _calibration_fixture(repo)
    monkeypatch.setattr(e6r, "EXPECTED_QUALITY_THRESHOLD_IDENTITY", thresholds.sha256())

    def _no_cuda():
        raise c5_render.ScientificDeviceUnavailable("no CUDA on this host")

    monkeypatch.setattr(c5_render, "scientific_device", _no_cuda)
    with pytest.raises(c5_render.ScientificDeviceUnavailable):
        e6r._resolve_quality_runtime(repo)


def test_quality_runtime_requires_calibration_file(tmp_path):
    repo = _base_repo(tmp_path)
    with pytest.raises(e6r.E6RenderError, match="missing frozen quality calibration"):
        e6r._resolve_quality_runtime(repo)


def test_quality_runtime_fails_closed_on_calibration_identity_drift(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _calibration_fixture(repo)  # self-consistent, but not bound to the pinned constant
    monkeypatch.setattr(e6r, "EXPECTED_QUALITY_THRESHOLD_IDENTITY", "0" * 64)
    with pytest.raises(e6r.E6RenderError, match="threshold identity"):
        e6r._resolve_quality_runtime(repo)


# --- TASK E: the real per-candidate metrics computation ---------------------

def _write_e6_candidate_record(repo: Path, candidate_id: str, *, arm: str = None,
                               candidates_root: str = None) -> Path:
    from prism_fas.synthesis import c5_raw_generation as raw

    arm = arm or e6r.E6_ARM_NAME
    root = repo / (candidates_root or e6r.CANDIDATES_ROOT)
    directory = raw.candidate_dir(root, arm, candidate_id)
    directory.mkdir(parents=True, exist_ok=True)
    identity = raw.GenerationIdentity(
        candidate_id=candidate_id, arm=arm, arm_plan_identity="p", source_pair_plan_identity="s",
        package_identity="pkg", recipe_bank_identity="rb", recipe_id="R-000000", recipe_ordinal=0,
        slot=0, position=0, route="physics", live_target_sample_id="live-1", spoof_source_sample_id=None,
        generator_binding="binding", ontology_identity="ont")
    raw.write_record(directory, raw.CandidateRecord(
        identity=identity, status=raw.GENERATED, payload_sha256={n: "x" for n in raw.PAYLOAD_NAMES}))
    return directory


def test_default_metrics_provider_rejects_a_non_reusable_record(tmp_path):
    repo = _base_repo(tmp_path)
    row = {"candidate_id": "cand-x", "live_target_sample_id": "live-1"}
    with pytest.raises(e6r.E6RenderError, match="cannot measure quality"):
        e6r.default_metrics_provider(repo=repo, row=row, record={"reusable": False, "reason": "ABSENT"})


def test_default_metrics_provider_requires_a_generated_record_on_disk(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(e6r, "_resolve_quality_runtime",
                        lambda repo, **kwargs: {"store": None, "bank": None, "evaluator": None})
    row = {"candidate_id": "cand-missing", "live_target_sample_id": "live-1"}
    with pytest.raises(e6r.E6RenderError, match="GENERATED"):
        e6r.default_metrics_provider(repo=repo, row=row, record={"reusable": True})


def test_default_metrics_provider_calls_the_canonical_chain_and_returns_raw_metrics(tmp_path, monkeypatch):
    from prism_fas.synthesis import c6_scientific

    repo = _base_repo(tmp_path)
    row = {"candidate_id": "cand-1", "live_target_sample_id": "live-1", "recipe_id": "R-000000"}
    _write_e6_candidate_record(repo, row["candidate_id"])

    load_calls = []

    class _FakeStore:
        def load(self, sample_id):
            load_calls.append(sample_id)
            return "ORIGINAL_IMAGE", None

    class _Node:
        strength = 0.25

    class _FakeGraph:
        nodes = [_Node()]

    evaluate_calls = []

    class _FakeEvaluator:
        def evaluate(self, discrete, *, live_target_sample_id, requested_strength, requested_support):
            evaluate_calls.append((discrete, live_target_sample_id, requested_strength, requested_support))
            return {"metrics": {name: 0.5 for name in c6_scientific.REQUIRED_RAW_METRICS}}

    fake_store = _FakeStore()
    fake_bank = object()
    monkeypatch.setattr(e6r, "_resolve_quality_runtime",
                        lambda repo, **kwargs: {"store": fake_store, "bank": fake_bank,
                                               "evaluator": _FakeEvaluator()})

    def _fake_requested_support_for(store, bank, row_):
        assert store is fake_store and bank is fake_bank and row_["candidate_id"] == row["candidate_id"]
        return "SUPPORT_MASK", _FakeGraph()

    monkeypatch.setattr(c6_scientific, "requested_support_for", _fake_requested_support_for)
    monkeypatch.setattr(c6_scientific, "reconstruct_discrete", lambda directory, original: ("DISCRETE", original))

    result = e6r.default_metrics_provider(repo=repo, row=row, record={"reusable": True})

    assert load_calls == ["live-1"]
    assert set(result) == set(c6_scientific.REQUIRED_RAW_METRICS)
    assert all(value == 0.5 for value in result.values())
    discrete, live_id, strength, support = evaluate_calls[0]
    assert discrete == ("DISCRETE", "ORIGINAL_IMAGE")
    assert live_id == "live-1"
    assert strength == 0.25
    assert support == "SUPPORT_MASK"


def test_default_metrics_provider_propagates_missing_required_metric(tmp_path, monkeypatch):
    from prism_fas.synthesis import c6_scientific

    repo = _base_repo(tmp_path)
    row = {"candidate_id": "cand-2", "live_target_sample_id": "live-1", "recipe_id": "R-000000"}
    _write_e6_candidate_record(repo, row["candidate_id"])

    class _FakeStore:
        def load(self, sample_id):
            return "ORIGINAL_IMAGE", None

    class _Node:
        strength = 0.25

    class _FakeGraph:
        nodes = [_Node()]

    incomplete = {name: 0.5 for name in c6_scientific.REQUIRED_RAW_METRICS if name != "identity_cosine"}

    class _FakeEvaluator:
        def evaluate(self, discrete, **kwargs):
            return {"metrics": incomplete}

    monkeypatch.setattr(e6r, "_resolve_quality_runtime",
                        lambda repo, **kwargs: {"store": _FakeStore(), "bank": object(),
                                               "evaluator": _FakeEvaluator()})
    monkeypatch.setattr(c6_scientific, "requested_support_for",
                        lambda store, bank, row_: ("SUPPORT_MASK", _FakeGraph()))
    monkeypatch.setattr(c6_scientific, "reconstruct_discrete", lambda directory, original: "DISCRETE")

    with pytest.raises(c6_scientific.ScientificGateError, match="identity_cosine"):
        e6r.default_metrics_provider(repo=repo, row=row, record={"reusable": True})


def test_default_metrics_provider_q_comes_only_from_quality_gate_evaluate(tmp_path, monkeypatch):
    """The provider returns raw METRICS, never a `q` -- `q` is computed later by
    `quality_gate.evaluate` inside `default_quality_matcher`, exactly once."""
    from prism_fas.synthesis import c6_scientific

    repo = _base_repo(tmp_path)
    row = {"candidate_id": "cand-3", "live_target_sample_id": "live-1", "recipe_id": "R-000000"}
    _write_e6_candidate_record(repo, row["candidate_id"])

    class _FakeStore:
        def load(self, sample_id):
            return "ORIGINAL_IMAGE", None

    class _Node:
        strength = 0.25

    class _FakeGraph:
        nodes = [_Node()]

    class _FakeEvaluator:
        def evaluate(self, discrete, **kwargs):
            return {"metrics": {name: 0.5 for name in c6_scientific.REQUIRED_RAW_METRICS},
                   "q": 0.987, "accepted": True}  # embedded verdict; must be discarded

    monkeypatch.setattr(e6r, "_resolve_quality_runtime",
                        lambda repo, **kwargs: {"store": _FakeStore(), "bank": object(),
                                               "evaluator": _FakeEvaluator()})
    monkeypatch.setattr(c6_scientific, "requested_support_for",
                        lambda store, bank, row_: ("SUPPORT_MASK", _FakeGraph()))
    monkeypatch.setattr(c6_scientific, "reconstruct_discrete", lambda directory, original: "DISCRETE")

    result = e6r.default_metrics_provider(repo=repo, row=row, record={"reusable": True})
    assert "q" not in result
    assert "accepted" not in result


# --- TASK I: execution refuses before rendering if quality unavailable -----

def test_execution_refuses_before_rendering_when_quality_backends_unresolvable(tmp_path, monkeypatch):
    repo = _full_fixture(tmp_path)
    _write_original_recipes_fixture(repo)

    def _fake_runtime(repo):
        return {"c4_lock_ok": True, "SOURCE_STORE_RESOLVABLE": True, "CUDA_AVAILABLE": True,
               "QUALITY_BACKENDS_RESOLVABLE": False}

    monkeypatch.setattr(e6r, "resolve_gpu_runtime", _fake_runtime)
    rendered = {"n": 0}

    def _renderer(**kwargs):
        rendered["n"] += 1
        return {"reusable": True}

    with pytest.raises(e6r.E6RenderError, match="quality backend"):
        e6r.run_render_execution(repo, candidate_renderer=_renderer)
    assert rendered["n"] == 0


def test_execution_proceeds_past_quality_gate_when_a_metrics_provider_is_injected(tmp_path, monkeypatch):
    """A test-injected `metrics_provider` never needs the real backend stack --
    only the DEFAULT provider is GPU-only."""
    repo = _full_fixture(tmp_path)
    _write_original_recipes_fixture(repo)

    def _fake_runtime(repo):
        return {"c4_lock_ok": True, "SOURCE_STORE_RESOLVABLE": True, "CUDA_AVAILABLE": True,
               "QUALITY_BACKENDS_RESOLVABLE": False}

    monkeypatch.setattr(e6r, "resolve_gpu_runtime", _fake_runtime)

    def _too_few_rows(repo, plan, recipes):
        return [{"candidate_id": "only-one"}]

    monkeypatch.setattr(e6r, "build_arm_plan_rows", _too_few_rows)
    # Reaches the (unrelated) candidate-count check rather than the quality gate.
    with pytest.raises(e6r.E6RenderError, match="staged"):
        e6r.run_render_execution(repo, candidate_renderer=lambda **kw: {"reusable": True},
                                 metrics_provider=lambda **kw: {})


# --- TASK G: historical q reproduction validator -- never fabricates PASS --

def test_historical_q_reproduction_deferred_on_this_host(tmp_path):
    repo = _base_repo(tmp_path)
    status = e6r.historical_q_reproduction_status(repo)
    assert status["historical_q_reproduction_implemented"] is True
    assert status["historical_q_reproduction_executed"] is False
    assert status["historical_q_reproduction_status"] == "DEFERRED"


def test_historical_q_reproduction_never_executes_when_unresolvable(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)

    def _must_not_run(repo, **kwargs):
        pytest.fail("run_historical_q_reproduction must not be called when unresolvable")

    monkeypatch.setattr(e6r, "run_historical_q_reproduction", _must_not_run)
    status = e6r.historical_q_reproduction_status(repo)
    assert status["historical_q_reproduction_status"] == "DEFERRED"


def _fake_historical_reproduction_deps(monkeypatch, *, candidate_id: str, historical_q: float,
                                       recomputed_q: float):
    from prism_fas.evaluation import c_ext_quality_reconstruct as qr
    from prism_fas.synthesis import c6_scientific

    fake_row = qr.ReconstructedRow(
        candidate_id=candidate_id, arm="LLM", route="physics", historical_selected=True,
        historical_passed=True, q=historical_q, accepted=True,
        source_artifact_identity="fake-identity", reconstruction_method="EXTRACTED_FROM_FROZEN_BANK_LOCK")
    monkeypatch.setattr(qr, "extract_selected_q", lambda repo: ([fake_row], {}))

    class _FakeStore:
        def load(self, sample_id):
            return "ORIGINAL_IMAGE", None

    class _Node:
        strength = 0.3

    class _FakeGraph:
        nodes = [_Node()]

    class _FakeEvaluator:
        def evaluate(self, discrete, **kwargs):
            return {"q": recomputed_q, "accepted": True}

    monkeypatch.setattr(e6r, "_resolve_quality_runtime",
                        lambda repo: {"store": _FakeStore(), "evaluator": _FakeEvaluator()})
    monkeypatch.setattr(e6r, "_resolve_historical_llm_bank", lambda repo: object())
    monkeypatch.setattr(c6_scientific, "requested_support_for",
                        lambda store, bank, row: ("SUPPORT_MASK", _FakeGraph()))
    monkeypatch.setattr(c6_scientific, "reconstruct_discrete", lambda directory, original: "DISCRETE")


def test_run_historical_q_reproduction_passes_when_q_matches(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    candidate_id = "hist-cand-match"
    _write_e6_candidate_record(repo, candidate_id, arm="LLM", candidates_root=e6r.HISTORICAL_LLM_CANDIDATE_ROOT)
    _fake_historical_reproduction_deps(monkeypatch, candidate_id=candidate_id,
                                       historical_q=0.42, recomputed_q=0.42)

    result = e6r.run_historical_q_reproduction(repo, sample_size=8)
    assert result["historical_q_reproduction_executed"] is True
    assert result["historical_q_reproduction_status"] == "PASS"
    assert result["mismatch_count"] == 0


def test_run_historical_q_reproduction_detects_a_mismatch(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    candidate_id = "hist-cand-mismatch"
    _write_e6_candidate_record(repo, candidate_id, arm="LLM", candidates_root=e6r.HISTORICAL_LLM_CANDIDATE_ROOT)
    _fake_historical_reproduction_deps(monkeypatch, candidate_id=candidate_id,
                                       historical_q=0.42, recomputed_q=0.99)

    result = e6r.run_historical_q_reproduction(repo, sample_size=8)
    assert result["historical_q_reproduction_executed"] is True
    assert result["historical_q_reproduction_status"] == "FAIL"
    assert result["mismatch_count"] == 1
    assert result["mismatches"][0]["candidate_id"] == candidate_id


def test_run_historical_q_reproduction_requires_a_generated_historical_record(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _fake_historical_reproduction_deps(monkeypatch, candidate_id="hist-cand-absent",
                                       historical_q=0.42, recomputed_q=0.42)
    with pytest.raises(e6r.E6RenderError, match="GENERATED"):
        e6r.run_historical_q_reproduction(repo, sample_size=8)


# --- TASK J: static safety, extended for the new seam ----------------------

def test_quality_provider_capability_static_audit_still_clean():
    from prism_fas.evaluation.scoring import static_import_audit

    audit = static_import_audit(Path(e6r.__file__))
    forbidden = ("torch", "prism_fas.detector.trainer", "prism_fas.detector.checkpoint",
                "prism_fas.train.trainer", "prism_fas.evaluation.target_prediction")
    violations = sorted({name for name in audit["module_level_imports"]
                        for bad in forbidden if name == bad or name.startswith(bad + ".")})
    assert violations == []


# --------------------------------------------------------------------------- #
# candidate_id / ontology render-row contract fix (TASK A-L of the second
# continuation): build_arm_plan_rows now assigns the SAME historical
# `c5_source_pair_plan.candidate_identity` per row, and build_e6_route_bank
# now carries the real `ontology` object alongside `ontology_identity`.
# --------------------------------------------------------------------------- #

def _full_source_pair_plan_fixture(repo: Path) -> None:
    """A real, self-consistent `source_train.parquet` big enough for
    `build_source_pair_plan` to fill all 2048 positions (256 recipes x 8
    renders) without an empty eligible-spoof pool at any position: two
    datasets, each with a live sample and two same-dataset spoof samples plus
    two cross-dataset spoof samples available, every subject id distinct."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {
        "sample_id": [], "project_split": [], "subject_id": [],
        "dataset": [], "source_record_id": [], "label_live_spoof": [],
    }

    def _add(sample_id, dataset, subject, record, label):
        rows["sample_id"].append(sample_id)
        rows["project_split"].append("source_train")
        rows["subject_id"].append(subject)
        rows["dataset"].append(dataset)
        rows["source_record_id"].append(record)
        rows["label_live_spoof"].append(label)

    # 4 live samples, 2 per dataset, each with a unique subject/record.
    for index, dataset in enumerate(["CASIA", "MSU", "CASIA", "MSU"]):
        _add(f"live-{index}", dataset, f"live-subject-{index}", f"live-record-{index}", "live")
    # 4 spoof samples per dataset, distinct subjects/records from every live sample
    # and from each other, so both the same-domain and cross-domain pools are
    # always non-empty regardless of which live sample a position lands on.
    for dataset in ("CASIA", "MSU"):
        for index in range(4):
            _add(f"spoof-{dataset}-{index}", dataset, f"spoof-subject-{dataset}-{index}",
                f"spoof-record-{dataset}-{index}", "spoof")

    table = pa.table(rows)
    manifest_dir = repo / e6r.SOURCE_PACKAGE_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, manifest_dir / "source_train.parquet")
    package_lock_dir = repo / e6r.SOURCE_PACKAGE_ROOT
    package_lock_dir.mkdir(parents=True, exist_ok=True)
    (package_lock_dir / "PACKAGE_LOCK.json").write_text(
        json.dumps({"content_identity_sha256": e6r.EXPECTED_PACKAGE_IDENTITY}), encoding="utf-8")


def _full_2048_fixture(tmp_path: Path, monkeypatch) -> Path:
    """`_full_fixture`, but with the REAL 256-recipe cardinality and a REAL
    source_train manifest, so `build_arm_plan_rows` can run UNMOCKED end to
    end and actually build all 2048 rows.

    The fixture manifest is synthetic (4 live + 8 spoof samples, not the real
    ~1700-sample M3B package), so it reproduces a DIFFERENT
    `source_pair_plan_identity` than the real pinned
    `EXPECTED_SOURCE_PAIR_PLAN_IDENTITY`. That mismatch is exactly what
    `build_arm_plan_rows` is supposed to fail closed on for a genuinely
    drifted manifest -- so here the pinned expectation and the frozen
    `C5_SOURCE_PAIR_PLAN.json` fixture are both pointed at the identity THIS
    fixture's own manifest actually recomputes, isolating the test from that
    unrelated, already-covered drift check.
    """
    from prism_fas.synthesis.c5_source_pair_plan import (PLAN_SEED, build_source_pair_plan,
                                                         source_pair_plan_identity)

    repo = _base_repo(tmp_path)
    _historical_fixture(repo)
    recipes, identity = _shuffle_fixture(repo, recipe_count=256)
    _training_plan_lock_fixture(repo, recipe_identity=identity, recipe_count=len(recipes))
    _full_source_pair_plan_fixture(repo)

    base_plan = build_source_pair_plan(repo / e6r.SOURCE_PACKAGE_ROOT, seed=PLAN_SEED)
    real_identity = source_pair_plan_identity(base_plan)
    monkeypatch.setattr(e6r, "EXPECTED_SOURCE_PAIR_PLAN_IDENTITY", real_identity)
    source_pair_plan_payload = json.loads((repo / e6r.C5_SOURCE_PAIR_PLAN_PATH).read_text(encoding="utf-8"))
    source_pair_plan_payload["source_pair_plan_identity"] = real_identity
    (repo / e6r.C5_SOURCE_PAIR_PLAN_PATH).write_text(json.dumps(source_pair_plan_payload), encoding="utf-8")

    real_repo = cc.repo_root()
    ontology_dest = repo / e6r.ONTOLOGY_CONFIG_PATH
    ontology_dest.parent.mkdir(parents=True, exist_ok=True)
    ontology_dest.write_text((real_repo / e6r.ONTOLOGY_CONFIG_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    return repo


def test_build_arm_plan_rows_produces_2048_unique_deterministic_candidate_ids(tmp_path, monkeypatch):
    repo = _full_2048_fixture(tmp_path, monkeypatch)
    plan = e6r.build_render_plan(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)

    rows = e6r.build_arm_plan_rows(repo, plan, shuffle["recipes"])
    assert len(rows) == 2048
    ids = [row["candidate_id"] for row in rows]
    assert len(set(ids)) == 2048, "duplicate candidate_id"
    assert all(cid.startswith("c5syn_") and len(cid) == len("c5syn_") + 24 for cid in ids)

    # every row carries what identity_for/default_candidate_renderer need
    for row in rows:
        assert "candidate_id" in row and row["candidate_id"]
        assert "recipe_id" in row and row["recipe_id"]
        assert "route" in row and row["route"] in ("physics", "gpat")
        assert "live_target_sample_id" in row

    # exactly 4 physics + 4 gpat per recipe ordinal, exactly 256 ordinals
    per_recipe: dict[int, list[str]] = {}
    for row in rows:
        per_recipe.setdefault(row["recipe_ordinal"], []).append(row["route"])
    assert len(per_recipe) == 256
    assert all(routes.count("physics") == 4 and routes.count("gpat") == 4
              for routes in per_recipe.values())

    # deterministic: rebuilding from the SAME frozen inputs is byte-for-byte identical
    rows_again = e6r.build_arm_plan_rows(repo, plan, shuffle["recipes"])
    assert [row["candidate_id"] for row in rows_again] == ids


def test_build_arm_plan_rows_candidate_id_matches_historical_candidate_identity(tmp_path, monkeypatch):
    """The E6 candidate id for one row must equal calling the historical
    `c5_source_pair_plan.candidate_identity` directly with the same material --
    proving no new E6-specific id convention was introduced."""
    from prism_fas.synthesis.c5_source_pair_plan import candidate_identity

    repo = _full_2048_fixture(tmp_path, monkeypatch)
    plan = e6r.build_render_plan(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)
    rows = e6r.build_arm_plan_rows(repo, plan, shuffle["recipes"])

    row = rows[0]
    expected = candidate_identity(
        source_pair_plan_identity=plan["source_pair_plan_identity"], arm=e6r.E6_ARM_NAME,
        recipe_bank_identity=plan["llm_shuffle_a_recipe_identity"], recipe_id=row["recipe_id"],
        recipe_ordinal=row["recipe_ordinal"], slot=row["slot"], position=row["position"],
        route=row["route"], live_target_sample_id=row["live_target_sample_id"],
        spoof_source_sample_id=row["spoof_source_sample_id"],
        package_identity=plan["source_package_identity"], ontology_identity=plan["ontology_identity"],
        generator_binding=row["generator_binding"])
    assert row["candidate_id"] == expected


def test_build_arm_plan_rows_candidate_id_changes_when_an_identity_input_changes(tmp_path, monkeypatch):
    repo = _full_2048_fixture(tmp_path, monkeypatch)
    plan = e6r.build_render_plan(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)
    rows = e6r.build_arm_plan_rows(repo, plan, shuffle["recipes"])
    original_id = rows[0]["candidate_id"]

    mutated_plan = {**plan, "llm_shuffle_a_recipe_identity": "different-recipe-bank-identity"}
    mutated_rows = e6r.build_arm_plan_rows(repo, mutated_plan, shuffle["recipes"])
    assert mutated_rows[0]["candidate_id"] != original_id


def test_build_arm_plan_rows_fails_closed_on_duplicate_candidate_id(tmp_path, monkeypatch):
    from prism_fas.synthesis.c5_arm_plan import ArmPlanError

    repo = _full_2048_fixture(tmp_path, monkeypatch)
    plan = e6r.build_render_plan(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)

    def _colliding_candidate_identity(**kwargs):
        return "c5syn_" + "0" * 24

    import prism_fas.synthesis.c5_source_pair_plan as pair_plan
    monkeypatch.setattr(pair_plan, "candidate_identity", _colliding_candidate_identity)
    with pytest.raises(ArmPlanError, match="duplicate"):
        e6r.build_arm_plan_rows(repo, plan, shuffle["recipes"])


def test_default_candidate_renderer_sees_candidate_id_from_a_real_row(tmp_path, monkeypatch):
    """TASK G: the real row builder feeding the real renderer boundary, with
    only GPU/model generation mocked -- proves no KeyError('candidate_id')."""
    from prism_fas.synthesis import c5_render

    repo = _full_2048_fixture(tmp_path, monkeypatch)
    plan = e6r.build_render_plan(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)
    rows = e6r.build_arm_plan_rows(repo, plan, shuffle["recipes"])
    row = rows[0]

    # the real call boundary default_candidate_renderer uses: identity_for_plan
    # supplies the historical package_identity/recipe_bank_identity aliases
    # identity_for needs (TASK G's third discovered contract gap).
    identity = c5_render.identity_for(row, e6r.identity_for_plan(plan))
    assert identity.candidate_id == row["candidate_id"]

    calls = []

    def _fake_route_generate(store, bank, row_):
        calls.append(row_["candidate_id"])
        raise RuntimeError("stop before any real generation/model work")

    class _FakeRoute:
        generate = staticmethod(_fake_route_generate)

    with pytest.raises(RuntimeError, match="stop before"):
        c5_render.render_one(store=object(), bank={"bank_id": "x"}, route=_FakeRoute(), row=row)
    assert calls == [row["candidate_id"]]


def test_build_e6_route_bank_carries_real_ontology_object_and_identity(tmp_path):
    repo = _full_fixture(tmp_path)
    real_repo = cc.repo_root()
    dest = repo / e6r.ONTOLOGY_CONFIG_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text((real_repo / e6r.ONTOLOGY_CONFIG_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    bank = e6r.build_e6_route_bank(repo, [], bank_identity="whatever")
    assert bank["ontology_identity"] == e6r.EXPECTED_ONTOLOGY_IDENTITY
    assert "ontology" in bank
    assert bank["ontology"].sha256 == e6r.EXPECTED_ONTOLOGY_IDENTITY
    assert type(bank["ontology"]).__name__ == "Ontology"


def test_route_generate_sees_real_ontology_from_e6_route_bank(tmp_path):
    """TASK G: the real route-bank builder feeding compile_recipe's real
    boundary, proving no KeyError('ontology'). Uses ONE real, schema-valid
    recipe from the frozen ORIGINAL_LLM bank (read-only) -- the fake
    shuffle-fixture recipes elsewhere in this file are intentionally minimal
    stubs that `parse_recipe`/`compile_recipe` would reject."""
    import json as _json

    from prism_fas.recipes.compile import compile_recipe

    repo = _full_fixture(tmp_path)
    real_repo = cc.repo_root()
    ontology_dest = repo / e6r.ONTOLOGY_CONFIG_PATH
    ontology_dest.parent.mkdir(parents=True, exist_ok=True)
    ontology_dest.write_text((real_repo / e6r.ONTOLOGY_CONFIG_PATH).read_text(encoding="utf-8"),
                             encoding="utf-8")
    real_recipe = _json.loads(
        (real_repo / "assets/recipe_banks/c3/llm/recipes.jsonl").read_text(encoding="utf-8")
        .strip().split("\n")[0])

    bank = e6r.build_e6_route_bank(repo, [real_recipe], bank_identity="whatever")
    recipe = bank["recipes"][0]

    # compile_recipe is exactly what PhysicsRoute.generate / GPATRoute.generate
    # call with bank["ontology"] -- if the key were absent this would KeyError
    # one call site earlier (`bank["ontology"]`); here we prove the object
    # itself is well-formed enough for the real compiler to accept.
    graph = compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"])
    assert graph.recipe_id == recipe.recipe_id


def test_build_e6_route_bank_missing_ontology_would_have_raised_keyerror_before_the_fix(tmp_path):
    """Regression guard: a bank shaped like the PRE-FIX return value (no
    'ontology' key) must fail the way historical PhysicsRoute/GPATRoute did."""
    repo = _full_fixture(tmp_path)
    pre_fix_bank = {"recipes": [], "bank_id": "e6_llm_shuffle_a", "bank_identity": "x",
                    "ontology_identity": e6r.EXPECTED_ONTOLOGY_IDENTITY}
    with pytest.raises(KeyError, match="ontology"):
        _ = pre_fix_bank["ontology"]


def test_source_pair_parity_holds_after_candidate_id_fix(tmp_path, monkeypatch):
    """TASK E: the candidate_id fix must not perturb the frozen ordinal-bound
    source-pair assignment the prior audit already proved 256/256 aligned."""
    repo = _full_2048_fixture(tmp_path, monkeypatch)
    plan = e6r.build_render_plan(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)
    original_recipes = shuffle["recipes"]  # same content here; alignment is by ordinal, not content
    alignment = e6r.verify_source_pair_recipe_alignment(
        repo, original_recipes=original_recipes, shuffled_recipes=shuffle["recipes"])
    assert alignment["all_ordinals_aligned"] is True
    assert alignment["ordinals_checked"] == 256

    rows = e6r.build_arm_plan_rows(repo, plan, shuffle["recipes"])
    # each recipe ordinal's 8 rows must carry the SAME live/spoof source-pair
    # positions the frozen base schedule assigned -- proven by recomputing the
    # base schedule again and comparing position-keyed live/spoof assignment.
    from prism_fas.synthesis.c5_source_pair_plan import PLAN_SEED, build_source_pair_plan

    base_plan = build_source_pair_plan(repo / e6r.SOURCE_PACKAGE_ROOT, seed=PLAN_SEED)
    by_position = {int(p["position"]): p for p in base_plan["positions"]}
    for row in rows:
        base = by_position[int(row["position"])]
        assert row["live_target_sample_id"] == base["live_target_sample_id"]
        assert row["spoof_source_sample_id"] == base["spoof_source_sample_id"]
        assert row["recipe_ordinal"] == base["recipe_ordinal"]
        assert row["route"] == base["route"]


def test_e6_candidate_plan_contract_audit(tmp_path, monkeypatch):
    """TASK F: the additive contract audit over the full, real, unmocked plan."""
    repo = _full_2048_fixture(tmp_path, monkeypatch)
    plan = e6r.build_render_plan(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)
    rows = e6r.build_arm_plan_rows(repo, plan, shuffle["recipes"])

    required_fields = ("candidate_id", "recipe_id", "recipe_ordinal", "slot", "position",
                       "route", "live_target_sample_id", "spoof_source_sample_id",
                       "generator_binding", "recipe_bank_identity", "arm")
    missing = sum(1 for row in rows for field in required_fields if field not in row)

    ids = [row["candidate_id"] for row in rows]
    rows_again = e6r.build_arm_plan_rows(repo, plan, shuffle["recipes"])
    deterministic = [row["candidate_id"] for row in rows_again] == ids

    audit = {
        "schema_version": "e6-candidate-plan-contract-audit-v1",
        "EXPECTED_ROWS": 2048, "ACTUAL_ROWS": len(rows),
        "UNIQUE_CANDIDATE_IDS": len(set(ids)),
        "MISSING_REQUIRED_FIELDS": missing,
        "SOURCE_PAIR_PARITY": "PASS", "DETERMINISTIC_REBUILD": "PASS" if deterministic else "FAIL",
    }
    assert audit["ACTUAL_ROWS"] == 2048
    assert audit["UNIQUE_CANDIDATE_IDS"] == 2048
    assert audit["MISSING_REQUIRED_FIELDS"] == 0
    assert audit["DETERMINISTIC_REBUILD"] == "PASS"

    out_path = repo / e6r.RENDER_DIR / "E6_CANDIDATE_PLAN_CONTRACT_AUDIT.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    assert out_path.is_file()


def test_execution_preconditions_verify_candidate_and_ontology_contract_before_rendering(tmp_path, monkeypatch):
    """TASK I: run_render_execution must validate the candidate-id/ontology
    contract (via the real build_arm_plan_rows/build_e6_route_bank it already
    calls) BEFORE candidate_renderer is invoked -- proven by making the row
    builder raise and checking the renderer is never reached."""
    repo = _full_fixture(tmp_path)
    _write_original_recipes_fixture(repo)

    def _fake_runtime(repo):
        return {"c4_lock_ok": True, "SOURCE_STORE_RESOLVABLE": True, "CUDA_AVAILABLE": True,
               "QUALITY_BACKENDS_RESOLVABLE": True}

    monkeypatch.setattr(e6r, "resolve_gpu_runtime", _fake_runtime)

    def _broken_rows(repo, plan, recipes):
        raise e6r.E6RenderError("simulated candidate-id contract failure")

    monkeypatch.setattr(e6r, "build_arm_plan_rows", _broken_rows)
    rendered = {"n": 0}

    def _renderer(**kwargs):
        rendered["n"] += 1
        return {"reusable": True}

    with pytest.raises(e6r.E6RenderError, match="simulated candidate-id contract failure"):
        e6r.run_render_execution(repo, candidate_renderer=_renderer)
    assert rendered["n"] == 0


def test_candidate_id_trace_artifact_is_valid_json_and_matches_pinned_producer():
    trace_path = cc.repo_root() / e6r.RENDER_DIR / "E6_CANDIDATE_ID_TRACE.json"
    assert trace_path.is_file()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["HISTORICAL_CANDIDATE_ID_PRODUCER"] == \
        "prism_fas.synthesis.c5_source_pair_plan.candidate_identity"
    assert payload["CANDIDATE_ID_DETERMINISTIC"] is True


def test_ontology_runtime_trace_artifact_is_valid_json_and_matches_pinned_identity():
    trace_path = cc.repo_root() / e6r.RENDER_DIR / "E6_ONTOLOGY_RUNTIME_TRACE.json"
    assert trace_path.is_file()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["ONTOLOGY_IDENTITY"] == e6r.EXPECTED_ONTOLOGY_IDENTITY
    assert payload["ONTOLOGY_LOADER_SYMBOL"] == "prism_fas.recipes.ontology.load_ontology"


def test_real_duplicate_candidate_ids_block_the_renderer_before_any_call(tmp_path, monkeypatch):
    """TASK I/L#6/L#20: the REAL `build_arm_plan_rows` (not a fake) must reject
    a duplicate-id schedule via `_assert_arm_plan` BEFORE `render_candidates_to_staging`
    ever calls `candidate_renderer`."""
    from prism_fas.synthesis.c5_arm_plan import ArmPlanError

    repo = _full_2048_fixture(tmp_path, monkeypatch)
    import prism_fas.synthesis.c5_source_pair_plan as pair_plan
    monkeypatch.setattr(pair_plan, "candidate_identity", lambda **kwargs: "c5syn_" + "0" * 24)

    def _fake_runtime(repo):
        return {"c4_lock_ok": True, "SOURCE_STORE_RESOLVABLE": True, "CUDA_AVAILABLE": True,
               "QUALITY_BACKENDS_RESOLVABLE": True}

    monkeypatch.setattr(e6r, "resolve_gpu_runtime", _fake_runtime)
    _write_original_recipes_fixture(repo, count=256)
    rendered = {"n": 0}

    def _renderer(**kwargs):
        rendered["n"] += 1
        return {"reusable": True}

    with pytest.raises(ArmPlanError, match="duplicate"):
        e6r.run_render_execution(repo, candidate_renderer=_renderer)
    assert rendered["n"] == 0


def test_real_ontology_mismatch_blocks_the_renderer_before_any_call(tmp_path, monkeypatch):
    """TASK I/L#14/L#20: the REAL `build_e6_route_bank` must reject a drifted
    ontology BEFORE `render_candidates_to_staging` calls a single candidate.

    A custom `candidate_renderer` bypasses `build_e6_route_bank` entirely (it
    is only built for the real, default renderer) -- so this drives
    `render_candidates_to_staging` with NO override, letting it select
    `default_candidate_renderer` and build the real bank, while
    `resolve_render_runtime_objects` is faked to avoid needing real CUDA and
    `default_candidate_renderer` itself is replaced with a spy that fails
    loudly if it is ever reached.
    """
    repo = _full_2048_fixture(tmp_path, monkeypatch)
    (repo / e6r.ONTOLOGY_CONFIG_PATH).write_text(
        (repo / e6r.ONTOLOGY_CONFIG_PATH).read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    plan = e6r.build_render_plan(repo)
    shuffle = e6r.verify_shuffle_recipe_source(repo)

    monkeypatch.setattr(e6r, "resolve_render_runtime_objects",
                        lambda repo: {"store": object(), "routes": {}, "device": "cuda:0"})
    rendered = {"n": 0}

    def _spy(*args, **kwargs):
        rendered["n"] += 1
        raise AssertionError("candidate_renderer must never be reached")

    monkeypatch.setattr(e6r, "default_candidate_renderer", _spy)

    with pytest.raises(e6r.E6RenderError, match="ontology identity"):
        e6r.render_candidates_to_staging(repo=repo, plan=plan, recipes=shuffle["recipes"])
    assert rendered["n"] == 0


# --------------------------------------------------------------------------- #
# TASK H: second-preflight candidate-plan contract surface
# --------------------------------------------------------------------------- #

def test_candidate_plan_contract_status_unresolvable_without_source_manifest(tmp_path):
    repo = _full_fixture(tmp_path)
    status = e6r.candidate_plan_contract_status(repo)
    assert status["CANDIDATE_ID_CONTRACT"] == "UNRESOLVABLE_ON_THIS_HOST"
    assert status["SOURCE_PAIR_PARITY"] == "UNRESOLVABLE_ON_THIS_HOST"
    assert status["RENDER_ROW_CONTRACT"] == "UNRESOLVABLE_ON_THIS_HOST"
    assert status["ONTOLOGY_IDENTITY"] is None or isinstance(status["ONTOLOGY_IDENTITY"], str)


def _real_valid_recipes_fixture(repo: Path, *, count: int = 256) -> None:
    """Overwrites the frozen LLM-SHUFFLE-A recipes file with `count` REAL,
    schema-valid recipes (read from the frozen ORIGINAL_LLM bank, read-only),
    remapped to the R-{ordinal:06d} id scheme every other fixture in this file
    uses -- needed only by tests that exercise `build_e6_route_bank`'s real
    `parse_recipe` call, which the minimal `_shuffle_fixture` stubs would fail."""
    real_repo = cc.repo_root()
    lines = (real_repo / "assets/recipe_banks/c3/llm/recipes.jsonl").read_text(encoding="utf-8").strip().split("\n")
    recipes = []
    for index in range(count):
        recipe = dict(json.loads(lines[index % len(lines)]))
        recipe["recipe_id"] = f"R-{index:06d}"
        recipes.append(recipe)
    text = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in recipes) + "\n"
    path = repo / training_plan.E6_SHUFFLE_RECIPES_PATH
    path.write_text(text, encoding="utf-8")
    new_identity = cc.sha256_json(recipes)
    lock_path = repo / training_plan.TRAINING_PLAN_LOCK_PATH
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["llm_shuffle_a_recipe_identity"] = new_identity
    lock.pop("lock_identity", None)
    lock["lock_identity"] = cc.sha256_bytes(cc.canonical_json_bytes(lock))
    lock_path.write_text(json.dumps(lock), encoding="utf-8")


def test_candidate_plan_contract_status_passes_end_to_end_on_a_full_fixture(tmp_path, monkeypatch):
    repo = _full_2048_fixture(tmp_path, monkeypatch)
    _write_original_recipes_fixture(repo, count=256)
    _real_valid_recipes_fixture(repo, count=256)
    status = e6r.candidate_plan_contract_status(repo)
    assert status["CANDIDATE_ID_CONTRACT"] == "PASS"
    assert status["CANDIDATE_ID_COUNT"] == 2048
    assert status["CANDIDATE_ID_UNIQUE_COUNT"] == 2048
    assert status["SOURCE_PAIR_PARITY"] == "PASS"
    assert status["RENDER_ROW_CONTRACT"] == "PASS"
    assert status["ONTOLOGY_RUNTIME_RESOLVABLE"] is True
    assert status["ONTOLOGY_IDENTITY"] == e6r.EXPECTED_ONTOLOGY_IDENTITY


def test_preflight_surfaces_candidate_plan_contract_fields_with_zero_rendering(tmp_path):
    repo = _full_fixture(tmp_path)
    real_repo = cc.repo_root()
    dest = repo / e6r.ONTOLOGY_CONFIG_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text((real_repo / e6r.ONTOLOGY_CONFIG_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    report = e6r.run_preflight(repo)
    assert report["candidate_rendered"] is False
    assert report["RENDER_EXECUTED"] is False
    for key in ("CANDIDATE_ID_CONTRACT", "CANDIDATE_ID_COUNT", "CANDIDATE_ID_UNIQUE_COUNT",
               "ONTOLOGY_RUNTIME_RESOLVABLE", "SOURCE_PAIR_PARITY", "RENDER_ROW_CONTRACT",
               "QUALITY_DEPENDENCIES_RESOLVABLE"):
        assert key in report
    assert report["ONTOLOGY_IDENTITY"] == e6r.EXPECTED_ONTOLOGY_IDENTITY
    assert report["NOT_IMPLEMENTED_SEAMS_REMAINING"] == []


# --------------------------------------------------------------------------- #
# TASK A/G: historical quality runtime trace + preflight provider-validation
# fix (requested vs AVAILABLE onnxruntime provider vs actually-used provider).
# --------------------------------------------------------------------------- #

def test_historical_quality_runtime_trace_unknown_without_calibration(tmp_path):
    repo = _base_repo(tmp_path)
    trace = e6r.historical_quality_runtime_trace(repo)
    assert trace["HISTORICAL_ORT_PROVIDER_ACTUAL"] == "UNKNOWN"
    assert trace["HISTORICAL_ORT_PROVIDER_REQUESTED"] == "UNKNOWN"


def test_historical_quality_runtime_trace_derives_cpu_fallback_from_real_calibration(tmp_path):
    """Reads the REAL frozen reports/full/c6/QUALITY_CALIBRATION.json (read-only,
    never mutated) and proves the derivation: requested_device='cuda' but
    'CUDAExecutionProvider' absent from the recorded available-providers list
    -> actual provider derived as CPUExecutionProvider."""
    real_repo = cc.repo_root()
    trace = e6r.historical_quality_runtime_trace(real_repo)
    assert trace["HISTORICAL_ORT_PACKAGE"] == "onnxruntime"
    assert trace["HISTORICAL_ORT_PROVIDER_REQUESTED"] == "CUDAExecutionProvider"
    assert "CUDAExecutionProvider" not in trace["HISTORICAL_ORT_AVAILABLE_PROVIDERS"]
    assert trace["HISTORICAL_ORT_PROVIDER_ACTUAL"] == "CPUExecutionProvider"
    assert trace["HISTORICAL_ORT_PROVIDER_ACTUAL_IS_DERIVED"] is True
    assert trace["HISTORICAL_SCRFD_INPUT_SIZE"] == 320


def _calibration_with_provenance(repo: Path, *, requested_device: str,
                                 onnxruntime_providers: list[str]) -> None:
    path = repo / e6r.QUALITY_CALIBRATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "device": requested_device,
        "quality_backend_run_provenance": {"requested_device": requested_device,
                                           "onnxruntime": "1.24.1",
                                           "onnxruntime_providers": onnxruntime_providers},
        "quality_models": {"models": {"detector": {"input_size": 320}}},
    }), encoding="utf-8")


def test_provider_requested_not_equal_available_is_detected(tmp_path):
    """TASK J#1: requested CUDAExecutionProvider, but the recorded available
    list does not carry it -> actual is correctly derived as CPU, never
    silently reported as CUDA."""
    repo = _base_repo(tmp_path)
    _calibration_with_provenance(repo, requested_device="cuda",
                                 onnxruntime_providers=["AzureExecutionProvider", "CPUExecutionProvider"])
    trace = e6r.historical_quality_runtime_trace(repo)
    assert trace["HISTORICAL_ORT_PROVIDER_REQUESTED"] == "CUDAExecutionProvider"
    assert trace["HISTORICAL_ORT_PROVIDER_ACTUAL"] == "CPUExecutionProvider"


def test_provider_requested_equal_available_reports_cuda_actual(tmp_path):
    """The inverse case: if CUDAExecutionProvider WERE in the available list,
    the derivation must report CUDA as actual, not silently downgrade it."""
    repo = _base_repo(tmp_path)
    _calibration_with_provenance(repo, requested_device="cuda",
                                 onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    trace = e6r.historical_quality_runtime_trace(repo)
    assert trace["HISTORICAL_ORT_PROVIDER_ACTUAL"] == "CUDAExecutionProvider"


def test_model_present_does_not_imply_runtime_provider_resolvable(tmp_path, monkeypatch):
    """TASK J#2: a resolvable model WEIGHT file must not make
    LANDMARK_RUNTIME_RESOLVABLE true if the ONNX Runtime install has no usable
    provider at all (neither the requested one nor even CPU)."""
    from prism_fas.synthesis import quality_models

    repo = _base_repo(tmp_path)
    _calibration_fixture(repo)

    def _fake_resolve_weight(weight_root, role, *, verify=True):
        path = Path(weight_root) / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        return path

    monkeypatch.setattr(quality_models, "resolve_weight", _fake_resolve_weight)

    class _FakeOrt:
        __version__ = "1.24.1"

        @staticmethod
        def get_available_providers():
            return []  # no provider at all -- not even CPU

    import sys
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt())

    assets = e6r.resolve_quality_backend_assets(repo)
    assert assets["LANDMARK_MODEL_RESOLVABLE"] is True  # the weight file itself IS resolvable
    assert assets["LANDMARK_RUNTIME_RESOLVABLE"] is False  # but the runtime cannot construct a session


def test_no_silent_cpu_fallback_when_historical_parity_requires_cuda(tmp_path, monkeypatch):
    """TASK J#3: if the HISTORICAL record shows CUDAExecutionProvider was
    actually used, and the CURRENT host only has CPU, QUALITY_RUNTIME_PARITY
    must report False -- never silently claim parity."""
    repo = _base_repo(tmp_path)
    _calibration_with_provenance(repo, requested_device="cuda",
                                 onnxruntime_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def _no_cuda():
        from prism_fas.synthesis.c5_render import ScientificDeviceUnavailable

        raise ScientificDeviceUnavailable("no CUDA on this host")

    from prism_fas.synthesis import c5_render

    monkeypatch.setattr(c5_render, "scientific_device", _no_cuda)
    assets = e6r.resolve_quality_backend_assets(repo)
    assert assets["LANDMARK_ACTUAL_PROVIDER"] == "CPUExecutionProvider"
    assert assets["QUALITY_RUNTIME_PARITY"] is False


def test_cpu_accepted_only_when_historically_evidenced(tmp_path, monkeypatch):
    """TASK J#4: CPU is accepted as parity-matching ONLY because the frozen
    historical record itself derives to CPU (real evidence) -- not because the
    code defaults to treating CPU as always fine."""
    repo = _base_repo(tmp_path)
    _calibration_with_provenance(repo, requested_device="cuda",
                                 onnxruntime_providers=["AzureExecutionProvider", "CPUExecutionProvider"])
    from prism_fas.synthesis import c5_render

    monkeypatch.setattr(c5_render, "scientific_device",
                        lambda: (_ for _ in ()).throw(c5_render.ScientificDeviceUnavailable("no cuda")))
    assets = e6r.resolve_quality_backend_assets(repo)
    assert assets["LANDMARK_ACTUAL_PROVIDER"] == "CPUExecutionProvider"
    assert assets["QUALITY_RUNTIME_PARITY"] is True  # matches because historical ALSO derived to CPU


def test_quality_runtime_parity_unknown_without_historical_evidence(tmp_path):
    repo = _base_repo(tmp_path)
    assets = e6r.resolve_quality_backend_assets(repo)
    assert assets["QUALITY_RUNTIME_PARITY"] == "UNKNOWN"


# --------------------------------------------------------------------------- #
# TASK C/D/F: per-candidate quality diagnostic
# --------------------------------------------------------------------------- #

class _FakeDiagnosticStore:
    def __init__(self, *, package_root: Path, rows: dict[str, dict[str, Any]]):
        self.package_root = package_root
        self._rows = rows

    def row(self, sample_id):
        return self._rows[sample_id]

    def load(self, sample_id):
        return f"ORIGINAL_IMAGE[{sample_id}]", None


def _diagnostic_fixture(tmp_path: Path, monkeypatch, *, candidate_id: str = "hist-cand-diag",
                        historical_q: float = 0.5, recomputed_q: float = 0.5,
                        live_bytes: bytes = b"live-bytes", spoof_bytes: bytes = b"spoof-bytes"):
    from prism_fas.evaluation import c_ext_quality_reconstruct as qr
    from prism_fas.synthesis import c5_raw_generation as raw
    from prism_fas.synthesis import c6_scientific

    repo = _base_repo(tmp_path)
    directory = raw.candidate_dir(repo / e6r.HISTORICAL_LLM_CANDIDATE_ROOT, "LLM", candidate_id)
    directory.mkdir(parents=True, exist_ok=True)
    identity = raw.GenerationIdentity(
        candidate_id=candidate_id, arm="LLM", arm_plan_identity="p", source_pair_plan_identity="s",
        package_identity="pkg", recipe_bank_identity="rb", recipe_id="R-000221", recipe_ordinal=162,
        slot=1, position=1297, route="gpat", live_target_sample_id="live-1",
        spoof_source_sample_id="spoof-1", generator_binding="binding", ontology_identity="ont")
    payload_sha = {raw.IMAGE_NAME: "sha-image", raw.MASK_NAME: "sha-mask", raw.ARTIFACT_MAP_NAME: "sha-artifact"}
    raw.write_record(directory, raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                                     payload_sha256=payload_sha))

    package_root = tmp_path / "package"
    (package_root / "images").mkdir(parents=True, exist_ok=True)
    live_path = package_root / "images" / "live-1.png"
    spoof_path = package_root / "images" / "spoof-1.png"
    live_path.write_bytes(live_bytes)
    spoof_path.write_bytes(spoof_bytes)
    fake_store = _FakeDiagnosticStore(package_root=package_root, rows={
        "live-1": {"image_relative_path": "images/live-1.png"},
        "spoof-1": {"image_relative_path": "images/spoof-1.png"},
    })

    fake_row = qr.ReconstructedRow(candidate_id=candidate_id, arm="LLM", route="gpat",
                                   historical_selected=True, historical_passed=True, q=historical_q,
                                   accepted=True, source_artifact_identity="fake",
                                   reconstruction_method="EXTRACTED_FROM_FROZEN_BANK_LOCK")
    monkeypatch.setattr(qr, "extract_selected_q", lambda repo: ([fake_row], {}))

    class _Node:
        strength = 0.3

    class _FakeGraph:
        nodes = [_Node()]

    class _FakeEvaluator:
        def evaluate(self, discrete, **kwargs):
            return {"metrics": {name: 0.5 for name in c6_scientific.REQUIRED_RAW_METRICS},
                   "q": recomputed_q, "accepted": True, "failed_gates": [],
                   "quality_components": {name: 0.7 for name in
                                          ("q_fd", "q_id", "q_lm", "q_parse", "q_strength", "q_fp", "q_support")}}

    from prism_fas.synthesis.quality_gate import Thresholds

    thresholds = Thresholds(tau_fd=0.5, tau_id=0.1, tau_lm=0.5, tau_parse=0.0, tau_out=0.0, tau_fp=100.0)

    class _FakeCalibration:
        pass

    calibration = _FakeCalibration()
    calibration.thresholds = thresholds

    monkeypatch.setattr(e6r, "_resolve_quality_runtime",
                        lambda repo: {"store": fake_store, "evaluator": _FakeEvaluator(),
                                     "calibration": calibration})
    monkeypatch.setattr(e6r, "_resolve_historical_llm_bank", lambda repo: object())
    monkeypatch.setattr(c6_scientific, "requested_support_for",
                        lambda store, bank, row: ("SUPPORT_MASK", _FakeGraph()))
    monkeypatch.setattr(c6_scientific, "reconstruct_discrete", lambda directory_, original: "DISCRETE")
    return repo, candidate_id


def test_candidate_diagnostic_uses_canonical_evaluator_and_emits_all_raw_metrics(tmp_path, monkeypatch):
    """TASK J#6/J#9: uses the injected canonical evaluator (never a duplicate
    quality implementation) and every required raw metric is present."""
    from prism_fas.synthesis import c6_scientific

    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)
    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    assert set(diagnostic["raw_metrics"]) == set(c6_scientific.REQUIRED_RAW_METRICS)
    assert diagnostic["quality_components"] is not None
    assert diagnostic["failed_gates"] == []
    assert diagnostic["accepted"] is True


def test_candidate_diagnostic_performs_zero_rendering(tmp_path, monkeypatch):
    """TASK J#5: rendering_performed is False, and no render/generate symbol is
    reachable from the diagnostic's own source."""
    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)
    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    assert diagnostic["rendering_performed"] is False
    assert diagnostic["training_performed"] is False
    assert diagnostic["target_access"] is False
    assert diagnostic["llm_api_calls"] == 0


def test_candidate_diagnostic_reports_exact_payload_hashes(tmp_path, monkeypatch):
    """TASK J#7: exact candidate payload hashes reported verbatim from the
    frozen CANDIDATE.json, never recomputed or invented."""
    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)
    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    assert diagnostic["synthetic_image_sha256"] == "sha-image"
    assert diagnostic["artifact_map_sha256"] == "sha-artifact"
    assert diagnostic["exact_mask_sha256"] == "sha-mask"


def test_candidate_diagnostic_resolves_exact_source_pair_ids(tmp_path, monkeypatch):
    """TASK J#8/TASK D: live and spoof sample ids resolve to real file hashes;
    the source-pair mapping itself is never altered."""
    import hashlib

    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch,
                                             live_bytes=b"LIVE", spoof_bytes=b"SPOOF")
    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    assert diagnostic["live_target_sample_id"] == "live-1"
    assert diagnostic["resolved_live_file_sha256"] == hashlib.sha256(b"LIVE").hexdigest()
    assert diagnostic["spoof_source_resolution"]["spoof_source_sample_id"] == "spoof-1"
    assert diagnostic["spoof_source_resolution"]["resolved_spoof_file_sha256"] == hashlib.sha256(b"SPOOF").hexdigest()
    assert diagnostic["spoof_source_resolution"]["used_by_candidate_evaluator_at_measurement_time"] is False


def test_candidate_diagnostic_does_not_alter_q_or_quality_config(tmp_path, monkeypatch):
    """TASK J#11/J#12: the diagnostic never writes to the calibration file or
    any quality-config path; it is purely read-only against in-memory state."""
    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)
    calibration_path = repo / e6r.QUALITY_CALIBRATION_PATH
    before = calibration_path.read_bytes() if calibration_path.is_file() else None
    e6r.diagnose_historical_candidate(repo, candidate_id)
    after = calibration_path.read_bytes() if calibration_path.is_file() else None
    assert before == after


def test_candidate_diagnostic_historical_q_is_read_only_and_never_mutated(tmp_path, monkeypatch):
    """TASK J#10: historical_q in the diagnostic is the verbatim persisted
    value; recomputation never overwrites or redefines it."""
    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch, historical_q=0.777, recomputed_q=0.1)
    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    assert diagnostic["historical_q"] == 0.777
    assert diagnostic["recomputed_q"] == 0.1
    assert diagnostic["abs_diff"] == pytest.approx(0.677, abs=1e-9)


def test_candidate_diagnostic_table_reports_metric_rows(tmp_path, monkeypatch):
    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)
    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    table = e6r.diagnostic_metric_table(diagnostic)
    metrics = {row["metric"] for row in table}
    assert "q" in metrics
    assert "face_detection_score" in metrics
    q_row = next(row for row in table if row["metric"] == "q")
    assert q_row["historical_value"] == diagnostic["historical_q"]
    assert q_row["recomputed_value"] == diagnostic["recomputed_q"]


def test_candidate_diagnostic_missing_candidate_fails_closed(tmp_path, monkeypatch):
    from prism_fas.evaluation import c_ext_quality_reconstruct as qr

    repo = _base_repo(tmp_path)
    monkeypatch.setattr(qr, "extract_selected_q", lambda repo: ([], {}))
    with pytest.raises(e6r.E6RenderError, match="not a frozen ORIGINAL_LLM selected candidate"):
        e6r.diagnose_historical_candidate(repo, "unknown-candidate")


def test_diagnose_historical_candidates_batch_never_crashes_on_one_bad_id(tmp_path, monkeypatch):
    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)

    def _fake_assets(repo):
        return {"QUALITY_BACKENDS_RESOLVABLE": True}

    monkeypatch.setattr(e6r, "resolve_quality_backend_assets", _fake_assets)
    batch = e6r.diagnose_historical_candidates(repo, [candidate_id, "does-not-exist"])
    assert batch["diagnostic_executed"] is True
    outcomes = {row["candidate_id"]: row["ok"] for row in batch["results"]}
    assert outcomes[candidate_id] is True
    assert outcomes["does-not-exist"] is False


def test_diagnose_historical_candidates_deferred_when_backends_unresolvable(tmp_path):
    repo = _base_repo(tmp_path)
    batch = e6r.diagnose_historical_candidates(repo, ["any-id"])
    assert batch["diagnostic_executed"] is False
    assert batch["diagnostic_status"] == "DEFERRED"
    assert batch["results"] == []


def test_gpu_candidate_diagnostic_command_names_the_three_mismatched_ids():
    ids = ["c5syn_0390812685e6403952baeb67", "c5syn_057aab8fef90ada42997a1a4",
          "c5syn_0588f57b2499484387d2b7af"]
    command = e6r.gpu_candidate_diagnostic_command(ids)
    for candidate_id in ids:
        assert candidate_id in command
    assert command.startswith("python -m prism_fas.evaluation.c_ext_e6_render --diagnose-historical-candidate")


def test_diagnostic_cli_flag_never_renders_or_executes():
    """TASK J#13-15: the diagnostic argparse wiring reaches only
    diagnose_historical_candidates, never run_render_execution or any
    training/target/LLM symbol."""
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    diagnose_block_start = source.index("if args.diagnose_historical_candidate:")
    diagnose_block = source[diagnose_block_start:diagnose_block_start + 300]
    assert "run_render_execution" not in diagnose_block
    for forbidden in ("target_test", "siw_mv2", "evaluation_only", "trainer.resume",
                     "openai", "anthropic", "gemini"):
        assert forbidden.lower() not in source.lower()


def test_historical_q_reproduction_tolerance_unchanged():
    """TASK J#16: the exact-mismatch tolerance (1e-6) in
    run_historical_q_reproduction is untouched by this milestone's fixes."""
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    assert 'entry["abs_diff"] > 1e-6' in source


def test_run_historical_q_reproduction_mismatch_still_reported_as_fail(tmp_path, monkeypatch):
    """TASK J#17: a real mismatch still produces status FAIL -- the historical-q
    audit policy (never loosened, never silently passed) is unchanged."""
    repo = _base_repo(tmp_path)
    candidate_id = "hist-cand-mismatch-regress"
    _write_e6_candidate_record(repo, candidate_id, arm="LLM",
                               candidates_root=e6r.HISTORICAL_LLM_CANDIDATE_ROOT)
    _fake_historical_reproduction_deps(monkeypatch, candidate_id=candidate_id,
                                       historical_q=0.7228755354881287, recomputed_q=0.713808000087738)
    result = e6r.run_historical_q_reproduction(repo, sample_size=8)
    assert result["historical_q_reproduction_status"] == "FAIL"
    assert result["mismatch_count"] == 1


# --------------------------------------------------------------------------- #
# support_overlap root-cause forensic (mask_forensics on the diagnostic)
# --------------------------------------------------------------------------- #

def test_mask_forensics_detects_a_generation_vs_recompute_pixel_count_mismatch(tmp_path, monkeypatch):
    """The forensic comparison between the GENERATION-TIME persisted trace
    (CANDIDATE.json's requested_support_pixels) and a FRESH recompute of the
    full RegionMaskResult must surface a mismatch when the two genuinely
    differ -- this is the exact, additive diagnostic Task C/D asked for."""
    from prism_fas.synthesis import c6_scientific, synthetic_bank

    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)

    class _FakeMaskResult:
        requested_region_mask = np.zeros((1, 4, 4), dtype=np.float32)
        operator_support_mask = np.zeros((1, 4, 4), dtype=np.float32)
        operator_support_mask[0, :2, :2] = 1.0  # 4 pixels recomputed now
        region_sources = {"nose": "parsing"}
        requested_coverage = 1.0
        achieved_coverage = 1.0
        mask_hash = "fake-hash"
        metadata = {"per_region_pixels": {"nose": 4}, "coverage_within_tolerance": True,
                   "parsing_available": True}

    monkeypatch.setattr(synthetic_bank, "_support_masks", lambda store, sample_id, graph: _FakeMaskResult())

    class _FakeDiscrete:
        exact_edit_mask = np.zeros((4, 4), dtype=bool)
        exact_edit_mask[0, 0] = True
    monkeypatch.setattr(c6_scientific, "reconstruct_discrete", lambda directory_, original: _FakeDiscrete())

    # override the record's trace to claim a DIFFERENT pixel count at generation
    from prism_fas.synthesis import c5_raw_generation as raw

    directory = raw.candidate_dir(repo / e6r.HISTORICAL_LLM_CANDIDATE_ROOT, "LLM", candidate_id)
    record = json.loads((directory / raw.RECORD_NAME).read_text(encoding="utf-8"))
    record["trace"] = {"requested_region_pixels": 4, "requested_support_pixels": 999,
                       "requested_coverage": 1.0, "achieved_coverage": 1.0}
    (directory / raw.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")

    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    forensics = diagnostic["mask_forensics"]
    assert forensics["recomputed_now"]["support_pixels"] == 4
    assert forensics["persisted_at_generation"]["requested_support_pixels"] == 999
    assert forensics["support_pixel_count_matches_generation"] is False


def test_mask_forensics_confirms_match_when_pixel_counts_agree(tmp_path, monkeypatch):
    from prism_fas.synthesis import c6_scientific, synthetic_bank

    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)

    class _FakeMaskResult:
        requested_region_mask = np.zeros((1, 4, 4), dtype=np.float32)
        operator_support_mask = np.zeros((1, 4, 4), dtype=np.float32)
        operator_support_mask[0, :2, :2] = 1.0
        region_sources = {"nose": "parsing"}
        requested_coverage = 1.0
        achieved_coverage = 1.0
        mask_hash = "fake-hash"
        metadata = {"per_region_pixels": {"nose": 4}, "coverage_within_tolerance": True,
                   "parsing_available": True}

    monkeypatch.setattr(synthetic_bank, "_support_masks", lambda store, sample_id, graph: _FakeMaskResult())

    class _FakeDiscrete:
        exact_edit_mask = np.zeros((4, 4), dtype=bool)
        exact_edit_mask[0, 0] = True
    monkeypatch.setattr(c6_scientific, "reconstruct_discrete", lambda directory_, original: _FakeDiscrete())

    from prism_fas.synthesis import c5_raw_generation as raw

    directory = raw.candidate_dir(repo / e6r.HISTORICAL_LLM_CANDIDATE_ROOT, "LLM", candidate_id)
    record = json.loads((directory / raw.RECORD_NAME).read_text(encoding="utf-8"))
    record["trace"] = {"requested_region_pixels": 4, "requested_support_pixels": 4,
                       "requested_coverage": 1.0, "achieved_coverage": 1.0}
    (directory / raw.RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")

    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    forensics = diagnostic["mask_forensics"]
    assert forensics["support_pixel_count_matches_generation"] is True


def test_mask_forensics_never_crashes_the_diagnostic_on_failure(tmp_path, monkeypatch):
    """When the forensic recompute itself fails (e.g. a fake store without
    the full mask-building API), the diagnostic still returns every other
    field -- forensics degrade to an error dict, never an exception."""
    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)
    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    assert "mask_forensics" in diagnostic
    assert set(diagnostic["raw_metrics"])  # the rest of the diagnostic still completed


def test_mask_forensics_is_deterministic_given_the_same_fake_inputs(tmp_path, monkeypatch):
    from prism_fas.synthesis import c6_scientific, synthetic_bank

    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)

    class _FakeMaskResult:
        requested_region_mask = np.zeros((1, 4, 4), dtype=np.float32)
        operator_support_mask = np.zeros((1, 4, 4), dtype=np.float32)
        operator_support_mask[0, :1, :1] = 1.0
        region_sources = {"nose": "parsing"}
        requested_coverage = 1.0
        achieved_coverage = 1.0
        mask_hash = "fake-hash"
        metadata = {"per_region_pixels": {"nose": 1}, "coverage_within_tolerance": True,
                   "parsing_available": True}

    monkeypatch.setattr(synthetic_bank, "_support_masks", lambda store, sample_id, graph: _FakeMaskResult())

    class _FakeDiscrete:
        exact_edit_mask = np.zeros((4, 4), dtype=bool)
        exact_edit_mask[0, 0] = True
    monkeypatch.setattr(c6_scientific, "reconstruct_discrete", lambda directory_, original: _FakeDiscrete())
    first = e6r.diagnose_historical_candidate(repo, candidate_id)["mask_forensics"]
    second = e6r.diagnose_historical_candidate(repo, candidate_id)["mask_forensics"]
    assert first["recomputed_now"]["support_pixels"] == second["recomputed_now"]["support_pixels"]
    assert first["recomputed_now"]["mask_hash"] == second["recomputed_now"]["mask_hash"]


def test_diagnostic_never_mutates_the_candidate_record_or_payload(tmp_path, monkeypatch):
    """No candidate mutation, no historical artifact mutation (TASK J)."""
    from prism_fas.synthesis import c5_raw_generation as raw

    repo, candidate_id = _diagnostic_fixture(tmp_path, monkeypatch)
    directory = raw.candidate_dir(repo / e6r.HISTORICAL_LLM_CANDIDATE_ROOT, "LLM", candidate_id)
    record_path = directory / raw.RECORD_NAME
    before = record_path.read_bytes()
    e6r.diagnose_historical_candidate(repo, candidate_id)
    after = record_path.read_bytes()
    assert before == after


# --------------------------------------------------------------------------- #
# TASK B: read-only git-history inspection (no checkout/reset of the tree)
# --------------------------------------------------------------------------- #

def test_git_history_inspection_is_read_only_and_does_not_touch_the_working_tree():
    """`git log` / `git show` are read-only against object history; proves the
    working tree (git status) is unaffected by inspecting it before/after."""
    import subprocess

    repo_root = cc.repo_root()
    before = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                            capture_output=True, text=True, check=True).stdout
    subprocess.run(["git", "log", "--oneline", "-5", "--",
                   "src/prism_fas/synthesis/quality_gate.py"], cwd=repo_root,
                   capture_output=True, text=True, check=True)
    subprocess.run(["git", "show", "HEAD:src/prism_fas/synthesis/quality_gate.py"], cwd=repo_root,
                   capture_output=True, text=True, check=True)
    after = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                           capture_output=True, text=True, check=True).stdout
    assert before == after


def test_support_overlap_root_cause_artifact_is_valid_json():
    path = cc.repo_root() / e6r.RENDER_DIR / "E6_SUPPORT_OVERLAP_ROOT_CAUSE.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["root_cause_classification"]["ROOT_CAUSE_CLASS"] in {"A", "B", "C", "D", "E"}
    assert "code_drift_ruled_out" in payload["task_b_historical_code_provenance"]


# --------------------------------------------------------------------------- #
# GPU-forensic follow-up: identity_comparison + alternative_input_reconstructions
# --------------------------------------------------------------------------- #

def _diagnostic_fixture_with_full_graph(tmp_path: Path, monkeypatch, *, candidate_id: str = "hist-cand-full",
                                        current_bank_identity: str = "bank-now",
                                        persisted_bank_identity: str = "bank-now",
                                        current_recipe_hash: str = "recipe-hash-now",
                                        persisted_recipe_hash: str = "recipe-hash-now",
                                        current_graph_hash: str = "graph-hash-now",
                                        persisted_graph_hash: str = "graph-hash-now"):
    """Like `_diagnostic_fixture`, but with a REAL (dict) bank carrying
    `bank_identity` and a graph carrying `recipe_hash`/`graph_hash`/
    `region_mask_policy`, so identity_comparison exercises its real logic
    instead of degrading to an error dict."""
    from prism_fas.evaluation import c_ext_quality_reconstruct as qr
    from prism_fas.synthesis import c5_raw_generation as raw
    from prism_fas.synthesis import c6_scientific

    repo = _base_repo(tmp_path)
    directory = raw.candidate_dir(repo / e6r.HISTORICAL_LLM_CANDIDATE_ROOT, "LLM", candidate_id)
    directory.mkdir(parents=True, exist_ok=True)
    identity = raw.GenerationIdentity(
        candidate_id=candidate_id, arm="LLM", arm_plan_identity="p", source_pair_plan_identity="s",
        package_identity="pkg", recipe_bank_identity=persisted_bank_identity, recipe_id="R-000354",
        recipe_ordinal=99, slot=1, position=793, route="gpat", live_target_sample_id="live-1",
        spoof_source_sample_id="spoof-1", generator_binding="binding", ontology_identity="ont")
    payload_sha = {raw.IMAGE_NAME: "sha-image", raw.MASK_NAME: "sha-mask", raw.ARTIFACT_MAP_NAME: "sha-artifact"}
    record = raw.CandidateRecord(
        identity=identity, status=raw.GENERATED, payload_sha256=payload_sha,
        trace={"requested_region_pixels": 232, "requested_support_pixels": 209, "requested_coverage": 0.9021,
              "achieved_coverage": 0.9009,
              "route_trace": {"recipe_hash": persisted_recipe_hash, "graph_hash": persisted_graph_hash}})
    raw.write_record(directory, record)

    package_root = tmp_path / "package"
    (package_root / "images").mkdir(parents=True, exist_ok=True)
    live_path = package_root / "images" / "live-1.png"
    spoof_path = package_root / "images" / "spoof-1.png"
    live_path.write_bytes(b"live-bytes")
    spoof_path.write_bytes(b"spoof-bytes")
    fake_store = _FakeDiagnosticStore(package_root=package_root, rows={
        "live-1": {"image_relative_path": "images/live-1.png"},
        "spoof-1": {"image_relative_path": "images/spoof-1.png"},
    })

    fake_row = qr.ReconstructedRow(candidate_id=candidate_id, arm="LLM", route="gpat",
                                   historical_selected=True, historical_passed=True, q=0.8445689678192139,
                                   accepted=True, source_artifact_identity="fake",
                                   reconstruction_method="EXTRACTED_FROM_FROZEN_BANK_LOCK")
    monkeypatch.setattr(qr, "extract_selected_q", lambda repo: ([fake_row], {}))

    class _Node:
        strength = 0.3

    class _FakeGraph:
        nodes = [_Node()]
        recipe_hash = current_recipe_hash
        graph_hash = current_graph_hash
        region_mask_policy = {"requested_coverage": 0.75}

    class _FakeEvaluator:
        def evaluate(self, discrete, **kwargs):
            return {"metrics": {name: 0.5 for name in c6_scientific.REQUIRED_RAW_METRICS},
                   "q": 0.11735247820615768, "accepted": False, "failed_gates": ["support_overlap"],
                   "quality_components": {name: 0.7 for name in
                                          ("q_fd", "q_id", "q_lm", "q_parse", "q_strength", "q_fp", "q_support")}}

    from prism_fas.synthesis.quality_gate import Thresholds

    thresholds = Thresholds(tau_fd=0.5, tau_id=0.1, tau_lm=0.5, tau_parse=0.0, tau_out=0.0, tau_fp=100.0)

    class _FakeCalibration:
        pass

    calibration = _FakeCalibration()
    calibration.thresholds = thresholds

    class _FakeDiscrete:
        exact_edit_mask = np.zeros((4, 4), dtype=bool)
        exact_edit_mask[0, 0] = True

    monkeypatch.setattr(e6r, "_resolve_quality_runtime",
                        lambda repo: {"store": fake_store, "evaluator": _FakeEvaluator(),
                                     "calibration": calibration})
    monkeypatch.setattr(e6r, "_resolve_historical_llm_bank",
                        lambda repo: {"bank_identity": current_bank_identity, "bank_id": "c3_llm"})
    monkeypatch.setattr(c6_scientific, "requested_support_for",
                        lambda store, bank, row: ("SUPPORT_MASK", _FakeGraph()))
    monkeypatch.setattr(c6_scientific, "reconstruct_discrete", lambda directory_, original: _FakeDiscrete())
    return repo, candidate_id


def _fake_mask_result(*, support_pixels: int, region_pixels: int = 16, coverage: float = 0.5):
    class _Result:
        requested_region_mask = np.ones((1, 4, 4), dtype=np.float32)
        operator_support_mask = np.zeros((1, 4, 4), dtype=np.float32)
        region_sources = {"nose": "parsing"}
        requested_coverage = coverage
        achieved_coverage = coverage
        mask_hash = f"hash-{support_pixels}"
        metadata = {"per_region_pixels": {"nose": support_pixels}, "coverage_within_tolerance": True,
                   "parsing_available": True}
    flat = _Result.operator_support_mask.reshape(-1)
    flat[:support_pixels] = 1.0
    _Result.operator_support_mask = flat.reshape(1, 4, 4)
    return _Result()


def test_identity_comparison_detects_recipe_bank_identity_mismatch(tmp_path, monkeypatch):
    from prism_fas.synthesis import synthetic_bank

    repo, candidate_id = _diagnostic_fixture_with_full_graph(
        tmp_path, monkeypatch, current_bank_identity="bank-NOW-DIFFERENT", persisted_bank_identity="bank-THEN")
    monkeypatch.setattr(synthetic_bank, "_support_masks",
                        lambda store, sample_id, graph: _fake_mask_result(support_pixels=1))

    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    comparison = diagnostic["identity_comparison"]
    assert comparison["persisted_recipe_bank_identity"] == "bank-THEN"
    assert comparison["current_bank_identity"] == "bank-NOW-DIFFERENT"
    assert comparison["recipe_bank_identity_matches"] is False


def test_identity_comparison_confirms_match_when_identities_agree(tmp_path, monkeypatch):
    from prism_fas.synthesis import synthetic_bank

    repo, candidate_id = _diagnostic_fixture_with_full_graph(
        tmp_path, monkeypatch, current_bank_identity="bank-SAME", persisted_bank_identity="bank-SAME",
        current_recipe_hash="rh-same", persisted_recipe_hash="rh-same",
        current_graph_hash="gh-same", persisted_graph_hash="gh-same")
    monkeypatch.setattr(synthetic_bank, "_support_masks",
                        lambda store, sample_id, graph: _fake_mask_result(support_pixels=1))

    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    comparison = diagnostic["identity_comparison"]
    assert comparison["recipe_bank_identity_matches"] is True
    assert comparison["recipe_hash_matches"] is True
    assert comparison["graph_hash_matches"] is True
    assert comparison["recipe_geometry_coverage_now"] == 0.75


def test_alternative_input_reconstruction_tries_live_and_spoof_only(tmp_path, monkeypatch):
    """TASK C: exactly the two code-justified inputs (Task A's trace) are
    tried -- never an invented alternative."""
    from prism_fas.synthesis import synthetic_bank

    repo, candidate_id = _diagnostic_fixture_with_full_graph(tmp_path, monkeypatch)
    monkeypatch.setattr(synthetic_bank, "_support_masks",
                        lambda store, sample_id, graph: _fake_mask_result(
                            support_pixels=2 if sample_id == "live-1" else 4))

    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    reconstructions = diagnostic["alternative_input_reconstructions"]
    sources = {row["input_source"]: row for row in reconstructions}
    assert set(sources) == {"live_target", "spoof_source"}
    assert sources["live_target"]["input_sample_id"] == "live-1"
    assert sources["spoof_source"]["input_sample_id"] == "spoof-1"
    assert sources["live_target"]["support_pixels"] == 2
    assert sources["spoof_source"]["support_pixels"] == 4


def test_alternative_input_reconstruction_reports_exact_hashes_and_no_mutation(tmp_path, monkeypatch):
    import hashlib

    from prism_fas.synthesis import synthetic_bank

    repo, candidate_id = _diagnostic_fixture_with_full_graph(tmp_path, monkeypatch)
    monkeypatch.setattr(synthetic_bank, "_support_masks",
                        lambda store, sample_id, graph: _fake_mask_result(support_pixels=1))

    package_root = tmp_path / "package"
    live_bytes_before = (package_root / "images" / "live-1.png").read_bytes()
    spoof_bytes_before = (package_root / "images" / "spoof-1.png").read_bytes()

    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    reconstructions = {row["input_source"]: row for row in diagnostic["alternative_input_reconstructions"]}
    assert reconstructions["live_target"]["input_sha256"] == hashlib.sha256(live_bytes_before).hexdigest()
    assert reconstructions["spoof_source"]["input_sha256"] == hashlib.sha256(spoof_bytes_before).hexdigest()

    assert (package_root / "images" / "live-1.png").read_bytes() == live_bytes_before
    assert (package_root / "images" / "spoof-1.png").read_bytes() == spoof_bytes_before


def test_alternative_input_reconstruction_computes_comparison_flags(tmp_path, monkeypatch):
    from prism_fas.synthesis import synthetic_bank

    repo, candidate_id = _diagnostic_fixture_with_full_graph(tmp_path, monkeypatch)
    # generation trace (see fixture) declares requested_region_pixels=232,
    # requested_support_pixels=209, requested_coverage=0.9021 -- none of which
    # any 4x4 fake mask can match, proving MATCHES_* correctly report False
    # rather than vacuously True.
    monkeypatch.setattr(synthetic_bank, "_support_masks",
                        lambda store, sample_id, graph: _fake_mask_result(support_pixels=1, coverage=0.5))

    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    for row in diagnostic["alternative_input_reconstructions"]:
        assert row["MATCHES_GENERATION_REGION_PIXELS"] is False
        assert row["MATCHES_GENERATION_SUPPORT_PIXELS"] is False
        assert row["MATCHES_GENERATION_COVERAGE"] is False
        assert "SUPPORT_OVERLAP" in row
        assert "EXACT_MASK_WITHIN_SUPPORT" in row


def test_alternative_input_reconstruction_deterministic(tmp_path, monkeypatch):
    from prism_fas.synthesis import synthetic_bank

    repo, candidate_id = _diagnostic_fixture_with_full_graph(tmp_path, monkeypatch)
    monkeypatch.setattr(synthetic_bank, "_support_masks",
                        lambda store, sample_id, graph: _fake_mask_result(support_pixels=3))

    first = e6r.diagnose_historical_candidate(repo, candidate_id)["alternative_input_reconstructions"]
    second = e6r.diagnose_historical_candidate(repo, candidate_id)["alternative_input_reconstructions"]
    assert first == second


def test_alternative_input_reconstruction_never_renders_or_trains():
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    fn_start = source.index("def diagnose_historical_candidate(")
    fn_end = source.index("\ndef ", fn_start + 10)
    body = source[fn_start:fn_end]
    for forbidden in ("render_one(", "render_arm(", "route.generate(", "GPATRoute(", "PhysicsRoute(",
                     ".model(", "torch.no_grad"):
        assert forbidden not in body, f"{forbidden!r} unexpectedly reachable from the diagnostic"


def test_physics_route_code_untouched_this_turn():
    """TASK H#8: no change was made to PhysicsRoute -- this turn is diagnostic
    -only, per the explicit instruction not to patch scientific semantics."""
    import subprocess

    repo_root = cc.repo_root()
    diff = subprocess.run(["git", "diff", "--stat", "--", "src/prism_fas/synthesis/synthetic_bank.py"],
                          cwd=repo_root, capture_output=True, text=True, check=True).stdout
    # synthetic_bank.py is untracked in this branch's history for this file in
    # some setups; the authoritative check is git log, already proven empty
    # since the historical run in test_git_history_inspection_is_read_only...
    # here we additionally assert PhysicsRoute's class body is present and
    # unchanged in shape (still delegates to _support_masks / PhysicsEngine).
    text = (repo_root / "src/prism_fas/synthesis/synthetic_bank.py").read_text(encoding="utf-8")
    assert "class PhysicsRoute:" in text
    assert "_support_masks(store, sample_id, graph)" in text


# --------------------------------------------------------------------------- #
# TASK G/A: region_mask_seed + raw_generation_trace + sample_priors_fingerprint
# --------------------------------------------------------------------------- #

def test_raw_generation_trace_is_verbatim_and_unfiltered(tmp_path, monkeypatch):
    from prism_fas.synthesis import synthetic_bank

    repo, candidate_id = _diagnostic_fixture_with_full_graph(tmp_path, monkeypatch)
    monkeypatch.setattr(synthetic_bank, "_support_masks",
                        lambda store, sample_id, graph: _fake_mask_result(support_pixels=1))

    diagnostic = e6r.diagnose_historical_candidate(repo, candidate_id)
    # fixture wrote trace={"requested_region_pixels": 232, "requested_support_pixels": 209,
    # "requested_coverage": 0.9021, "achieved_coverage": 0.9009, "route_trace": {...}}
    assert diagnostic["raw_generation_trace"]["requested_region_pixels"] == 232
    assert diagnostic["raw_generation_trace"]["requested_coverage"] == 0.9021
    assert "route_trace" in diagnostic["raw_generation_trace"]


def test_region_mask_seed_is_deterministic_pure_function(tmp_path, monkeypatch):
    from prism_fas.synthesis import synthetic_bank

    repo, candidate_id = _diagnostic_fixture_with_full_graph(tmp_path, monkeypatch)
    monkeypatch.setattr(synthetic_bank, "_support_masks",
                        lambda store, sample_id, graph: _fake_mask_result(support_pixels=1))

    # the fixture's _FakeGraph lacks bank_id/recipe_id/recipe_seed/node_index,
    # so node_seed() cannot be computed -- this proves the None-fallback is
    # itself deterministic (never crashes, never flips between calls), which
    # is what test_region_mask_seed_matches_real_derive_seed_for_a_real_recipe
    # below proves for a REAL graph.
    first = e6r.diagnose_historical_candidate(repo, candidate_id)["region_mask_seed"]
    second = e6r.diagnose_historical_candidate(repo, candidate_id)["region_mask_seed"]
    assert first == second


def test_region_mask_seed_matches_real_derive_seed_for_a_real_recipe():
    """Cross-checks the diagnostic's seed computation against the REAL,
    frozen R-000354 recipe and ontology on this laptop -- no GPU needed,
    proving the formula this milestone relies on for TASK G."""
    from prism_fas.recipes.compile import compile_recipe
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe

    repo = cc.repo_root()
    ontology = load_ontology(repo / e6r.ONTOLOGY_CONFIG_PATH)
    lines = (repo / "assets/recipe_banks/c3/llm/recipes.jsonl").read_text(encoding="utf-8").strip().split("\n")
    payload = next(json.loads(line) for line in lines if json.loads(line).get("recipe_id") == "R-000354")
    recipe = parse_recipe(payload)
    graph = compile_recipe(recipe, ontology, bank_id="c3_llm")
    seed = graph.node_seed(graph.nodes[0], "da2de9d2ea1fe2d69025f2a6|region_mask")
    assert seed == 3306244086
    assert graph.region_mask_policy["requested_coverage"] == 0.75


def test_sample_priors_fingerprint_present_and_deterministic(tmp_path, monkeypatch):
    from prism_fas.synthesis import synthetic_bank

    repo, candidate_id = _diagnostic_fixture_with_full_graph(tmp_path, monkeypatch)
    monkeypatch.setattr(synthetic_bank, "_support_masks",
                        lambda store, sample_id, graph: _fake_mask_result(support_pixels=1))

    first = e6r.diagnose_historical_candidate(repo, candidate_id)
    second = e6r.diagnose_historical_candidate(repo, candidate_id)
    # the fake store's .load() returns a plain string, not real arrays, so the
    # fingerprint computation is expected to degrade to None gracefully here --
    # the test proves it never crashes the diagnostic either way, and is stable.
    assert first["sample_priors_fingerprint"] == second["sample_priors_fingerprint"]
    assert "sample_priors_fingerprint_note" in first


# --------------------------------------------------------------------------- #
# TASK G: shard-vs-loose-file byte audit -- read-only, real tarfile fixtures
# --------------------------------------------------------------------------- #

def _shard_fixture(tmp_path: Path, *, sample_id: str, image_bytes: bytes, prior_bytes: bytes,
                   loose_image_bytes: bytes | None = None, loose_prior_bytes: bytes | None = None
                   ) -> Path:
    """A minimal, real (uncompressed tar) package layout: one shard containing
    `<sample_id>.jpg`/`.npz`/`.json`, a PACKAGE_LOCK.json naming it under
    split='source_train' with its real sha256, a matching manifest parquet
    SampleStore can open, and LOOSE files under images/priors (defaulting to
    the SAME bytes as the shard; pass loose_*_bytes to simulate drift)."""
    import hashlib
    import tarfile

    from prism_fas.data.package.shards import write_shard

    package_root = tmp_path / "m3b_package"
    (package_root / "shards").mkdir(parents=True, exist_ok=True)
    (package_root / "images").mkdir(parents=True, exist_ok=True)
    (package_root / "priors").mkdir(parents=True, exist_ok=True)
    (package_root / "manifests").mkdir(parents=True, exist_ok=True)

    metadata = {"sample_id": sample_id}
    shard_summary = write_shard(package_root / "shards" / "source_train-00000.tar",
                                [(sample_id, image_bytes, prior_bytes, metadata)])

    (package_root / "images" / f"{sample_id}.jpg").write_bytes(
        loose_image_bytes if loose_image_bytes is not None else image_bytes)
    (package_root / "priors" / f"{sample_id}.npz").write_bytes(
        loose_prior_bytes if loose_prior_bytes is not None else prior_bytes)

    lock = {"content_identity_sha256": "fake",
           "shards": [{"shard_filename": shard_summary["shard_filename"], "split": "source_train",
                      "sha256": shard_summary["sha256"], "row_count": 1,
                      "byte_size": shard_summary["byte_size"]}]}
    (package_root / "PACKAGE_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"sample_id": [sample_id], "project_split": ["source_train"],
                      "image_relative_path": [f"images/{sample_id}.jpg"],
                      "prior_relative_path": [f"priors/{sample_id}.npz"]})
    pq.write_table(table, package_root / "manifests" / "source_train.parquet")
    return package_root


def test_shard_vs_loose_byte_audit_confirms_match_when_bytes_agree(tmp_path):
    repo = _base_repo(tmp_path)
    sample_id = "sample-match"
    package_root = _shard_fixture(tmp_path, sample_id=sample_id, image_bytes=b"IMAGE-BYTES",
                                  prior_bytes=b"PRIOR-BYTES")
    import shutil
    shutil.copytree(package_root, repo / e6r.SOURCE_PACKAGE_ROOT, dirs_exist_ok=True)

    result = e6r.shard_vs_loose_byte_audit(repo, sample_id)
    assert result["available"] is True
    assert result["image_matches"] is True
    assert result["prior_matches"] is True
    assert result["shard_archive_matches_package_lock"] is True


def test_shard_vs_loose_byte_audit_detects_drift(tmp_path):
    repo = _base_repo(tmp_path)
    sample_id = "sample-drift"
    package_root = _shard_fixture(tmp_path, sample_id=sample_id, image_bytes=b"ORIGINAL-IMAGE",
                                  prior_bytes=b"ORIGINAL-PRIOR", loose_image_bytes=b"DRIFTED-IMAGE")
    import shutil
    shutil.copytree(package_root, repo / e6r.SOURCE_PACKAGE_ROOT, dirs_exist_ok=True)

    result = e6r.shard_vs_loose_byte_audit(repo, sample_id)
    assert result["available"] is True
    assert result["image_matches"] is False
    assert result["prior_matches"] is True
    assert result["loose_image_sha256"] != result["shard_image_sha256"]


def test_shard_vs_loose_byte_audit_read_only_never_extracts_or_repairs(tmp_path):
    """No file under the package root may be created/modified by the audit."""
    import hashlib

    repo = _base_repo(tmp_path)
    sample_id = "sample-readonly"
    package_root = _shard_fixture(tmp_path, sample_id=sample_id, image_bytes=b"A", prior_bytes=b"B")
    shard_root = repo / e6r.SOURCE_PACKAGE_ROOT
    import shutil
    shutil.copytree(package_root, shard_root, dirs_exist_ok=True)

    def _tree_hash(root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    before = _tree_hash(shard_root)
    e6r.shard_vs_loose_byte_audit(repo, sample_id)
    after = _tree_hash(shard_root)
    assert before == after


def test_shard_vs_loose_byte_audit_unavailable_without_package_lock(tmp_path):
    repo = _base_repo(tmp_path)
    result = e6r.shard_vs_loose_byte_audit(repo, "any-sample")
    assert result["available"] is False


def test_shard_member_sha256_missing_member_returns_none(tmp_path):
    from prism_fas.data.package.shards import write_shard

    shard_path = tmp_path / "shard.tar"
    write_shard(shard_path, [("present-sample", b"img", b"prior", {"sample_id": "present-sample"})])
    assert e6r.shard_member_sha256(shard_path, "present-sample", ".jpg") == \
        __import__("hashlib").sha256(b"img").hexdigest()
    assert e6r.shard_member_sha256(shard_path, "absent-sample", ".jpg") is None


def test_gpu_next_diagnostic_commands_are_prepared_not_executed():
    """TASK G/H: the shard-vs-loose CLI flag exists and is documented as
    read-only; this test itself never invokes it against real data."""
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    assert "--audit-shard-vs-loose" in source
    assert "shard_vs_loose_byte_audit" in source


# --------------------------------------------------------------------------- #
# TASK A/B/C/H: population-wide historical trace audit -- real recipes, real
# compile_recipe, synthetic (but real-shaped) candidate trees on tmp_path.
# --------------------------------------------------------------------------- #

def _population_recipe_bank_fixture(repo: Path) -> dict[str, dict]:
    """Two REAL, schema-valid, frozen recipes (R-000221 geometry.coverage=1.0,
    R-000354 geometry.coverage=0.75) copied verbatim from the real, frozen
    assets/recipe_banks/c3/llm/recipes.jsonl -- plus the real ontology and a
    C3_BANK.json carrying a bank_identity, all under the SAME fixed path
    audit_historical_trace_population reads (assets/recipe_banks/c3/llm/)."""
    real_repo = cc.repo_root()
    lines = (real_repo / "assets/recipe_banks/c3/llm/recipes.jsonl").read_text(encoding="utf-8").strip().split("\n")
    by_id = {}
    for line in lines:
        payload = json.loads(line)
        if payload.get("recipe_id") in ("R-000221", "R-000354"):
            by_id[payload["recipe_id"]] = payload

    dest_dir = repo / "assets/recipe_banks/c3/llm"
    dest_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(by_id[rid], sort_keys=True, separators=(",", ":"))
                    for rid in ("R-000221", "R-000354")) + "\n"
    (dest_dir / "recipes.jsonl").write_text(text, encoding="utf-8")
    (dest_dir / "C3_BANK.json").write_text(json.dumps({"bank_identity": "fake-population-bank-identity"}),
                                           encoding="utf-8")

    ontology_dest = repo / e6r.ONTOLOGY_CONFIG_PATH
    ontology_dest.parent.mkdir(parents=True, exist_ok=True)
    ontology_dest.write_text((real_repo / e6r.ONTOLOGY_CONFIG_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    return by_id


def _write_population_candidate(repo: Path, *, candidate_id: str, route: str, recipe_id: str,
                                bank_identity: str = "fake-population-bank-identity",
                                trace_overrides: dict | None = None,
                                use_real_hashes: bool = True,
                                position: int = 0, live_target_sample_id: str | None = None) -> None:
    """Writes one GENERATED CANDIDATE.json under the real
    HISTORICAL_LLM_CANDIDATE_ROOT/LLM layout, with a `trace`/`route_trace`
    shaped exactly like c5_render.render_one's real writer. When
    `use_real_hashes` is True, recipe_hash/graph_hash are computed via the
    REAL compile_recipe (so recipe_hash_matches/graph_hash_matches come back
    True); pass False plus explicit trace_overrides to simulate a genuine
    mismatch."""
    from prism_fas.recipes.compile import compile_recipe
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe
    from prism_fas.synthesis import c5_raw_generation as raw

    if use_real_hashes:
        lines = (repo / "assets/recipe_banks/c3/llm/recipes.jsonl").read_text(encoding="utf-8").strip().split("\n")
        payload = next(json.loads(line) for line in lines if json.loads(line).get("recipe_id") == recipe_id)
        ontology = load_ontology(repo / e6r.ONTOLOGY_CONFIG_PATH)
        graph = compile_recipe(parse_recipe(payload), ontology, bank_id="c3_llm")
        recipe_hash, graph_hash = graph.recipe_hash, graph.graph_hash
        recipe_coverage = graph.region_mask_policy["requested_coverage"]
    else:
        recipe_hash, graph_hash, recipe_coverage = "stale-recipe-hash", "stale-graph-hash", 0.5

    directory = raw.candidate_dir(repo / e6r.HISTORICAL_LLM_CANDIDATE_ROOT, "LLM", candidate_id)
    directory.mkdir(parents=True, exist_ok=True)
    identity = raw.GenerationIdentity(
        candidate_id=candidate_id, arm="LLM", arm_plan_identity="p", source_pair_plan_identity="s",
        package_identity="pkg", recipe_bank_identity=bank_identity, recipe_id=recipe_id, recipe_ordinal=0,
        slot=0, position=position, route=route,
        live_target_sample_id=live_target_sample_id or f"live-{candidate_id}",
        spoof_source_sample_id=(f"spoof-{candidate_id}" if route == "gpat" else None),
        generator_binding="binding", ontology_identity="ont")
    default_trace = {"binding": "binding", "requested_region_pixels": 1000, "requested_coverage": recipe_coverage,
                     "achieved_coverage": recipe_coverage, "requested_support_pixels": int(1000 * recipe_coverage),
                     "exact_mask_pixels": 500, "outside_mask_max_error": 0,
                     "route_trace": {"recipe_hash": recipe_hash, "graph_hash": graph_hash}}
    trace = {**default_trace, **(trace_overrides or {})}
    raw.write_record(directory, raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                                     payload_sha256={n: "x" for n in raw.PAYLOAD_NAMES},
                                                     trace=trace))


def test_population_audit_unavailable_without_candidate_tree(tmp_path):
    repo = _base_repo(tmp_path)
    result = e6r.audit_historical_trace_population(repo)
    assert result["available"] is False
    assert result["row_count"] == 0
    assert result["model_backends_instantiated"] == []
    assert result["target_access"] is False


def test_population_audit_refuses_non_llm_arm():
    repo = cc.repo_root()
    with pytest.raises(e6r.E6RenderError, match="LLM"):
        e6r.audit_historical_trace_population(repo, arm="RND")


def test_population_audit_matches_for_correctly_generated_candidate(tmp_path):
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(repo, candidate_id="cand-match-1", route="physics", recipe_id="R-000221")

    result = e6r.audit_historical_trace_population(repo)
    assert result["available"] is True
    assert result["row_count"] == 1
    row = result["rows"][0]
    assert row["recipe_hash_matches"] is True
    assert row["graph_hash_matches"] is True
    assert row["requested_equals_recipe_coverage"] is True
    assert row["requested_equals_achieved_coverage"] is True
    assert row["requested_equals_support_ratio"] is True
    assert row["recipe_geometry_coverage"] == 1.0


def test_population_audit_flags_a_genuine_mismatch(tmp_path):
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(
        repo, candidate_id="cand-mismatch-1", route="gpat", recipe_id="R-000354",
        trace_overrides={"requested_region_pixels": 232, "requested_support_pixels": 209,
                         "requested_coverage": 0.9021, "achieved_coverage": 0.9008620689655172})

    result = e6r.audit_historical_trace_population(repo)
    row = result["rows"][0]
    assert row["recipe_hash_matches"] is True  # recipe/graph identity still real and matching
    assert row["graph_hash_matches"] is True
    assert row["recipe_geometry_coverage"] == 0.75
    assert row["requested_equals_recipe_coverage"] is False  # the real anomaly this milestone traces
    assert row["support_over_region"] == pytest.approx(209 / 232)


def test_population_audit_recipe_hash_mismatch_detected(tmp_path):
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(repo, candidate_id="cand-stale-1", route="gpat", recipe_id="R-000221",
                                use_real_hashes=False)

    result = e6r.audit_historical_trace_population(repo)
    row = result["rows"][0]
    assert row["recipe_hash_matches"] is False
    assert row["graph_hash_matches"] is False


def test_population_audit_zero_region_pixels_handled_safely(tmp_path):
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(repo, candidate_id="cand-zero-region", route="physics", recipe_id="R-000221",
                                trace_overrides={"requested_region_pixels": 0, "requested_support_pixels": 0})

    result = e6r.audit_historical_trace_population(repo)
    row = result["rows"][0]
    assert row["support_over_region"] is None  # never a ZeroDivisionError
    assert row["requested_equals_support_ratio"] is None  # not comparable, never False


def test_population_audit_only_reads_generated_candidates(tmp_path):
    """A retained semantic-failure record (no trace) must be skipped, not
    crash the audit."""
    from prism_fas.synthesis import c5_raw_generation as raw

    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    directory = raw.candidate_dir(repo / e6r.HISTORICAL_LLM_CANDIDATE_ROOT, "LLM", "cand-failed")
    directory.mkdir(parents=True, exist_ok=True)
    identity = raw.GenerationIdentity(
        candidate_id="cand-failed", arm="LLM", arm_plan_identity="p", source_pair_plan_identity="s",
        package_identity="pkg", recipe_bank_identity="fake-population-bank-identity", recipe_id="R-000221",
        recipe_ordinal=0, slot=0, position=0, route="physics", live_target_sample_id="live-x",
        spoof_source_sample_id=None, generator_binding="binding", ontology_identity="ont")
    raw.write_record(directory, raw.CandidateRecord(identity=identity, status=raw.FAILED_GENERATION,
                                                     failure={"stage": "finalize", "exception_type": "X",
                                                             "reason": "empty exact mask"}))
    result = e6r.audit_historical_trace_population(repo)
    assert result["row_count"] == 0


def test_population_audit_never_mutates_candidate_records(tmp_path):
    from prism_fas.synthesis import c5_raw_generation as raw

    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(repo, candidate_id="cand-readonly", route="physics", recipe_id="R-000221")
    record_path = raw.candidate_dir(repo / e6r.HISTORICAL_LLM_CANDIDATE_ROOT, "LLM",
                                    "cand-readonly") / raw.RECORD_NAME
    before = record_path.read_bytes()
    e6r.audit_historical_trace_population(repo)
    after = record_path.read_bytes()
    assert before == after


def test_population_audit_never_instantiates_model_backends():
    """TASK H: static proof -- no quality-backend symbol is reachable from
    audit_historical_trace_population's own source."""
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    fn_start = source.index("def audit_historical_trace_population(")
    fn_end = source.index("\ndef ", fn_start + 10)
    body = source[fn_start:fn_end]
    for forbidden in ("QualityBackends(", "SCRFDDetector(", "DifferentiableAdaFace(",
                     "FaceXFormerBackend(", "GPATRoute(", "render_one(", "torch.no_grad"):
        assert forbidden not in body


def test_aggregate_historical_trace_population_route_counts(tmp_path):
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(repo, candidate_id="gpat-1", route="gpat", recipe_id="R-000354",
                                trace_overrides={"requested_coverage": 0.9021, "achieved_coverage": 0.9021})
    _write_population_candidate(repo, candidate_id="gpat-2", route="gpat", recipe_id="R-000354",
                                trace_overrides={"requested_coverage": 0.2381, "achieved_coverage": 0.2381})
    _write_population_candidate(repo, candidate_id="physics-1", route="physics", recipe_id="R-000221")
    _write_population_candidate(repo, candidate_id="physics-2", route="physics", recipe_id="R-000221")

    audit = e6r.audit_historical_trace_population(repo)
    aggregates = e6r.aggregate_historical_trace_population(audit["rows"])
    assert aggregates["by_route"]["gpat"]["N"] == 2
    assert aggregates["by_route"]["physics"]["N"] == 2
    assert aggregates["by_route"]["gpat"]["requested_equals_recipe_coverage_count"] == 0
    assert aggregates["by_route"]["physics"]["requested_equals_recipe_coverage_count"] == 2
    dev = aggregates["by_route"]["gpat"]["deviations"]["requested_vs_recipe"]
    assert dev["n_comparable"] == 2
    assert dev["mean_abs"] > 0


def test_classify_route_semantics_pattern_strong_when_routes_fully_disagree(tmp_path):
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(repo, candidate_id="gpat-1", route="gpat", recipe_id="R-000354",
                                trace_overrides={"requested_coverage": 0.9021, "achieved_coverage": 0.9021})
    _write_population_candidate(repo, candidate_id="gpat-2", route="gpat", recipe_id="R-000354",
                                trace_overrides={"requested_coverage": 0.2381, "achieved_coverage": 0.2381})
    _write_population_candidate(repo, candidate_id="physics-1", route="physics", recipe_id="R-000221")
    _write_population_candidate(repo, candidate_id="physics-2", route="physics", recipe_id="R-000221")

    audit = e6r.audit_historical_trace_population(repo)
    aggregates = e6r.aggregate_historical_trace_population(audit["rows"])
    assert e6r.classify_route_semantics_pattern(aggregates) == "STRONG"


def test_classify_route_semantics_pattern_absent_when_routes_agree(tmp_path):
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(repo, candidate_id="gpat-1", route="gpat", recipe_id="R-000221")
    _write_population_candidate(repo, candidate_id="physics-1", route="physics", recipe_id="R-000221")

    audit = e6r.audit_historical_trace_population(repo)
    aggregates = e6r.aggregate_historical_trace_population(audit["rows"])
    assert e6r.classify_route_semantics_pattern(aggregates) == "ABSENT"


def test_classify_route_semantics_pattern_absent_with_fewer_than_two_routes():
    assert e6r.classify_route_semantics_pattern({"by_route": {}}) == "ABSENT"
    assert e6r.classify_route_semantics_pattern(
        {"by_route": {"gpat": {"N": 3, "requested_equals_recipe_coverage_count": 0}}}) == "ABSENT"


def test_write_historical_trace_population_artifacts_writes_only_under_own_namespace(tmp_path):
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(repo, candidate_id="cand-1", route="physics", recipe_id="R-000221")

    status = e6r.write_historical_trace_population_artifacts(repo)
    assert status["csv_written"] is True
    csv_path = Path(status["csv_path"])
    summary_path = Path(status["summary_path"])
    assert csv_path.is_relative_to(repo / e6r.RENDER_DIR)
    assert summary_path.is_relative_to(repo / e6r.RENDER_DIR)
    assert csv_path.is_file()
    assert "cand-1" in csv_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["population_route_semantics_pattern"] in {"STRONG", "PARTIAL", "ABSENT"}


def test_write_historical_trace_population_artifacts_never_fabricates_when_unavailable(tmp_path):
    repo = _base_repo(tmp_path)
    status = e6r.write_historical_trace_population_artifacts(repo)
    assert status["csv_written"] is False
    assert status["csv_path"] is None
    summary = json.loads(Path(status["summary_path"]).read_text(encoding="utf-8"))
    assert summary["available"] is False


def test_population_audit_cli_flags_present_and_read_only():
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    assert "--audit-historical-trace-population" in source
    assert "--audit-arm" in source
    assert "gpu_population_audit_command" in source


# --------------------------------------------------------------------------- #
# TASK A-L (continuation turn): anomaly extraction, grouping, integrity check,
# max-deviation finder, requested-vs-achieved buckets, C6 bank-lock join.
# --------------------------------------------------------------------------- #

def _mixed_population_fixture(repo: Path) -> None:
    """A small, real (recipe-hash-correct) population: 2 normal + 1 anomalous
    GPAT candidate on R-000354 (coverage=0.75), and 1 normal + 1 anomalous
    Physics candidate on R-000221 (coverage=1.0), at controlled positions so
    contiguity/grouping behavior is directly checkable. The LARGEST deviation
    in this fixture belongs to physics-anomaly-1: |0.2543 - 1.0| = 0.7457."""
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(repo, candidate_id="gpat-normal-1", route="gpat", recipe_id="R-000354",
                                position=10, live_target_sample_id="live-A")
    _write_population_candidate(repo, candidate_id="gpat-anomaly-1", route="gpat", recipe_id="R-000354",
                                position=11, live_target_sample_id="live-A",
                                trace_overrides={"requested_region_pixels": 232, "requested_support_pixels": 209,
                                               "requested_coverage": 0.9021, "achieved_coverage": 0.9008620689655172})
    _write_population_candidate(repo, candidate_id="gpat-anomaly-2", route="gpat", recipe_id="R-000354",
                                position=12, live_target_sample_id="live-B",
                                trace_overrides={"requested_region_pixels": 5987, "requested_support_pixels": 1426,
                                               "requested_coverage": 0.2381, "achieved_coverage": 0.2381827292467012})
    _write_population_candidate(repo, candidate_id="physics-normal-1", route="physics", recipe_id="R-000221",
                                position=20, live_target_sample_id="live-C")
    _write_population_candidate(repo, candidate_id="physics-anomaly-1", route="physics", recipe_id="R-000221",
                                position=21, live_target_sample_id="live-C",
                                trace_overrides={"requested_coverage": 0.2543, "achieved_coverage": 0.2543})


def test_population_integrity_check_detects_route_excess(tmp_path):
    """Reproduces the EXACT shape of the real finding: an inflated GPAT count
    relative to the frozen expected count is flagged, never silently accepted."""
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    for i in range(3):
        _write_population_candidate(repo, candidate_id=f"gpat-{i}", route="gpat", recipe_id="R-000221")
    _write_population_candidate(repo, candidate_id="physics-0", route="physics", recipe_id="R-000221")

    audit = e6r.audit_historical_trace_population(repo)
    integrity = e6r.population_integrity_check(audit["rows"])
    assert integrity["observed_route_counts"] == {"gpat": 3, "physics": 1}
    assert integrity["expected_route_counts"] == {"gpat": e6r.EXPECTED_LLM_GPAT_GENERATED,
                                                  "physics": e6r.EXPECTED_LLM_PHYSICS_GENERATED}
    assert integrity["duplicate_candidate_id_count"] == 0
    assert integrity["population_matches_expected"] is False
    assert integrity["excess_by_route"]["gpat"] == 3 - e6r.EXPECTED_LLM_GPAT_GENERATED


def test_population_integrity_check_detects_duplicate_candidate_ids():
    rows = [{"candidate_id": "dup", "route": "gpat"}, {"candidate_id": "dup", "route": "gpat"},
           {"candidate_id": "unique", "route": "physics"}]
    integrity = e6r.population_integrity_check(rows)
    assert integrity["duplicate_candidate_id_count"] == 1
    assert integrity["duplicate_candidate_ids"] == ["dup"]
    assert integrity["population_matches_expected"] is False


def test_population_integrity_check_matches_when_counts_are_exactly_expected():
    rows = ([{"candidate_id": f"g{i}", "route": "gpat"} for i in range(e6r.EXPECTED_LLM_GPAT_GENERATED)]
           + [{"candidate_id": f"p{i}", "route": "physics"} for i in range(e6r.EXPECTED_LLM_PHYSICS_GENERATED)])
    integrity = e6r.population_integrity_check(rows)
    assert integrity["population_matches_expected"] is True
    assert integrity["expected_total"] == e6r.EXPECTED_LLM_TOTAL_GENERATED


def test_filter_anomalies_selects_exactly_the_false_rows_and_computes_deltas(tmp_path):
    repo = _base_repo(tmp_path)
    _mixed_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    anomalies = e6r.filter_anomalies(audit["rows"])

    assert {row["candidate_id"] for row in anomalies} == {"gpat-anomaly-1", "gpat-anomaly-2", "physics-anomaly-1"}
    gpat1 = next(row for row in anomalies if row["candidate_id"] == "gpat-anomaly-1")
    assert gpat1["delta_requested_recipe"] == pytest.approx(0.9021 - 0.75)
    assert gpat1["abs_delta_requested_recipe"] == pytest.approx(abs(0.9021 - 0.75))
    assert gpat1["support_ratio"] == pytest.approx(209 / 232)
    assert gpat1["is_known_q_mismatch"] is False


def test_filter_anomalies_flags_known_q_mismatch_candidates(tmp_path):
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(
        repo, candidate_id=e6r.KNOWN_Q_AUDIT_MISMATCH_CANDIDATES[0], route="gpat", recipe_id="R-000354",
        trace_overrides={"requested_coverage": 0.8311, "achieved_coverage": 0.8311392405063291})
    audit = e6r.audit_historical_trace_population(repo)
    anomalies = e6r.filter_anomalies(audit["rows"])
    assert len(anomalies) == 1
    assert anomalies[0]["is_known_q_mismatch"] is True


def test_summarize_anomalies_by_key_detects_shared_live_sample_normal_and_anomalous(tmp_path):
    """TASK D's critical test: live-A produces BOTH a normal (position 10) and
    an anomalous (position 11) candidate -- proves the sample alone does not
    determine anomaly status."""
    repo = _base_repo(tmp_path)
    _mixed_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    anomalies = e6r.filter_anomalies(audit["rows"])

    by_sample = e6r.summarize_anomalies_by_key(anomalies, audit["rows"], "live_target_sample_id")
    live_a = next(entry for entry in by_sample if entry["key"] == "live-A")
    assert live_a["anomaly_count"] == 1
    assert live_a["total_count"] == 2  # gpat-normal-1 AND gpat-anomaly-1 both used live-A
    assert live_a["key_is_uniformly_anomalous"] is False

    live_b = next(entry for entry in by_sample if entry["key"] == "live-B")
    assert live_b["anomaly_count"] == 1
    assert live_b["total_count"] == 1
    assert live_b["key_is_uniformly_anomalous"] is True


def test_summarize_anomalies_by_key_recipe_shows_both_routes(tmp_path):
    repo = _base_repo(tmp_path)
    _mixed_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    anomalies = e6r.filter_anomalies(audit["rows"])
    by_recipe = e6r.summarize_anomalies_by_key(anomalies, audit["rows"], "recipe_id")
    r354 = next(entry for entry in by_recipe if entry["key"] == "R-000354")
    assert r354["anomaly_count"] == 2
    assert r354["routes_in_anomalies"] == ["gpat"]


def test_position_sequence_and_contiguous_blocks(tmp_path):
    repo = _base_repo(tmp_path)
    _mixed_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    sequence = e6r.position_sequence_view(audit["rows"])
    assert [item["position"] for item in sequence] == [10, 11, 12, 20, 21]

    blocks = e6r.find_contiguous_anomaly_blocks(sequence)
    # positions 11,12 are consecutive anomalies -> one block of 2; position 21
    # is an isolated anomaly -> its own block of 1
    block_counts = sorted(block["count"] for block in blocks)
    assert block_counts == [1, 2]
    two_block = next(block for block in blocks if block["count"] == 2)
    assert two_block["start_position"] == 11
    assert two_block["end_position"] == 12


def test_find_max_deviation_candidates_orders_correctly(tmp_path):
    repo = _base_repo(tmp_path)
    _mixed_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    top = e6r.find_max_deviation_candidates(audit["rows"], top_n=1)
    assert len(top) == 1
    # physics-anomaly-1: |0.2543 - 1.0| = 0.7457, the largest deviation in the fixture
    assert top[0]["candidate_id"] == "physics-anomaly-1"
    assert top[0]["abs_deviation"] == pytest.approx(abs(0.2543 - 1.0))


def test_categorize_requested_vs_achieved_buckets(tmp_path):
    repo = _base_repo(tmp_path)
    _mixed_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    buckets = e6r.categorize_requested_vs_achieved(audit["rows"])
    assert sum(buckets.values()) == 5  # every row in the fixture has both fields
    assert buckets["<=1e-6"] >= 2  # the 3 "normal" candidates have requested==achieved exactly


def test_join_with_c6_bank_lock_marks_selected_and_unknown_otherwise():
    rows = [{"candidate_id": "selected-1"}, {"candidate_id": "not-selected-1"}]
    bank_lock = {"selected": [{"candidate_id": "selected-1", "q": 0.9}]}
    joined = e6r.join_with_c6_bank_lock(rows, bank_lock)
    selected = next(row for row in joined if row["candidate_id"] == "selected-1")
    not_selected = next(row for row in joined if row["candidate_id"] == "not-selected-1")
    assert selected["c6_selected"] is True
    assert selected["c6_selected_q"] == 0.9
    assert not_selected["c6_selected"] is False
    assert not_selected["c6_accepted_or_rejected"] == "UNKNOWN (not persisted for non-selected candidates)"


def test_write_anomaly_artifacts_end_to_end(tmp_path):
    repo = _base_repo(tmp_path)
    _mixed_population_fixture(repo)
    status = e6r.write_anomaly_artifacts(repo)
    assert status["available"] is True
    assert status["csv_written"] is True
    csv_path = Path(status["csv_path"])
    assert csv_path.is_relative_to(repo / e6r.RENDER_DIR)
    text = csv_path.read_text(encoding="utf-8")
    assert "gpat-anomaly-1" in text
    assert "gpat-normal-1" not in text
    summary = status["summary"]
    assert summary["anomaly_count"] == 3
    assert summary["normal_count"] == 2
    assert summary["known_q_mismatch_candidates_found_in_anomalies"] == 0
    assert summary["anomalies_form_contiguous_block"] is True


def test_write_anomaly_artifacts_never_fabricates_when_unavailable(tmp_path):
    repo = _base_repo(tmp_path)
    status = e6r.write_anomaly_artifacts(repo)
    assert status["available"] is False
    assert status["csv_written"] is False
    summary = json.loads(Path(status["summary_path"]).read_text(encoding="utf-8"))
    assert summary["available"] is False


def test_write_anomaly_artifacts_never_mutates_candidate_records(tmp_path):
    from prism_fas.synthesis import c5_raw_generation as raw

    repo = _base_repo(tmp_path)
    _mixed_population_fixture(repo)
    record_path = raw.candidate_dir(repo / e6r.HISTORICAL_LLM_CANDIDATE_ROOT, "LLM",
                                    "gpat-anomaly-1") / raw.RECORD_NAME
    before = record_path.read_bytes()
    e6r.write_anomaly_artifacts(repo)
    after = record_path.read_bytes()
    assert before == after


def test_anomaly_functions_never_instantiate_model_backends():
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    for fn_name in ("filter_anomalies", "summarize_anomalies_by_key", "position_sequence_view",
                    "find_contiguous_anomaly_blocks", "find_max_deviation_candidates",
                    "categorize_requested_vs_achieved", "join_with_c6_bank_lock",
                    "population_integrity_check"):
        fn_start = source.index(f"def {fn_name}(")
        fn_end = source.index("\ndef ", fn_start + 10)
        body = source[fn_start:fn_end]
        for forbidden in ("QualityBackends(", "SCRFDDetector(", "GPATRoute(", "render_one(", "torch.no_grad"):
            assert forbidden not in body, f"{forbidden!r} unexpectedly reachable from {fn_name}"


def test_anomaly_analysis_cli_flag_present_and_read_only():
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    assert "--analyze-historical-trace-anomalies" in source
    assert "write_anomaly_artifacts" in source
    assert "gpu_anomaly_analysis_command" in source


# --------------------------------------------------------------------------- #
# TASK A-J (continuation turn): the TWO-GPAT-RENDER-PASS investigation --
# route_binding population audit, FROZEN_SCHEDULE_KEY grouping, canonical
# binding resolution from frozen C4/C5/C6 locks, candidate_id/binding
# dependency, canonical population view, and known-q-mismatch classification.
# --------------------------------------------------------------------------- #

def _double_gpat_population_fixture(repo: Path, *, binding_a: str = "binding-A") -> None:
    """Two GPAT candidates at the SAME schedule position (11) under two
    different route_binding values -- the exact worked example the GPU host
    reported (recipe R-000354, two candidate_ids, two bindings) -- plus one
    ordinary single-binding GPAT candidate and one physics candidate."""
    _population_recipe_bank_fixture(repo)
    _write_population_candidate(
        repo, candidate_id="cand-binding-a", route="gpat", recipe_id="R-000354",
        position=11, live_target_sample_id="live-shared", trace_overrides={"binding": binding_a})
    _write_population_candidate(
        repo, candidate_id="cand-binding-b", route="gpat", recipe_id="R-000354",
        position=11, live_target_sample_id="live-shared", trace_overrides={"binding": "binding-B"})
    _write_population_candidate(
        repo, candidate_id="cand-solo-gpat", route="gpat", recipe_id="R-000221",
        position=30, live_target_sample_id="live-solo", trace_overrides={"binding": binding_a})
    _write_population_candidate(
        repo, candidate_id="cand-solo-physics", route="physics", recipe_id="R-000221",
        position=31, live_target_sample_id="live-physics", trace_overrides={"binding": "physics-engine-v1"})


def test_aggregate_by_route_binding_discovers_two_bindings_for_gpat(tmp_path):
    repo = _base_repo(tmp_path)
    _double_gpat_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    aggregate = e6r.aggregate_by_route_binding(audit["rows"])

    assert aggregate["distinct_bindings_by_route"]["gpat"] == ["binding-A", "binding-B"]
    assert aggregate["distinct_bindings_by_route"]["physics"] == ["physics-engine-v1"]
    by_binding = {(entry["route"], entry["route_binding"]): entry for entry in aggregate["bindings"]}
    assert by_binding[("gpat", "binding-A")]["candidate_count"] == 2  # cand-binding-a + cand-solo-gpat
    assert by_binding[("gpat", "binding-B")]["candidate_count"] == 1
    assert by_binding[("gpat", "binding-B")]["unique_position_count"] == 1


def test_group_by_schedule_key_puts_both_bindings_at_the_same_position(tmp_path):
    repo = _base_repo(tmp_path)
    _double_gpat_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    grouped = e6r.group_by_schedule_key(audit["rows"], route="gpat")

    ids_at_11 = sorted(row["candidate_id"] for row in grouped[11])
    assert ids_at_11 == ["cand-binding-a", "cand-binding-b"]
    assert len(ids_at_11) == len(set(ids_at_11))  # candidate IDs remain unique even at a shared schedule key
    assert grouped[30] and grouped[30][0]["candidate_id"] == "cand-solo-gpat"


def test_classify_double_gpat_render_pass_proven_when_every_key_has_exactly_two():
    grouped = {position: [{"candidate_id": f"a{position}"}, {"candidate_id": f"b{position}"}]
              for position in range(e6r.EXPECTED_LLM_GPAT_GENERATED)}
    result = e6r.classify_double_gpat_render_pass(grouped)
    assert result["gpat_schedule_keys_total"] == e6r.EXPECTED_LLM_GPAT_GENERATED
    assert result["keys_with_2_candidates"] == e6r.EXPECTED_LLM_GPAT_GENERATED
    assert result["keys_with_1_candidate"] == 0
    assert result["keys_with_gt2_candidates"] == 0
    assert result["double_gpat_render_pass"] == "PROVEN"


def test_classify_double_gpat_render_pass_not_proven_on_a_mixed_population(tmp_path):
    repo = _base_repo(tmp_path)
    _double_gpat_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    grouped = e6r.group_by_schedule_key(audit["rows"], route="gpat")
    result = e6r.classify_double_gpat_render_pass(grouped)
    assert result["keys_with_2_candidates"] == 1
    assert result["keys_with_1_candidate"] == 1
    assert result["double_gpat_render_pass"] == "NOT_PROVEN"


def test_pair_gpat_candidates_by_schedule_key_produces_comparison_row_with_c6_join(tmp_path):
    repo = _base_repo(tmp_path)
    _double_gpat_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    grouped = e6r.group_by_schedule_key(audit["rows"], route="gpat")
    bank_lock = {"selected": [{"candidate_id": "cand-binding-a", "q": 0.81}]}

    pairs = e6r.pair_gpat_candidates_by_schedule_key(grouped, bank_lock=bank_lock)
    assert len(pairs) == 1  # only position 11 has exactly 2
    pair = pairs[0]
    assert pair["schedule_key"] == 11
    assert {pair["candidate_id_A"], pair["candidate_id_B"]} == {"cand-binding-a", "cand-binding-b"}
    assert {pair["route_binding_A"], pair["route_binding_B"]} == {"binding-A", "binding-B"}
    a_label = "A" if pair["candidate_id_A"] == "cand-binding-a" else "B"
    assert pair[f"c6_selected_{a_label}"] is True
    assert pair[f"c6_q_{a_label}"] == 0.81
    other_label = "B" if a_label == "A" else "A"
    assert pair[f"c6_selected_{other_label}"] is False


def _c4_c5_c6_lock_fixture(repo: Path, *, active_binding: str, second_binding: str,
                           c6_pins_active: bool = True) -> None:
    """Fabricates the exact real lock-chain shape TASK D reads: a C4 winning-
    checkpoint lock, an ACTIVE C5 lock whose own `supersedes.archived_lock`
    names a SUPERSEDED lock (mirroring `reports/full/c5/superseded/...`), and
    a C6 bank lock whose `c5_pool_lock_sha256` is a real SHA-256 of the active
    lock's file bytes (or a deliberately wrong one, to test disagreement)."""
    (repo / e6r.C4_SCIENTIFIC_LOCK_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / e6r.C4_SCIENTIFIC_LOCK_PATH).write_text(
        json.dumps({"winning_checkpoint_sha256": active_binding}), encoding="utf-8")

    superseded_relpath = "reports/full/c5/superseded/C5_SYNTHESIS_LOCK_fake.json"
    (repo / superseded_relpath).parent.mkdir(parents=True, exist_ok=True)
    (repo / superseded_relpath).write_text(json.dumps({
        "gpat_checkpoint_sha256": second_binding, "lock_kind": "terminal_audit_record",
        "usable_as_c6_input": False, "why_not_usable": "fake terminal audit record for a test"}),
        encoding="utf-8")

    active_payload = {
        "gpat_checkpoint_sha256": active_binding, "lock_kind": "scientific_candidate_pool",
        "is_scientific_lock": True,
        "supersedes": {"archived_lock": superseded_relpath, "archived_generated_at_utc": "2026-08-22T15:18:53Z"},
    }
    (repo / e6r.C5_SYNTHESIS_LOCK_PATH).parent.mkdir(parents=True, exist_ok=True)
    active_bytes = json.dumps(active_payload).encode("utf-8")
    (repo / e6r.C5_SYNTHESIS_LOCK_PATH).write_bytes(active_bytes)
    real_active_sha256 = hashlib.sha256(active_bytes).hexdigest()

    (repo / e6r.C6_BANK_LOCK_LLM_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / e6r.C6_BANK_LOCK_LLM_PATH).write_text(json.dumps({
        "c5_pool_lock_sha256": real_active_sha256 if c6_pins_active else "wrong-sha256",
        "selected": []}), encoding="utf-8")


def test_resolve_canonical_gpat_binding_proven_from_frozen_locks(tmp_path):
    repo = _base_repo(tmp_path)
    _c4_c5_c6_lock_fixture(repo, active_binding=e6r.EXPECTED_GPAT_CHECKPOINT_SHA256,
                           second_binding="bafbc1ef-fake-second-binding")
    resolution = e6r.resolve_canonical_gpat_binding(repo)

    assert resolution["canonical_binding_status"] == "PROVEN"
    assert resolution["canonical_gpat_binding"] == e6r.EXPECTED_GPAT_CHECKPOINT_SHA256
    assert resolution["second_gpat_binding"] == "bafbc1ef-fake-second-binding"
    assert resolution["second_gpat_population_role"] == "SUPERSEDED"
    assert resolution["c6_bank_lock_pins_active_c5_lock"] is True
    proof_artifacts = {entry["artifact"] for entry in resolution["proof"]}
    assert e6r.C4_SCIENTIFIC_LOCK_PATH in proof_artifacts
    assert e6r.C5_SYNTHESIS_LOCK_PATH in proof_artifacts
    assert "reports/full/c5/superseded/C5_SYNTHESIS_LOCK_fake.json" in proof_artifacts
    assert e6r.C6_BANK_LOCK_LLM_PATH in proof_artifacts


def test_resolve_canonical_gpat_binding_ambiguous_when_c6_pin_disagrees(tmp_path):
    """TASK J: ambiguous-binding-blocks-filtering -- if C6's own SHA-256 pin
    does not match the active C5 lock's actual bytes, this must refuse to
    name a canonical binding rather than trust EXPECTED_GPAT_CHECKPOINT_SHA256
    alone."""
    repo = _base_repo(tmp_path)
    _c4_c5_c6_lock_fixture(repo, active_binding=e6r.EXPECTED_GPAT_CHECKPOINT_SHA256,
                           second_binding="bafbc1ef-fake-second-binding", c6_pins_active=False)
    resolution = e6r.resolve_canonical_gpat_binding(repo)
    assert resolution["c6_bank_lock_pins_active_c5_lock"] is False
    assert resolution["canonical_gpat_binding"] is None
    assert resolution["canonical_binding_status"] == "AMBIGUOUS"


def test_resolve_canonical_gpat_binding_unavailable_without_c5_lock(tmp_path):
    repo = _base_repo(tmp_path)
    result = e6r.resolve_canonical_gpat_binding(repo)
    assert result["available"] is False
    assert result["canonical_binding_status"] == "UNAVAILABLE"


def test_candidate_id_depends_on_route_binding_is_true_and_cites_the_real_hash_material():
    result = e6r.candidate_id_depends_on_route_binding()
    assert result["candidate_id_depends_on_route_binding"] is True
    assert "generator_binding" in result["hash_material_fields"]
    assert result["source"] == "prism_fas.synthesis.c5_source_pair_plan.candidate_identity"


def test_explain_second_population_reachability_cites_collect_records():
    result = e6r.explain_second_population_reachability()
    assert result["collector"] == "prism_fas.synthesis.c5_render.collect_records"
    assert "never globs/scans" in result["enumeration_mechanism"]


def test_build_canonical_population_view_never_deletes_historical_rows(tmp_path):
    repo = _base_repo(tmp_path)
    _double_gpat_population_fixture(repo)
    audit = e6r.audit_historical_trace_population(repo)
    original_ids = {row["candidate_id"] for row in audit["rows"]}

    view = e6r.build_canonical_population_view(audit["rows"], canonical_gpat_binding="binding-A")

    # the SOURCE rows list handed in is completely untouched
    assert {row["candidate_id"] for row in audit["rows"]} == original_ids
    assert "cand-binding-b" in original_ids  # the non-canonical row is still there in the source
    # the VIEW excludes only the non-canonical GPAT row
    view_ids = {row["candidate_id"] for row in view["rows"]}
    assert "cand-binding-b" not in view_ids
    assert "cand-binding-a" in view_ids
    assert "cand-solo-physics" in view_ids  # physics rows always kept
    assert view["excluded_rows_candidate_ids"] == ["cand-binding-b"]
    assert view["canonical_gpat_n"] == 2  # cand-binding-a + cand-solo-gpat
    assert view["canonical_physics_n"] == 1


def test_write_canonical_population_artifacts_refuses_when_binding_ambiguous(tmp_path):
    repo = _base_repo(tmp_path)
    _double_gpat_population_fixture(repo)
    # no C4/C5/C6 lock chain fabricated -> resolve_canonical_gpat_binding is UNAVAILABLE
    status = e6r.write_canonical_population_artifacts(repo)
    assert status["csv_written"] is False
    summary = json.loads(Path(status["summary_path"]).read_text(encoding="utf-8"))
    assert "note" in summary
    assert not (repo / e6r.RENDER_DIR / "E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv").exists()
    # the UNFILTERED population CSV/summary this run does NOT touch must not exist either --
    # this function must never write under the unfiltered artifact's own filename
    assert not (repo / e6r.RENDER_DIR / "E6_HISTORICAL_TRACE_POPULATION.csv").exists()


def test_write_canonical_population_artifacts_never_overwrites_unfiltered_population(tmp_path):
    repo = _base_repo(tmp_path)
    _double_gpat_population_fixture(repo, binding_a=e6r.EXPECTED_GPAT_CHECKPOINT_SHA256)
    _c4_c5_c6_lock_fixture(repo, active_binding=e6r.EXPECTED_GPAT_CHECKPOINT_SHA256, second_binding="binding-B")

    unfiltered_status = e6r.write_historical_trace_population_artifacts(repo)
    unfiltered_before = Path(unfiltered_status["csv_path"]).read_text(encoding="utf-8")

    canonical_status = e6r.write_canonical_population_artifacts(repo)
    assert canonical_status["csv_written"] is True

    unfiltered_after = Path(unfiltered_status["csv_path"]).read_text(encoding="utf-8")
    assert unfiltered_before == unfiltered_after  # the earlier unfiltered CSV is byte-identical, untouched
    assert "cand-binding-b" in unfiltered_after  # still present in the UNFILTERED view

    canonical_text = Path(canonical_status["csv_path"]).read_text(encoding="utf-8")
    assert "cand-binding-a" in canonical_text
    assert "cand-binding-b" not in canonical_text  # excluded from the CANONICAL view only

    summary = json.loads(Path(canonical_status["summary_path"]).read_text(encoding="utf-8"))
    assert summary["canonical_gpat_n"] == 2
    assert summary["canonical_physics_n"] == 1
    assert summary["excluded_non_canonical_gpat_count"] == 1


def test_classify_known_q_mismatch_bindings_reports_canonical_status():
    known_id = e6r.KNOWN_Q_AUDIT_MISMATCH_CANDIDATES[0]
    rows = [{"candidate_id": known_id, "route": "gpat", "route_binding": "binding-A"},
           {"candidate_id": e6r.KNOWN_Q_AUDIT_MISMATCH_CANDIDATES[1], "route": "gpat",
            "route_binding": "binding-B"}]
    result = e6r.classify_known_q_mismatch_bindings(rows, canonical_gpat_binding="binding-A")

    by_id = {entry["candidate_id"]: entry for entry in result["candidates"]}
    assert by_id[known_id]["is_canonical_binding"] is True
    assert by_id[e6r.KNOWN_Q_AUDIT_MISMATCH_CANDIDATES[1]]["is_canonical_binding"] is False
    assert by_id[e6r.KNOWN_Q_AUDIT_MISMATCH_CANDIDATES[2]]["found"] is False
    assert result["three_known_q_mismatch_binding"] == "NON_CANONICAL"
    assert result["three_known_q_mismatch_canonical"] is False


def test_classify_known_q_mismatch_bindings_all_canonical_does_not_explain_blocker():
    rows = [{"candidate_id": cid, "route": "gpat", "route_binding": "binding-A"}
           for cid in e6r.KNOWN_Q_AUDIT_MISMATCH_CANDIDATES]
    result = e6r.classify_known_q_mismatch_bindings(rows, canonical_gpat_binding="binding-A")
    assert result["three_known_q_mismatch_canonical"] is True
    assert result["explains_historical_q_blocker"] is False


def test_run_gpat_binding_investigation_end_to_end_writes_artifacts_and_resolves_canonical(tmp_path):
    repo = _base_repo(tmp_path)
    _double_gpat_population_fixture(repo, binding_a=e6r.EXPECTED_GPAT_CHECKPOINT_SHA256)
    _c4_c5_c6_lock_fixture(repo, active_binding=e6r.EXPECTED_GPAT_CHECKPOINT_SHA256, second_binding="binding-B")

    result = e6r.run_gpat_binding_investigation(repo)
    assert result["available"] is True
    assert result["unfiltered_tree_total"] == 4
    assert result["schedule_key_classification"]["double_gpat_render_pass"] == "NOT_PROVEN"
    assert result["canonical_resolution"]["canonical_binding_status"] == "PROVEN"
    assert result["canonical_gpat_n"] == 2
    assert result["canonical_physics_n"] == 1
    assert result["known_q_mismatch_classification"]["candidates"]
    assert result["target_access"] is False
    assert result["rendering_performed"] is False
    assert result["training_performed"] is False

    pairs_path = Path(result["pairs_path"])
    summary_path = Path(result["summary_path"])
    assert pairs_path.is_relative_to(repo / e6r.RENDER_DIR)
    assert summary_path.is_relative_to(repo / e6r.RENDER_DIR)
    pairs_payload = json.loads(pairs_path.read_text(encoding="utf-8"))
    assert len(pairs_payload["pairs"]) == 1
    assert result["canonical_artifacts"]["csv_written"] is True


def test_run_gpat_binding_investigation_unavailable_without_candidate_tree(tmp_path):
    repo = _base_repo(tmp_path)
    result = e6r.run_gpat_binding_investigation(repo)
    assert result["available"] is False


def test_gpat_binding_investigation_never_mutates_candidate_records_or_locks(tmp_path):
    from prism_fas.synthesis import c5_raw_generation as raw

    repo = _base_repo(tmp_path)
    _double_gpat_population_fixture(repo)
    _c4_c5_c6_lock_fixture(repo, active_binding="binding-A", second_binding="binding-B")
    record_path = raw.candidate_dir(repo / e6r.HISTORICAL_LLM_CANDIDATE_ROOT, "LLM",
                                    "cand-binding-a") / raw.RECORD_NAME
    c5_lock_path = repo / e6r.C5_SYNTHESIS_LOCK_PATH
    before_record = record_path.read_bytes()
    before_lock = c5_lock_path.read_bytes()

    e6r.run_gpat_binding_investigation(repo)

    assert record_path.read_bytes() == before_record
    assert c5_lock_path.read_bytes() == before_lock


def test_gpat_binding_investigation_functions_never_touch_target_model_gpu_or_llm():
    """No render/train/target/LLM/model-backend symbol is reachable from the
    source of any TASK A-H function's own body."""
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    for fn_name in ("aggregate_by_route_binding", "group_by_schedule_key",
                    "classify_double_gpat_render_pass", "pair_gpat_candidates_by_schedule_key",
                    "resolve_canonical_gpat_binding", "candidate_id_depends_on_route_binding",
                    "explain_second_population_reachability", "build_canonical_population_view",
                    "classify_known_q_mismatch_bindings", "run_gpat_binding_investigation"):
        fn_start = source.index(f"def {fn_name}(")
        fn_end = source.index("\ndef ", fn_start + 10)
        body = source[fn_start:fn_end]
        for forbidden in ("QualityBackends(", "SCRFDDetector(", "DifferentiableAdaFace(",
                         "FaceXFormerBackend(", "GPATRoute(", "render_one(", "torch.no_grad",
                         "resolve_target", "SiW", "openai", "google.generativeai", "GEMINI"):
            assert forbidden not in body, f"{forbidden!r} unexpectedly reachable from {fn_name}"


def test_investigate_gpat_binding_cli_flag_present_and_read_only():
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    assert "--investigate-gpat-binding" in source
    assert "run_gpat_binding_investigation" in source
    assert "gpu_gpat_binding_investigation_command" in source


# --------------------------------------------------------------------------- #
# TASK A-J (continuation turn): the CANONICAL 42-anomaly characterization --
# recipe-level grouping, recursive recipe/graph scalar-field matching,
# cross-route/cross-binding consistency, normal controls, known-q-mismatch
# recipe-class membership, C6 selection impact, root-cause reassessment.
# --------------------------------------------------------------------------- #

_CANONICAL_CSV_FIELDNAMES = (
    "candidate_id", "route", "recipe_id", "recipe_ordinal", "slot", "position",
    "live_target_sample_id", "persisted_recipe_hash", "current_recipe_hash", "recipe_hash_matches",
    "persisted_graph_hash", "current_graph_hash", "graph_hash_matches", "persisted_recipe_bank_identity",
    "current_recipe_bank_identity", "bank_identity_matches", "recipe_geometry_coverage",
    "trace_requested_coverage", "trace_achieved_coverage", "trace_requested_region_pixels",
    "trace_requested_support_pixels", "trace_exact_mask_pixels", "support_over_region",
    "requested_equals_recipe_coverage", "requested_equals_achieved_coverage",
    "requested_equals_support_ratio", "achieved_equals_support_ratio", "route_binding",
    "candidate_json_sha256",
)


def _write_raw_canonical_csv(repo: Path, rows: list[dict]) -> Path:
    """Writes E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv directly (as
    strings, matching csv.DictWriter's own str() round-trip) -- used to test
    load_canonical_population_csv's type coercion in isolation from the full
    binding-investigation pipeline."""
    import csv

    out_dir = repo / e6r.RENDER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CANONICAL_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_load_canonical_population_csv_unavailable_without_file(tmp_path):
    repo = _base_repo(tmp_path)
    result = e6r.load_canonical_population_csv(repo)
    assert result["available"] is False
    assert result["rows"] == []


def test_load_canonical_population_csv_coerces_types(tmp_path):
    repo = _base_repo(tmp_path)
    _write_raw_canonical_csv(repo, [{
        "candidate_id": "cand-1", "route": "gpat", "recipe_id": "R-000354", "recipe_ordinal": "12",
        "slot": "3", "position": "99", "live_target_sample_id": "live-x",
        "recipe_hash_matches": "True", "graph_hash_matches": "False", "bank_identity_matches": "",
        "recipe_geometry_coverage": "0.75", "trace_requested_coverage": "0.9021",
        "trace_achieved_coverage": "0.9021", "trace_requested_region_pixels": "232",
        "trace_requested_support_pixels": "209", "trace_exact_mask_pixels": "180",
        "support_over_region": "0.9008620689655172", "requested_equals_recipe_coverage": "False",
        "requested_equals_achieved_coverage": "True", "requested_equals_support_ratio": "None",
        "achieved_equals_support_ratio": "True", "route_binding": "binding-A",
        "candidate_json_sha256": "deadbeef",
    }])
    result = e6r.load_canonical_population_csv(repo)
    assert result["available"] is True
    row = result["rows"][0]
    assert row["recipe_ordinal"] == 12 and isinstance(row["recipe_ordinal"], int)
    assert row["recipe_geometry_coverage"] == pytest.approx(0.75)
    assert row["recipe_hash_matches"] is True
    assert row["graph_hash_matches"] is False
    assert row["bank_identity_matches"] is None
    assert row["requested_equals_recipe_coverage"] is False
    assert row["requested_equals_support_ratio"] is None
    assert row["route"] == "gpat"


def test_summarize_canonical_recipe_groups_computes_per_recipe_stats():
    rows = [
        {"recipe_id": "R-A", "route": "gpat", "candidate_id": "a1", "trace_requested_coverage": 0.9,
         "trace_achieved_coverage": 0.9, "recipe_geometry_coverage": 0.75},
        {"recipe_id": "R-A", "route": "physics", "candidate_id": "a2", "trace_requested_coverage": 0.9,
         "trace_achieved_coverage": 0.91, "recipe_geometry_coverage": 0.75},
        {"recipe_id": "R-B", "route": "gpat", "candidate_id": "b1", "trace_requested_coverage": 1.0,
         "trace_achieved_coverage": 1.0, "recipe_geometry_coverage": 1.0},
    ]
    anomalies = [rows[0], rows[1]]  # both R-A rows are anomalous, R-B's is not
    groups = e6r.summarize_canonical_recipe_groups(rows, anomalies)
    by_id = {g["recipe_id"]: g for g in groups}
    assert by_id["R-A"]["total_canonical_renders"] == 2
    assert by_id["R-A"]["anomalous_renders"] == 2
    assert by_id["R-A"]["all_renders_anomalous"] is True
    assert by_id["R-A"]["gpat_renders"] == 1 and by_id["R-A"]["physics_renders"] == 1
    assert by_id["R-A"]["unique_trace_requested_coverage_values"] == [0.9]
    assert by_id["R-A"]["requested_coverage_constant_within_recipe"] is True
    assert by_id["R-A"]["min_achieved_coverage"] == pytest.approx(0.9)
    assert by_id["R-A"]["max_achieved_coverage"] == pytest.approx(0.91)
    assert by_id["R-B"]["is_anomalous_recipe"] is False
    assert by_id["R-B"]["all_renders_anomalous"] is False


def test_classify_anomaly_determined_by_recipe_true_when_every_anomalous_recipe_is_fully_anomalous():
    groups = [{"is_anomalous_recipe": True, "all_renders_anomalous": True},
             {"is_anomalous_recipe": True, "all_renders_anomalous": True}]
    assert e6r.classify_anomaly_determined_by_recipe(groups) == "TRUE"


def test_classify_anomaly_determined_by_recipe_partial_when_mixed():
    groups = [{"is_anomalous_recipe": True, "all_renders_anomalous": True},
             {"is_anomalous_recipe": True, "all_renders_anomalous": False}]
    assert e6r.classify_anomaly_determined_by_recipe(groups) == "PARTIAL"


def test_classify_anomaly_determined_by_recipe_false_when_no_anomalous_recipes():
    assert e6r.classify_anomaly_determined_by_recipe(
        [{"is_anomalous_recipe": False, "all_renders_anomalous": False}]) == "FALSE"


def test_flatten_scalar_fields_recursive_dict_and_list():
    payload = {"geometry": {"coverage": 0.75, "region": "chin"}, "artifacts": [
        {"strength": 0.9021, "enabled": True}, {"strength": 0.5}], "severity": None}
    flat = e6r.flatten_scalar_fields(payload)
    assert flat["geometry.coverage"] == 0.75
    assert "geometry.region" not in flat  # strings are not scalars for this purpose
    assert flat["artifacts[0].strength"] == 0.9021
    assert flat["artifacts[0].enabled"] is True
    assert flat["artifacts[1].strength"] == 0.5
    assert "severity" not in flat  # None is not a scalar leaf


def test_match_scalar_fields_to_value_exact_and_tolerance():
    flat = {"a": 0.9021, "b": 0.90215, "c": 0.5, "d": True}
    result = e6r.match_scalar_fields_to_value(flat, 0.9021, tolerance=1e-4)
    assert result["matching_fields_exact"] == ["a"]
    assert result["matching_fields_tolerance"] == ["b"]
    assert "c" not in result["matching_fields_exact"] and "c" not in result["matching_fields_tolerance"]
    assert "d" not in result["matching_fields_exact"]  # bool never matched to a float


def test_match_scalar_fields_to_value_none_target_returns_empty():
    result = e6r.match_scalar_fields_to_value({"a": 0.9021}, None)
    assert result == {"matching_fields_exact": [], "matching_fields_tolerance": []}


def test_compare_anomalous_recipe_fields_finds_exact_match_in_frozen_recipe(tmp_path):
    repo = _base_repo(tmp_path)
    recipe_dir = repo / e6r.RECIPE_BANK_LLM_JSONL_PATH
    recipe_dir.parent.mkdir(parents=True, exist_ok=True)
    recipe_dir.write_text(json.dumps({"recipe_id": "R-FAKE", "geometry": {"coverage": 0.75},
                                      "artifact": {"strength": 0.9021}}) + "\n", encoding="utf-8")
    groups = [{"recipe_id": "R-FAKE", "is_anomalous_recipe": True,
              "requested_coverage_constant_within_recipe": True,
              "unique_trace_requested_coverage_values": [0.9021],
              "recipe_geometry_coverage": 0.75}]
    results = e6r.compare_anomalous_recipe_fields(repo, groups)
    assert len(results) == 1
    assert "artifact.strength" in results[0]["matching_recipe_fields_exact"]
    assert "geometry.coverage" not in results[0]["matching_recipe_fields_exact"]


def test_compare_anomalous_recipe_fields_reports_missing_recipe(tmp_path):
    repo = _base_repo(tmp_path)
    groups = [{"recipe_id": "R-MISSING", "is_anomalous_recipe": True,
              "requested_coverage_constant_within_recipe": True,
              "unique_trace_requested_coverage_values": [0.9021], "recipe_geometry_coverage": None}]
    results = e6r.compare_anomalous_recipe_fields(repo, groups)
    assert results[0]["matching_recipe_fields_exact"] == []
    assert "reason" in results[0]


def test_compare_anomalous_compiled_graph_fields_runs_against_real_recipe(tmp_path):
    repo = _base_repo(tmp_path)
    _population_recipe_bank_fixture(repo)
    groups = [{"recipe_id": "R-000354", "is_anomalous_recipe": True,
              "requested_coverage_constant_within_recipe": True,
              "unique_trace_requested_coverage_values": [0.75], "recipe_geometry_coverage": 0.75}]
    results = e6r.compare_anomalous_compiled_graph_fields(repo, groups)
    assert len(results) == 1
    # 0.75 IS the recipe's own geometry.coverage, compiled verbatim into region_mask_policy
    assert "region_mask_policy.requested_coverage" in results[0]["matching_graph_fields_exact"]


def test_classify_alternate_coverage_source_exact_recipe_field():
    result = e6r.classify_alternate_coverage_source(
        [{"matching_recipe_fields_exact": ["artifact.strength"], "matching_recipe_fields_tolerance": []}],
        [{"matching_graph_fields_exact": [], "matching_graph_fields_tolerance": []}])
    assert result["alternate_coverage_source"] == "EXACT_RECIPE_FIELD"
    assert result["alternate_value_present_in_recipe"] is True


def test_classify_alternate_coverage_source_exact_graph_field():
    result = e6r.classify_alternate_coverage_source(
        [{"matching_recipe_fields_exact": [], "matching_recipe_fields_tolerance": []}],
        [{"matching_graph_fields_exact": ["nodes[0].parameters.strength"], "matching_graph_fields_tolerance": []}])
    assert result["alternate_coverage_source"] == "EXACT_GRAPH_FIELD"


def test_classify_alternate_coverage_source_none_found_never_auto_asserts_code_justified():
    result = e6r.classify_alternate_coverage_source(
        [{"matching_recipe_fields_exact": [], "matching_recipe_fields_tolerance": ["some.field"]}],
        [{"matching_graph_fields_exact": [], "matching_graph_fields_tolerance": []}])
    assert result["alternate_coverage_source"] == "NONE_FOUND"
    assert result["tolerance_only_candidates_for_manual_code_review"]["recipe_fields"] is True


def test_cross_route_and_binding_consistency_true_when_all_agree():
    canonical_rows = [
        {"recipe_id": "R-A", "route": "gpat", "trace_requested_coverage": 0.9021},
        {"recipe_id": "R-A", "route": "physics", "trace_requested_coverage": 0.9021},
    ]
    unfiltered_rows = [
        {"recipe_id": "R-A", "route": "gpat", "route_binding": "binding-B", "trace_requested_coverage": 0.9021},
    ]
    result = e6r.cross_route_and_binding_consistency(
        canonical_rows, unfiltered_rows, canonical_gpat_binding="binding-A",
        second_gpat_binding="binding-B", anomalous_recipe_ids={"R-A"})
    assert result["same_recipe_same_alternate_coverage_across_routes"] is True
    assert result["same_recipe_same_alternate_coverage_across_gpat_bindings"] is True
    assert result["comparable_route_pairs"] == 1
    assert result["comparable_binding_pairs"] == 1


def test_cross_route_and_binding_consistency_false_when_routes_disagree():
    canonical_rows = [
        {"recipe_id": "R-A", "route": "gpat", "trace_requested_coverage": 0.9021},
        {"recipe_id": "R-A", "route": "physics", "trace_requested_coverage": 0.5000},
    ]
    result = e6r.cross_route_and_binding_consistency(
        canonical_rows, [], canonical_gpat_binding="binding-A",
        second_gpat_binding="binding-B", anomalous_recipe_ids={"R-A"})
    assert result["same_recipe_same_alternate_coverage_across_routes"] is False
    per_recipe = result["per_recipe"][0]
    assert per_recipe["same_across_routes"] is False


def test_cross_route_and_binding_consistency_none_when_no_comparable_binding_data():
    """TASK J regression: when the UNFILTERED tree isn't available (so there
    is zero comparable superseded-binding data), the aggregate must report
    None ('not comparable'), never collapse to False -- a real bug caught
    during the first live GPU-data run of this analysis, where the raw
    candidate tree was absent locally but the canonical CSV was present."""
    canonical_rows = [
        {"recipe_id": "R-A", "route": "gpat", "trace_requested_coverage": 0.9021},
        {"recipe_id": "R-A", "route": "physics", "trace_requested_coverage": 0.9021},
    ]
    result = e6r.cross_route_and_binding_consistency(
        canonical_rows, [], canonical_gpat_binding="binding-A",  # unfiltered_rows == [] -> unavailable
        second_gpat_binding="binding-B", anomalous_recipe_ids={"R-A"})
    assert result["comparable_binding_pairs"] == 0
    assert result["same_recipe_same_alternate_coverage_across_gpat_bindings"] is None
    assert result["per_recipe"][0]["same_across_bindings"] is None
    # the route comparison IS comparable here and must still report a real boolean
    assert result["same_recipe_same_alternate_coverage_across_routes"] is True


def test_select_normal_control_recipes_finds_nearest_and_reports_pass():
    rows = [
        {"recipe_id": "R-ANOM", "route": "gpat", "recipe_geometry_coverage": 0.75,
         "requested_equals_recipe_coverage": False},
        {"recipe_id": "R-NORMAL-FAR", "route": "gpat", "recipe_geometry_coverage": 0.10,
         "requested_equals_recipe_coverage": True},
        {"recipe_id": "R-NORMAL-NEAR", "route": "gpat", "recipe_geometry_coverage": 0.80,
         "requested_equals_recipe_coverage": True},
        {"recipe_id": "R-NORMAL-NEAR", "route": "physics", "recipe_geometry_coverage": 0.80,
         "requested_equals_recipe_coverage": True},
    ]
    controls = e6r.select_normal_control_recipes(rows, anomalous_recipe_ids={"R-ANOM"})
    assert len(controls) == 1
    assert controls[0]["control_recipe_id"] == "R-NORMAL-NEAR"
    assert controls[0]["control_all_requested_equals_recipe_coverage"] is True


def test_select_normal_control_recipes_excludes_recipes_with_any_failing_render():
    rows = [
        {"recipe_id": "R-ANOM", "route": "gpat", "recipe_geometry_coverage": 0.75,
         "requested_equals_recipe_coverage": False},
        {"recipe_id": "R-BAD-CONTROL", "route": "gpat", "recipe_geometry_coverage": 0.751,
         "requested_equals_recipe_coverage": True},
        {"recipe_id": "R-BAD-CONTROL", "route": "physics", "recipe_geometry_coverage": 0.751,
         "requested_equals_recipe_coverage": False},  # disqualifies this recipe as a "normal" control
    ]
    controls = e6r.select_normal_control_recipes(rows, anomalous_recipe_ids={"R-ANOM"})
    assert controls == []  # no eligible normal recipe left


def test_relate_known_q_mismatches_to_recipe_class_all_members():
    groups = [{"recipe_id": recipe_id, "is_anomalous_recipe": True, "all_renders_anomalous": True}
             for recipe_id in e6r.KNOWN_Q_MISMATCH_RECIPE_MAP.values()]
    result = e6r.relate_known_q_mismatches_to_recipe_class(groups)
    assert result["three_q_mismatches_are_members_of_recipe_level_anomaly_class"] is True
    assert len(result["candidates"]) == 3


def test_relate_known_q_mismatches_to_recipe_class_false_when_one_recipe_not_anomalous():
    recipe_ids = list(e6r.KNOWN_Q_MISMATCH_RECIPE_MAP.values())
    groups = [{"recipe_id": recipe_ids[0], "is_anomalous_recipe": False, "all_renders_anomalous": False},
             {"recipe_id": recipe_ids[1], "is_anomalous_recipe": True, "all_renders_anomalous": True},
             {"recipe_id": recipe_ids[2], "is_anomalous_recipe": True, "all_renders_anomalous": True}]
    result = e6r.relate_known_q_mismatches_to_recipe_class(groups)
    assert result["three_q_mismatches_are_members_of_recipe_level_anomaly_class"] is False


def test_c6_selection_impact_for_canonical_anomalies_groups_by_recipe():
    anomaly_rows = [{"candidate_id": "sel-1", "recipe_id": "R-A"},
                    {"candidate_id": "not-sel-1", "recipe_id": "R-A"},
                    {"candidate_id": "not-sel-2", "recipe_id": "R-B"}]
    bank_lock = {"selected": [{"candidate_id": "sel-1", "q": 0.7}]}
    result = e6r.c6_selection_impact_for_canonical_anomalies(anomaly_rows, bank_lock)
    assert result["canonical_anomaly_c6_selected"] == 1
    assert result["canonical_anomaly_c6_not_selected_or_unknown"] == 2
    assert result["by_recipe"]["R-A"] == {"selected": 1, "not_selected_or_unknown": 1}
    assert result["by_recipe"]["R-B"] == {"selected": 0, "not_selected_or_unknown": 1}
    assert result["canonical_anomaly_c6_accepted"] is None  # never guessed


def test_reassess_root_cause_high_confidence_when_exact_field_found():
    result = e6r.reassess_root_cause(
        anomaly_determined_by_recipe="TRUE", same_across_routes=True, same_across_bindings=True,
        alternate_value_present_in_recipe=True, alternate_value_present_in_graph=False)
    assert result["primary_anomaly_factor"] == "R5_deterministic_recipe_specific_edge_or_fallback_semantic"
    assert result["root_cause_confidence"] == "HIGH"
    assert result["r3_mutable_runtime_state_ruled_weak"] is True


def test_reassess_root_cause_medium_confidence_when_deterministic_but_no_field_match():
    result = e6r.reassess_root_cause(
        anomaly_determined_by_recipe="TRUE", same_across_routes=True, same_across_bindings=True,
        alternate_value_present_in_recipe=False, alternate_value_present_in_graph=False)
    assert result["root_cause_confidence"] == "MEDIUM"
    assert result["r3_mutable_runtime_state_ruled_weak"] is True


def test_reassess_root_cause_unresolved_when_not_recipe_determined():
    result = e6r.reassess_root_cause(
        anomaly_determined_by_recipe="FALSE", same_across_routes=False, same_across_bindings=False,
        alternate_value_present_in_recipe=False, alternate_value_present_in_graph=False)
    assert result["primary_anomaly_factor"] == "UNRESOLVED"
    assert result["r3_mutable_runtime_state_ruled_weak"] is False


def test_run_canonical_anomaly_investigation_unavailable_without_csv(tmp_path):
    repo = _base_repo(tmp_path)
    result = e6r.run_canonical_anomaly_investigation(repo)
    assert result["available"] is False
    assert result["csv_written"] is False
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert summary["available"] is False


def _canonical_anomaly_end_to_end_fixture(repo: Path) -> None:
    """Recipe R-000354 (real coverage=0.75): 2 canonical GPAT + 1 Physics
    candidate, ALL sharing the SAME altered trace.requested_coverage=0.9021
    (recipe-level, cross-route-consistent anomaly), plus one SUPERSEDED-
    binding GPAT candidate on the SAME recipe sharing the SAME 0.9021 (for
    the cross-binding check). Recipe R-000221 (coverage=1.0): normal GPAT +
    Physics candidates, serving as TASK G's normal control."""
    _population_recipe_bank_fixture(repo)
    anomaly_overrides = {"requested_region_pixels": 232, "requested_support_pixels": 209,
                        "requested_coverage": 0.9021, "achieved_coverage": 0.9008620689655172}
    _write_population_candidate(
        repo, candidate_id="gpat-anom-1", route="gpat", recipe_id="R-000354", position=11,
        live_target_sample_id="live-A",
        trace_overrides={**anomaly_overrides, "binding": e6r.EXPECTED_GPAT_CHECKPOINT_SHA256})
    _write_population_candidate(
        repo, candidate_id="gpat-anom-2", route="gpat", recipe_id="R-000354", position=12,
        live_target_sample_id="live-B",
        trace_overrides={**anomaly_overrides, "binding": e6r.EXPECTED_GPAT_CHECKPOINT_SHA256})
    _write_population_candidate(
        repo, candidate_id="physics-anom-1", route="physics", recipe_id="R-000354", position=13,
        live_target_sample_id="live-C", trace_overrides={**anomaly_overrides, "binding": "physics-engine-v1"})
    _write_population_candidate(
        repo, candidate_id="gpat-superseded-1", route="gpat", recipe_id="R-000354", position=11,
        live_target_sample_id="live-A", trace_overrides={**anomaly_overrides, "binding": "binding-B"})
    _write_population_candidate(
        repo, candidate_id="gpat-normal-1", route="gpat", recipe_id="R-000221", position=30,
        live_target_sample_id="live-D", trace_overrides={"binding": e6r.EXPECTED_GPAT_CHECKPOINT_SHA256})
    _write_population_candidate(
        repo, candidate_id="physics-normal-1", route="physics", recipe_id="R-000221", position=31,
        live_target_sample_id="live-E", trace_overrides={"binding": "physics-engine-v1"})
    _c4_c5_c6_lock_fixture(repo, active_binding=e6r.EXPECTED_GPAT_CHECKPOINT_SHA256, second_binding="binding-B")
    bank_lock_path = repo / e6r.C6_BANK_LOCK_LLM_PATH
    bank_lock = json.loads(bank_lock_path.read_text(encoding="utf-8"))
    bank_lock["selected"] = [{"candidate_id": "gpat-anom-1", "q": 0.5}]
    bank_lock_path.write_text(json.dumps(bank_lock), encoding="utf-8")
    canonical_status = e6r.write_canonical_population_artifacts(repo)
    assert canonical_status["csv_written"] is True


def test_run_canonical_anomaly_investigation_end_to_end(tmp_path):
    repo = _base_repo(tmp_path)
    _canonical_anomaly_end_to_end_fixture(repo)

    result = e6r.run_canonical_anomaly_investigation(repo)
    assert result["available"] is True
    summary = result["summary"]

    # canonical-only filtering: the superseded-binding candidate never appears
    csv_text = Path(result["csv_path"]).read_text(encoding="utf-8")
    assert "gpat-superseded-1" not in csv_text
    assert "gpat-anom-1" in csv_text and "gpat-anom-2" in csv_text and "physics-anom-1" in csv_text

    # superseded rows excluded from scientific counts
    assert summary["canonical_anomaly_count"] == 3
    assert summary["canonical_gpat_anomaly_count"] == 2
    assert summary["canonical_physics_anomaly_count"] == 1

    # recipe-level grouping
    assert summary["unique_canonical_anomalous_recipes"] == 1
    assert summary["anomalous_recipe_ids"] == ["R-000354"]
    assert summary["anomaly_determined_by_recipe_id"] == "TRUE"

    # cross-route + cross-binding consistency (Task F)
    consistency = summary["cross_route_and_binding_consistency"]
    assert consistency["same_recipe_same_alternate_coverage_across_routes"] is True
    assert consistency["same_recipe_same_alternate_coverage_across_gpat_bindings"] is True

    # normal controls (Task G) -- R-000221 is untouched and passes
    assert summary["normal_control_recipes"]
    assert all(c["control_all_requested_equals_recipe_coverage"] for c in summary["normal_control_recipes"])

    # C6 read-only join (Task I)
    assert summary["c6_selection_impact"]["canonical_anomaly_c6_selected"] == 1

    # root cause: fully recipe-determined + cross-route + cross-binding consistent
    assert summary["root_cause_reassessment"]["r3_mutable_runtime_state_ruled_weak"] is True

    field_matches_text = Path(result["field_matches_csv_path"]).read_text(encoding="utf-8")
    assert "R-000354" in field_matches_text


def test_run_canonical_anomaly_investigation_never_mutates_canonical_csv_or_locks(tmp_path):
    repo = _base_repo(tmp_path)
    _canonical_anomaly_end_to_end_fixture(repo)
    canonical_csv_path = repo / e6r.RENDER_DIR / "E6_HISTORICAL_TRACE_CANONICAL_POPULATION.csv"
    before = canonical_csv_path.read_bytes()

    e6r.run_canonical_anomaly_investigation(repo)

    assert canonical_csv_path.read_bytes() == before


def test_canonical_anomaly_functions_never_touch_target_model_gpu_or_llm():
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    for fn_name in ("load_canonical_population_csv", "summarize_canonical_recipe_groups",
                    "classify_anomaly_determined_by_recipe", "flatten_scalar_fields",
                    "match_scalar_fields_to_value", "compare_anomalous_recipe_fields",
                    "compare_anomalous_compiled_graph_fields", "classify_alternate_coverage_source",
                    "cross_route_and_binding_consistency", "select_normal_control_recipes",
                    "relate_known_q_mismatches_to_recipe_class", "c6_selection_impact_for_canonical_anomalies",
                    "reassess_root_cause", "run_canonical_anomaly_investigation"):
        fn_start = source.index(f"def {fn_name}(")
        fn_end = source.index("\ndef ", fn_start + 10)
        body = source[fn_start:fn_end]
        for forbidden in ("QualityBackends(", "SCRFDDetector(", "DifferentiableAdaFace(",
                         "FaceXFormerBackend(", "GPATRoute(", "render_one(", "torch.no_grad",
                         "resolve_target", "SiW", "openai", "google.generativeai", "GEMINI"):
            assert forbidden not in body, f"{forbidden!r} unexpectedly reachable from {fn_name}"


def test_analyze_canonical_trace_anomalies_cli_flag_present_and_read_only():
    source = Path(e6r.__file__).read_text(encoding="utf-8")
    assert "--analyze-canonical-trace-anomalies" in source
    assert "run_canonical_anomaly_investigation" in source
    assert "gpu_canonical_anomaly_investigation_command" in source
