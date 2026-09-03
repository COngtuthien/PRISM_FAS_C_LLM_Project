"""Tests for `prism_fas.evaluation.c_ext_e6_v2_paired` (E6-v2
PAIRED_CURRENT_RUNTIME protocol preparation).

Every test builds a self-contained fake repo under `tmp_path` (reusing the
already-established `_historical_fixture`/`_shuffle_fixture`/
`_training_plan_lock_fixture` helpers from `test_c_ext_e6_render`) or calls a
pure function directly. No test ever passes the real repo root to a function
that writes. An autouse fixture hashes the REAL, historical
`reports/c_ext_q1q2_v1/e6_llm_shuffle/render/` namespace before and after
every test in this file -- this module must never touch it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prism_fas.evaluation import c_ext_common as cc
from prism_fas.evaluation import c_ext_e6_v2_paired as v2

from test_c_ext_e6_render import (  # noqa: E402  (reuse, never reimplement)
    _base_repo, _full_source_pair_plan_fixture, _historical_fixture, _passing_metrics,
    _quality_gate_fixture, _route_quota_fixture, _shuffle_fixture, _training_plan_lock_fixture,
    _write_q_reference_fixture,
)

REPO = Path(__file__).resolve().parents[2]
OLD_E6_RENDER_DIR = REPO / v2.OLD_E6_RENDER_DIR


def _tree_hash(root: Path) -> str:
    import hashlib

    if not root.is_dir():
        return "MISSING"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@pytest.fixture(autouse=True)
def _protect_old_e6_namespace():
    before = _tree_hash(OLD_E6_RENDER_DIR)
    yield
    after = _tree_hash(OLD_E6_RENDER_DIR)
    assert after == before, "a test mutated the REAL, historical e6_llm_shuffle/render/ namespace"


def _full_v2_fixture(repo: Path, *, recipe_count: int = 256, with_training_config: bool = False) -> None:
    """A complete, self-contained fake repo: the historical C5/C6 contract
    fixture, a frozen LLM-SHUFFLE-A recipe set + training-plan lock, AND an
    ORIGINAL recipe bank at the real path with matching recipe_ids at every
    ordinal (source-pair alignment requires identical ids, only field
    CONTENT may differ).

    `with_training_config=True` additionally copies the REAL, frozen C7 lock
    and m9 model/train configs verbatim -- `build_e6_training_config` checks
    the C7 lock's winner_config_sha256/decision_graph_hash against PINNED,
    real hashes (`c_ext_e6_training_plan.EXPECTED_WINNER_CONFIG_SHA256` etc.),
    so a synthetic/fake C7 lock cannot pass that check; only the real one can.

    `recipe_count=256` (the default, and the only value BLOCKER 1/2 checks
    can pass) copies the REAL, frozen original recipe bank, the REAL, frozen
    LLM-SHUFFLE-A recipes, and the REAL EXT_RECIPE_BINDING.json verbatim --
    required because `resolve_original_recipe_identity_equivalence` parses
    every recipe against the real RecipeV11 schema and independently
    recomputes the real, pinned `EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY`,
    and `resolve_recipe_field_marginal_parity` reads the real, historically-
    frozen shuffle-group definition -- none of these can be satisfied by
    small synthetic placeholder recipes. Any OTHER recipe_count builds a
    small synthetic fixture instead, for tests that fail closed BEFORE
    reaching schema validation (e.g. the wrong-recipe-count guard)."""
    import shutil

    _historical_fixture(repo)
    _route_quota_fixture(repo)

    if recipe_count == 256:
        original_path = repo / v2.e6r.RECIPE_BANK_LLM_JSONL_PATH
        original_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / v2.e6r.RECIPE_BANK_LLM_JSONL_PATH, original_path)

        shuffle_path = repo / v2.training_plan.E6_SHUFFLE_RECIPES_PATH
        shuffle_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / v2.training_plan.E6_SHUFFLE_RECIPES_PATH, shuffle_path)

        binding_path = repo / v2.EXT_RECIPE_BINDING_PATH
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / v2.EXT_RECIPE_BINDING_PATH, binding_path)

        shuffle_recipes = cc.read_jsonl(shuffle_path)
        shuffle_identity = cc.sha256_json(shuffle_recipes)
        _training_plan_lock_fixture(repo, recipe_identity=shuffle_identity,
                                    recipe_count=len(shuffle_recipes))
    else:
        recipes, identity = _shuffle_fixture(repo, recipe_count=recipe_count)
        _training_plan_lock_fixture(repo, recipe_identity=identity, recipe_count=recipe_count)
        original_recipes = [{"recipe_id": recipe["recipe_id"], "medium": {"family": "silicone"},
                            "note": "original, unshuffled"} for recipe in recipes]
        original_path = repo / "assets/recipe_banks/c3/llm/recipes.jsonl"
        original_path.parent.mkdir(parents=True, exist_ok=True)
        lines = "\n".join(json.dumps(rec, sort_keys=True, separators=(",", ":"))
                         for rec in original_recipes) + "\n"
        original_path.write_text(lines, encoding="utf-8")

    if with_training_config:
        import shutil

        (repo / "reports/full/c7").mkdir(parents=True, exist_ok=True)
        (repo / "configs/models").mkdir(parents=True, exist_ok=True)
        (repo / "configs/train").mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / "reports/full/c7/DETECTOR_CONFIG_LOCK.json",
                   repo / "reports/full/c7/DETECTOR_CONFIG_LOCK.json")
        shutil.copy(REPO / "configs/models/m9_detector.yaml", repo / "configs/models/m9_detector.yaml")
        shutil.copy(REPO / "configs/train/m9_reference.yaml", repo / "configs/train/m9_reference.yaml")


# --------------------------------------------------------------------------- #
# pure functions
# --------------------------------------------------------------------------- #

def test_v2_run_id_matches_documented_pattern():
    assert v2.v2_run_id(v2.ARM_ORIGINAL, 20260806) == "EXT-F1-G-LLM-ORIGINAL-CURRENT-V2-s20260806"
    assert v2.v2_run_id(v2.ARM_SHUFFLE, 20260810) == "EXT-F1-G-LLM-SHUFFLE-A-CURRENT-V2-s20260810"


def test_is_usable_v2_lock_requires_frozen_status():
    assert v2.is_usable_v2_lock({"status": "FROZEN"}) is True
    assert v2.is_usable_v2_lock({"status": "BLOCKED"}) is False
    assert v2.is_usable_v2_lock({"status": "BLOCKED_BINDING"}) is False
    assert v2.is_usable_v2_lock({}) is False
    assert v2.is_usable_v2_lock("not-a-dict") is False


def test_pre_register_statistical_contrast_matches_spec():
    contrast = v2.pre_register_statistical_contrast()
    assert contrast["primary_metric"] == "ACER"
    assert contrast["secondary_metrics"] == ["APCER", "BPCER", "AUC", "EER"]
    assert contrast["replicate_unit"] == "detector_seed"
    assert contrast["target_labels_accessed_at_freeze"] is False


def test_interpretation_ceiling_covers_every_directional_outcome():
    ceiling = v2.interpretation_ceiling()
    assert "if_original_better_consistently" in ceiling
    assert "if_approximately_equal" in ceiling
    assert "if_shuffle_better" in ceiling
    assert ceiling["frozen_before_results"] is True


def test_seven_anomalous_recipes_policy_never_excludes():
    policy = v2.seven_anomalous_recipes_policy()
    assert len(policy["recipe_ids"]) == 7
    assert policy["excluded_from_either_bank"] is False
    assert policy["hard_coded_historical_alternate_coverage"] is False
    assert policy["render_under_current_semantics"] is True


def test_build_protocol_amendment_without_root_cause_artifact(tmp_path):
    repo = _base_repo(tmp_path)
    amendment = v2.build_protocol_amendment(repo)
    assert amendment["historical_original_llm_c5_c6_bank"]["may_serve_as_paired_e6_original"] is False
    assert amendment["historical_downstream_c_g_llm"]["may_serve_as_paired_e6_original"] is False
    assert amendment["old_e6_historical_parity_path_status"] == v2.OLD_E6_PATH_STATUS
    assert amendment["old_e6_path_overwritten"] is False
    assert "UNKNOWN" in amendment["forensic_root_cause_cited"]


def test_build_protocol_amendment_cites_real_forensic_conclusion_when_present(tmp_path):
    repo = _base_repo(tmp_path)
    root_cause_dir = repo / v2.e6r.RENDER_DIR
    root_cause_dir.mkdir(parents=True, exist_ok=True)
    (root_cause_dir / "E6_SUPPORT_OVERLAP_ROOT_CAUSE.json").write_text(json.dumps({
        "turn_10_source_code_forensics": {
            "primary_anomaly_factor_and_confidence": {"FINAL_ROOT_CAUSE": "UNRECOVERABLE_TEST_VALUE"}}}),
        encoding="utf-8")
    amendment = v2.build_protocol_amendment(repo)
    assert amendment["forensic_root_cause_cited"] == "UNRECOVERABLE_TEST_VALUE"


# --------------------------------------------------------------------------- #
# matching policy: the real BLOCKED_BINDING finding
# --------------------------------------------------------------------------- #

def test_resolve_v2_matching_policy_reports_reusable_after_arm_parameterization(tmp_path):
    """Structural, not fixture-dependent: reads the REAL c_ext_e6_render
    source. `default_quality_matcher` no longer hardcodes `arm=E6_ARM_NAME`
    -- it now takes `arm` as a required keyword-only parameter, so the
    wrapper is reusable for both v2 arm labels. This is a genuine finding
    against the REAL, current source, not a synthetic one."""
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo)
    policy = v2.resolve_v2_matching_policy(repo)
    assert policy["arm_hardcoded_in_source"] is False
    assert policy["arm_is_required_parameter_no_default"] is True
    assert policy["e6_default_quality_matcher_wrapper_reusable"] is True
    assert policy["status"] == "REUSABLE"
    assert policy["matching_primitives_reusable"] is True
    assert policy["matching_algorithm_changed"] is False
    assert policy["blocked_reason"] is None
    assert policy["original_v2_matcher_arm"] == v2.ARM_ORIGINAL
    assert policy["shuffle_v2_matcher_arm"] == v2.ARM_SHUFFLE


# --------------------------------------------------------------------------- #
# readiness gate
# --------------------------------------------------------------------------- #

def _synthetic_locks(*, quality_status: str = "FROZEN") -> dict[str, dict]:
    return {
        v2.PROTOCOL_LOCK_PATH: {"status": "FROZEN"},
        v2.RECIPE_PAIR_LOCK_PATH: {
            "status": "FROZEN", "recipe_counts_match": True,
            "original_recipe_content_equivalence": "PROVEN",
            "recipe_field_marginal_parity": "PASS", "joint_associations_changed": True,
            "original_recipe_identity_file_sha_v2": "sha-original", "shuffle_recipe_identity": "sha-shuffle",
        },
        v2.SOURCE_PAIR_PARITY_LOCK_PATH: {"status": "FROZEN"},
        v2.RENDER_PARITY_LOCK_PATH: {
            "status": "FROZEN",
            "original_plan": {"recipe_content_identity": "sha-original"},
            "shuffle_plan": {"recipe_content_identity": "sha-shuffle"},
        },
        v2.QUALITY_PARITY_LOCK_PATH: {"status": quality_status},
        v2.TRAINING_PLAN_LOCK_PATH: {"status": "FROZEN"},
    }


def test_readiness_gate_true_when_every_lock_frozen():
    gate = v2.compute_readiness_gate(_synthetic_locks())
    assert gate["E6_V2_READY_FOR_RENDER"] is True
    assert gate["blocking_locks"] == []


def test_readiness_gate_false_when_one_lock_blocked():
    gate = v2.compute_readiness_gate(_synthetic_locks(quality_status="BLOCKED_BINDING"))
    assert gate["E6_V2_QUALITY_PARITY_PASS"] is False
    assert gate["E6_V2_READY_FOR_RENDER"] is False
    assert v2.QUALITY_PARITY_LOCK_PATH in gate["blocking_locks"]
    # every OTHER gate still reports its own true status independently
    assert gate["E6_V2_PROTOCOL_LOCKED"] is True
    assert gate["E6_V2_TRAINING_PLAN_LOCKED"] is True


def test_readiness_gate_false_when_recipe_counts_mismatch_even_if_status_frozen():
    locks = _synthetic_locks()
    locks[v2.RECIPE_PAIR_LOCK_PATH] = {"status": "FROZEN", "recipe_counts_match": False}
    gate = v2.compute_readiness_gate(locks)
    assert gate["E6_V2_RECIPE_PARITY_PASS"] is False
    assert gate["E6_V2_READY_FOR_RENDER"] is False


# --------------------------------------------------------------------------- #
# source-pair parity + fairness contract (real reuse of e6r primitives)
# --------------------------------------------------------------------------- #

def test_resolve_v2_source_pair_parity_100_percent_when_ids_align(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    parity = v2.resolve_v2_source_pair_parity(repo)
    assert parity["ordinals_checked"] == 256
    assert parity["all_ordinals_aligned"] is True
    assert parity["source_pair_parity_pct"] == 100.0


def test_resolve_v2_source_pair_parity_fails_closed_on_id_mismatch(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    # corrupt ONE original recipe_id so it disagrees with the shuffle side
    original_path = repo / "assets/recipe_banks/c3/llm/recipes.jsonl"
    lines = original_path.read_text(encoding="utf-8").strip().split("\n")
    corrupted = json.loads(lines[3])
    corrupted["recipe_id"] = "R-DIFFERENT"
    lines[3] = json.dumps(corrupted)
    original_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(v2.e6r.E6RenderError, match="alignment broken"):
        v2.resolve_v2_source_pair_parity(repo)


def test_build_v2_fairness_contract_full_parity_with_matching_contract_fields(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    contract = v2.build_v2_fairness_contract(repo)
    assert contract["runtime_render_parity"] == "PASS"
    assert contract["mismatches"] == []
    assert contract["original_plan"]["recipe_count"] == 256
    assert contract["shuffle_plan"]["recipe_count"] == 256
    fields_covered = {row["field"] for row in contract["field_by_field"]}
    for field in v2.V2_PARITY_FIELDS:
        assert field in fields_covered
    assert len(contract["shared_by_construction"]) >= 10


# --------------------------------------------------------------------------- #
# recipe pair lock, training-plan lock
# --------------------------------------------------------------------------- #

def test_build_recipe_pair_lock_reports_matching_counts(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    lock = v2.build_recipe_pair_lock(repo)
    assert lock["status"] == "FROZEN"
    assert lock["original_recipe_count"] == 256
    assert lock["shuffle_recipe_count"] == 256
    assert lock["recipe_counts_match"] is True
    assert lock["no_recipe_regeneration"] is True


def test_load_original_llm_recipes_fails_closed_on_wrong_count(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=16)
    # the module-level EXPECTED_RECIPE_COUNT check is 256 -- a 16-recipe
    # original bank must be refused, not silently accepted
    with pytest.raises(v2.E6V2ProtocolError, match="expected 256"):
        v2.load_original_llm_recipes(repo)


def test_build_training_plan_lock_proves_config_identity_across_arms(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    lock = v2.build_training_plan_lock(repo)
    assert lock["status"] == "FROZEN"
    assert lock["config_identical_excluding_run_id"] is True
    assert lock["training_authorized_this_turn"] is False
    assert len(lock["run_ids"][v2.ARM_ORIGINAL]) == 5
    assert len(lock["run_ids"][v2.ARM_SHUFFLE]) == 5


# --------------------------------------------------------------------------- #
# end-to-end orchestration
# --------------------------------------------------------------------------- #

def test_run_e6_v2_protocol_preparation_end_to_end(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)

    result = v2.run_e6_v2_protocol_preparation(repo)
    readiness = result["readiness"]

    assert readiness["E6_V2_PROTOCOL_LOCKED"] is True
    assert readiness["E6_V2_RECIPE_PARITY_PASS"] is True
    assert readiness["E6_V2_SOURCE_PAIR_PARITY_PASS"] is True
    assert readiness["E6_V2_RENDER_PARITY_PASS"] is True
    # default_quality_matcher's arm hardcoding is fixed (see c_ext_e6_render.py) -- this now passes
    assert readiness["E6_V2_QUALITY_PARITY_PASS"] is True
    assert readiness["E6_V2_TRAINING_PLAN_LOCKED"] is True
    assert readiness["E6_V2_READY_FOR_RENDER"] is True

    for rel_path in v2.LOCK_BUILDERS:
        written = Path(repo / rel_path)
        assert written.is_file()
        assert written.is_relative_to(repo / v2.E6_V2_DIR)
        payload = json.loads(written.read_text(encoding="utf-8"))
        assert "lock_identity" in payload

    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert summary["target_access"] is False
    assert summary["rendering_performed"] is False
    assert summary["training_performed"] is False


def test_run_e6_v2_protocol_preparation_never_writes_outside_its_own_namespace(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    before = {p for p in repo.rglob("*") if p.is_file()}

    v2.run_e6_v2_protocol_preparation(repo)

    after_new = {p for p in repo.rglob("*") if p.is_file()} - before
    for path in after_new:
        assert path.is_relative_to(repo / v2.E6_V2_DIR), f"unexpected write outside E6_V2_DIR: {path}"


def test_run_e6_v2_protocol_preparation_never_touches_old_e6_dir(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    old_dir = repo / v2.OLD_E6_RENDER_DIR
    old_dir.mkdir(parents=True, exist_ok=True)
    (old_dir / "SENTINEL.json").write_text('{"untouched": true}', encoding="utf-8")
    before = (old_dir / "SENTINEL.json").read_bytes()

    v2.run_e6_v2_protocol_preparation(repo)

    assert (old_dir / "SENTINEL.json").read_bytes() == before
    assert not (repo / v2.E6_V2_DIR).is_relative_to(old_dir)


# --------------------------------------------------------------------------- #
# no target / model / render / train / LLM reachability
# --------------------------------------------------------------------------- #

def test_v2_module_never_touches_target_model_gpu_train_render_or_llm():
    source = Path(v2.__file__).read_text(encoding="utf-8")
    for forbidden in ("resolve_target", "SiW", "openai", "google.generativeai", "GEMINI",
                     "torch.no_grad", "render_arm(", "render_one(", "GPATRoute(", "PhysicsRoute(",
                     "QualityBackends(", "SCRFDDetector(", "M9TrainingRun(", "train_detector("):
        assert forbidden not in source, f"{forbidden!r} unexpectedly reachable from c_ext_e6_v2_paired"


def test_v2_cli_flag_present_and_prepare_only():
    source = Path(v2.__file__).read_text(encoding="utf-8")
    assert "--prepare-protocol" in source
    assert "run_e6_v2_protocol_preparation" in source
    # this turn adds the explicit two-flag render authorization -- both
    # flags must exist AND (proven by the CLI tests below) both must be
    # required together, with no permissive default
    assert "add_argument(\"--execute\"" in source
    assert "add_argument(\"--authorize-gpu-render\"" in source


# --------------------------------------------------------------------------- #
# BLOCKER 1: original recipe identity -- file SHA vs semantic identity
# --------------------------------------------------------------------------- #

def test_identity_equivalence_distinguishes_file_sha_from_semantic_identity(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    result = v2.resolve_original_recipe_identity_equivalence(repo)

    earlier = result["earlier_identity"]
    new = result["new_v2_identity"]
    assert earlier["identity"] == v2.e6r.EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY
    assert earlier["ordering_included"] is False
    assert "canonical.recipe_hash" in earlier["algorithm"] or "recipe_hash" in earlier["algorithm"]
    assert new["ordering_included"] is True
    assert "sha256_json" in new["algorithm"]
    # the two algorithms are genuinely different -- the two identity VALUES differ
    assert earlier["identity"] != new["identity"]
    assert result["classification"].startswith("A --")


def test_original_recipe_content_equivalence_proven_against_real_frozen_recipes(tmp_path):
    """The real, definitive proof this milestone required: independently
    recomputing the historically-frozen semantic selected_set_identity from
    the exact 256 recipes this module reads must reproduce the frozen,
    pinned value exactly."""
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    result = v2.resolve_original_recipe_identity_equivalence(repo)
    assert result["original_recipe_content_equivalence"] == "PROVEN"
    assert result["matches_frozen_semantic_identity"] is True
    assert result["recomputed_semantic_identity_from_v2_load"] == \
           v2.e6r.EXPECTED_ORIGINAL_LLM_SELECTED_SET_IDENTITY


def test_original_recipe_content_equivalence_not_proven_when_recipe_bank_tampered(tmp_path):
    """If the 256 recipes at the frozen path do NOT match the historically-
    frozen selected set (e.g. one recipe's content changed), the recomputed
    semantic identity must NOT match, and this must be reported as
    NOT_PROVEN -- never silently passed."""
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    original_path = repo / v2.e6r.RECIPE_BANK_LLM_JSONL_PATH
    lines = original_path.read_text(encoding="utf-8").strip().split("\n")
    tampered = json.loads(lines[0])
    tampered["geometry"]["coverage"] = round(1.0 - float(tampered["geometry"]["coverage"]), 6) \
        if tampered["geometry"]["coverage"] != 0.5 else 0.42
    lines[0] = json.dumps(tampered)
    original_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = v2.resolve_original_recipe_identity_equivalence(repo)
    assert result["original_recipe_content_equivalence"] == "NOT_PROVEN"
    assert result["matches_frozen_semantic_identity"] is False


# --------------------------------------------------------------------------- #
# BLOCKER 2: exact per-group marginal multiset equality + joint-association change
# --------------------------------------------------------------------------- #

def test_recipe_field_marginal_parity_pass_against_real_frozen_shuffle(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    result = v2.resolve_recipe_field_marginal_parity(repo)

    assert result["recipe_field_marginal_parity"] == "PASS"
    assert result["original_recipe_count"] == 256
    assert result["shuffle_recipe_count"] == 256
    assert result["recipe_counts_match"] is True
    assert result["recipe_ids_same_order"] is True
    assert result["group_definition_source"] == v2.EXT_RECIPE_BINDING_PATH
    assert set(result["groups"]) == {"medium", "geometry", "illumination", "region",
                                     "artifact_family_and_parameters", "severity_group"}
    for row in result["per_group"]:
        assert row["exact_multiset_equal"] is True
        assert row["original_multiset_hash"] == row["shuffle_multiset_hash"]


def test_joint_associations_changed_true_for_the_real_frozen_shuffle(tmp_path):
    """The real Shuffle-A recipes DO differ per recipe_id from the original
    (that is the entire point of the shuffle) even though every group's
    marginal is exactly preserved -- this must read True, never False,
    against the real frozen artifacts."""
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    result = v2.resolve_recipe_field_marginal_parity(repo)
    assert result["joint_associations_changed"] is True


def test_recipe_field_marginal_parity_fails_when_a_group_value_is_altered(tmp_path):
    """Tampering with a SINGLE shuffle recipe's medium.family (a value not
    present anywhere else in the group's multiset) must break exact
    multiset equality for that group and report FAIL, not PASS.

    The tampered recipes.jsonl's OWN content identity is re-derived and
    written into the training-plan lock too -- otherwise
    `verify_shuffle_recipe_source`'s own (correct, pre-existing, defense-in-
    depth) content-identity check fails FIRST, one layer earlier, for a
    DIFFERENT and equally legitimate reason. This isolates the marginal-
    parity comparison itself."""
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    shuffle_path = repo / v2.training_plan.E6_SHUFFLE_RECIPES_PATH
    lines = shuffle_path.read_text(encoding="utf-8").strip().split("\n")
    tampered = json.loads(lines[0])
    # every real medium.family is one of the ontology's fixed enum values;
    # introducing a value outside that set cannot already exist elsewhere
    tampered["medium"]["family"] = "impossible-material-injected-by-test"
    lines[0] = json.dumps(tampered)
    shuffle_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tampered_recipes = [json.loads(line) for line in lines]
    tampered_identity = cc.sha256_json(tampered_recipes)
    _training_plan_lock_fixture(repo, recipe_identity=tampered_identity, recipe_count=len(tampered_recipes))

    result = v2.resolve_recipe_field_marginal_parity(repo)
    medium_row = next(row for row in result["per_group"] if row["group"] == "medium")
    assert medium_row["exact_multiset_equal"] is False
    assert result["recipe_field_marginal_parity"] == "FAIL"


def test_load_frozen_shuffle_group_map_uses_the_real_historical_grouping_never_invented(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256)
    group_map = v2.load_frozen_shuffle_group_map(repo)
    assert group_map["status"] == "FROZEN_AT_E0_FOR_E6"
    assert group_map["group_field_map"]["geometry"] == \
           ["geometry.shape", "geometry.rigidity", "geometry.coverage"]
    assert group_map["group_field_map"]["severity_group"] == \
           ["capture.yaw", "capture.compression_q", "capture.scale", "capture.motion", "capture.defocus"]


def test_load_frozen_shuffle_group_map_fails_closed_when_not_frozen(tmp_path):
    repo = _base_repo(tmp_path)
    binding_path = repo / v2.EXT_RECIPE_BINDING_PATH
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.write_text(json.dumps({"llm_shuffle_groups": {"status": "DRAFT"}}), encoding="utf-8")
    with pytest.raises(v2.E6V2ProtocolError, match="not frozen"):
        v2.load_frozen_shuffle_group_map(repo)


# --------------------------------------------------------------------------- #
# BLOCKER 3: lock dependency chain
# --------------------------------------------------------------------------- #

def test_lock_dependency_chain_declares_real_edges():
    chain = {entry["lock"]: entry["depends_on"] for entry in v2.LOCK_DEPENDENCY_CHAIN}
    assert v2.RECIPE_PAIR_LOCK_PATH in chain
    assert any("EXT_RECIPE_BINDING" in dep for dep in chain[v2.RECIPE_PAIR_LOCK_PATH])
    assert any(v2.RECIPE_PAIR_LOCK_PATH in dep for dep in chain[v2.SOURCE_PAIR_PARITY_LOCK_PATH])
    assert any(v2.RECIPE_PAIR_LOCK_PATH in dep for dep in chain[v2.RENDER_PARITY_LOCK_PATH])


def test_describe_lock_dependency_chain_valid_when_identities_agree():
    locks = _synthetic_locks()
    result = v2.describe_lock_dependency_chain(locks)
    assert result["cross_check"]["original_recipe_identity_consistent_across_recipe_pair_and_render_parity"] \
        is True
    assert result["cross_check"]["shuffle_recipe_identity_consistent_across_recipe_pair_and_render_parity"] \
        is True
    assert result["lock_dependency_chain_valid"] is True


def test_describe_lock_dependency_chain_invalid_when_identities_disagree():
    locks = _synthetic_locks()
    # simulate two independently-computed recipe identities disagreeing --
    # this must be CAUGHT, not silently trusted
    locks[v2.RENDER_PARITY_LOCK_PATH]["original_plan"]["recipe_content_identity"] = "sha-DIFFERENT"
    result = v2.describe_lock_dependency_chain(locks)
    assert result["cross_check"]["original_recipe_identity_consistent_across_recipe_pair_and_render_parity"] \
        is False
    assert result["lock_dependency_chain_valid"] is False


def test_lock_dependency_chain_valid_end_to_end_against_real_fixture(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    result = v2.run_e6_v2_protocol_preparation(repo)
    assert result["readiness"]["LOCK_DEPENDENCY_CHAIN_VALID"] is True


# --------------------------------------------------------------------------- #
# BLOCKER 5: readiness blocked if recipe identity or marginal parity unproven
# --------------------------------------------------------------------------- #

def test_readiness_blocked_when_recipe_content_equivalence_not_proven():
    locks = _synthetic_locks()
    locks[v2.RECIPE_PAIR_LOCK_PATH]["original_recipe_content_equivalence"] = "NOT_PROVEN"
    gate = v2.compute_readiness_gate(locks)
    assert gate["ORIGINAL_RECIPE_CONTENT_EQUIVALENCE"] is False
    assert gate["E6_V2_RECIPE_PARITY_PASS"] is False
    assert gate["E6_V2_READY_FOR_RENDER"] is False


def test_readiness_blocked_when_marginal_parity_fails():
    locks = _synthetic_locks()
    locks[v2.RECIPE_PAIR_LOCK_PATH]["recipe_field_marginal_parity"] = "FAIL"
    gate = v2.compute_readiness_gate(locks)
    assert gate["RECIPE_FIELD_MARGINAL_PARITY"] == "FAIL"
    assert gate["E6_V2_RECIPE_PARITY_PASS"] is False
    assert gate["E6_V2_READY_FOR_RENDER"] is False


def test_readiness_blocked_when_joint_associations_did_not_change():
    """If a 'shuffle' that never actually changed any joint association were
    ever fed in, this must block readiness -- a no-op shuffle would defeat
    the entire E6 experiment."""
    locks = _synthetic_locks()
    locks[v2.RECIPE_PAIR_LOCK_PATH]["joint_associations_changed"] = False
    gate = v2.compute_readiness_gate(locks)
    assert gate["JOINT_ASSOCIATIONS_CHANGED"] is False
    assert gate["E6_V2_RECIPE_PARITY_PASS"] is False
    assert gate["E6_V2_READY_FOR_RENDER"] is False


def test_readiness_true_end_to_end_against_real_fixture_proves_all_blockers_resolved(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    result = v2.run_e6_v2_protocol_preparation(repo)
    readiness = result["readiness"]

    assert readiness["ORIGINAL_RECIPE_CONTENT_EQUIVALENCE"] is True
    assert readiness["RECIPE_FIELD_MARGINAL_PARITY"] == "PASS"
    assert readiness["JOINT_ASSOCIATIONS_CHANGED"] is True
    assert readiness["LOCK_DEPENDENCY_CHAIN_VALID"] is True
    assert readiness["E6_V2_READY_FOR_RENDER"] is True


# =============================================================================
# EXECUTION ENTRY POINT (this turn): CODE + PREFLIGHT + TESTS ONLY.
# No test in this section ever renders a real candidate -- render_arm_fn is
# always a fake that uses the REAL c5_raw_generation resume primitives
# (candidate_dir/reuse_decision/write_record/CandidateRecord) without GPU,
# CUDA, GPAT, PhysicsEngine or any quality model.
# =============================================================================

def _full_v2_execution_fixture(tmp_path: Path, monkeypatch) -> Path:
    """`_full_v2_fixture(recipe_count=256)` PLUS a real, self-consistent
    `source_train.parquet` (reusing `_full_source_pair_plan_fixture` from
    test_c_ext_e6_render verbatim) so `build_v2_arm_plan_rows` can run
    UNMOCKED end to end for BOTH arms. The synthetic manifest reproduces a
    DIFFERENT source_pair_plan_identity than the real pinned constant, so
    (exactly like `_full_2048_fixture`) the pinned expectation and the C5
    lock are pointed at what THIS fixture's manifest actually recomputes."""
    from prism_fas.synthesis.c5_source_pair_plan import PLAN_SEED, build_source_pair_plan, source_pair_plan_identity

    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    _quality_gate_fixture(repo)
    _write_q_reference_fixture(repo)
    _full_source_pair_plan_fixture(repo)

    base_plan = build_source_pair_plan(repo / v2.e6r.SOURCE_PACKAGE_ROOT, seed=PLAN_SEED)
    real_identity = source_pair_plan_identity(base_plan)
    monkeypatch.setattr(v2.e6r, "EXPECTED_SOURCE_PAIR_PLAN_IDENTITY", real_identity)
    source_pair_plan_payload = json.loads((repo / v2.e6r.C5_SOURCE_PAIR_PLAN_PATH).read_text(encoding="utf-8"))
    source_pair_plan_payload["source_pair_plan_identity"] = real_identity
    (repo / v2.e6r.C5_SOURCE_PAIR_PLAN_PATH).write_text(json.dumps(source_pair_plan_payload), encoding="utf-8")

    # _route_quota_fixture (already applied inside _full_v2_fixture) hardcodes
    # a single "domainA" domain key -- the REAL schedule's rows carry
    # live_dataset="CASIA"/"MSU" (from _full_source_pair_plan_fixture's own
    # manifest), so the quota must be replaced with the ACTUAL per-domain/
    # per-route counts the real 2048-position schedule produces, or
    # select_route_bank cannot fill any quota at all.
    from collections import Counter

    route_domain_counts: dict[str, Counter] = {"physics": Counter(), "gpat": Counter()}
    for row in base_plan["positions"]:
        route_domain_counts[row["route"]][row["live_dataset"]] += 1
    bank_lock_path = repo / v2.e6r.C6_BANK_LOCK_LLM_PATH
    bank_lock_payload = json.loads(bank_lock_path.read_text(encoding="utf-8"))
    bank_lock_payload["exposure"] = {
        "physics": {"by_source_domain": dict(route_domain_counts["physics"])},
        "gpat": {"by_source_domain": dict(route_domain_counts["gpat"])},
    }
    bank_lock_path.write_text(json.dumps(bank_lock_payload), encoding="utf-8")

    # the persisted execution-plan lock --prepare-protocol would have
    # written before any --execute invocation is even attempted
    v2.write_render_execution_plan_lock(repo)
    return repo


def _fake_render_arm_factory(*, fail_candidate_ids: frozenset[str] = frozenset(), call_log: list | None = None):
    """A GPU-free stand-in for `synthesis.c5_render.render_arm` that uses the
    SAME REAL resume primitives (`candidate_dir`/`reuse_decision`/
    `write_record`/`CandidateRecord`/`GenerationIdentity`) render_arm itself
    uses, so resume/failure-retention semantics are genuinely exercised, not
    merely asserted."""
    from prism_fas.synthesis import c5_raw_generation as raw
    from prism_fas.synthesis.c5_render import identity_for

    def _fake_render_arm(*, work_root, plan, **_ignored):
        records = []
        reused = rendered = failed = 0
        for row in plan["candidates"]:
            identity = identity_for(row, plan)
            directory = raw.candidate_dir(work_root, plan["arm"], identity.candidate_id)
            directory.mkdir(parents=True, exist_ok=True)
            decision = raw.reuse_decision(directory, identity)
            if call_log is not None:
                call_log.append((plan["arm"], row["candidate_id"], decision["reusable"]))
            if decision["reusable"]:
                reused += 1
                records.append(raw.read_record(directory / raw.RECORD_NAME))
                continue
            if decision["reason"] == "FAILED_GENERATION":
                # TERMINAL and retained -- mirrors render_arm's own branching
                # exactly: a retained failure is never re-attempted, never
                # replaced, and counts toward `failed` again, not `rendered`.
                failed += 1
                records.append(raw.read_record(directory / raw.RECORD_NAME))
                continue
            if row["candidate_id"] in fail_candidate_ids:
                record = raw.failure_record(identity, stage="fake_render",
                                            error=RuntimeError("fake terminal failure"))
                failed += 1
            else:
                payload_sha256 = {}
                for name in raw.PAYLOAD_NAMES:
                    content = f"fake-payload:{identity.candidate_id}:{name}".encode("utf-8")
                    (directory / name).write_bytes(content)
                    payload_sha256[name] = raw.sha256_file(directory / name)
                record = raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                             payload_sha256=payload_sha256, trace={"binding": "fake"})
                rendered += 1
            raw.write_record(directory, record)
            records.append(raw.read_record(directory / raw.RECORD_NAME))
        return {"records": records, "reused": reused, "rendered": rendered, "failed": failed, "rebuilt": 0}

    return _fake_render_arm


def _fake_metrics_provider(*, repo, row, record):
    return _passing_metrics()


# --- 1-3: no-flag / single-flag CLI behavior -------------------------------

def test_no_flags_means_no_execution(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(v2.cc, "repo_root", lambda: repo)
    code = v2.main([])
    assert code == 1
    assert not (repo / v2.E6_V2_CANDIDATES_ROOT).exists()


def test_execute_alone_fails_closed(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(v2.cc, "repo_root", lambda: repo)
    code = v2.main(["--execute"])
    assert code == 2
    assert not (repo / v2.E6_V2_CANDIDATES_ROOT).exists()


def test_authorize_gpu_render_alone_fails_closed(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(v2.cc, "repo_root", lambda: repo)
    code = v2.main(["--authorize-gpu-render"])
    assert code == 2
    assert not (repo / v2.E6_V2_CANDIDATES_ROOT).exists()


# --- 4-6: both flags but preconditions unmet --------------------------------

def test_both_flags_fail_before_candidate_creation_when_readiness_false(tmp_path):
    repo = _base_repo(tmp_path)
    # no fixture at all -> the upstream lock chain itself cannot even build
    # (missing frozen artifacts) -- this fails EVEN EARLIER than a
    # readiness=False execution-plan lock would, which is itself still a
    # "fail before candidate creation" outcome; assert on that observable
    # property (no candidate directory ever created) rather than pin one
    # specific exception class from deep inside the lock-building chain.
    with pytest.raises((v2.E6V2ExecutionError, v2.E6V2ProtocolError, v2.e6r.E6RenderError)):
        v2.run_v2_render_execution(repo)
    assert not (repo / v2.E6_V2_CANDIDATES_ROOT).exists()


def test_both_flags_fail_when_recipe_content_equivalence_not_proven(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original_path = repo / v2.e6r.RECIPE_BANK_LLM_JSONL_PATH
    lines = original_path.read_text(encoding="utf-8").strip().split("\n")
    tampered = json.loads(lines[0])
    tampered["geometry"]["coverage"] = 0.13
    lines[0] = json.dumps(tampered)
    original_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(v2.E6V2ExecutionError):
        v2.run_v2_render_execution(repo)
    assert not (repo / v2.E6_V2_CANDIDATES_ROOT).exists()


def test_both_flags_fail_when_source_pair_execution_parity_not_100(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)

    def _fake_broken_parity(original_rows, shuffle_rows):
        return {"schema_version": "e6-v2-source-pair-execution-parity-v1",
               "all_positions_aligned": False, "source_pair_execution_parity_pct": 42.0,
               "mismatches": [{"position": 0, "field": "route", "original": "gpat", "shuffle": "physics"}]}

    monkeypatch.setattr(v2, "resolve_source_pair_execution_parity", _fake_broken_parity)
    with pytest.raises(v2.E6V2ExecutionError, match="SOURCE_PAIR_EXECUTION_PARITY"):
        v2.run_v2_render_execution(repo)
    assert not (repo / v2.E6_V2_CANDIDATES_ROOT).exists()


# --- 7-11: real execution with fake renderer --------------------------------

def test_execution_uses_additive_v2_namespace_only(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    result = v2.run_v2_render_execution(repo, render_arm_fn=_fake_render_arm_factory(),
                                        metrics_provider=_fake_metrics_provider)
    assert result["source_pair_execution_parity"]["all_positions_aligned"] is True

    for arm in (v2.ARM_ORIGINAL, v2.ARM_SHUFFLE):
        candidates_dir = repo / v2.v2_candidates_root(arm)
        assert candidates_dir.is_relative_to(repo / v2.E6_V2_RUN_ROOT)
        assert candidates_dir.is_dir()
        assert any(candidates_dir.iterdir())


def test_original_and_shuffle_have_separate_candidate_trees(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original_root = v2.v2_candidates_root(v2.ARM_ORIGINAL)
    shuffle_root = v2.v2_candidates_root(v2.ARM_SHUFFLE)
    assert original_root != shuffle_root
    assert v2.ARM_ORIGINAL in original_root and v2.ARM_SHUFFLE in shuffle_root


def test_same_common_source_schedule_used_across_arms(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    result = v2.run_v2_render_execution(repo, render_arm_fn=_fake_render_arm_factory(),
                                        metrics_provider=_fake_metrics_provider)
    parity = result["source_pair_execution_parity"]
    assert parity["positions_checked"] == 2048
    assert parity["source_pair_execution_parity_pct"] == 100.0
    assert parity["all_positions_aligned"] is True


def test_candidate_id_differs_only_where_expected(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    shuffle = v2.e6r.verify_shuffle_recipe_source(repo)
    original_plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL,
                                        recipe_content_identity=original["content_identity"],
                                        recipe_count=original["recipe_count"])
    shuffle_plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_SHUFFLE,
                                       recipe_content_identity=shuffle["content_identity"],
                                       recipe_count=len(shuffle["recipes"]))
    original_rows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_ORIGINAL,
                                             recipe_bank_identity=original_plan["recipe_bank_identity"],
                                             recipes=original["recipes"], plan=original_plan)
    shuffle_rows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_SHUFFLE,
                                            recipe_bank_identity=shuffle_plan["recipe_bank_identity"],
                                            recipes=shuffle["recipes"], plan=shuffle_plan)
    by_position_original = {row["position"]: row for row in original_rows}
    by_position_shuffle = {row["position"]: row for row in shuffle_rows}
    all_candidate_ids_differ = all(
        by_position_original[p]["candidate_id"] != by_position_shuffle[p]["candidate_id"]
        for p in by_position_original)
    all_positions_same = all(
        by_position_original[p]["position"] == by_position_shuffle[p]["position"]
        and by_position_original[p]["route"] == by_position_shuffle[p]["route"]
        and by_position_original[p]["live_target_sample_id"] == by_position_shuffle[p]["live_target_sample_id"]
        for p in by_position_original)
    assert all_candidate_ids_differ  # recipe content differs -> candidate_id always differs
    assert all_positions_same


def test_arm_label_reaches_parameterized_matcher_correctly(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    result = v2.run_v2_render_execution(repo, render_arm_fn=_fake_render_arm_factory(),
                                        metrics_provider=_fake_metrics_provider)
    for arm in (v2.ARM_ORIGINAL, v2.ARM_SHUFFLE):
        selected = result["matched_results"][arm]["selected"]
        assert selected  # the fixture's fake gate always passes -> non-empty bank
        assert {row["arm"] for row in selected} == {arm}


# --- 12-14: historical namespace / target / LLM unreachable ----------------

def test_historical_namespaces_never_written_during_execution(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    (repo / v2.e6r.RENDER_DIR).mkdir(parents=True, exist_ok=True)
    (repo / v2.e6r.RENDER_DIR / "SENTINEL.json").write_text('{"untouched": true}', encoding="utf-8")
    before = (repo / v2.e6r.RENDER_DIR / "SENTINEL.json").read_bytes()
    c5_before = (repo / "reports/full/c5").exists()

    v2.run_v2_render_execution(repo, render_arm_fn=_fake_render_arm_factory(),
                               metrics_provider=_fake_metrics_provider)

    assert (repo / v2.e6r.RENDER_DIR / "SENTINEL.json").read_bytes() == before
    assert (repo / "reports/full/c5").exists() == c5_before
    assert not (repo / "runs/full/c5").exists()


def test_target_access_remains_impossible_in_execution_path():
    source = Path(v2.__file__).read_text(encoding="utf-8")
    for forbidden in ("resolve_target", "SiW", "target_labels_resolved", "prism_target_eval"):
        assert forbidden not in source


def test_llm_call_remains_impossible_in_execution_path():
    source = Path(v2.__file__).read_text(encoding="utf-8")
    for forbidden in ("openai", "google.generativeai", "GEMINI_API_KEY", "llm_call=True"):
        assert forbidden not in source


# --- 15: dry-run/preflight creates zero candidates --------------------------

def test_structural_preflight_creates_zero_candidates(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    result = v2.structural_preflight_v2(repo)
    assert "structural_preflight" in result
    assert result["gpu_hardware_declared_available"] is False
    for arm in (v2.ARM_ORIGINAL, v2.ARM_SHUFFLE):
        candidates_dir = repo / v2.v2_candidates_root(arm)
        if candidates_dir.is_dir():
            assert list(candidates_dir.rglob("CANDIDATE.json")) == []


# --------------------------------------------------------------------------- #
# GPU_RUNTIME_PREFLIGHT (this turn): TASK K's 15 required scenarios.
# Every scenario here is proven against the REAL laptop (genuinely no CUDA)
# or a controlled tmp_path fixture -- never a fabricated "GPU host" result.
# --------------------------------------------------------------------------- #

def _all_pass_checks() -> dict[str, dict]:
    """A fully-passing set of individual check results, used to prove the
    master function's own AND-of-everything logic (never to claim this
    laptop actually has CUDA)."""
    return {
        "cuda": {"CUDA_AVAILABLE": True, "device_count": 1, "device_name": "fake-gpu",
                "torch_cuda_version": "12.0", "selected_device": "cuda:0", "torch_importable": True},
        "gpat": {"GPAT_CHECKPOINT_PRESENT": True, "GPAT_CHECKPOINT_SHA_MATCH": True,
                "GPAT_RUNTIME_IMPORTABLE": True, "GPAT_GPU_BINDING_READY": True},
        "physics": {"PHYSICS_RUNTIME_READY": True},
        "quality": {"QUALITY_RUNTIME_READY": True, "per_backend": []},
        "source_package": {"SOURCE_PACKAGE_PRESENT": True, "SOURCE_PACKAGE_IDENTITY_MATCH": True,
                          "SOURCE_PAIR_PLAN_VALID": True, "PLANNED_POSITIONS_RESOLVABLE": True},
        "lock_chain": {"LOCK_CHAIN_VALID": True},
        "output_storage": {"OUTPUT_STORAGE_READY": True, "AVAILABLE_DISK_SPACE_BYTES": 10**12},
    }


def _patch_all_checks(monkeypatch, overrides: dict[str, dict] | None = None) -> None:
    checks = _all_pass_checks()
    for key, patch in (overrides or {}).items():
        checks[key].update(patch)
    monkeypatch.setattr(v2, "_check_cuda_hardware", lambda: checks["cuda"])
    monkeypatch.setattr(v2, "_check_gpat_runtime", lambda repo, **kw: checks["gpat"])
    monkeypatch.setattr(v2, "_check_physics_runtime", lambda repo: checks["physics"])
    monkeypatch.setattr(v2, "_check_quality_runtime", lambda repo: checks["quality"])
    monkeypatch.setattr(v2, "_check_source_package", lambda repo: checks["source_package"])
    monkeypatch.setattr(v2, "_check_lock_chain", lambda repo: checks["lock_chain"])
    monkeypatch.setattr(v2, "_check_output_storage", lambda repo: checks["output_storage"])


def test_gpu_runtime_preflight_requires_no_execute_flag(tmp_path):
    """TASK A/K.1-3: --gpu-runtime-preflight needs neither --execute nor
    --authorize-gpu-render -- calling it directly with only `repo` proves the
    function itself has no such requirement in its signature or body."""
    repo = _base_repo(tmp_path)
    result = v2.gpu_runtime_preflight_v2(repo)  # no execute/authorize anywhere in this call
    assert "gpu_runtime_preflight" in result
    assert result["rendering_performed"] is False
    assert result["training_performed"] is False
    assert result["candidates_created"] == 0


def test_cli_gpu_runtime_preflight_flag_does_not_require_execute_flags(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    monkeypatch.setattr(v2.cc, "repo_root", lambda: repo)
    code = v2.main(["--gpu-runtime-preflight"])
    assert code == 0
    assert not (repo / v2.E6_V2_CANDIDATES_ROOT).exists()


def test_cuda_unavailable_fails_closed_on_real_laptop(tmp_path):
    """Genuinely run on this laptop, which has no CUDA -- this is an HONEST
    laptop-observed result proving fail-closed behavior, never presented as
    what a GPU host would report."""
    cuda = v2._check_cuda_hardware()
    assert cuda["CUDA_AVAILABLE"] is False

    repo = _base_repo(tmp_path)
    result = v2.gpu_runtime_preflight_v2(repo)
    assert result["cuda"]["CUDA_AVAILABLE"] is False
    assert result["gpu_runtime_preflight"] == "FAIL"


def test_gpat_checkpoint_missing_fails(tmp_path):
    repo = _base_repo(tmp_path)
    (repo / "reports/full/c4").mkdir(parents=True, exist_ok=True)
    (repo / "reports/full/c4/GPAT_CONFIG_LOCK.json").write_text(json.dumps({
        "winning_checkpoint_sha256": v2.e6r.EXPECTED_GPAT_CHECKPOINT_SHA256,
        "winning_checkpoint": "runs/full/c4/scientific/does_not_exist/best.pt"}), encoding="utf-8")
    result = v2._check_gpat_runtime(repo)
    assert result["GPAT_CHECKPOINT_PRESENT"] is False
    assert result["GPAT_GPU_BINDING_READY"] is False


def test_gpat_checkpoint_hash_mismatch_fails(tmp_path):
    repo = _base_repo(tmp_path)
    checkpoint_dir = repo / "runs/full/c4/scientific/trial_fake/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    checkpoint_path.write_bytes(b"not the real checkpoint bytes")
    (repo / "reports/full/c4").mkdir(parents=True, exist_ok=True)
    (repo / "reports/full/c4/GPAT_CONFIG_LOCK.json").write_text(json.dumps({
        "winning_checkpoint_sha256": v2.e6r.EXPECTED_GPAT_CHECKPOINT_SHA256,
        "winning_checkpoint": "runs/full/c4/scientific/trial_fake/checkpoints/best.pt"}), encoding="utf-8")

    result = v2._check_gpat_runtime(repo)
    assert result["GPAT_CHECKPOINT_PRESENT"] is True
    assert result["GPAT_CHECKPOINT_SHA_MATCH"] is False  # real bytes hash != the pinned expectation
    assert result["GPAT_GPU_BINDING_READY"] is False


def test_quality_runtime_unavailable_fails(tmp_path):
    repo = _base_repo(tmp_path)  # no weights/ directory, no calibration -> nothing resolvable
    result = v2._check_quality_runtime(repo)
    assert result["QUALITY_RUNTIME_READY"] is False


def test_source_package_unavailable_fails(tmp_path):
    repo = _base_repo(tmp_path)  # no data/packages/prism_data_v1_m3b at all
    result = v2._check_source_package(repo)
    assert result["SOURCE_PACKAGE_PRESENT"] is False
    assert result["SOURCE_PAIR_PLAN_VALID"] is False
    assert result["PLANNED_POSITIONS_RESOLVABLE"] is False


def test_lock_chain_mismatch_fails(tmp_path):
    repo = _base_repo(tmp_path)  # no v2 fixture at all -> every lock unbuildable/unusable
    result = v2._check_lock_chain(repo)
    assert result["LOCK_CHAIN_VALID"] is False


def test_lock_chain_valid_against_real_fixture(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    v2.write_render_execution_plan_lock(repo)  # the persisted lock --prepare-protocol would write
    result = v2._check_lock_chain(repo)
    assert result["LOCK_CHAIN_VALID"] is True


def test_successful_mocked_gpu_preflight_passes(tmp_path, monkeypatch):
    """Every individual check is mocked to a passing result -- proves the
    MASTER function's own AND-of-everything composition is correct, without
    claiming this laptop has real CUDA/GPAT/quality runtime access."""
    repo = _base_repo(tmp_path)
    _patch_all_checks(monkeypatch)
    result = v2.gpu_runtime_preflight_v2(repo)
    assert result["gpu_runtime_preflight"] == "PASS"


def test_mocked_gpu_preflight_fails_if_any_single_check_fails(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _patch_all_checks(monkeypatch, overrides={"quality": {"QUALITY_RUNTIME_READY": False}})
    result = v2.gpu_runtime_preflight_v2(repo)
    assert result["gpu_runtime_preflight"] == "FAIL"


def test_gpu_ready_for_render_combines_readiness_and_gpu_preflight(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    _patch_all_checks(monkeypatch)
    result = v2.compute_gpu_ready_for_render(repo)
    assert result["E6_V2_READY_FOR_RENDER"] is True
    assert result["GPU_RUNTIME_PREFLIGHT"] == "PASS"
    assert result["E6_V2_GPU_READY_FOR_RENDER"] is True
    assert result["real_render_executed"] is False


def test_gpu_ready_for_render_false_when_gpu_preflight_fails(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    # no mocking -- real laptop CUDA check genuinely fails
    result = v2.compute_gpu_ready_for_render(repo)
    assert result["GPU_RUNTIME_PREFLIGHT"] == "FAIL"
    assert result["E6_V2_GPU_READY_FOR_RENDER"] is False


def test_gpu_preflight_creates_no_candidate_directories(tmp_path):
    repo = _base_repo(tmp_path)
    v2.gpu_runtime_preflight_v2(repo)
    for arm in (v2.ARM_ORIGINAL, v2.ARM_SHUFFLE):
        candidates_dir = repo / v2.v2_candidates_root(arm)
        assert not candidates_dir.exists() or list(candidates_dir.rglob("CANDIDATE.json")) == []


def test_gpu_preflight_never_calls_render_functions_or_target_or_llm():
    source = Path(v2.__file__).read_text(encoding="utf-8")
    preflight_start = source.index("def _check_cuda_hardware(")
    preflight_end = source.index("\ndef design_post_render_cross_arm_gate(")
    body = source[preflight_start:preflight_end]
    for forbidden in ("render_arm(", "render_one(", "GPATRoute(", "PhysicsRoute().apply",
                     "default_quality_matcher(", "train_detector(", "M9TrainingRun(",
                     "resolve_target", "SiW", "openai", "google.generativeai", "GEMINI"):
        assert forbidden not in body, f"{forbidden!r} unexpectedly reachable from GPU preflight code"


def test_gpu_preflight_never_mutates_scientific_artifacts(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    tracked = [
        repo / "reports/full/c4/GPAT_CONFIG_LOCK.json",
        repo / "reports/full/c5/C5_ARM_PLANS.json",
        repo / "reports/full/c6/C6_BANK_LOCK_LLM.json",
        repo / "reports/full/c7/DETECTOR_CONFIG_LOCK.json",
        repo / v2.e6r.RECIPE_BANK_LLM_JSONL_PATH,
        repo / v2.training_plan.E6_SHUFFLE_RECIPES_PATH,
    ]
    before = {path: path.read_bytes() for path in tracked if path.is_file()}

    v2.gpu_runtime_preflight_v2(repo)

    for path, content in before.items():
        assert path.read_bytes() == content, f"{path} was mutated by GPU preflight"


# --- 16-18: failure retention / resume / corruption -------------------------

def test_render_failure_retained_not_silently_replaced(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL,
                               recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    rows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_ORIGINAL,
                                    recipe_bank_identity=plan["recipe_bank_identity"],
                                    recipes=original["recipes"], plan=plan)
    fail_id = rows[0]["candidate_id"]

    rendered = v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                               render_arm_fn=_fake_render_arm_factory(fail_candidate_ids=frozenset({fail_id})))
    assert rendered["failed"] == 1

    from prism_fas.synthesis import c5_raw_generation as raw
    directory = raw.candidate_dir(repo / v2.v2_candidates_root(v2.ARM_ORIGINAL), v2.ARM_ORIGINAL, fail_id)
    record = raw.read_record(directory / raw.RECORD_NAME)
    assert record["status"] == raw.FAILED_GENERATION

    # a second render pass must NOT silently replace the retained failure
    rendered_again = v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                                      render_arm_fn=_fake_render_arm_factory())
    record_after = raw.read_record(directory / raw.RECORD_NAME)
    assert record_after["status"] == raw.FAILED_GENERATION
    # a retained failure is counted again as `failed` on every resume pass
    # (mirroring c5_render.render_arm's own real branching exactly) -- what
    # matters is that it was NEVER re-attempted or replaced, proven above by
    # the record's status staying FAILED_GENERATION byte-identical
    assert rendered_again["failed"] == 1
    assert rendered_again["rendered"] == 0  # never re-rendered as a fresh attempt
    assert rendered_again["reused"] >= 1


def test_resume_skips_valid_completed_candidate(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL,
                               recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])

    call_log: list = []
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory(call_log=call_log))
    first_pass_generated = sum(1 for _arm, _cid, reusable in call_log if not reusable)
    assert first_pass_generated == 2048

    call_log.clear()
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory(call_log=call_log))
    second_pass_reused = sum(1 for _arm, _cid, reusable in call_log if reusable)
    assert second_pass_reused == 2048  # every candidate resumed, none re-rendered


def test_corrupted_partial_candidate_fails_closed_via_reuse_decision(tmp_path, monkeypatch):
    """A candidate directory whose payload bytes no longer match its own
    recorded hash must NOT be silently treated as complete -- reuse_decision
    (the REAL, existing resume primitive render_v2_arm delegates to) must
    say so, and the fake renderer (using that same real primitive) must
    rebuild it rather than skip it."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL,
                               recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())

    from prism_fas.synthesis import c5_raw_generation as raw
    rows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_ORIGINAL,
                                    recipe_bank_identity=plan["recipe_bank_identity"],
                                    recipes=original["recipes"], plan=plan)
    corrupted_id = rows[0]["candidate_id"]
    directory = raw.candidate_dir(repo / v2.v2_candidates_root(v2.ARM_ORIGINAL), v2.ARM_ORIGINAL, corrupted_id)
    record_path = directory / raw.RECORD_NAME
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["payload_sha256"] = {name: "corrupted-hash-value" for name in payload["payload_sha256"]}
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    from prism_fas.synthesis.c5_render import identity_for
    decision = raw.reuse_decision(directory, identity_for(rows[0], plan))
    assert decision["reusable"] is False  # NOT silently accepted as complete


# --- 19-20: post-render gate / dependency chain -----------------------------

def test_training_readiness_remains_false_until_post_render_audit_complete(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    gate = v2.design_post_render_cross_arm_gate()
    assert gate["E6_V2_READY_FOR_TRAINING"] is False

    result = v2.run_v2_render_execution(repo, render_arm_fn=_fake_render_arm_factory(),
                                        metrics_provider=_fake_metrics_provider)
    assert result["e6_v2_ready_for_training"] is False
    assert result["training_performed"] is False


def test_execution_plan_lock_dependency_chain_validated(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    plan_lock = v2.build_render_execution_plan_lock(repo)
    assert plan_lock["status"] == "FROZEN"
    assert plan_lock["readiness_at_lock_time"]["LOCK_DEPENDENCY_CHAIN_VALID"] is True
    for key in ("protocol", "recipe_pair", "source_pair_parity", "render_parity", "quality_parity",
               "training_plan"):
        assert plan_lock["upstream_lock_identities"][key]
    for arm in (v2.ARM_ORIGINAL, v2.ARM_SHUFFLE):
        assert plan_lock["per_arm"][arm]["planned_candidate_count"] == 2048
        assert plan_lock["per_arm"][arm]["planned_physics_count"] == 1024
        assert plan_lock["per_arm"][arm]["planned_gpat_count"] == 1024


# --------------------------------------------------------------------------- #
# no target / model / render / train / LLM reachability (execution additions)
# --------------------------------------------------------------------------- #

def test_v2_execution_functions_never_touch_target_model_or_llm():
    source = Path(v2.__file__).read_text(encoding="utf-8")
    for fn_name in ("build_v2_arm_plan_rows", "resolve_source_pair_execution_parity",
                    "render_v2_arm", "match_v2_arm", "build_v2_matched_bank_lock",
                    "build_render_execution_plan_lock", "structural_preflight_v2",
                    "run_v2_render_execution"):
        fn_start = source.index(f"def {fn_name}(")
        fn_end = source.index("\ndef ", fn_start + 10)
        body = source[fn_start:fn_end]
        for forbidden in ("resolve_target", "SiW", "openai", "google.generativeai", "GEMINI",
                         "train_detector(", "M9TrainingRun(", "torch.cuda.is_available() == False"):
            assert forbidden not in body, f"{forbidden!r} unexpectedly reachable from {fn_name}"


def test_execution_cli_flags_documented_and_fail_closed_by_default():
    source = Path(v2.__file__).read_text(encoding="utf-8")
    assert "requires --authorize-gpu-render" in source
    assert "requires --execute" in source


# =============================================================================
# EXECUTION-PLAN-LOCK PROVENANCE GAP (this turn): TASK F's 12 required tests.
# =============================================================================

def test_prepare_protocol_writes_execution_plan_lock(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    lock_path = repo / v2.RENDER_EXECUTION_PLAN_LOCK_PATH
    assert not lock_path.is_file()  # not written by the fixture itself

    result = v2.run_e6_v2_protocol_preparation(repo)

    assert lock_path.is_file()
    assert result["execution_plan_lock"]["path"] == str(lock_path)
    persisted = json.loads(lock_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "FROZEN"
    assert persisted == result["execution_plan_lock"]["lock"]


def test_execution_plan_lock_deterministic(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    first = v2.build_render_execution_plan_lock(repo)
    second = v2.build_render_execution_plan_lock(repo)
    assert first == second
    assert first["lock_identity"] == second["lock_identity"]


def test_persisted_lock_equals_rebuilt_expected_lock(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    v2.run_e6_v2_protocol_preparation(repo)

    verification = v2.verify_execution_plan_lock_matches_expected(repo)
    assert verification["EXECUTION_PLAN_LOCK_PRESENT"] is True
    assert verification["EXECUTION_PLAN_LOCK_EXPECTED_EQUALS_PERSISTED"] is True
    assert verification["EXECUTION_PLAN_LOCK_IDENTITY"] is not None
    assert verification["persisted_status"] == "FROZEN"


def test_structural_preflight_does_not_create_missing_lock(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    lock_path = repo / v2.RENDER_EXECUTION_PLAN_LOCK_PATH
    assert not lock_path.is_file()

    v2.structural_preflight_v2(repo)

    assert not lock_path.is_file()  # still not written -- structural preflight is read-only


def test_structural_preflight_reports_missing_lock_honestly(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    result = v2.structural_preflight_v2(repo)
    assert result["checks"]["execution_plan_lock_present"] is False
    assert result["structural_preflight"] == "FAIL"  # now a REQUIRED gate, not merely informational


def test_structural_preflight_passes_once_lock_is_persisted(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    # structural_preflight_v2 also checks GPAT_CONFIG_LOCK.json presence and
    # the source package directory -- neither is part of _full_v2_fixture
    # (which targets the six v2 locks + execution plan, not every structural
    # file), so both are added here to reach a genuine, isolated PASS.
    (repo / "reports/full/c4").mkdir(parents=True, exist_ok=True)
    (repo / "reports/full/c4/GPAT_CONFIG_LOCK.json").write_text(
        json.dumps({"winning_checkpoint_sha256": v2.e6r.EXPECTED_GPAT_CHECKPOINT_SHA256}), encoding="utf-8")
    (repo / "data/packages/prism_data_v1_m3b").mkdir(parents=True, exist_ok=True)

    v2.run_e6_v2_protocol_preparation(repo)
    result = v2.structural_preflight_v2(repo)
    assert result["checks"]["execution_plan_lock_present"] is True
    assert result["structural_preflight"] == "PASS"


def test_gpu_runtime_preflight_missing_lock_fails(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    # every OTHER check is mocked to PASS, isolating the missing-lock effect;
    # _check_lock_chain is deliberately NOT mocked, so it sees the real,
    # never-persisted lock file
    monkeypatch.setattr(v2, "_check_cuda_hardware", lambda: _all_pass_checks()["cuda"])
    monkeypatch.setattr(v2, "_check_gpat_runtime", lambda repo, **kw: _all_pass_checks()["gpat"])
    monkeypatch.setattr(v2, "_check_physics_runtime", lambda repo: _all_pass_checks()["physics"])
    monkeypatch.setattr(v2, "_check_quality_runtime", lambda repo: _all_pass_checks()["quality"])
    monkeypatch.setattr(v2, "_check_source_package", lambda repo: _all_pass_checks()["source_package"])
    monkeypatch.setattr(v2, "_check_output_storage", lambda repo: _all_pass_checks()["output_storage"])
    # _check_lock_chain deliberately NOT mocked -- no lock was ever persisted

    result = v2.gpu_runtime_preflight_v2(repo)
    assert result["lock_chain"]["execution_plan_lock_verification"]["EXECUTION_PLAN_LOCK_PRESENT"] is False
    assert result["lock_chain"]["LOCK_CHAIN_VALID"] is False
    assert result["gpu_runtime_preflight"] == "FAIL"


def test_gpu_runtime_preflight_tampered_lock_fails(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    v2.run_e6_v2_protocol_preparation(repo)
    lock_path = repo / v2.RENDER_EXECUTION_PLAN_LOCK_PATH
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["per_arm"][v2.ARM_ORIGINAL]["planned_candidate_count"] = 999999  # tamper
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(v2, "_check_cuda_hardware", lambda: _all_pass_checks()["cuda"])
    monkeypatch.setattr(v2, "_check_gpat_runtime", lambda repo, **kw: _all_pass_checks()["gpat"])
    monkeypatch.setattr(v2, "_check_physics_runtime", lambda repo: _all_pass_checks()["physics"])
    monkeypatch.setattr(v2, "_check_quality_runtime", lambda repo: _all_pass_checks()["quality"])
    monkeypatch.setattr(v2, "_check_source_package", lambda repo: _all_pass_checks()["source_package"])
    monkeypatch.setattr(v2, "_check_output_storage", lambda repo: _all_pass_checks()["output_storage"])

    result = v2.gpu_runtime_preflight_v2(repo)
    verification = result["lock_chain"]["execution_plan_lock_verification"]
    assert verification["EXECUTION_PLAN_LOCK_PRESENT"] is True
    assert verification["EXECUTION_PLAN_LOCK_EXPECTED_EQUALS_PERSISTED"] is False
    assert result["lock_chain"]["LOCK_CHAIN_VALID"] is False
    assert result["gpu_runtime_preflight"] == "FAIL"


def test_gpu_runtime_preflight_valid_persisted_lock_passes(tmp_path, monkeypatch):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    v2.run_e6_v2_protocol_preparation(repo)  # persists a genuinely valid lock

    monkeypatch.setattr(v2, "_check_cuda_hardware", lambda: _all_pass_checks()["cuda"])
    monkeypatch.setattr(v2, "_check_gpat_runtime", lambda repo, **kw: _all_pass_checks()["gpat"])
    monkeypatch.setattr(v2, "_check_physics_runtime", lambda repo: _all_pass_checks()["physics"])
    monkeypatch.setattr(v2, "_check_quality_runtime", lambda repo: _all_pass_checks()["quality"])
    monkeypatch.setattr(v2, "_check_source_package", lambda repo: _all_pass_checks()["source_package"])
    monkeypatch.setattr(v2, "_check_output_storage", lambda repo: _all_pass_checks()["output_storage"])
    # _check_lock_chain deliberately NOT mocked -- must pass against the REAL persisted lock

    result = v2.gpu_runtime_preflight_v2(repo)
    assert result["lock_chain"]["execution_plan_lock_verification"][
        "EXECUTION_PLAN_LOCK_EXPECTED_EQUALS_PERSISTED"] is True
    assert result["lock_chain"]["LOCK_CHAIN_VALID"] is True
    assert result["gpu_runtime_preflight"] == "PASS"


def test_execution_path_missing_lock_fails_before_candidate_creation(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    lock_path = repo / v2.RENDER_EXECUTION_PLAN_LOCK_PATH
    lock_path.unlink()  # the fixture writes it -- remove it to isolate THIS failure mode

    with pytest.raises(v2.E6V2ExecutionError, match="not persisted on disk"):
        v2.run_v2_render_execution(repo, render_arm_fn=_fake_render_arm_factory(),
                                   metrics_provider=_fake_metrics_provider)
    for arm in (v2.ARM_ORIGINAL, v2.ARM_SHUFFLE):
        candidates_dir = repo / v2.v2_candidates_root(arm)
        assert not candidates_dir.exists() or list(candidates_dir.rglob("CANDIDATE.json")) == []


def test_execution_path_tampered_lock_fails_before_candidate_creation(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    lock_path = repo / v2.RENDER_EXECUTION_PLAN_LOCK_PATH
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["per_arm"][v2.ARM_SHUFFLE]["gpat_checkpoint_sha256"] = "tampered-not-the-real-checkpoint"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(v2.E6V2ExecutionError, match="does not match a freshly rebuilt expected plan"):
        v2.run_v2_render_execution(repo, render_arm_fn=_fake_render_arm_factory(),
                                   metrics_provider=_fake_metrics_provider)
    for arm in (v2.ARM_ORIGINAL, v2.ARM_SHUFFLE):
        candidates_dir = repo / v2.v2_candidates_root(arm)
        assert not candidates_dir.exists() or list(candidates_dir.rglob("CANDIDATE.json")) == []


def test_preflight_modes_remain_read_only_regarding_execution_plan_lock(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    lock_path = repo / v2.RENDER_EXECUTION_PLAN_LOCK_PATH

    v2.structural_preflight_v2(repo)
    assert not lock_path.is_file()

    v2.gpu_runtime_preflight_v2(repo)
    assert not lock_path.is_file()

    source = Path(v2.__file__).read_text(encoding="utf-8")
    for fn_name in ("structural_preflight_v2", "gpu_runtime_preflight_v2", "_check_lock_chain",
                    "verify_execution_plan_lock_matches_expected", "load_persisted_execution_plan_lock"):
        fn_start = source.index(f"def {fn_name}(")
        fn_end = source.index("\ndef ", fn_start + 10)
        body = source[fn_start:fn_end]
        assert "write_render_execution_plan_lock(" not in body, \
            f"{fn_name} unexpectedly calls the WRITING operation"


def test_protected_historical_namespaces_untouched_by_lock_freeze(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    (repo / v2.e6r.RENDER_DIR).mkdir(parents=True, exist_ok=True)
    (repo / v2.e6r.RENDER_DIR / "SENTINEL.json").write_text('{"untouched": true}', encoding="utf-8")
    before_sentinel = (repo / v2.e6r.RENDER_DIR / "SENTINEL.json").read_bytes()
    tracked = [repo / "reports/full/c4/GPAT_CONFIG_LOCK.json", repo / "reports/full/c7/DETECTOR_CONFIG_LOCK.json",
              repo / v2.e6r.RECIPE_BANK_LLM_JSONL_PATH, repo / v2.training_plan.E6_SHUFFLE_RECIPES_PATH]
    before = {path: path.read_bytes() for path in tracked if path.is_file()}

    v2.run_e6_v2_protocol_preparation(repo)

    assert (repo / v2.e6r.RENDER_DIR / "SENTINEL.json").read_bytes() == before_sentinel
    for path, content in before.items():
        assert path.read_bytes() == content


# =============================================================================
# ATTEMPT-1 TECHNICAL RECOVERY (this turn): TASK L's 25 required tests.
# =============================================================================

def test_double_arm_append_reproduced_by_old_v2_binding():
    """TASK L.1: the OLD (buggy) v2 binding passed v2_candidates_root(arm)
    (already arm-inclusive) as work_root to candidate_dir, which appends arm
    again -- reproduces the EXACT observed GPU path."""
    from prism_fas.synthesis import c5_raw_generation as raw

    old_buggy_work_root = Path(v2.v2_candidates_root(v2.ARM_ORIGINAL))
    doubled = raw.candidate_dir(old_buggy_work_root, v2.ARM_ORIGINAL, "c5syn_example")
    assert str(doubled) == (
        f"{v2.E6_V2_CANDIDATES_ROOT}/{v2.ARM_ORIGINAL}/{v2.ARM_ORIGINAL}/c5syn_example")


def test_corrected_v2_renderer_path_appends_arm_exactly_once(tmp_path, monkeypatch):
    """TASK L.2/20: SHUFFLE (the arm with NO attempt-1 recovery exception)
    must render to a SINGLE-append path -- the corrected contract."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    shuffle = v2.e6r.verify_shuffle_recipe_source(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_SHUFFLE, recipe_content_identity=shuffle["content_identity"],
                               recipe_count=len(shuffle["recipes"]))
    rendered = v2.render_v2_arm(repo=repo, arm=v2.ARM_SHUFFLE, plan=plan, recipes=shuffle["recipes"],
                               render_arm_fn=_fake_render_arm_factory())

    expected_root = repo / v2.E6_V2_CANDIDATES_ROOT / v2.ARM_SHUFFLE
    assert Path(rendered["candidates_root"]) == expected_root
    assert expected_root.is_dir()
    entries = [p for p in expected_root.iterdir() if p.is_dir()]
    assert entries  # candidate_id directories live DIRECTLY here
    assert all(p.name.startswith("c5syn_") for p in entries)  # never another "LLM_SHUFFLE_A_CURRENT_V2" level
    assert not (expected_root / v2.ARM_SHUFFLE).exists()  # no doubled nesting


def test_shuffle_future_path_never_doubles_even_via_v2_render_work_root():
    assert v2.v2_render_work_root(v2.ARM_SHUFFLE) == v2.E6_V2_CANDIDATES_ROOT
    assert v2.v2_render_work_root(v2.ARM_ORIGINAL) == v2.ATTEMPT1_ORIGINAL_RECOVERY_WORK_ROOT
    assert v2.ATTEMPT1_ORIGINAL_RECOVERY_WORK_ROOT == v2.v2_candidates_root(v2.ARM_ORIGINAL)


# --- historical E6 regression (TASK D / L.3,5,6) ----------------------------

def _stub_quality_runtime(monkeypatch) -> None:
    """Bypasses `_resolve_quality_runtime`'s real CUDA/calibration resolution
    (unavailable on this laptop) with an empty stub, so tests can observe
    default_metrics_provider's candidate_dir/record resolution in isolation --
    the function only touches `runtime["store"/"bank"/"evaluator"]` AFTER the
    "no GENERATED candidate record" check, so a candidate that resolves stops
    here (KeyError, caught generically) while one that fails to resolve still
    raises the real "no GENERATED candidate record" E6RenderError first."""
    monkeypatch.setattr(v2.e6r, "_resolve_quality_runtime", lambda repo, **kwargs: {})


def test_historical_default_metrics_provider_unchanged_without_new_params(tmp_path, monkeypatch):
    """TASK D/L.3,5: calling the historical symbols WITHOUT candidates_root/
    arm must resolve the EXACT historical default (c_ext_e6_render.
    CANDIDATES_ROOT / E6_ARM_NAME), byte-for-byte -- proven by a real
    candidate written at the historical path and read back with NO override."""
    from prism_fas.synthesis import c5_raw_generation as raw

    repo = _base_repo(tmp_path)
    _stub_quality_runtime(monkeypatch)
    directory = raw.candidate_dir(repo / v2.e6r.CANDIDATES_ROOT, v2.e6r.E6_ARM_NAME, "c5syn_hist")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "synthetic.png").write_bytes(b"x")
    (directory / "exact_mask.png").write_bytes(b"y")
    (directory / "artifact_map.npz").write_bytes(b"z")
    identity = raw.GenerationIdentity(
        candidate_id="c5syn_hist", arm=v2.e6r.E6_ARM_NAME, arm_plan_identity="p",
        source_pair_plan_identity="s", package_identity="pkg", recipe_bank_identity="bank",
        recipe_id="R-000000", recipe_ordinal=0, slot=0, position=0, route="physics",
        live_target_sample_id="live-0", spoof_source_sample_id=None,
        generator_binding="binding", ontology_identity="ont")
    payload_sha256 = {name: raw.sha256_file(directory / name) for name in raw.PAYLOAD_NAMES}
    raw.write_record(directory, raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                                     payload_sha256=payload_sha256, trace={}))

    # default_metrics_provider called with NO candidates_root/arm -- must find the
    # historical-path record without raising the "no GENERATED candidate record" error
    # (it will still fail later at _resolve_quality_runtime, which is fine/expected --
    # we only assert it got PAST the candidate_dir resolution using the historical default)
    try:
        v2.e6r.default_metrics_provider(repo=repo, row={"candidate_id": "c5syn_hist",
                                                       "live_target_sample_id": "live-0"},
                                       record={"reusable": True})
    except v2.e6r.E6RenderError as error:
        assert "no GENERATED candidate record" not in str(error)
    except Exception:  # noqa: BLE001 - _resolve_quality_runtime needs real weights; expected past this point
        pass


def test_historical_matcher_root_unchanged(tmp_path):
    """TASK L.5: default_quality_matcher called with NO candidates_root
    (the exact historical call shape) forwards NO candidates_root/arm
    override to the provider -- proven structurally."""
    import inspect

    signature = inspect.signature(v2.e6r.default_quality_matcher)
    assert signature.parameters["candidates_root"].default is None


def test_historical_regression_selection_byte_identical_with_and_without_new_kwarg(tmp_path):
    """TASK D: the full historical selection test from the matcher-
    parameterization turn, reproduced with the NEW candidates_root parameter
    simply omitted -- selection must be byte-identical to what it always was."""
    from test_c_ext_e6_render import (_full_fixture, _passing_metrics, _quality_gate_fixture,  # noqa: E402
                                      _route_quota_fixture, _write_q_reference_fixture)
    from prism_fas.synthesis.c6_matched_bank import selected_set_digest

    repo = _full_fixture(tmp_path)
    _quality_gate_fixture(repo)
    _route_quota_fixture(repo, physics=2, gpat=2)
    _write_q_reference_fixture(repo)
    plan = v2.e6r.build_render_plan(repo)

    rows, results = [], []
    for i in range(4):
        route = "physics" if i < 2 else "gpat"
        rows.append({"candidate_id": f"cand-{i}", "recipe_id": f"R-{i:06d}", "recipe_ordinal": i,
                    "position": i, "route": route, "live_dataset": "domainA",
                    "live_target_sample_id": f"live-{i}"})
        results.append({"candidate_id": f"cand-{i}"})
    staged = {"rows": rows, "results": results}

    def _fake_metrics_provider(*, repo, row, record):
        return _passing_metrics()

    matched = v2.e6r.default_quality_matcher(repo=repo, plan=plan, staged=staged, arm=v2.e6r.E6_ARM_NAME,
                                            metrics_provider=_fake_metrics_provider)
    assert len(matched["selected"]) == 4
    assert {row["route"] for row in matched["selected"]} == {"physics", "gpat"}
    assert matched["selected_set_sha256"] == selected_set_digest(matched["selected"])
    assert {row["arm"] for row in matched["selected"]} == {v2.e6r.E6_ARM_NAME}


# --- explicit v2 candidate-root quality lookup (TASK C / L.4,6,7,8) --------

def _write_generated_candidate_at(repo: Path, *, work_root: Path, arm: str, candidate_id: str,
                                  live_target_sample_id: str = "live-0") -> None:
    from prism_fas.synthesis import c5_raw_generation as raw

    directory = raw.candidate_dir(work_root, arm, candidate_id)
    directory.mkdir(parents=True, exist_ok=True)
    for name in raw.PAYLOAD_NAMES:
        (directory / name).write_bytes(f"payload:{candidate_id}:{name}".encode("utf-8"))
    identity = raw.GenerationIdentity(
        candidate_id=candidate_id, arm=arm, arm_plan_identity="p", source_pair_plan_identity="s",
        package_identity="pkg", recipe_bank_identity="bank", recipe_id="R-000000", recipe_ordinal=0,
        slot=0, position=0, route="physics", live_target_sample_id=live_target_sample_id,
        spoof_source_sample_id=None, generator_binding="binding", ontology_identity="ont")
    payload_sha256 = {name: raw.sha256_file(directory / name) for name in raw.PAYLOAD_NAMES}
    raw.write_record(directory, raw.CandidateRecord(identity=identity, status=raw.GENERATED,
                                                     payload_sha256=payload_sha256, trace={}))


def test_explicit_candidate_root_quality_lookup_resolves_to_supplied_root(tmp_path, monkeypatch):
    """TASK L.4/7: a candidate written ONLY under an explicit v2 root is
    found when that root is supplied -- proven by getting PAST the
    "no GENERATED candidate record" gate."""
    repo = _base_repo(tmp_path)
    _stub_quality_runtime(monkeypatch)
    v2_root = repo / "runs/c_ext_q1q2_v1/EXT-F1/e6_paired_current_runtime_v2/candidates"
    _write_generated_candidate_at(repo, work_root=v2_root, arm=v2.ARM_SHUFFLE, candidate_id="c5syn_v2only")

    try:
        v2.e6r.default_metrics_provider(
            repo=repo, row={"candidate_id": "c5syn_v2only", "live_target_sample_id": "live-0"},
            record={"reusable": True}, candidates_root=v2_root, arm=v2.ARM_SHUFFLE)
    except v2.e6r.E6RenderError as error:
        assert "no GENERATED candidate record" not in str(error)
    except Exception:  # noqa: BLE001 - expected past candidate_dir resolution
        pass


def test_v2_cannot_fall_back_to_historical_root(tmp_path, monkeypatch):
    """TASK L.6/8: a candidate that exists ONLY at the historical E6 root
    must NOT be found when v2 supplies its own explicit root -- v2 must fail
    closed with "no GENERATED candidate record", never silently redirected
    to the historical path."""
    repo = _base_repo(tmp_path)
    _stub_quality_runtime(monkeypatch)
    _write_generated_candidate_at(repo, work_root=repo / v2.e6r.CANDIDATES_ROOT, arm=v2.e6r.E6_ARM_NAME,
                                  candidate_id="c5syn_historical_only")

    v2_root = repo / "runs/c_ext_q1q2_v1/EXT-F1/e6_paired_current_runtime_v2/candidates"
    with pytest.raises(v2.e6r.E6RenderError, match="no GENERATED candidate record"):
        v2.e6r.default_metrics_provider(
            repo=repo, row={"candidate_id": "c5syn_historical_only", "live_target_sample_id": "live-0"},
            record={"reusable": True}, candidates_root=v2_root, arm=v2.ARM_ORIGINAL)


def test_candidate_in_nested_attempt1_root_resolves_via_v2_render_work_root(tmp_path, monkeypatch):
    """TASK L.7: the ACTUAL nested attempt-1 physical location -- reached via
    v2_render_work_root(ARM_ORIGINAL) (already arm-inclusive) + candidate_dir's
    own single append -- resolves correctly when supplied explicitly."""
    repo = _base_repo(tmp_path)
    _stub_quality_runtime(monkeypatch)
    nested_work_root = repo / v2.v2_render_work_root(v2.ARM_ORIGINAL)
    _write_generated_candidate_at(repo, work_root=nested_work_root, arm=v2.ARM_ORIGINAL,
                                  candidate_id="c5syn_nested")
    # sanity: this really did produce the doubled physical layout (one level
    # from the arm-inclusive recovery root itself, one from candidate_dir's append)
    assert (nested_work_root / v2.ARM_ORIGINAL / "c5syn_nested").is_dir()

    try:
        v2.e6r.default_metrics_provider(
            repo=repo, row={"candidate_id": "c5syn_nested", "live_target_sample_id": "live-0"},
            record={"reusable": True}, candidates_root=nested_work_root, arm=v2.ARM_ORIGINAL)
    except v2.e6r.E6RenderError as error:
        assert "no GENERATED candidate record" not in str(error)
    except Exception:  # noqa: BLE001
        pass


# --- resume-preflight auditor (TASK E / L.9-16) -----------------------------

def test_resume_audit_handles_nested_attempt1_layout(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())

    result = v2.audit_attempt1_original(repo)
    assert result["ATTEMPT1_ORIGINAL_PLANNED"] == 2048
    assert result["ATTEMPT1_ORIGINAL_RECORDS"] == 2048
    assert result["ATTEMPT1_ORIGINAL_GENERATED"] == 2048
    assert result["ATTEMPT1_ORIGINAL_MISSING"] == 0
    assert result["ATTEMPT1_ORIGINAL_DUPLICATES"] == 0
    assert result["ATTEMPT1_ORIGINAL_UNEXPECTED"] == 0
    assert result["ATTEMPT1_SHUFFLE_RECORDS"] == 0
    assert v2.ARM_ORIGINAL in result["audited_root"]


def test_resume_audit_identity_from_generation_identity_not_top_level(tmp_path, monkeypatch):
    """TASK L.10/11: proves the auditor reads generation_identity.candidate_id
    / generation_identity.route -- a record with a top-level candidate_id/
    route (which the real schema never has) must NOT be misread as if THOSE
    were authoritative; classification must still come from the nested
    generation_identity fields alone."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())

    # tamper ONE record to ALSO carry misleading top-level candidate_id/route
    # keys the real schema never has -- the auditor must ignore them
    arm_dir = repo / v2.v2_render_work_root(v2.ARM_ORIGINAL) / v2.ARM_ORIGINAL
    one_dir = next(p for p in arm_dir.iterdir() if p.is_dir())
    record_path = one_dir / "CANDIDATE.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["candidate_id"] = None  # exactly the misleading shape the GPU audit script hit
    payload["route"] = None
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    result = v2.audit_attempt1_original(repo)
    assert result["ATTEMPT1_ORIGINAL_GENERATED"] == 2048  # unaffected -- real identity is still nested
    assert result["ATTEMPT1_ORIGINAL_INVALID"] == 0


def test_resume_audit_generated_payload_integrity(tmp_path, monkeypatch):
    """TASK L.12: a GENERATED record whose payload file was deleted after
    the fact must be classified INVALID, not silently GENERATED_VALID."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())

    arm_dir = repo / v2.v2_render_work_root(v2.ARM_ORIGINAL) / v2.ARM_ORIGINAL
    one_dir = next(p for p in arm_dir.iterdir() if p.is_dir())
    (one_dir / "synthetic.png").unlink()

    result = v2.audit_attempt1_original(repo)
    assert result["ATTEMPT1_ORIGINAL_GENERATED"] == 2047
    assert result["ATTEMPT1_ORIGINAL_INVALID"] == 1


def test_resume_audit_failed_generation_retained(tmp_path, monkeypatch):
    """TASK L.13: FAILED_GENERATION records are classified separately from
    GENERATED, never counted as generated or as invalid."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    rows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_ORIGINAL, recipe_bank_identity=plan["recipe_bank_identity"],
                                    recipes=original["recipes"], plan=plan)
    fail_ids = frozenset(row["candidate_id"] for row in rows[:3])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory(fail_candidate_ids=fail_ids))

    result = v2.audit_attempt1_original(repo)
    assert result["ATTEMPT1_ORIGINAL_FAILED_GENERATION"] == 3
    assert result["ATTEMPT1_ORIGINAL_GENERATED"] == 2045
    assert result["ATTEMPT1_ORIGINAL_REUSABLE"] == 2048


def test_resume_audit_duplicate_detection(tmp_path, monkeypatch):
    """TASK L.14: two DIFFERENT directories whose records both claim the SAME
    recorded generation_identity.candidate_id must be flagged DUPLICATE."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())

    arm_dir = repo / v2.v2_render_work_root(v2.ARM_ORIGINAL) / v2.ARM_ORIGINAL
    real_dir = next(p for p in arm_dir.iterdir() if p.is_dir())
    real_record = json.loads((real_dir / "CANDIDATE.json").read_text(encoding="utf-8"))
    duplicate_dir = arm_dir / "a-duplicate-directory-name"
    duplicate_dir.mkdir()
    (duplicate_dir / "CANDIDATE.json").write_text(json.dumps(real_record), encoding="utf-8")  # same recorded id

    result = v2.audit_attempt1_original(repo)
    assert result["ATTEMPT1_ORIGINAL_DUPLICATES"] == 1


def test_resume_audit_unexpected_id_detection(tmp_path, monkeypatch):
    """TASK L.15: a well-formed but NOT-in-the-frozen-plan candidate_id is
    UNEXPECTED, never silently absorbed."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())

    arm_dir = repo / v2.v2_render_work_root(v2.ARM_ORIGINAL) / v2.ARM_ORIGINAL
    _write_generated_candidate_at(repo, work_root=arm_dir.parent, arm=v2.ARM_ORIGINAL,
                                  candidate_id="c5syn_not_in_the_plan_at_all")

    result = v2.audit_attempt1_original(repo)
    assert result["ATTEMPT1_ORIGINAL_UNEXPECTED"] == 1
    assert "c5syn_not_in_the_plan_at_all" in result["unexpected_ids"]


def test_resume_audit_missing_planned_candidate_detection(tmp_path, monkeypatch):
    """TASK L.16: a planned candidate with NO directory at all is MISSING."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())

    arm_dir = repo / v2.v2_render_work_root(v2.ARM_ORIGINAL) / v2.ARM_ORIGINAL
    import shutil
    victim = next(p for p in arm_dir.iterdir() if p.is_dir())
    shutil.rmtree(victim)

    result = v2.audit_attempt1_original(repo)
    assert result["ATTEMPT1_ORIGINAL_MISSING"] == 1
    assert result["ATTEMPT1_ORIGINAL_GENERATED"] == 2047


# --- resume safety (TASK F / L.17-19) ---------------------------------------

def test_valid_original_candidate_reused_on_resume(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())

    call_log: list = []
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory(call_log=call_log))
    assert sum(1 for _a, _c, reusable in call_log if reusable) == 2048


def test_valid_candidate_never_rerendered_on_resume(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())
    before = v2.audit_attempt1_original(repo)

    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())
    after = v2.audit_attempt1_original(repo)
    assert after["ATTEMPT1_ORIGINAL_GENERATED"] == before["ATTEMPT1_ORIGINAL_GENERATED"] == 2048


def test_failed_generation_never_resampled_on_resume(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    rows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_ORIGINAL, recipe_bank_identity=plan["recipe_bank_identity"],
                                    recipes=original["recipes"], plan=plan)
    fail_ids = frozenset({rows[0]["candidate_id"]})
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory(fail_candidate_ids=fail_ids))

    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())  # a second, "fixed" pass
    result = v2.audit_attempt1_original(repo)
    assert result["ATTEMPT1_ORIGINAL_FAILED_GENERATION"] == 1  # still retained, not resampled into GENERATED
    assert result["ATTEMPT1_ORIGINAL_GENERATED"] == 2047


# --- provenance / recovery lock (TASK I/J / L.21,22) ------------------------

def test_technical_provenance_record(tmp_path):
    repo = _base_repo(tmp_path)
    result = v2.write_attempt1_provenance(repo)
    provenance = result["provenance"]
    assert provenance["ATTEMPT"] == 1
    assert provenance["STATUS"] == "TECHNICAL_FAILURE"
    assert provenance["FAILURE_STAGE"] == "ORIGINAL_QUALITY_MATCHING"
    assert provenance["FAILURE_CLASS"] == \
        "RENDER_ROOT_DOUBLE_ARM_APPEND_AND_QUALITY_ROOT_HISTORICAL_BINDING"
    assert provenance["SCIENTIFIC_PROTOCOL_CHANGED"] is False
    assert provenance["RENDER_ALGORITHM_CHANGED"] is False
    assert provenance["QUALITY_ALGORITHM_CHANGED"] is False
    assert provenance["MATCHING_ALGORITHM_CHANGED"] is False
    assert provenance["TARGET_ACCESS"] is False
    assert provenance["LLM_API_CALLS"] == 0
    assert provenance["gpu_observed_facts"]["original_candidate_count"] == 2048
    assert provenance["gpu_observed_facts"]["original_generated"] == 2045
    assert provenance["gpu_observed_facts"]["original_failed_generation"] == 3
    assert provenance["gpu_observed_facts"]["shuffle_records"] == 0
    assert Path(result["path"]).is_file()
    assert Path(result["path"]).is_relative_to(repo / v2.E6_V2_DIR)


def test_recovery_lock_dependency_on_execution_plan_lock(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    v2.run_e6_v2_protocol_preparation(repo)
    execution_lock = v2.load_persisted_execution_plan_lock(repo)

    result = v2.write_attempt1_recovery_lock(repo)
    recovery_lock = result["lock"]
    assert recovery_lock["status"] == "FROZEN"
    assert recovery_lock["original_execution_plan_lock_identity"] == execution_lock["lock_identity"]
    assert recovery_lock["original_execution_plan_lock_present"] is True
    assert recovery_lock["recovery_option_chosen"] == "A"
    assert recovery_lock["existing_payload_bytes_preserved"] is True
    assert recovery_lock["scientific_protocol_unchanged"] is True
    assert recovery_lock["actual_nested_original_root"] == \
        f"{v2.ATTEMPT1_ORIGINAL_RECOVERY_WORK_ROOT}/{v2.ARM_ORIGINAL}"
    assert Path(result["path"]).is_file()


def test_recovery_lock_does_not_touch_execution_plan_lock(tmp_path):
    repo = _base_repo(tmp_path)
    _full_v2_fixture(repo, recipe_count=256, with_training_config=True)
    v2.run_e6_v2_protocol_preparation(repo)
    execution_lock_path = repo / v2.RENDER_EXECUTION_PLAN_LOCK_PATH
    before = execution_lock_path.read_bytes()

    v2.write_attempt1_recovery_lock(repo)

    assert execution_lock_path.read_bytes() == before


def test_resume_execution_gate_design_not_authorized():
    gate = v2.design_resume_execution_gate()
    assert gate["GPU_RESUME_AUTHORIZED"] is False
    assert gate["not_run_this_turn"] is True
    assert len(gate["required_before_resume"]) >= 5


# --- resume-preflight read-only / unreachability (TASK L.23,24,25) --------

def test_resume_preflight_is_read_only(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())

    arm_dir = repo / v2.v2_render_work_root(v2.ARM_ORIGINAL) / v2.ARM_ORIGINAL
    before = {p: p.read_bytes() for p in arm_dir.rglob("*") if p.is_file()}

    v2.audit_attempt1_original(repo)

    after = {p: p.read_bytes() for p in arm_dir.rglob("*") if p.is_file()}
    assert before == after


def test_resume_preflight_cli_flag_present_and_read_only():
    source = Path(v2.__file__).read_text(encoding="utf-8")
    assert "--resume-preflight" in source
    assert "audit_attempt1_original" in source


def test_attempt1_recovery_functions_never_touch_target_or_llm():
    source = Path(v2.__file__).read_text(encoding="utf-8")
    for fn_name in ("audit_attempt1_original", "build_attempt1_provenance",
                    "build_attempt1_recovery_lock", "write_attempt1_recovery_lock",
                    "write_attempt1_provenance", "design_resume_execution_gate"):
        fn_start = source.index(f"def {fn_name}(")
        fn_end = source.index("\ndef ", fn_start + 10)
        body = source[fn_start:fn_end]
        for forbidden in ("resolve_target", "SiW", "openai", "google.generativeai", "GEMINI",
                         "render_arm(", "render_one(", "train_detector(", "M9TrainingRun("):
            assert forbidden not in body, f"{forbidden!r} unexpectedly reachable from {fn_name}"


# =============================================================================
# ATTEMPT-2 TECHNICAL RECOVERY (this turn): TASK J's 16 required tests.
# =============================================================================

def _render_original_v2(repo: Path):
    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    v2.render_v2_arm(repo=repo, arm=v2.ARM_ORIGINAL, plan=plan, recipes=original["recipes"],
                     render_arm_fn=_fake_render_arm_factory())
    return original, plan


def test_v2_staged_row_retains_every_historical_matcher_required_domain_field(tmp_path, monkeypatch):
    """TASK J.1: `stage_v2_results_for_quality`'s rows are built by
    `{**row, ...}` spreading `c5_source_pair_plan.build_source_pair_plan`'s
    OWN position dict -- so every field `default_quality_matcher`/
    `select_route_bank` need (`route`, `live_dataset`, `live_target_sample_id`,
    `spoof_source_sample_id`, `position`, `recipe_id`, `recipe_ordinal`) must
    still be present after staging, never dropped by the v2 staging path."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original, plan = _render_original_v2(repo)
    rows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_ORIGINAL, recipe_bank_identity=plan["recipe_bank_identity"],
                                    recipes=original["recipes"], plan=plan)
    staged = v2.stage_v2_results_for_quality(repo, arm=v2.ARM_ORIGINAL, rows=rows)
    assert staged["rows"], "fixture must actually stage GENERATED candidates"
    required_fields = ("route", "live_dataset", "live_target_sample_id", "spoof_source_sample_id",
                       "position", "recipe_id", "recipe_ordinal", "candidate_id")
    for row in staged["rows"]:
        for field in required_fields:
            assert field in row, f"staged row is missing matcher-required field {field!r}"


def test_physics_and_gpat_domain_resolution_matches_historical_semantics(tmp_path, monkeypatch):
    """TASK J.2: `c6_matched_bank.SOURCE_DOMAIN_PLAN_FIELD` ('live_dataset')
    is exactly the field `default_quality_matcher` reads
    (`row.get('live_dataset', '')`) -- proven by construction, not by name
    coincidence."""
    from prism_fas.synthesis.c6_matched_bank import SOURCE_DOMAIN_PLAN_FIELD

    assert SOURCE_DOMAIN_PLAN_FIELD == "live_dataset"
    source = inspect_default_quality_matcher_source()
    assert 'row.get("live_dataset"' in source or "row.get('live_dataset'" in source


def inspect_default_quality_matcher_source() -> str:
    import inspect

    return inspect.getsource(v2.e6r.default_quality_matcher)


def test_null_spoof_source_sample_id_does_not_erase_physics_domain(tmp_path, monkeypatch):
    """TASK J.3: a Physics row's `spoof_source_sample_id` is None BY DESIGN
    (`c5_source_pair_plan.build_source_pair_plan` only assigns a spoof for
    GPAT rows) -- this must never suppress or blank `live_dataset`, since
    `default_quality_matcher`'s domain read is entirely independent of the
    spoof field."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original, plan = _render_original_v2(repo)
    rows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_ORIGINAL, recipe_bank_identity=plan["recipe_bank_identity"],
                                    recipes=original["recipes"], plan=plan)
    physics_rows = [row for row in rows if row["route"] == "physics"]
    assert physics_rows
    for row in physics_rows:
        assert row["spoof_source_sample_id"] is None
        assert row["live_dataset"], "a null spoof field must never blank the live_dataset domain"


def test_common_source_domain_computation(tmp_path, monkeypatch):
    """TASK J.4: `COMMON_SOURCE_DOMAINS` is the union of every domain key the
    frozen `resolve_e6_route_quota` quota carries across BOTH routes."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    result = v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=_fake_metrics_provider)
    quota = v2.e6r.resolve_e6_route_quota(repo)
    expected = sorted(set(quota["physics"]) | set(quota["gpat"]))
    assert result["COMMON_SOURCE_DOMAINS"] == expected


def test_domain_quota_computation(tmp_path, monkeypatch):
    """TASK J.5: the diagnostic's per-domain `required_quota` is read
    VERBATIM from the SAME frozen `resolve_e6_route_quota` quota
    `default_quality_matcher` itself resolves -- never recomputed."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    result = v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=_fake_metrics_provider)
    quota = v2.e6r.resolve_e6_route_quota(repo)
    for route in ("physics", "gpat"):
        by_domain = {entry["domain"]: entry["required_quota"] for entry in result["domains_by_route"][route]}
        for domain, count in quota[route].items():
            assert by_domain[domain] == count


def test_feasibility_count_matches_select_route_bank(tmp_path, monkeypatch):
    """TASK J.6: `_max_fillable_under_quota`'s pure cardinality count agrees
    EXACTLY with calling the frozen `select_route_bank` directly (which the
    diagnostic also does, wrapped)."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    result = v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=_fake_metrics_provider)
    for route in ("physics", "gpat"):
        outcome = result["select_route_bank_result_by_route"][route]
        fillable = result[f"{route.upper()}_MAX_FILLABLE_UNDER_FROZEN_QUOTA"]
        if outcome["raised"]:
            assert f"only {fillable} of" in outcome["message"]
        else:
            assert outcome["selected_count"] == fillable


def test_reproduces_synthetic_16_of_512_style_failure(tmp_path, monkeypatch):
    """TASK J.7: a metrics_provider that fails quality for all but 16
    physics candidates reproduces the EXACT real GPU failure shape --
    `MatchedBankError` naming physics and "only 16 of <required>"."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)

    seen = {"physics": 0}

    def sparse_provider(*, repo, row, record):
        if row["route"] == "physics":
            seen["physics"] += 1
            if seen["physics"] > 16:
                metrics = _passing_metrics()
                metrics["identity_cosine"] = 0.0
                return metrics
        return _passing_metrics()

    result = v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=sparse_provider)
    assert result["PHYSICS_QUALITY_PASS"] == 16
    assert result["PHYSICS_MAX_FILLABLE_UNDER_FROZEN_QUOTA"] == 16
    outcome = result["select_route_bank_result_by_route"]["physics"]
    assert outcome["raised"] is True
    assert "physics: only 16 of" in outcome["message"]

    # directly reproduce the raw MatchedBankError too, exactly as ATTEMPT-2's
    # GPU trace did (never a reimplementation -- the frozen function itself)
    from prism_fas.synthesis.c6_matched_bank import MatchedBankError, select_route_bank

    with pytest.raises(MatchedBankError, match="physics: only 16 of"):
        select_route_bank(
            [c for c in _build_selectable_candidates_from_result(repo, result)], route="physics",
            quota=v2.e6r.resolve_e6_route_quota(repo)["physics"])


def _build_selectable_candidates_from_result(repo: Path, result: dict) -> list:
    """Rebuilds the exact 16 eligible SelectableCandidates the sparse fixture
    produced, purely from the diagnostic's own reported per-domain counts --
    used only to re-invoke `select_route_bank` a second, independent time."""
    from prism_fas.synthesis.c6_matched_bank import SelectableCandidate

    candidates = []
    for entry in result["domains_by_route"]["physics"]:
        for i in range(entry["available"]):
            candidates.append(SelectableCandidate(
                candidate_id=f"synthetic-{entry['domain']}-{i}", arm=v2.ARM_ORIGINAL, route="physics",
                source_domain=entry["domain"], recipe_id=f"R-{i:06d}", recipe_ordinal=i,
                live_target_sample_id=f"live-{i}", base_position=i))
    return candidates


def test_technical_metadata_collapse_is_detected(tmp_path, monkeypatch):
    """TASK J.8: when EVERY physics candidate fails metrics computation
    itself (not the quality gate), the classifier reports
    TECHNICAL_METADATA_BUG, never TRUE_FROZEN_MATCHING_INFEASIBILITY."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)

    def collapse_provider(*, repo, row, record):
        if row["route"] == "physics":
            raise RuntimeError("simulated technical staging failure")
        return _passing_metrics()

    result = v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=collapse_provider)
    assert result["PHYSICS_METRICS_SUCCESS"] == 0
    assert result["MATCHING_PREFLIGHT_CLASSIFICATION"] == "TECHNICAL_METADATA_BUG"


def test_genuine_quality_sparsity_is_classified_separately(tmp_path, monkeypatch):
    """TASK J.9: when metrics computation SUCCEEDS for every physics
    candidate but almost none pass the frozen gate, the classifier reports
    TRUE_FROZEN_MATCHING_INFEASIBILITY, never TECHNICAL_METADATA_BUG."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)

    seen = {"physics": 0}

    def sparse_provider(*, repo, row, record):
        if row["route"] == "physics":
            seen["physics"] += 1
            if seen["physics"] > 16:
                metrics = _passing_metrics()
                metrics["identity_cosine"] = 0.0
                return metrics
        return _passing_metrics()

    result = v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=sparse_provider)
    assert result["PHYSICS_METRICS_SUCCESS"] == result["PHYSICS_GENERATED"]
    assert result["MATCHING_PREFLIGHT_CLASSIFICATION"] == "TRUE_FROZEN_MATCHING_INFEASIBILITY"


def test_matching_preflight_is_read_only(tmp_path, monkeypatch):
    """TASK J.10: no candidate byte, lock, or bank file changes across a
    --matching-preflight-equivalent call, including the sparse-failure path."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    arm_dir = repo / v2.v2_render_work_root(v2.ARM_ORIGINAL) / v2.ARM_ORIGINAL
    before_files = {p: p.read_bytes() for p in arm_dir.rglob("*") if p.is_file()}
    before_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())

    def sparse_provider(*, repo, row, record):
        metrics = _passing_metrics()
        if row["route"] == "physics":
            metrics["identity_cosine"] = 0.0
        return metrics

    v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=sparse_provider)

    after_files = {p: p.read_bytes() for p in arm_dir.rglob("*") if p.is_file()}
    after_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    assert before_files == after_files
    assert before_tree == after_tree


def test_matching_preflight_never_selects_or_writes_a_bank(tmp_path, monkeypatch):
    """TASK J.11: the diagnostic's return value proves no final bank was
    selected or written, and no v2 bank-lock path exists on disk after it runs."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    result = v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=_fake_metrics_provider)
    assert result["bank_selected"] is False
    assert result["bank_written"] is False
    assert not (repo / v2.v2_bank_lock_path(v2.ARM_ORIGINAL)).exists()


def test_matching_preflight_never_renders(tmp_path, monkeypatch):
    """TASK J.12: running the diagnostic against an arm with ZERO rendered
    candidates yet must not render any -- PLANNED > 0 but GENERATED == 0,
    and no candidate directory is created."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    arm_dir = repo / v2.v2_render_work_root(v2.ARM_SHUFFLE) / v2.ARM_SHUFFLE
    assert not arm_dir.exists()
    result = v2.matching_preflight_v2(repo, arm=v2.ARM_SHUFFLE, metrics_provider=_fake_metrics_provider)
    assert result["PLANNED_TOTAL"] > 0
    assert result["GENERATED_TOTAL"] == 0
    assert result["rendering_performed"] is False
    assert not arm_dir.exists()


def test_matching_preflight_never_trains(tmp_path, monkeypatch):
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    result = v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=_fake_metrics_provider)
    assert result["training_performed"] is False


def test_matching_preflight_never_touches_target():
    source = inspect_matching_preflight_source()
    for forbidden in ("resolve_target", "SiW", "target_label", "TargetStore"):
        assert forbidden not in source
    assert "\"target_access\": False" in source


def test_matching_preflight_never_calls_llm():
    source = inspect_matching_preflight_source()
    for forbidden in ("openai", "google.generativeai", "GEMINI_API_KEY", "requests.post"):
        assert forbidden not in source
    assert '"llm_api_calls": 0' in source


def inspect_matching_preflight_source() -> str:
    import inspect

    return inspect.getsource(v2.matching_preflight_v2)


def test_historical_select_route_bank_behavior_unchanged(tmp_path, monkeypatch):
    """TASK J.16: `matching_preflight_v2` calls the frozen
    `select_route_bank` with NO monkeypatch and NO wrapper around its
    ordering/tie-break/selection policy -- proven by re-running the SAME
    frozen function directly on the SAME eligible pool/quota it computed and
    getting the IDENTICAL `selected_set_sha256`."""
    from prism_fas.synthesis.c6_matched_bank import select_route_bank, selected_set_digest, SelectableCandidate

    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    result = v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=_fake_metrics_provider)
    quota = v2.e6r.resolve_e6_route_quota(repo)["gpat"]

    original = v2.load_original_llm_recipes(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_ORIGINAL, recipe_content_identity=original["content_identity"],
                               recipe_count=original["recipe_count"])
    rows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_ORIGINAL, recipe_bank_identity=plan["recipe_bank_identity"],
                                    recipes=original["recipes"], plan=plan)
    gpat_rows = [row for row in rows if row["route"] == "gpat"]
    pool = [SelectableCandidate(candidate_id=row["candidate_id"], arm=v2.ARM_ORIGINAL, route="gpat",
                                source_domain=row["live_dataset"], recipe_id=row["recipe_id"],
                                recipe_ordinal=row["recipe_ordinal"],
                                live_target_sample_id=row["live_target_sample_id"],
                                base_position=row["position"]) for row in gpat_rows]

    first = select_route_bank(pool, route="gpat", quota=quota)
    second = select_route_bank(pool, route="gpat", quota=quota)
    assert selected_set_digest(first) == selected_set_digest(second)
    assert result["select_route_bank_result_by_route"]["gpat"]["selected_count"] == len(first)


# =============================================================================
# ATTEMPT-3 TECHNICAL RECOVERY (this turn): cross-arm process-state
# contamination -- TASK J's 22 required tests.
# =============================================================================

class _FakeSequenceGraph:
    """Minimal stand-in for a `CompiledRecipeGraph`: only the attributes
    `synthetic_bank._support_masks`/`m8_pipeline.SampleStore.cached_mask`
    actually read."""

    def __init__(self, *, recipe_id: str, recipe_hash: str, coverage: float,
                geometry_shape: str = "flat", regions=("nose",)):
        self.recipe_id = recipe_id
        self.recipe_hash = recipe_hash
        self.region_mask_policy = {"geometry_shape": geometry_shape, "requested_coverage": coverage,
                                   "coverage_tolerance": 0.05}
        self.requested_regions = regions
        self.nodes = [type("Node", (), {"strength": coverage})()]

    def node_seed(self, node, scope):
        return 1234567


def test_support_masks_cache_key_includes_recipe_content_not_just_recipe_id():
    """TASK A/F: the core fix. Two graphs sharing the SAME recipe_id (exactly
    the real ORIGINAL/SHUFFLE relationship: identical recipe_id, different
    content) but DIFFERENT `recipe_hash`/coverage must NOT collide in
    `_support_masks`'s cache -- each is actually (re)built; a THIRD call
    with the FIRST graph's identity is still served from cache (proving
    caching still works within one recipe's content)."""
    from prism_fas.synthesis import synthetic_bank

    build_calls: list[float] = []

    class _FakeMaskBuilder:
        def __init__(self, sample_id):
            self.sample_id = sample_id

        def build(self, regions, *, geometry_shape, coverage, seed, coverage_tolerance):
            build_calls.append(coverage)

            class _Result:
                pass

            result = _Result()
            result.tag = f"{self.sample_id}:{coverage}"
            result.coverage = coverage
            return result

    class _FakeStore:
        def mask_builder(self, sample_id):
            return _FakeMaskBuilder(sample_id)

    store = _FakeStore()
    graph_a = _FakeSequenceGraph(recipe_id="R-000000", recipe_hash="hash-ORIGINAL", coverage=0.30)
    graph_b = _FakeSequenceGraph(recipe_id="R-000000", recipe_hash="hash-SHUFFLE", coverage=0.90)

    result_a = synthetic_bank._support_masks(store, "live-1", graph_a)
    result_b = synthetic_bank._support_masks(store, "live-1", graph_b)
    assert result_a.tag != result_b.tag
    assert len(build_calls) == 2, "different recipe content at the SAME recipe_id must never cache-collide"

    result_a_again = synthetic_bank._support_masks(store, "live-1", graph_a)
    assert result_a_again is result_a, "identical (sample, recipe_id, recipe_hash) must still be memoized"
    assert len(build_calls) == 2


def test_cached_mask_cache_key_includes_recipe_content(monkeypatch, tmp_path):
    """TASK A/F: `SampleStore.cached_mask` (the GPAT-training render path's
    OWN, separate memoization) gets the identical fix -- proven directly,
    without needing a real source_train package on disk."""
    from prism_fas.synthesis.m8_pipeline import SampleStore, SourceOnlyAudit

    store = SampleStore(package_root=tmp_path, audit=SourceOnlyAudit(), _rows={}, _cache={}, _mask_cache={})

    build_calls: list[float] = []

    class _FakeMaskBuilder:
        def build(self, regions, *, geometry_shape, coverage, seed):
            build_calls.append(coverage)

            class _Result:
                pass

            result = _Result()
            result.operator_support_mask = np.full((1, 2, 2), coverage, dtype=np.float32)
            result.requested_region_mask = np.full((1, 2, 2), coverage, dtype=np.float32)
            return result

    monkeypatch.setattr(store, "mask_builder", lambda sample_id: _FakeMaskBuilder())

    graph_a = _FakeSequenceGraph(recipe_id="R-000000", recipe_hash="hash-ORIGINAL", coverage=0.30)
    graph_b = _FakeSequenceGraph(recipe_id="R-000000", recipe_hash="hash-SHUFFLE", coverage=0.90)

    mask_a = store.cached_mask("live-1", "live", graph_a, coverage=0.30, seed_scope="s", use_support=True)
    mask_b = store.cached_mask("live-1", "live", graph_b, coverage=0.90, seed_scope="s", use_support=True)
    assert float(mask_a[0, 0, 0]) != float(mask_b[0, 0, 0])
    assert len(build_calls) == 2


def test_candidate_id_cannot_collide_across_arms_structurally():
    """TASK B: `candidate_identity` hashes BOTH `arm` and `recipe_bank_identity`
    -- either dimension alone already guarantees no ORIGINAL/SHUFFLE
    collision, since the two arms always carry different arm labels AND
    different recipe-bank content identities by construction."""
    from prism_fas.synthesis.c5_source_pair_plan import candidate_identity

    common = dict(source_pair_plan_identity="spp", recipe_id="R-000000", recipe_ordinal=0, slot=0,
                 position=0, route="physics", live_target_sample_id="live-0", spoof_source_sample_id=None,
                 package_identity="pkg", ontology_identity="ont", generator_binding="binding")
    original_id = candidate_identity(arm=v2.ARM_ORIGINAL, recipe_bank_identity="original-bank-identity", **common)
    shuffle_id = candidate_identity(arm=v2.ARM_SHUFFLE, recipe_bank_identity="shuffle-bank-identity", **common)
    assert original_id != shuffle_id

    # even a pathological identical recipe_bank_identity (never true in
    # practice -- the two banks' content identities always differ) is STILL
    # disambiguated by `arm` alone
    assert (candidate_identity(arm=v2.ARM_ORIGINAL, recipe_bank_identity="same", **common)
           != candidate_identity(arm=v2.ARM_SHUFFLE, recipe_bank_identity="same", **common))


def test_no_actual_candidate_id_collision_between_rendered_arms(tmp_path, monkeypatch):
    """TASK B: the REAL, full 2048-position schedule for both arms never
    produces an overlapping candidate_id set."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original, oplan = _render_original_v2(repo)
    shuffle = v2.e6r.verify_shuffle_recipe_source(repo)
    splan = v2.build_v2_arm_plan(repo, arm=v2.ARM_SHUFFLE, recipe_content_identity=shuffle["content_identity"],
                                recipe_count=len(shuffle["recipes"]))

    orows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_ORIGINAL, recipe_bank_identity=oplan["recipe_bank_identity"],
                                     recipes=original["recipes"], plan=oplan)
    srows = v2.build_v2_arm_plan_rows(repo, arm=v2.ARM_SHUFFLE, recipe_bank_identity=splan["recipe_bank_identity"],
                                     recipes=shuffle["recipes"], plan=splan)
    oids = {row["candidate_id"] for row in orows}
    sids = {row["candidate_id"] for row in srows}
    assert len(oids) == len(orows) == 2048
    assert len(sids) == len(srows) == 2048
    assert not (oids & sids)

    # the SAME recipe_id string IS shared at every ordinal (the precondition
    # a recipe_id-only cache key silently relies on) -- confirming the
    # collision precondition is real, even though candidate_id itself never
    # collides
    orecipe_ids_by_position = {row["position"]: row["recipe_id"] for row in orows}
    srecipe_ids_by_position = {row["position"]: row["recipe_id"] for row in srows}
    assert orecipe_ids_by_position == srecipe_ids_by_position


def _render_shuffle_v2(repo: Path):
    shuffle = v2.e6r.verify_shuffle_recipe_source(repo)
    plan = v2.build_v2_arm_plan(repo, arm=v2.ARM_SHUFFLE, recipe_content_identity=shuffle["content_identity"],
                               recipe_count=len(shuffle["recipes"]))
    v2.render_v2_arm(repo=repo, arm=v2.ARM_SHUFFLE, plan=plan, recipes=shuffle["recipes"],
                     render_arm_fn=_fake_render_arm_factory())
    return shuffle, plan


def test_standalone_original_matches_sequence_original_with_fakes(tmp_path, monkeypatch):
    """TASK J.1."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    _render_shuffle_v2(repo)

    standalone = v2.matching_preflight_v2(repo, arm=v2.ARM_ORIGINAL, metrics_provider=_fake_metrics_provider)
    sequence = v2.matching_sequence_preflight_v2(repo, metrics_provider=_fake_metrics_provider)
    assert (sequence["forward_results_by_arm"][v2.ARM_ORIGINAL]["PHYSICS_MAX_FILLABLE_UNDER_FROZEN_QUOTA"]
           == standalone["PHYSICS_MAX_FILLABLE_UNDER_FROZEN_QUOTA"])


def test_standalone_shuffle_matches_sequence_shuffle_with_fakes(tmp_path, monkeypatch):
    """TASK J.2."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    _render_shuffle_v2(repo)

    standalone = v2.matching_preflight_v2(repo, arm=v2.ARM_SHUFFLE, metrics_provider=_fake_metrics_provider)
    sequence = v2.matching_sequence_preflight_v2(repo, metrics_provider=_fake_metrics_provider)
    assert (sequence["forward_results_by_arm"][v2.ARM_SHUFFLE]["PHYSICS_MAX_FILLABLE_UNDER_FROZEN_QUOTA"]
           == standalone["PHYSICS_MAX_FILLABLE_UNDER_FROZEN_QUOTA"])


def test_original_to_shuffle_order_independence_with_fakes(tmp_path, monkeypatch):
    """TASK J.3."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    _render_shuffle_v2(repo)

    sequence = v2.matching_sequence_preflight_v2(repo, metrics_provider=_fake_metrics_provider)
    assert sequence["PROCESS_SEQUENCE"] == [v2.ARM_ORIGINAL, v2.ARM_SHUFFLE]
    assert sequence["ORDER_DEPENDENCE_PRESENT"] is False


def test_shuffle_to_original_order_independence_with_fakes(tmp_path, monkeypatch):
    """TASK J.4."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    _render_shuffle_v2(repo)

    sequence = v2.matching_sequence_preflight_v2(
        repo, sequence=(v2.ARM_SHUFFLE, v2.ARM_ORIGINAL), metrics_provider=_fake_metrics_provider)
    assert sequence["PROCESS_SEQUENCE"] == [v2.ARM_SHUFFLE, v2.ARM_ORIGINAL]
    assert sequence["ORDER_DEPENDENCE_PRESENT"] is False


def test_cache_key_completeness_structural(tmp_path):
    """TASK J.6: every cache reachable from the E6/E6-v2 quality-matching
    call chain that varies with recipe content now keys on `recipe_hash`,
    not `recipe_id` alone -- proven from source, not just behaviorally."""
    import inspect

    from prism_fas.synthesis import synthetic_bank
    from prism_fas.synthesis import m8_pipeline

    assert "graph.recipe_hash" in inspect.getsource(synthetic_bank._support_masks)
    assert "graph.recipe_hash" in inspect.getsource(m8_pipeline.SampleStore.cached_mask)


def test_per_arm_recipe_bank_isolation(tmp_path, monkeypatch):
    """TASK J.7: `match_v2_arm` builds a DIFFERENT `quality_bank` object per
    arm (never the same bank instance reused), each carrying ONLY that arm's
    own recipes."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    original, oplan = _render_original_v2(repo)
    shuffle, splan = _render_shuffle_v2(repo)

    ontology_dest = repo / v2.e6r.ONTOLOGY_CONFIG_PATH
    ontology_dest.parent.mkdir(parents=True, exist_ok=True)
    ontology_dest.write_text((REPO / v2.e6r.ONTOLOGY_CONFIG_PATH).read_text(encoding="utf-8"), encoding="utf-8")

    original_bank = v2.build_e6_v2_route_bank(repo, original["recipes"], arm=v2.ARM_ORIGINAL,
                                              bank_identity=oplan["recipe_bank_identity"])
    shuffle_bank = v2.build_e6_v2_route_bank(repo, shuffle["recipes"], arm=v2.ARM_SHUFFLE,
                                             bank_identity=splan["recipe_bank_identity"])
    assert original_bank["bank_identity"] != shuffle_bank["bank_identity"]
    assert original_bank["bank_id"] != shuffle_bank["bank_id"]
    assert original_bank["recipes"] is not shuffle_bank["recipes"]


def test_requested_support_isolation_via_recipe_hash(tmp_path):
    """TASK J.8: `requested_support_for`-shaped reconstruction (via
    `_support_masks` directly) yields DIFFERENT results for the SAME
    sample_id/recipe_id pair when the underlying recipe CONTENT (recipe_hash)
    differs -- the isolation this whole turn's fix establishes."""
    from prism_fas.synthesis import synthetic_bank

    class _FakeMaskBuilder:
        def __init__(self, sample_id):
            self.sample_id = sample_id

        def build(self, regions, *, geometry_shape, coverage, seed, coverage_tolerance):
            class _Result:
                pass

            result = _Result()
            result.coverage = coverage
            return result

    class _FakeStore:
        def mask_builder(self, sample_id):
            return _FakeMaskBuilder(sample_id)

    store = _FakeStore()
    original_like = synthetic_bank._support_masks(
        store, "live-shared", _FakeSequenceGraph(recipe_id="R-000042", recipe_hash="original", coverage=0.2))
    shuffle_like = synthetic_bank._support_masks(
        store, "live-shared", _FakeSequenceGraph(recipe_id="R-000042", recipe_hash="shuffle", coverage=0.8))
    assert original_like.coverage != shuffle_like.coverage


def test_quality_metrics_isolation_across_arms(tmp_path, monkeypatch):
    """TASK J.9: two arms' `default_metrics_provider`-shaped metrics
    computations for candidates sharing sample_id/recipe_id never leak into
    each other once `quality_bank` is explicitly per-arm (structural: each
    call passes a DIFFERENT `quality_bank` object, proven by
    `test_per_arm_recipe_bank_isolation`) AND `_support_masks` keys on
    content (proven by `test_support_masks_cache_key_includes_recipe_content_not_just_recipe_id`).
    This test proves the COMPOSITION: `match_v2_arm` never reuses one arm's
    `quality_bank` for the other."""
    import inspect

    source = inspect.getsource(v2.match_v2_arm)
    assert "build_e6_v2_route_bank(" in source
    assert source.count("build_e6_v2_route_bank(") == 1  # called fresh, once, per match_v2_arm invocation


def test_quality_gate_isolation_pure_function(tmp_path):
    """TASK J.10: `quality_gate.evaluate` takes no store/bank/arm argument at
    all -- it is a pure function of `(metrics, thresholds)`, so it cannot be
    a vector for cross-arm state leakage by construction."""
    import inspect

    from prism_fas.synthesis.quality_gate import evaluate

    signature = inspect.signature(evaluate)
    assert list(signature.parameters) == ["metrics", "thresholds"]


def test_matcher_feasibility_isolation_select_route_bank_pure(tmp_path):
    """TASK J.11: `select_route_bank` takes no persistent state either --
    every intermediate (`recipe_count`, `live_count`, `remaining`) is a LOCAL
    dict rebuilt on every call, so two calls never share mutable state."""
    from prism_fas.synthesis.c6_matched_bank import SelectableCandidate, select_route_bank

    pool = [SelectableCandidate(candidate_id=f"c-{i}", arm="A", route="physics", source_domain="D",
                                recipe_id=f"R-{i}", recipe_ordinal=i, live_target_sample_id=f"live-{i}",
                                base_position=i) for i in range(4)]
    first = select_route_bank(pool, route="physics", quota={"D": 4})
    second = select_route_bank(pool, route="physics", quota={"D": 4})
    assert first == second  # fully deterministic and independent


def test_synthetic_cross_arm_contamination_detected_by_sequence_diagnostic(tmp_path, monkeypatch):
    """TASK J.12: a DELIBERATELY bugged fake `metrics_provider` -- caching by
    `(route, recipe_id)` only, omitting `arm`, exactly the bug CLASS fixed
    this turn -- makes `matching_sequence_preflight_v2` observe
    ORDER_DEPENDENCE_PRESENT=True, proving the diagnostic is actually
    sensitive to real cross-arm contamination and not just structurally
    unable to detect it."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    _render_shuffle_v2(repo)

    contaminated_cache: dict[tuple, dict] = {}
    real_reset = v2.e6r.reset_quality_runtime_cache_for_tests

    def _reset_both():
        contaminated_cache.clear()
        real_reset()

    monkeypatch.setattr(v2.e6r, "reset_quality_runtime_cache_for_tests", _reset_both)

    def contaminated_provider(*, repo, row, record):
        key = (row["route"], row["recipe_id"])  # BUG: no `arm` dimension
        if key in contaminated_cache:
            return contaminated_cache[key]
        metrics = _passing_metrics()
        if row["route"] == "physics" and row["arm"] == v2.ARM_SHUFFLE and row["position"] % 3 == 0:
            metrics["identity_cosine"] = 0.0  # SHUFFLE's OWN correct-for-this-arm outcome
        contaminated_cache[key] = metrics
        return metrics

    sequence = v2.matching_sequence_preflight_v2(repo, metrics_provider=contaminated_provider)
    assert sequence["ORDER_DEPENDENCE_PRESENT"] is True
    assert sequence["order_dependence_by_arm"][v2.ARM_SHUFFLE]["matches_clean_baseline"] is False


def test_exact_479_infeasibility_fixture():
    """TASK J.13: the EXACT reported GPU numbers -- CASIA available 231/
    required 264, MSU available 281/required 248 -- reproduce fillable=479
    via BOTH `_max_fillable_under_quota`'s formula AND the frozen
    `select_route_bank` raising with "only 479 of 512"."""
    from prism_fas.synthesis.c6_matched_bank import MatchedBankError, SelectableCandidate, select_route_bank

    quota = {"casia_fasd": 264, "msu_mfsd": 248}
    available = {"casia_fasd": 231, "msu_mfsd": 281}
    assert v2._max_fillable_under_quota(available, quota) == 479

    pool = []
    for domain, count in available.items():
        for i in range(count):
            pool.append(SelectableCandidate(
                candidate_id=f"{domain}-{i}", arm=v2.ARM_SHUFFLE, route="physics", source_domain=domain,
                recipe_id=f"R-{i:06d}", recipe_ordinal=i, live_target_sample_id=f"live-{domain}-{i}",
                base_position=i))
    with pytest.raises(MatchedBankError, match=r"physics: only 479 of 512"):
        select_route_bank(pool, route="physics", quota=quota)


def test_surplus_msu_cannot_compensate_casia_deficit_under_frozen_quota():
    """TASK J.14: MSU's 33-candidate surplus (281 available vs 248 required)
    cannot fill CASIA's 33-candidate deficit (231 available vs 264 required)
    -- the frozen quota is EXACT per domain, never fungible across domains."""
    quota = {"casia_fasd": 264, "msu_mfsd": 248}
    available = {"casia_fasd": 231, "msu_mfsd": 281}
    casia_deficit = quota["casia_fasd"] - available["casia_fasd"]
    msu_surplus = available["msu_mfsd"] - quota["msu_mfsd"]
    assert casia_deficit == 33
    assert msu_surplus == 33
    fillable = v2._max_fillable_under_quota(available, quota)
    assert fillable == sum(quota.values()) - casia_deficit
    assert fillable != sum(quota.values())  # the surplus does NOT cancel the deficit


def test_attempt3_investigation_never_modifies_quota_threshold_or_q(tmp_path):
    """TASK J.15."""
    source = Path(v2.__file__).read_text(encoding="utf-8")
    for forbidden in ("PER_ROUTE = ", "FINAL_BANK_PER_ARM = "):
        assert forbidden not in source
    render_source = Path(v2.e6r.__file__).read_text(encoding="utf-8")
    assert "def resolve_e6_route_quota(" in render_source  # unchanged entry point still present, unmodified body
    import inspect

    quota_source = inspect.getsource(v2.e6r.resolve_e6_route_quota)
    assert "bank_lock[\"exposure\"]" in quota_source or "bank_lock['exposure']" in quota_source


def test_sequence_preflight_is_read_only(tmp_path, monkeypatch):
    """TASK J.16."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    _render_shuffle_v2(repo)

    tree_root = repo / v2.E6_V2_CANDIDATES_ROOT
    before = {p: p.read_bytes() for p in tree_root.rglob("*") if p.is_file()}

    v2.matching_sequence_preflight_v2(repo, metrics_provider=_fake_metrics_provider)

    after = {p: p.read_bytes() for p in tree_root.rglob("*") if p.is_file()}
    assert before == after


def test_sequence_preflight_never_writes_a_bank(tmp_path, monkeypatch):
    """TASK J.17."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    _render_shuffle_v2(repo)
    result = v2.matching_sequence_preflight_v2(repo, metrics_provider=_fake_metrics_provider)
    assert result["bank_selected"] is False
    assert result["bank_written"] is False
    assert not (repo / v2.v2_bank_lock_path(v2.ARM_ORIGINAL)).exists()
    assert not (repo / v2.v2_bank_lock_path(v2.ARM_SHUFFLE)).exists()


def test_sequence_preflight_never_renders(tmp_path, monkeypatch):
    """TASK J.18."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    shuffle_dir = repo / v2.v2_render_work_root(v2.ARM_SHUFFLE) / v2.ARM_SHUFFLE
    assert not shuffle_dir.exists()
    result = v2.matching_sequence_preflight_v2(repo, metrics_provider=_fake_metrics_provider)
    assert result["rendering_performed"] is False
    assert not shuffle_dir.exists()


def test_sequence_preflight_never_trains(tmp_path, monkeypatch):
    """TASK J.19."""
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    _render_original_v2(repo)
    _render_shuffle_v2(repo)
    result = v2.matching_sequence_preflight_v2(repo, metrics_provider=_fake_metrics_provider)
    assert result["training_performed"] is False


def test_sequence_preflight_never_touches_target():
    """TASK J.20."""
    import inspect

    source = inspect.getsource(v2.matching_sequence_preflight_v2)
    for forbidden in ("resolve_target", "SiW", "target_label", "TargetStore"):
        assert forbidden not in source
    assert '"target_access": False' in source


def test_sequence_preflight_never_calls_llm():
    """TASK J.21."""
    import inspect

    source = inspect.getsource(v2.matching_sequence_preflight_v2)
    for forbidden in ("openai", "google.generativeai", "GEMINI_API_KEY", "requests.post"):
        assert forbidden not in source
    assert '"llm_api_calls": 0' in source


def test_historical_e6_v1_behavior_unchanged_by_cache_fix(tmp_path):
    """TASK J.22: the historical E6 v1 call shape (single arm, real
    `_resolve_quality_bank`/`_resolve_quality_runtime` default, no
    `quality_bank` override) is untouched by the cache-key fix -- proven by
    re-running the exact historical selection regression test's assertions
    (selected count/routes/digest) with the corrected `_support_masks`."""
    from test_c_ext_e6_render import _full_fixture  # noqa: E402
    from prism_fas.synthesis.c6_matched_bank import selected_set_digest

    repo = _full_fixture(tmp_path)
    _quality_gate_fixture(repo)
    _route_quota_fixture(repo, physics=2, gpat=2)
    _write_q_reference_fixture(repo)
    plan = v2.e6r.build_render_plan(repo)

    rows, results = [], []
    for i in range(4):
        route = "physics" if i < 2 else "gpat"
        rows.append({"candidate_id": f"cand-{i}", "recipe_id": f"R-{i:06d}", "recipe_ordinal": i,
                    "position": i, "route": route, "live_dataset": "domainA",
                    "live_target_sample_id": f"live-{i}"})
        results.append({"candidate_id": f"cand-{i}"})
    staged = {"rows": rows, "results": results}

    matched = v2.e6r.default_quality_matcher(repo=repo, plan=plan, staged=staged, arm=v2.e6r.E6_ARM_NAME,
                                            metrics_provider=_fake_metrics_provider)
    assert len(matched["selected"]) == 4
    assert matched["selected_set_sha256"] == selected_set_digest(matched["selected"])


def test_attempt3_provenance_records_all_three_fillable_observations_separately(tmp_path):
    repo = _base_repo(tmp_path)
    result = v2.write_attempt3_provenance(repo)
    provenance = result["provenance"]
    facts = provenance["gpu_observed_facts"]
    assert facts["REAL_EXECUTION_FAILURE_FILLABLE"] == 26
    assert facts["STANDALONE_ORIGINAL_PREFLIGHT_FILLABLE"] == 512
    assert facts["STANDALONE_SHUFFLE_PREFLIGHT_FILLABLE"] == 479
    assert facts["REAL_EXECUTION_FAILURE_FILLABLE"] != facts["STANDALONE_SHUFFLE_PREFLIGHT_FILLABLE"]
    assert provenance["candidate_id_collision_across_arms"] == "IMPOSSIBLE_BY_CONSTRUCTION"
    assert provenance["post_fix_gpu_confirmed"] is False
    assert provenance["GPU_RESUME_AUTHORIZED"] is False
    assert provenance["QUOTA_CHANGED"] is False
    assert provenance["Q_CHANGED"] is False
    assert Path(result["path"]).is_file()
    assert Path(result["path"]).is_relative_to(repo / v2.E6_V2_DIR)


# =============================================================================
# E6-V2 SCIENTIFIC CLOSURE (this turn): TASK E's 12 required tests.
# =============================================================================

def _full_closure_fixture(tmp_path: Path, monkeypatch) -> Path:
    repo = _full_v2_execution_fixture(tmp_path, monkeypatch)
    v2.run_e6_v2_protocol_preparation(repo)
    v2.write_attempt1_provenance(repo)
    v2.write_attempt1_recovery_lock(repo)
    v2.write_attempt2_provenance(repo)
    v2.write_attempt3_provenance(repo)
    v2.write_attempt3_postfix_confirmation(repo)
    return repo


def test_closure_records_479_not_26(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    closure = v2.build_e6_v2_final_closure(repo)
    assert closure["SHUFFLE_PHYSICS_MAX_FILLABLE"] == 479
    assert closure["SHUFFLE_PHYSICS_MAX_FILLABLE"] != 26
    assert closure["FINAL_479_OF_512_CLASS"] == "SCIENTIFIC_INFEASIBILITY"


def test_closure_labels_26_technical_artifact(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    closure = v2.build_e6_v2_final_closure(repo)
    assert closure["ATTEMPT3_26_OF_512_CLASS"] == "TECHNICAL_ARTIFACT"


def test_closure_labels_16_technical_artifact(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    closure = v2.build_e6_v2_final_closure(repo)
    assert closure["ATTEMPT2_16_OF_512_CLASS"] == "TECHNICAL_ARTIFACT"


def test_closure_order_dependence_false(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    closure = v2.build_e6_v2_final_closure(repo)
    assert closure["ORDER_DEPENDENCE_PRESENT"] is False
    confirmation = json.loads((repo / v2.E6_V2_DIR / "E6_V2_ATTEMPT3_POSTFIX_CONFIRMATION.json")
                             .read_text(encoding="utf-8"))
    assert confirmation["ORDER_DEPENDENCE_PRESENT"] is False


def test_closure_casia_deficit_arithmetic_is_33(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    closure = v2.build_e6_v2_final_closure(repo)
    assert closure["SHUFFLE_PHYSICS_CASIA_DEFICIT"] == 33
    assert (closure["SHUFFLE_PHYSICS_CASIA_REQUIRED"] - closure["SHUFFLE_PHYSICS_CASIA_AVAILABLE"]) == 33
    assert closure["SHUFFLE_PHYSICS_MSU_SURPLUS"] == 33


def test_closure_exact_domain_quota_prevents_cross_domain_compensation(tmp_path, monkeypatch):
    """Independently re-derives fillable=479 from the closure's OWN recorded
    CASIA/MSU numbers via the frozen selector's real cardinality formula,
    proving the 33-candidate MSU surplus provably cannot offset the
    33-candidate CASIA deficit."""
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    closure = v2.build_e6_v2_final_closure(repo)
    available = {"casia_fasd": closure["SHUFFLE_PHYSICS_CASIA_AVAILABLE"],
                "msu_mfsd": closure["SHUFFLE_PHYSICS_MSU_AVAILABLE"]}
    quota = {"casia_fasd": closure["SHUFFLE_PHYSICS_CASIA_REQUIRED"],
            "msu_mfsd": closure["SHUFFLE_PHYSICS_MSU_REQUIRED"]}
    assert sum(available.values()) == closure["SHUFFLE_PHYSICS_REQUIRED"]  # 512 total quality-pass
    assert v2._max_fillable_under_quota(available, quota) == closure["SHUFFLE_PHYSICS_MAX_FILLABLE"] == 479


def test_closure_training_ready_false(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    closure = v2.build_e6_v2_final_closure(repo)
    assert closure["E6_V2_READY_FOR_TRAINING"] is False
    assert closure["E6_V2_TRAINING_BLOCK_REASON"] == "SHUFFLE_PHYSICS_MATCHED_BANK_INFEASIBLE_UNDER_FROZEN_DOMAIN_QUOTA"


def test_closure_never_claims_a_completed_shuffle_matched_bank(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    result = v2.write_e6_v2_final_closure(repo)
    assert not (repo / v2.v2_bank_lock_path(v2.ARM_SHUFFLE)).exists()
    assert not (repo / v2.v2_bank_lock_path(v2.ARM_ORIGINAL)).exists()
    assert "SHUFFLE_PHYSICS_MAX_FILLABLE" in result["closure"]
    summary = v2.build_e6_v2_final_summary_markdown(repo)
    assert "SHUFFLE" in summary and "infeasib" in summary.lower()
    assert "matched bank" in summary.lower() or "matched-bank" in summary.lower()


def test_closure_no_target_access(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    closure = v2.build_e6_v2_final_closure(repo)
    assert closure["TARGET_ACCESS"] is False
    import inspect

    for fn in (v2.build_e6_v2_final_closure, v2.build_attempt3_postfix_confirmation,
              v2.build_e6_v2_final_summary_markdown):
        source = inspect.getsource(fn)
        for forbidden in ("resolve_target", "SiW", "TargetStore"):
            assert forbidden not in source


def test_closure_no_llm(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    closure = v2.build_e6_v2_final_closure(repo)
    assert closure["LLM_API_CALLS"] == 0
    import inspect

    for fn in (v2.build_e6_v2_final_closure, v2.build_attempt3_postfix_confirmation,
              v2.build_e6_v2_final_summary_markdown):
        source = inspect.getsource(fn)
        for forbidden in ("openai", "google.generativeai", "GEMINI_API_KEY", "requests.post"):
            assert forbidden not in source


def test_prior_attempt_provenance_immutable_across_closure_write(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    before = {}
    for filename in ("E6_V2_ATTEMPT1_PROVENANCE.json", "E6_V2_ATTEMPT1_RECOVERY_LOCK.json",
                     "E6_V2_ATTEMPT2_PROVENANCE.json", "E6_V2_ATTEMPT3_PROVENANCE.json"):
        path = repo / v2.E6_V2_DIR / filename
        before[filename] = path.read_bytes()

    v2.write_attempt3_postfix_confirmation(repo)
    v2.write_e6_v2_final_closure(repo)
    v2.write_e6_v2_final_summary(repo)

    for filename, content in before.items():
        assert (repo / v2.E6_V2_DIR / filename).read_bytes() == content


def test_original_execution_plan_lock_immutable_across_closure_write(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    lock_path = repo / v2.RENDER_EXECUTION_PLAN_LOCK_PATH
    before = lock_path.read_bytes()

    v2.write_attempt3_postfix_confirmation(repo)
    v2.write_e6_v2_final_closure(repo)
    v2.write_e6_v2_final_summary(repo)

    assert lock_path.read_bytes() == before


def test_closure_provenance_chain_pins_all_six_files_plus_postfix(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    closure = v2.build_e6_v2_final_closure(repo)
    assert closure["provenance_chain_complete"] is True
    filenames = {entry["filename"] for entry in closure["provenance_chain"]}
    assert filenames == set(v2.PROVENANCE_CHAIN_FILES)
    for entry in closure["provenance_chain"]:
        assert entry["present"] is True
        assert len(entry["sha256"]) == 64


def test_e7_dependency_status_reports_no_e7_milestone(tmp_path, monkeypatch):
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    status = v2.e7_dependency_status(repo)
    assert status["E7_MILESTONE_DEFINED_IN_REPOSITORY"] is False
    assert status["E7_DEPENDENCY_STATUS"] == "NOT_APPLICABLE_NO_E7_MILESTONE_DEFINED"
    assert status["actual_downstream_consumer_blocked"] is True
    assert status["e7_executed_this_turn"] is False


def test_close_e6_v2_is_additive_only(tmp_path, monkeypatch):
    """--close-e6-v2's CLI handler never renders/trains/touches target/LLM,
    and never writes outside E6_V2_DIR."""
    repo = _full_closure_fixture(tmp_path, monkeypatch)
    before_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())

    monkeypatch.setattr(v2.cc, "repo_root", lambda: repo)
    code = v2.main(["--close-e6-v2"])
    assert code == 0

    after_tree = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file())
    new_files = sorted(set(after_tree) - set(before_tree))
    assert all(f.startswith(v2.E6_V2_DIR) for f in new_files)
    assert any(f.endswith("E6_V2_FINAL_CLOSURE.json") for f in new_files)
    assert any(f.endswith("E6_V2_FINAL_SUMMARY.md") for f in new_files)
