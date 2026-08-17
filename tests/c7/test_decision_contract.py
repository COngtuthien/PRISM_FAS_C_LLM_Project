"""The v1.5 Track-G / Track-R decision contract (§13.4, §13.5, §16.2).

§13.5 exists because Version B shipped a regional detector whose decision did not
depend on its regional branch. The guards it mandates are structural, so these
tests are structural too: they check what is instantiated, what the logit
actually depends on, and what the identity records — not what a metric says.

Two properties matter most and are tested from both directions.

**Track R's logit really depends on all three branches.** Proven by autograd
norms and by feature intervention, and separately by the fact that the inherited
`prism_noisy_or` fusion — which combines post-hoc SCORES — does *not* satisfy it.
That contrast is why `glr_concat` had to be added rather than reused.

**Nothing inherited moved.** `glr_concat` is additive. Every pre-existing variant
keeps its exact architecture identity, so the frozen M9 reference checkpoint
stays loadable, and a test asserts that rather than trusting it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prism_fas.detector import decision_audit as audit
from prism_fas.detector.trainer import M9TrainingConfig, batch_contract_for
from prism_fas.detector.variant import (GLR_BRANCH_DIM, GLR_CONCAT_WIDTH, GLR_HIDDEN_DIM,
                                        ResolvedExperimentVariant)
from prism_fas.evaluation.variant_audit import audit_batch, build_audit_detector
from prism_fas.pipeline.adapters.c7 import TRACK_G_FLAGS, TRACK_R_FLAGS, TRACK_R_K4_FLAGS


def variant(flags: dict) -> ResolvedExperimentVariant:
    return ResolvedExperimentVariant.resolve(flags)


def model_and_batch(flags: dict):
    resolved = variant(flags)
    model = build_audit_detector(resolved)
    config = M9TrainingConfig(run_id="t", variant=resolved, steps_per_epoch=2)
    return model, audit_batch(resolved, batch_contract_for("G5", config))


# --- both tracks resolve -----------------------------------------------------

def test_both_v15_tracks_are_executable() -> None:
    for flags in (TRACK_G_FLAGS, TRACK_R_FLAGS, TRACK_R_K4_FLAGS):
        executable, reason = variant(flags).executable()
        assert executable, reason


def test_the_frozen_score_names_follow_the_track() -> None:
    """§16.1 fixes the calibration logit and the thresholded score per track."""
    track_g, track_r = variant(TRACK_G_FLAGS), variant(TRACK_R_FLAGS)
    assert (track_g.decision_logit_name, track_g.decision_score_name) == \
        ("global_logit_G", "p_G")
    assert (track_r.decision_logit_name, track_r.decision_score_name) == \
        ("fused_logit_R", "p_R")
    assert track_r.decision_head_type == "track_r_glr_concat_v1"


def test_the_fusion_geometry_is_the_frozen_one() -> None:
    """§13.4.2 freezes 256-D branch summaries, 768 concat, 256 hidden."""
    assert (GLR_BRANCH_DIM, GLR_CONCAT_WIDTH, GLR_HIDDEN_DIM) == (256, 768, 256)
    payload = variant(TRACK_R_FLAGS).architecture_payload()
    assert payload["branch_summary_dim"] == 256
    assert payload["fusion_concat_width"] == 768
    assert payload["fusion_hidden_dim"] == 256
    assert payload["fusion_activation"] == "GELU"


# --- Track G is global-only --------------------------------------------------

def test_track_g_instantiates_no_local_region_manifold_or_prompt_module() -> None:
    """§13.4.1: Track G EXCLUDES them rather than computing and ignoring them."""
    model, batch = model_and_batch(TRACK_G_FLAGS)
    report = audit.audit_track_g(model, batch)
    assert report["passed"], report["checks"]
    assert report["forbidden_modules_instantiated"] == []
    assert all(report["absent_outputs"].values())


def test_track_g_score_is_not_a_fusion() -> None:
    model, batch = model_and_batch(TRACK_G_FLAGS)
    with torch.no_grad():
        output = model(batch)
    assert torch.allclose(output.s_final, output.p_global)
    assert output.s_region is None and output.p_prompt_spoof is None


# --- Track R depends on every branch -----------------------------------------

def test_the_fused_logit_has_a_nonzero_gradient_on_local_and_region() -> None:
    """§13.5: both norms must be finite and strictly greater than 1e-8."""
    model, batch = model_and_batch(TRACK_R_FLAGS)
    report = audit.autograd_dependency(model, batch)
    assert report["passed"], report["gradient_norms"]
    assert report["gradient_norms"]["local"] > audit.GRADIENT_MINIMUM
    assert report["gradient_norms"]["region_fusion"] > audit.GRADIENT_MINIMUM
    assert report["all_finite"]


def test_zeroing_or_permuting_any_branch_moves_the_logit() -> None:
    model, batch = model_and_batch(TRACK_R_FLAGS)
    report = audit.feature_intervention(model, batch)
    assert report["passed"], report["max_absolute_shift"]
    for case in ("local_zeroed", "region_zeroed", "region_permuted", "global_zeroed"):
        assert report["max_absolute_shift"][case] > audit.INTERVENTION_TOLERANCE, case


def test_the_region_branch_is_a_logit_input_not_a_post_hoc_score() -> None:
    """The distinction §13.5 turns on, asserted on both fusions."""
    glr, noisy_or = variant(TRACK_R_FLAGS), ResolvedExperimentVariant.reference()
    assert glr.region_enters_decision_logit
    assert not glr.fuses_region_evidence          # s_region is not a fusion input
    assert not noisy_or.region_enters_decision_logit
    assert noisy_or.fuses_region_evidence         # the inherited post-hoc fusion


def test_the_inherited_noisy_or_fusion_would_not_satisfy_the_guard() -> None:
    """Why a new decision head was required rather than reusing the old one.

    The reference variant computes regions and fuses `s_region` as a probability
    AFTER the logit exists. Its final logit is therefore a function of the global
    embedding alone, which is exactly the shape §13.5 declares invalid for a
    Track-R primary decision.
    """
    reference = ResolvedExperimentVariant.reference()
    with pytest.raises(audit.DecisionAuditError, match="requires a Track-R decision head"):
        model = build_audit_detector(reference)
        config = M9TrainingConfig(run_id="t", variant=reference, steps_per_epoch=2)
        audit.audit_track_r(model, audit_batch(reference, batch_contract_for("G5", config)))


# --- checkpoint state and identity -------------------------------------------

def test_every_trainable_branch_reaches_the_optimizer_and_the_frozen_tower_does_not() -> None:
    model, _batch = model_and_batch(TRACK_R_FLAGS)
    groups = model.parameter_groups(backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.05)
    report = audit.checkpoint_state_audit(model, groups)
    assert report["passed"], report
    assert report["frozen_global_tower_parameters_in_optimizer"] == []
    assert not report["frozen_global_tower_registered_as_submodule"]
    assert all(report["branches_fully_in_optimizer"].values())


def test_the_decision_graph_identity_is_computable_and_distinct_per_track() -> None:
    graphs = {}
    for name, flags in (("G", TRACK_G_FLAGS), ("R", TRACK_R_FLAGS)):
        model, _batch = model_and_batch(flags)
        graphs[name] = audit.decision_graph_hash(model)["decision_graph_hash"]
    assert graphs["G"] != graphs["R"]
    assert all(len(value) == 64 for value in graphs.values())


def test_two_tracks_never_share_an_architecture_identity() -> None:
    assert variant(TRACK_G_FLAGS).architecture_identity() \
        != variant(TRACK_R_FLAGS).architecture_identity()
    assert variant(TRACK_R_FLAGS).architecture_identity() \
        != variant(TRACK_R_K4_FLAGS).architecture_identity()


# --- calibration identity ----------------------------------------------------

def test_the_calibration_guard_accepts_a_matched_pair() -> None:
    report = audit.calibration_identity_guard(
        decision_logit_name="fused_logit_R", calibration_logit_name="fused_logit_R",
        thresholded_quantity="p_R", decision_score_name="p_R")
    assert report["passed"]


def test_the_calibration_guard_refuses_a_temperature_from_another_logit() -> None:
    """§16.2, the Version-B G7 v1->v2 regression, made structural."""
    report = audit.calibration_identity_guard(
        decision_logit_name="fused_logit_R", calibration_logit_name="global_logit_G",
        thresholded_quantity="p_R", decision_score_name="p_R")
    assert not report["passed"]
    assert not report["calibration_fits_the_decision_logit"]


def test_the_calibration_guard_refuses_thresholding_a_different_quantity() -> None:
    report = audit.calibration_identity_guard(
        decision_logit_name="fused_logit_R", calibration_logit_name="fused_logit_R",
        thresholded_quantity="s_final", decision_score_name="p_R")
    assert not report["passed"]
    assert not report["threshold_applies_to_the_decision_score"]


# --- the addition is additive ------------------------------------------------

def test_every_inherited_variant_keeps_its_exact_architecture_identity() -> None:
    """Adding glr_concat must not move any pre-existing identity.

    The frozen M9 reference checkpoint is loadable only while its architecture
    identity is byte-for-byte what it was, so the new decision-graph fields are
    added to the payload ONLY for the new fusion.
    """
    reference = ResolvedExperimentVariant.reference()
    payload = reference.architecture_payload()
    assert set(payload) == {
        "variant_schema_version", "local_branch", "global_branch", "fusion", "region",
        "manifold", "prototype_k", "prompt", "manifold_scope", "manifold_slots",
        "fuses_region_evidence", "fuses_prompt_evidence"}
    for field in ("decision_head_type", "decision_logit_name", "branch_summary_dim"):
        assert field not in payload


def test_the_new_fusion_records_its_decision_graph_in_the_identity() -> None:
    payload = variant(TRACK_R_FLAGS).architecture_payload()
    for field in ("decision_head_type", "decision_logit_name", "decision_score_name",
                  "region_enters_decision_logit", "local_enters_decision_logit"):
        assert field in payload, field


def test_the_code_and_the_matrix_config_declare_the_same_fusions(repo) -> None:
    import yaml

    from prism_fas.detector.variant import FLAG_VOCABULARY

    declared = yaml.safe_load(
        (repo / "configs/experiments/m10_matrix.yaml").read_text(encoding="utf-8"))
    assert tuple(declared["flag_vocabulary"]["fusion"]) == FLAG_VOCABULARY["fusion"]
    assert "glr_concat" in FLAG_VOCABULARY["fusion"]


def test_glr_concat_requires_the_branches_it_fuses() -> None:
    """`g`, `l` and `r` are mandatory inputs, so a variant missing one is refused.

    The refusal happens at RESOLUTION, not at execution: the contradictory flag
    set cannot be constructed at all, so there is no window in which a Track-R
    variant exists without one of the branches its logit is defined over.
    """
    from prism_fas.detector.variant import VariantError

    for missing in ({"region": "off"}, {"global_branch": "off"}, {"local_branch": "off"}):
        with pytest.raises(VariantError, match="contradictory flags"):
            variant({**TRACK_R_FLAGS, **missing})
