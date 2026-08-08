"""M10 variant tests: every Table 59 baseline and Table 60 ablation switch.

The point of these tests is not that the switches exist. It is that each one
CHANGES SOMETHING REAL — a module that is or is not instantiated, a term that is or
is not computed, a tensor the gradient does or does not reach, an identity that does
or does not differ. A flag that lives only in YAML and changes no behaviour is the
failure mode this file exists to catch, and several tests assert exactly that a
configuration fails closed rather than being silently repaired.

They also pin the four B08 reference identities, because the M9 reference checkpoint
must stay loadable and B08 seed 20260806 must stay reusable.

No real dataset, no real weights, no network: fixture-scale structures with the real
shapes and the real contracts.
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch
import yaml

from prism_fas.detector.checkpoint import M9CheckpointError, RunIdentity, StageTransitionError, \
    check_stage_transition, load_checkpoint, save_checkpoint
from prism_fas.detector.config import batch_contract_from, detector_config_from, load_yaml, \
    training_config_from
from prism_fas.detector.contracts import LIVE, REGION_COUNT, SPOOF, DetectorBatch, \
    DetectorContractError
from prism_fas.detector.losses import compute_losses, image_level_outlier_loss, \
    loss_contract_identity, loss_graph_delta, outlier_loss
from prism_fas.detector.manifold import GLOBAL_SITE_NAMES, ManifoldError, PrototypeState
from prism_fas.detector.prism_detector import DetectorConfig, PRISMDetector, architecture_delta
from prism_fas.detector.sampler import DOMAINS, M9BatchSampler, SamplerError
from prism_fas.detector.trainer import M9TrainingConfig, batch_contract_for, enabled_terms, \
    stage_for_epoch, training_delta
from prism_fas.detector.variant import FLAG_KEYS, FLAG_VOCABULARY, REFERENCE_FLAGS, \
    ResolvedExperimentVariant, VariantError, describe_difference
from prism_fas.evaluation.experiment_matrix import build_plan
from prism_fas.evaluation.variant_audit import audit_matrix, audit_variant, build_audit_detector, \
    audit_batch
from prism_fas.evaluation.target_prediction import VariantCapabilities, build_prediction_row, \
    validate_predictions

PROJECT = Path(__file__).resolve().parents[1]
MATRIX = PROJECT / "configs" / "experiments" / "m10_matrix.yaml"

# The four identities the M9 reference run wrote its checkpoints with. If any of
# these moves, `B08-s20260806` can no longer bind the M9 reference run and the
# experiment contract section 2.1 requires it to be retrained instead.
M9_REFERENCE_IDENTITIES = {
    "detector_config": "9a94d841a9e6b5382cde8ae4d78b761060d4cbb08e3ee0cc58bd4d0d767a8e7d",
    "training_config_hash": "34a971b8b8c94fc3b9665b46c284933747302b877c94d18143bfaf6b7d71e1d0",
    "batch_contract": "ab31b33fb4a10704aad89c6034e9c9a33c412fd146000a22c9e3382a3e4fe070",
    "loss_contract": "c9d30e936ca4064284ea1f090f043d4089b9db66f84c2b3cf3a849a990e798ce"}
M10_MATRIX_IDENTITY = "a4972b0dc23946c4ad169f2c856fc9b5e0387baca45b2c9a4895f8180d9c2dd5"


@pytest.fixture(scope="module")
def plan() -> dict:
    return build_plan(MATRIX)


def variant(**flags) -> ResolvedExperimentVariant:
    return ResolvedExperimentVariant.resolve(flags)


def row_variant(plan: dict, experiment_id: str) -> ResolvedExperimentVariant:
    from prism_fas.detector.variant import variant_from_row
    for entry in plan["rows"]:
        if entry["experiment_id"] == experiment_id: return variant_from_row(entry)
    raise AssertionError(f"{experiment_id} is not in the matrix")


# ============================================================================
# THE FLAG VOCABULARY IS ONE VOCABULARY
# ============================================================================

def test_code_and_matrix_declare_the_same_flag_vocabulary():
    """A value the matrix declares but the code cannot honour is exactly the gap
    this milestone exists to close, so the two lists must be identical."""
    declared = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))["flag_vocabulary"]
    assert set(declared) == set(FLAG_VOCABULARY)
    for key, values in declared.items():
        assert tuple(values) == FLAG_VOCABULARY[key], key


def test_code_and_matrix_declare_the_same_reference_flags():
    declared = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))["reference_flags"]
    assert declared == REFERENCE_FLAGS


def test_every_matrix_row_resolves_to_a_variant(plan):
    for entry in plan["rows"]:
        resolved = row_variant(plan, entry["experiment_id"])
        assert resolved.flags() == {key: entry["flags"][key] for key in FLAG_KEYS}


def test_the_matrix_identity_is_unchanged(plan):
    """Implementing the switches must not redefine the scientific matrix."""
    assert plan["m10_matrix_identity"] == M10_MATRIX_IDENTITY


# ============================================================================
# FAIL CLOSED
# ============================================================================

@pytest.mark.parametrize("flags", [
    {"local_branch": "resnet"},                       # outside the vocabulary
    {"manifold": "off", "prototype_k": 4},            # no manifold cannot carry prototypes
    {"manifold": "global_center", "prototype_k": 4},  # a global center is exactly one
    {"region": "off"},                                # noisy-or needs the regional term
    {"local_branch": "off", "global_branch": "off"},  # no branch at all
    {"synthetic": "none"},                            # q with nothing to weight
    {"prompt": "frozen_prompt", "region": "off", "fusion": "simple_concat",
     "manifold": "off", "prototype_k": 0, "outlier_loss": "off", "synthetic": "none",
     "quality_weighting": "off", "recipe_conditioning": "off"},
])
def test_unknown_or_contradictory_combinations_fail_closed(flags):
    with pytest.raises(VariantError):
        variant(**flags)


def test_a_blocked_frame_density_is_not_executable():
    resolved = variant(frames_per_video=16)
    ok, reason = resolved.executable()
    assert ok is False and "frozen frame plan" in reason
    with pytest.raises(VariantError): resolved.require_executable()


# ============================================================================
# THE B08 REFERENCE IDENTITIES ARE UNCHANGED
# ============================================================================

def test_the_m9_reference_identities_are_byte_stable():
    reference = ResolvedExperimentVariant.reference()
    model_payload = load_yaml(PROJECT / "configs/models/m9_detector.yaml")
    training_payload = load_yaml(PROJECT / "configs/train/m9_reference.yaml")
    detector = detector_config_from(model_payload, reference)
    training = training_config_from(training_payload, variant=reference)
    contract = batch_contract_from(training_payload, reference)
    assert detector.identity() == M9_REFERENCE_IDENTITIES["detector_config"]
    assert training.hash() == M9_REFERENCE_IDENTITIES["training_config_hash"]
    assert contract.identity() == M9_REFERENCE_IDENTITIES["batch_contract"]
    assert loss_contract_identity(
        training.loss_weights,
        {"clean_cap": training.clean_cap, "mil_temperature": training.mil_temperature,
         "prompt_temperature": training.prompt_temperature,
         **loss_graph_delta(reference)}) == M9_REFERENCE_IDENTITIES["loss_contract"]


def test_the_reference_variant_carries_no_delta():
    reference = ResolvedExperimentVariant.reference()
    assert architecture_delta(reference) == {}
    assert training_delta(reference) == {}
    assert loss_graph_delta(reference) == {}


# ============================================================================
# EVERY SWITCH ENTERS AN IDENTITY
# ============================================================================

ARCHITECTURE_SWITCHES = [
    {"local_branch": "off", "fusion": "single_logit", "region": "off", "manifold": "off",
     "prototype_k": 0, "synthetic": "none", "recipe_conditioning": "off",
     "quality_weighting": "off", "outlier_loss": "off", "prompt": "off"},
    {"global_branch": "off", "fusion": "single_logit", "region": "off", "manifold": "off",
     "prototype_k": 0, "synthetic": "none", "recipe_conditioning": "off",
     "quality_weighting": "off", "outlier_loss": "off", "prompt": "off"},
    {"fusion": "simple_concat", "region": "off", "manifold": "off", "prototype_k": 0,
     "synthetic": "none", "recipe_conditioning": "off", "quality_weighting": "off",
     "outlier_loss": "off", "prompt": "off"},
    {"manifold": "global_center", "prototype_k": 1, "synthetic": "none",
     "recipe_conditioning": "off", "quality_weighting": "off", "prompt": "off"},
    {"prototype_k": 2},
    {"prompt": "off"},
    {"prompt": "adapter"},
]
TRAINING_SWITCHES = [
    {"sampler": "naive_concat"},
    {"synthetic": "physics_only"},
    {"synthetic": "gpat_only"},
    {"quality_weighting": "hard_gate_only"},
    {"outlier_loss": "image_level"},
    {"recipe_conditioning": "random_operators"},
]


@pytest.mark.parametrize("flags", ARCHITECTURE_SWITCHES)
def test_an_architecture_switch_changes_the_architecture_identity(flags):
    reference = ResolvedExperimentVariant.reference()
    changed = variant(**flags)
    assert changed.architecture_identity() != reference.architecture_identity()
    assert architecture_delta(changed) != {}


@pytest.mark.parametrize("flags", TRAINING_SWITCHES)
def test_a_training_switch_changes_the_training_identity(flags):
    """These change what is OPTIMIZED without changing a parameter shape, so an
    architecture hash cannot catch them; the training identity must."""
    reference = ResolvedExperimentVariant.reference()
    changed = variant(**flags)
    assert changed.training_identity() != reference.training_identity()
    assert changed.identity() != reference.identity()


@pytest.mark.parametrize("flags", ARCHITECTURE_SWITCHES + TRAINING_SWITCHES)
def test_every_switch_changes_the_full_variant_identity(flags):
    assert variant(**flags).identity() != ResolvedExperimentVariant.reference().identity()


def test_distinct_matrix_rows_have_distinct_identities(plan):
    """Two rows with the same flags must hash the same; two with different flags
    must not. Otherwise a checkpoint could silently change meaning."""
    by_identity: dict[str, set] = {}
    for entry in plan["rows"]:
        resolved = row_variant(plan, entry["experiment_id"])
        by_identity.setdefault(resolved.identity(), set()).add(
            json.dumps(resolved.flags(), sort_keys=True))
    for identity, flag_sets in by_identity.items():
        assert len(flag_sets) == 1, f"{identity} covers more than one flag set"


# ============================================================================
# LOCAL AND GLOBAL BRANCH SWITCHES
# ============================================================================

def test_local_branch_off_instantiates_no_local_module(plan):
    model = build_audit_detector(row_variant(plan, "B01-s20260806"))
    assert model.local_backbone is None
    assert not hasattr(model, "local_projection") and not hasattr(model, "local_head")
    names = {name for name, _ in model.named_children()}
    assert not any(name.startswith("local") for name in names)
    # No unused trainable branch may reach the optimizer.
    groups = model.parameter_groups(backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.05)
    assert [group["name"] for group in groups] == ["heads"]


def test_global_branch_off_never_runs_or_attaches_the_frozen_tower(plan):
    model = build_audit_detector(row_variant(plan, "B00-s20260806"))
    assert model.global_tower is None
    with pytest.raises(DetectorContractError):
        model.attach_global_tower(object())
    assert model.variant.architecture_payload()["global_branch"] == "off"


def test_b00_is_local_only_and_b01_is_global_only(plan):
    b00, b01 = row_variant(plan, "B00-s20260806"), row_variant(plan, "B01-s20260806")
    assert (b00.has_local, b00.has_global) == (True, False)
    assert (b01.has_local, b01.has_global) == (False, True)
    assert b00.architecture_identity() != b01.architecture_identity()


def test_the_single_branch_baselines_still_produce_a_finite_score(plan):
    for experiment_id in ("B00-s20260806", "B01-s20260806"):
        resolved = row_variant(plan, experiment_id)
        model = build_audit_detector(resolved)
        contract = batch_contract_for("G5", M9TrainingConfig(variant=resolved))
        out = model(audit_batch(resolved, contract))
        assert torch.isfinite(out.s_final).all()
        assert out.region_embeddings is None and out.region_distances is None
        assert out.s_region is None and out.p_prompt_spoof is None


# ============================================================================
# FUSION
# ============================================================================

def test_the_three_fusions_have_different_output_dependencies(plan):
    """`single_logit`, `simple_concat` and `prism_noisy_or` are three different
    computational graphs, not one under three names."""
    single = row_variant(plan, "B00-s20260806")
    concat = row_variant(plan, "B02-s20260806")
    noisy_or = row_variant(plan, "B08-s20260806")
    models = {name: build_audit_detector(v) for name, v in
              (("single", single), ("concat", concat), ("noisy_or", noisy_or))}
    children = {name: {child for child, _ in model.named_children()}
                for name, model in models.items()}
    # Only the Table 34 path owns GlobalHead; only the two simpler ones own a
    # fusion classifier; only concat owns the concat projection.
    assert "global_head" in children["noisy_or"] and "global_head" not in children["concat"]
    assert "fusion_classifier" in children["concat"] and "fusion_classifier" not in children["noisy_or"]
    assert "fusion_projection" in children["concat"] and "fusion_projection" not in children["single"]
    assert "global_projection" not in children["single"]


def test_only_the_noisy_or_fusion_consumes_region_and_prompt_evidence(plan):
    for experiment_id, fused in (("B00-s20260806", False), ("B02-s20260806", False),
                                 ("B05-s20260806", False), ("B08-s20260806", True)):
        resolved = row_variant(plan, experiment_id)
        assert resolved.fuses_region_evidence is fused, experiment_id


def test_simple_concat_score_is_the_classifier_not_a_noisy_or(plan):
    """B05 has a manifold, but its score comes from the concat classifier. Feeding
    the manifold into a noisy-or would silently make B05 a regional detector."""
    resolved = row_variant(plan, "B05-s20260806")
    model = build_audit_detector(resolved)
    contract = batch_contract_for("G5", M9TrainingConfig(variant=resolved))
    out = model(audit_batch(resolved, contract))
    assert out.region_distances is not None            # the manifold exists
    assert out.s_region is None                        # but it does not fuse
    assert torch.allclose(out.s_final, out.p_global)


# ============================================================================
# REGION
# ============================================================================

def test_region_off_removes_the_whole_semantic_region_path(plan):
    resolved = row_variant(plan, "A05-region-global_only-s20260806")
    assert resolved.has_region_path is False
    model = build_audit_detector(resolved)
    children = {name for name, _ in model.named_children()}
    assert not ({"region_query", "region_attention", "region_pool", "region_norm"} & children)
    assert model.prompt_head is None
    contract = batch_contract_for("G5", M9TrainingConfig(variant=resolved))
    out = model(audit_batch(resolved, contract))
    assert out.region_embeddings is None
    terms = resolved.active_loss_terms()
    assert terms["L_prompt"] is False and terms["L_clean"] is False


def test_region_on_keeps_the_soft_prior_path(plan):
    model = build_audit_detector(row_variant(plan, "B08-s20260806"))
    children = {name for name, _ in model.named_children()}
    assert {"region_query", "region_attention", "region_pool", "region_norm"} <= children


# ============================================================================
# MANIFOLD AND PROTOTYPE K
# ============================================================================

def test_manifold_off_has_no_manifold_state_no_g2_and_no_manifold_losses(plan):
    resolved = row_variant(plan, "B02-s20260806")
    assert resolved.has_manifold is False and resolved.manifold_slots == 0
    assert resolved.required_stages() == ("G1", "G5", "G6")
    model = build_audit_detector(resolved)
    assert model.manifold is None and not hasattr(model, "distance_scale")
    terms = resolved.active_loss_terms()
    assert not any(terms[name] for name in ("L_real", "L_out", "L_clean"))


def test_global_center_is_exactly_one_site_and_multi_prototype_is_nine(plan):
    b06 = row_variant(plan, "B06-s20260806")
    b07 = row_variant(plan, "B07-s20260806")
    assert (b06.manifold_scope, b06.manifold_slots, b06.prototype_k) == ("global", 1, 1)
    assert (b07.manifold_scope, b07.manifold_slots, b07.prototype_k) == ("regional", REGION_COUNT, 4)
    assert build_audit_detector(b06).manifold.region_names == GLOBAL_SITE_NAMES
    assert len(build_audit_detector(b07).manifold.region_names) == REGION_COUNT
    # H2 is exactly this pair, so they must differ in the manifold and nothing else.
    assert set(describe_difference(b06, b07)) == {"manifold", "prototype_k"}


def test_b06_and_b07_produce_different_distance_shapes(plan):
    for experiment_id, slots in (("B06-s20260806", 1), ("B07-s20260806", REGION_COUNT)):
        resolved = row_variant(plan, experiment_id)
        report = audit_variant(resolved, experiment_id=experiment_id)
        assert report["output_components"]["manifold_slots"] == slots
        assert report["implementable"], report["findings"]


def test_a_global_prototype_state_cannot_be_loaded_into_a_regional_manifold():
    from prism_fas.detector.manifold import RealManifold
    import numpy as np
    regional = RealManifold(4, k=1, regions=REGION_COUNT)
    state = PrototypeState(centers=np.zeros((1, 1, 4)), variances=np.full((1, 1, 4), 1e-4),
                           counts=np.ones((1, 1), dtype=np.int64), valid=np.ones((1, 1), dtype=bool),
                           epsilon=1e-4, region_names=GLOBAL_SITE_NAMES)
    with pytest.raises(ManifoldError):
        regional.load_state(state)


@pytest.mark.parametrize("k", [1, 2, 6])
def test_prototype_k_changes_state_shape_and_identity_not_only_metadata(k, plan):
    reference = row_variant(plan, "B08-s20260806")
    changed = reference.with_flags(prototype_k=k)
    assert changed.architecture_identity() != reference.architecture_identity()
    model = build_audit_detector(changed)
    assert model.manifold.centers.shape[1] == k
    assert model.manifold.k == k


def test_the_reference_k_stays_four(plan):
    assert row_variant(plan, "B08-s20260806").prototype_k == 4


# ============================================================================
# SYNTHETIC ROUTE
# ============================================================================

@pytest.mark.parametrize("experiment_id,routes", [
    ("B03-s20260806", ("physics",)),
    ("B04-s20260806", ("gpat",)),
    ("A03-synthetic_route-physics_only-s20260806", ("physics",)),
    ("A03-synthetic_route-gpat_only-s20260806", ("gpat",)),
    ("B08-s20260806", ("physics", "gpat")),
])
def test_the_synthetic_route_flag_restricts_the_declared_routes(plan, experiment_id, routes):
    resolved = row_variant(plan, experiment_id)
    assert resolved.synthetic_routes == routes
    contract = batch_contract_for("G5", M9TrainingConfig(variant=resolved))
    assert tuple(contract.routes) == routes
    assert contract.require_both_routes is (len(routes) == 2)


def test_a_single_route_row_never_asks_for_both_routes(plan):
    resolved = row_variant(plan, "B03-s20260806")
    with pytest.raises(SamplerError):
        batch_contract_for("G5", M9TrainingConfig(variant=resolved)).__class__(
            real_live=12, real_spoof=12, synthetic=8, routes=("physics",),
            require_both_routes=True).validate()


def test_a_synthetic_none_row_declares_an_empty_synthetic_pool(plan):
    """Regression: the route filter was written `if allowed and route not in
    allowed`, which short-circuits on the EMPTY allowed set that `synthetic: none`
    produces — so all 871 accepted rows entered pools that were never sampled, and
    `run.json` reported 871 synthetic samples for a row that used none. Training was
    never affected (the batch quota is 0), but the report was wrong. Found by
    reading the real runs' artifacts, not by inspection."""
    # Exactly the rows that declare `synthetic: none`. A05 is NOT one of them — it
    # inherits `bank_physics_gpat` from B08 and only ablates the region path, which
    # the real A05 run confirms (`declared_routes: [physics, gpat]`, pool 871).
    for experiment_id in ("B00-s20260806", "B01-s20260806", "B02-s20260806",
                          "B05-s20260806", "B06-s20260806", "B07-s20260806"):
        resolved = row_variant(plan, experiment_id)
        assert resolved.uses_synthetic is False, experiment_id
        assert resolved.synthetic_routes == ()
        contract = batch_contract_for("G5", M9TrainingConfig(variant=resolved))
        assert contract.synthetic == 0
    # And A05 really does keep the full bank, so the fix must not silence it.
    a05 = row_variant(plan, "A05-region-global_only-s20260806")
    assert a05.uses_synthetic is True and a05.synthetic_routes == ("physics", "gpat")


def test_a_route_restricted_sampler_draws_only_that_route(plan):
    resolved = row_variant(plan, "A03-synthetic_route-physics_only-s20260806")
    contract = batch_contract_for("G5", M9TrainingConfig(variant=resolved))
    physics, gpat = list(range(0, 40)), list(range(40, 80))
    sampler = M9BatchSampler(
        real_live={"casia_fasd": list(range(0, 20)), "msu_mfsd": list(range(20, 40))},
        real_spoof={"casia_fasd": list(range(40, 60)), "msu_mfsd": list(range(60, 80))},
        synthetic_routes={"physics": physics, "gpat": gpat}, contract=contract,
        seed=20260806, steps_per_epoch=4, identity="toy")
    drawn = {index for plan_ in sampler.epoch_plans(0) for index in plan_.synthetic}
    assert drawn and drawn <= set(physics)
    assert not (drawn & set(gpat))


# ============================================================================
# SYNTHETIC = NONE BATCH CONTRACT
# ============================================================================

def test_synthetic_none_uses_a_real_only_g5_batch_not_a_fabricated_one(plan):
    """B00-B07 must not be handed eight fabricated synthetic slots, and their
    synthetic loss terms must be structurally inactive rather than constant."""
    resolved = row_variant(plan, "B07-s20260806")
    contract = batch_contract_for("G5", M9TrainingConfig(variant=resolved))
    assert contract.synthetic == 0 and contract.phase == "real_only"
    assert contract.real_live == contract.real_spoof == 16
    assert contract.domain_balance is True
    terms = resolved.active_loss_terms()
    for name in ("L_cls_syn", "L_local", "L_out", "L_clean", "L_prompt"):
        assert terms[name] is False, name


def test_a_synthetic_row_keeps_the_table_36_composition(plan):
    contract = batch_contract_for("G5", M9TrainingConfig(variant=row_variant(plan, "B08-s20260806")))
    assert (contract.real_live, contract.real_spoof, contract.synthetic) == (12, 12, 8)
    assert contract.batch_size == 32


# ============================================================================
# RECIPE CONDITIONING
# ============================================================================

def test_recipe_conditioning_off_stops_the_recipe_identity_reaching_the_loss(plan):
    b04 = row_variant(plan, "B04-s20260806")
    assert b04.recipe_conditioning == "off"
    assert b04.consumes_recipe_identity is False
    assert b04.recipe_source == "none"


def test_random_operators_names_a_separate_source_only_artifact(plan):
    resolved = row_variant(plan, "A02-recipe-random_operators-s20260806")
    assert resolved.recipe_source == "m10_random_operator_bank"
    assert resolved.recipe_source != ResolvedExperimentVariant.reference().recipe_source
    # It is a different training identity, so a B08 checkpoint cannot be resumed as it.
    assert resolved.training_identity() != ResolvedExperimentVariant.reference().training_identity()


# ============================================================================
# QUALITY WEIGHTING
# ============================================================================

def test_hard_gate_only_reuses_the_same_accepted_samples_at_equal_weight(plan):
    reference = row_variant(plan, "B08-s20260806")
    hard_gate = row_variant(plan, "A04-quality_weighting-hard_gate_only-s20260806")
    # Same bank, same routes, same acceptance: only the WEIGHT differs.
    assert hard_gate.synthetic == reference.synthetic
    assert hard_gate.synthetic_routes == reference.synthetic_routes
    assert hard_gate.uses_quality_weight is False and reference.uses_quality_weight is True
    assert set(describe_difference(reference, hard_gate)) == {"quality_weighting"}


def test_q_changes_the_weighting_only_and_never_the_label(plan):
    """`q` multiplies the synthetic bracket. It must never move a label, and the two
    quality variants must see identical labels on an identical batch."""
    reference = row_variant(plan, "B08-s20260806")
    hard_gate = row_variant(plan, "A04-quality_weighting-hard_gate_only-s20260806")
    contract = batch_contract_for("G5", M9TrainingConfig(variant=reference))
    batch = audit_batch(reference, contract)
    labels = batch.label.clone()
    # ONE model and ONE forward pass, scored under both variants: A04 and B08 share
    # an architecture, so any difference in the per-term values can only come from
    # the quality flag itself rather than from two different weight initializations.
    assert reference.architecture_identity() == hard_gate.architecture_identity()
    model = build_audit_detector(reference)
    _seed_prototypes(model, reference, batch)
    out = model(batch)
    results = {name: compute_losses(out, batch, model.manifold, text_embeddings=model.text_matrix(),
                                    enabled=enabled_terms("G5", resolved), variant=resolved)
               for name, resolved in (("q", reference), ("hard", hard_gate))}
    assert torch.equal(batch.label, labels)
    assert results["q"].weights["q_bar"] == pytest.approx(0.8)
    assert results["hard"].weights["q_bar"] == pytest.approx(1.0)
    # The per-term values are identical; only the bracket weight differs.
    for name in ("L_cls_real", "L_cls_syn", "L_MIL"):
        assert results["q"].terms[name].detach() == pytest.approx(
            float(results["hard"].terms[name].detach()), rel=1e-6)


def _seed_prototypes(model, resolved, batch) -> None:
    from prism_fas.detector.manifold import initialize_prototypes
    if model.manifold is None: return
    with torch.no_grad():
        probe = model(batch)
        sites = (probe.aux or {}).get("manifold_embeddings", probe.region_embeddings)
    model.manifold.load_state(initialize_prototypes(
        sites.repeat(4, 1, 1).numpy(), probe.region_valid.repeat(4, 1).numpy(),
        k=int(resolved.prototype_k), seed=20260806, region_names=model.manifold.region_names))


# ============================================================================
# OUTLIER LOSS
# ============================================================================

def test_image_level_and_mask_aware_ask_different_questions():
    """The mask-aware term reads `m_r` and pushes only the ATTACKED regions. The
    image-level term never reads `m_r`. On a fixture where the attacked and the
    clean regions have deliberately different distances, the two must disagree."""
    distance = torch.tensor([[0.1, 0.1, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])
    attack = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    synthetic = torch.tensor([True])
    valid = torch.ones(1, REGION_COUNT, dtype=torch.bool)
    masked = float(outlier_loss(distance, attack, synthetic, valid, margin=3.0))
    image = float(image_level_outlier_loss(distance, synthetic, valid, margin=3.0))
    # Mask-aware sees two attacked regions well inside the margin: 2 * (3 - 0.1).
    assert masked == pytest.approx(5.8)
    # Image-level averages every site: mean = (0.2 + 35)/9 = 3.911..., above the
    # margin, so the hinge is exactly zero and the attack is invisible to it.
    assert image == pytest.approx(0.0)
    assert masked != image


def test_moving_a_clean_region_changes_image_level_but_not_mask_aware():
    """The converse: image-level responds to a region the recipe never touched."""
    attack = torch.tensor([[1.0] + [0.0] * 8])
    synthetic = torch.tensor([True])
    valid = torch.ones(1, REGION_COUNT, dtype=torch.bool)
    near = torch.tensor([[0.5] + [0.5] * 8])
    far = torch.tensor([[0.5] + [9.0] * 8])
    assert float(outlier_loss(near, attack, synthetic, valid, margin=3.0)) == \
        pytest.approx(float(outlier_loss(far, attack, synthetic, valid, margin=3.0)))
    assert float(image_level_outlier_loss(near, synthetic, valid, margin=3.0)) != \
        pytest.approx(float(image_level_outlier_loss(far, synthetic, valid, margin=3.0)))


def test_image_level_has_no_clean_companion_term(plan):
    """`L_clean` is defined by `1 - m_r`; a term that never reads the mask cannot
    have one, so it is structurally inactive rather than merely down-weighted."""
    a07 = row_variant(plan, "A07-outlier-image_level-s20260806")
    reference = row_variant(plan, "B08-s20260806")
    assert a07.active_loss_terms()["L_out"] is True
    assert a07.active_loss_terms()["L_clean"] is False
    assert reference.active_loss_terms()["L_clean"] is True
    assert set(describe_difference(reference, a07)) == {"outlier_loss"}


# ============================================================================
# PROMPT
# ============================================================================

def test_prompt_off_removes_the_head_and_the_loss(plan):
    resolved = row_variant(plan, "A08-prompt-off-s20260806")
    model = build_audit_detector(resolved)
    assert model.prompt_head is None and model.text_matrix() is None
    assert resolved.active_loss_terms()["L_prompt"] is False
    contract = batch_contract_for("G5", M9TrainingConfig(variant=resolved))
    out = model(audit_batch(resolved, contract))
    assert out.prompt_logits is None and out.p_prompt_spoof is None


def test_the_adapter_is_a_no_op_at_step_zero_and_trainable_after(plan):
    """`adapter` must start numerically identical to `frozen_prompt`, so the
    ablation measures what the adapter LEARNS rather than a different init."""
    resolved = row_variant(plan, "A08-prompt-adapter-s20260806")
    model = build_audit_detector(resolved)
    assert model.text_adapter is not None
    base = model.prompt_head.text_embeddings
    assert torch.allclose(model.text_matrix(), base, atol=1e-6)
    # It is trainable, and it is in the heads group, not a fine-tuned text tower.
    parameters = {name for name, _ in model.text_adapter.named_parameters()}
    assert parameters == {"norm.weight", "norm.bias", "down.weight", "down.bias", "up.weight"}
    with torch.no_grad(): model.text_adapter.up.weight.fill_(0.01)
    assert not torch.allclose(model.text_matrix(), base, atol=1e-6)


def test_the_adapter_never_loads_or_finetunes_a_text_encoder(plan):
    model = build_audit_detector(row_variant(plan, "A08-prompt-adapter-s20260806"))
    assert not hasattr(model, "text_model") and not hasattr(model, "text_tower")
    # The cached matrix is still a buffer, so no gradient can reach the cache itself.
    assert not model.prompt_head.text_embeddings.requires_grad


def test_the_three_prompt_values_have_three_identities(plan):
    identities = {value: variant(prompt=value).architecture_identity()
                  for value in ("off", "frozen_prompt", "adapter")}
    assert len(set(identities.values())) == 3


# ============================================================================
# SAMPLER
# ============================================================================

def test_naive_concat_is_not_domain_balanced(plan):
    resolved = row_variant(plan, "A01-data_balance-naive_concat-s20260806")
    assert resolved.domain_balance is False
    contract = batch_contract_for("G5", M9TrainingConfig(variant=resolved))
    assert contract.domain_balance is False
    assert set(describe_difference(ResolvedExperimentVariant.reference(), resolved)) == {"sampler"}


def test_naive_concat_does_not_quietly_preserve_domain_balance():
    """A control that silently rebalanced would make H1 unanswerable. With a pool
    that is deliberately lopsided, the realized domain counts must follow the pool."""
    from prism_fas.detector.sampler import BatchContract
    contract = BatchContract(real_live=12, real_spoof=12, synthetic=0, phase="real_only",
                             domain_balance=False, require_both_routes=False).validate()
    sampler = M9BatchSampler(
        real_live={"casia_fasd": list(range(0, 90)), "msu_mfsd": list(range(90, 100))},
        real_spoof={"casia_fasd": list(range(100, 190)), "msu_mfsd": list(range(190, 200))},
        synthetic_routes={"physics": [0], "gpat": [1]}, contract=contract,
        seed=20260806, steps_per_epoch=20, identity="toy")
    casia = msu = 0
    for plan_ in sampler.epoch_plans(0):
        for index in list(plan_.real_live) + list(plan_.real_spoof):
            if index % 100 < 90: casia += 1
            else: msu += 1
    # A domain-balanced sampler would give 50/50; the naive pool is 90/10.
    assert casia > 4 * msu


def test_naive_concat_still_guarantees_live_and_spoof_in_every_batch():
    from prism_fas.detector.sampler import BatchContract
    contract = BatchContract(real_live=12, real_spoof=12, synthetic=0, phase="real_only",
                             domain_balance=False, require_both_routes=False).validate()
    sampler = M9BatchSampler(
        real_live={"casia_fasd": [0], "msu_mfsd": [1]},
        real_spoof={"casia_fasd": list(range(2, 200)), "msu_mfsd": list(range(200, 400))},
        synthetic_routes={"physics": [0], "gpat": [1]}, contract=contract,
        seed=20260806, steps_per_epoch=25, identity="toy")
    for plan_ in sampler.epoch_plans(0):
        assert plan_.real_live and plan_.real_spoof
        assert plan_.size == 24


def test_both_samplers_are_deterministic_under_the_declared_seed():
    from prism_fas.detector.sampler import BatchContract
    pools = dict(real_live={"casia_fasd": list(range(0, 20)), "msu_mfsd": list(range(20, 40))},
                 real_spoof={"casia_fasd": list(range(40, 60)), "msu_mfsd": list(range(60, 80))},
                 synthetic_routes={"physics": list(range(0, 10)), "gpat": list(range(10, 20))})
    for balanced in (True, False):
        contract = BatchContract(real_live=12, real_spoof=12, synthetic=8,
                                 domain_balance=balanced).validate()
        first = M9BatchSampler(**pools, contract=contract, seed=20260806, steps_per_epoch=5,
                               identity="toy").fingerprint(0)
        second = M9BatchSampler(**pools, contract=contract, seed=20260806, steps_per_epoch=5,
                                identity="toy").fingerprint(0)
        assert first == second


# ============================================================================
# STAGE RESOLUTION
# ============================================================================

def test_g2_exists_exactly_where_prototypes_exist(plan):
    for entry in plan["rows"]:
        if entry.get("status") == "BLOCKED": continue
        resolved = row_variant(plan, entry["experiment_id"])
        assert ("G2" in resolved.required_stages()) is resolved.has_manifold, entry["experiment_id"]


def test_a_variant_without_a_manifold_may_not_enter_g2():
    resolved = variant(manifold="off", prototype_k=0, region="off", fusion="simple_concat",
                       synthetic="none", recipe_conditioning="off", quality_weighting="off",
                       outlier_loss="off", prompt="off")
    with pytest.raises(StageTransitionError):
        check_stage_transition("G1", "G2", order=resolved.required_stages())
    assert check_stage_transition("G1", "G5", order=resolved.required_stages()) == "G5"


def test_a_variant_with_a_manifold_may_not_skip_g2():
    resolved = ResolvedExperimentVariant.reference()
    with pytest.raises(StageTransitionError):
        check_stage_transition("G1", "G5", order=resolved.required_stages())


def test_the_trainer_enters_stages_against_its_own_declared_flow(plan):
    """Regression: `M9Trainer.enter_stage` checked the transition against the full
    G1->G2->G5->G6 default before handing it to the lineage, so a manifold-free
    variant was refused its own legitimate G1 -> G5. Found by the first real-data
    smoke, not by inspection - B00, B01 and B02 all failed on it."""
    import inspect
    from prism_fas.detector import trainer as trainer_module
    source = inspect.getsource(trainer_module.M9Trainer.enter_stage)
    assert "order=self.stages" in source
    # And the declared flows themselves round-trip through the stage machine.
    for experiment_id in ("B00-s20260806", "B02-s20260806", "B08-s20260806"):
        stages = row_variant(plan, experiment_id).required_stages()
        current = None
        for stage in stages:
            current = check_stage_transition(current, stage, order=stages)
        assert current == "G6"


def test_the_total_schedule_length_is_the_same_for_every_variant(plan):
    lengths = set()
    for entry in plan["rows"]:
        if entry.get("status") == "BLOCKED": continue
        config = M9TrainingConfig(variant=row_variant(plan, entry["experiment_id"]))
        lengths.add(config.total_epochs)
        stages = [stage_for_epoch(epoch, config) for epoch in range(config.total_epochs)]
        assert ("G2" in stages) is config.variant.has_manifold
    assert lengths == {35}


def test_g6_is_source_dev_only_for_every_variant(plan):
    for entry in plan["rows"]:
        if entry.get("status") == "BLOCKED": continue
        config = M9TrainingConfig(variant=row_variant(plan, entry["experiment_id"]))
        for metric in (config.selection_metric, config.tie_break_metric, config.calibration_metric):
            assert metric.startswith("source_dev/")


# ============================================================================
# LOSS GRAPH FOLLOWS ACTIVE FEATURES
# ============================================================================

def test_the_computed_loss_graph_equals_the_declared_one(plan):
    for entry in plan["rows"]:
        if entry.get("status") == "BLOCKED": continue
        report = audit_variant(row_variant(plan, entry["experiment_id"]),
                               experiment_id=entry["experiment_id"])
        assert report["implementable"], (entry["experiment_id"], report["findings"])
        assert report["active_loss_terms"] == \
            row_variant(plan, entry["experiment_id"]).stage_loss_terms("G5")


def test_an_inactive_term_is_an_exact_structural_zero(plan):
    resolved = row_variant(plan, "B02-s20260806")
    report = audit_variant(resolved, experiment_id="B02")
    for name, active in report["active_loss_terms"].items():
        if not active: assert report["loss_values"][name] == 0.0, name


def test_q_never_weights_a_non_synthetic_term(plan):
    """Read the declared total literally: `q` multiplies only the synthetic bracket."""
    reference = row_variant(plan, "B08-s20260806")
    contract = batch_contract_for("G5", M9TrainingConfig(variant=reference))
    batch = audit_batch(reference, contract)
    model = build_audit_detector(reference)
    _seed_prototypes(model, reference, batch)
    out = model(batch)
    weights = compute_losses(out, batch, model.manifold, text_embeddings=model.text_matrix(),
                             enabled=enabled_terms("G5", reference), variant=reference).weights
    assert weights["q_bar"] == pytest.approx(0.8)


# ============================================================================
# CHECKPOINT IDENTITY AND RESUME
# ============================================================================

def _identity(resolved, model) -> RunIdentity:
    config = M9TrainingConfig(variant=resolved)
    contract = batch_contract_for("G5", config)
    return RunIdentity(
        source_package_identity="pkg", m8_bank_identity="bank",
        architecture_identity=model.architecture_identity(), siglip2_identity="siglip",
        recipe_text_cache_identity="cache", config_hash=config.hash(),
        loss_contract_hash=loss_contract_identity(config.loss_weights, loss_graph_delta(resolved)),
        batch_contract_hash=contract.identity(), dataset_contract_identity="dataset").validate()


def test_a_manifold_free_variant_can_save_and_reload_a_checkpoint(tmp_path, plan):
    """Regression: `prototype_payload` called `manifold.export_state()`
    unconditionally, so every manifold-free variant crashed at its first save. Found
    by the second real-data smoke, not by inspection — B00, B01 and B02 all hit it.
    The absent state is recorded EXPLICITLY, never as a zero-filled one."""
    from prism_fas.detector.checkpoint import apply_checkpoint, load_checkpoint
    resolved = row_variant(plan, "B02-s20260806")
    assert resolved.has_manifold is False
    model = build_audit_detector(resolved)
    identity = _identity(resolved, model)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model=model, optimizer=None, scheduler=None, scaler=None, epoch=0,
                    global_step=1, stage="G1", identity=identity, sampler_state={},
                    prototype_identity="", best_metrics={}, stage_lineage=[])
    payload = load_checkpoint(path, expected_identity=identity, allowed_stages=("G1", "G5", "G6"))
    assert payload["prototype_state"]["manifold"] == "absent"
    assert payload["prototype_state"]["initialized"] is False
    restored = apply_checkpoint(payload, model=model)
    assert restored["global_step"] == 1


def test_a_prototype_state_cannot_cross_the_manifold_boundary(plan):
    """A checkpoint carrying prototypes must not load into a manifold-free model,
    nor an absent state into one that has a manifold."""
    from prism_fas.detector.checkpoint import _restore_prototypes
    with_manifold = build_audit_detector(row_variant(plan, "B08-s20260806"))
    with pytest.raises(M9CheckpointError):
        _restore_prototypes(with_manifold.manifold, {"manifold": "absent", "initialized": False})
    with pytest.raises(M9CheckpointError):
        _restore_prototypes(None, {"initialized": True, "centers": [], "variances": [],
                                   "counts": [], "valid": [], "epsilon": 1e-4, "region_names": []})


def test_a_checkpoint_cannot_be_resumed_under_a_different_scientific_switch(tmp_path, plan):
    """`prompt=off` must not load as `prompt=frozen_prompt`, and the failure must
    name exactly which identity disagreed."""
    trained = row_variant(plan, "A08-prompt-off-s20260806")
    other = row_variant(plan, "B08-s20260806")
    model = build_audit_detector(trained)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model=model, optimizer=None, scheduler=None, scaler=None, epoch=0,
                    global_step=1, stage="G1", identity=_identity(trained, model),
                    sampler_state={}, prototype_identity="", best_metrics={}, stage_lineage=[])
    with pytest.raises(M9CheckpointError) as error:
        load_checkpoint(path, expected_identity=_identity(other, build_audit_detector(other)))
    assert "architecture_identity" in str(error.value)


def test_a_training_only_switch_also_blocks_resume(tmp_path, plan):
    """A04 has the same parameter shapes as B08, so only the training identity can
    catch it. It must still refuse, or an ablation would silently inherit B08."""
    trained = row_variant(plan, "B08-s20260806")
    other = row_variant(plan, "A04-quality_weighting-hard_gate_only-s20260806")
    assert trained.architecture_identity() == other.architecture_identity()
    model = build_audit_detector(trained)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model=model, optimizer=None, scheduler=None, scaler=None, epoch=0,
                    global_step=1, stage="G1", identity=_identity(trained, model),
                    sampler_state={}, prototype_identity="", best_metrics={}, stage_lineage=[])
    with pytest.raises(M9CheckpointError) as error:
        load_checkpoint(path, expected_identity=_identity(other, build_audit_detector(other)))
    assert "config_hash" in str(error.value) or "loss_contract_hash" in str(error.value)


# ============================================================================
# G7 INFERENCE ADAPTER
# ============================================================================

def test_an_absent_component_is_null_and_never_a_fabricated_zero(plan):
    resolved = row_variant(plan, "B00-s20260806")
    capabilities = VariantCapabilities.from_variant(resolved)
    assert capabilities.has_region is False and capabilities.has_prompt is False
    row = build_prediction_row(sample_id="a", video_id="v", frame_id=0, p_global=0.7,
                               s_region=None, p_prompt=None, threshold=0.5, unknown_threshold=None,
                               top_region_ids=[], region_distances=[], checkpoint_hash="c",
                               calibration_hash="k", inference_config_hash="i", variant="B00")
    assert row["s_region"] is None and row["p_prompt"] is None
    assert row["region_status"] == row["prompt_status"] == "not_applicable"
    # `s_final` is the same formula with fewer factors, so it collapses to p_global.
    assert row["s_final"] == pytest.approx(0.7)
    validate_predictions([row])


def test_a_global_center_reports_a_fused_score_without_region_detail(plan):
    capabilities = VariantCapabilities.from_variant(row_variant(plan, "B06-s20260806"))
    assert capabilities.has_region is True and capabilities.has_region_detail is False
    row = build_prediction_row(sample_id="a", video_id="v", frame_id=0, p_global=0.4,
                               s_region=0.25, p_prompt=None, threshold=0.5, unknown_threshold=None,
                               top_region_ids=[], region_distances=[], checkpoint_hash="c",
                               calibration_hash="k", inference_config_hash="i", variant="B06")
    assert row["region_status"] == "computed" and row["top_region_ids"] == []
    validate_predictions([row])


def test_the_baselines_are_not_forced_through_the_b08_fusion(plan):
    for experiment_id in ("B00-s20260806", "B01-s20260806", "B02-s20260806", "B05-s20260806"):
        capabilities = VariantCapabilities.from_variant(row_variant(plan, experiment_id))
        assert capabilities.has_region is False and capabilities.has_prompt is False


# ============================================================================
# THE MATRIX DIFFERENCE AUDIT
# ============================================================================

@pytest.mark.parametrize("experiment_id,expected", [
    ("A01-data_balance-naive_concat-s20260806", {"sampler"}),
    ("A02-recipe-random_operators-s20260806", {"recipe_conditioning"}),
    ("A03-synthetic_route-physics_only-s20260806", {"synthetic"}),
    ("A03-synthetic_route-gpat_only-s20260806", {"synthetic"}),
    ("A04-quality_weighting-hard_gate_only-s20260806", {"quality_weighting"}),
    ("A06-prototype_k-k1-s20260806", {"prototype_k"}),
    ("A06-prototype_k-k2-s20260806", {"prototype_k"}),
    ("A06-prototype_k-k6-s20260806", {"prototype_k"}),
    ("A07-outlier-image_level-s20260806", {"outlier_loss"}),
    ("A08-prompt-off-s20260806", {"prompt"}),
    ("A08-prompt-adapter-s20260806", {"prompt"}),
])
def test_each_single_dimension_ablation_changes_exactly_that_dimension(plan, experiment_id, expected):
    """No accidental extra flag drift. An ablation that moved two dimensions would
    make its hypothesis untestable."""
    reference = ResolvedExperimentVariant.reference()
    assert set(describe_difference(reference, row_variant(plan, experiment_id))) == expected


def test_the_multi_dimension_rows_declare_every_dimension_they_change(plan):
    reference = ResolvedExperimentVariant.reference()
    a05 = describe_difference(reference, row_variant(plan, "A05-region-global_only-s20260806"))
    # A05 removes the semantic regions; the manifold, the fusion, the outlier term
    # and the PromptHead are all defined over them, so all five move together.
    assert set(a05) == {"region", "fusion", "manifold", "prototype_k", "outlier_loss", "prompt"}
    b00 = describe_difference(reference, row_variant(plan, "B00-s20260806"))
    assert "global_branch" in b00 and "local_branch" not in b00
    b01 = describe_difference(reference, row_variant(plan, "B01-s20260806"))
    assert "local_branch" in b01 and "global_branch" not in b01


def test_the_parity_row_shares_the_reference_scientific_configuration(plan):
    reference = row_variant(plan, "B08-s20260806")
    parity = row_variant(plan, "A09-backend-pc_bounded_parity-s20260806")
    assert describe_difference(reference, parity) == {}
    assert parity.identity() == reference.identity()


# ============================================================================
# THE B08 / M9 REFERENCE BINDING
# ============================================================================

def _m9_like_run(stages=("G1", "G2", "G5", "G6"), **overrides) -> dict:
    run = {"seed": 20260806, "status": "COMPLETED", "target_test_opened": False,
           "identity": {"architecture_identity": "arch", "config_hash": "cfg"},
           "stage_lineage": [{"stage": stage, "status": "COMPLETED"} for stage in stages]}
    run.update(overrides)
    return run


def test_the_m9_reference_binding_ignores_the_target_prediction_stage(plan):
    """G7 is the M10 target-prediction stage and has not run for ANY row. Requiring
    it here refused the binding for the one reason that is true of every row alike —
    which is what the first run of this validator actually did."""
    from prism_fas.evaluation.experiment_registry import validate_m9_reference_binding
    row = next(entry for entry in plan["rows"] if entry["experiment_id"] == "B08-s20260806")
    assert "G7" in row["required_stages"]
    result = validate_m9_reference_binding(
        row, _m9_like_run(), {"architecture_identity": "arch", "config_hash": "cfg"})
    assert result["binding_accepted"] is True, result["mismatched"]
    assert result["checks"]["every_declared_training_stage_completed"]["matches"] is True
    assert result["checks"]["target_prediction_still_pending"]["matches"] is True


@pytest.mark.parametrize("run,reason", [
    (_m9_like_run(identity={"architecture_identity": "other", "config_hash": "cfg"}),
     "architecture_identity"),
    (_m9_like_run(seed=20260807), "seed_matches"),
    (_m9_like_run(status="FAILED"), "run_completed"),
    (_m9_like_run(stages=("G1", "G2", "G5")), "every_declared_training_stage_completed"),
    (_m9_like_run(target_test_opened=True), "target_never_opened"),
])
def test_the_binding_is_refused_on_any_real_mismatch(plan, run, reason):
    """An identity binding, not an approximation: a near-match is never a match."""
    from prism_fas.evaluation.experiment_registry import validate_m9_reference_binding
    row = next(entry for entry in plan["rows"] if entry["experiment_id"] == "B08-s20260806")
    result = validate_m9_reference_binding(
        row, run, {"architecture_identity": "arch", "config_hash": "cfg"})
    assert result["binding_accepted"] is False
    assert reason in result["mismatched"]
    assert "REFUSED" in result["decision"]


def test_the_recorded_b08_binding_evidence_was_accepted():
    """The real evidence artifact, re-read from disk."""
    path = PROJECT / "reports" / "m10" / "B08_M9_REFERENCE_BINDING.json"
    if not path.is_file(): pytest.skip("binding evidence has not been produced in this checkout")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["experiment_id"] == "B08-s20260806"
    assert evidence["reference_run_id"] == "m9_reference_seed20260806"
    assert evidence["binding_accepted"] is True and evidence["mismatched"] == []
    assert evidence["m10_matrix_identity"] == M10_MATRIX_IDENTITY
    assert evidence["target_labels_opened"] is False


# ============================================================================
# THE IMPLEMENTABILITY AUDIT
# ============================================================================

def test_every_executable_row_is_implementable(plan):
    report = audit_matrix(plan)
    assert report["m10_matrix_identity"] == M10_MATRIX_IDENTITY
    assert report["audited_rows"] == 38
    assert report["implementable_rows"] == 38
    assert report["not_implementable"] == []
    assert report["all_executable_rows_implementable"] is True
    assert report["scientific_configs_not_covered"] == []
    assert report["rows_sharing_the_reference_architecture_despite_a_delta"] == []


def test_blocked_rows_stay_blocked_for_their_own_declared_reasons(plan):
    report = audit_matrix(plan)
    blocked = {row["experiment_id"]: row["blocked_reason"] for row in report["blocked_rows"]}
    assert set(blocked) == {"A09-backend-pc_full_training", "A10-frame_count-f16",
                            "A10-frame_count-f32", "A10-frame_count-f48_64"}
    assert all(reason for reason in blocked.values())


def test_no_optimizer_group_is_empty_in_any_row(plan):
    report = audit_matrix(plan)
    for row in report["rows"]:
        if not row.get("audited"): continue
        assert row["optimizer_groups"], row["experiment_id"]
        for group in row["optimizer_groups"]:
            assert group["parameters"] > 0, (row["experiment_id"], group["name"])
