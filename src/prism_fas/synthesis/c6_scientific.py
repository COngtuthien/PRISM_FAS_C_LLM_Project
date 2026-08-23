"""The scientific C6 pipeline: fit, gate, choose a profile, build matched banks.

Every numeric decision here is made by a canonical module. This file owns the
Version-C sequencing and one thing the canonical modules did not have: the
matched-feasibility test, which is stronger than the arm-count test that
preceded it.

`ArmFeasibility` asks whether an arm has 512 accepted Physics and 512 accepted
GPAT. That is necessary and, since C6_MATCHED_BANK_SELECTOR_V1 was frozen, no
longer sufficient: the three arms must also be able to fill ONE identical
source-domain quota vector per route. An arm can hold 600 accepted Physics
candidates and still be unable to match the others if they are concentrated in
the wrong dataset. The engineering rehearsal keeps the older, weaker test
unchanged — this is a scientific-path addition, not a redefinition.

Nothing in this module opens a target artifact, and the calibration reads
`source_train` only, through `quality_calibration.calibrate`'s own audited
`SampleStore`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:                    # pragma: no cover - typing only
    from .quality_gate import Thresholds

from . import c6_matched_bank as selector
from .c5_source_pair_plan import ARMS, GPAT, PHYSICS

SCHEMA_VERSION = "prism-c6-scientific-v1"

#: The gate profiles §11.4 preregisters. Exactly these three, in this order.
from .gate_profiles import PROFILE_ORDER  # noqa: E402  (re-exported deliberately)


class ScientificGateError(RuntimeError):
    """The scientific C6 gate cannot proceed under the frozen contract."""


# --- matched feasibility -----------------------------------------------------

@dataclass(frozen=True)
class MatchedFeasibility:
    """Whether ONE profile can produce three matched banks.

    Typed rather than a bare bool because the two ways of failing mean different
    things and both must reach the report: an arm short of 512 on a route, and
    three arms that individually have enough but cannot agree on a common
    source-domain quota.
    """

    profile: str
    arm_route_counts: dict[str, dict[str, int]]
    route_quotas: dict[str, dict[str, Any]]
    arms_meet_route_floor: bool
    common_quota_feasible: bool

    @property
    def feasible(self) -> bool:
        """Cardinality and matched feasibility. Reliability is NOT an input.

        C6_RELIABILITY_SEQUENCE = OPTION_B_POST_SELECTION_CLOSURE_GATE: the
        profile is chosen on the frozen cardinality contract alone and frozen
        immediately, and the bank-level probe then runs on the FINAL banks as a
        closure gate. Letting a probe result in here would make it a selection
        objective, which is the reading §3.1.1 rules out.
        """
        return self.arms_meet_route_floor and self.common_quota_feasible

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.profile,
                "arm_route_counts": {arm: dict(counts) for arm, counts
                                     in self.arm_route_counts.items()},
                "route_quotas": dict(self.route_quotas),
                "arms_meet_route_floor": self.arms_meet_route_floor,
                "common_quota_feasible": self.common_quota_feasible,
                "feasible": self.feasible,
                "reliability_used_for_selection": False,
                "rule": ("§11.4 route floor AND one identical source-domain quota "
                         "vector per route across RND/DET/LLM. Bank-level "
                         "reliability is a post-selection closure gate, not a "
                         "selection objective")}


def assess_profile(profile: str, accepted_by_arm: Mapping[str, Sequence[
                       selector.SelectableCandidate]],
                   plans: Mapping[str, Mapping[str, Any]]) -> MatchedFeasibility:
    """One profile's feasibility: the route floor and the common domain quota.

    Reliability is deliberately absent from the signature. It used to be a
    parameter that defaulted to True — a fail-open — and under Option B it is
    not a selection input at all, so the safest shape is one that cannot receive
    it.
    """
    counts = {
        arm: {route: sum(1 for candidate in candidates if candidate.route == route)
              for route in selector.ROUTES}
        for arm, candidates in accepted_by_arm.items()}
    floor_met = (set(counts) == set(ARMS) and all(
        counts[arm][route] >= selector.PER_ROUTE
        for arm in ARMS for route in selector.ROUTES))

    quotas = selector.route_quotas(plans, accepted_by_arm)
    return MatchedFeasibility(
        profile=profile, arm_route_counts=counts,
        route_quotas={route: quota.as_dict() for route, quota in quotas.items()},
        arms_meet_route_floor=floor_met,
        common_quota_feasible=selector.matched_feasible(quotas))


@dataclass
class ProfileDecision:
    """Which profile was selected, and every profile that was refused and why."""

    selected: str | None
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.selected is None

    def as_dict(self) -> dict[str, Any]:
        return {"selected_profile": self.selected,
                "profile_order": list(PROFILE_ORDER),
                "evaluations": list(self.evaluations),
                "c6_scientific_failed": self.failed,
                "on_failure": ("§11.4: if no profile qualifies, C6 FAILS. The gate "
                               "is never widened, no arm gets its own threshold, "
                               "the target distribution is not altered, the "
                               "selector is not changed and no candidate is "
                               "regenerated")}


def select_strictest_profile(assessments: Sequence[MatchedFeasibility]) -> ProfileDecision:
    """The strictest matched-feasible, reliable profile — in the frozen order.

    Every profile is assessed and recorded even after one qualifies, because a
    refusal is evidence: it says how close the next-stricter profile came.
    """
    by_name = {item.profile: item for item in assessments}
    missing = [name for name in PROFILE_ORDER if name not in by_name]
    if missing:
        raise ScientificGateError(
            f"§11.4 requires exactly {PROFILE_ORDER}; missing {missing}")
    if len(assessments) != len(PROFILE_ORDER):
        raise ScientificGateError(
            f"§11.4 evaluates exactly three profiles; got {len(assessments)}")

    decision = ProfileDecision(selected=None)
    for name in PROFILE_ORDER:                     # STRICT, NOMINAL, PERMISSIVE
        item = by_name[name]
        decision.evaluations.append(item.as_dict())
        if item.feasible and decision.selected is None:
            decision.selected = name
    return decision


# --- calibration and candidate evaluation ------------------------------------

def fit_nominal_calibration(package_root: Path, config: Mapping[str, Any],
                            backends: Any, **limits: Any) -> dict[str, Any]:
    """Fit NOMINAL from the source_train benign population. §11.4, at C6.

    Delegated wholesale to `quality_calibration.calibrate`, which opens
    `source_train` through an audited `SampleStore` and refuses a forbidden
    split. This is a C6 OUTPUT: nothing upstream supplies it, and it must never
    be hand-written.
    """
    from .quality_calibration import calibrate

    return calibrate(Path(package_root), dict(config), backends, **limits)


def build_common_profiles(nominal: Mapping[str, float], *,
                          nominal_source: str) -> dict[str, Any]:
    """STRICT / NOMINAL / PERMISSIVE by the frozen §11.4 formulas.

    One threshold set per profile, applied unchanged to all three arms. The
    formulas live in `gate_profiles.derive_profile` and are not restated here.
    """
    from .gate_profiles import build_profiles

    return build_profiles(nominal, nominal_source=nominal_source)


def reconstruct_discrete(directory: Path, original: Any) -> Any:
    """Rebuild a `DiscreteResult` from a stored C5 candidate.

    C5 wrote the three payloads through `finalize_discrete` and hashed the exact
    bytes; this decodes those same bytes back with the canonical decoders rather
    than re-rendering, so what C6 measures is literally what C5 produced and what
    C7 will train on. Re-rendering to obtain the metrics would measure a second
    generation that no lock covers.
    """
    import numpy as np

    from . import c5_raw_generation as raw
    from .synthetic_bank import (ARTIFACT_MAP_KEY, DiscreteResult, decode_npz,
                                 decode_png, to_uint8)

    directory = Path(directory)
    image_png = (directory / raw.IMAGE_NAME).read_bytes()
    mask_png = (directory / raw.MASK_NAME).read_bytes()
    artifact_npz = (directory / raw.ARTIFACT_MAP_NAME).read_bytes()

    image_uint8 = decode_png(image_png)
    exact = decode_png(mask_png).astype(bool)
    if exact.ndim == 3:
        exact = exact[..., 0]
    artifact_map = decode_npz(artifact_npz, ARTIFACT_MAP_KEY)
    original_uint8 = to_uint8(np.asarray(original, dtype=np.float32))
    return DiscreteResult(
        image_uint8=image_uint8, original_uint8=original_uint8,
        exact_edit_mask=exact, artifact_map=artifact_map,
        image_png=image_png, mask_png=mask_png, artifact_map_npz=artifact_npz,
        outside_mask_max_error=0, requested_support_pixels=int(exact.sum()),
        exact_mask_pixels=int(exact.sum()))


def requested_support_for(store: Any, bank: Mapping[str, Any],
                          row: Mapping[str, Any]) -> Any:
    """The region support C5 asked for, rebuilt through the canonical builder.

    `exact_mask.png` records the pixels that actually changed, which is not the
    same set as the support the recipe requested — the difference is exactly what
    `support_overlap` measures. So the requested support is rebuilt with
    `synthetic_bank._support_masks`, the same memoized, deterministic function
    the physics engine and the GPAT route used at render time.
    """
    import numpy as np

    from ..recipes.compile import compile_recipe
    from .synthetic_bank import _recipe, _support_masks

    recipe = _recipe(dict(bank), row["recipe_id"])
    graph = compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"])
    masks = _support_masks(store, row["live_target_sample_id"], graph)
    return np.asarray(masks.operator_support_mask)[0].astype(bool), graph


#: The raw metric fields `quality_gate.evaluate` requires of its input. Taken
#: from the canonical gate rather than restated: these are exactly the keys it
#: reads before applying any threshold.
REQUIRED_RAW_METRICS: tuple[str, ...] = (
    "face_detection_score", "identity_cosine", "landmark_nme",
    "outside_mask_parsing_dice", "outside_mask_max_error",
    "measured_artifact_strength", "requested_artifact_strength",
    "fingerprint_score", "support_overlap")

#: Canonical diagnostics `CandidateEvaluator` also emits. Carried, never gated.
DIAGNOSTIC_RAW_METRICS: tuple[str, ...] = ("reference_detection_score",
                                           "landmark_detected")

#: Threshold-dependent outputs of the evaluator's own embedded NOMINAL pass.
#: They must never enter the measurement layer: a decision taken under one
#: threshold set cannot be reused for another, and carrying it would let the
#: calibration's NOMINAL leak into the STRICT and PERMISSIVE assessments.
THRESHOLD_DEPENDENT_FIELDS: tuple[str, ...] = (
    "accepted", "failed_gates", "gates", "q", "quality_components",
    "threshold_hash", "strength_bounds", "recipe_match")


def raw_metrics_of(result: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    """The raw metric map out of a `CandidateEvaluator` result, validated.

    `CandidateEvaluator.evaluate` returns `quality_gate.evaluate(...)` — a gate
    ENVELOPE whose raw measurements sit under `["metrics"]`, alongside an
    acceptance decision taken under the calibration's own NOMINAL thresholds.
    C6 wants the measurements and must not inherit that decision.

    Unwrapping is explicit and validated rather than best-effort: a missing
    `metrics` block or a missing required field fails closed here, where the
    shape is known, instead of surfacing three substages later as a
    `QualityGateError` about one absent key.
    """
    import math

    if "metrics" not in result:
        raise ScientificGateError(
            f"{candidate_id}: the evaluator result carries no 'metrics' block; "
            f"CandidateEvaluator returns a gate envelope and C6 measures from "
            f"its nested raw metrics")
    metrics = result["metrics"]
    if not isinstance(metrics, dict):
        raise ScientificGateError(
            f"{candidate_id}: 'metrics' is {type(metrics).__name__}, not a map")

    missing = [name for name in REQUIRED_RAW_METRICS if name not in metrics]
    if missing:
        raise ScientificGateError(
            f"{candidate_id}: the raw metric set is missing {missing}")
    for name in REQUIRED_RAW_METRICS:
        value = float(metrics[name])
        if not math.isfinite(value):
            raise ScientificGateError(
                f"{candidate_id}: raw metric {name!r} is not finite")

    keep = set(REQUIRED_RAW_METRICS) | set(DIAGNOSTIC_RAW_METRICS)
    unexpected = sorted(set(metrics) - keep)
    if unexpected:
        raise ScientificGateError(
            f"{candidate_id}: the raw metric set carries unrecognized fields "
            f"{unexpected}; C6 does not silently unwrap arbitrary nested objects")
    return {name: metrics[name] for name in sorted(metrics)}


def evaluate_pool(evaluator: Any, store: Any, bank: Mapping[str, Any], *,
                  candidate_root: Path, arm: str,
                  rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every generated candidate's RAW metric set, measured exactly once.

    Threshold-independent by construction: what is stored describes the
    candidate's bytes and nothing about any gate. The three profiles are applied
    afterwards by `gate_candidates`, so the pool is measured once and gated three
    times rather than measured three times.

    The evaluator's own embedded NOMINAL verdict is discarded here. It is an
    implementation side-effect of `CandidateEvaluator` returning a gate envelope,
    not a scientific decision, and reusing it would silently give NOMINAL a
    privileged position among the three profiles.
    """
    from . import c5_raw_generation as raw

    metrics: dict[str, dict[str, Any]] = {}
    for row in rows:
        directory = raw.candidate_dir(candidate_root, arm, row["candidate_id"])
        record = raw.read_record(directory / raw.RECORD_NAME) or {}
        if record.get("status") != raw.GENERATED:
            # A semantic failure has no bytes. It is never measured, never
            # accepted and never rejected; it stays retained provenance.
            continue
        original, _ = store.load(row["live_target_sample_id"])
        support, graph = requested_support_for(store, bank, row)
        discrete = reconstruct_discrete(directory, original)
        result = evaluator.evaluate(
            discrete, live_target_sample_id=row["live_target_sample_id"],
            requested_strength=float(graph.nodes[0].strength),
            requested_support=support)
        metrics[row["candidate_id"]] = raw_metrics_of(result, row["candidate_id"])
    return metrics


def gate_candidates(metrics_by_candidate: Mapping[str, Mapping[str, Any]],
                    thresholds: "Thresholds") -> list[dict[str, Any]]:
    """Apply ONE threshold set to every candidate. Binary for selection.

    `quality_gate.evaluate` returns the hard-gate verdict and the §11.2 weight
    `q`. Both are recorded; only `accepted` reaches the selector.

    `thresholds` must be a canonical `quality_gate.Thresholds`, not the raw
    mapping a `GateProfile` stores. The annotation used to be `Any`, which is how
    a `dict` reached `evaluate` and surfaced as `'dict' object has no attribute
    'tau_fd'` at the first gating call. `GateProfile.as_thresholds()` is the
    conversion; there is deliberately no dict-accepting compatibility path,
    because two accepted representations at the gating boundary is what allowed
    the mismatch in the first place.
    """
    from .quality_gate import Thresholds, evaluate

    if not isinstance(thresholds, Thresholds):
        raise ScientificGateError(
            f"gate_candidates requires a quality_gate.Thresholds, got "
            f"{type(thresholds).__name__}. Use GateProfile.as_thresholds(); the "
            f"raw mapping is for hashing and serialization only.")

    decisions: list[dict[str, Any]] = []
    for candidate_id in sorted(metrics_by_candidate):
        outcome = evaluate(dict(metrics_by_candidate[candidate_id]), thresholds)
        decisions.append({"candidate_id": candidate_id,
                          "accepted": bool(outcome["accepted"]),
                          "q": outcome.get("q"),
                          "failed_gates": outcome.get("failed_gates", [])})
    return decisions


def eligible_candidates(decisions: Sequence[Mapping[str, Any]],
                        pool: Mapping[str, selector.SelectableCandidate]
                        ) -> list[selector.SelectableCandidate]:
    """Accepted candidates only, carrying `q` for later training weighting.

    A C5 semantic failure never appears here: it has no payload, so it was never
    evaluated, so it holds no decision. Its record stays provenance.
    """
    eligible: list[selector.SelectableCandidate] = []
    for decision in decisions:
        if not decision["accepted"]:
            continue
        candidate = pool.get(decision["candidate_id"])
        if candidate is None:
            raise ScientificGateError(
                f"the gate accepted {decision['candidate_id']!r}, which is not a "
                "verified generated candidate in the C5 pool")
        eligible.append(selector.SelectableCandidate(
            candidate_id=candidate.candidate_id, arm=candidate.arm,
            route=candidate.route, source_domain=candidate.source_domain,
            recipe_id=candidate.recipe_id, recipe_ordinal=candidate.recipe_ordinal,
            live_target_sample_id=candidate.live_target_sample_id,
            base_position=candidate.base_position, q=decision.get("q")))
    return eligible


def provenance_closure(pool_candidate_ids: Sequence[str],
                       semantic_failure_ids: Sequence[str],
                       decisions: Sequence[Mapping[str, Any]],
                       selected_ids: Sequence[str]) -> dict[str, Any]:
    """Prove nothing was dropped: every planned slot lands in exactly one class.

    L.8 forbids winner-only cleanup, so this is checked rather than asserted:
    selected + accepted-but-not-selected + rejected + semantic failures must
    reconstruct the whole planned schedule.
    """
    accepted = {row["candidate_id"] for row in decisions if row["accepted"]}
    rejected = {row["candidate_id"] for row in decisions if not row["accepted"]}
    selected = set(selected_ids)
    failures = set(semantic_failure_ids)
    planned = set(pool_candidate_ids) | failures

    not_selected = accepted - selected
    covered = selected | not_selected | rejected | failures

    # The partition, proved rather than counted. The four classes must be
    # pairwise disjoint and must union to exactly the planned schedule: no
    # planned candidate silently lost, and none in two classes at once.
    classes = {"selected": selected, "accepted_not_selected": not_selected,
               "rejected": rejected, "semantic_failed": failures}
    overlaps: dict[str, list[str]] = {}
    names = sorted(classes)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = classes[left] & classes[right]
            if shared:
                overlaps[f"{left}&{right}"] = sorted(shared)[:16]

    # A semantic failure has no payload, so it can never have been measured and
    # can never carry a gate decision. One that does means the two evidence
    # sources disagree about what happened to that candidate.
    decided = accepted | rejected
    failures_with_decisions = sorted(failures & decided)

    partitioned = (covered == planned and not overlaps
                   and not failures_with_decisions
                   and not (selected - accepted))
    return {"planned": len(planned), "selected": len(selected),
            "accepted_not_selected": len(not_selected),
            "rejected": len(rejected), "semantic_failed": len(failures),
            "covered": len(covered),
            "closed": partitioned,
            "pairwise_disjoint": not overlaps,
            "category_overlaps": overlaps,
            "semantic_failures_carrying_a_gate_decision": failures_with_decisions[:16],
            "unaccounted": sorted(planned - covered)[:32],
            "selected_outside_accepted": sorted(selected - accepted)[:32],
            "partition": ("the frozen planned schedule partitions exactly into "
                          "selected + accepted_not_selected + rejected + "
                          "semantic_failed, pairwise disjoint"),
            "rule": ("no loser cleanup: rejected candidates, accepted-but-not-"
                     "selected candidates and C5 semantic failures are all "
                     "retained and addressable")}


def bank_lock_payload(*, arm: str, bank: Mapping[str, Any],
                      selector_contract: Mapping[str, Any],
                      profile: str, threshold_identity: str,
                      c5_pool_lock_sha256: str,
                      provenance: Mapping[str, Any]) -> dict[str, Any]:
    """One arm's synthetic BANK_LOCK: what is in the bank and where it came from."""
    return {
        "schema_version": SCHEMA_VERSION, "generated_at_utc": None,
        "arm": arm, "is_scientific_lock": True,
        "final_bank_size": bank["size"], "by_route": dict(bank["by_route"]),
        "exposure": dict(bank["exposure"]),
        "selected_set_sha256": bank["selected_set_sha256"],
        "selected": list(bank["selected"]),
        "quality_profile": profile,
        "quality_threshold_identity": threshold_identity,
        "selector_identity_sha256": selector_contract["selector_identity_sha256"],
        "selector_name": selector_contract["selector_name"],
        "c5_pool_lock_sha256": c5_pool_lock_sha256,
        "provenance_closure": dict(provenance),
        "q_used_for_selection": False,
        "q_purpose": "§11.2 synthetic sample-quality TRAINING WEIGHT only",
        # BA_sep is staged at the detector-level barrier. Recorded as pending,
        # never as a pass. This does NOT make the bank unusable for C7/C8 source
        # training; it means the P3 path stays locked until the barrier resolves.
        "ba_sep_stage": "C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C",
        "ba_sep_used_for_profile_selection": False,
        "c6_bank_level_ba_probe": "not_applicable",
        "detector_reliability_pending": True,
        "usable_for_c7_c8_source_training": True,
        "target_access": 0,
        "no_target_capability_proof": {"target_roots_mounted": [],
                                       "target_labels_resolved": 0},
    }


def threshold_identity(thresholds: Any) -> str:
    """One identity per threshold set, shared by all three arms under a profile."""
    payload = thresholds.as_dict() if hasattr(thresholds, "as_dict") else dict(thresholds)
    import json

    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


def candidate_pool(plans: Mapping[str, Mapping[str, Any]]
                   ) -> dict[str, dict[str, selector.SelectableCandidate]]:
    """The selector's view of every planned slot, per arm, from the frozen plans.

    Built from the plan rather than from the records, so `source_domain`,
    `base_position` and `live_target_sample_id` come from the schedule all three
    arms share. `q` is left unset here; it arrives with the gate decision.
    """
    pool: dict[str, dict[str, selector.SelectableCandidate]] = {}
    for arm in ARMS:
        plan = plans[arm]
        pool[arm] = {row["candidate_id"]: selector.SelectableCandidate(
            candidate_id=row["candidate_id"], arm=arm, route=row["route"],
            source_domain=str(row[selector.SOURCE_DOMAIN_PLAN_FIELD]),
            recipe_id=row["recipe_id"], recipe_ordinal=int(row["recipe_ordinal"]),
            live_target_sample_id=row["live_target_sample_id"],
            base_position=int(row["position"])) for row in plan["candidates"]}
    return pool


__all__ = ["SCHEMA_VERSION", "PROFILE_ORDER", "ScientificGateError",
           "MatchedFeasibility", "assess_profile", "ProfileDecision",
           "select_strictest_profile", "fit_nominal_calibration",
           "build_common_profiles", "gate_candidates", "eligible_candidates",
           "provenance_closure", "bank_lock_payload", "threshold_identity",
           "reconstruct_discrete", "requested_support_for", "evaluate_pool",
           "candidate_pool", "raw_metrics_of", "REQUIRED_RAW_METRICS",
           "DIAGNOSTIC_RAW_METRICS", "THRESHOLD_DEPENDENT_FIELDS"]
