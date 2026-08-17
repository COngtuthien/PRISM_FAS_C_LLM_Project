"""SEARCH_PLAN: the frozen envelope a bounded source search may explore.

§15.2.2 and L.6 both require the same thing before any candidate executes: the
inherited anchors, the candidate grid, the evaluation order, the source-only
metric tuple and the tie-break must already be materialized into an
identity-bearing plan. A search that decided its own candidate set as it went
would be unbounded discretion wearing a search's clothes.

So a plan is built, hashed and written *first*, and the engine in
`coordinate.py` can only ever execute what the plan already declares. Two things
follow from that and are enforced here rather than left to the caller:

* **Candidates come from the anchor, not from the author.** A coordinate's
  candidate set is ``anchor x multipliers``, clipped to the coordinate's legal
  range, deduplicated and sorted ascending by canonical value — §15.2.2 says
  "evaluate in ascending canonical value order", so the order is a property of
  the plan rather than of whoever iterates it.
* **An absent scalar is skipped, not invented.** §15.2.3 is explicit for GPAT
  and the same rule reads naturally for the detector envelope: a coordinate
  whose anchor cannot be resolved from the inherited configuration is recorded
  as inapplicable, with the reason, and contributes no trials. Substituting a
  plausible default would be inventing an anchor the spec says does not exist.

The plan holds no scientific constant of its own. Multipliers, coordinate order,
selection tuples and tie-breaks are transcribed from the spec sections named on
each declaration, and the anchors are read from inherited configuration at build
time and recorded with their source.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "prism-search-plan-v1"

#: §15.2.2, learning rate / weight decay / active scalar loss weights.
MULTIPLIERS_HALF_ONE_TWO: tuple[float, ...] = (0.5, 1.0, 2.0)
#: §15.2.2, warm-up fraction/steps.
MULTIPLIERS_HALF_ONE_ONEHALF: tuple[float, ...] = (0.5, 1.0, 1.5)

#: §15.2.2: warm-up fraction is clipped to this range after multiplication.
WARMUP_FRACTION_RANGE: tuple[float, float] = (0.0, 0.20)

#: §15.2.2, verbatim coordinate order for the detector/loss envelope. The
#: K=4-only weights are appended by the builder when that variant is active.
DETECTOR_COORDINATE_ORDER: tuple[str, ...] = (
    "learning_rate", "weight_decay", "warmup", "lambda_syn", "lambda_local",
    "lambda_MIL", "lambda_P", "lambda_risk")

#: §15.2.3, the GPAT scalars, in the order the spec lists them.
GPAT_COORDINATE_ORDER: tuple[str, ...] = (
    "learning_rate", "weight_decay", "residual_loss_weight",
    "identity_preservation_weight", "geometry_preservation_weight")

#: §15.4, P1/P2 checkpoint tuple, minimized lexicographically.
P1P2_SELECTION_TUPLE: tuple[str, ...] = (
    "video_ACER", "video_BPCER", "NLL", "ECE", "epoch")

#: §15.4, P3-ready checkpoint tuple over CASIA-dev and MSU-dev, equal weight.
P3_READY_SELECTION_TUPLE: tuple[str, ...] = (
    "mean_domain_video_ACER", "max_domain_video_ACER", "mean_domain_video_BPCER",
    "mean_domain_NLL", "mean_domain_ECE", "epoch")

#: §15.2.3, the GPAT selection tuple. The leading flag is the hard-invariant
#: failure, which is why any failing configuration ranks after every passing one.
GPAT_SELECTION_TUPLE: tuple[str, ...] = (
    "hard_invariant_failure", "neutral_support_validation_objective",
    "identity_drift", "low_frequency_geometry_drift", "outside_mask_error")

#: §15.2.2 / §15.4: the only tie-break, applied after every numeric field.
CANONICAL_TIE_BREAK = "canonical_config_sha256_ascending"

#: §15.2.2 search budget/deadline sentence, transcribed per milestone.
LOCK_DEADLINES: dict[str, str] = {
    "C4": "the GPAT envelope closes at C4 before C5",
    "C6": "the quality-gate profile closes at C6 before C8",
    "C7": "the detector/loss envelope closes at C7 before C8",
}


class SearchPlanError(ValueError):
    """A plan cannot be built as declared, or contradicts its own envelope."""


def canonical_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """The tie-break identity of one candidate configuration.

    Every numeric field is rounded before hashing so a float repr difference on
    another platform cannot reorder a tie-break that the spec requires to be
    deterministic.
    """
    return sha256_text(canonical_text(_round(dict(config))))


def _round(value: Any, decimals: int = 12) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, decimals) + 0.0
    if isinstance(value, dict):
        return {key: _round(item, decimals) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(item, decimals) for item in value]
    return value


@dataclass(frozen=True)
class Coordinate:
    """One searchable scalar, its anchor and the exact values it may take."""

    name: str
    anchor: float | None
    multipliers: tuple[float, ...]
    #: Legal range applied after multiplication, when the spec declares one.
    clip: tuple[float, float] | None = None
    #: Non-negativity, declared per §15.2.2 ("no negative values").
    non_negative: bool = True
    #: Why this coordinate contributes no trials, when it does not.
    skip_reason: str = ""
    anchor_source: str = ""
    spec_clause: str = ""

    @property
    def applicable(self) -> bool:
        return self.anchor is not None and not self.skip_reason

    @property
    def candidates(self) -> tuple[float, ...]:
        """``anchor x multipliers``, clipped, deduplicated, ascending.

        §15.2.2 fixes two special cases and both are here rather than in the
        engine: an anchor of exactly zero collapses the candidate set to {0}
        (multiplying zero cannot explore anything), and evaluation order is
        ascending canonical value.
        """
        if not self.applicable:
            return ()
        anchor = float(self.anchor or 0.0)
        if anchor == 0.0:
            return (0.0,)
        values = {self._clip(anchor * multiplier) for multiplier in self.multipliers}
        return tuple(sorted(values))

    def _clip(self, value: float) -> float:
        if self.non_negative:
            value = max(0.0, value)
        if self.clip is not None:
            low, high = self.clip
            value = min(max(value, low), high)
        return round(value, 12) + 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "anchor": self.anchor,
            "anchor_source": self.anchor_source,
            "multipliers": list(self.multipliers),
            "clip": list(self.clip) if self.clip else None,
            "non_negative": self.non_negative,
            "applicable": self.applicable,
            "skip_reason": self.skip_reason,
            "candidates": list(self.candidates),
            "spec_clause": self.spec_clause,
        }


@dataclass(frozen=True)
class SearchPlan:
    """A frozen, identity-bearing search envelope.

    `identity` covers the envelope and nothing else — not the clock, not the
    machine, not the profile that will execute it — so the same plan built on the
    collaborator's GPU hashes to the same value it hashes to here. That is what
    lets a later full run prove it executed *this* plan.
    """

    plan_id: str
    milestone: str
    coordinates: tuple[Coordinate, ...]
    selection_tuple: tuple[str, ...]
    base_config: dict[str, Any] = field(default_factory=dict)
    tie_break: str = CANONICAL_TIE_BREAK
    one_pass: bool = True
    revisit_permitted: bool = False
    lock_deadline: str = ""
    expansion_policy: str = (
        "expanding a candidate set or starting a second pass is "
        "USER_APPROVAL_REQUIRED (§15.2.2)")
    spec_clause: str = ""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = [coordinate.name for coordinate in self.coordinates]
        if len(names) != len(set(names)):
            raise SearchPlanError(
                f"plan {self.plan_id!r} declares a coordinate twice: {names}")
        if not self.selection_tuple:
            raise SearchPlanError(
                f"plan {self.plan_id!r} declares no selection tuple; L.6 requires the "
                "winner to be chosen by a frozen metric tuple")

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return tuple(coordinate.name for coordinate in self.coordinates)

    @property
    def active_coordinates(self) -> tuple[Coordinate, ...]:
        return tuple(item for item in self.coordinates if item.applicable)

    @property
    def total_trials(self) -> int:
        return sum(len(item.candidates) for item in self.active_coordinates)

    def identity_material(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "milestone": self.milestone,
            "coordinate_order": list(self.coordinate_order),
            "coordinates": [item.as_dict() for item in self.coordinates],
            "selection_tuple": list(self.selection_tuple),
            "tie_break": self.tie_break,
            "one_pass": self.one_pass,
            "revisit_permitted": self.revisit_permitted,
            "base_config": _round(dict(self.base_config)),
            "lock_deadline": self.lock_deadline,
        }

    @property
    def identity(self) -> str:
        return sha256_text(canonical_text(self.identity_material()))

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_material(),
            "search_plan_identity": self.identity,
            "total_declared_trials": self.total_trials,
            "expansion_policy": self.expansion_policy,
            "spec_clause": self.spec_clause,
            "notes": list(self.notes),
            "retention_policy": (
                "every attempted configuration is retained: PASS, FAIL, DIVERGED and "
                "INTERRUPTED trials all stay addressable and losing configs are never "
                "deleted after a winner exists (L.6, L.8)"),
        }

    def config_for(self, coordinate_name: str, value: float,
                   current: Mapping[str, Any]) -> dict[str, Any]:
        """The full candidate config: the current best with one coordinate moved.

        §15.2.2: "at each coordinate evaluate the unique candidate values while
        all other coordinates remain at the current best".
        """
        config = dict(self.base_config)
        config.update(dict(current))
        config[coordinate_name] = value
        return config


#: The three ways an anchor lookup can end. `AMBIGUOUS` exists because §15.2.2
#: says an anchor that is not uniquely inherited is USER_APPROVAL_REQUIRED — it
#: is not a value to guess at, and it is not the same as an absent one.
RESOLVED, ABSENT, AMBIGUOUS = "RESOLVED", "ABSENT", "AMBIGUOUS"


@dataclass(frozen=True)
class AnchorResolution:
    """What the inherited configuration says about one coordinate's anchor.

    Kept separate from `Coordinate` so the audit can report *why* a coordinate
    was skipped in the vocabulary the spec uses. A skipped-because-absent scalar
    is a legitimate, spec-directed outcome; a skipped-because-ambiguous one is a
    decision the user still owes, and collapsing them would hide the second.
    """

    name: str
    state: str
    value: float | None = None
    path: str = ""
    candidate_paths: tuple[str, ...] = ()
    detail: str = ""

    @property
    def unique(self) -> bool:
        return self.state == RESOLVED

    @property
    def needs_user_decision(self) -> bool:
        return self.state == AMBIGUOUS

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state, "value": self.value,
                "path": self.path, "candidate_paths": list(self.candidate_paths),
                "detail": self.detail,
                "classification": ("USER_APPROVAL_REQUIRED" if self.needs_user_decision
                                   else "FROZEN_INHERITED_ANCHOR" if self.unique
                                   else "NOT_APPLICABLE")}


def resolve_anchors(source: Mapping[str, Any],
                    candidates: Mapping[str, Sequence[str]]) -> dict[str, AnchorResolution]:
    """Resolve each coordinate's anchor against every path it might live at.

    Exactly one hit is an anchor. Zero is an absent scalar, which §15.2.3 says to
    skip rather than invent. More than one is the case the inherited M8/M9
    configurations actually hit — they declare a learning rate per component
    (encoder / recipe / generator, backbone / head) rather than one — and the
    spec's answer there is a user decision, not a preference expressed in code.
    """
    resolutions: dict[str, AnchorResolution] = {}
    for name, paths in candidates.items():
        hits = [(path, resolve_anchor(source, path)) for path in paths]
        found = [(path, value) for path, value in hits if value is not None]
        if len(found) == 1:
            path, value = found[0]
            resolutions[name] = AnchorResolution(name=name, state=RESOLVED, value=value,
                                                 path=path,
                                                 candidate_paths=tuple(paths))
        elif not found:
            resolutions[name] = AnchorResolution(
                name=name, state=ABSENT, candidate_paths=tuple(paths),
                detail=("no inherited scalar exists at any declared path; §15.2.3 skips an "
                        "absent scalar rather than inventing one"))
        else:
            resolutions[name] = AnchorResolution(
                name=name, state=AMBIGUOUS, candidate_paths=tuple(paths),
                path=", ".join(path for path, _value in found),
                detail=("the inherited configuration declares more than one candidate "
                        f"scalar ({', '.join(f'{p}={v}' for p, v in found)}); §15.2.2 "
                        "requires a uniquely inherited anchor and classifies a "
                        "non-unique one as USER_APPROVAL_REQUIRED"))
    return resolutions


def coordinate_from_resolution(resolution: AnchorResolution, *,
                               multipliers: Sequence[float],
                               clip: tuple[float, float] | None = None,
                               spec_clause: str = "", active: bool = True,
                               inactive_reason: str = "") -> Coordinate:
    """Turn one anchor resolution into a plan coordinate, keeping its reason."""
    if not active:
        return Coordinate(name=resolution.name, anchor=None, multipliers=tuple(multipliers),
                          clip=clip, skip_reason=inactive_reason or
                          f"{resolution.name} is not an active term in this variant",
                          anchor_source=resolution.path, spec_clause=spec_clause)
    if not resolution.unique:
        return Coordinate(name=resolution.name, anchor=None, multipliers=tuple(multipliers),
                          clip=clip, skip_reason=f"{resolution.state}: {resolution.detail}",
                          anchor_source=resolution.path or ", ".join(resolution.candidate_paths),
                          spec_clause=spec_clause)
    return Coordinate(name=resolution.name, anchor=resolution.value,
                      multipliers=tuple(multipliers), clip=clip,
                      anchor_source=resolution.path, spec_clause=spec_clause)


def resolve_anchor(source: Mapping[str, Any], path: str) -> float | None:
    """Read a dotted anchor path out of inherited configuration.

    Returns None when the scalar does not exist, which the caller turns into an
    inapplicable coordinate. Deliberately not a default-returning lookup: the
    difference between "the anchor is 0.0" and "there is no such anchor" decides
    whether a coordinate is searched at all.
    """
    cursor: Any = source
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    if isinstance(cursor, bool) or not isinstance(cursor, (int, float)):
        return None
    return float(cursor)


def build_coordinate(name: str, *, source: Mapping[str, Any], path: str,
                     multipliers: Sequence[float], clip: tuple[float, float] | None = None,
                     spec_clause: str = "", active: bool = True,
                     inactive_reason: str = "") -> Coordinate:
    """One coordinate, resolved against inherited configuration.

    Two independent ways a coordinate becomes inapplicable, kept separate so the
    audit says which happened: the anchor is not resolvable at all, or the term
    it belongs to is inactive in this variant ("skip inactive terms", §15.2.2).
    """
    if not active:
        return Coordinate(name=name, anchor=None, multipliers=tuple(multipliers),
                          clip=clip, skip_reason=inactive_reason or
                          "the loss term is inactive in this variant; §15.2.2 skips "
                          "inactive terms rather than inventing a weight for them",
                          anchor_source=path, spec_clause=spec_clause)
    anchor = resolve_anchor(source, path)
    if anchor is None:
        return Coordinate(name=name, anchor=None, multipliers=tuple(multipliers),
                          clip=clip, skip_reason=(
                              f"no inherited anchor at {path!r}; §15.2.3 requires an absent "
                              "scalar to be skipped, not invented"),
                          anchor_source=path, spec_clause=spec_clause)
    return Coordinate(name=name, anchor=anchor, multipliers=tuple(multipliers),
                      clip=clip, anchor_source=path, spec_clause=spec_clause)


#: Where each GPAT scalar might live in the inherited M8 configuration. More than
#: one path per coordinate is deliberate: the resolver reports ambiguity instead
#: of preferring whichever entry happens to be listed first.
GPAT_ANCHOR_PATHS: dict[str, tuple[str, ...]] = {
    "learning_rate": ("optimizer.learning_rate", "optimizer.encoder_lr",
                      "optimizer.recipe_lr", "optimizer.generator_lr"),
    "weight_decay": ("optimizer.weight_decay",),
    "residual_loss_weight": ("loss.residual",),
    "identity_preservation_weight": ("loss.identity",),
    "geometry_preservation_weight": ("loss.geometry", "loss.geometry_preservation"),
}

#: Where each detector scalar might live in the inherited M9 configuration.
DETECTOR_ANCHOR_PATHS: dict[str, tuple[str, ...]] = {
    "learning_rate": ("optimizer.learning_rate", "optimizer.backbone_lr",
                      "optimizer.head_lr"),
    "weight_decay": ("optimizer.weight_decay",),
    "warmup": ("scheduler.warmup_fraction",),
    "lambda_syn": ("loss.weights.lambda_syn",),
    "lambda_local": ("loss.weights.lambda_local",),
    "lambda_MIL": ("loss.weights.lambda_MIL",),
    "lambda_P": ("loss.weights.lambda_P",),
    "lambda_risk": ("loss.weights.lambda_risk",),
    # K=4-only weights: the manifold-dependent terms, active only when the
    # explicit K=4 secondary variant is enabled (§13.2, §15.2.2).
    "lambda_M": ("loss.weights.lambda_M",),
    "lambda_out": ("loss.weights.lambda_out",),
    "lambda_clean": ("loss.weights.lambda_clean",),
}

#: The K=4-only scalar weights, appended after `lambda_risk` exactly where the
#: §15.2.2 coordinate order puts them.
K4_ONLY_WEIGHTS: tuple[str, ...] = ("lambda_M", "lambda_out", "lambda_clean")


def gpat_search_plan(anchors: Mapping[str, Any], *,
                     anchor_paths: Mapping[str, Sequence[str]] | None = None,
                     base_config: Mapping[str, Any] | None = None
                     ) -> tuple[SearchPlan, dict[str, AnchorResolution]]:
    """The §15.2.3 neutral-GPAT envelope: one pass, five scalars, x{0.5,1,2}.

    Returns the plan and the anchor resolutions beside it, because a caller
    needs to know *why* a coordinate is inactive before deciding whether the
    envelope is executable at all.
    """
    paths = {name: tuple(value) for name, value
             in dict(anchor_paths or GPAT_ANCHOR_PATHS).items()}
    resolutions = resolve_anchors(anchors, paths)
    coordinates = tuple(
        coordinate_from_resolution(resolutions[name],
                                   multipliers=MULTIPLIERS_HALF_ONE_TWO,
                                   spec_clause="§15.2.3")
        for name in GPAT_COORDINATE_ORDER)
    plan = SearchPlan(
        plan_id="c4_gpat_coordinate_v1",
        milestone="C4",
        coordinates=coordinates,
        selection_tuple=GPAT_SELECTION_TUPLE,
        base_config=dict(base_config or {}),
        lock_deadline=LOCK_DEADLINES["C4"],
        spec_clause="§15.2.3",
        notes=("architecture and support ontology are FROZEN by §8.3 and C4; this plan "
               "searches scalars only and can express no architectural change",
               "any hard geometry/identity invariant failure ranks after every passing "
               "configuration, which is why it is the leading tuple field"))
    return plan, resolutions


def detector_search_plan(anchors: Mapping[str, Any], *,
                         active_terms: Mapping[str, bool] | None = None,
                         anchor_paths: Mapping[str, Sequence[str]] | None = None,
                         base_config: Mapping[str, Any] | None = None,
                         selection_tuple: Sequence[str] = P3_READY_SELECTION_TUPLE,
                         k4_weights: Sequence[str] = ()
                         ) -> tuple[SearchPlan, dict[str, AnchorResolution]]:
    """The §15.2.2 detector/loss envelope in its exact coordinate order.

    `active_terms` carries the variant's own answer to "is this loss term
    active": a manifold-OFF Track R has no L_real/L_out/L_clean, and §15.2.2
    says to skip inactive terms. The K=4-only scalar weights are appended last,
    exactly where the spec's order puts them, and only when that variant is on.
    """
    paths = {name: tuple(value) for name, value
             in dict(anchor_paths or DETECTOR_ANCHOR_PATHS).items()}
    order = (*DETECTOR_COORDINATE_ORDER, *k4_weights)
    resolutions = resolve_anchors(anchors, {name: paths.get(name, (name,))
                                            for name in order})
    active = dict(active_terms or {})
    coordinates: list[Coordinate] = []
    for name in order:
        multipliers = (MULTIPLIERS_HALF_ONE_ONEHALF if name == "warmup"
                       else MULTIPLIERS_HALF_ONE_TWO)
        clip = WARMUP_FRACTION_RANGE if name == "warmup" else None
        k4 = name in k4_weights
        coordinates.append(coordinate_from_resolution(
            resolutions[name], multipliers=multipliers, clip=clip,
            spec_clause="§15.2.2 (K=4-only)" if k4 else "§15.2.2",
            active=active.get(name, True),
            inactive_reason=("the K=4 manifold variant is not active" if k4 else
                             f"{name} is not an active loss term in this variant")))
    plan = SearchPlan(
        plan_id="c7_detector_coordinate_v1",
        milestone="C7",
        coordinates=tuple(coordinates),
        selection_tuple=tuple(selection_tuple),
        base_config=dict(base_config or {}),
        lock_deadline=LOCK_DEADLINES["C7"],
        spec_clause="§15.2.2",
        notes=("optimizer family is FROZEN to the inherited Version-B family; this plan "
               "cannot express a family switch",
               "batch/microbatch is ENGINEERING_ADAPTIVE and is deliberately absent from "
               "this envelope: effective scientific batch composition is fixed"))
    return plan, resolutions


def anchor_resolution_report(resolutions: Mapping[str, AnchorResolution]) -> dict[str, Any]:
    """A machine-readable account of which anchors are executable, and which owe
    a user decision before the full profile may run this envelope."""
    rows = [item.as_dict() for item in resolutions.values()]
    ambiguous = sorted(name for name, item in resolutions.items()
                       if item.needs_user_decision)
    absent = sorted(name for name, item in resolutions.items() if item.state == ABSENT)
    return {
        "coordinates": rows,
        "resolved": sorted(name for name, item in resolutions.items() if item.unique),
        "absent": absent,
        "ambiguous": ambiguous,
        "executable_under_full": not ambiguous,
        "blocking_reason": (
            "" if not ambiguous else
            f"{ambiguous} have no uniquely inherited anchor. §15.2.2 classifies a "
            "non-unique inherited scalar as USER_APPROVAL_REQUIRED, so the full profile "
            "cannot execute this envelope until the user binds each one to a named "
            "scalar. Engineering readiness does not require that decision; scientific "
            "execution does."),
        "absent_note": (
            "an absent scalar is skipped rather than invented (§15.2.3); it is not a "
            "blocker" if absent else ""),
    }


#: Builders keyed by milestone, so an adapter asks for its own plan by name
#: rather than importing a specific function and drifting from the registry.
PLAN_BUILDERS: dict[str, Callable[..., tuple[SearchPlan, dict[str, AnchorResolution]]]] = {
    "C4": gpat_search_plan,
    "C7": detector_search_plan,
}


__all__ = ["SCHEMA_VERSION", "MULTIPLIERS_HALF_ONE_TWO", "MULTIPLIERS_HALF_ONE_ONEHALF",
           "WARMUP_FRACTION_RANGE", "DETECTOR_COORDINATE_ORDER", "GPAT_COORDINATE_ORDER",
           "P1P2_SELECTION_TUPLE", "P3_READY_SELECTION_TUPLE", "GPAT_SELECTION_TUPLE",
           "CANONICAL_TIE_BREAK", "LOCK_DEADLINES", "SearchPlanError", "canonical_text",
           "sha256_text", "canonical_config_sha256", "Coordinate", "SearchPlan",
           "resolve_anchor", "build_coordinate", "gpat_search_plan",
           "detector_search_plan", "PLAN_BUILDERS", "RESOLVED", "ABSENT", "AMBIGUOUS",
           "AnchorResolution", "resolve_anchors", "coordinate_from_resolution",
           "anchor_resolution_report", "GPAT_ANCHOR_PATHS", "DETECTOR_ANCHOR_PATHS",
           "K4_ONLY_WEIGHTS"]
