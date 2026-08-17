"""The approved learning-rate anchor interpretation (§15.2.2, §15.2.3).

`learning_rate` names no scalar in either component. C4's GPAT has three
per-group learning rates and C7's detector has two, all live under one AdamW, so
"the inherited anchor" is a *vector* rather than a number. The user approved
interpretation B: one learning-rate coordinate whose candidate is a multiplier
applied to every active group, holding the inherited ratios fixed.

This module turns that decision into something the search engine can execute. It
holds no scientific constant of its own — every anchor, ratio and multiplier is
read from `configs/search/lr_anchor_decision.yaml`, which is the decision record.

Two properties are worth stating because they are what make the interpretation
safe rather than merely convenient:

* **m = 1.0 is the inherited configuration exactly.** The anchor trial reproduces
  Version B, so the search still starts where §15.2.2 says it starts.
* **The ratio is a property of the plan, not of the evaluator.** `lr_for_groups`
  derives every group's rate from the frozen anchor vector, so an evaluator
  cannot accidentally search one group and freeze another.

Track G is present here for completeness and carries no multiplier: it has one
applicable scalar, which the frozen rules already resolve.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "prism-lr-anchor-decision-v1"
DECISION_CONFIG = Path("configs") / "search" / "lr_anchor_decision.yaml"

#: Interpretation ids, as the dossier named them.
COMMON_MULTIPLIER = "B_common_multiplier"
UNIQUE_ANCHOR = "UNIQUE_INHERITED_ANCHOR"


class LRDecisionError(ValueError):
    """The decision record is missing, unapproved or self-inconsistent."""


@dataclass(frozen=True)
class LRAnchorDecision:
    """One component's approved learning-rate interpretation."""

    component: str
    interpretation: str
    anchor_vector: dict[str, float]
    multipliers: tuple[float, ...]
    coordinate_name: str
    preserved_ratio: tuple[float, ...]
    parameter_groups: tuple[str, ...]
    compliance_class: str
    anchor_source: str
    rationale: str = ""

    @property
    def searches_a_multiplier(self) -> bool:
        return self.interpretation == COMMON_MULTIPLIER

    @property
    def candidates(self) -> tuple[float, ...]:
        """The multiplier values, ascending. Track G contributes none of its own."""
        return tuple(sorted(self.multipliers)) if self.searches_a_multiplier else ()

    def lr_for_groups(self, multiplier: float) -> dict[str, float]:
        """Every group's learning rate at one multiplier.

        The inherited ratio survives by construction: each group is scaled by the
        same factor, so no evaluator can search one group while silently freezing
        another.
        """
        return {name: round(float(anchor) * float(multiplier), 15) + 0.0
                for name, anchor in self.anchor_vector.items()}

    def ratio_preserved(self, multiplier: float) -> bool:
        scaled = self.lr_for_groups(multiplier)
        base = list(self.anchor_vector.values())
        moved = list(scaled.values())
        if not base or base[0] == 0:
            return True
        return all(abs((moved[index] / moved[0]) - (base[index] / base[0])) < 1e-12
                   for index in range(len(base)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "interpretation": self.interpretation,
            "compliance_class": self.compliance_class,
            "coordinate_name": self.coordinate_name if self.searches_a_multiplier else None,
            "anchor_vector": dict(self.anchor_vector),
            "anchor_source": self.anchor_source,
            "multipliers": list(self.candidates),
            "preserved_ratio": list(self.preserved_ratio),
            "parameter_groups": list(self.parameter_groups),
            "lr_at_each_multiplier": {str(value): self.lr_for_groups(value)
                                      for value in self.candidates},
            "anchor_trial_reproduces_version_b":
                self.lr_for_groups(1.0) == {name: float(value) for name, value
                                            in self.anchor_vector.items()}
                if self.searches_a_multiplier else True,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class LRDecisionRecord:
    """The whole approved record, plus the identity of the bytes it came from."""

    components: dict[str, LRAnchorDecision]
    config_path: str
    config_sha256: str
    decision_status: str
    dossier: str
    dossier_identity: str
    raw: dict[str, Any]

    def for_component(self, name: str) -> LRAnchorDecision:
        try:
            return self.components[name]
        except KeyError:
            raise LRDecisionError(
                f"no approved learning-rate decision for {name!r}; the record covers "
                f"{sorted(self.components)}") from None

    @property
    def approved(self) -> bool:
        return self.decision_status == "APPROVED"

    def identity_material(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision_status": self.decision_status,
            "components": {name: item.as_dict()
                           for name, item in sorted(self.components.items())},
        }

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(self.identity_material(), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_material(),
            "decision_identity": self.identity,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "dossier": self.dossier,
            "dossier_identity": self.dossier_identity,
            "not_approved": list(self.raw.get("not_approved") or ()),
            "search_envelope_unchanged": dict(self.raw.get("search_envelope_unchanged") or {}),
        }


def load_decision(repo: Path) -> LRDecisionRecord:
    """Read the approved record, refusing anything that is not approved.

    An unapproved or absent record is not a reason to fall back to a default —
    falling back would silently re-open the decision the record exists to close.
    """
    import yaml

    path = Path(repo) / DECISION_CONFIG
    if not path.exists():
        raise LRDecisionError(
            f"the learning-rate decision record is missing at {DECISION_CONFIG.as_posix()}; "
            "C4 and C7 cannot build a search plan without it")
    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes.decode("utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise LRDecisionError(
            f"decision record schema {payload.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    status = str(payload.get("decision_status"))
    if status != "APPROVED":
        raise LRDecisionError(
            f"the learning-rate decision record is {status!r}, not APPROVED; a search "
            "plan built from an unapproved decision would re-open a closed question")

    multipliers = tuple(float(value) for value in payload["multipliers"])
    coordinate = str(payload["coordinate_name"])
    components: dict[str, LRAnchorDecision] = {}
    for key, component in (("c4_gpat", "C4"), ("c7_track_r", "C7_TRACK_R"),
                           ("c7_track_g", "C7_TRACK_G")):
        block = dict(payload[key])
        if not block.get("approved"):
            raise LRDecisionError(f"{key} is present but not marked approved")
        interpretation = str(block["interpretation"])
        components[component] = LRAnchorDecision(
            component=component,
            interpretation=interpretation,
            anchor_vector={name: float(value)
                           for name, value in dict(block["anchor_vector"]).items()},
            multipliers=multipliers if interpretation == COMMON_MULTIPLIER else (),
            coordinate_name=coordinate,
            preserved_ratio=tuple(float(value)
                                  for value in block.get("preserved_ratio") or ()),
            parameter_groups=tuple(block.get("parameter_groups") or ()),
            compliance_class=str(block["compliance_class"]),
            anchor_source=str(block.get("anchor_source", "")),
            rationale=str(block.get("rationale", "")).strip())

    return LRDecisionRecord(
        components=components,
        config_path=DECISION_CONFIG.as_posix(),
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        decision_status=status,
        dossier=str(payload.get("dossier", "")),
        dossier_identity=str(payload.get("dossier_identity", "")),
        raw=payload)


def lr_coordinate(decision: LRAnchorDecision) -> Any:
    """The single learning-rate coordinate this decision authorizes.

    Returns a `Coordinate` anchored at multiplier 1.0, so its candidate set is
    exactly {0.5, 1.0, 2.0} and the anchor trial reproduces Version B. Track G
    returns an inapplicable coordinate carrying its reason, which is how the
    engine already represents "resolved, nothing to search".
    """
    from prism_fas.search.plan import Coordinate

    if not decision.searches_a_multiplier:
        return Coordinate(
            name=decision.coordinate_name, anchor=None,
            multipliers=(), skip_reason=(
                f"{decision.compliance_class}: {decision.component} has exactly one "
                f"applicable inherited LR scalar "
                f"({', '.join(decision.anchor_vector)}), so there is no ambiguity to "
                "search and no multiplier to apply"),
            anchor_source=decision.anchor_source, spec_clause="§13.4.1 / §15.2.3")
    return Coordinate(
        name=decision.coordinate_name, anchor=1.0,
        multipliers=decision.multipliers, non_negative=True,
        anchor_source=decision.anchor_source,
        spec_clause="§15.2.2 learning rate, approved interpretation B")


__all__ = ["SCHEMA_VERSION", "DECISION_CONFIG", "COMMON_MULTIPLIER", "UNIQUE_ANCHOR",
           "LRDecisionError", "LRAnchorDecision", "LRDecisionRecord", "load_decision",
           "lr_coordinate"]
