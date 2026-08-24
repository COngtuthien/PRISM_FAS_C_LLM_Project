"""C7_SOURCE_SEARCH_SYNTHETIC_ARM — the frozen search population, as a record.

§15.2.2 fixes the detector/loss envelope completely: the coordinate order, the
candidate multipliers, one pass, the ranking tuple, the tie-break. It does not
say which of C6's three matched banks supplies the synthetic quarter of the batch
while the search runs, and that is not an oversight this module may repair by
picking one.

It matters because the arm IS the treatment. C8 compares RND, DET and LLM at one
frozen detector configuration per track; if that configuration were tuned on the
LLM bank, every later LLM-vs-control comparison would carry a tuning advantage no
statistic removes. So the arm is a decision record with a name, a reason and an
identity, read from `configs/search/c7_source_search_decision.yaml`, and this
module refuses anything that is not FROZEN. An absent or unfrozen record is not a
reason to fall back to a default; falling back is precisely how an unrecorded
scientific choice enters a result.

Two properties of the frozen decision are enforced here rather than trusted:

* **One search per TRACK, never per arm.** `tracks` is a list and `training_arm`
  is a scalar. There is no shape this record can take that authorizes three
  searches over three banks, which is the confound the whole design removes.
* **The prohibited alternatives are part of the identity.** Changing the arm
  after the freeze changes `decision_identity`, which is bound into the search
  plan, into every trial and into DETECTOR_CONFIG_LOCK — so a swapped arm
  invalidates resume and fails C8's parent-identity validation rather than
  quietly producing a second set of numbers.

The engineering readiness path never calls this. A CPU rehearsal proves the
search ALGORITHM and needs no training population.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "prism-c7-source-search-decision-v1"
DECISION_CONFIG = Path("configs") / "search" / "c7_source_search_decision.yaml"

#: The decision this record closes, named so it can be referred to by id.
DECISION_ID = "C7_SOURCE_SEARCH_SYNTHETIC_ARM"

#: The reason code a blocked scientific C7 reports.
NEEDS_SCIENTIFIC_DECISION = "NEEDS_SCIENTIFIC_DECISION"

#: The statuses that mean "decided". `APPROVED` is retained because the sibling
#: LR record uses it; `FROZEN` is what this record declares.
FROZEN_STATUSES: tuple[str, ...] = ("FROZEN", "APPROVED")

#: The fields that must be frozen before a scientific trial may run. Named
#: individually so a partial record blocks on exactly what it is missing.
REQUIRED_FIELDS: tuple[str, ...] = (
    "decision_id", "training_arm", "tracks", "protocol", "selection_tuple_name",
    "trial_schedule")

#: What each field may be. A record naming something outside these is rejected
#: rather than passed through to the trainer.
PERMITTED: dict[str, tuple[str, ...]] = {
    "decision_id": (DECISION_ID,),
    "training_arm": ("RND", "DET", "LLM"),
    "tracks": ("G", "R"),
    "protocol": ("P1", "P2", "P3"),
    "selection_tuple_name": ("P1P2", "P3_READY"),
    "trial_schedule": ("frozen_m9_schedule",),
}

#: Fields whose value is a list drawn from `PERMITTED`.
LIST_FIELDS: tuple[str, ...] = ("tracks",)


class C7DecisionError(ValueError):
    """The C7 search decision record is missing, unfrozen or inconsistent."""

    reason_code = NEEDS_SCIENTIFIC_DECISION


@dataclass(frozen=True)
class C7SearchDecision:
    """The frozen search population for the bounded C7 pass of each track."""

    decision_id: str
    training_arm: str
    tracks: tuple[str, ...]
    protocol: str
    selection_tuple_name: str
    trial_schedule: str
    decision_status: str
    source: str
    spec_status: str
    timing: str
    rationale: str
    approved_by: str
    frozen_on: str
    prohibited_alternatives: tuple[str, ...]
    config_path: str
    config_sha256: str
    raw: dict[str, Any]

    @property
    def approved(self) -> bool:
        return self.decision_status in FROZEN_STATUSES

    @property
    def frozen_before_any_trial(self) -> bool:
        return self.timing == "BEFORE_FIRST_C7_SCIENTIFIC_TRIAL"

    def permits_arm(self, arm: str) -> bool:
        """Whether this decision authorizes searching against `arm`'s bank.

        Exactly one arm is authorized. A caller asking about any other gets
        False, which is what makes "no RND or LLM candidate bytes may enter C7
        SOURCE_SEARCH training" a check rather than a comment.
        """
        return str(arm) == self.training_arm

    def identity_material(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision_id": self.decision_id,
            "decision_status": self.decision_status,
            "training_arm": self.training_arm,
            "tracks": list(self.tracks),
            "protocol": self.protocol,
            "selection_tuple_name": self.selection_tuple_name,
            "trial_schedule": self.trial_schedule,
            "prohibited_alternatives_after_freeze": list(self.prohibited_alternatives),
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
            "value": self.training_arm,
            "source": self.source,
            "spec_status": self.spec_status,
            "timing": self.timing,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "approved_by": self.approved_by,
            "frozen_on": self.frozen_on,
            "rationale": self.rationale,
            "not_approved": list(self.raw.get("not_approved") or ()),
            "per_arm_search_authorized": False,
            "pooled_search_bank_authorized": False,
            "target_access": 0,
        }


def load_decision(repo: Path) -> C7SearchDecision:
    """Read the frozen record, refusing anything that is not frozen."""
    import yaml

    path = Path(repo) / DECISION_CONFIG
    if not path.exists():
        raise C7DecisionError(
            f"the C7 source-search decision record is missing at "
            f"{DECISION_CONFIG.as_posix()}. §15.2.2 does not fix which of C6's three "
            "matched banks the bounded search trains against, and choosing one here "
            "would put an unrecorded treatment choice into every downstream number")
    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise C7DecisionError(
            f"decision record schema {payload.get('schema_version')!r} != {SCHEMA_VERSION!r}")

    status = str(payload.get("decision_status"))
    missing = [name for name in REQUIRED_FIELDS if not payload.get(name)]
    if missing:
        raise C7DecisionError(
            f"the C7 source-search decision record freezes none of {missing}; every "
            "one of them changes which numbers the search produces")

    illegal: list[str] = []
    for name in REQUIRED_FIELDS:
        values = (payload[name] if name in LIST_FIELDS else [payload[name]])
        for value in values:
            if str(value) not in PERMITTED[name]:
                illegal.append(f"{name}={value!r}")
    if illegal:
        raise C7DecisionError(
            f"the C7 source-search decision record declares {illegal}, which is "
            f"outside the permitted values {PERMITTED}")

    tracks = tuple(str(value) for value in payload["tracks"])
    if len(set(tracks)) != len(tracks):
        raise C7DecisionError(f"the record declares a track twice: {list(tracks)}")

    decision = C7SearchDecision(
        decision_id=str(payload["decision_id"]),
        training_arm=str(payload["training_arm"]),
        tracks=tracks,
        protocol=str(payload["protocol"]),
        selection_tuple_name=str(payload["selection_tuple_name"]),
        trial_schedule=str(payload["trial_schedule"]),
        decision_status=status,
        source=str(payload.get("source", "")),
        spec_status=str(payload.get("spec_status", "")),
        timing=str(payload.get("timing", "")),
        rationale=str(payload.get("rationale", "")).strip(),
        approved_by=str(payload.get("approved_by", "")),
        frozen_on=str(payload.get("frozen_on", "")),
        prohibited_alternatives=tuple(
            str(item) for item in payload.get("prohibited_alternatives_after_freeze") or ()),
        config_path=DECISION_CONFIG.as_posix(),
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw=dict(payload))

    if str(payload.get("value")) != decision.training_arm:
        raise C7DecisionError(
            f"the record declares value={payload.get('value')!r} and "
            f"training_arm={decision.training_arm!r}; they name the same thing and "
            "must agree")
    if status not in FROZEN_STATUSES:
        raise C7DecisionError(
            f"the C7 source-search decision record is {status!r}, not one of "
            f"{FROZEN_STATUSES}. A search plan built from an unfrozen decision "
            "would freeze a detector configuration on a training population "
            "nobody chose")
    if not decision.frozen_before_any_trial:
        raise C7DecisionError(
            f"the record declares timing={decision.timing!r}; a search-population "
            "decision taken after a scientific trial exists is a decision taken "
            "from a result")
    return decision


def decision_report(repo: Path) -> dict[str, Any]:
    """Non-raising form, for a precondition gate that must name what is unresolved."""
    try:
        decision = load_decision(repo)
    except C7DecisionError as error:
        return {"resolved": False, "reason_code": error.reason_code,
                "error": str(error), "config_path": DECISION_CONFIG.as_posix(),
                "decision_id": DECISION_ID,
                "required_fields": list(REQUIRED_FIELDS), "decision": None}
    return {"resolved": True, "reason_code": "", "error": "",
            "config_path": decision.config_path,
            "decision_id": decision.decision_id,
            "required_fields": list(REQUIRED_FIELDS),
            "decision": decision.as_dict()}


__all__ = ["SCHEMA_VERSION", "DECISION_CONFIG", "DECISION_ID",
           "NEEDS_SCIENTIFIC_DECISION", "FROZEN_STATUSES", "REQUIRED_FIELDS",
           "PERMITTED", "LIST_FIELDS", "C7DecisionError", "C7SearchDecision",
           "load_decision", "decision_report"]
