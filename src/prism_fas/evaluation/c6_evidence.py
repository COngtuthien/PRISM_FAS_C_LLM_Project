"""The strict C6 closure contract, read back from what C6 actually wrote.

C7 and C8 train against the three matched synthetic banks C6 froze. "C6 ran" is
not the precondition for that; "C6 closed, and these are the banks it closed
over" is. The difference is the whole point of this module: `reports/full/c6`
existing proves nothing, and a stage that accepted its existence would train
against a directory rather than against an experiment.

So every field the C6 closure asserts is re-read here from the artifacts C6
serialized, and disagreement is refused rather than discovered later:

* the profile-selection lock is a SCIENTIFIC lock and names the selected profile;
* the matched-bank artifact was built under that same profile, by
  `C6_MATCHED_BANK_SELECTOR_V1`, and records the selector identity;
* each of the three arm BANK_LOCKs binds that same selector identity and the
  same quality-threshold identity;
* each arm's final bank is exactly `FINAL_BANK_PER_ARM`, split exactly
  `PER_ROUTE` Physics + `PER_ROUTE` GPAT;
* each arm's provenance closure is `closed` with nothing unaccounted;
* `q_used_for_selection` is false — q is a §11.2 training weight and was never a
  selector input;
* `usable_for_c7_c8_source_training` is true and `target_access` is 0.

The constants are imported from the modules that own them
(`synthesis.c6_matched_bank`, `synthesis.c6_scientific`) rather than restated,
so this verifier cannot drift into being a weaker second opinion about what C6
requires. It reads JSON locks. It opens no candidate payload, no source package
and no target.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prism_fas.synthesis import c6_matched_bank as selector

SCHEMA_VERSION = "prism-c6-evidence-v1"

#: Where a scientific C6 writes. Not configurable: a lock trusted by C7/C8 lives
#: in the scientific namespace or it is not the lock they mean.
C6_REPORTS = "reports/full/c6"

PROFILE_SELECTION_LOCK = "C6_PROFILE_SELECTION_LOCK.json"
MATCHED_BANKS = "C6_MATCHED_BANKS.json"
BANK_LOCK_TEMPLATE = "C6_BANK_LOCK_{arm}.json"

#: The three arms, from the selector that produced them.
ARMS: tuple[str, ...] = selector.ARMS


class C6EvidenceError(RuntimeError):
    """The frozen C6 closure is absent, incomplete or self-inconsistent.

    One exception type with a machine-readable `reason_code`, because the
    operator's response differs by kind: an absent lock is fetched from the GPU
    host, a disagreeing one means something was rebuilt under C6 and the run
    must stop rather than train against a bank C6 did not freeze.
    """

    reason_code = "C6_EVIDENCE_INVALID"

    def __init__(self, message: str, *, reason_code: str | None = None,
                 problems: list[str] | None = None) -> None:
        super().__init__(message)
        if reason_code:
            self.reason_code = reason_code
        self.problems = list(problems or ())


class C6EvidenceMissing(C6EvidenceError):
    """A required C6 artifact is not on this machine."""

    reason_code = "C6_EVIDENCE_ABSENT"


@dataclass(frozen=True)
class ArmBankEvidence:
    """One arm's verified matched bank, as C7/C8 will address it."""

    arm: str
    lock_path: str
    selected_set_sha256: str
    selector_identity_sha256: str
    quality_profile: str
    quality_threshold_identity: str
    c5_pool_lock_sha256: str
    final_bank_size: int
    by_route: dict[str, int]
    candidate_ids: tuple[str, ...]
    selected: tuple[dict[str, Any], ...] = field(repr=False, default=())

    def as_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "lock": self.lock_path,
                "selected_set_sha256": self.selected_set_sha256,
                "selector_identity_sha256": self.selector_identity_sha256,
                "quality_profile": self.quality_profile,
                "quality_threshold_identity": self.quality_threshold_identity,
                "c5_pool_lock_sha256": self.c5_pool_lock_sha256,
                "final_bank_size": self.final_bank_size,
                "by_route": dict(self.by_route),
                "candidates": len(self.candidate_ids)}


@dataclass(frozen=True)
class C6Evidence:
    """The whole verified C6 closure, and the identities it binds."""

    reports_root: str
    selected_profile: str
    threshold_identity: str
    selector_identity_sha256: str
    selector_name: str
    c5_pool_lock_sha256: str
    banks: dict[str, ArmBankEvidence]

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(sorted(self.banks))

    def bank(self, arm: str) -> ArmBankEvidence:
        try:
            return self.banks[arm]
        except KeyError:
            raise C6EvidenceError(
                f"C6 froze no bank for arm {arm!r}; the verified arms are "
                f"{self.arms}") from None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "reports_root": self.reports_root,
            "selected_profile": self.selected_profile,
            "quality_threshold_identity": self.threshold_identity,
            "selector_identity_sha256": self.selector_identity_sha256,
            "selector_name": self.selector_name,
            "c5_pool_lock_sha256": self.c5_pool_lock_sha256,
            "final_bank_per_arm": selector.FINAL_BANK_PER_ARM,
            "per_route": selector.PER_ROUTE,
            "banks": {arm: item.as_dict() for arm, item in sorted(self.banks.items())},
            "target_access": 0,
            "verified_by": "prism_fas.evaluation.c6_evidence.verify_c6_evidence",
        }


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise C6EvidenceMissing(
            f"{path.as_posix()} is absent; it is part of the frozen C6 closure and "
            "there is no substitute for it on a scientific path")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise C6EvidenceError(
            f"{path.as_posix()} is not readable JSON: {error}") from error
    if not isinstance(payload, dict):
        raise C6EvidenceError(f"{path.as_posix()} is not a JSON object")
    return payload


def verify_c6_evidence(repo: Path, *, reports_root: str = C6_REPORTS) -> C6Evidence:
    """Verify the C6 closure, or refuse. Never returns a partial answer.

    Collects EVERY disagreement before raising, because an operator fixing a
    stale C6 tree needs the whole list rather than the first item of it.
    """
    root = Path(repo) / reports_root
    problems: list[str] = []

    profile_lock = _read(root / PROFILE_SELECTION_LOCK)
    matched = _read(root / MATCHED_BANKS)

    if profile_lock.get("is_scientific_lock") is not True:
        problems.append(
            f"{PROFILE_SELECTION_LOCK} does not declare is_scientific_lock=true "
            f"(got {profile_lock.get('is_scientific_lock')!r}); a rehearsal's "
            "profile selection may never govern a scientific bank")
    if profile_lock.get("fixture_backed") is not False:
        problems.append(
            f"{PROFILE_SELECTION_LOCK} does not declare fixture_backed=false "
            f"(got {profile_lock.get('fixture_backed')!r})")
    if profile_lock.get("ba_sep_not_used_for_profile_selection") is not True:
        problems.append(
            f"{PROFILE_SELECTION_LOCK} does not record that BA_sep was excluded "
            "from profile selection; the frozen sequence requires it")
    if int(profile_lock.get("target_access", -1)) != 0:
        problems.append(
            f"{PROFILE_SELECTION_LOCK} records target_access="
            f"{profile_lock.get('target_access')!r}, not 0")

    selected_profile = str(profile_lock.get("selected_profile") or "")
    threshold_identity = str(profile_lock.get("threshold_identity") or "")
    if not selected_profile:
        problems.append(f"{PROFILE_SELECTION_LOCK} names no selected_profile")
    if not threshold_identity:
        problems.append(f"{PROFILE_SELECTION_LOCK} carries no threshold_identity")

    if matched.get("selected_profile") != selected_profile:
        problems.append(
            f"{MATCHED_BANKS} was built under profile "
            f"{matched.get('selected_profile')!r} but {PROFILE_SELECTION_LOCK} "
            f"froze {selected_profile!r}; the banks and the lock describe "
            "different C6 runs")
    if matched.get("selector") != selector.SELECTOR_NAME:
        problems.append(
            f"{MATCHED_BANKS} names selector {matched.get('selector')!r}, not the "
            f"frozen {selector.SELECTOR_NAME!r}")
    if matched.get("fixture_backed") is not False:
        problems.append(f"{MATCHED_BANKS} does not declare fixture_backed=false")

    contract = dict(matched.get("selector_identity") or {})
    selector_identity = str(contract.get("selector_identity_sha256") or "")
    if not selector_identity:
        problems.append(f"{MATCHED_BANKS} carries no selector_identity_sha256")
    if contract.get("quality_profile_identity") not in (None, threshold_identity):
        problems.append(
            "the selector identity was computed over quality profile identity "
            f"{contract.get('quality_profile_identity')!r} but the profile lock "
            f"froze {threshold_identity!r}")

    pool_lock = str(contract.get("c5_pool_lock_sha256") or "")

    banks: dict[str, ArmBankEvidence] = {}
    for arm in ARMS:
        name = BANK_LOCK_TEMPLATE.format(arm=arm)
        payload = _read(root / name)
        problems.extend(_arm_problems(name, arm, payload,
                                      selector_identity=selector_identity,
                                      threshold_identity=threshold_identity,
                                      selected_profile=selected_profile,
                                      pool_lock=pool_lock))
        rows = tuple(dict(row) for row in (payload.get("selected") or ()))
        banks[arm] = ArmBankEvidence(
            arm=arm, lock_path=f"{reports_root}/{name}",
            selected_set_sha256=str(payload.get("selected_set_sha256") or ""),
            selector_identity_sha256=str(payload.get("selector_identity_sha256") or ""),
            quality_profile=str(payload.get("quality_profile") or ""),
            quality_threshold_identity=str(
                payload.get("quality_threshold_identity") or ""),
            c5_pool_lock_sha256=str(payload.get("c5_pool_lock_sha256") or ""),
            final_bank_size=int(payload.get("final_bank_size") or 0),
            by_route={str(key): int(value) for key, value
                      in dict(payload.get("by_route") or {}).items()},
            candidate_ids=tuple(str(row.get("candidate_id")) for row in rows),
            selected=rows)

    if problems:
        raise C6EvidenceError(
            "the frozen C6 closure does not verify: " + "; ".join(problems[:12])
            + (f" (and {len(problems) - 12} more)" if len(problems) > 12 else ""),
            problems=problems)

    return C6Evidence(
        reports_root=reports_root, selected_profile=selected_profile,
        threshold_identity=threshold_identity,
        selector_identity_sha256=selector_identity,
        selector_name=str(matched.get("selector")),
        c5_pool_lock_sha256=pool_lock, banks=banks)


def _arm_problems(name: str, arm: str, payload: dict[str, Any], *,
                  selector_identity: str, threshold_identity: str,
                  selected_profile: str, pool_lock: str) -> list[str]:
    """Every way one arm's BANK_LOCK can disagree with the closure it belongs to."""
    problems: list[str] = []

    if payload.get("arm") != arm:
        problems.append(f"{name} declares arm {payload.get('arm')!r}, not {arm!r}")
    if payload.get("is_scientific_lock") is not True:
        problems.append(f"{name} does not declare is_scientific_lock=true")
    if payload.get("fixture_backed") is not False:
        problems.append(f"{name} does not declare fixture_backed=false")

    size = int(payload.get("final_bank_size") or 0)
    if size != selector.FINAL_BANK_PER_ARM:
        problems.append(
            f"{name} holds {size} samples; §11.3 fixes the final bank at "
            f"{selector.FINAL_BANK_PER_ARM} per arm")
    routes = {str(key): int(value) for key, value
              in dict(payload.get("by_route") or {}).items()}
    for route in selector.ROUTES:
        if routes.get(route) != selector.PER_ROUTE:
            problems.append(
                f"{name} route {route} holds {routes.get(route)!r}, not "
                f"{selector.PER_ROUTE}")

    rows = list(payload.get("selected") or ())
    if len(rows) != size:
        problems.append(
            f"{name} declares {size} samples but serializes {len(rows)} selected "
            "rows; the lock does not describe its own bank")
    identifiers = [str(row.get("candidate_id") or "") for row in rows]
    if len(set(identifiers)) != len(identifiers):
        problems.append(f"{name} selects the same candidate more than once")
    if any(not value for value in identifiers):
        problems.append(f"{name} has a selected row with no candidate_id")

    if payload.get("selector_identity_sha256") != selector_identity:
        problems.append(
            f"{name} binds selector identity "
            f"{payload.get('selector_identity_sha256')!r} but the matched-bank "
            f"artifact recorded {selector_identity!r}")
    if payload.get("selector_name") != selector.SELECTOR_NAME:
        problems.append(
            f"{name} names selector {payload.get('selector_name')!r}, not "
            f"{selector.SELECTOR_NAME!r}")
    if payload.get("quality_profile") != selected_profile:
        problems.append(
            f"{name} was gated under profile {payload.get('quality_profile')!r} "
            f"but C6 froze {selected_profile!r}")
    if payload.get("quality_threshold_identity") != threshold_identity:
        problems.append(
            f"{name} binds threshold identity "
            f"{payload.get('quality_threshold_identity')!r} but the profile lock "
            f"froze {threshold_identity!r}")
    if pool_lock and payload.get("c5_pool_lock_sha256") != pool_lock:
        problems.append(
            f"{name} was built over C5 pool {payload.get('c5_pool_lock_sha256')!r} "
            f"but the selector identity was computed over {pool_lock!r}")

    closure = dict(payload.get("provenance_closure") or {})
    if closure.get("closed") is not True:
        problems.append(
            f"{name} provenance_closure.closed is {closure.get('closed')!r}; an "
            "unclosed bank has candidates whose fate is unaccounted for")
    unaccounted = list(closure.get("unaccounted") or ())
    if unaccounted:
        problems.append(
            f"{name} leaves {len(unaccounted)} candidate(s) unaccounted, starting "
            f"at {unaccounted[:3]}")

    # q is a §11.2 training weight. A bank that ranked by it would have selected
    # on quality, which is a different experiment from the one C6 preregistered.
    if payload.get("q_used_for_selection") is not False:
        problems.append(
            f"{name} does not record q_used_for_selection=false; q is a training "
            "weight and may never be a selector input")
    if payload.get("usable_for_c7_c8_source_training") is not True:
        problems.append(
            f"{name} does not declare usable_for_c7_c8_source_training=true")
    if int(payload.get("target_access", -1)) != 0:
        problems.append(
            f"{name} records target_access={payload.get('target_access')!r}, not 0")
    proof = dict(payload.get("no_target_capability_proof") or {})
    if int(proof.get("target_labels_resolved", -1)) != 0 or proof.get(
            "target_roots_mounted") not in ([], ()):
        problems.append(f"{name} carries no clean no-target-capability proof")

    return problems


def evidence_report(repo: Path, *, reports_root: str = C6_REPORTS) -> dict[str, Any]:
    """Non-raising form, for a precondition gate that must name what is wrong.

    Shaped like the rows `RequiredInput.resolve` produces, so an unsatisfied C6
    closure BLOCKS a stage exactly as a missing file does and lands in the same
    gate report.
    """
    try:
        evidence = verify_c6_evidence(repo, reports_root=reports_root)
    except C6EvidenceError as error:
        return {"valid": False, "reason_code": error.reason_code,
                "error": str(error), "problems": list(error.problems),
                "reports_root": reports_root, "evidence": None}
    return {"valid": True, "reason_code": "", "error": "", "problems": [],
            "reports_root": reports_root, "evidence": evidence.as_dict()}


__all__ = ["SCHEMA_VERSION", "C6_REPORTS", "PROFILE_SELECTION_LOCK", "MATCHED_BANKS",
           "BANK_LOCK_TEMPLATE", "ARMS", "C6EvidenceError", "C6EvidenceMissing",
           "ArmBankEvidence", "C6Evidence", "verify_c6_evidence", "evidence_report"]
