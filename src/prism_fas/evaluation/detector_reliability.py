"""DETECTOR_RELIABILITY_LOCK_C — the pre-target reliability barrier.

The v1.5 text contains a staging incompatibility, and this module is where the
typed decision that resolves it lives.

    §3.1.1 evaluates BA_sep AFTER the common C6 synthetic gate is frozen.
    §17 places the reliability gates BEFORE P3 target evaluation.
    The C6 stage row reads "shortcut gates pass or STOP".
    The only canonical synthetic-vs-real probe uses DETECTOR evidence
    (p_global, s_region, nine normalized regional distances).
    C6 has no detector. C7 implements one and C8 trains it.

So "C6 shortcut gates pass or STOP" is not executable as written: at C6 there is
nothing to probe with. Rather than invent an image-level bank probe — a new
feature extractor, classifier, split, training budget and seed policy that v1.5
never froze — the synthetic-vs-real gate moves to the stage where its evidence
exists:

    SYNTHETIC_VS_REAL_RELIABILITY_STAGE
        = C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C

This is a superseding decision, recorded as one. It does not rewrite the earlier
reading; it records why that reading cannot execute and what replaces it. It was
frozen before any target access and before any valid C6 profile or bank result
was used to choose it.

What moving the probe does NOT solve is the executable protocol. That remains
`NEEDS_SCIENTIFIC_DECISION`, and this module refuses rather than guessing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "prism-detector-reliability-lock-c-v1"
LOCK_NAME = "DETECTOR_RELIABILITY_LOCK_C.json"

#: Where the lock lives, and the deadline it enforces.
LOCK_PATH = "reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json"
STAGE = "C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C"

#: Resolution states. UNRESOLVED and NOT_APPLICABLE_AT_C6 are never passes.
PASSED, FAILED, BLOCKED, UNRESOLVED = "PASSED", "FAILED", "BLOCKED", "UNRESOLVED"
NOT_APPLICABLE_AT_C6 = "NOT_APPLICABLE_AT_C6"

#: Every test that must resolve after C8 and before C9 closes SOURCE_MATRIX_LOCK_C.
REQUIRED_DETECTOR_RELIABILITY_TESTS: tuple[str, ...] = (
    "synthetic_vs_real_spoof_probe",
    "residual_scale_zero",
    "recipe_region_shift",
    "artifact_map_swap",
    "cross_route_synthetic",
    "benign_jpeg_corruption",
    "benign_resize_corruption",
    "benign_color_corruption",
    "crop_padding_interpolation",
)

#: Declared with no legitimate population. It stays BLOCKED and is never a pass;
#: it is also not in the required set, so it cannot hold the barrier shut.
CANONICALLY_BLOCKED_TESTS: tuple[str, ...] = ("benign_glasses_makeup_lowlight",)

#: The frozen scientific meaning of BA_sep, carried over unchanged from §3.1.1.
BA_SEP_CEILING = 0.75
BA_SEP_SEEDS_REQUIRED = 3
BA_SEP_DEFINITION = (
    "BA_sep_arm = mean synthetic-vs-real probe balanced accuracy over the three "
    "frozen source-only probe seeds on the same matched source split; lower is "
    "better; the hard reliability gate is BA_sep <= 0.75")

#: The hard gate is NOT the hypothesis. Kept apart so passing one is never read
#: as supporting the other.
C_H4_SUPPORT_RULE = (
    "C-H4 is SUPPORTED only if BA_sep_LLM <= 0.75 AND BA_sep_LLM < BA_sep_DET "
    "AND BA_sep_LLM < BA_sep_RND with the 95% paired source-sample bootstrap CI "
    "upper bound < 0 for each control difference, AND the validity condition, "
    "AND the recipe-diversity condition. Passing the hard gate implies none of "
    "this")

PROBE_PROTOCOL_UNRESOLVED = "DETECTOR_BA_SEP_PROBE_PROTOCOL_NEEDS_SCIENTIFIC_DECISION"
EVIDENCE_VECTOR_UNRESOLVED = "DETECTOR_BA_SEP_EVIDENCE_VECTOR_NEEDS_SCIENTIFIC_DECISION"
PROBE_SEEDS_UNRESOLVED = "DETECTOR_BA_SEP_PROBE_SEEDS_NEEDS_SCIENTIFIC_DECISION"

#: The executable protocol. `None` until every result-affecting field below is
#: frozen. No BA number may be produced while this is None.
#:
#: Deliberately never assigned a hard-coded dict in this module — a Python
#: literal is not how any other frozen Version-C scientific decision in this
#: project is bound (compare `search.c7_decision.load_decision`,
#: `search.lr_decision.load_decision`: a config file, resolved by a function
#: that takes `repo`). `probe_protocol_status(repo=...)` below reads the
#: frozen protocol from its own config file via `load_probe_protocol`; this
#: constant stays the honest "no repo context" answer.
DETECTOR_BA_SEP_PROBE_PROTOCOL: dict[str, Any] | None = None

#: Where a frozen, user-approved probe protocol lives, once one exists.
#: Currently: Option 1 V2 (common Track-G decision evidence, group-safe on
#: source_record_id), approved before any BA_sep value was observed. V2
#: supersedes V1 with a pre-execution group-identity correction only; no
#: BA_sep value was ever observed under either version. See
#: reports/readiness/C9_BA_SEP_OPTION1_PROTOCOL_FREEZE.md (V1, historical)
#: and reports/readiness/C9_BA_SEP_OPTION1_V2_PREEXECUTION_CORRECTION.md
#: (V2, current).
PROBE_PROTOCOL_CONFIG_PATH = "configs/evaluation/c9_detector_ba_sep_option1_v2.yaml"

#: Everything the protocol must bind before the first probe execution. None of
#: it may be chosen after observing a BA value.
PROBE_PROTOCOL_REQUIRED_FIELDS: tuple[str, ...] = (
    "real_spoof_population", "synthetic_population", "source_domains",
    "matched_source_split", "class_balancing_rule", "sample_unit",
    "detector_checkpoint_identity", "evidence_vector_definition",
    "preprocessing", "feature_normalization", "linear_probe_implementation",
    "regularization", "optimizer_or_solver", "training_budget",
    "train_validation_split", "probe_seed_values",
    "balanced_accuracy_implementation", "per_seed_aggregation",
    "mean_aggregation", "ba_ceiling",
)

#: Why the evidence vector is not recoverable, recorded so the gap is auditable.
EVIDENCE_VECTOR_AUDIT: tuple[str, ...] = (
    "Version-B recorded the probe as 'a linear probe on the detector's own "
    "evidence vector (p_global, s_region, nine normalized regional distances)'. "
    "s_region and the nine regional distances are REGIONAL quantities, produced "
    "by a Track-R detector",
    "Version-C Track-R primary rows are DET and LLM only: C-H3 is 'LLM "
    "structured vs deterministic structured, same Track-R detector'. There is "
    "no preregistered Track-R RND row",
    "but BA_sep is required for all three arms — C-H4 needs BA_sep_LLM < "
    "BA_sep_RND — so a common evidence representation across RND, DET and LLM "
    "is needed and none is uniquely recoverable",
    "each way out is a new scientific choice: adding a Track-R RND experiment, "
    "substituting a Track-G vector for RND, using different feature spaces per "
    "arm, or dropping RND from BA_sep. None may be taken silently",
)

#: Why the seeds are not recoverable.
PROBE_SEED_AUDIT: tuple[str, ...] = (
    "§3.1.1 says 'the three frozen source-only probe seeds' and never names them",
    "§18.3 fixes the seed family 20260806-20260810 for 5-seed rows and the first "
    "three for 3-seed rows — but that policy is scoped to hypothesis TRAINING "
    "rows, and the probe is not a training row",
    "whether the probe inherits that family is therefore not normative on the "
    "current audit, and seeds may never be chosen after seeing a BA value",
)


class DetectorReliabilityError(RuntimeError):
    """The pre-target reliability barrier cannot be resolved as declared."""


def load_probe_protocol(repo: Path) -> dict[str, Any] | None:
    """The frozen probe protocol, read from `PROBE_PROTOCOL_CONFIG_PATH`, or
    `None`.

    Never invents a value: a missing file, a file that fails to parse, a file
    not declaring `status: FROZEN_NOT_RUN`, or a file missing any
    `PROBE_PROTOCOL_REQUIRED_FIELDS` entry all return `None` — the same
    "not yet resolved" answer this module always gave while
    `DETECTOR_BA_SEP_PROBE_PROTOCOL` was a hard-coded `None`. Loading a real
    protocol never marks any TEST result `PASSED`; that is a separate,
    later, executed measurement.
    """
    import yaml

    path = Path(repo) / PROBE_PROTOCOL_CONFIG_PATH
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "FROZEN_NOT_RUN":
        return None
    missing = [field for field in PROBE_PROTOCOL_REQUIRED_FIELDS if field not in payload]
    if missing:
        return None
    return payload


def probe_protocol_status(repo: Path | None = None) -> dict[str, Any]:
    """Whether a BA_sep number may be produced at all.

    With no `repo`, this is the module-constant answer
    (`DETECTOR_BA_SEP_PROBE_PROTOCOL`, always `None`). With a `repo`, the
    frozen protocol is read from its own config file (`load_probe_protocol`)
    — resolved becomes `True` only because every required protocol field is
    now explicitly frozen there, never because a Python literal changed.
    """
    protocol = load_probe_protocol(repo) if repo is not None else DETECTOR_BA_SEP_PROBE_PROTOCOL
    resolved = protocol is not None
    missing = ([] if resolved
               else [field for field in PROBE_PROTOCOL_REQUIRED_FIELDS])
    return {
        "resolved": resolved,
        "protocol": protocol,
        "protocol_identity": protocol_identity(protocol) if protocol else None,
        "reason_code": None if resolved else PROBE_PROTOCOL_UNRESOLVED,
        "unresolved_fields": missing,
        "open_decisions": [] if resolved else [
            PROBE_PROTOCOL_UNRESOLVED, EVIDENCE_VECTOR_UNRESOLVED,
            PROBE_SEEDS_UNRESOLVED],
        "evidence_vector_audit": list(EVIDENCE_VECTOR_AUDIT),
        "probe_seed_audit": list(PROBE_SEED_AUDIT),
        "ba_sep_definition": BA_SEP_DEFINITION,
        "ba_ceiling": BA_SEP_CEILING,
        "seeds_required": BA_SEP_SEEDS_REQUIRED,
        "may_execute": resolved,
        "rule": ("no BA_sep number may be produced until every result-affecting "
                 "protocol field is frozen; nothing may be chosen after "
                 "observing a BA value"),
    }


#: Metadata keys excluded from the protocol identity: they describe WHEN and
#: BY WHOM the protocol was frozen, never WHAT was frozen. Including them
#: would make the identity move on a re-save that changed nothing scientific.
_PROTOCOL_IDENTITY_EXCLUDED_KEYS = frozenset({
    "frozen_on", "approved_by", "status", "no_ba_sep_observed_before_freeze",
    "not_resolved_by_this_freeze", "schema_version", "decision_id", "resolves_test",
})


def protocol_identity(protocol: Mapping[str, Any]) -> str:
    """sha256 over every result-affecting protocol field, sorted keys, no
    timestamps/hostnames/results. Changes if and only if a result-affecting
    field changes."""
    material = {key: value for key, value in protocol.items()
               if key not in _PROTOCOL_IDENTITY_EXCLUDED_KEYS}
    return hashlib.sha256(json.dumps(
        material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def barrier_state(results: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The barrier's resolution state, computed from explicit per-test verdicts.

    An absent verdict is UNRESOLVED, never a pass. A canonically BLOCKED test is
    reported as BLOCKED and is not in the required set, so it neither passes nor
    holds the barrier shut for want of a population that does not exist.
    """
    results = dict(results or {})
    unexpected = sorted(set(results) - set(REQUIRED_DETECTOR_RELIABILITY_TESTS)
                        - set(CANONICALLY_BLOCKED_TESTS))
    if unexpected:
        raise DetectorReliabilityError(
            f"verdicts supplied for undeclared tests: {unexpected}")

    per_test = {name: results.get(name, UNRESOLVED)
                for name in REQUIRED_DETECTOR_RELIABILITY_TESTS}
    for name in CANONICALLY_BLOCKED_TESTS:
        per_test[name] = BLOCKED

    required = {name: per_test[name] for name in REQUIRED_DETECTOR_RELIABILITY_TESTS}
    failed = sorted(name for name, state in required.items() if state == FAILED)
    unresolved = sorted(name for name, state in required.items()
                        if state in (UNRESOLVED, NOT_APPLICABLE_AT_C6))
    blocked = sorted(name for name, state in required.items() if state == BLOCKED)

    if failed:
        overall = FAILED
    elif unresolved or blocked:
        overall = UNRESOLVED if unresolved else BLOCKED
    else:
        overall = PASSED

    return {
        "schema_version": SCHEMA_VERSION, "stage": STAGE,
        "per_test": per_test, "required": list(REQUIRED_DETECTOR_RELIABILITY_TESTS),
        "canonically_blocked": list(CANONICALLY_BLOCKED_TESTS),
        "failed": failed, "unresolved": unresolved, "blocked": blocked,
        "overall": overall,
        "unresolved_is_not_a_pass": True,
        "c9_may_close": overall == PASSED,
        "on_failure": (
            "DETECTOR_RELIABILITY_LOCK_C = FAILED blocks C9 SOURCE_MATRIX_LOCK_C, "
            "C10, C11 and target prediction. All negative evidence is preserved. "
            "C6 is never reopened, no other C6 profile is chosen, C5 is never "
            "regenerated, banks are not tuned, checkpoints are not cherry-picked, "
            "probe seeds are not rechosen and the 0.75 ceiling is not loosened. "
            "Any redesign afterwards requires a new approved protocol version"),
        "target_access": 0,
    }


def verify_lock(repo: Path, lock_path: Path | None = None) -> dict[str, Any]:
    """Validate a DETECTOR_RELIABILITY_LOCK_C for C9's precondition.

    Structural, not a PROJECT_STATE line: the lock must exist, declare itself,
    resolve every required test, bind the probe protocol identity and the
    detector checkpoint identities, and record zero target access.
    """
    path = Path(lock_path or (Path(repo) / LOCK_PATH))
    problems: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        payload, problems = {}, ["the lock could not be read as JSON"]

    if not payload:
        problems.append(f"{LOCK_NAME} is absent or empty")
    else:
        if payload.get("schema_version") != SCHEMA_VERSION:
            problems.append("schema_version does not match the frozen contract")
        if payload.get("stage") != STAGE:
            problems.append(f"stage is not {STAGE}")
        if payload.get("overall") != PASSED:
            problems.append(f"overall is {payload.get('overall')!r}, not {PASSED}")
        per_test = dict(payload.get("per_test") or {})
        for name in REQUIRED_DETECTOR_RELIABILITY_TESTS:
            state = per_test.get(name, UNRESOLVED)
            if state != PASSED:
                problems.append(f"{name} is {state}")
        if not payload.get("probe_protocol_identity"):
            problems.append("no probe protocol identity is bound")
        if not payload.get("detector_checkpoint_identities"):
            problems.append("no detector checkpoint identity is bound")
        if int(payload.get("target_access", -1)) != 0:
            problems.append("target_access is not recorded as 0")

    return {"valid": not problems, "problems": problems, "payload": payload,
            "lock_path": path.as_posix(), "required_stage": STAGE,
            "rule": ("C9 SOURCE_MATRIX_LOCK_C may close only over a valid "
                     f"{LOCK_NAME}; an unresolved required test never counts as "
                     "a pass")}


def lock_payload(*, results: Mapping[str, str], probe_protocol_identity: str,
                 detector_checkpoint_identities: Mapping[str, str],
                 ba_sep_by_arm: Mapping[str, float] | None = None) -> dict[str, Any]:
    """The barrier lock, built from explicit verdicts and bound identities."""
    state = barrier_state(results)
    return {**state,
            "probe_protocol_identity": probe_protocol_identity,
            "detector_checkpoint_identities": dict(detector_checkpoint_identities),
            "ba_sep_by_arm": dict(ba_sep_by_arm or {}),
            "ba_sep_definition": BA_SEP_DEFINITION,
            "ba_ceiling": BA_SEP_CEILING,
            "c_h4_support_rule_is_separate": C_H4_SUPPORT_RULE,
            "identity_sha256": _identity(state, probe_protocol_identity,
                                         detector_checkpoint_identities)}


def _identity(state: Mapping[str, Any], protocol: str,
              checkpoints: Mapping[str, str]) -> str:
    material = json.dumps({"per_test": state["per_test"], "protocol": protocol,
                           "checkpoints": dict(checkpoints)},
                          sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


__all__ = ["SCHEMA_VERSION", "LOCK_NAME", "LOCK_PATH", "STAGE", "PASSED", "FAILED",
           "BLOCKED", "UNRESOLVED", "NOT_APPLICABLE_AT_C6",
           "REQUIRED_DETECTOR_RELIABILITY_TESTS", "CANONICALLY_BLOCKED_TESTS",
           "BA_SEP_CEILING", "BA_SEP_SEEDS_REQUIRED", "BA_SEP_DEFINITION",
           "C_H4_SUPPORT_RULE", "DETECTOR_BA_SEP_PROBE_PROTOCOL",
           "PROBE_PROTOCOL_CONFIG_PATH",
           "PROBE_PROTOCOL_REQUIRED_FIELDS", "PROBE_PROTOCOL_UNRESOLVED",
           "EVIDENCE_VECTOR_UNRESOLVED", "PROBE_SEEDS_UNRESOLVED",
           "EVIDENCE_VECTOR_AUDIT", "PROBE_SEED_AUDIT",
           "DetectorReliabilityError", "load_probe_protocol", "probe_protocol_status",
           "barrier_state", "verify_lock", "lock_payload"]
