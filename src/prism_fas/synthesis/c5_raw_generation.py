"""C5 raw candidate generation — Version-C, and deliberately gate-free.

The Version-B `SyntheticBankGenerator` does generation AND quality evaluation in
one pass: it loads a `FrozenCalibration` in `__post_init__` and binds three
calibration hashes into its generation identity. That is correct for the
Version-B milestone it was written for, and it is untouched — this module imports
its route primitives and nothing else.

Version-C splits the two across the frozen stage boundary. C5 renders; C6 gates.
So this path stops the moment a candidate is finalized and hashed. It never
constructs a `CandidateEvaluator`, never loads a calibration, never calls
`quality_gate.evaluate`, and writes no acceptance decision. A candidate that C6
will later reject is still a completed C5 candidate, and its bytes and its
identity do not depend on the threshold that will reject it.

Three properties the record layer exists to guarantee:

**Exactly one terminal outcome per planned candidate.** A planned position is
either a success record with verified payload hashes, or a failure record with
its reason. Nothing is resampled to fill a gap; the budget is frozen at 2048 per
arm and a failure leaves 2047 usable, which is C5's problem to report rather than
to paper over.

**Reuse only on full agreement.** A record is reused when its generation identity
matches AND every payload still hashes to what it recorded. A missing or altered
payload rebuilds that exact candidate — the same identity, the same inputs — and
nothing else.

**The record is written last.** Its presence therefore means the payloads beside
it are complete, which is what lets a later process trust them without redoing
the work.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "prism-c5-raw-candidate-v1"
RECORD_NAME = "CANDIDATE.json"

#: Terminal states. A planned candidate ends in exactly one of them, and the
#: presence of RECORD_NAME is what makes an outcome terminal.
GENERATED = "generated"
FAILED_GENERATION = "failed_generation"

#: NOT a terminal state, and deliberately not written to RECORD_NAME.
#:
#: C5_RUNTIME_RECOVERY_V1 separates what a candidate IS from what happened while
#: trying to make it. A CUDA fault, an OOM, a filesystem error or an unexpected
#: RuntimeError says nothing about the candidate. Recording one as
#: FAILED_GENERATION would spend one of the frozen 2048 on a machine problem and
#: make the loss permanent, since a terminal failure is never retried. So it is
#: written beside the candidate as operational provenance, the pass aborts, and
#: the next invocation retries this exact identity — recovery-ladder L1, "retry
#: the identical frozen configuration for transient/runtime failure" (§0.1).
RUNTIME_ATTEMPT_FAILURE = "runtime_incomplete"
RUNTIME_ATTEMPT_DIR = "runtime_attempts"
RUNTIME_ATTEMPT_SCHEMA = "prism-c5-runtime-attempt-v1"

#: Payload file names inside a candidate directory. Frozen so a record written by
#: one process is addressable by the next.
IMAGE_NAME = "synthetic.png"
MASK_NAME = "exact_mask.png"
ARTIFACT_MAP_NAME = "artifact_map.npz"
PAYLOAD_NAMES = (IMAGE_NAME, MASK_NAME, ARTIFACT_MAP_NAME)


class RawGenerationError(RuntimeError):
    """C5 raw generation could not proceed under the frozen contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_reason(error: BaseException) -> str:
    """An exception can carry a host path; a scientific record may not."""
    import re

    return re.sub(r"([A-Za-z]:\\|/home/|/Users/)\S*", "[redacted-path]",
                  str(error))[:400]


def sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def candidate_dir(work_root: Path, arm: str, candidate_id: str) -> Path:
    """Deterministic, addressable by identity alone.

    Keyed on the candidate id rather than on a counter, so a resumed process
    finds the same directory without carrying any ordering in memory.
    """
    return Path(work_root) / arm / candidate_id


@dataclass(frozen=True)
class GenerationIdentity:
    """Everything that can change this candidate's bytes, and nothing else.

    Notably absent, by the frozen C5/C6 boundary: `threshold_sha256`,
    `fingerprint_reference_sha256`, `calibration_sha256`, the selected quality
    profile and any acceptance decision. Those describe how a candidate is
    JUDGED, and C6 chooses them after these candidates exist.
    """

    candidate_id: str
    arm: str
    arm_plan_identity: str
    source_pair_plan_identity: str
    package_identity: str
    recipe_bank_identity: str
    recipe_id: str
    recipe_ordinal: int
    slot: int
    position: int
    route: str
    live_target_sample_id: str
    spoof_source_sample_id: str | None
    generator_binding: str
    ontology_identity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self.candidate_id, "arm": self.arm,
            "arm_plan_identity": self.arm_plan_identity,
            "source_pair_plan_identity": self.source_pair_plan_identity,
            "package_identity": self.package_identity,
            "recipe_bank_identity": self.recipe_bank_identity,
            "recipe_id": self.recipe_id, "recipe_ordinal": int(self.recipe_ordinal),
            "slot": int(self.slot), "position": int(self.position),
            "route": self.route,
            "live_target_sample_id": self.live_target_sample_id,
            "spoof_source_sample_id": self.spoof_source_sample_id,
            "generator_binding": self.generator_binding,
            "ontology_identity": self.ontology_identity,
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass
class CandidateRecord:
    """One planned candidate's terminal outcome, as written to disk."""

    identity: GenerationIdentity
    status: str
    payload_sha256: dict[str, str] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "status": self.status,
                "generation_identity": self.identity.as_dict(),
                "generation_identity_sha256": self.identity.digest(),
                "payload_sha256": dict(self.payload_sha256),
                "payloads": list(PAYLOAD_NAMES) if self.status == GENERATED else [],
                "trace": dict(self.trace), "failure": self.failure,
                "binds_quality_calibration": False,
                "scientific_eligible": True}


def read_record(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_record(directory: Path, record: CandidateRecord) -> Path:
    """Written LAST, atomically. Its presence means the payloads are complete."""
    from prism_fas.pipeline.state import atomic_write_json

    path = Path(directory) / RECORD_NAME
    atomic_write_json(path, record.as_dict())
    return path


def reuse_decision(directory: Path, identity: GenerationIdentity) -> dict[str, Any]:
    """Whether a previous process's candidate may stand, and why not if it may not.

    Four answers, each meaning something different to the caller:

        ABSENT          nothing was recorded here; render it
        STALE           recorded under a different generation identity, so it is
                        not this candidate at all; it is never called "reused"
        PAYLOAD_MISSING recorded as generated, but a payload is gone
        PAYLOAD_CHANGED a payload no longer hashes to what the record says
        REUSABLE        identity agrees and every hash still matches
    """
    directory = Path(directory)
    payload = read_record(directory / RECORD_NAME)
    if payload is None:
        return {"reusable": False, "reason": "ABSENT", "candidate_id": identity.candidate_id}
    recorded = payload.get("generation_identity_sha256")
    if recorded != identity.digest():
        return {"reusable": False, "reason": "STALE",
                "candidate_id": identity.candidate_id,
                "recorded_identity_sha256": recorded,
                "expected_identity_sha256": identity.digest()}
    if payload.get("status") == FAILED_GENERATION:
        # A retained failure. It is terminal and addressable, but it is not a
        # usable payload, and it is never replaced by a different candidate.
        return {"reusable": False, "reason": "FAILED_GENERATION",
                "candidate_id": identity.candidate_id,
                "failure": payload.get("failure")}
    for name in PAYLOAD_NAMES:
        path = directory / name
        if not path.is_file():
            return {"reusable": False, "reason": "PAYLOAD_MISSING",
                    "candidate_id": identity.candidate_id, "payload": name}
        if sha256_file(path) != payload.get("payload_sha256", {}).get(name):
            return {"reusable": False, "reason": "PAYLOAD_CHANGED",
                    "candidate_id": identity.candidate_id, "payload": name}
    return {"reusable": True, "reason": "REUSABLE",
            "candidate_id": identity.candidate_id,
            "payload_sha256": payload.get("payload_sha256", {})}


def write_payload_bytes(directory: Path, result: Any) -> dict[str, str]:
    """The three raw payloads, as the canonical finalizer already encoded them.

    `finalize_discrete` has produced `image_png`, `mask_png` and
    `artifact_map_npz` and has already decoded each one back to check it
    round-trips. Re-encoding here with a second imaging library would produce
    different bytes for the same candidate and would make the recorded hashes
    describe an encoder rather than a result, so these bytes are written
    verbatim. No gate is consulted.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, payload in ((IMAGE_NAME, result.image_png),
                          (MASK_NAME, result.mask_png),
                          (ARTIFACT_MAP_NAME, result.artifact_map_npz)):
        temporary = directory / f".{name}.partial"
        temporary.write_bytes(payload)
        temporary.replace(directory / name)
        written[name] = _sha256_bytes(payload)
    return written


def failure_record(identity: GenerationIdentity, *, stage: str,
                   error: BaseException) -> CandidateRecord:
    """A retained, TERMINAL failure OF THE CANDIDATE ITSELF.

    Reserved for a deterministic candidate-semantic failure: one that is a pure
    function of the frozen inputs and would recur identically on every rerun, so
    that retrying it could only produce the same answer. The budget does not grow
    to compensate, and the arm is short by one.

    A runtime failure is not this, and must never arrive here. See
    `runtime_attempt_record`.
    """
    return CandidateRecord(
        identity=identity, status=FAILED_GENERATION,
        failure={"stage": stage, "error_type": type(error).__name__,
                 "sanitized_reason": sanitize_reason(error),
                 "replacement_generated": False,
                 "deterministic_candidate_semantic": True,
                 "rule": "the frozen budget is 2048 candidates per arm; a failed "
                         "candidate is retained and never resampled"})


def runtime_attempt_dir(directory: Path) -> Path:
    return Path(directory) / RUNTIME_ATTEMPT_DIR


def runtime_attempt_record(identity: GenerationIdentity, *, stage: str,
                           error: BaseException, ordinal: int,
                           diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    """One operational attempt that failed for a reason outside the candidate.

    It binds the generation identity so the provenance is addressable, and it
    carries no field the candidate layer reads as an outcome. The operational
    fields — the timestamp, the ordinal, the backend diagnostics — are recorded
    here precisely BECAUSE they may not touch `GenerationIdentity`: a candidate
    whose identity moved every time the machine hiccuped could not be retried as
    the same candidate.
    """
    return {"schema_version": RUNTIME_ATTEMPT_SCHEMA,
            "outcome": RUNTIME_ATTEMPT_FAILURE,
            "attempt_ordinal": int(ordinal),
            "candidate_id": identity.candidate_id, "arm": identity.arm,
            "position": int(identity.position), "route": identity.route,
            "generation_identity_sha256": identity.digest(),
            "stage": stage, "error_type": type(error).__name__,
            "sanitized_reason": sanitize_reason(error),
            "recorded_at_utc": _utc(),
            "diagnostics": dict(diagnostics or {}),
            "terminal": False, "candidate_consumed": False,
            "rule": ("C5_RUNTIME_RECOVERY_V1: a runtime failure is not a candidate "
                     "outcome. No CANDIDATE.json is written, the pass aborts, and "
                     "the next invocation retries this exact candidate identity "
                     "(recovery-ladder L1). It is never reclassified as a "
                     "generation failure and never resampled.")}


def append_runtime_attempt(directory: Path, identity: GenerationIdentity, *,
                           stage: str, error: BaseException,
                           diagnostics: dict[str, Any] | None = None) -> Path:
    """Record an attempt beside the candidate, keeping every earlier one.

    One file per attempt under a subdirectory, so appending cannot lose or
    rewrite what a previous process observed. Earlier attempts are kept because
    the SECOND occurrence of the same fault is the evidence that says "stop
    retrying and diagnose the implementation or the environment" (L0). A file
    that only ever holds the latest attempt cannot show that.
    """
    from prism_fas.pipeline.state import atomic_write_json

    root = runtime_attempt_dir(directory)
    root.mkdir(parents=True, exist_ok=True)
    ordinal = len(runtime_attempts(directory)) + 1
    record = runtime_attempt_record(identity, stage=stage, error=error,
                                    ordinal=ordinal, diagnostics=diagnostics)
    path = root / f"RUNTIME_ATTEMPT_{ordinal:04d}.json"
    atomic_write_json(path, record)
    return path


def runtime_attempts(directory: Path) -> list[dict[str, Any]]:
    """Every recorded attempt for this candidate, oldest first."""
    root = runtime_attempt_dir(directory)
    if not root.is_dir():
        return []
    found = [read_record(path) for path in sorted(root.glob("RUNTIME_ATTEMPT_*.json"))]
    return [item for item in found if item is not None]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts a completion lock can bind, computed from the records themselves."""
    by_arm: dict[str, dict[str, int]] = {}
    by_route: dict[str, int] = {}
    generated = failed = 0
    for record in records:
        identity = record["generation_identity"]
        arm = by_arm.setdefault(identity["arm"], {"generated": 0, "failed": 0,
                                                  "physics": 0, "gpat": 0})
        if record["status"] == GENERATED:
            arm["generated"] += 1
            arm[identity["route"]] += 1
            by_route[identity["route"]] = by_route.get(identity["route"], 0) + 1
            generated += 1
        else:
            arm["failed"] += 1
            failed += 1
    return {"records": len(records), "generated": generated, "failed": failed,
            "per_arm": {arm: dict(counts) for arm, counts in sorted(by_arm.items())},
            "per_route": dict(sorted(by_route.items()))}


def record_set_digest(records: list[dict[str, Any]]) -> str:
    """Identity over the terminal outcome of every planned candidate."""
    material = sorted(
        f"{item['generation_identity']['candidate_id']}:{item['status']}:"
        f"{item.get('generation_identity_sha256', '')}" for item in records)
    return hashlib.sha256("|".join(material).encode("utf-8")).hexdigest()


def payload_set_digest(records: list[dict[str, Any]]) -> str:
    """Identity over the bytes C6 will read."""
    material = sorted(
        f"{item['generation_identity']['candidate_id']}:{name}:{digest}"
        for item in records if item["status"] == GENERATED
        for name, digest in sorted(item.get("payload_sha256", {}).items()))
    return hashlib.sha256("|".join(material).encode("utf-8")).hexdigest()


__all__ = ["SCHEMA_VERSION", "RECORD_NAME", "GENERATED", "FAILED_GENERATION",
           "IMAGE_NAME", "MASK_NAME", "ARTIFACT_MAP_NAME", "PAYLOAD_NAMES",
           "RawGenerationError", "GenerationIdentity", "CandidateRecord",
           "candidate_dir", "read_record", "write_record", "reuse_decision",
           "RUNTIME_ATTEMPT_FAILURE", "RUNTIME_ATTEMPT_DIR", "RUNTIME_ATTEMPT_SCHEMA",
           "runtime_attempt_dir", "runtime_attempt_record", "append_runtime_attempt",
           "runtime_attempts", "sanitize_reason",
           "write_payload_bytes", "failure_record", "summarize", "record_set_digest",
           "payload_set_digest", "sha256_file"]
