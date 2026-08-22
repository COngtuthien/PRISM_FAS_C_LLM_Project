"""C5 scientific rendering — the loop, and nothing the loop does not need.

Every renderer here is imported. `PhysicsRoute` is the final M7 engine,
`GPATRoute` is the frozen C4 checkpoint bound by its SHA-256, and
`finalize_discrete` is the frozen §3 discretization. This module contributes the
Version-C parts only: which bank an arm renders through, which route each
position takes, and what happens to a candidate that fails.

What it does NOT contribute is a judgement. `SyntheticBankGenerator` evaluates
quality inside its generation loop; this loop stops at `finalize_discrete`. A
`CandidateEvaluator` is never constructed, a `FrozenCalibration` is never loaded
and `quality_gate.evaluate` is never called, because C6 owns that and owns it
after these candidates exist.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from . import c5_raw_generation as raw
from .c5_arm_plan import ArmPlanError, arm_bank_root
from .c5_source_pair_plan import GPAT, PHYSICS

#: The M7 ontology all three C3 banks were generated against. A C3 arm bank
#: records `ontology_identity` but does not carry the ontology itself, so it is
#: resolved from the repository and then checked against what the bank recorded.
ONTOLOGY_CONFIG = "configs/recipes/ontology_m7.yaml"
GPAT_CONFIG = "configs/synthesis/gpat_m8.yaml"


class RenderError(RuntimeError):
    """C5 scientific rendering cannot proceed under the frozen contract."""


class ScientificDeviceUnavailable(RenderError):
    """A scientific GPAT render was asked for on a host with no CUDA device."""

    reason_code = "SCIENTIFIC_DEVICE_UNAVAILABLE"


class SemanticGenerationFailure(RenderError):
    """This candidate cannot exist, and would fail identically on every rerun.

    The deterministic candidate-level failure class. Membership is decided by
    proof, not by resemblance: a failure belongs here only when it is a pure
    function of the frozen inputs, so that retrying it could only produce the
    same answer and retention costs nothing recoverable.

    The one authorized member is the empty exact mask — the artifact did not
    survive uint8 quantization, so the finalized image is byte-identical to the
    live image it was rendered from and there is no candidate to keep. A
    `SyntheticBankError`, a CUDA fault or an OSError is NOT a member merely
    because it happened during rendering; classifying one as semantic would
    permanently spend a candidate on something a rerun might well have produced.
    """

    reason_code = "C5_SEMANTIC_GENERATION_FAILURE"


class RuntimeAttemptFailure(RenderError):
    """The pass could not continue for a reason outside the candidate.

    Raised after the attempt has been recorded, and it aborts the render pass
    rather than consuming the candidate. Nothing terminal is written, no later
    candidate is attempted, and no retry happens inside this process: the next
    `train.py` invocation IS the recovery-ladder L1 retry of the identical
    frozen configuration.
    """

    reason_code = "C5_RUNTIME_ATTEMPT_FAILURE"

    def __init__(self, message: str, *, candidate_id: str, arm: str, position: int,
                 route: str, stage: str, error_type: str, attempt_ordinal: int,
                 attempt_path: str) -> None:
        super().__init__(message)
        self.candidate_id = candidate_id
        self.arm = arm
        self.position = position
        self.route = route
        self.stage = stage
        self.error_type = error_type
        self.attempt_ordinal = attempt_ordinal
        self.attempt_path = attempt_path

    def as_dict(self) -> dict[str, Any]:
        return {"reason_code": self.reason_code, "candidate_id": self.candidate_id,
                "arm": self.arm, "position": self.position, "route": self.route,
                "stage": self.stage, "error_type": self.error_type,
                "attempt_ordinal": self.attempt_ordinal,
                "attempt_record": self.attempt_path,
                "candidate_consumed": False, "terminal": False}


def scientific_device() -> str:
    """CUDA, or nothing. The GPAT route never silently falls back to the CPU.

    3072 GPAT renders through the frozen checkpoint on a CPU would not finish,
    and if they did they would carry bytes produced outside the precision
    contract the checkpoint was frozen under. The physics route is CPU-bound and
    genuinely device-free; this gate exists for the GPAT half.
    """
    from .gpat_trainer import resolve_device

    device = resolve_device(None)
    if not str(device).startswith("cuda"):
        raise ScientificDeviceUnavailable(
            f"scientific C5 GPAT rendering requires CUDA and this host resolved "
            f"{device!r}. Run C5 on the GPU host.")
    return str(device)


def route_bank(repo: Path, arm: str) -> dict[str, Any]:
    """One C3 arm bank in the shape `PhysicsRoute` and `GPATRoute` consume.

    A C3 bank is `C3_BANK.json` + `recipes.jsonl`; an M7 bank is a directory of
    seven files including its own `ontology.yaml`. `load_bank` reads the second
    shape, so this builds the route bank from the first rather than converting a
    C3 bank into M7 format — the C3 banks are frozen and are not rewritten.
    """
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe

    root = arm_bank_root(repo, arm)
    lock = json.loads((root / "C3_BANK.json").read_text(encoding="utf-8"))
    ontology = load_ontology(Path(repo) / ONTOLOGY_CONFIG)
    recorded = str(lock.get("ontology_identity", ""))
    if recorded and ontology.sha256 != recorded:
        raise ArmPlanError(
            f"the {arm} C3 bank was generated against ontology {recorded} and this "
            f"repository resolves {ontology.sha256}; the bank and the ontology "
            "must be the pair that was frozen together")

    recipes = [parse_recipe(json.loads(line)) for line
               in (root / "recipes.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    return {"root": root, "lock": lock, "ontology": ontology, "recipes": recipes,
            "bank_id": f"c3_{arm.lower()}",
            "bank_identity": str(lock["bank_identity"]),
            "ontology_identity": ontology.sha256}


def build_routes(repo: Path, *, checkpoint_path: Path, checkpoint_sha256: str,
                 expected_identity: dict[str, str], device: str) -> dict[str, Any]:
    """The two frozen routes, constructed once for the whole pass.

    `GPATRoute.__init__` re-hashes the checkpoint and loads it `strict=True`
    against the identity C4 locked, so a checkpoint that is not the frozen one
    fails here rather than 3072 renders later.
    """
    from .m8_pipeline import load_gpat_config
    from .synthetic_bank import GPATRoute, PhysicsRoute

    return {
        PHYSICS: PhysicsRoute(),
        GPAT: GPATRoute(Path(checkpoint_path), load_gpat_config(Path(repo) / GPAT_CONFIG),
                        expected_sha256=checkpoint_sha256,
                        expected_identity=dict(expected_identity), device=device,
                        # The A02 conditioning exemption is not declared for C5.
                        conditioning_control=None),
    }


def identity_for(row: dict[str, Any], plan: dict[str, Any]) -> raw.GenerationIdentity:
    """The generation identity of one planned candidate. No calibration input."""
    return raw.GenerationIdentity(
        candidate_id=row["candidate_id"], arm=plan["arm"],
        arm_plan_identity=plan["arm_plan_identity"],
        source_pair_plan_identity=plan["source_pair_plan_identity"],
        package_identity=plan["package_identity"],
        recipe_bank_identity=plan["recipe_bank_identity"],
        recipe_id=row["recipe_id"], recipe_ordinal=int(row["recipe_ordinal"]),
        slot=int(row["slot"]), position=int(row["position"]), route=row["route"],
        live_target_sample_id=row["live_target_sample_id"],
        spoof_source_sample_id=row.get("spoof_source_sample_id"),
        generator_binding=row["generator_binding"],
        ontology_identity=plan["ontology_identity"])


def render_one(store: Any, bank: dict[str, Any], route: Any,
               row: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """One candidate, generated and finalized. Nothing judges it.

    The route emits floats; `finalize_discrete` is the frozen procedure that
    turns them into the bytes a candidate consists of. Its exact mask is the set
    of pixels that really differ AFTER uint8 quantization, so an artifact too
    weak to survive rounding finalizes to an empty mask — a file identical to
    the live image it was rendered from.

    That is a generation failure, and a different thing from a quality
    rejection: the candidate has no artifact at all, not merely an unconvincing
    one. Recording it as a success would hand C6 a bank padded with unmodified
    live images to gate, so it is refused here and retained as a failure.
    """
    from .synthetic_bank import finalize_discrete

    output = route.generate(store, bank, row)
    original, _ = store.load(row["live_target_sample_id"])
    result = finalize_discrete(output.image, original, output.requested_support,
                               output.artifact_map)
    if int(result.exact_mask_pixels) == 0:
        # Deterministic in the frozen inputs: the same recipe on the same sample
        # will finalize to the same empty mask every time, so this is terminal.
        raise SemanticGenerationFailure(
            f"{row['candidate_id']}: the artifact did not survive uint8 "
            f"quantization and finalized to an empty exact mask over "
            f"{int(result.requested_support_pixels)} requested support pixels")
    trace = {"binding": output.binding,
             "requested_region_pixels": int(output.requested_region_pixels),
             "requested_coverage": float(output.requested_coverage),
             "achieved_coverage": float(output.achieved_coverage),
             "requested_support_pixels": int(result.requested_support_pixels),
             "exact_mask_pixels": int(result.exact_mask_pixels),
             "outside_mask_max_error": int(result.outside_mask_max_error),
             "route_trace": dict(output.trace)}
    return result, trace


def render_arm(*, work_root: Path, plan: dict[str, Any], store: Any,
               bank: dict[str, Any], routes: dict[str, Any],
               limit: int | None = None,
               progress: Callable[[dict[str, Any]], None] | None = None
               ) -> dict[str, Any]:
    """Every planned candidate of one arm, exactly once.

    Resume, corruption repair and failure retention are one decision each, taken
    per candidate from what is on disk:

    * a record whose identity agrees and whose payloads still hash is reused and
      not re-rendered — a restarted pass does not redo finished work;
    * a record whose payloads are missing or altered rebuilds THAT candidate,
      under the same identity and the same inputs, and nothing else;
    * a retained failure stays a failure. It is not retried into a success and
      it is not replaced by an extra render, because the budget is frozen.

    `limit` renders a prefix of the plan. It is for a bounded rehearsal of this
    loop, never for a scientific pass, and the caller that passes it is
    responsible for saying so in its artifact — a truncated arm is not an arm.
    """
    work_root = Path(work_root)
    rows = list(plan["candidates"])[:limit] if limit is not None else list(plan["candidates"])
    records: list[dict[str, Any]] = []
    reused = rendered = failed = rebuilt = 0

    for row in rows:
        identity = identity_for(row, plan)
        directory = raw.candidate_dir(work_root, plan["arm"], identity.candidate_id)
        decision = raw.reuse_decision(directory, identity)

        if decision["reusable"]:
            reused += 1
            records.append(raw.read_record(directory / raw.RECORD_NAME))
            if progress is not None:
                progress({"candidate_id": identity.candidate_id, "action": "reused"})
            continue
        if decision["reason"] == "FAILED_GENERATION":
            # Terminal and retained. Re-running the pass does not resample it.
            failed += 1
            records.append(raw.read_record(directory / raw.RECORD_NAME))
            if progress is not None:
                progress({"candidate_id": identity.candidate_id, "action": "retained_failure"})
            continue
        if decision["reason"] in ("PAYLOAD_MISSING", "PAYLOAD_CHANGED"):
            rebuilt += 1

        stage = f"render_{row['route']}"
        prior_attempts = len(raw.runtime_attempts(directory))
        try:
            result, trace = render_one(store, bank, routes[row["route"]], row)
            hashes = raw.write_payload_bytes(directory, result)
            record = raw.CandidateRecord(
                identity=identity, status=raw.GENERATED, payload_sha256=hashes,
                # Operational, never part of the identity or the payload bytes:
                # a candidate that took three attempts is the same candidate.
                trace={**trace, "prior_runtime_attempts": prior_attempts})
            rendered += 1
            action = "rendered"

        except SemanticGenerationFailure as error:
            # The candidate itself cannot exist. Terminal, retained, and the arm
            # is short by one; the pass continues with the next candidate.
            record = raw.failure_record(identity, stage=stage, error=error)
            failed += 1
            action = "semantic_failure"

        except (KeyboardInterrupt, SystemExit):
            # Somebody stopped the process. That is not an outcome for this
            # candidate and not a runtime attempt worth recording: nothing was
            # learned about the render. Everything already completed stays on
            # disk, and the next invocation resumes this exact candidate.
            raise

        except Exception as error:                       # noqa: BLE001
            # Anything else — CUDA, OOM, filesystem, codec, an unexpected
            # SyntheticBankError — says nothing about the candidate. Record the
            # attempt as operational provenance and abort the pass. No
            # CANDIDATE.json, no in-process retry, no later candidate.
            attempt = raw.append_runtime_attempt(
                directory, identity, stage=stage, error=error,
                diagnostics={"route": row["route"], "arm": plan["arm"],
                             "prior_runtime_attempts": prior_attempts})
            if progress is not None:
                progress({"candidate_id": identity.candidate_id,
                          "action": "runtime_incomplete"})
            raise RuntimeAttemptFailure(
                f"C5 aborted on a runtime failure at {identity.candidate_id} "
                f"({plan['arm']} position {identity.position}, route "
                f"{row['route']}): {type(error).__name__}: "
                f"{raw.sanitize_reason(error)}. The candidate was NOT consumed; "
                f"rerun to retry this identical candidate.",
                candidate_id=identity.candidate_id, arm=plan["arm"],
                position=int(identity.position), route=row["route"], stage=stage,
                error_type=type(error).__name__,
                attempt_ordinal=prior_attempts + 1,
                attempt_path=attempt.as_posix()) from error

        # Written last: its presence is what makes the payloads beside it
        # trustworthy to the next process.
        raw.write_record(directory, record)
        records.append(record.as_dict())
        if progress is not None:
            progress({"candidate_id": identity.candidate_id, "action": action})

    return {"arm": plan["arm"], "planned": len(plan["candidates"]),
            "attempted": len(rows), "records": records,
            "reused": reused, "rendered": rendered, "rebuilt": rebuilt,
            "failed": failed,
            "record_set_digest": raw.record_set_digest(records),
            "payload_set_digest": raw.payload_set_digest(records),
            "summary": raw.summarize(records)}


def collect_records(work_root: Path, plans: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Every terminal record the plans name, read back from disk.

    Read back rather than carried in memory, so a completion lock describes what
    a later process would find rather than what this process believed.
    """
    found: list[dict[str, Any]] = []
    for arm, plan in plans.items():
        for row in plan["candidates"]:
            path = raw.candidate_dir(work_root, arm, row["candidate_id"]) / raw.RECORD_NAME
            payload = raw.read_record(path)
            if payload is not None:
                found.append(payload)
    return found


def completeness(plans: dict[str, dict[str, Any]],
                 records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Whether every planned candidate reached a terminal state, and how.

    Deliberately three numbers, not one. "Complete" means every planned position
    has an outcome; it does not mean every position has a payload, and C5 may not
    round the second up to the first. A pass with 6143 generated candidates and
    one retained failure is complete AND short, and both facts travel to C6.
    """
    by_id = {record["generation_identity"]["candidate_id"]: record
             for record in records}
    planned = [(arm, row["candidate_id"]) for arm, plan in plans.items()
               for row in plan["candidates"]]
    missing = [candidate_id for _, candidate_id in planned if candidate_id not in by_id]
    generated = [record for record in by_id.values()
                 if record["status"] == raw.GENERATED]
    failures = [record for record in by_id.values()
                if record["status"] == raw.FAILED_GENERATION]
    return {
        "planned": len(planned), "terminal": len(by_id),
        "generated": len(generated), "failed": len(failures),
        "missing_candidate_ids": missing[:32], "missing": len(missing),
        # Every planned position reached an outcome. This is what "the stage ran
        # to the end" means, and it is the only thing FINALIZE_C5 requires.
        "every_planned_candidate_is_terminal": not missing,
        # ...and this is the separate fact C6 needs, never conflated with it.
        "every_planned_candidate_is_usable": not missing and not failures,
        "failed_candidate_ids": [record["generation_identity"]["candidate_id"]
                                 for record in failures][:32],
        "rule": ("a generation failure is retained and reported; it is never "
                 "resampled and the frozen 2048-per-arm budget never grows"),
    }


__all__ = ["ONTOLOGY_CONFIG", "GPAT_CONFIG", "RenderError", "ScientificDeviceUnavailable",
           "SemanticGenerationFailure", "RuntimeAttemptFailure",
           "scientific_device", "route_bank", "build_routes", "identity_for",
           "render_one", "render_arm", "collect_records", "completeness"]
