"""C11 — label-isolated P3 prediction.

C11 runs inference on the target and must leave no way to recover the answer from
what it writes. §19.2 and the C11 acceptance are specific about what a prediction
may not carry: no ground-truth label, no attack family, no raw path, no
subject/session taxonomy, no hidden target metadata. That list is the substance
of this adapter — most of its checks are about what is *absent* from the rows it
produced.

The prediction fixtures are built by `adapters.tiny.prediction_rows`, which
constructs rows out of video ids and scores alone. That matters: a fixture that
happened to carry a label field would let the forbidden-column audit pass on a
payload the real validator would reject, and the audit would be measuring the
fixture rather than the contract.

Locks come in two layers, both exercised here. Each row gets a PREDICTION_LOCK
binding its checkpoint, calibration, inference config and package identity; the
locks together form a TARGET_PREDICTION_LOCKSET. §19.2 requires the lockset to be
validated **twice** before label capability is granted, so it is validated twice
and both results are recorded.

No real SiW inference happens. The prediction rows are fixtures and say so.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (assert_fixture_permitted,
                                                EngineeringAdapter, RequiredInput, check,
                                                resume_decision, stage_reports_dir, utc,
                                                write_artifact)
from prism_fas.pipeline.execution import ExecutionContext
from prism_fas.pipeline.adapters.tiny import prediction_rows

STAGE_ID = "C11"

BUILD_PREDICTIONS = "BUILD_PREDICTIONS"
LABEL_ISOLATION_AUDIT = "LABEL_ISOLATION_AUDIT"
PREDICTION_LOCKS = "PREDICTION_LOCKS"
DOUBLE_VALIDATION = "DOUBLE_VALIDATION"

MODES: tuple[str, ...] = (BUILD_PREDICTIONS, LABEL_ISOLATION_AUDIT, PREDICTION_LOCKS,
                          DOUBLE_VALIDATION)

#: Everything a prediction row may never carry (§19.2, C11 acceptance).
FORBIDDEN_FIELDS: tuple[str, ...] = (
    "label", "true_label", "label_live_spoof", "true_target", "target", "class_target",
    "attack_type", "attack_family", "taxonomy", "subject", "subject_id", "session",
    "session_id", "raw_path", "source_path", "video_path", "ground_truth")

#: Prediction rows per variant, at the frozen 4 frames per video — so 40 videos.
#:
#: Sized by a downstream requirement rather than by taste: C12's paired video
#: bootstrap resamples videos with replacement, and the canonical metrics
#: correctly refuse to compute APCER/ACER on a resample that contains only one
#: class. With a handful of videos that happens routinely and the smoke would
#: fail for a reason unrelated to the code under test. Forty balanced videos make
#: a single-class resample vanishingly unlikely.
#:
#: This is not a scientific budget dimension. L.12's reducible dimensions are
#: samples, steps, epochs and seeds; the number of fixture rows needed to make a
#: statistical path well defined is none of those.
FIXTURE_ROWS = 160


@dataclass
class C11Adapter(EngineeringAdapter):
    """The C11 execution adapter. Prediction and lock logic are imported."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Label-isolated P3 prediction"
    modes: tuple[str, ...] = MODES
    requires_gpu: bool = True

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("c9_source_lock", "reports/full/c9/SOURCE_MATRIX_LOCK_C.json",
                          "the source freeze that must precede any prediction"),
            RequiredInput("c10_target_lock", "reports/full/c10/TARGET_PACKAGE_LOCK.json",
                          "the target package and capability lock"),
            RequiredInput("target_feature_package", "data/processed/prism_target_eval_v2",
                          "the label-free SiW feature package, mounted READ-ONLY"),
        )

    def workflow(self, request: AdapterRequest,
                 context: ExecutionContext) -> list[AdapterResult]:
        reports = stage_reports_dir(request, STAGE_ID)
        rows, build = self._build(request, reports)
        locks, lock_result = self._locks(request, rows, reports)
        return [build, self._isolation(request, rows, reports), lock_result,
                self._double_validation(request, locks, reports)]

    # --- modes ----------------------------------------------------------------

    def _build(self, request: AdapterRequest,
               reports: Path) -> tuple[list[dict[str, Any]], AdapterResult]:
        """Build the fixture prediction rows the label-isolation contract is tested on.

        Guarded first: these rows are constructed from video ids and scores by
        `adapters.tiny.prediction_rows`, and a scientific C11 must produce
        predictions by running the frozen checkpoints over the sealed feature
        package. Writing constructed scores into a scientific PREDICTION_LOCK
        would put invented numbers behind a lockset C12 is required to validate
        twice before label capability is granted.
        """
        from prism_fas.evaluation.target_prediction import (VariantCapabilities,
                                                            build_prediction_row,
                                                            prediction_logical_identity,
                                                            validate_predictions)

        assert_fixture_permitted(request.context,
                                 "the C11 constructed target prediction rows")

        checks: list[dict[str, Any]] = []
        from prism_fas.detector.variant import ResolvedExperimentVariant
        from prism_fas.pipeline.adapters.c7 import TRACK_G_FLAGS, TRACK_R_FLAGS

        rows: list[dict[str, Any]] = []
        for name, flags in (("C-G-LLM", TRACK_G_FLAGS), ("C-R-LLM", TRACK_R_FLAGS)):
            variant = ResolvedExperimentVariant.resolve(flags)
            capabilities = VariantCapabilities.from_variant(variant)
            for source in prediction_rows(FIXTURE_ROWS, seed=hash(name) % 10_000):
                rows.append(build_prediction_row(
                    sample_id=f"{name}:{source['sample_id']}",
                    video_id=f"{name}:{source['video_id']}",
                    frame_id=source["frame_id"], p_global=source["p_global"],
                    s_region=0.25 if capabilities.has_region else None,
                    p_prompt=None,   # §13.4.4: null on ordinary target frames
                    threshold=source["threshold"], unknown_threshold=None,
                    top_region_ids=[0, 1] if capabilities.has_region_detail
                    and capabilities.has_region else [],
                    region_distances=[0.1] * 9 if capabilities.has_region_detail
                    and capabilities.has_region else [],
                    checkpoint_hash=f"fixture-checkpoint-{name}",
                    calibration_hash=f"fixture-calibration-{name}",
                    inference_config_hash="fixture-inference-config",
                    variant=name))

        by_variant: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_variant.setdefault(row["variant"], []).append(row)

        report = validate_predictions(rows)
        checks.append(check(
            "c11_prediction_rows_validate", bool(report),
            f"{len(rows)} prediction rows satisfy the frozen prediction schema",
            rows=len(rows), validation=report,
            builder="prism_fas.evaluation.target_prediction (canonical)"))
        checks.append(check(
            "c11_prompt_is_not_applicable_at_target",
            all(row.get("p_prompt") is None for row in rows),
            "p_prompt is null on ordinary target frames rather than 0.0",
            rule="§12.3 and §13.4.4: the recipe/attack mask does not exist for a real "
                 "target sample, so a numeric 0.0 would be a fabricated measurement"))
        # `s_region` follows whether the variant FUSES a regional evidence term,
        # not whether it has a region branch. Under the v1.5 manifold-OFF Track R
        # the region enters the LOGIT and no s_region scalar exists at all, so
        # both tracks correctly write null — and writing a value here would be
        # reporting a quantity the decision never used.
        expected_region = {name: VariantCapabilities.from_variant(
            ResolvedExperimentVariant.resolve(flags)).has_region
            for name, flags in (("C-G-LLM", TRACK_G_FLAGS), ("C-R-LLM", TRACK_R_FLAGS))}
        checks.append(check(
            "c11_region_fields_follow_variant_capability",
            all((row.get("s_region") is not None) == expected_region[row["variant"]]
                for row in rows),
            "s_region is written exactly when the variant fuses a regional evidence term",
            fuses_region_evidence=expected_region,
            note="v1.5 Track R fuses the region into fused_logit_R rather than into a "
                 "post-hoc s_region score, and its manifold is OFF, so s_region is null",
            rows_with_s_region=sum(1 for row in rows if row.get("s_region") is not None)))

        # The row table is bulk payload, so it follows the repository's existing
        # rule for prediction tables: held on disk, referenced by hash, kept out
        # of Git. The committed artifact carries the counts, the validation and
        # the content identity — everything a reader needs to judge the run —
        # and names where the rows live.
        rows_path = reports / "C11_PREDICTION_ROWS.json"
        write_artifact(request, rows_path, {
            "schema_version": "c11-prediction-rows-v1", "generated_at_utc": utc(),
            "rows": rows, "row_count": len(rows), "fixture_backed": request.context.fixtures_permitted})
        rows_identity = prediction_logical_identity(rows)

        artifact = write_artifact(request, reports / "C11_PREDICTIONS.json", {
            "schema_version": "c11-predictions-v1", "generated_at_utc": utc(),
            "mode": BUILD_PREDICTIONS, "row_count": len(rows),
            "rows_by_variant": {name: len(group) for name, group in by_variant.items()},
            "prediction_logical_identity": rows_identity,
            "rows_artifact": rows_path.relative_to(request.repo).as_posix(),
            "rows_artifact_in_git": False,
            "validation": report, "fixture_backed": request.context.fixtures_permitted,
            "real_target_inference_performed": False,
            "note": "fixture prediction rows built from video ids and scores alone. No "
                    "SiW feature was read and no model was run against the target"})
        return rows, self.result(request, mode=BUILD_PREDICTIONS, checks=checks,
                                 artifacts=[artifact])

    def _isolation(self, request: AdapterRequest, rows: list[dict[str, Any]],
                   reports: Path) -> AdapterResult:
        checks: list[dict[str, Any]] = []
        present = sorted({key for row in rows for key in row
                          if key.lower() in FORBIDDEN_FIELDS})
        checks.append(check(
            "c11_no_forbidden_field_in_any_row", not present,
            "no prediction row carries a label, attack family, taxonomy or raw path",
            forbidden_fields=list(FORBIDDEN_FIELDS), found=present,
            rows_scanned=len(rows)))

        values = " ".join(str(value) for row in rows for value in row.values()).lower()
        leaked = [token for token in ("live", "spoof", "replay", "print", "mask3d",
                                      "paper", "silicone")
                  if f'"{token}"' in values]
        checks.append(check(
            "c11_no_class_or_family_token_in_values", not leaked,
            "no row VALUE spells out a class or attack family",
            tokens_checked=["live", "spoof", "replay", "print", "mask3d", "paper",
                            "silicone"], found=leaked))

        video_ids = {row["video_id"] for row in rows}
        checks.append(check(
            "c11_video_id_is_opaque",
            all(":" in value and "/" not in value and "\\" not in value
                for value in video_ids),
            "video identifiers are opaque ids rather than filesystem paths",
            sample=sorted(video_ids)[:3]))

        decisions = {row.get("decision") for row in rows}
        checks.append(check(
            "c11_decision_is_derived_from_the_score_not_a_label",
            None not in decisions,
            "each row's decision comes from its own calibrated score and threshold",
            distinct_decisions=sorted(str(value) for value in decisions)))

        artifact = write_artifact(request, reports / "C11_LABEL_ISOLATION.json", {
            "schema_version": "c11-label-isolation-v1", "generated_at_utc": utc(),
            "mode": LABEL_ISOLATION_AUDIT,
            "forbidden_fields": list(FORBIDDEN_FIELDS),
            "forbidden_fields_found": present,
            "target_labels_opened": False, "target_labels_resolved": 0,
            "rows_scanned": len(rows), "fixture_backed": request.context.fixtures_permitted,
            "procedural_note": ("this is procedural isolation of the Version-C prediction "
                                "process. It is not a claim that researchers have never "
                                "seen SiW labels historically (§19.1)")})
        return self.result(request, mode=LABEL_ISOLATION_AUDIT, checks=checks,
                           artifacts=[artifact])

    def _locks(self, request: AdapterRequest, rows: list[dict[str, Any]],
               reports: Path) -> tuple[dict[str, Any], AdapterResult]:
        from prism_fas.evaluation.target_prediction import (build_lockset,
                                                            build_prediction_lock,
                                                            prediction_logical_identity,
                                                            validate_prediction_lock)

        checks: list[dict[str, Any]] = []
        locks: list[dict[str, Any]] = []
        by_variant: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_variant.setdefault(row["variant"], []).append(row)

        for variant, group in sorted(by_variant.items()):
            lock = build_prediction_lock(
                experiment_id=variant, variant=variant, seed=20260806, rows=group,
                checkpoint_sha256=f"fixture-checkpoint-{variant}",
                source_calibration_sha256=f"fixture-calibration-{variant}",
                calibration_hash=f"fixture-calibration-{variant}",
                inference_config_hash="fixture-inference-config",
                target_feature_package_identity="fixture-package-identity",
                target_package_id="fixture_target_eval",
                threshold=0.5, unknown_threshold=None, engineering_smoke=True)
            locks.append(lock)
            report = validate_prediction_lock(
                lock, group,
                expected_checkpoint_sha256=f"fixture-checkpoint-{variant}",
                expected_calibration_hash=f"fixture-calibration-{variant}",
                expected_inference_config_hash="fixture-inference-config",
                expected_package_identity="fixture-package-identity")
            checks.append(check(
                f"c11_prediction_lock_{variant.lower().replace('-', '_')}", bool(report),
                f"{variant}'s prediction lock builds and validates against its own rows",
                rows=len(group), lock_keys=sorted(lock)[:8]))

        # The locked logical identity binds the DECISION quantities — s_final,
        # decision_score, confidence and the decision itself — so those are what
        # a tamper test must move. Mutating an upstream field that the identity
        # does not hash would prove nothing about the lock.
        first_variant = sorted(by_variant)[0]
        tamper_cases: list[dict[str, Any]] = []
        for field in ("decision_score", "s_final", "confidence"):
            tampered = list(by_variant[first_variant])
            tampered[0] = {**tampered[0], field: float(tampered[0][field]) * 0.5 + 0.25}
            try:
                validate_prediction_lock(locks[0], tampered)
                tamper_cases.append({"field": field, "refused": False})
            except Exception as error:
                tamper_cases.append({"field": field, "refused": True,
                                     "error": f"{type(error).__name__}: {str(error)[:120]}"})
        dropped = list(by_variant[first_variant])[:-1]
        try:
            validate_prediction_lock(locks[0], dropped)
            tamper_cases.append({"field": "row_count", "refused": False})
        except Exception as error:
            tamper_cases.append({"field": "row_count", "refused": True,
                                 "error": f"{type(error).__name__}: {str(error)[:120]}"})

        checks.append(check(
            "c11_lock_refuses_mutated_predictions",
            all(case["refused"] for case in tamper_cases),
            "changing any locked decision quantity, or dropping a row, invalidates the lock",
            cases=tamper_cases,
            identity_binds=["sample_id", "video_id", "frame_id", "s_final",
                            "decision_score", "confidence", "decision"],
            rule="§19.2: predictions are frozen; no optimizer or checkpoint mutation is "
                 "permitted and the rows may not move under their own lock"))

        # The global lockset is the artifact that unlocks label capability at C12,
        # and the canonical builder refuses any lock marked as an engineering
        # smoke. That refusal is the guarantee C11 readiness most needs, so it is
        # asserted rather than worked around: under a non-eligible profile there
        # is no way to produce a lockset at all, and therefore no way for a smoke
        # run to reach the scorer's label capability.
        lockset_refused, refusal = False, ""
        try:
            build_lockset(locks, matrix_identity="fixture-matrix",
                          registry_identity="fixture-registry",
                          target_feature_package_identity="fixture-package-identity")
        except Exception as error:
            lockset_refused, refusal = True, f"{type(error).__name__}: {error}"
        checks.append(check(
            "c11_lockset_refuses_engineering_smoke_predictions", lockset_refused,
            "the global lockset cannot be built from engineering-smoke prediction locks",
            refusal=refusal, locks_offered=len(locks),
            consequence="a smoke run can never produce the artifact that grants the "
                        "scorer label capability at C12",
            builder="prism_fas.evaluation.target_prediction.build_lockset (canonical)"))
        checks.append(check(
            "c11_logical_identity_is_content_addressed", True,
            "prediction identity is derived from the rows themselves",
            identities={variant: prediction_logical_identity(group)[:16]
                        for variant, group in sorted(by_variant.items())}))

        artifact = write_artifact(request, reports / "C11_PREDICTION_LOCKSET.json", {
            "schema_version": "c11-prediction-lockset-v1", "generated_at_utc": utc(),
            "mode": PREDICTION_LOCKS, "locks": locks,
            "lockset": None, "lockset_refused": lockset_refused,
            "lockset_refusal": refusal,
            "is_scientific_lockset": False, "fixture_backed": request.context.fixtures_permitted,
            "why_no_lockset": ("every lock here is marked engineering_smoke, and the "
                               "canonical builder refuses those. The scientific lockset is "
                               "built at C11 under the full profile from real label-free "
                               "inference")})
        return {"lockset": None, "locks": locks,
                "rows_by_variant": by_variant}, self.result(
            request, mode=PREDICTION_LOCKS, checks=checks, artifacts=[artifact])

    def _double_validation(self, request: AdapterRequest, bundle: dict[str, Any],
                           reports: Path) -> AdapterResult:
        """§19.2: validate twice before label capability is granted.

        The double validation is applied to the per-row locks, which is what
        exists under a non-eligible profile. Each pass re-derives the lock's
        identity from the rows rather than re-reading a cached verdict — two
        passes that both consult the same memo would agree for the wrong reason.
        """
        from prism_fas.evaluation.target_prediction import (prediction_logical_identity,
                                                            validate_prediction_lock)

        checks: list[dict[str, Any]] = []
        predictions = bundle["rows_by_variant"]

        def pass_over() -> dict[str, Any]:
            results: dict[str, Any] = {}
            for lock in bundle["locks"]:
                rows = predictions[lock["experiment_id"]]
                try:
                    validate_prediction_lock(lock, rows)
                    results[lock["experiment_id"]] = {
                        "valid": True,
                        "logical_identity": prediction_logical_identity(rows)}
                except Exception as error:
                    results[lock["experiment_id"]] = {
                        "valid": False, "error": f"{type(error).__name__}: {error}"}
            return results

        first, second = pass_over(), pass_over()
        agreed = first == second
        checks.append(check(
            "c11_locks_validated_twice_and_agree",
            agreed and all(item["valid"] for item in first.values()),
            "two independent validation passes re-derived identical results",
            first_pass=first, second_pass=second, agreed=agreed,
            rule="§19.2: create per-row PREDICTION_LOCK and a global "
                 "TARGET_PREDICTION_LOCKSET; validate twice before label capability is "
                 "granted to the scorer"))
        checks.append(check(
            "c11_no_label_capability_granted_here", True,
            "C11 grants no label capability; that happens at C12 and only after a real "
            "lockset exists",
            label_capability_granted=False, stage_permission="G7: target_label_root=deny",
            lockset_present=bundle["lockset"] is not None))

        artifact = write_artifact(request, reports / "C11_DOUBLE_VALIDATION.json", {
            "schema_version": "c11-double-validation-v1", "generated_at_utc": utc(),
            "mode": DOUBLE_VALIDATION, "first_pass": first, "second_pass": second,
            "agreed": agreed, "fixture_backed": request.context.fixtures_permitted})

        decision = resume_decision(request, "c11_prediction_lockset",
                                   reports / "C11_PREDICTION_LOCKSET.json",
                                   expected_identity="c11-prediction-lockset-v1",
                                   identity_key="schema_version")
        checks.append(check(
            "c11_resume_is_identity_aware", decision["identity_matches"],
            "resume validates the lockset artifact by identity", **decision))
        return self.result(request, mode=DOUBLE_VALIDATION, checks=checks,
                           artifacts=[artifact])


__all__ = ["STAGE_ID", "MODES", "BUILD_PREDICTIONS", "LABEL_ISOLATION_AUDIT",
           "PREDICTION_LOCKS", "DOUBLE_VALIDATION", "FORBIDDEN_FIELDS", "C11Adapter"]
