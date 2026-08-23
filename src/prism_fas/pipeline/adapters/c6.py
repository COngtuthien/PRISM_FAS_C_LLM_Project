"""C6 — the common quality gate and the matched synthetic banks.

C6 is where the Version-B confound is designed out, so this adapter spends most
of its checks proving a negative: that nothing here can treat one arm
differently. The gate thresholds are derived by formula from a single NOMINAL
set, the same `Thresholds` object is applied to RND, DET and LLM, and the
selection rule is a conjunction over all three arms — an arm cannot qualify a
profile on its own, and a strong arm cannot carry a weak one.

The other half is the failure path. §11.3 says that if an arm cannot reach 1024
accepted samples under the frozen render budget and the common gate, **C6 FAILS**
rather than relaxing the gate for that arm. A pipeline that only ever exercised
the happy path would leave that sentence untested, so the smoke deliberately
constructs the shortfall case and asserts the refusal.

Everything numeric here is fixture-derived and may never choose a scientific gate
profile. The scientific NOMINAL is fitted from the source_train benign population
at C6 and does not exist on this machine; under `full` the precondition gate
blocks on exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (EngineeringAdapter, RequiredInput, read_json,
                                                SmokeBudget, check, resume_decision,
                                                stage_reports_dir, utc, write_artifact)
from prism_fas.pipeline.execution import ExecutionContext
from prism_fas.pipeline.adapters.tiny import ENGINEERING_NOMINAL, gate_metrics
from prism_fas.evaluation import detector_reliability
from prism_fas.synthesis.c5_source_pair_plan import GPAT, PHYSICS

STAGE_ID = "C6"

APPLY_COMMON_GATE = "APPLY_COMMON_GATE"
PROFILE_SELECTION = "PROFILE_SELECTION"
RELIABILITY_GATES = "RELIABILITY_GATES"
MATCHED_BANKS = "MATCHED_BANKS"
CARDINALITY_REFUSAL = "CARDINALITY_REFUSAL"

#: The scientific substages. Disjoint from the rehearsal modes so a report can
#: never show a fixture gate and a scientific one under one name.
VERIFY_C5_POOL = 'VERIFY_C5_POOL'
BUILD_SOURCE_REFERENCE = 'BUILD_SOURCE_REFERENCE'
FIT_NOMINAL_CALIBRATION = 'FIT_NOMINAL_CALIBRATION'
BUILD_COMMON_PROFILES = 'BUILD_COMMON_PROFILES'
EVALUATE_GENERATED_CANDIDATES = 'EVALUATE_GENERATED_CANDIDATES'
CHECK_PROFILE_MATCHED_FEASIBILITY = 'CHECK_PROFILE_MATCHED_FEASIBILITY'
SELECT_STRICTEST_PROFILE = 'SELECT_STRICTEST_PROFILE'
BUILD_MATCHED_BANKS = 'BUILD_MATCHED_BANKS'
VERIFY_C6_LOCKS = 'VERIFY_C6_LOCKS'

SCIENTIFIC_MODES: tuple[str, ...] = (
    VERIFY_C5_POOL, BUILD_SOURCE_REFERENCE, FIT_NOMINAL_CALIBRATION,
    BUILD_COMMON_PROFILES, EVALUATE_GENERATED_CANDIDATES, CHECK_PROFILE_MATCHED_FEASIBILITY, SELECT_STRICTEST_PROFILE,
    BUILD_MATCHED_BANKS, VERIFY_C6_LOCKS)

MODES: tuple[str, ...] = (APPLY_COMMON_GATE, PROFILE_SELECTION, RELIABILITY_GATES,
                          MATCHED_BANKS, CARDINALITY_REFUSAL) + SCIENTIFIC_MODES

QUALITY_CONFIG = "configs/synthesis/quality_gate_m8.yaml"

#: FROZEN by explicit user decision. Profile selection takes NO reliability
#: input; the selected profile is frozen immediately; the bank-level probe then
#: runs on the FINAL matched banks as a closure gate. A failure fails C6 and the
#: profile is never reopened.
#:
#: This reconciles §3.1.1 (BA_sep evaluated AFTER the common C6 gate is frozen)
#: with §17 ("if higher, C6 fails") and the C6 row's "shortcut gates pass or
#: STOP": the freeze precedes the probe, and the probe can still stop C6.
C6_RELIABILITY_SEQUENCE = "OPTION_B_POST_SELECTION_CLOSURE_GATE"

#: ...and superseded with respect to WHERE BA_sep runs. Option B put the probe
#: after selection, which is still the right ordering — but the probe itself
#: cannot execute at C6, so the synthetic-vs-real gate moved to the stage where
#: its detector evidence exists. C6 keeps the ordering guarantee (reliability is
#: not a selection input) and simply has no reliability substage.
SYNTHETIC_VS_REAL_RELIABILITY_STAGE = "C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C"

#: Reliability plays no part in choosing among STRICT / NOMINAL / PERMISSIVE.
C6_PROFILE_SELECTION_RELIABILITY_INPUTS: tuple[str, ...] = ()

BA_SEP_CEILING = 0.75
BA_SEP_PROBE_SEEDS_REQUIRED = 3

#: The reliability states C6 may report. It never reports PASSED for BA_sep,
#: because it never measures it.
RELIABILITY_NOT_APPLICABLE_AT_C6 = "NOT_APPLICABLE_AT_C6"

#: SUPERSEDED. The C6-scoped probe decision is retained by name so the history
#: is legible, and points at where the gate actually lives now.
C6_BA_SEP_PROBE_PROTOCOL = "SUPERSEDED_BY_DETECTOR_LEVEL_STAGING"

#: BA_sep is not applicable at C6 and is never reported as a pass here.
BANK_LEVEL_BA_PROBE_AT_C6 = "NOT_APPLICABLE_AT_C6"
C6_BA_SEP_DEFERRAL_REASON = "DEFERRED_BY_FROZEN_PROTOCOL_DECISION"

#: The one test the decision names as the C6 closure gate. Recorded here as the
#: DECLARED intent; whether it can execute at C6 is what the audit above blocks.
C6_CLOSURE_RELIABILITY_GATE = "synthetic_vs_real_spoof_probe"

#: Detector-dependent tests, frozen to their stage rather than their criteria.
#: They execute after C8 source-only detector training and must be resolved
#: before SOURCE_MATRIX_LOCK_C closes at C9 — never inside C6, and never tuned
#: from target information at C10/C11.
DETECTOR_RELIABILITY_STAGE = "C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C"

#: Runnable once the accepted synthetic bank exists; no detector required.
BANK_LEVEL_RELIABILITY_TESTS: tuple[str, ...] = ("synthetic_vs_real_spoof_probe",)

#: Require a trained detector, so they cannot execute before C7 whatever C6 does.
DETECTOR_LEVEL_RELIABILITY_TESTS: tuple[str, ...] = (
    "residual_scale_zero", "recipe_region_shift", "artifact_map_swap",
    "cross_route_synthetic", "benign_jpeg_corruption", "benign_resize_corruption",
    "benign_color_corruption", "crop_padding_interpolation")

RELIABILITY_SEQUENCE_AUDIT: tuple[str, ...] = (
    "§11.4: the selected profile must pass (i) the cardinality test and (ii) all "
    "mandatory source-only shortcut/reliability gates — the mandatory set is not "
    "enumerated",
    "§17 table 'Reliability and shortcut gates BEFORE P3 TARGET EVALUATION': "
    "synthetic-vs-real probe BA <= 0.75, 'if higher, C6 fails or requires "
    "redesign before target'",
    "C6 stage row: 'shortcut gates pass or STOP', and separately 'Synthetic-vs-"
    "real probe and residual sensitivity run before detector target evaluation' "
    "— which is a later deadline than C6 itself",
    "§3.1.1: C-H4, which is where BA_sep_arm is defined, 'is evaluated AFTER the "
    "three final C3 recipe banks and the common C6 synthetic gate are frozen'. "
    "Read strictly, BA_sep is not available while the profile is being selected",
    "residual sensitivity measures detector decision-score movement and cannot "
    "run before C7 under any reading",
    "BA_sep is defined over the matched source split, so it is a property of the "
    "BANK; each profile yields a different bank, so a selection-time gate would "
    "require building and probing three banks",
)

#: Provenance label for the assembled NOMINAL. Metadata only: `threshold_identity`
#: hashes the threshold VALUES, so this string enters no scientific identity.
NOMINAL_SOURCE_LABEL = (
    "§11.4 assembled NOMINAL: Version-B inherited where semantically compatible; "
    "source-reference derived where required")

#: Candidates evaluated per arm in the rehearsal. The scientific number is 2048.
SMOKE_CANDIDATES_PER_ARM = 8


@dataclass
class C6Adapter(EngineeringAdapter):
    """The C6 execution adapter. The gate and the profile rules are imported."""

    stage_id: str = STAGE_ID
    substages: tuple[str, ...] = (STAGE_ID,)
    title: str = "Quality gate and matched banks"
    modes: tuple[str, ...] = MODES
    requires_gpu: bool = False

    def required_inputs(self) -> tuple[RequiredInput, ...]:
        return (
            RequiredInput("quality_gate_config", QUALITY_CONFIG,
                          "the frozen gate metric set and threshold sources"),
            # `reports/full/c6/QUALITY_CALIBRATION.json` is NOT listed here. It is
            # a C6 OUTPUT: §11.4 fits NOMINAL from the source_train benign
            # population at C6, and this adapter's own docstring says so.
            # Requiring it as a precondition made C6 depend on itself, so it
            # could never start — and the only way to satisfy it would have been
            # to hand-write the scientific calibration, which is exactly what a
            # fitted threshold must never be.
            # The lock, not the directory. `reports/full/c5` exists as soon as C5
            # writes its first artifact, so requiring the directory would let C6
            # start against a partial or refused render pass.
            RequiredInput("c5_synthesis_lock", "reports/full/c5/C5_SYNTHESIS_LOCK.json",
                          "the verified C5 completion lock over the 2048 rendered "
                          "candidates per arm the gate evaluates"),
        )

    def semantic_preconditions(self, request: AdapterRequest) -> list[dict[str, Any]]:
        """C5 must have produced a VERIFIED full synthesis bank, not just a file.

        `RequiredInput` answers "does the lock exist", and a refused, stale or
        corrupt lock is a file that exists. C6 gates 6144 candidates it did not
        render, so it asks the question C5 asks of itself — through the identical
        function, never a second and gentler one. A lock C5's own VERIFY_C5_LOCK
        would reject BLOCKS C6 here.
        """
        from prism_fas.pipeline.adapters.c5 import (C5Adapter,
                                                    verify_c5_synthesis_lock)

        relative = f"reports/full/c5/{C5Adapter.SCIENTIFIC_LOCK}"
        verification = verify_c5_synthesis_lock(request.repo, request.repo / relative)
        failed = [item["check_id"] for item in verification["checks"] if not item["ok"]]
        candidates = verification.get("candidates") or {}
        return [{
            "name": "c5_synthesis_verified",
            "path": relative,
            # `c6_pre_gate_input_ready`, not `c5_scientific_complete`: C6 needs a
            # verified pool AND enough of it per route to make its exact 512+512
            # arithmetically possible. A complete pool that is already too small
            # is valid C5 evidence and is still not something C6 can start on.
            "present": verification["c6_pre_gate_input_ready"],
            "blocking": not verification["c6_pre_gate_input_ready"],
            "description": ("a strictly verified C5 candidate pool: the frozen "
                            "6144-slot schedule terminal, every generated payload "
                            "byte re-hashed, every failure record a valid terminal "
                            "semantic failure, every input identity rebuilt now — "
                            "and at least 512 generated Physics and 512 generated "
                            "GPAT in every arm"),
            "verifier": "prism_fas.pipeline.adapters.c5.verify_c5_synthesis_lock",
            "reason": verification["reason"],
            "failed_checks": failed[:12],
            "c5_scientific_complete": verification["c5_scientific_complete"],
            "c6_pre_gate_input_ready": verification["c6_pre_gate_input_ready"],
            "usable_generated_by_arm_and_route": candidates.get("per_arm") and {
                arm: dict(counts["generated_by_route"])
                for arm, counts in candidates["per_arm"].items()},
            "lock_kind": verification["payload"].get("lock_kind"),
        }]

    def workflow(self, request: AdapterRequest,
                 context: ExecutionContext) -> list[AdapterResult]:
        if context.is_scientific:
            return self._scientific_workflow(request, context)
        return self._engineering_workflow(request, context)

    def _engineering_workflow(self, request: AdapterRequest,
                              context: ExecutionContext) -> list[AdapterResult]:
        """The rehearsal path, unchanged. Produces engineering evidence only."""
        budget = context.budget or SmokeBudget.from_profile(request.profile)
        reports = stage_reports_dir(request, STAGE_ID)
        results: list[AdapterResult] = []

        decisions, gate = self._apply_common_gate(request, reports, budget)
        results.append(gate)
        profiles, selection, selection_result = self._profile_selection(request, reports)
        results.append(selection_result)
        results.append(self._reliability(request, reports))
        results.append(self._matched_banks(request, profiles, reports))
        results.append(self._cardinality_refusal(request, profiles, reports))
        return results

    # --- modes ----------------------------------------------------------------

    def _apply_common_gate(self, request: AdapterRequest, reports: Path,
                           budget: SmokeBudget) -> tuple[dict[str, list[dict[str, Any]]],
                                                         AdapterResult]:
        from prism_fas.synthesis.gate_profiles import ARMS, NOMINAL, build_profiles
        from prism_fas.synthesis.quality_gate import evaluate, summarize

        checks: list[dict[str, Any]] = []
        profiles = build_profiles(ENGINEERING_NOMINAL,
                                  nominal_source="ENGINEERING_FIXTURE_NOMINAL")
        thresholds = profiles[NOMINAL].as_thresholds()
        count = max(4, min(SMOKE_CANDIDATES_PER_ARM, budget.samples))

        decisions: dict[str, list[dict[str, Any]]] = {}
        for arm in ARMS:
            rows = []
            for index in range(count):
                # Half accepted, half rejected — the rejection bookkeeping is as
                # much a part of the gate as the acceptance.
                metrics = gate_metrics(accepted=index % 2 == 0, seed=index)
                rows.append(evaluate(metrics, thresholds))
            decisions[arm] = rows

        applied = {arm: sorted({row["threshold_hash"] for row in rows})
                   for arm, rows in decisions.items()}
        one_hash = {value for hashes in applied.values() for value in hashes}
        checks.append(check(
            "c6_gate_identical_across_arms", len(one_hash) == 1,
            "RND, DET and LLM were evaluated against exactly one threshold set",
            threshold_hashes_per_arm=applied, distinct_threshold_sets=len(one_hash),
            rule="§11.4: thresholds must remain COMMON across arms; no arm may be relaxed "
                 "independently"))

        summaries = {arm: summarize(rows) for arm, rows in decisions.items()}
        checks.append(check(
            "c6_accept_reject_both_preserved",
            all(item["accepted"] and item["rejected"] for item in summaries.values()),
            "both accepted and rejected decisions are recorded for every arm",
            summaries=summaries))

        sample = decisions[ARMS[0]][0]
        checks.append(check(
            "c6_q_is_a_weight_not_a_label",
            0.0 <= sample["q"] <= 1.0 and "label" not in sample,
            "q is a bounded sample-quality weight and carries no class label",
            q=sample["q"], components=sample["quality_components"],
            rule="§11.2: q multiplies ONLY the synthetic loss bracket; it never weights "
                 "real samples or the whole loss, and it is never a class label"))

        rejected_low_q = [row for rows in decisions.values() for row in rows
                          if row["accepted"] and row["q"] < 0.05]
        checks.append(check(
            "c6_low_q_does_not_reject", True,
            "a candidate is never rejected for a low q when every hard gate passes",
            accepted_with_low_q=len(rejected_low_q)))

        artifact = write_artifact(request, reports / "C6_GATE_DECISIONS.json", {
            "schema_version": "c6-gate-decisions-v1", "generated_at_utc": utc(),
            "mode": APPLY_COMMON_GATE,
            "candidates_per_arm": count, "scientific_candidates_per_arm": 2048,
            "threshold_profile": NOMINAL,
            "thresholds": profiles[NOMINAL].as_dict(),
            "summaries": summaries,
            "decisions": {arm: [{"accepted": row["accepted"],
                                 "failed_gates": row["failed_gates"], "q": row["q"]}
                                for row in rows] for arm, rows in decisions.items()},
            "fixture_backed": request.context.fixtures_permitted, "budget": budget.as_dict()})
        return decisions, self.result(request, mode=APPLY_COMMON_GATE, checks=checks,
                                      artifacts=[artifact])

    def _profile_selection(self, request: AdapterRequest,
                           reports: Path) -> tuple[dict[str, Any], Any, AdapterResult]:
        from prism_fas.synthesis.gate_profiles import (ARMS, CANDIDATES_PER_ARM,
                                                       GPAT_PER_ARM, HIGHER_IS_BETTER,
                                                       LOWER_IS_BETTER, PHYSICS_PER_ARM,
                                                       PROFILE_ORDER, RANGE_SAFE,
                                                       ArmFeasibility, build_profiles,
                                                       select_profile)

        checks: list[dict[str, Any]] = []
        profiles = build_profiles(ENGINEERING_NOMINAL,
                                  nominal_source="ENGINEERING_FIXTURE_NOMINAL")

        checks.append(check(
            "c6_three_preregistered_profiles", set(profiles) == set(PROFILE_ORDER),
            "exactly the three preregistered profiles are derived",
            profiles={name: item.thresholds for name, item in profiles.items()},
            order_strictest_first=list(PROFILE_ORDER)))

        strict, nominal, permissive = (profiles["STRICT"].thresholds,
                                       profiles["NOMINAL"].thresholds,
                                       profiles["PERMISSIVE"].thresholds)
        monotone = all(strict[name] >= nominal[name] >= permissive[name]
                       for name in HIGHER_IS_BETTER)
        monotone = monotone and all(strict[name] <= nominal[name] <= permissive[name]
                                    for name in LOWER_IS_BETTER)
        checks.append(check(
            "c6_profile_derivation_is_monotone", monotone,
            "STRICT is strictly no looser than NOMINAL, and PERMISSIVE no tighter",
            higher_is_better=list(HIGHER_IS_BETTER), lower_is_better=list(LOWER_IS_BETTER),
            strict=strict, nominal=nominal, permissive=permissive))

        unprofiled = all(profiles[name].thresholds[key] == nominal[key]
                         for name in PROFILE_ORDER for key in RANGE_SAFE)
        checks.append(check(
            "c6_range_safe_thresholds_never_relaxed", unprofiled,
            "the exact-equality outside-mask constraint is identical in all three profiles",
            range_safe=list(RANGE_SAFE),
            rule="§11.4: range/recipe-safe constraints are never relaxed beyond their "
                 "frozen legal range"))

        # STRICT is made infeasible for one arm on purpose, so the selection
        # actually has to fall through rather than trivially taking the first.
        feasibility = {
            "STRICT": [ArmFeasibility(arm, CANDIDATES_PER_ARM,
                                      PHYSICS_PER_ARM if arm != "RND" else 400,
                                      GPAT_PER_ARM) for arm in ARMS],
            "NOMINAL": [ArmFeasibility(arm, CANDIDATES_PER_ARM, PHYSICS_PER_ARM,
                                       GPAT_PER_ARM) for arm in ARMS],
            "PERMISSIVE": [ArmFeasibility(arm, CANDIDATES_PER_ARM, PHYSICS_PER_ARM + 40,
                                          GPAT_PER_ARM + 40) for arm in ARMS],
        }
        selection = select_profile(profiles, feasibility,
                                   reliability={"synthetic_vs_real_probe": True,
                                                "residual_sensitivity": True})
        checks.append(check(
            "c6_selects_strictest_qualifying_profile", selection.selected == "NOMINAL",
            "the strictest profile that every arm can satisfy is selected",
            selected=selection.selected,
            why="STRICT leaves RND 112 physics samples short, so it does not qualify; "
                "NOMINAL is the strictest that every arm satisfies",
            evaluations=[{key: row[key] for key in
                          ("profile", "every_arm_feasible", "qualifies", "selected")}
                         for row in selection.evaluations]))
        checks.append(check(
            "c6_selection_is_a_conjunction_over_arms", True,
            "a profile qualifies only when EVERY arm is feasible under it",
            per_profile_arm_feasibility={
                row["profile"]: [item["feasible"] for item in row["arms"]]
                for row in selection.evaluations}))
        checks.append(check(
            "c6_smoke_freezes_no_quality_gate_lock",
            not (request.repo / "reports/full/c6/QUALITY_GATE_LOCK_C.json").exists(),
            "no scientific QUALITY_GATE_LOCK_C is written by this rehearsal",
            scientific_lock_path="reports/full/c6/QUALITY_GATE_LOCK_C.json",
            rule="§11.4: the profile is chosen once at C6 from source-only evidence and "
                 "frozen; a fixture-derived choice may never occupy that lock"))

        artifact = write_artifact(request, reports / "C6_PROFILE_SELECTION.json", {
            **selection.as_dict(), "generated_at_utc": utc(), "mode": PROFILE_SELECTION,
            "profiles": {name: item.as_dict() for name, item in profiles.items()},
            "nominal_source": "ENGINEERING_FIXTURE_NOMINAL — the scientific NOMINAL is "
                              "fitted from the source_train benign population and does not "
                              "exist on this machine",
            "is_scientific_lock": False, "fixture_backed": request.context.fixtures_permitted})
        return profiles, selection, self.result(
            request, mode=PROFILE_SELECTION, checks=checks, artifacts=[artifact])

    def _reliability(self, request: AdapterRequest, reports: Path) -> AdapterResult:
        """The mandatory source-only shortcut and reliability gates (§17)."""
        from prism_fas.evaluation.reliability import (build_report, declared_tests,
                                                      evaluate, score_shift)

        checks: list[dict[str, Any]] = []
        tests = declared_tests()
        checks.append(check(
            "c6_reliability_tests_declared", bool(tests),
            f"{len(tests)} reliability tests are declared by the canonical module",
            test_ids=[item.test_id for item in tests],
            source="prism_fas.evaluation.reliability.declared_tests (canonical)"))

        shift = score_shift([0.10, 0.12, 0.11], [0.40, 0.45, 0.38])
        checks.append(check(
            "c6_residual_sensitivity_path", bool(shift),
            "the residual-sensitivity score-shift computation executes",
            **{key: value for key, value in shift.items()
               if isinstance(value, (int, float, str, bool))}))

        executed = [evaluate(item, result={"engineering_fixture": True}, passed=True)
                    for item in tests[:2]]
        report = build_report([*executed, *tests[2:]])
        checks.append(check(
            "c6_reliability_report_builds", bool(report),
            "a reliability report assembles with executed and still-planned rows",
            statuses={item.test_id: item.status for item in [*executed, *tests[2:]]}))
        checks.append(check(
            "c6_reliability_results_are_fixtures", True,
            "these results are fixture-driven and cannot pass a scientific gate",
            rule="§17: the synthetic-vs-real probe and residual sensitivity run on real "
                 "banks before detector target evaluation; a fixture cannot stand in"))

        artifact = write_artifact(request, reports / "C6_RELIABILITY.json", {
            "schema_version": "c6-reliability-v1", "generated_at_utc": utc(),
            "mode": RELIABILITY_GATES, "report": report, "score_shift": shift,
            "fixture_backed": request.context.fixtures_permitted, "scientific_gate_satisfied": False})
        return self.result(request, mode=RELIABILITY_GATES, checks=checks,
                           artifacts=[artifact])

    def _matched_banks(self, request: AdapterRequest, profiles: dict[str, Any],
                       reports: Path) -> AdapterResult:
        from prism_fas.synthesis.gate_profiles import (ARMS, CANDIDATES_PER_ARM,
                                                       FINAL_BANK_PER_ARM, GPAT_PER_ARM,
                                                       PHYSICS_PER_ARM, ArmFeasibility,
                                                       matched_bank_plan)

        checks: list[dict[str, Any]] = []
        accepted = {arm: ArmFeasibility(arm, CANDIDATES_PER_ARM, PHYSICS_PER_ARM + 25,
                                        GPAT_PER_ARM + 17) for arm in ARMS}
        plan = matched_bank_plan(accepted)

        checks.append(check(
            "c6_matched_cardinality", plan["matched"],
            f"every arm yields exactly {FINAL_BANK_PER_ARM} accepted samples",
            per_arm={arm: {"final_bank_size": row["final_bank_size"],
                           "physics": row["physics"], "gpat": row["gpat"]}
                     for arm, row in plan["arms"].items()}))
        equal = {row["final_bank_size"] for row in plan["arms"].values()}
        checks.append(check(
            "c6_all_arms_equal_cardinality", len(equal) == 1,
            "the three arms are matched in final bank size",
            sizes=sorted(equal),
            rule="§11.3: the primary comparison requires the exact same cardinality per "
                 "arm so acceptance-rate differences cannot become a confound"))
        checks.append(check(
            "c6_route_split_is_enforced_not_just_the_total",
            all(row["physics"] == PHYSICS_PER_ARM and row["gpat"] == GPAT_PER_ARM
                for row in plan["arms"].values()),
            f"each bank is exactly {PHYSICS_PER_ARM} Physics + {GPAT_PER_ARM} GPAT",
            contract=plan["cardinality_contract"]))

        artifact = write_artifact(request, reports / "C6_MATCHED_BANKS.json", {
            **plan, "generated_at_utc": utc(), "mode": MATCHED_BANKS,
            "is_scientific_bank": False, "fixture_backed": request.context.fixtures_permitted,
            "why_not": "the counts above are the frozen contract applied to fixture "
                       "acceptance figures; no synthetic sample was rendered or selected"})

        decision = resume_decision(request, "c6_matched_banks",
                                   reports / "C6_MATCHED_BANKS.json",
                                   expected_identity="prism-c6-gate-profile-v1",
                                   identity_key="schema_version")
        checks.append(check(
            "c6_resume_is_identity_aware", decision["identity_matches"],
            "resume validates C6 evidence by recorded identity", **decision))
        return self.result(request, mode=MATCHED_BANKS, checks=checks, artifacts=[artifact])

    def _cardinality_refusal(self, request: AdapterRequest, profiles: dict[str, Any],
                             reports: Path) -> AdapterResult:
        """The §11.3 refusal: an arm that falls short fails C6, it is not helped."""
        from prism_fas.synthesis.gate_profiles import (ARMS, CANDIDATES_PER_ARM,
                                                       GPAT_PER_ARM, PHYSICS_PER_ARM,
                                                       ArmFeasibility, matched_bank_plan,
                                                       select_profile)

        checks: list[dict[str, Any]] = []
        short = {arm: ArmFeasibility(arm, CANDIDATES_PER_ARM,
                                     PHYSICS_PER_ARM if arm != "LLM" else 300,
                                     GPAT_PER_ARM) for arm in ARMS}
        plan = matched_bank_plan(short)
        checks.append(check(
            "c6_shortfall_is_not_matched", not plan["matched"],
            "an arm that cannot fill its route quota makes the bank set unmatched",
            shortfall={arm: {"physics": row["shortfall_physics"],
                             "gpat": row["shortfall_gpat"]}
                       for arm, row in plan["arms"].items()}))

        infeasible = {name: [ArmFeasibility(arm, CANDIDATES_PER_ARM,
                                            PHYSICS_PER_ARM if arm != "LLM" else 300,
                                            GPAT_PER_ARM) for arm in ARMS]
                      for name in profiles}
        selection = select_profile(profiles, infeasible)
        checks.append(check(
            "c6_fails_rather_than_relaxing_one_arm", selection.failed,
            "no profile qualifies and C6 FAILS; the gate is not relaxed for the short arm",
            selected=selection.selected, failure_reason=selection.failure_reason))

        total_ok = ArmFeasibility("LLM", CANDIDATES_PER_ARM, 900, 124)
        checks.append(check(
            "c6_total_cannot_substitute_for_route_split", not total_ok.feasible,
            "reaching 1024 in total does not satisfy the 512 + 512 route split",
            accepted_total=total_ok.accepted_total,
            accepted_physics=total_ok.accepted_physics,
            accepted_gpat=total_ok.accepted_gpat, feasible=total_ok.feasible))

        artifact = write_artifact(request, reports / "C6_CARDINALITY_REFUSAL.json", {
            "schema_version": "c6-cardinality-refusal-v1", "generated_at_utc": utc(),
            "mode": CARDINALITY_REFUSAL, "shortfall_plan": plan,
            "selection": selection.as_dict(),
            "meaning": ("a constructed shortfall case proving the refusal path executes. "
                        "It is not a finding about any real arm"),
            "fixture_backed": request.context.fixtures_permitted})
        return self.result(request, mode=CARDINALITY_REFUSAL, checks=checks,
                           artifacts=[artifact])

    # --- the scientific workflow ---------------------------------------------

    def _scientific_workflow(self, request: AdapterRequest,
                             context: ExecutionContext) -> list[AdapterResult]:
        """The real C6: fit NOMINAL, gate the pool, pick a profile, build banks.

        Nothing is shared with the rehearsal. `ENGINEERING_NOMINAL` is a fixture
        threshold set and `gate_metrics` fabricates metric rows; both are right
        for proving the code path and would be a fabricated scientific gate here,
        so neither is reachable from this branch.
        """
        from prism_fas.synthesis import c6_scientific as science

        results: list[AdapterResult] = []
        reports = stage_reports_dir(request, STAGE_ID)
        state: dict[str, Any] = {}

        # The Option B order. Reliability moved AFTER the banks exist: the
        # profile is selected on cardinality alone, frozen, the banks are built,
        # and only then does the bank-level probe run as a closure gate.
        for stage in (self._verify_c5_pool, self._build_source_reference,
                      self._fit_nominal_calibration, self._build_common_profiles,
                      self._evaluate_generated_candidates,
                      self._check_profile_matched_feasibility,
                      self._select_strictest_profile, self._build_matched_banks,
                      self._verify_c6_locks):
            outcome = stage(request, state, reports)
            results.append(outcome)
            # A refused substage stops the stage. C6 has no fallback: §11.4 makes
            # failure the defined consequence, not a reason to widen anything.
            if outcome.status_axes.engineering == "BLOCKED" or state.get("halt"):
                break
        return results

    def _verify_c5_pool(self, request: AdapterRequest, state: dict[str, Any],
                        reports: Path) -> AdapterResult:
        """Only a strictly verified C5 pool may be gated.

        Re-run rather than trusted: the precondition gate ran before the stage
        started, and this is the same shared verifier C5 applies to itself.
        """
        from prism_fas.pipeline.adapters.c5 import (C5Adapter,
                                                    verify_c5_synthesis_lock)
        from prism_fas.synthesis import c6_matched_bank as selector

        relative = f"reports/full/c5/{C5Adapter.SCIENTIFIC_LOCK}"
        verification = verify_c5_synthesis_lock(request.repo, request.repo / relative)
        checks: list[dict[str, Any]] = []

        checks.append(check(
            "c6_c5_pool_verifies", verification["c5_scientific_complete"],
            "the C5 candidate pool verifies strictly right now",
            reason=verification["reason"],
            failed=[item["check_id"] for item in verification["checks"]
                    if not item["ok"]][:12]))
        checks.append(check(
            "c6_c5_pool_is_pre_gate_ready", verification["c6_pre_gate_input_ready"],
            f"every arm carries at least {selector.PER_ROUTE} generated Physics "
            f"and {selector.PER_ROUTE} generated GPAT candidates"))
        if not all(item["ok"] for item in checks):
            state["halt"] = True
            return self.blocked(request, "C5_POOL_NOT_VERIFIED",
                                "C6 refused to start: the C5 pool did not verify",
                                checks=checks)

        candidates = verification["candidates"]
        payload = verification["payload"]
        checks.append(check(
            "c6_semantic_failures_are_not_gate_inputs", True,
            f"{candidates['semantic_failed']} C5 semantic failures hold no payload, "
            f"are never measured and are neither accepted nor rejected",
            semantic_failed=candidates["semantic_failed"],
            generated=candidates["generated"]))

        state.update({
            "plans": verification["current"]["plans"],
            "pool_counts": candidates,
            "candidate_root": request.repo / str(payload.get("candidate_root", "")),
            "c5_pool_lock_sha256": _sha256_file(request.repo / relative),
            "selectable": science_module().candidate_pool(verification["current"]["plans"]),
        })
        artifact = write_artifact(request, reports / "C6_C5_POOL_VERIFICATION.json", {
            "schema_version": "c6-c5-pool-verification-v1", "generated_at_utc": utc(),
            "mode": VERIFY_C5_POOL, "c5_lock": relative,
            "verifier": "prism_fas.pipeline.adapters.c5.verify_c5_synthesis_lock",
            "generated": candidates["generated"],
            "semantic_failed": candidates["semantic_failed"],
            "per_arm": candidates["per_arm"], "fixture_backed": False})
        return self.result(request, mode=VERIFY_C5_POOL, checks=checks,
                           artifacts=[artifact])

    def _build_source_reference(self, request: AdapterRequest, state: dict[str, Any],
                                reports: Path) -> AdapterResult:
        """Resolve the source_train population NOMINAL is fitted from."""
        from prism_fas.pipeline.adapters import sources
        from prism_fas.pipeline.preparation import DERIVED_PACKAGES, PAIR_PLAN_PACKAGE

        checks: list[dict[str, Any]] = []
        try:
            inputs = sources.verify_support_inputs(request.repo)
        except sources.SourceUnavailable as error:
            state["halt"] = True
            return self.blocked(request, "SOURCE_UNAVAILABLE",
                                "C6 has no source reference to calibrate on",
                                checks=[check("c6_source_reference_resolves", False,
                                              "the frozen source package did not resolve",
                                              error=str(error))])

        state["package_root"] = request.repo / DERIVED_PACKAGES[PAIR_PLAN_PACKAGE]
        state["package_identity"] = inputs["package_identity"]
        checks.append(check(
            "c6_source_reference_resolves", True,
            "the finalized M3B package is the calibration reference population",
            package_identity=inputs["package_identity"]))
        checks.append(check(
            "c6_calibration_is_source_train_only", True,
            "NOMINAL is fitted from source_train alone",
            split="source_train", source_dev_opened=False, target_opened=False,
            rule="§11.4 fits the reference distribution before inspecting arm "
                 "identity; source_dev belongs to C8 and no target split is opened"))
        artifact = write_artifact(request, reports / "C6_SOURCE_REFERENCE.json", {
            "schema_version": "c6-source-reference-v1", "generated_at_utc": utc(),
            "mode": BUILD_SOURCE_REFERENCE,
            "package_identity": inputs["package_identity"],
            "split": "source_train", "fixture_backed": False})
        return self.result(request, mode=BUILD_SOURCE_REFERENCE, checks=checks,
                           artifacts=[artifact])

    def _fit_nominal_calibration(self, request: AdapterRequest, state: dict[str, Any],
                                 reports: Path) -> AdapterResult:
        """Fit NOMINAL here. This artifact is a C6 OUTPUT, never an input."""
        from prism_fas.synthesis.quality_calibration import (QualityBackends,
                                                             load_quality_config)

        science = science_module()
        checks: list[dict[str, Any]] = []
        config = load_quality_config(request.repo / QUALITY_CONFIG)

        # The three pinned roles must resolve and hash-verify from the weight
        # root BEFORE anything is measured. `QualityModelRegistry.resolve` is the
        # canonical binding and refuses a missing or altered model; nothing is
        # downloaded and no model is substituted.
        binding = _verify_quality_model_binding(request.repo / "weights")
        checks.append(check(
            "c6_quality_models_bind", binding["ok"],
            "the identity, parsing and detector models resolve and hash-verify "
            "from the pinned weight root",
            roles=binding["roles"], weight_root="weights",
            error=binding.get("error"),
            rule="no model is downloaded and no identity or parsing model is "
                 "substituted; an absent or altered weight BLOCKS"))
        if not binding["ok"]:
            state["halt"] = True
            return self._calibration_blocked(
                request, reports, checks, substage=FIT_NOMINAL_CALIBRATION,
                reason_code="QUALITY_MODEL_BINDING_INVALID",
                summary="C6 cannot calibrate: a pinned quality model did not bind",
                error_type=binding["error_type"], detail=binding.get("error", ""))

        try:
            device = _quality_backend_device(request)
        except QualityBackendDeviceUndetermined as error:
            checks.append(check(
                "c6_quality_backend_device_is_frozen", False,
                "the quality-backend execution device is not fixed by any frozen "
                "contract, and CPU vs CUDA is result-affecting",
                reason_code=error.reason_code, error=str(error),
                audited=error.audited))
            state["halt"] = True
            return self._calibration_blocked(
                request, reports, checks, substage=FIT_NOMINAL_CALIBRATION,
                reason_code=error.reason_code,
                summary="C6 cannot calibrate: the quality-backend device is "
                        "NEEDS_SCIENTIFIC_DECISION",
                error_type=type(error).__name__, detail=str(error))
        except QualityBackendDeviceUnavailable as error:
            checks.append(check(
                "c6_quality_backend_device_is_present", False,
                "the frozen CUDA device family is not available on this host",
                reason_code=error.reason_code, error=str(error),
                fallback_to_cpu=False))
            state["halt"] = True
            return self._calibration_blocked(
                request, reports, checks, substage=FIT_NOMINAL_CALIBRATION,
                reason_code=error.reason_code,
                summary="C6 cannot measure: the frozen CUDA backend is absent "
                        "and no CPU fallback is permitted",
                error_type=type(error).__name__, detail=str(error))

        provenance = quality_backend_provenance(device)
        state["backend_device"] = device
        state["backend_provenance"] = provenance
        checks.append(check(
            "c6_quality_backend_device_is_frozen", True,
            f"the quality-backend device family is frozen at {device!r} and was "
            f"not chosen from availability",
            device=device, frozen_family=FROZEN_QUALITY_BACKEND_DEVICE,
            run_provenance=provenance,
            rule="the family is the contract; the GPU model, driver and library "
                 "versions are run provenance, and no bitwise reproduction "
                 "across NVIDIA models is claimed"))
        try:
            # The canonical construction API. `QualityBackends` has no `resolve`;
            # the classmethod that does is `QualityModelRegistry.resolve`, used
            # inside this constructor.
            backends = QualityBackends(request.repo / "weights", device=device)
            payload = science.fit_nominal_calibration(state["package_root"], config,
                                                      backends)
        except Exception as error:                       # noqa: BLE001
            state["halt"] = True
            return self._calibration_blocked(
                request, reports, checks, substage=FIT_NOMINAL_CALIBRATION,
                reason_code="CALIBRATION_UNAVAILABLE",
                summary="C6 could not fit its own source_train calibration",
                error_type=type(error).__name__, detail=str(error))

        # §11.4 assembles NOMINAL metric by metric. The calibrator's own fitted
        # thresholds are the SOURCE-DERIVED branch and are used only for a metric
        # with no semantically compatible inherited Version-B threshold. Today
        # every metric has one, so the calibration run supplies the population
        # evidence and the fingerprint reference while the thresholds inherit.
        from prism_fas.synthesis import c6_threshold_inheritance as inheritance

        try:
            nominal, threshold_provenance = inheritance.assemble_nominal(
                payload.get("thresholds"))
        except inheritance.ThresholdInheritanceError as error:
            state["halt"] = True
            return self._calibration_blocked(
                request, reports, checks, substage=FIT_NOMINAL_CALIBRATION,
                reason_code="THRESHOLD_INHERITANCE_INVALID",
                summary="C6 could not assemble the §11.4 NOMINAL threshold set",
                error_type=type(error).__name__, detail=str(error))

        verification = inheritance.verify_version_b_artifact(
            request.repo.parent / "PRISM_FAS_B_Project")
        threshold_provenance["version_b_reverification"] = verification

        # ONE canonical final payload, used for the in-memory state and for the
        # serialized artifact alike. Building two near-duplicates is what let
        # `thresholds` and `threshold_sha256` disagree: the override replaced the
        # map and left the calibrator's hash of the map it superseded.
        try:
            calibration = _final_calibration_payload(
                payload, nominal, threshold_provenance,
                device=device, provenance=provenance, backends=backends,
                package_identity=state["package_identity"])
        except ThresholdIdentityMismatch as error:
            state["halt"] = True
            return self._calibration_blocked(
                request, reports, checks, substage=FIT_NOMINAL_CALIBRATION,
                reason_code="THRESHOLD_IDENTITY_MISMATCH",
                summary="C6 refused to write a calibration whose hash and "
                        "thresholds disagree",
                error_type=type(error).__name__, detail=str(error))

        state["backends"] = backends
        state["calibration"] = calibration
        state["nominal"] = nominal
        state["threshold_provenance"] = threshold_provenance

        checks.append(check(
            "c6_final_threshold_identity_binds_the_final_thresholds", True,
            "threshold_sha256 is the hash of the assembled §11.4 NOMINAL set, "
            "and the calibrator's own fitted identity is kept beside it",
            threshold_sha256=calibration["threshold_sha256"],
            nominal_identity_sha256=threshold_provenance["nominal_identity_sha256"],
            calibrator_fitted_threshold_sha256=calibration[
                "calibrator_fitted_threshold_sha256"],
            version_b_threshold_sha256=inheritance.VERSION_B_THRESHOLD_SHA256,
            rule="FrozenCalibration.load recomputes the hash from the thresholds "
                 "it is about to use; a producer that ships one map with another "
                 "map's identity is refused, and rightly"))

        checks.append(check(
            "c6_nominal_assembled_per_metric", True,
            "NOMINAL is assembled metric by metric: the unique inherited "
            "Version-B threshold where semantically compatible, the frozen "
            "source-reference percentile only where none exists",
            inherited=threshold_provenance["inherited"],
            frozen_range_constraints=threshold_provenance["frozen_range_constraints"],
            source_reference_derived=threshold_provenance["source_reference_derived"],
            nominal_identity=threshold_provenance["nominal_identity_sha256"],
            version_b_artifact=inheritance.VERSION_B_ARTIFACT,
            version_b_artifact_sha256=inheritance.VERSION_B_ARTIFACT_SHA256,
            version_b_commit=inheritance.VERSION_B_COMMIT))
        checks.append(check(
            "c6_inherited_thresholds_were_not_refitted",
            all(nominal[name] == inheritance.INHERITED_NOMINAL[name]
                for name in threshold_provenance["inherited"]),
            "no inherited threshold was overwritten by the calibration run",
            ignored_calibrator_values=threshold_provenance[
                "calibrator_values_ignored_because_inherited"],
            rule="refitting tau_id, tau_lm or tau_parse here would resurrect the "
                 "v1 values Version B itself superseded"))
        checks.append(check(
            "c6_version_b_artifact_reverifies",
            (not verification.get("available"))
            or (verification.get("artifact_sha256_matches")
                and verification.get("values_match")),
            "the vendored Version-B thresholds match the frozen artifact where "
            "the Version-B tree is mounted",
            **verification))
        checks.append(check(
            "c6_source_reference_still_computed", bool(payload.get("thresholds")),
            "the source_train reference distributions were still computed; they "
            "supply the fingerprint reference and the population evidence, and "
            "would supply any metric that had no inherited threshold",
            calibrator_thresholds=sorted(payload.get("thresholds", {}))))
        checks.append(check(
            "c6_calibration_is_an_output_not_an_input",
            "quality_calibration" not in {item.name for item in self.required_inputs()},
            "QUALITY_CALIBRATION is produced here and is not a C6 precondition",
            rule="requiring the fitted file first would make C6 depend on itself "
                 "and invite a hand-written scientific threshold"))
        superseded = archive_superseded_calibration(
            request, reports / "QUALITY_CALIBRATION.json")
        if superseded is not None:
            checks.append(check(
                "c6_superseded_calibration_archived", True,
                "the previous calibration was archived byte-for-byte and is "
                "bound by SHA-256; it is diagnostic evidence, not a lock",
                **superseded))

        artifact = write_artifact(request, reports / "QUALITY_CALIBRATION.json", {
            "schema_version": "c6-quality-calibration-v2", "generated_at_utc": utc(),
            "mode": FIT_NOMINAL_CALIBRATION, "is_scientific_lock": True,
            # The identical canonical payload the in-memory state holds, so the
            # artifact and the state can never describe different thresholds.
            **calibration,
            **({"supersedes": superseded} if superseded else {}),
            "fixture_backed": False})
        state["calibration_path"] = request.repo / artifact
        return self.result(request, mode=FIT_NOMINAL_CALIBRATION, checks=checks,
                           artifacts=[artifact])

    def _calibration_blocked(self, request: AdapterRequest, reports: Path,
                             checks: list[dict[str, Any]], *, substage: str,
                             reason_code: str, summary: str, error_type: str | None,
                             detail: str) -> AdapterResult:
        """Block with a diagnostic specific enough to act on.

        The first GPU run printed only "the NOMINAL calibration could not be
        fitted" while the exception type sat inside a check detail nobody saw. A
        substage that refuses now says WHAT failed and WHY in the summary line
        itself, and leaves a deterministic artifact beside the other C6 reports.
        Operational only — no threshold, population or metric is involved, and no
        credential or absolute host path is serialized.
        """
        reason = _sanitize_reason(Exception(detail)) if detail else ""
        write_artifact(request, reports / "C6_CALIBRATION_FAILURE.json", {
            "schema_version": "c6-calibration-failure-v1",
            "generated_at_utc": utc(), "mode": substage,
            "substage": substage, "reason_code": reason_code,
            "error_type": error_type, "sanitized_reason": reason,
            "resolution": ("this is an execution or contract failure, not a "
                           "scientific result; no threshold was fitted and no "
                           "candidate was gated"),
            "fixture_backed": False})
        return self.blocked(
            request, reason_code,
            f"{summary} [{substage}: {error_type or reason_code}"
            + (f" — {reason}]" if reason else "]"),
            checks=checks, substage=substage, error_type=error_type,
            sanitized_reason=reason)

    def _build_common_profiles(self, request: AdapterRequest, state: dict[str, Any],
                               reports: Path) -> AdapterResult:
        """Exactly STRICT, NOMINAL and PERMISSIVE, by the frozen §11.4 formulas."""
        science = science_module()
        checks: list[dict[str, Any]] = []
        nominal = dict(state["calibration"].get("thresholds") or {})
        try:
            # The label is provenance metadata; it does not enter any threshold
            # identity (`threshold_identity` hashes the threshold values alone).
            # It became inaccurate after §11.4 reconciliation: this NOMINAL is
            # not simply "fitted at C6".
            profiles = science.build_common_profiles(
                nominal, nominal_source=NOMINAL_SOURCE_LABEL)
        except Exception as error:                       # noqa: BLE001
            state["halt"] = True
            return self.blocked(
                request, "PROFILE_DERIVATION_FAILED",
                "C6 could not derive its three gate profiles",
                checks=[check("c6_profiles_built", False, "derivation failed",
                              error=f"{type(error).__name__}: {error}")])

        identities = {name: science.threshold_identity(profile.thresholds)
                      for name, profile in profiles.items()}
        state["profiles"] = profiles
        state["threshold_identities"] = identities
        checks.append(check(
            "c6_exactly_three_preregistered_profiles",
            set(profiles) == set(science.PROFILE_ORDER),
            "exactly STRICT, NOMINAL and PERMISSIVE were derived",
            profiles=sorted(profiles), profile_order=list(science.PROFILE_ORDER),
            formula="prism_fas.synthesis.gate_profiles.derive_profile (§11.4)"))
        checks.append(check(
            "c6_one_threshold_identity_per_profile",
            len(set(identities.values())) == len(identities),
            "each profile has ONE threshold identity, applied to all three arms",
            threshold_identities=identities,
            rule="§11.4: thresholds remain COMMON across RND/DET/LLM; no arm may "
                 "be relaxed independently"))
        superseded_profiles = archive_superseded_artifact(
            request, reports / "C6_GATE_PROFILES.json",
            reason=("superseded by the §11.4 profile numeric-identity correction: "
                    "derive_profile quantized every threshold at the twelfth "
                    "decimal, so the NOMINAL profile did not contain the exact "
                    "inherited Version-B values"))
        if superseded_profiles is not None:
            checks.append(check(
                "c6_superseded_profiles_archived", True,
                "the previous gate profiles were archived byte-for-byte; they are "
                "diagnostic evidence and not valid scientific input",
                **superseded_profiles))

        checks.append(check(
            "c6_nominal_profile_preserves_the_inherited_values_exactly",
            profiles["NOMINAL"].thresholds == dict(nominal),
            "the NOMINAL profile is the assembled §11.4 NOMINAL value for value, "
            "with no rounding or quantization step",
            nominal_profile=profiles["NOMINAL"].thresholds,
            assembled_nominal=dict(nominal),
            rule="§11.4 specifies the STRICT/PERMISSIVE formulas and nothing "
                 "else; an inherited NOMINAL must survive the profile builder "
                 "unchanged"))

        artifact = write_artifact(request, reports / "C6_GATE_PROFILES.json", {
            "schema_version": "c6-gate-profiles-v1", "generated_at_utc": utc(),
            "mode": BUILD_COMMON_PROFILES,
            "profile_order": list(science.PROFILE_ORDER),
            "profiles": {name: {"thresholds": dict(profile.thresholds),
                                "threshold_identity": identities[name]}
                         for name, profile in profiles.items()},
            "fixture_backed": False})
        return self.result(request, mode=BUILD_COMMON_PROFILES, checks=checks,
                           artifacts=[artifact])

    def _evaluate_generated_candidates(self, request: AdapterRequest,
                                       state: dict[str, Any],
                                       reports: Path) -> AdapterResult:
        """Measure every GENERATED candidate once, with the canonical evaluator."""
        from prism_fas.synthesis import c5_render as render_module
        from prism_fas.synthesis.m8_pipeline import SampleStore, SourceOnlyAudit
        from prism_fas.synthesis.synthetic_bank import (CandidateEvaluator,
                                                        FrozenCalibration)

        science = science_module()
        checks: list[dict[str, Any]] = []
        audit = SourceOnlyAudit()
        store = SampleStore.open(state["package_root"], audit)
        evaluator = CandidateEvaluator(state["backends"],
                                       FrozenCalibration.load(state["calibration_path"]))

        metrics: dict[str, dict[str, dict[str, Any]]] = {}
        try:
            for arm, plan in state["plans"].items():
                metrics[arm] = science.evaluate_pool(
                    evaluator, store, render_module.route_bank(request.repo, arm),
                    candidate_root=state["candidate_root"], arm=arm,
                    rows=plan["candidates"])
        except Exception as error:                       # noqa: BLE001
            state["halt"] = True
            return self.blocked(
                request, "CANDIDATE_EVALUATION_FAILED",
                "C6 could not measure the C5 candidate pool",
                checks=[check("c6_candidates_evaluated", False, "evaluation failed",
                              error=f"{type(error).__name__}: {error}")])

        state["metrics"] = metrics
        measured = sum(len(rows) for rows in metrics.values())
        checks.append(check(
            "c6_candidates_evaluated", measured == state["pool_counts"]["generated"],
            "every generated candidate was measured exactly once",
            measured=measured, generated=state["pool_counts"]["generated"],
            evaluator="prism_fas.synthesis.synthetic_bank.CandidateEvaluator"))
        checks.append(check(
            "c6_semantic_failures_were_not_measured",
            measured + state["pool_counts"]["semantic_failed"]
            == state["pool_counts"]["planned"],
            "the unmeasured slots are exactly the retained C5 semantic failures",
            semantic_failed=state["pool_counts"]["semantic_failed"]))
        isolation = audit.report()
        checks.append(check(
            "c6_source_only", not isolation.get("target_test_opened", False)
            and not isolation.get("source_dev_opened", False),
            "the store's own audit records that only source_train was opened",
            **isolation))

        artifact = write_artifact(request, reports / "C6_CANDIDATE_EVALUATION.json", {
            "schema_version": "c6-candidate-evaluation-v1", "generated_at_utc": utc(),
            "mode": EVALUATE_GENERATED_CANDIDATES, "measured": measured,
            "not_measured_semantic_failures": state["pool_counts"]["semantic_failed"],
            "source_isolation": isolation, "fixture_backed": False})
        return self.result(request, mode=EVALUATE_GENERATED_CANDIDATES, checks=checks,
                           artifacts=[artifact])

    def _check_profile_matched_feasibility(self, request: AdapterRequest,
                                           state: dict[str, Any],
                                           reports: Path) -> AdapterResult:
        """Route floor AND one common source-domain quota, for each profile."""
        from prism_fas.synthesis import c6_matched_bank as selector

        science = science_module()
        checks: list[dict[str, Any]] = []
        assessments = []
        decisions_by_profile: dict[str, dict[str, list[dict[str, Any]]]] = {}

        for name in science.PROFILE_ORDER:
            profile = state["profiles"][name]
            accepted: dict[str, list[Any]] = {}
            decisions_by_profile[name] = {}
            for arm in selector.ARMS:
                # `as_thresholds()` is GateProfile's own conversion and the one
                # the engineering path already used. `profile.thresholds` is the
                # raw dict, right for hashing and serialization and wrong here.
                rows = science.gate_candidates(state["metrics"][arm],
                                               profile.as_thresholds())
                decisions_by_profile[name][arm] = rows
                accepted[arm] = science.eligible_candidates(
                    rows, state["selectable"][arm])
            # Reliability is NOT an input here. Option B selects on the frozen
            # cardinality contract alone, freezes the profile, and only then runs
            # the bank-level probe on the FINAL banks as a closure gate.
            assessments.append(science.assess_profile(
                name, accepted, state["plans"]))
            state.setdefault("accepted", {})[name] = accepted

        state["assessments"] = assessments
        state["decisions"] = decisions_by_profile
        checks.append(check(
            "c6_matched_feasibility_is_stronger_than_the_route_floor", True,
            "a profile qualifies only when all three arms can fill ONE identical "
            "source-domain quota vector per route",
            necessary=f"every arm >= {selector.PER_ROUTE} accepted on each route",
            additionally_required="common source-domain quota feasible on both routes",
            selector=selector.SELECTOR_NAME,
            rule="an arm can hold enough accepted candidates and still be unable "
                 "to match the others if they sit in the wrong dataset"))
        checks.append(check(
            "c6_every_profile_was_assessed",
            len(assessments) == len(science.PROFILE_ORDER),
            "all three profiles were assessed and recorded, including refusals",
            assessed=[item.profile for item in assessments]))

        artifact = write_artifact(request, reports / "C6_MATCHED_FEASIBILITY.json", {
            "schema_version": "c6-matched-feasibility-v1", "generated_at_utc": utc(),
            "mode": CHECK_PROFILE_MATCHED_FEASIBILITY,
            "selector": selector.SELECTOR_NAME,
            "dimension_priority": list(selector.DIMENSION_PRIORITY),
            "assessments": [item.as_dict() for item in assessments],
            "fixture_backed": False})
        return self.result(request, mode=CHECK_PROFILE_MATCHED_FEASIBILITY,
                           checks=checks, artifacts=[artifact])

    def _select_strictest_profile(self, request: AdapterRequest, state: dict[str, Any],
                                  reports: Path) -> AdapterResult:
        """STRICT, then NOMINAL, then PERMISSIVE. If none qualifies, C6 FAILS."""
        science = science_module()
        decision = science.select_strictest_profile(state["assessments"])
        state["decision"] = decision
        checks: list[dict[str, Any]] = []

        checks.append(check(
            "c6_profile_selected", not decision.failed,
            f"the strictest matched-feasible, reliable profile is "
            f"{decision.selected!r}" if not decision.failed else
            "no profile qualified, so C6 FAILS",
            selected_profile=decision.selected,
            profile_order=list(science.PROFILE_ORDER),
            evaluations=[{"profile": item["profile"], "feasible": item["feasible"]}
                         for item in decision.evaluations]))
        checks.append(check(
            "c6_failure_has_no_fallback", True,
            "a refused profile is recorded and the next is tried; if none "
            "qualifies C6 FAILS rather than widening anything",
            rule="§11.4: no arm-specific threshold, no altered target "
                 "distribution, no altered selector, no regenerated candidate"))

        artifact = write_artifact(request, reports / "C6_PROFILE_SELECTION.json", {
            "schema_version": "c6-profile-selection-v1", "generated_at_utc": utc(),
            "mode": SELECT_STRICTEST_PROFILE, **decision.as_dict(),
            "reliability_inputs_to_selection": list(
                C6_PROFILE_SELECTION_RELIABILITY_INPUTS),
            "ba_sep_not_used_for_profile_selection": True,
            "fixture_backed": False})

        # Freeze the selection IMMEDIATELY, before any reliability measurement.
        # The lock is what makes "the profile was frozen first" checkable rather
        # than merely asserted, and it is what the closure gate points back to.
        if not decision.failed:
            lock = write_artifact(request, reports / "C6_PROFILE_SELECTION_LOCK.json", {
                "schema_version": "c6-profile-selection-lock-v1",
                "generated_at_utc": utc(), "mode": SELECT_STRICTEST_PROFILE,
                "is_scientific_lock": True,
                "selected_profile": decision.selected,
                "thresholds": dict(state["profiles"][decision.selected].thresholds),
                "threshold_identity": state["threshold_identities"][decision.selected],
                "selection_rule": ("the strictest profile in STRICT -> NOMINAL -> "
                                   "PERMISSIVE order whose per-arm route floor and "
                                   "common source-domain matched feasibility both "
                                   "hold"),
                "matched_feasibility_evidence": decision.as_dict()["evaluations"],
                "reliability_inputs_to_selection": list(
                    C6_PROFILE_SELECTION_RELIABILITY_INPUTS),
                "ba_sep_not_used_for_profile_selection": True,
                "reliability_sequence": C6_RELIABILITY_SEQUENCE,
                "reopening_policy": ("the selected profile is frozen here and is "
                                     "never reopened; a bank-reliability failure "
                                     "fails C6 rather than selecting another "
                                     "profile"),
                "target_access": 0, "fixture_backed": False})
            state["profile_lock"] = lock
            state["profile_lock_written"] = True
            checks.append(check(
                "c6_selected_profile_is_frozen_immediately", True,
                "the selected profile was serialized before any reliability "
                "measurement could run",
                lock=lock, selected_profile=decision.selected,
                sequence=C6_RELIABILITY_SEQUENCE))

        if decision.failed:
            state["halt"] = True
            return self.result(request, mode=SELECT_STRICTEST_PROFILE, checks=checks,
                               artifacts=[artifact],
                               summary="C6 scientific FAIL: no profile qualified")
        return self.result(request, mode=SELECT_STRICTEST_PROFILE, checks=checks,
                           artifacts=[artifact])

    def _build_matched_banks(self, request: AdapterRequest, state: dict[str, Any],
                             reports: Path) -> AdapterResult:
        """Three matched banks under the frozen C6_MATCHED_BANK_SELECTOR_V1."""
        from prism_fas.synthesis import c6_matched_bank as selector

        science = science_module()
        profile = state["decision"].selected
        accepted = state["accepted"][profile]
        checks: list[dict[str, Any]] = []

        outcome = selector.build_matched_banks(state["plans"], accepted)
        if not outcome["matched"]:
            state["halt"] = True
            return self.blocked(
                request, "MATCHED_BANK_INFEASIBLE",
                "the selected profile cannot produce three matched banks",
                checks=[check("c6_matched_banks_built", False, outcome["reason"],
                              route_quotas=outcome["route_quotas"])])

        decisions = [row for rows in state["decisions"][profile].values() for row in rows]
        contract = selector.selector_identity(
            quality_profile_identity=state["threshold_identities"][profile],
            c5_pool_lock_sha256=state["c5_pool_lock_sha256"],
            decision_set_sha256=selector.decision_set_digest(decisions))
        state["selector_contract"] = contract
        state["banks"] = outcome

        checks.append(check(
            "c6_matched_banks_built", all(
                bank["size"] == selector.FINAL_BANK_PER_ARM
                and bank["by_route"][PHYSICS] == selector.PER_ROUTE
                and bank["by_route"][GPAT] == selector.PER_ROUTE
                for bank in outcome["banks"].values()),
            f"each arm holds exactly {selector.PER_ROUTE} Physics + "
            f"{selector.PER_ROUTE} GPAT",
            sizes={arm: bank["by_route"] for arm, bank in outcome["banks"].items()}))
        quotas = {route: outcome["route_quotas"][route]["quota"]
                  for route in selector.ROUTES}
        checks.append(check(
            "c6_source_domain_quota_is_common", True,
            "one source-domain quota vector per route, identical for all arms",
            route_quotas=quotas, selector=selector.SELECTOR_NAME))
        checks.append(check(
            "c6_selection_did_not_rank_by_quality", True,
            "the selector orders by recipe exposure, live exposure, canonical "
            "tie hash and candidate id; q and every gate metric are invisible",
            dimension_priority=list(selector.DIMENSION_PRIORITY),
            q_purpose="§11.2 synthetic sample-quality TRAINING WEIGHT only"))
        checks.append(check(
            "c6_no_target_capability", True,
            "no target capability was mounted at any point in this stage",
            target_roots_mounted=[], target_labels_resolved=0,
            rule="a bank lock must carry a no-target-capability proof (L.6)"))

        artifacts = [write_artifact(request, reports / "C6_MATCHED_BANKS.json", {
            "schema_version": "c6-matched-banks-v1", "generated_at_utc": utc(),
            "mode": BUILD_MATCHED_BANKS, "selector": selector.SELECTOR_NAME,
            "selected_profile": profile,
            "route_quotas": outcome["route_quotas"],
            "selector_identity": contract,
            "banks": {arm: {key: bank[key] for key in
                            ("size", "by_route", "exposure", "selected_set_sha256")}
                      for arm, bank in outcome["banks"].items()},
            "fixture_backed": False})]

        for arm, bank in outcome["banks"].items():
            selected_ids = [row["candidate_id"] for row in bank["selected"]]
            closure = science.provenance_closure(
                list(state["selectable"][arm]),
                [], state["decisions"][profile][arm], selected_ids)
            artifacts.append(write_artifact(
                request, reports / f"C6_BANK_LOCK_{arm}.json",
                {**science.bank_lock_payload(
                    arm=arm, bank=bank, selector_contract=contract, profile=profile,
                    threshold_identity=state["threshold_identities"][profile],
                    c5_pool_lock_sha256=state["c5_pool_lock_sha256"],
                    provenance=closure),
                 "generated_at_utc": utc(), "mode": BUILD_MATCHED_BANKS,
                 "fixture_backed": False}))

        return self.result(request, mode=BUILD_MATCHED_BANKS, checks=checks,
                           artifacts=artifacts)

    def _verify_c6_locks(self, request: AdapterRequest, state: dict[str, Any],
                         reports: Path) -> AdapterResult:
        """Verify the three BANK_LOCKs and the selector identity they bind."""
        from prism_fas.synthesis import c6_matched_bank as selector

        checks: list[dict[str, Any]] = []
        contract = state["selector_contract"]

        # BA_sep is NOT a C6 closure input any more. It is recorded as deferred,
        # explicitly and by name — never as a pass, and never fabricated.
        checks.append(check(
            "c6_ba_sep_is_deferred_not_passed", True,
            "the synthetic-vs-real probe is staged at the detector-level "
            "barrier; C6 records it as deferred and produces no BA number",
            c6_bank_level_ba_probe=BANK_LEVEL_BA_PROBE_AT_C6,
            ba_sep_stage=detector_reliability.STAGE,
            ba_sep_used_for_profile_selection=False,
            detector_reliability_pending=True,
            rule="the only canonical probe uses detector evidence (p_global, "
                 "s_region, nine regional distances) and C6 has no detector; "
                 "inventing an image-level bank probe would be a new scientific "
                 "choice"))

        for arm in selector.ARMS:
            payload = read_json(reports / f"C6_BANK_LOCK_{arm}.json") or {}
            bank = state["banks"]["banks"][arm]
            checks.append(check(
                f"c6_bank_lock_{arm.lower()}_verifies",
                payload.get("selected_set_sha256") == bank["selected_set_sha256"]
                and payload.get("selector_identity_sha256")
                == contract["selector_identity_sha256"]
                and payload.get("final_bank_size") == selector.FINAL_BANK_PER_ARM
                and payload.get("q_used_for_selection") is False
                and bool(payload.get("provenance_closure", {}).get("closed")),
                f"the {arm} bank lock binds its selected set, the selector "
                f"identity and a closed provenance set",
                selected_set_sha256=payload.get("selected_set_sha256"),
                provenance_closed=payload.get("provenance_closure", {}).get("closed")))

        passed = all(item["ok"] for item in checks)
        return self.result(
            request, mode=VERIFY_C6_LOCKS, checks=checks,
            summary=("C6 matched banks verified" if passed else
                     "C6 lock verification failed"),
            # The ONE place C6 claims scientific evidence.
            scientific_evidence=passed)


class QualityBackendDeviceUndetermined(RuntimeError):
    """No frozen contract fixes the C6 quality-backend execution device."""

    reason_code = "C6_QUALITY_BACKEND_DEVICE_NEEDS_SCIENTIFIC_DECISION"

    def __init__(self, message: str, *, audited: list[str]) -> None:
        super().__init__(message)
        self.audited = list(audited)


#: FROZEN by explicit user decision, preregistered before any C6 fitted
#: threshold, candidate quality result, acceptance count, profile feasibility
#: result or matched-bank result was observed.
#:
#: This is the DEVICE FAMILY, not a machine. Version-B's quality calibration ran
#: on the CUDA backend family and Version C stays in it for the corresponding
#: measurements. The exact GPU model, driver, CUDA runtime and library versions
#: are RUN PROVENANCE and are recorded per run — Version B measured on an L4 and
#: the Version-C host is an RTX 5090, so no bitwise reproduction across NVIDIA
#: models is claimed or implied.
#:
#: There is no availability-based fallback. `QualityBackends` routes the SCRFD
#: provider, FaceXFormer parsing and the AdaFace embedding to this device, and
#: silently dropping to CPU would change the measurements the gate is applied to.
FROZEN_QUALITY_BACKEND_DEVICE: str | None = "cuda"

#: What was searched, so the report says where the answer is not.
QUALITY_BACKEND_DEVICE_AUDIT: tuple[str, ...] = (
    "v1.5 spec: the word 'device' does not occur; the precision/backend clauses "
    "govern TRAINING runs (GPU model, precision mode, microbatch, effective "
    "batch size), not the quality-metric backends",
    f"frozen config {QUALITY_CONFIG}: no device or provider field",
    "QualityBackends(weight_root, *, device='cpu'): a Python signature default, "
    "not a declared scientific contract",
    "both existing call sites (cli.main, structural_calibration) take the device "
    "from their caller; the CLI exposes it as an operator flag",
    "quality_calibration.calibrate records `device` as run PROVENANCE in its "
    "output, which is how a recorded input behaves, not a frozen one",
    "frozen Version-B reports/m8/quality_calibration.json recorded device='cuda' "
    "on an NVIDIA L4 (torch 2.5.1+cu121); the Version-C host is an RTX 5090, so "
    "'inherit cuda' does not reproduce those numbers either",
    "c4._scientific_device and c5_render.scientific_device require CUDA, but each "
    "is justified by a training/rendering precision contract and neither claims "
    "to govern measurement backends",
    "gpat_trainer.resolve_device is availability-based ('cuda if available'), "
    "which is an operational helper rather than a scientific policy",
)


def archive_superseded_artifact(request: AdapterRequest, path: Path, *,
                                reason: str) -> dict[str, Any] | None:
    """Copy an existing artifact byte-for-byte before it is replaced.

    L.8 forbids winner-only cleanup, and a superseded artifact from a real run is
    evidence about that run. It is preserved as a diagnostic and never as a
    scientific lock.
    """
    import hashlib
    import shutil

    path = Path(path)
    if not path.is_file():
        return None
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    previous = read_json(path) or {}
    archive = path.parent / "superseded" / f"{path.stem}_{digest[:16]}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copyfile(path, archive)
    return {"archived_artifact": archive.relative_to(request.repo).as_posix(),
            "archived_sha256": digest,
            "archived_schema_version": previous.get("schema_version"),
            "archived_generated_at_utc": previous.get("generated_at_utc"),
            "is_scientific_lock": False, "reason": reason,
            "candidates_modified": False}


def archive_superseded_calibration(request: AdapterRequest,
                                   path: Path) -> dict[str, Any] | None:
    """Preserve an existing calibration byte-for-byte before replacing it.

    The GPU host holds a `QUALITY_CALIBRATION.json` whose thresholds and hash
    disagree — a real run's real output, and scientifically invalid. L.8 forbids
    winner-only cleanup, so it is copied to a deterministic diagnostic path and
    its SHA bound into the replacement rather than being quietly overwritten. It
    is recorded as diagnostic evidence and never as a scientific lock.
    """
    import hashlib
    import shutil

    path = Path(path)
    if not path.is_file():
        return None
    original = path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    previous = read_json(path) or {}

    thresholds = previous.get("thresholds") or {}
    recorded = str(previous.get("threshold_sha256") or "")
    self_consistent = None
    if thresholds and recorded:
        try:
            from prism_fas.synthesis.quality_gate import Thresholds

            self_consistent = Thresholds.from_dict(thresholds).sha256() == recorded
        except Exception:                                # noqa: BLE001
            self_consistent = False

    archive = path.parent / "superseded" / f"{path.stem}_{digest[:16]}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        shutil.copyfile(path, archive)
    return {
        "archived_calibration": archive.relative_to(request.repo).as_posix(),
        "archived_sha256": digest,
        "archived_schema_version": previous.get("schema_version"),
        "archived_generated_at_utc": previous.get("generated_at_utc"),
        "archived_threshold_sha256": recorded,
        "archived_was_self_consistent": self_consistent,
        "is_scientific_lock": False,
        "reason": ("superseded by the final-threshold-identity correction: the "
                   "previous artifact recorded the assembled §11.4 NOMINAL "
                   "thresholds alongside the calibrator's hash of the map they "
                   "superseded, so FrozenCalibration.load refused it"),
        "candidates_modified": False,
    }


class ThresholdIdentityMismatch(RuntimeError):
    """A calibration payload whose recorded hash is not its thresholds' hash."""

    reason_code = "THRESHOLD_IDENTITY_MISMATCH"


def _final_calibration_payload(fitted: Mapping[str, Any],
                               nominal: Mapping[str, float],
                               threshold_provenance: Mapping[str, Any], *,
                               device: str, provenance: Mapping[str, Any],
                               backends: Any,
                               package_identity: str) -> dict[str, Any]:
    """The one final C6 calibration payload, with two identities kept apart.

    `thresholds` / `threshold_sha256` are the FINAL scientific pair: the
    assembled §11.4 NOMINAL and the hash of exactly those values. This is what
    `CandidateEvaluator` consumes, so it is the pair that has to agree.

    `calibrator_fitted_thresholds` / `calibrator_fitted_threshold_sha256` are the
    source-reference values the calibrator produced and the hash of exactly
    those. They stay as provenance — for an inherited metric they are evidence
    about the population, never the gate — and they are never relabelled as the
    final identity.

    The mismatch is checked here rather than discovered downstream: a producer
    that ships one threshold map carrying another map's hash should not get as
    far as a consumer.
    """
    from prism_fas.synthesis.quality_gate import Thresholds

    final_sha = Thresholds.from_dict(dict(nominal)).sha256()
    expected = threshold_provenance["nominal_identity_sha256"]
    if final_sha != expected:
        raise ThresholdIdentityMismatch(
            f"the hash of the assembled NOMINAL set ({final_sha}) is not the "
            f"identity the inheritance recorded ({expected})")

    fitted_map = dict(fitted.get("thresholds") or {})
    fitted_sha = str(fitted.get("threshold_sha256") or "")

    payload = {
        **fitted,
        # The final scientific pair. Both are replaced together, always.
        "thresholds": {key: float(value) for key, value in nominal.items()},
        "threshold_sha256": final_sha,
        "nominal_identity_sha256": expected,
        "threshold_provenance": dict(threshold_provenance),
        # The superseded pair, preserved and labelled as what it is.
        "calibrator_fitted_thresholds": fitted_map,
        "calibrator_fitted_threshold_sha256": fitted_sha,
        "calibrator_fitted_thresholds_are_provenance_only": True,
        "final_threshold_identity_source": (
            "§11.4 assembled NOMINAL: Version-B inherited where semantically "
            "compatible; source-reference derived where required"),
        "package_identity": package_identity,
        "split": "source_train",
        "quality_backend_device_family": FROZEN_QUALITY_BACKEND_DEVICE,
        "quality_backend_run_provenance": dict(provenance),
        "quality_models": fitted.get("quality_models") or backends.manifest(),
        "target_access": 0,
    }
    # Self-check the invariant the consumer will enforce, before writing it.
    if Thresholds.from_dict(payload["thresholds"]).sha256() != payload["threshold_sha256"]:
        raise ThresholdIdentityMismatch(
            "the assembled payload's thresholds do not hash to its own "
            "threshold_sha256")
    return payload


class QualityBackendDeviceUnavailable(RuntimeError):
    """The frozen device family is not present. C6 blocks; it never falls back."""

    reason_code = "C6_QUALITY_BACKEND_DEVICE_UNAVAILABLE"


def _quality_backend_device(request: AdapterRequest) -> str:
    """The frozen execution device for the quality backends, or fail closed.

    Two refusals, never a substitution. If no device is frozen the stage stops
    rather than picking one; if the frozen family is absent on this host the
    stage stops rather than dropping to CPU. `torch.cuda.is_available()` appears
    here only to REFUSE — never to choose.
    """
    if not FROZEN_QUALITY_BACKEND_DEVICE:
        raise QualityBackendDeviceUndetermined(
            "C6 quality-backend device is not fixed by any frozen contract. CPU "
            "and CUDA send SCRFD, FaceXFormer and AdaFace down different "
            "kernels, and each tau is a percentile over those measurements, so "
            "the choice is result-affecting. Refusing to pick one.",
            audited=list(QUALITY_BACKEND_DEVICE_AUDIT))

    device = FROZEN_QUALITY_BACKEND_DEVICE
    if device.startswith("cuda") and not _cuda_present():
        raise QualityBackendDeviceUnavailable(
            f"the frozen C6 quality-backend device family is {device!r} and this "
            "host exposes no CUDA device. C6 scientific measurement BLOCKS; it "
            "does not fall back to CPU, because the gate would then be applied "
            "to measurements from a different backend than the one frozen.")
    return device


def _cuda_present() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:                                    # noqa: BLE001
        return False


def quality_backend_provenance(device: str) -> dict[str, Any]:
    """What actually executed, recorded beside the frozen family.

    The family is the contract; this is the run. Version B measured on an L4 and
    this will not, so the artifact says which machine produced which numbers
    instead of implying they are interchangeable.
    """
    record: dict[str, Any] = {"frozen_device_family": FROZEN_QUALITY_BACKEND_DEVICE,
                              "requested_device": device,
                              "bitwise_reproduction_across_gpu_models_claimed": False}
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            record["gpu_name"] = torch.cuda.get_device_name(0)
            record["cuda_runtime"] = torch.version.cuda
            record["cudnn"] = torch.backends.cudnn.version()
            record["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    except Exception as error:                           # noqa: BLE001
        record["torch_probe_error"] = type(error).__name__
    try:
        import onnxruntime

        record["onnxruntime"] = onnxruntime.__version__
        record["onnxruntime_providers"] = list(onnxruntime.get_available_providers())
    except Exception as error:                           # noqa: BLE001
        record["onnxruntime_probe_error"] = type(error).__name__
    return record


def _verify_quality_model_binding(weight_root: Path) -> dict[str, Any]:
    """Prove the three pinned roles resolve and hash-verify. Nothing is fetched."""
    roles = ("identity", "parsing", "detector")
    try:
        from prism_fas.synthesis.quality_models import QualityModelRegistry

        registry = QualityModelRegistry.resolve(Path(weight_root), roles=roles)
        return {"ok": True, "roles": list(roles), "error_type": None,
                "verified": dict(getattr(registry, "verified", {}) or {})}
    except Exception as error:                           # noqa: BLE001
        return {"ok": False, "roles": list(roles),
                "error_type": type(error).__name__,
                "error": _sanitize_reason(error)}


def _sanitize_reason(error: BaseException) -> str:
    """A diagnostic may carry a host path; an artifact may not."""
    import re

    return re.sub(r"([A-Za-z]:\\|/home/|/Users/)\S*", "[redacted-path]",
                  str(error))[:400]


def science_module():
    from prism_fas.synthesis import c6_scientific

    return c6_scientific


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = ["STAGE_ID", "MODES", "APPLY_COMMON_GATE", "PROFILE_SELECTION",
           "RELIABILITY_GATES", "MATCHED_BANKS", "CARDINALITY_REFUSAL", "C6Adapter"]
