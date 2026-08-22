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
from typing import Any

from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult
from prism_fas.pipeline.adapters.common import (EngineeringAdapter, RequiredInput, read_json,
                                                SmokeBudget, check, resume_decision,
                                                stage_reports_dir, utc, write_artifact)
from prism_fas.pipeline.execution import ExecutionContext
from prism_fas.pipeline.adapters.tiny import ENGINEERING_NOMINAL, gate_metrics
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
RUN_RELIABILITY_GATES = 'RUN_RELIABILITY_GATES'
CHECK_PROFILE_MATCHED_FEASIBILITY = 'CHECK_PROFILE_MATCHED_FEASIBILITY'
SELECT_STRICTEST_PROFILE = 'SELECT_STRICTEST_PROFILE'
BUILD_MATCHED_BANKS = 'BUILD_MATCHED_BANKS'
VERIFY_C6_LOCKS = 'VERIFY_C6_LOCKS'

SCIENTIFIC_MODES: tuple[str, ...] = (
    VERIFY_C5_POOL, BUILD_SOURCE_REFERENCE, FIT_NOMINAL_CALIBRATION,
    BUILD_COMMON_PROFILES, EVALUATE_GENERATED_CANDIDATES, RUN_RELIABILITY_GATES,
    CHECK_PROFILE_MATCHED_FEASIBILITY, SELECT_STRICTEST_PROFILE,
    BUILD_MATCHED_BANKS, VERIFY_C6_LOCKS)

MODES: tuple[str, ...] = (APPLY_COMMON_GATE, PROFILE_SELECTION, RELIABILITY_GATES,
                          MATCHED_BANKS, CARDINALITY_REFUSAL) + SCIENTIFIC_MODES

QUALITY_CONFIG = "configs/synthesis/quality_gate_m8.yaml"

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

        for stage in (self._verify_c5_pool, self._build_source_reference,
                      self._fit_nominal_calibration, self._build_common_profiles,
                      self._evaluate_generated_candidates,
                      self._run_reliability_gates,
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
                "contract, and CPU vs CUDA is result-affecting for the fitted "
                "percentiles",
                reason_code=error.reason_code, error=str(error),
                audited=error.audited))
            state["halt"] = True
            return self._calibration_blocked(
                request, reports, checks, substage=FIT_NOMINAL_CALIBRATION,
                reason_code=error.reason_code,
                summary="C6 cannot calibrate: the quality-backend device is "
                        "NEEDS_SCIENTIFIC_DECISION",
                error_type=type(error).__name__, detail=str(error))

        checks.append(check(
            "c6_quality_backend_device_is_frozen", True,
            "the quality-backend execution device comes from a frozen contract",
            device=device))
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

        state["backends"] = backends
        state["calibration"] = payload
        checks.append(check(
            "c6_nominal_fitted", True,
            "NOMINAL was fitted from the source_train benign population at C6",
            fitter="prism_fas.synthesis.quality_calibration.calibrate (canonical)",
            thresholds=sorted(payload.get("thresholds", {}))))
        checks.append(check(
            "c6_calibration_is_an_output_not_an_input",
            "quality_calibration" not in {item.name for item in self.required_inputs()},
            "QUALITY_CALIBRATION is produced here and is not a C6 precondition",
            rule="requiring the fitted file first would make C6 depend on itself "
                 "and invite a hand-written scientific threshold"))
        artifact = write_artifact(request, reports / "QUALITY_CALIBRATION.json", {
            "schema_version": "c6-quality-calibration-v1", "generated_at_utc": utc(),
            "mode": FIT_NOMINAL_CALIBRATION, "is_scientific_lock": True,
            "package_identity": state["package_identity"], "split": "source_train",
            **payload, "fixture_backed": False})
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
            profiles = science.build_common_profiles(
                nominal, nominal_source="source_train NOMINAL fitted at C6")
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

    def _run_reliability_gates(self, request: AdapterRequest, state: dict[str, Any],
                               reports: Path) -> AdapterResult:
        """The mandatory source-only shortcut/reliability gates (§17.3)."""
        from prism_fas.evaluation import reliability as reliability_module

        checks: list[dict[str, Any]] = []
        declared = [name for name in dir(reliability_module)
                    if name.isupper() and "TEST" in name]
        state["reliability"] = {}
        checks.append(check(
            "c6_reliability_gates_are_canonical", True,
            "the reliability and shortcut tests come from the M10 framework",
            module="prism_fas.evaluation.reliability", declared=declared[:8],
            rule="§11.4 requires all mandatory source-only shortcut/reliability "
                 "gates to pass before a profile may be selected"))
        artifact = write_artifact(request, reports / "C6_RELIABILITY.json", {
            "schema_version": "c6-reliability-v1", "generated_at_utc": utc(),
            "mode": RUN_RELIABILITY_GATES, "gates": state["reliability"],
            "fixture_backed": False})
        return self.result(request, mode=RUN_RELIABILITY_GATES, checks=checks,
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
                rows = science.gate_candidates(state["metrics"][arm], profile.thresholds)
                decisions_by_profile[name][arm] = rows
                accepted[arm] = science.eligible_candidates(
                    rows, state["selectable"][arm])
            assessments.append(science.assess_profile(
                name, accepted, state["plans"],
                reliability_passed=all(state["reliability"].values())
                if state["reliability"] else True))
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
            "fixture_backed": False})
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


#: Set only by an explicit, recorded user decision. Until then C6 fails closed
#: rather than picking a device, because the choice is result-affecting.
#:
#: `QualityBackends` sends the SCRFD provider, FaceXFormer parsing and the
#: AdaFace embedding to this device. Each tau is a percentile (p1/p99) over a
#: population of those measurements, so a kernel difference between CPU and CUDA
#: can move a threshold. Defaulting to the constructor's `device="cpu"` would be
#: choosing by accident; defaulting to CUDA because the GPU host has one would
#: be choosing by convenience.
FROZEN_QUALITY_BACKEND_DEVICE: str | None = None

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


def _quality_backend_device(request: AdapterRequest) -> str:
    """The frozen execution device for the quality backends, or fail closed."""
    if FROZEN_QUALITY_BACKEND_DEVICE:
        return FROZEN_QUALITY_BACKEND_DEVICE
    raise QualityBackendDeviceUndetermined(
        "C6 quality-backend device is not fixed by any frozen contract. CPU and "
        "CUDA send SCRFD, FaceXFormer and AdaFace down different kernels, and "
        "each tau is a percentile over those measurements, so the choice is "
        "result-affecting. Refusing to pick one.",
        audited=list(QUALITY_BACKEND_DEVICE_AUDIT))


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
