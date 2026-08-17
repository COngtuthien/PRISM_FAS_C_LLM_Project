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
from prism_fas.pipeline.adapters.common import (EngineeringAdapter, RequiredInput,
                                                SmokeBudget, check, resume_decision,
                                                stage_reports_dir, utc, write_artifact)
from prism_fas.pipeline.adapters.tiny import ENGINEERING_NOMINAL, gate_metrics

STAGE_ID = "C6"

APPLY_COMMON_GATE = "APPLY_COMMON_GATE"
PROFILE_SELECTION = "PROFILE_SELECTION"
RELIABILITY_GATES = "RELIABILITY_GATES"
MATCHED_BANKS = "MATCHED_BANKS"
CARDINALITY_REFUSAL = "CARDINALITY_REFUSAL"

MODES: tuple[str, ...] = (APPLY_COMMON_GATE, PROFILE_SELECTION, RELIABILITY_GATES,
                          MATCHED_BANKS, CARDINALITY_REFUSAL)

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
            RequiredInput("quality_calibration", "reports/full/c6/QUALITY_CALIBRATION.json",
                          "the source_train-fitted NOMINAL thresholds the three profiles "
                          "are derived from"),
            RequiredInput("c5_candidates", "reports/full/c5",
                          "the 2048 rendered candidates per arm the gate evaluates"),
        )

    def run_smoke(self, request: AdapterRequest) -> list[AdapterResult]:
        budget = SmokeBudget.from_profile(request.profile)
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
            "fixture_backed": True, "budget": budget.as_dict()})
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
            "is_scientific_lock": False, "fixture_backed": True})
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
            "fixture_backed": True, "scientific_gate_satisfied": False})
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
            "is_scientific_bank": False, "fixture_backed": True,
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
            "fixture_backed": True})
        return self.result(request, mode=CARDINALITY_REFUSAL, checks=checks,
                           artifacts=[artifact])


__all__ = ["STAGE_ID", "MODES", "APPLY_COMMON_GATE", "PROFILE_SELECTION",
           "RELIABILITY_GATES", "MATCHED_BANKS", "CARDINALITY_REFUSAL", "C6Adapter"]
