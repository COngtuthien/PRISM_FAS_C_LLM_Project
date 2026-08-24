"""Seed `reports/evidence/NEGATIVE_EVIDENCE_INDEX.json` from what is already recorded.

Every entry below is transcribed from a record that already exists — a block in
`docs/PROJECT_STATE.md`, a stage artifact, or a decision record in `configs/`.
Nothing here invents an event, a count or a log path: an event whose log was not
kept records an empty `logs` list, which is the honest thing and is also what the
index reports as `entries_without_retained_evidence`.

Re-running this script is safe. It merges by `entry_id`, so a later run updates
an entry in place rather than duplicating it, and it never removes one.

    python scripts/seed_negative_evidence_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.reporting.negative_evidence import (  # noqa: E402
    APPENDIX, BLOCKED_PROTOCOL_DECISION, DISCUSSION, ENGINEERING_FAILURE, LIMITATIONS,
    RESULTS, SCIENTIFIC_NEGATIVE_RESULT, NegativeEvidence, merge, read_index, write_index)

ENTRIES = [
    # --- C5: the 62 retained semantic generation failures --------------------
    NegativeEvidence(
        entry_id="C5-SEMANTIC-GENERATION-FAILURES-2026-08",
        stage="C5", substage="RENDER_CANDIDATES",
        classification=SCIENTIFIC_NEGATIVE_RESULT,
        occurred_on="2026-08",
        result_affecting=True,
        reason=(
            "62 of 6144 planned candidates finalized to an EMPTY exact mask: the "
            "physics artifact did not survive uint8 quantization, so the rendered "
            "file is byte-identical to the live image it was edited from. That is a "
            "candidate with no artifact at all, which is a different thing from an "
            "unconvincing one, and recording it as a success would have handed C6 a "
            "bank padded with unmodified live images to gate. All 62 are Physics "
            "route; GPAT rendered 3072/3072. Per arm: DET 28, RND 20, LLM 14."),
        artifacts=("runs/full/c5/scientific/candidates/<ARM>/<candidate_id>/CANDIDATE.json",),
        paper_eligibility=(RESULTS, DISCUSSION, APPENDIX),
        detail={
            "planned": 6144, "terminal": 6144, "generated": 6082,
            "semantic_failed": 62, "runtime_unresolved": 0,
            "per_arm": {"DET": {"generated": 2020, "physics_failures": 28},
                        "LLM": {"generated": 2034, "physics_failures": 14},
                        "RND": {"generated": 2028, "physics_failures": 20}},
            "immutability": (
                "never retried, deleted, regenerated or re-paired to another live "
                "sample or recipe. The 2048-per-arm budget did not grow, and the "
                "PhysicsEngine, the quantization and the empty-mask condition are "
                "unchanged"),
            "why_it_is_a_result": (
                "the failure rate differs by arm (28/20/14), which is a property of "
                "the recipe banks the arms produced and therefore a measurement"),
            "source_record": "docs/PROJECT_STATE.md first_scientific_gpu_run.c5"}),

    # --- C6: the two failed GPU runs -----------------------------------------
    NegativeEvidence(
        entry_id="C6-THRESHOLD-TYPE-CONTRACT-FAILURE-2026-08-23",
        stage="C6", substage="CHECK_PROFILE_MATCHED_FEASIBILITY",
        classification=ENGINEERING_FAILURE,
        occurred_on="2026-08-23",
        result_affecting=False,
        reason=(
            "the scientific adapter passed `GateProfile.thresholds` — a "
            "dict[str, float] — into `gate_candidates`, which reaches "
            "`quality_gate.evaluate` and reads `thresholds.tau_fd`. The run aborted "
            "at the first gating call with \"'dict' object has no attribute "
            "'tau_fd'\". A producer/consumer TYPE-CONTRACT mismatch: no threshold "
            "value, no §11.4 rule, no calibration, no CUDA and no matched-bank "
            "logic was involved. Candidate measurement had completed; no profile "
            "was assessed, none was selected and no bank was built."),
        artifacts=(),
        logs=(),
        paper_eligibility=(APPENDIX,),
        detail={
            "fix": ("one call site — `profile.as_thresholds()`, GateProfile's own "
                    "conversion, which the engineering rehearsal already used"),
            "contract_tightened": (
                "`gate_candidates(thresholds: Any)` became "
                "`thresholds: Thresholds` with a fail-fast isinstance refusal. No "
                "dict-accepting compatibility path was added: two accepted "
                "representations at the gating boundary is what allowed it"),
            "why_tests_missed_it": (
                "every existing test built the object itself, performing the "
                "conversion the production code omitted; none drove the scientific "
                "adapter's own wiring"),
            "onnx_warnings": (
                "the repeated SCRFD VerifyOutputSizes warnings did not cause this "
                "traceback; SCRFD model, input size and provider are unchanged"),
            "source_record": "docs/PROJECT_STATE.md c6_failed_run_2026_08_23"}),
    NegativeEvidence(
        entry_id="C6-BANK-PROVENANCE-CLOSURE-FAILURE-2026-08-23",
        stage="C6", substage="VERIFY_C6_LOCKS",
        classification=ENGINEERING_FAILURE,
        occurred_on="2026-08-23",
        result_affecting=False,
        reason=(
            "profile assessment, profile selection and matched-bank construction all "
            "completed and were VALID; the run then blocked at VERIFY_C6_LOCKS with "
            "c6_bank_lock_{rnd,det,llm}_verifies failing, because the bank provenance "
            "closure did not account for the C5 semantic failures. A provenance "
            "ACCOUNTING failure, not a scientific gate failure: the selected profile "
            "and the matched banks were not wrong."),
        artifacts=(),
        logs=(),
        paper_eligibility=(APPENDIX,),
        detail={
            "failed_checks": ["c6_bank_lock_rnd_verifies", "c6_bank_lock_det_verifies",
                              "c6_bank_lock_llm_verifies"],
            "retention": ("the failed bank-lock artifacts are retained; a corrected run "
                          "supersedes them under the existing archive policy with "
                          "SHA-256 and reason"),
            "resolved_by": ("provenance_closure now accounts for semantic failures; "
                            "planned=2048, covered=2048, closed=true, unaccounted=[] "
                            "for all three arms"),
            "source_record": "docs/PROJECT_STATE.md c6_lock_verification_run_2026_08_23"}),

    # --- Blocked protocol decisions ------------------------------------------
    NegativeEvidence(
        entry_id="DETECTOR-BA-SEP-PROBE-PROTOCOL-UNFROZEN",
        stage="POST_C8", substage="DETECTOR_RELIABILITY_LOCK_C",
        classification=BLOCKED_PROTOCOL_DECISION,
        occurred_on="2026-08-23",
        result_affecting=True,
        reason=(
            "the synthetic-vs-real separability probe (BA_sep <= 0.75) is frozen to "
            "run AFTER C8 and BEFORE C9, but its PROTOCOL, its evidence vector and "
            "its seed count are still unfrozen. §3.1.1, §17 and the C6 stage row do "
            "not compose into one staging, and the probe needs detector evidence "
            "(p_global, s_region, nine regional distances) that does not exist until "
            "C7. C6 records BA_sep as DEFERRED and produces no BA number rather than "
            "inventing an image-level bank probe, which would have been a new "
            "scientific choice made to close a gate."),
        artifacts=("reports/full/c6/C6_BANK_LOCK_RND.json",
                   "reports/full/c6/C6_BANK_LOCK_DET.json",
                   "reports/full/c6/C6_BANK_LOCK_LLM.json"),
        paper_eligibility=(LIMITATIONS, APPENDIX),
        detail={
            "unresolved": ["DETECTOR_BA_SEP_PROBE_PROTOCOL",
                           "DETECTOR_BA_SEP_EVIDENCE_VECTOR",
                           "DETECTOR_BA_SEP_PROBE_SEEDS"],
            "frozen_staging": "C8_CLOSURE_BEFORE_C9_SOURCE_MATRIX_LOCK_C",
            "consequence": "C9 remains BLOCKED; C8 may complete",
            "may_not_be_resolved_from": (
                "C8 outcomes. Choosing the missing protocol after seeing which "
                "detectors trained well would make the gate a function of the "
                "result it is supposed to test"),
            "source_record": ("docs/PROJECT_STATE.md "
                              "synthetic_vs_real_reliability_stage")}),
    NegativeEvidence(
        entry_id="C7-SOURCE-SEARCH-TRAINING-POPULATION-UNFROZEN",
        stage="C7", substage="SOURCE_SEARCH",
        classification=BLOCKED_PROTOCOL_DECISION,
        occurred_on="2026-08-24",
        result_affecting=True,
        reason=(
            "SUPERSEDED on 2026-08-24 by "
            "C7-SOURCE-SEARCH-SYNTHETIC-ARM-FROZEN-DET-2026-08-24, and retained "
            "because the period during which the path was implemented and BLOCKED is "
            "itself the evidence that the decision was not taken from a result. "
            "§15.2.2 fixes the detector/loss envelope completely — coordinate order, "
            "multipliers, one pass, ranking tuple, tie-break — but does not say which "
            "of C6's three matched banks the bounded search trains against. The arm "
            "IS the treatment factor C8 compares, so tuning on LLM would give every "
            "later comparison a tuning advantage no statistic removes. The scientific "
            "path was implemented and blocked on the record rather than choosing a "
            "default."),
        artifacts=("configs/search/c7_source_search_decision.yaml",),
        paper_eligibility=(LIMITATIONS, APPENDIX),
        detail={
            "unresolved_fields": ["training_arm", "protocol", "track",
                                  "selection_tuple_name", "trial_schedule"],
            "options_recorded": {"RND": "the control arm; removes the confound",
                                 "DET": "CONFOUNDS C-H1/C-H2",
                                 "LLM": "CONFOUNDS C-H1/C-H2 most directly"},
            "status_now": "RESOLVED: DET, frozen 2026-08-24",
            "blocked": "the first scientific C7 trial, until the freeze",
            "does_not_block": ("C7 engineering readiness, the C7/C8 rehearsals or any "
                               "implementation work"),
            "verifier": "prism_fas.search.c7_decision.load_decision"}),

    # --- Engineering defects this milestone found and fixed -------------------
    NegativeEvidence(
        entry_id="C8-FIXTURE-EXECUTOR-REACHABLE-UNDER-SCIENCE-2026-08-24",
        stage="C8", substage="EXECUTE_ROWS",
        classification=ENGINEERING_FAILURE,
        occurred_on="2026-08-24",
        result_affecting=False,
        reason=(
            "C8 had ONE workflow whose row executor called `audit_batch` and "
            "`build_audit_detector` unconditionally and stepped `SmokeBudget.steps` "
            "times. The scheduler was already correct — under a scientific context "
            "`ExecutionContext.limit` returns the full 42 and never reads SMOKE_ROWS "
            "— so a full-profile run would have trained all 42 rows on fixture "
            "batches through an audit model and written 42 PASS manifests, which C9 "
            "would then have frozen. Found by audit before any GPU run; no fixture "
            "metric was ever written to a scientific namespace."),
        artifacts=("src/prism_fas/pipeline/adapters/c8.py",),
        paper_eligibility=(APPENDIX,),
        detail={
            "fix": ("`_engineering_workflow` / `_scientific_workflow` split, with "
                    "`assert_fixture_permitted` as the first statement of `_run_one` "
                    "and of `_failure_preservation`"),
            "generalization": ("the same shape as the C4 defect: correct engineering "
                               "reached by a scientific profile"),
            "regression": "tests/pipeline/test_scientific_fixture_leakage.py"}),
    NegativeEvidence(
        entry_id="C7-SCIENTIFIC-LOCK-ASSERTED-ABSENT-2026-08-24",
        stage="C7", substage="SOURCE_SEARCH",
        classification=ENGINEERING_FAILURE,
        occurred_on="2026-08-24",
        result_affecting=False,
        reason=(
            "C7's only search path used a deterministic analytic objective and "
            "asserted that `reports/full/c7/DETECTOR_CONFIG_LOCK.json` does NOT "
            "exist — while C8 declares that exact path as a required input. The full "
            "profile could therefore never legitimately produce the lock C8 needs, "
            "and had one existed from an earlier run every rehearsal would have "
            "failed on a file it was right to have produced. A scientific "
            "execution-path gap, not a wrong value."),
        artifacts=("src/prism_fas/pipeline/adapters/c7.py",),
        paper_eligibility=(APPENDIX,),
        detail={
            "fix": ("a real `_scientific_workflow` that trains through M9Trainer and "
                    "writes the lock, plus a corrected engineering check that asserts "
                    "THIS RUN wrote no scientific lock rather than that none exists"),
            "regression": "tests/pipeline/test_c7_scientific_path.py"}),
    NegativeEvidence(
        entry_id="C10-FIXTURE-LABELS-WRITABLE-INTO-SEALED-TARGET-2026-08-24",
        stage="C10", substage="BUILD_FIXTURE_PACKAGE",
        classification=ENGINEERING_FAILURE,
        occurred_on="2026-08-24",
        result_affecting=False,
        reason=(
            "`_build_fixture` called `sources.target_roots(...)` and then `mkdir`-ed "
            "every returned root and wrote a `labels.json` of invented labels into "
            "it. Under a scientific context that call returns the REAL sealed target "
            "package roots, so the stage would have written fabricated labels inside "
            "the artifact the firewall exists to protect. Unreachable in practice "
            "because the precondition gate blocks on the absent C9 lock; structurally "
            "reachable, and found by audit before any target package existed."),
        artifacts=("src/prism_fas/pipeline/adapters/c10.py",),
        paper_eligibility=(APPENDIX,),
        detail={"fix": "`assert_fixture_permitted` as the first statement of "
                       "`_build_fixture`, before any path is created",
                "target_accessed": False,
                "regression": "tests/pipeline/test_scientific_fixture_leakage.py"}),
    # --- the shared search engine, and the smaller rehearsal wiring defects ---
    NegativeEvidence(
        entry_id="SEARCH-ENGINE-COMPLETED-PASS-RESUME-2026-08-24",
        stage="C7", substage="SOURCE_SEARCH",
        classification=ENGINEERING_FAILURE,
        occurred_on="2026-08-24",
        result_affecting=False,
        reason=(
            "resuming a COMPLETED coordinate pass re-walked the coordinates with "
            "`best` already restored to the final winning vector, so the EARLY "
            "coordinates' trials were regenerated with the LATE coordinates moved. "
            "Those are different configurations with different canonical hashes, so "
            "they missed the reuse table and executed: a rerun of a finished search "
            "would have silently retrained GPU trials, and the trials it produced "
            "would not have been the ones the pass selected from. Found by the C7 "
            "rehearsal, in the engine C4 and C7 share, before any GPU run."),
        artifacts=("src/prism_fas/search/coordinate.py",),
        paper_eligibility=(APPENDIX,),
        detail={
            "fix": ("a COMPLETED state under a matching plan identity is RETURNED "
                    "rather than re-walked; L.11 applied at pass granularity"),
            "also_affects": "C4, which shares the engine and had never exercised the "
                            "completed-pass resume branch",
            "regression": ("tests/pipeline/test_search_engine.py::"
                           "test_resuming_a_completed_pass_re_executes_nothing")}),
    NegativeEvidence(
        entry_id="C7-REHEARSAL-WIRING-DEFECTS-2026-08-24",
        stage="C7", substage="SCIENTIFIC_SOURCE_SEARCH",
        classification=ENGINEERING_FAILURE,
        occurred_on="2026-08-24",
        result_affecting=False,
        reason=(
            "four defects the C7 production-path rehearsal caught in one sitting, "
            "each of which would have surfaced as a failed or wrong GPU run. (1) The "
            "coordinate-order check compared against the literal `learning_rate` "
            "while the approved LR decision replaces that coordinate in place with "
            "`learning_rate_multiplier`, so it failed a CORRECTLY executing search. "
            "(2) A `check()` call passed `decision_logit_name` twice and raised "
            "TypeError. (3) The lock verifier grepped `metrics_source` for "
            "\"analytic\", which the honest description \"no analytic objective\" "
            "trips over - failing a real lock and passing one that merely omitted the "
            "phrase; it is a declared boolean now. (4) The trial runner returned the "
            "status \"FAILED\", which is outside `coordinate.TRIAL_STATUS`, so the "
            "summary on disk and the leaderboard disagreed about the same "
            "configuration."),
        artifacts=("src/prism_fas/pipeline/adapters/c7.py",
                   "tests/pipeline/test_c7_scientific_path.py"),
        paper_eligibility=(APPENDIX,),
        detail={
            "why_it_matters": (
                "none is a scientific finding; together they are the case for "
                "rehearsing a production path against stubs before spending GPU "
                "hours, which is the methodology point"),
            "found_by": "tests/pipeline/test_c7_scientific_path.py"}),
    NegativeEvidence(
        entry_id="C8-ROW-ARTIFACT-DEFECTS-2026-08-24",
        stage="C8", substage="EXECUTE_ROWS",
        classification=ENGINEERING_FAILURE,
        occurred_on="2026-08-24",
        result_affecting=False,
        reason=(
            "two defects the C8 row rehearsal caught. (1) The per-row training "
            "history was written only inside a loop over the stage lineage, so a row "
            "could finish with no `train_history.jsonl` at all - and L.8 requires "
            "every atomic run to emit its own durable artifacts, an absent history "
            "being indistinguishable from a run that never happened. (2) The run "
            "manifest recorded `trainer.variant` rather than the variant the ROW "
            "resolved from its preregistered flags; the same object in production, "
            "but a trainer that substituted a variant would have agreed with itself "
            "instead of being caught."),
        artifacts=("src/prism_fas/pipeline/adapters/c8.py",
                   "tests/pipeline/test_c8_scientific_path.py"),
        paper_eligibility=(APPENDIX,),
        detail={
            "fix": ("the row summary history row is written unconditionally, before "
                    "the per-stage rows; the manifest reads `config.variant`"),
            "found_by": ("tests/pipeline/test_c8_scientific_path.py and "
                         "tests/pipeline/test_c7_search_arm_decision.py")}),
    NegativeEvidence(
        entry_id="C7-SOURCE-SEARCH-SYNTHETIC-ARM-FROZEN-DET-2026-08-24",
        stage="C7", substage="SOURCE_SEARCH",
        classification=BLOCKED_PROTOCOL_DECISION,
        occurred_on="2026-08-24",
        result_affecting=True,
        reason=(
            "RESOLVED. The under-specified C7 input - which of C6's three matched "
            "banks supplies the synthetic quarter of the batch during the bounded "
            "search - was frozen to DET by explicit scientific decision, before any "
            "C7 scientific metric existed. DET is the structured non-LLM control and "
            "the only generator arm primary in BOTH tracks (Track G runs RND/DET/LLM, "
            "Track R runs DET/LLM), so one non-treatment anchor serves both track "
            "searches; tuning on LLM would have handed the proposed treatment a "
            "hyperparameter advantage over its own controls. Retained here as a "
            "LIMITATION because the choice is real and v1.5 does not make it: a "
            "configuration selected on DET is not guaranteed optimal for RND or LLM, "
            "and the paper must say so rather than imply the spec chose it."),
        artifacts=("configs/search/c7_source_search_decision.yaml",),
        paper_eligibility=(LIMITATIONS, APPENDIX),
        detail={
            "supersedes": "C7-SOURCE-SEARCH-TRAINING-POPULATION-UNFROZEN",
            "decision_identity":
                "ed4f6b777d9f95f089a76191b863e2fb2df0b9e13434470ffd736d6e511b474e",
            "timing": "BEFORE_FIRST_C7_SCIENTIFIC_TRIAL",
            "spec_status": "UNDER_SPECIFIED_IN_V1_5",
            "not_a_spec_claim": "v1.5 does not choose DET; this is a closure decision",
            "searches": "one bounded pass per TRACK, never one per generator arm",
            "prohibited_after_freeze": ["RND", "LLM", "per-arm search",
                                        "pooled or result-dependent arm selection",
                                        "a second pass that changes the arm"],
            "regression": "tests/pipeline/test_c7_search_arm_decision.py"}),
]


def main() -> int:
    existing = (read_index(REPO) or {}).get("entries", [])
    entries = merge(existing, ENTRIES)
    path = write_index(REPO, entries)
    index = read_index(REPO) or {}
    print(f"wrote {path}: {index['entry_count']} entries, "
          f"identity {index['index_identity'][:16]}")
    for name, count in sorted(index["by_classification"].items()):
        print(f"  {name}: {count}")
    without = index["entries_without_retained_evidence"]
    if without:
        print(f"  entries with no retained artifact or log: {without}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
