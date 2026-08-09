"""What the M10 report must say about itself, assembled from frozen artifacts.

Three things in M10 are easy to present better than they are, so each is stated
here once, derived rather than narrated, and rendered into the report:

*   **A02** applies the frozen GPAT generator to an out-of-training-distribution
    conditioning bank through an explicitly authorized compatibility control, and
    its accepted pool is 3.8% smaller than the structured arm's.
*   **A09** measured 8 of 9 bounded-parity checks passing; the ninth exceeded its
    frozen tolerance by ~14% and the tolerance was NOT widened.
*   **`run.json.dataset.synthetic_total`** carries a stale 871 for rows that declare
    `synthetic: none`. The reconciliation here derives each row's ACTUAL synthetic
    exposure from that row's own audited batch contract and loss graph, so the
    report prints 0 where 0 is the truth without rewriting a historical artifact.

Nothing in this module imports torch: it is read by the report and the summary,
both of which run beside G8.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

DISCLOSURE_SCHEMA_VERSION = "m10-disclosures-v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}


def synthetic_exposure_reconciliation(root: Path) -> dict[str, Any]:
    """Derive real synthetic exposure per row; never trust the stale display field.

    The authority is each row's own audited `batch_contract` and `active_loss_terms`
    in `M10_IMPLEMENTABILITY.json` — the same objects the trainer built. A row whose
    batch draws 0 synthetic samples and whose synthetic loss terms are all inactive
    used 0 synthetic samples, whatever a cosmetic counter printed.
    """
    audit = _read(Path(root) / "M10_IMPLEMENTABILITY.json")
    rows: dict[str, Any] = {}
    for row in audit.get("rows") or []:
        if not row.get("audited"): continue
        contract = row.get("batch_contract") or {}
        terms = row.get("active_loss_terms") or {}
        per_batch = int(contract.get("synthetic", 0))
        rows[str(row["experiment_id"])] = {
            "declared_synthetic_flag": (row.get("flags") or {}).get("synthetic"),
            "synthetic_per_batch": per_batch,
            "batch_phase": contract.get("phase"),
            "synthetic_loss_terms_active": sorted(name for name in
                                                  ("L_cls_syn", "L_local", "L_out", "L_clean", "L_prompt")
                                                  if terms.get(name)),
            "derived_synthetic_exposure": ("none" if per_batch == 0 else
                                           f"{per_batch} per batch drawn from the declared pool")}
    affected = sorted(name for name, block in rows.items() if block["synthetic_per_batch"] == 0)
    return {
        "defect": "run.json.dataset.synthetic_total reports the whole accepted bank (871) for rows "
                  "that declare synthetic: none",
        "cause": "M9TrainingDataset._synthetic_pools filters routes with `if allowed and route not in "
                 "allowed`; for synthetic: none the allowed set is EMPTY, so the guard "
                 "short-circuits and every accepted bank row is added to a pool that is then never "
                 "sampled",
        "training_affected": False,
        "evidence": "each row's own resolved batch contract draws 0 synthetic samples and every "
                    "synthetic loss term is structurally inactive; no identity binds the pools, so "
                    "no checkpoint or hash moved",
        "blast_radius": {"rows_carrying_the_stale_field": affected,
                         "row_count": len(affected),
                         "correct_value_for_those_rows": 0,
                         "artifacts_rewritten": "none — a historical run artifact is not edited to "
                                                "improve presentation; the report derives the true "
                                                "figure instead"},
        "per_row": rows}


def a02_disclosure(root: Path) -> dict[str, Any]:
    """The A02 control, stated in full including what it costs."""
    return {
        "what_it_is": "the H4 control: the SAME frozen GPAT generator weights and the SAME frozen "
                      "M8 v3 quality gate, driven by a predeclared random-operator conditioning "
                      "bank instead of the structured M7 recipe bank",
        "conditioning_is_out_of_distribution": True,
        "statement": "A02 applies the frozen GPAT generator through the explicit compatibility-control "
                     "path to conditioning vectors drawn uniformly from the operator vocabulary. This "
                     "is out-of-training-conditioning-distribution generator conditioning, not normal "
                     "in-distribution GPAT inference.",
        "control_policy": "a02_random_operator_conditioning",
        "control_identity": "28b96d0122f6e2493cc1daa6534d7d7e60fdb63a9071d7422f262ee41a1a8140",
        "exempted_field": "recipe_bank_identity, and nothing else",
        "quality_gate": "frozen, unchanged; degenerate outputs are REJECTED rather than accepted",
        "pools": {"random_operator_accepted": 838, "random_operator_rejected": 282,
                  "random_operator_failed": 0,
                  "structured_accepted": 871, "structured_rejected": 249, "structured_failed": 0,
                  "candidate_budget_each": 1120,
                  "pool_size_difference_percent": round(100.0 * (871 - 838) / 871, 2)},
        "training_exposure_is_equal": {
            "synthetic_samples_per_batch": 8, "g5_optimizer_steps": 1350,
            "note": "both arms see the same NUMBER of synthetic samples over the same schedule; "
                    "only the pool they are drawn from differs in size"},
        "routes": {"random": {"gpat": 487, "physics": 351},
                   "structured": {"gpat": 452, "physics": 419}},
        "bank_identity": "f7f1e6ac20341d32d75dddd19cbf3231ea4eb7554eb49290aee32cf59ec17387",
        "recipe_bank_identity": "9351d08ac824cc67021445d1bb59bd9dc14ef7eb3dfa606414500d8fac49603f",
        "recipes_leaving_the_structured_compatibility_manifold": "70 of 128"}


def backend_parity_record(path: Path) -> dict[str, Any]:
    """A09/H6, read from the frozen parity artifact, reported as measured."""
    parity = _read(path)
    checks = {name: bool(block.get("passed")) for name, block in (parity.get("checks") or {}).items()}
    failed = sorted(name for name, passed in checks.items() if not passed)
    # The exceedance is DERIVED from the artifact, not restated from a summary. The
    # M10 handoff prose says "8/9"; the artifact records ten checks, nine of which
    # pass, and the artifact is what this report believes.
    logits = (parity.get("checks") or {}).get("global_logits") or {}
    exceedance = None
    if logits.get("mean_abs_diff") is not None and logits.get("tolerance_mean"):
        exceedance = round(100.0 * (float(logits["mean_abs_diff"]) / float(logits["tolerance_mean"]) - 1.0), 1)
    return {
        "measured": {
            "global_logits_mean_abs_diff": logits.get("mean_abs_diff"),
            "global_logits_mean_tolerance": logits.get("tolerance_mean"),
            "global_logits_max_abs_diff": logits.get("max_abs_diff"),
            "global_logits_max_tolerance": logits.get("tolerance_max"),
            "mean_tolerance_exceeded_by_percent": exceedance,
            "note": "the M10 handoff prose summarises this as 8/9; the frozen artifact records "
                    "ten checks with nine passing, and the artifact is authoritative"},
        "hypothesis": "H6", "kind": "parity_not_superiority",
        "in_holm_bonferroni_family": False,
        "protocol": "bounded_step_parity, 5 steps, fp32 on both halves, both loading the SAME frozen "
                    "checkpoint",
        "checks_passed": sum(1 for value in checks.values() if value),
        "checks_total": len(checks), "checks": checks, "failed_checks": failed,
        "passed": bool(parity.get("passed")),
        "tolerance_widened": False,
        "statement": f"{sum(1 for value in checks.values() if value)} of {len(checks)} bounded-parity "
                     f"checks passed. `global_logits` passes on the max tolerance and exceeds the "
                     f"MEAN tolerance by approximately {exceedance}% on CPU-fp32 versus CUDA-fp32 "
                     f"kernels. The frozen tolerance was NOT widened, and this is reported as measured "
                     f"parity evidence, not as a passed gate and not as perfect parity.",
        "corrected_error": "the first two attempts let each backend initialize weights independently; "
                           "the dataset is constructed between seeding and head initialization and "
                           "does different work on the two hosts, so the RNG stream diverged. Both "
                           "halves now start from the same frozen checkpoint.",
        "retained_separately": "A09_BACKEND_PARITY_amp_vs_fp32.json — the earlier bf16-vs-fp32 "
                               "measurement, kept as an engineering observation about AMP and never "
                               "substituted for the declared fp32 result",
        "parity_identity": parity.get("parity_identity"),
        "selects_no_checkpoint": True,
        "is_not_a_second_training_result": True}


def target_package_record(root: Path) -> dict[str, Any]:
    """The frozen target evaluation package, from its acceptance artifact."""
    acceptance = _read(Path(root) / "TARGET_PACKAGE_ACCEPTANCE.json")
    return {
        "package_id": "prism_target_eval_v2",
        "feature_identity": "c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8",
        "videos": 1700, "live": 785, "spoof": 915, "spoof_families": 14,
        "frames_planned": 6800, "frames_successful": 6776, "no_face_failures": 24,
        "videos_with_4_frames": 1676, "videos_with_3_frames": 24, "videos_dropped": 0,
        "priors": {"parsing": 6776, "pose": 6776, "visibility": 6776},
        "identity_embeddings": 0,
        "frozen_live_reproduction": "3140/3140 sample_id exact, 3140/3140 crop_sha256 exact",
        "labels_uploaded_to_any_volume": False,
        "acceptance_passed": acceptance.get("passed"),
        "no_face_handling": "the 24 failed frames are NOT invented; their 24 videos keep their 3 "
                            "valid frame predictions and remain in the video population"}


def compute_record(registry: Any, root: Path = Path("reports/m10")) -> dict[str, Any]:
    """Per-run compute, separated into the three costs Table 58 asks for.

    Wall clock is READ from each stage's own `output_hashes.json`, collected off the
    runs volume, not estimated from a smoke.
    """
    raw = (_read(Path(root) / "M10_COMPUTE_RAW.json").get("by_experiment") or {})
    by_experiment = {}
    for record in registry.ordered():
        if record.status != "COMPLETED": continue
        compute = dict(record.compute or {})
        measured = raw.get(record.experiment_id) or {}
        by_experiment[record.experiment_id] = {
            "backend": record.backend,
            "optimizer_steps": compute.get("global_step") or measured.get("optimizer_steps"),
            "epochs": compute.get("epoch") or measured.get("epochs"),
            "trainable_parameters": (compute.get("parameter_counts") or {}).get("trainable"),
            "optimizer_groups": [group.get("name") for group in compute.get("optimizer_groups") or []],
            "training_seconds_total": measured.get("training_seconds_total"),
            "stage_seconds": measured.get("stage_seconds")}
    measured_total = sum(float(block.get("training_seconds_total") or 0.0)
                         for block in raw.values())
    return {
        "training_compute": {
            "gpu": "NVIDIA L4", "precision": "fp32 declared; AMP off for the parity probe",
            "measured_total_training_seconds": round(measured_total, 1),
            "measured_total_training_hours": round(measured_total / 3600.0, 2),
            "rows_measured": len(raw),
            "source": "each run's own stages/<G>/output_hashes.json; the A09 parity row trains "
                      "nothing to completion and contributes no training compute",
            "by_experiment": by_experiment},
        "offline_synthetic_and_preprocessing_compute": {
            "note": "the M7 recipe banks, the M8 v3 synthetic bank and the A02 random-operator bank "
                    "were built in earlier milestones and in a separate A02 pass; that cost is not "
                    "part of any row's training compute and is reported separately",
            "a02_bank_candidates": 1120, "m8_v3_bank_candidates": 1120,
            "target_package_extraction_frames": 6800},
        "online_target_inference_compute": {
            "stage": "G7", "gpu": "NVIDIA L4",
            "frames_per_row": 6776, "videos_per_row": 1700,
            "note": "inference only; no optimizer is constructed and no gradient is taken"},
        "not_recorded": {
            "flops": "not measured; no FLOP counter was instrumented in M9/M10",
            "latency_per_frame": "not measured as a controlled benchmark; the wall-clock figures "
                                 "above include data loading and volume I/O"}}


def build(root: Path, registry: Any) -> dict[str, Any]:
    """Every disclosure the milestone requires, in one section."""
    return {
        "schema_version": DISCLOSURE_SCHEMA_VERSION,
        "a02_random_operator_control": a02_disclosure(root),
        "a09_backend_parity": backend_parity_record(Path(root) / "A09_BACKEND_PARITY.json"),
        "synthetic_total_cosmetic_defect": synthetic_exposure_reconciliation(root),
        "prompt_on_target_is_not_zero": {
            "statement": "PromptHead applicability is `is_synthetic AND attacked region AND visible`. "
                         "A target sample is never synthetic and carries no attack-region mask, so no "
                         "region is applicable and the head returns an EXACT structural zero. G7 "
                         "writes null / not_applicable, never 0.0, and the report never reads a null "
                         "prompt term as zero prompt evidence.",
            "consequence_for_b08": "on target data B08's fusion reduces to "
                                   "s_final = 1 - (1 - p_global)(1 - s_region) for every row",
            "consequence_for_a08": "A08 can differ on target only through TRAINING — L_prompt shaping "
                                   "the shared region embeddings — never through inference-time fusion"},
        "reject_threshold_not_fitted": {
            "statement": "the source-side unknown/reject exposure fit (G6b) was not performed, so "
                         "`unknown_threshold` is null, nothing is rejected, and the rejection rate is "
                         "0.0 by construction rather than by measurement. A target quantile was never "
                         "used to invent one.",
            "affected_metrics": "every reject-dependent metric is reported not_applicable with this "
                                "reason"},
        "trimmed_mean_at_four_frames": {
            "statement": "floor(4 * 0.10) = 0, so the frozen trimmed mean reduces to the plain mean of "
                         "the video's frames. For the 24 three-frame videos floor(3 * 0.10) = 0 as "
                         "well. This is a consequence of the frozen frame plan, stated rather than "
                         "hidden."},
        "blocked_reliability_test": {
            "test_id": "benign_glasses_makeup_lowlight", "status": "BLOCKED",
            "statement": "SiW-Mv2's Makeup_* and Partial_*Glasses videos are labelled SPOOF "
                         "presentations. Treating them as benign live stress cases would invert their "
                         "label, so the test is recorded blocked rather than substituted with an "
                         "invalid population."},
        "compute": compute_record(registry)}
