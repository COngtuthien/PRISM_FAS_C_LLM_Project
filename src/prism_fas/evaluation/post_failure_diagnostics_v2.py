"""C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 — pre-execution scientific correction
of `post_failure_diagnostics` (V1).

V1 (`configs/evaluation/c9_post_failure_source_diagnostics_v1.yaml`,
identity `cb05271e26d9a421f2f9277599523e185026e1eab644febc07c75432d26f3fc5`)
was never scientifically executed and is preserved unchanged as historical
pre-execution design evidence. This module corrects five pre-execution
defects found by audit before any GPU run:

  A. the benign-corruption acceptance threshold was self-normalizing
     (derived from the SAME corruption's own calibration-group shift) —
     corrected here to a DISJOINT reference-calibration design: a
     pre-existing, independently-frozen M8 benign-perturbation family
     (`prism_fas.synthesis.quality_calibration.BENIGN_VARIANTS`,
     `BENIGN_NOISE_STD`) is forwarded on the CALIBRATION group to derive a
     fixed tolerance, and the corruption under test is forwarded ONLY on the
     EVALUATION group and scored against that fixed tolerance;
  B. `cross_route_synthetic` reused the BA_sep separability ceiling (LOW is
     good) under a canonical test name whose declared pass_rule is
     "performance is retained across routes" (HIGH is good) — reclassified
     `NEEDS_SCIENTIFIC_DECISION` here; V1's separability-probe attempt is
     preserved unchanged and unused;
  C. an existing complete result set was validated too weakly before
     re-reporting — `validate_existing_diagnostics_result` below is the
     canonical, comprehensive check;
  D. the C8 matrix identity was handwritten placeholder text — bound here
     as the real `source_matrix.build_plan().identity`, cross-checked
     against `C8_ACCEPTANCE.json`, fail-closed on mismatch or absence;
  E. the group-safe split was only proven globally — proven per domain here.

**THIS IS NOT A BA_sep REVISION, A RELIABILITY-BARRIER RESCUE PROTOCOL, A C9
PASS PROTOCOL, OR A TARGET PROTOCOL.** No function in this module writes to
`reports/full/c8/reliability/synthetic_vs_real_spoof_probe/`,
`reports/full/c8/DETECTOR_RELIABILITY_LOCK_C.json`, or the V1 diagnostics
namespace (`post_failure_diagnostics.DIAGNOSTICS_DIR`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from prism_fas.evaluation.post_failure_diagnostics import (
    CORRUPTION_FUNCTIONS, PostFailureDiagnosticsError,
    calibration_evaluation_split, color_corrupt, forward_corruption_evidence_for_arm,
    jpeg_corrupt, resize_corrupt, resolve_source_dev_live_records,
    resolve_synthetic_population_by_route, run_cross_route_diagnostic_for_arm)

#: Where every V2 diagnostic artifact lives — a SEPARATE namespace from V1's,
#: which is itself separate from BA_sep's own reliability directory.
DIAGNOSTICS_DIR = "reports/full/c8/reliability/post_failure_source_diagnostics_v2"
PROTOCOL_BINDING_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_PROTOCOL_BINDING.json"
POPULATION_BINDING_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_POPULATION_BINDING.json"
CHECKPOINT_BINDING_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_CHECKPOINT_BINDING.json"
RESULT_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_RESULT.json"
PER_TEST_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_PER_TEST.json"
PROVENANCE_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_PROVENANCE.json"
VERDICT_PATH = f"{DIAGNOSTICS_DIR}/DIAGNOSTICS_VERDICT.json"

BINDING_ARTIFACT_PATHS: dict[str, str] = {
    "protocol": PROTOCOL_BINDING_PATH, "population": POPULATION_BINDING_PATH,
    "checkpoint": CHECKPOINT_BINDING_PATH,
}
RESULT_ARTIFACT_PATHS: dict[str, str] = {
    "result": RESULT_PATH, "per_test": PER_TEST_PATH,
    "provenance": PROVENANCE_PATH, "verdict": VERDICT_PATH,
}

PROTOCOL_CONFIG_PATH = "configs/evaluation/c9_post_failure_source_diagnostics_v2.yaml"
C8_ACCEPTANCE_PATH = "reports/full/c8/C8_ACCEPTANCE.json"

#: cross_route_synthetic is NEEDS_SCIENTIFIC_DECISION under V2 (Defect B) —
#: only the three benign-corruption tests are executable.
EXECUTABLE_TESTS: tuple[str, ...] = (
    "benign_jpeg_corruption", "benign_resize_corruption", "benign_color_corruption",
)
BENIGN_CORRUPTION_TESTS: tuple[str, ...] = EXECUTABLE_TESTS
ALL_TESTS: tuple[str, ...] = (
    "benign_jpeg_corruption", "benign_resize_corruption", "benign_color_corruption",
    "cross_route_synthetic", "residual_scale_zero", "recipe_region_shift",
    "artifact_map_swap", "crop_padding_interpolation",
)
DOMAINS: tuple[str, ...] = ("casia_fasd", "msu_mfsd")


# ==============================================================================
# 1. Protocol
# ==============================================================================

def load_protocol(repo: Path) -> dict[str, Any]:
    """The frozen V2 protocol, or a refusal naming why it is absent. Never
    invents a value."""
    import yaml

    path = Path(repo) / PROTOCOL_CONFIG_PATH
    if not path.is_file():
        raise PostFailureDiagnosticsError(
            f"C9_POST_FAILURE_SOURCE_DIAGNOSTICS_V2 is not frozen (expected "
            f"{PROTOCOL_CONFIG_PATH} to exist and declare status: FROZEN_NOT_RUN)")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, Exception) as error:                # noqa: BLE001
        raise PostFailureDiagnosticsError(f"{PROTOCOL_CONFIG_PATH} did not parse: {error}") from error
    if not isinstance(payload, dict) or payload.get("status") != "FROZEN_NOT_RUN":
        raise PostFailureDiagnosticsError(
            f"{PROTOCOL_CONFIG_PATH} does not declare status: FROZEN_NOT_RUN")
    return payload


_PROTOCOL_IDENTITY_EXCLUDED_KEYS = frozenset({
    "frozen_on", "approved_by", "status", "schema_version", "decision_id",
    "document_kind", "no_diagnostic_metric_observed_before_freeze",
})


def protocol_identity(protocol: Mapping[str, Any]) -> str:
    """sha256 over every result-affecting protocol field, sorted keys, no
    timestamps. Changes if and only if a result-affecting field changes."""
    material = {key: value for key, value in protocol.items()
               if key not in _PROTOCOL_IDENTITY_EXCLUDED_KEYS}
    return hashlib.sha256(json.dumps(
        material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def active_protocol_identity(repo: Path) -> str:
    return protocol_identity(load_protocol(repo))


# ==============================================================================
# 2. C8 matrix identity binding (Defect D)
# ==============================================================================

def bind_c8_matrix_identity(repo: Path) -> dict[str, Any]:
    """The REAL canonical C8 matrix identity, cross-checked against
    `C8_ACCEPTANCE.json`'s own `matrix_identity` field. Fails closed if the
    acceptance file is absent or the two identities disagree — never binds a
    handwritten placeholder."""
    from prism_fas.evaluation.source_matrix import build_plan

    canonical = build_plan().identity
    path = Path(repo) / C8_ACCEPTANCE_PATH
    if not path.is_file():
        raise PostFailureDiagnosticsError(
            f"{C8_ACCEPTANCE_PATH} is absent; the canonical C8 matrix identity cannot "
            "be cross-checked before binding it, fail closed")
    try:
        accepted = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PostFailureDiagnosticsError(f"{C8_ACCEPTANCE_PATH} did not parse: {error}") from error
    bound = accepted.get("matrix_identity")
    if bound != canonical:
        raise PostFailureDiagnosticsError(
            f"{C8_ACCEPTANCE_PATH}.matrix_identity ({bound!r}) does not match the "
            f"canonical source_matrix.build_plan().identity ({canonical!r}); fail "
            "closed rather than bind a stale or drifted C8 matrix identity")
    return {"c8_matrix_identity": canonical, "c8_acceptance_matrix_identity": bound,
           "c8_acceptance_path": C8_ACCEPTANCE_PATH}


# ==============================================================================
# 3. Per-domain calibration/evaluation group safety (Defect E)
# ==============================================================================

def _group_set_identity(groups: set) -> str:
    return hashlib.sha256("|".join(sorted(groups)).encode("utf-8")).hexdigest()


def verify_per_domain_group_safety(records: Sequence[Mapping[str, str]],
                                   split: Mapping[str, Sequence[str]], *,
                                   domains: Sequence[str] = DOMAINS) -> dict[str, Any]:
    """Fail closed unless EVERY domain has a non-empty calibration group set,
    a non-empty evaluation group set, and an empty calibration/evaluation
    intersection — a global (pooled) check alone is not sufficient."""
    calibration_groups = set(split["calibration"])
    evaluation_groups = set(split["evaluation"])
    by_domain: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for domain in domains:
        domain_groups = {str(r["stable_group_identity"]) for r in records
                         if r["source_domain"] == domain}
        domain_calibration = domain_groups & calibration_groups
        domain_evaluation = domain_groups & evaluation_groups
        intersection = domain_calibration & domain_evaluation
        by_domain[domain] = {
            "sample_count": sum(1 for r in records if r["source_domain"] == domain),
            "calibration_unique_groups": len(domain_calibration),
            "evaluation_unique_groups": len(domain_evaluation),
            "calibration_identity": _group_set_identity(domain_calibration),
            "evaluation_identity": _group_set_identity(domain_evaluation),
        }
        if not domain_calibration:
            problems.append(f"{domain}: zero calibration groups")
        if not domain_evaluation:
            problems.append(f"{domain}: zero evaluation groups")
        if intersection:
            problems.append(f"{domain}: calibration/evaluation intersection non-empty "
                            f"({len(intersection)})")
    if problems:
        raise PostFailureDiagnosticsError(
            "per-domain calibration/evaluation split is degenerate, fail closed: "
            + "; ".join(problems))
    return {"per_domain": by_domain, "domains": list(domains)}


# ==============================================================================
# 4. Reference benign-control forwarding and threshold derivation (Defect A)
# ==============================================================================

def forward_reference_benign_evidence_for_arm(repo: Path, checkpoints: Sequence[Any], *,
                                              sample_ids: Sequence[str],
                                              domains: Sequence[str] = DOMAINS
                                              ) -> dict[str, dict[str, Any]]:
    """BEFORE evidence, plus AFTER evidence for each of the frozen M8
    reference benign variants
    (`quality_calibration.BENIGN_VARIANTS`/`BENIGN_NOISE_STD`), averaged over
    an arm's 5 checkpoints exactly as `forward_corruption_evidence_for_arm`
    does for a tested corruption. These are the SAME frozen, independently
    derived benign transforms M8 already froze for its own quality-gate
    calibration — reused verbatim, never a second perturbation
    implementation, and deliberately disjoint from JPEG/resize/color.

    Returns `{sample_id: {"before": vector, "after_by_variant": {name: vector}}}`.
    """
    import torch
    from dataclasses import replace

    from prism_fas.detector.dataset import LIVE, M9ValidationDataset, collate_items
    from prism_fas.evaluation.synthetic_real_probe import (average_checkpoint_evidence,
                                                            construct_row_trainer,
                                                            forward_checkpoint_evidence)
    from prism_fas.pipeline.adapters import sources
    from prism_fas.synthesis.quality_calibration import (BENIGN_NOISE_STD, BENIGN_VARIANTS,
                                                          benign_variant)

    if not checkpoints:
        raise PostFailureDiagnosticsError("at least one checkpoint is required")
    inputs = sources.verify_detector_inputs(repo)
    package_root = Path(repo) / inputs["package_root"]

    before_by_checkpoint: list[dict[str, np.ndarray]] = []
    after_by_checkpoint: list[dict[str, dict[str, np.ndarray]]] = []
    for binding in checkpoints:
        trainer = construct_row_trainer(repo, binding)
        dataset = M9ValidationDataset(package_root, trainer.loader_config,
                                      cache_root=trainer.cache_root, domains=domains)
        wanted = set(sample_ids)
        positions = [position for position in dataset.positions
                    if dataset.sample_id_of(position) in wanted]
        found = {dataset.sample_id_of(p) for p in positions}
        missing = wanted - found
        if missing:
            raise PostFailureDiagnosticsError(
                f"{len(missing)} requested source_dev sample_id(s) not found for the "
                f"reference benign population (first offenders: {sorted(missing)[:5]}); "
                "fail closed")
        before: dict[str, np.ndarray] = {}
        after: dict[str, dict[str, np.ndarray]] = {}
        with torch.no_grad():
            for position in positions:
                item = dataset.item(position)
                if item.label != LIVE:
                    raise PostFailureDiagnosticsError(
                        f"{item.sample_id!r} is not LIVE; the reference benign population "
                        "never forwards a spoof sample")
                clean_batch = collate_items([item]).to(trainer.device)
                before[item.sample_id] = forward_checkpoint_evidence(trainer.model, clean_batch)
                per_variant: dict[str, np.ndarray] = {}
                for variant in BENIGN_VARIANTS:
                    variant_image = benign_variant(item.image, variant, sample_id=item.sample_id,
                                                   noise_std=BENIGN_NOISE_STD)
                    variant_item = replace(item, image=variant_image)
                    variant_batch = collate_items([variant_item]).to(trainer.device)
                    per_variant[variant["name"]] = forward_checkpoint_evidence(
                        trainer.model, variant_batch)
                after[item.sample_id] = per_variant
        before_by_checkpoint.append(before)
        after_by_checkpoint.append(after)

    merged: dict[str, dict[str, Any]] = {}
    for sample_id in sample_ids:
        before_vectors = [d[sample_id] for d in before_by_checkpoint]
        per_variant_after: dict[str, np.ndarray] = {}
        for variant in BENIGN_VARIANTS:
            name = variant["name"]
            vectors = [d[sample_id][name] for d in after_by_checkpoint]
            per_variant_after[name] = average_checkpoint_evidence(vectors)
        merged[sample_id] = {"before": average_checkpoint_evidence(before_vectors),
                             "after_by_variant": per_variant_after}
    return merged


def reference_delta_plus_for_arm(repo: Path, arm: str, checkpoints: Sequence[Any],
                                 calibration_ids: Sequence[str]) -> list[float]:
    """Pooled `delta_plus = max(p_after - p_before, 0)` over every
    (calibration sample, reference variant) pair for one arm."""
    evidence = forward_reference_benign_evidence_for_arm(
        repo, checkpoints, sample_ids=calibration_ids)
    values: list[float] = []
    for sample_id in calibration_ids:
        before_p = float(evidence[sample_id]["before"][1])
        for vector in evidence[sample_id]["after_by_variant"].values():
            values.append(max(float(vector[1]) - before_p, 0.0))
    return values


def derive_reference_threshold(reference_delta_plus: Sequence[float]) -> dict[str, Any]:
    """`tau_mean = reference_mean + 3*reference_std`,
    `tau_tail = reference_p95 + 3*reference_std` — both computed ONLY from
    the pooled reference-variant `delta_plus` values on the calibration
    group, never from the tested corruption. Fails closed on an empty
    reference population."""
    values = np.asarray(reference_delta_plus, dtype=np.float64)
    if values.size == 0:
        raise PostFailureDiagnosticsError(
            "cannot derive a reference threshold from zero reference samples")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    p95 = float(np.percentile(values, 95))
    return {"reference_mean_delta_plus": mean, "reference_std_delta_plus": std,
           "reference_p95_delta_plus": p95, "reference_samples": int(values.size),
           "tau_mean": mean + 3.0 * std, "tau_tail": p95 + 3.0 * std,
           "formula_mean": "reference_mean_delta_plus + 3 * reference_std_delta_plus",
           "formula_tail": "reference_p95_delta_plus + 3 * reference_std_delta_plus"}


def corruption_verdict(evaluation_mean_delta_plus: float, evaluation_p95_delta_plus: float,
                       tau_mean: float, tau_tail: float) -> str:
    """`PASS` iff BOTH the evaluation mean and the evaluation p95 stay at or
    below their respective reference-derived tolerances — protects against a
    systematic mean increase AND a large upper-tail increase. Ties PASS on
    each (`<=`)."""
    mean_ok = float(evaluation_mean_delta_plus) <= float(tau_mean)
    tail_ok = float(evaluation_p95_delta_plus) <= float(tau_tail)
    return "PASS" if (mean_ok and tail_ok) else "FAIL"


def run_benign_corruption_diagnostic_for_arm(repo: Path, test_id: str, arm: str,
                                             checkpoints: Sequence[Any], *,
                                             evaluation_ids: Sequence[str],
                                             reference_threshold: Mapping[str, Any]
                                             ) -> dict[str, Any]:
    """One arm's full V2 benign-corruption diagnostic: forwards the
    EVALUATION group's `source_dev` LIVE samples, clean and corrupted,
    through all 5 of the arm's checkpoints; scores `delta_plus` against the
    ALREADY-DERIVED, disjoint reference threshold. Never forwards the
    tested corruption on the calibration group, and never derives a
    threshold from the tested corruption's own effect."""
    corruption_fn = CORRUPTION_FUNCTIONS[test_id]
    evidence = forward_corruption_evidence_for_arm(
        repo, checkpoints, sample_ids=evaluation_ids, corruption_fn=corruption_fn)

    def _p_global(vector: np.ndarray) -> float:
        return float(vector[1])

    delta_plus = [max(_p_global(evidence[sid]["after"]) - _p_global(evidence[sid]["before"]), 0.0)
                 for sid in evaluation_ids]
    values = np.asarray(delta_plus, dtype=np.float64)
    evaluation_mean = float(values.mean())
    evaluation_p95 = float(np.percentile(values, 95))
    verdict = corruption_verdict(evaluation_mean, evaluation_p95,
                                 reference_threshold["tau_mean"], reference_threshold["tau_tail"])
    return {"arm": arm, "test_id": test_id, "reference_threshold": dict(reference_threshold),
           "evaluation": {"samples": int(values.size), "mean_delta_plus": evaluation_mean,
                          "p95_delta_plus": evaluation_p95},
           "verdict": verdict}


__all__ = [
    "DIAGNOSTICS_DIR", "PROTOCOL_BINDING_PATH", "POPULATION_BINDING_PATH",
    "CHECKPOINT_BINDING_PATH", "RESULT_PATH", "PER_TEST_PATH", "PROVENANCE_PATH",
    "VERDICT_PATH", "BINDING_ARTIFACT_PATHS", "RESULT_ARTIFACT_PATHS",
    "PROTOCOL_CONFIG_PATH", "C8_ACCEPTANCE_PATH", "EXECUTABLE_TESTS",
    "BENIGN_CORRUPTION_TESTS", "ALL_TESTS", "DOMAINS",
    "load_protocol", "protocol_identity", "active_protocol_identity",
    "bind_c8_matrix_identity", "verify_per_domain_group_safety",
    "forward_reference_benign_evidence_for_arm", "reference_delta_plus_for_arm",
    "derive_reference_threshold", "corruption_verdict",
    "run_benign_corruption_diagnostic_for_arm",
    # re-exported, reused-verbatim V1 primitives (never reimplemented)
    "jpeg_corrupt", "resize_corrupt", "color_corrupt", "CORRUPTION_FUNCTIONS",
    "calibration_evaluation_split", "resolve_source_dev_live_records",
    "resolve_synthetic_population_by_route", "run_cross_route_diagnostic_for_arm",
    "validate_existing_diagnostics_result",
]


# ==============================================================================
# 5. Existing-result validation (Defect C)
# ==============================================================================

def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def validate_existing_diagnostics_result(repo: Path) -> dict[str, Any]:
    """Canonical, comprehensive validation of an on-disk V2 diagnostics
    result set — used by BOTH `--execute` (second and later calls) and
    `--status`. NEVER re-runs a forward pass, probe fit or corruption; only
    re-derives cheap identities and recomputes the overall verdict from the
    RECORDED per-test statuses.

    Returns `{"valid": bool, "problems": [...], "docs": {...}}`. `problems`
    is empty iff `valid`. A complete-but-invalid result set must never be
    silently recomputed or overwritten by its caller.
    """
    problems: list[str] = []
    repo = Path(repo)

    binding_docs: dict[str, Any] = {}
    for name, relative in BINDING_ARTIFACT_PATHS.items():
        doc = _read_json_or_none(repo / relative)
        if doc is None:
            problems.append(f"binding artifact {name!r} ({relative}) is missing or unparseable")
        binding_docs[name] = doc

    result_docs: dict[str, Any] = {}
    for name, relative in RESULT_ARTIFACT_PATHS.items():
        doc = _read_json_or_none(repo / relative)
        if doc is None:
            problems.append(f"result artifact {name!r} ({relative}) is missing or unparseable")
        result_docs[name] = doc

    if problems:
        return {"valid": False, "problems": problems, "docs": {**binding_docs, **result_docs}}

    result, per_test_doc = result_docs["result"], result_docs["per_test"]
    provenance, verdict_doc = result_docs["provenance"], result_docs["verdict"]
    protocol_binding = binding_docs["protocol"]
    population_binding = binding_docs["population"]
    checkpoint_binding = binding_docs["checkpoint"]

    # C. active V2 protocol identity must match every bound/recorded identity.
    try:
        active_id = active_protocol_identity(repo)
    except PostFailureDiagnosticsError as error:
        return {"valid": False, "problems": [f"active protocol unresolvable: {error}"],
               "docs": {**binding_docs, **result_docs}}

    for name, doc in (("protocol_binding", protocol_binding),
                      ("population_binding", population_binding),
                      ("checkpoint_binding", checkpoint_binding),
                      ("result", result), ("per_test", per_test_doc),
                      ("provenance", provenance), ("verdict", verdict_doc)):
        if doc.get("protocol_identity") != active_id:
            problems.append(f"{name}.protocol_identity does not match the active V2 protocol identity")

    # A. the four result artifacts must agree with each other, not just parse.
    if result.get("per_test") != per_test_doc.get("per_test"):
        problems.append("result.per_test does not match per_test.per_test — a tampered or "
                        "diverged artifact")

    # B. bindings cross-reference each other consistently.
    if protocol_binding.get("checkpoint_binding_identity") != \
            checkpoint_binding.get("checkpoint_binding_identity_sha256"):
        problems.append("protocol_binding.checkpoint_binding_identity does not match "
                        "checkpoint_binding's own identity")
    if protocol_binding.get("population_binding_identity") != \
            population_binding.get("population_binding_identity_sha256"):
        problems.append("protocol_binding.population_binding_identity does not match "
                        "population_binding's own identity")

    # D/F. checkpoint hashes/cardinality/seeds and C6 bank identities, re-derived.
    try:
        from prism_fas.evaluation.synthetic_real_probe import ARMS
        from prism_fas.pipeline.adapters import sources

        inputs = sources.verify_detector_inputs(repo, arms=ARMS)
        if checkpoint_binding.get("source_package_identity") != inputs["package_identity"]:
            problems.append("checkpoint_binding.source_package_identity does not match the "
                            "currently resolvable source package identity")
        current_bank_identities = {arm: str(inputs["c6"]["banks"][arm]["selected_set_sha256"])
                                  for arm in ARMS}
        if dict(checkpoint_binding.get("c6_bank_identities") or {}) != current_bank_identities:
            problems.append("checkpoint_binding.c6_bank_identities does not match the currently "
                            "resolvable C6 bank identities")
        checkpoints = list(checkpoint_binding.get("checkpoints") or [])
        by_arm: dict[str, list[Any]] = {arm: [] for arm in ARMS}
        for item in checkpoints:
            by_arm.setdefault(str(item.get("arm")), []).append(item)
        for arm in ARMS:
            if len(by_arm.get(arm, [])) != 5:
                problems.append(f"checkpoint_binding has {len(by_arm.get(arm, []))} checkpoints "
                                f"for arm {arm!r}, expected 5")
            seeds = sorted(int(item["seed"]) for item in by_arm.get(arm, []) if "seed" in item)
            if seeds != sorted({20260806, 20260807, 20260808, 20260809, 20260810}):
                problems.append(f"checkpoint_binding seeds for arm {arm!r} do not match the "
                                "frozen 5-seed family")
    except Exception as error:                            # noqa: BLE001
        problems.append(f"could not re-derive checkpoint/source identities: "
                        f"{type(error).__name__}: {error}")

    # D. C8 matrix identity, re-derived and cross-checked.
    try:
        current_c8 = bind_c8_matrix_identity(repo)
        if provenance.get("c8_matrix_identity") != current_c8["c8_matrix_identity"]:
            problems.append("provenance.c8_matrix_identity does not match the currently "
                            "resolvable canonical C8 matrix identity")
    except PostFailureDiagnosticsError as error:
        problems.append(f"C8 matrix identity could not be re-verified: {error}")

    # G. population identities, re-derived byte-for-byte.
    protocol: dict[str, Any] | None = None
    try:
        protocol = load_protocol(repo)
        live_records = resolve_source_dev_live_records(repo)
        group_ids = sorted({r["stable_group_identity"] for r in live_records})
        shared = protocol["benign_corruption_shared"]
        split = calibration_evaluation_split(
            group_ids, namespace=shared["split_hash_namespace"], seed=int(shared["split_seed"]))
        verify_per_domain_group_safety(live_records, split, domains=DOMAINS)
        calibration_groups, evaluation_groups = set(split["calibration"]), set(split["evaluation"])
        recomputed_calibration_ids = sorted(
            r["sample_id"] for r in live_records if r["stable_group_identity"] in calibration_groups)
        recomputed_evaluation_ids = sorted(
            r["sample_id"] for r in live_records if r["stable_group_identity"] in evaluation_groups)
        bound_population = population_binding.get("benign_corruption") or {}
        if list(bound_population.get("calibration_sample_ids") or []) != recomputed_calibration_ids:
            problems.append("population_binding calibration_sample_ids no longer match the "
                            "recomputable population identity")
        if list(bound_population.get("evaluation_sample_ids") or []) != recomputed_evaluation_ids:
            problems.append("population_binding evaluation_sample_ids no longer match the "
                            "recomputable population identity")
    except PostFailureDiagnosticsError as error:
        problems.append(f"population identity could not be re-derived: {error}")

    # H. executable/blocked test set matches the frozen V2 protocol exactly.
    if protocol is None:
        try:
            protocol = load_protocol(repo)
        except PostFailureDiagnosticsError:
            protocol = None
    per_test = dict((per_test_doc or {}).get("per_test") or {})
    if protocol is not None:
        declared_executable = {test_id for test_id, cfg in protocol["tests"].items()
                              if cfg["classification"] == "EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL"}
        if declared_executable != set(EXECUTABLE_TESTS):
            problems.append("frozen protocol's executable test set no longer matches "
                            "post_failure_diagnostics_v2.EXECUTABLE_TESTS")
        if set(per_test) != set(ALL_TESTS):
            problems.append("per_test does not cover exactly the eight declared tests")
        for test_id, cfg in protocol["tests"].items():
            if cfg["classification"] != "EXECUTABLE_WITH_NEW_FROZEN_PROTOCOL":
                recorded = per_test.get(test_id) or {}
                if recorded.get("status") != "BLOCKED" or \
                        recorded.get("classification") != cfg["classification"]:
                    problems.append(f"{test_id}: recorded per_test entry no longer matches its "
                                    "frozen BLOCKED classification")

    # J. per-test verdict consistency, recomputed from recorded per-arm verdicts only.
    for test_id in EXECUTABLE_TESTS:
        entry = per_test.get(test_id) or {}
        per_arm = dict(entry.get("per_arm") or {})
        if not per_arm:
            problems.append(f"{test_id}: no per_arm verdicts recorded")
            continue
        recomputed_status = "PASS" if all(
            per_arm.get(arm, {}).get("verdict") == "PASS" for arm in ("RND", "DET", "LLM")) else "FAIL"
        if entry.get("status") != recomputed_status:
            problems.append(f"{test_id}: recorded status {entry.get('status')!r} does not match "
                            f"the status recomputed from its own recorded per-arm verdicts "
                            f"({recomputed_status!r})")

    # K. overall verdict recomputed from RECORDED per-test statuses only.
    executed_statuses = [per_test.get(t, {}).get("status") for t in EXECUTABLE_TESTS]
    recomputed_overall = "PASS" if all(status == "PASS" for status in executed_statuses) else "FAIL"
    if verdict_doc.get("overall_diagnostics_verdict") != recomputed_overall:
        problems.append(f"verdict.overall_diagnostics_verdict "
                        f"({verdict_doc.get('overall_diagnostics_verdict')!r}) does not match the "
                        f"value recomputed from recorded per-test statuses "
                        f"({recomputed_overall!r})")

    # L. c9_may_close is false everywhere it appears.
    for name, doc in (("verdict", verdict_doc), ("result", result)):
        if "c9_may_close" in doc and doc["c9_may_close"] is not False:
            problems.append(f"{name}.c9_may_close is not False")

    # M. BA_sep observed verdict must remain FAIL everywhere.
    for name, doc in result_docs.items():
        if doc.get("ba_sep_observed_verdict") != "FAIL":
            problems.append(f"{name}.ba_sep_observed_verdict is not 'FAIL'")
        if doc.get("detector_reliability_lock_c_observed_overall") != "FAILED":
            problems.append(f"{name}.detector_reliability_lock_c_observed_overall is not 'FAILED'")

    # N. target_access must be 0 everywhere it appears.
    for name, doc in {**binding_docs, **result_docs}.items():
        if "target_access" in doc and int(doc["target_access"]) != 0:
            problems.append(f"{name}.target_access is not 0")

    # O. the CURRENT (live, on this host) DETECTOR_RELIABILITY_LOCK_C must still be FAILED.
    from prism_fas.evaluation import detector_reliability

    live_lock = _read_json_or_none(repo / detector_reliability.LOCK_PATH)
    if live_lock is not None and live_lock.get("overall") != "FAILED":
        problems.append("the live DETECTOR_RELIABILITY_LOCK_C on this host is no longer FAILED; "
                        "fail closed rather than re-report a stale result")

    return {"valid": not problems, "problems": problems, "docs": {**binding_docs, **result_docs}}
