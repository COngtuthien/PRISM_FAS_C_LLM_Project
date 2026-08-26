"""The `synthetic_vs_real_spoof_probe` reliability test, under the frozen,
user-approved `C9_DETECTOR_BA_SEP_OPTION1_V2` protocol.

Option 1 (`reports/readiness/C9_DETECTOR_RELIABILITY_DECISION_DOSSIER.md`,
`reports/readiness/C9_BA_SEP_OPTION1_PROTOCOL_FREEZE.md`): a common Track-G
decision-evidence representation — `[global_logit_G, p_global]` — because
Track G is the only frozen Version-C primary detector representation that
exists for all three RND/DET/LLM arms under the same architecture, and it is
the only representation available from the 42 existing C8 rows without new
training.

V2 (`reports/readiness/C9_BA_SEP_OPTION1_V2_PREEXECUTION_CORRECTION.md`)
supersedes V1 with a PRE-EXECUTION correction only: V1 declared its matched
source split `group_safe: true` while actually partitioning on the SAMPLE
identity (`sample_id` / `synthetic_id`), not the underlying source record —
so samples derived from the same source video/record could straddle probe
train and probe validation. V2 separates SAMPLE identity (what gets
selected, via `PopulationRecord.sample_identity`) from GROUP identity (what
the split partitions on, via `PopulationRecord.stable_group_identity`,
always `source_record_id`) and closes the leak. No BA_sep value was ever
observed under V1 or before the V2 freeze. The frozen protocol config is
`configs/evaluation/c9_detector_ba_sep_option1_v2.yaml`
(`prism_fas.evaluation.detector_reliability.PROBE_PROTOCOL_CONFIG_PATH`).

**THIS MODULE MUST NOT BE CALLED WITH REAL SCIENTIFIC DATA IN A CONTEXT THAT
HAS NOT EXPLICITLY AUTHORIZED A SCIENTIFIC BA_sep RUN.** Every function
through `resolve_checkpoint_set`/`resolve_arm_populations` is read-only
metadata resolution — it reuses the exact canonical readers C7/C8 already use
(`source_matrix.build_plan`, `source_evidence.load_row_evidence`,
`c6_evidence.verify_c6_evidence`, `sources.verify_detector_inputs`,
`c6_bank.open_arm_bank`) and never opens an image or a checkpoint's weights.
`forward_checkpoint_evidence` is a pure function over an ALREADY-BUILT model
and batch and is safe to unit-test with a fake model.

`execute_joint_probe` (see `reports/readiness/C9_BA_SEP_OPTION1_V2_RUNNER_INTEGRATION_FIX.md`)
is the SANCTIONED, REAL, joint (all-three-arm) execution path — it strict-loads
real checkpoints through `construct_row_trainer` (the exact C8 row-construction
path, including C8's own scientific device resolver,
`pipeline.adapters.c7._scientific_device` — never CPU-hard-coded, never a
duplicated CUDA policy) and forwards real evidence through
`forward_evidence_for_records` (batched exactly like C8's own cross-source
evaluation). It is reached only through
`prism_fas.evaluation.synthetic_real_probe_runner --execute`, which requires
a prior successful `--bind-only` bound to the currently active protocol
identity. The old single-arm `run_scientific_probe(repo, arm)` entry point is
RETIRED: the frozen protocol balances jointly across RND/DET/LLM, so no
single arm may ever be probed in isolation — it raises
`SyntheticRealProbeError` rather than computing a partial result. Nothing in
this repository's test suite calls `execute_joint_probe` against real data;
every test exercises it with a monkeypatched `construct_row_trainer`/
`forward_evidence_for_records` boundary and fixture evidence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: §18.1/§18.3: the three treatment arms every Track-G primary row covers.
ARMS: tuple[str, ...] = ("RND", "DET", "LLM")

#: One checkpoint per C8 training seed, per arm, from the P3-ready Track-G rows.
CHECKPOINTS_PER_ARM = 5
TOTAL_CHECKPOINTS = 15

#: The frozen Option-1 evidence vector. Exactly these two fields, nothing else.
EVIDENCE_FIELDS: tuple[str, ...] = ("global_logit_G", "p_global")
EVIDENCE_DIMENSION = 2

#: Fields the protocol explicitly forbids adding to the evidence vector
#: (§2 of the freeze task). Listed so a static test can assert none of them
#: is ever read by `forward_checkpoint_evidence`.
FORBIDDEN_EVIDENCE_FIELDS: tuple[str, ...] = (
    "s_region", "region_distances", "local_logits", "region_embeddings",
    "p_prompt_spoof", "z_global", "generator_identity", "route_identity",
    "quality_weight", "recipe_metadata",
)

TRAIN_LABEL, VALIDATION_LABEL = "train", "validation"
REAL_SPOOF_CLASS, SYNTHETIC_SPOOF_CLASS = 0, 1


class SyntheticRealProbeError(RuntimeError):
    """The Option-1 probe cannot proceed with the inputs given."""


# ==============================================================================
# 1. Protocol
# ==============================================================================

def load_protocol(repo: Path) -> dict[str, Any]:
    """The frozen Option-1 protocol, or a refusal naming why it is absent."""
    from prism_fas.evaluation import detector_reliability

    protocol = detector_reliability.load_probe_protocol(repo)
    if protocol is None:
        raise SyntheticRealProbeError(
            "C9_DETECTOR_BA_SEP_OPTION1_V2 is not frozen "
            f"(expected {detector_reliability.PROBE_PROTOCOL_CONFIG_PATH} to exist, "
            "declare status: FROZEN_NOT_RUN, and carry every "
            "PROBE_PROTOCOL_REQUIRED_FIELDS entry)")
    return protocol


def protocol_identity(repo: Path) -> str:
    """sha256 over every result-affecting protocol field. Changes if and only
    if a result-affecting field of the frozen config changes."""
    from prism_fas.evaluation import detector_reliability

    return detector_reliability.protocol_identity(load_protocol(repo))


# ==============================================================================
# 2. Checkpoint binding — resolved from real C8 manifests, never chosen
# ==============================================================================

#: The single decision logit every bound checkpoint must declare. Refused,
#: not silently accepted, if a row's manifest names a different one.
REQUIRED_DECISION_LOGIT_NAME = "global_logit_G"


@dataclass(frozen=True)
class CheckpointBinding:
    """One resolved, hash-verified P3-ready Track-G checkpoint."""

    arm: str
    seed: int
    row_id: str
    run_identity: str
    config_identity: str
    checkpoint_sha256: str
    checkpoint_path: str
    checkpoint_kind: str
    decision_logit_name: str
    decision_graph_hash: str


def track_g_p3_rows(arm: str) -> list[Any]:
    """The exact five preregistered P3-ready Track-G rows for one arm, in
    seed order. No P1/P2 row and no other arm is ever included."""
    from prism_fas.evaluation.source_matrix import build_plan

    if arm not in ARMS:
        raise SyntheticRealProbeError(f"unknown arm {arm!r}; the protocol covers {ARMS}")
    plan = build_plan()
    rows = [row for row in plan.rows
            if row.track == "G" and row.protocol == "P3" and row.arm == arm]
    rows.sort(key=lambda row: row.seed)
    if len(rows) != CHECKPOINTS_PER_ARM:
        raise SyntheticRealProbeError(
            f"expected {CHECKPOINTS_PER_ARM} P3-ready Track-G rows for arm {arm!r}, "
            f"found {len(rows)}; the C8 matrix plan may have drifted")
    return rows


def resolve_checkpoint_set(repo: Path, arm: str) -> list[CheckpointBinding]:
    """The five real, hash-verified C8 checkpoints for one arm's P3-ready
    Track-G rows. Reuses `source_evidence.load_row_evidence` — the same
    manifest-reading, byte-verifying reader C9 itself uses — rather than a
    second implementation. Refuses (raises) unless all five are present,
    PASS, byte-verified, AND declare the required decision logit; never
    falls back to a partial set and never selects among available
    checkpoints by any criterion.

    `RowEvidence` (the shared C9 evidence contract) does not carry the
    checkpoint's own relative path or its `kind` (which checkpoint.pt the row
    actually saved — never chosen here by any metric) — reading
    those, plus `decision_graph_hash`, off the SAME already-hash-verified
    manifest `load_row_evidence` opened is completing what that shared
    contract deliberately leaves out, not a second implementation of it.
    """
    import json

    from prism_fas.evaluation import source_evidence
    from prism_fas.evaluation.source_matrix import build_plan

    plan = build_plan()
    target_rows = {row.row_id: row for row in track_g_p3_rows(arm)}
    evidence, problems = source_evidence.load_row_evidence(repo, plan)
    by_row_id = {item.row_id: item for item in evidence}

    bindings: list[CheckpointBinding] = []
    missing: list[str] = []
    for row_id in sorted(target_rows, key=lambda name: target_rows[name].seed):
        row = target_rows[row_id]
        item = by_row_id.get(row_id)
        if item is None or item.status != "PASS" or not item.checkpoint_sha256:
            missing.append(row_id)
            continue
        if item.decision_logit_name != REQUIRED_DECISION_LOGIT_NAME:
            missing.append(row_id)
            continue
        manifest_path = source_evidence.row_directory(
            Path(repo) / source_evidence.C8_RUNS, row) / source_evidence.RUN_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_meta = dict(manifest.get("checkpoint") or {})
        checkpoint_path = str(checkpoint_meta.get("path") or "")
        checkpoint_kind = str(checkpoint_meta.get("kind") or "")
        if not checkpoint_path or not checkpoint_kind:
            missing.append(row_id)
            continue
        bindings.append(CheckpointBinding(
            arm=arm, seed=row.seed, row_id=row_id, run_identity=item.run_identity,
            config_identity=item.config_identity,
            checkpoint_sha256=str(item.checkpoint_sha256),
            checkpoint_path=checkpoint_path, checkpoint_kind=checkpoint_kind,
            decision_logit_name=item.decision_logit_name,
            decision_graph_hash=str(manifest.get("decision_graph_hash") or "")))
    if missing or len(bindings) != CHECKPOINTS_PER_ARM:
        raise SyntheticRealProbeError(
            f"arm {arm!r}: {len(bindings)}/{CHECKPOINTS_PER_ARM} P3-ready Track-G "
            f"checkpoints resolved; missing/invalid: {missing or [r for r in target_rows if r not in by_row_id]}. "
            f"reader problems: {problems[:5]}")
    return bindings


def resolve_all_checkpoint_sets(repo: Path) -> dict[str, list[CheckpointBinding]]:
    """All 15 checkpoints, across all three arms. Raises unless every one
    resolves — the protocol requires the full set before the first probe
    seed runs, not a per-arm-as-available subset."""
    resolved = {arm: resolve_checkpoint_set(repo, arm) for arm in ARMS}
    total = sum(len(items) for items in resolved.values())
    if total != TOTAL_CHECKPOINTS:
        raise SyntheticRealProbeError(
            f"resolved {total}/{TOTAL_CHECKPOINTS} checkpoints across all arms")
    return resolved


#: Excludes its OWN field too: once written to disk and read back, the
#: binding dict contains `checkpoint_binding_identity_sha256` itself: without
#: excluding it here, re-verifying an already-bound artifact would compute a
#: DIFFERENT hash than the one originally stored (self-referential material),
#: and every legitimate re-verification (`execute_joint_probe`, a repeated
#: `--bind-only`) would wrongly report a mismatch.
_BINDING_IDENTITY_EXCLUDED_KEYS = frozenset({
    "bound_at_utc", "checkpoint_binding_identity_sha256"})


def checkpoint_binding_identity(binding: Mapping[str, Any]) -> str:
    """sha256 over the checkpoint binding, sorted keys, no timestamp."""
    material = {key: value for key, value in binding.items()
               if key not in _BINDING_IDENTITY_EXCLUDED_KEYS}
    return hashlib.sha256(json.dumps(
        material, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()


def build_checkpoint_binding(repo: Path) -> dict[str, Any]:
    """The one joint, atomic, all-three-arm checkpoint binding (§3, §6A of
    the runner-integration-fix task): all 15 real, hash-verified P3-ready
    Track-G checkpoints, bound together with the source package identity and
    all three C6 arm bank identities. Raises (fails closed) unless every one
    of the 15 resolves. No performance metric enters this binding — it is
    pure identity, resolved before any evidence is ever forwarded.
    """
    from prism_fas.pipeline.adapters import sources

    inputs = sources.verify_detector_inputs(repo, arms=ARMS)
    by_arm = resolve_all_checkpoint_sets(repo)
    bank_identities = {arm: str(inputs["c6"]["banks"][arm]["selected_set_sha256"])
                       for arm in ARMS}

    checkpoints = [
        {"arm": binding.arm, "seed": binding.seed, "row_id": binding.row_id,
         "run_identity": binding.run_identity, "config_identity": binding.config_identity,
         "checkpoint_relative_path": binding.checkpoint_path,
         "checkpoint_sha256": binding.checkpoint_sha256,
         "decision_graph_hash": binding.decision_graph_hash,
         "decision_logit_name": binding.decision_logit_name}
        for arm in ARMS for binding in by_arm[arm]]

    counts = {arm: sum(1 for item in checkpoints if item["arm"] == arm) for arm in ARMS}
    if any(counts[arm] != CHECKPOINTS_PER_ARM for arm in ARMS) or len(checkpoints) != TOTAL_CHECKPOINTS:
        raise SyntheticRealProbeError(
            f"joint checkpoint binding requires exactly {CHECKPOINTS_PER_ARM} per arm "
            f"and {TOTAL_CHECKPOINTS} total; resolved {counts}")

    binding = {
        "schema_version": "c9-ba-sep-execution-binding-v1",
        "protocol_identity": protocol_identity(repo),
        "source_package_identity": inputs["package_identity"],
        "c6_bank_identities": bank_identities,
        "checkpoints": checkpoints,
        "checkpoints_per_arm": {arm: counts[arm] for arm in ARMS},
        "total_checkpoints": len(checkpoints),
        "target_access": 0,
    }
    binding["checkpoint_binding_identity_sha256"] = checkpoint_binding_identity(binding)
    return binding


# ==============================================================================
# 3b. Joint (all-three-arm) population resolution and the pre-selected plan
#     (§4, §5, §6B of the runner-integration-fix task)
# ==============================================================================

def resolve_joint_populations(repo: Path) -> tuple[list[PopulationRecord],
                                                     dict[str, list[PopulationRecord]]]:
    """`(real_spoof, {arm: synthetic_spoof})` — the real population resolved
    ONCE and shared, the synthetic population resolved once per arm. Never a
    per-arm real resolution: there is exactly one real_spoof_population."""
    real_spoof = resolve_real_spoof_population(repo)
    synthetic_by_arm = {arm: resolve_synthetic_population(repo, arm) for arm in ARMS}
    return real_spoof, synthetic_by_arm


_PLAN_IDENTITY_EXCLUDED_KEYS = frozenset({"bound_at_utc", "population_plan_identity_sha256"})


def population_plan_identity(plan: Mapping[str, Any]) -> str:
    """sha256 over the population plan, sorted keys, no timestamp — and
    excluding its own field, for the same self-reference reason
    `checkpoint_binding_identity` does (see `_BINDING_IDENTITY_EXCLUDED_KEYS`).
    """
    material = {key: value for key, value in plan.items()
               if key not in _PLAN_IDENTITY_EXCLUDED_KEYS}
    return hashlib.sha256(json.dumps(
        material, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")).hexdigest()


def build_population_plan(repo: Path, *, protocol: Mapping[str, Any] | None = None
                          ) -> dict[str, Any]:
    """The one joint, atomic, all-three-arm pre-selected population plan.

    For every `(probe_seed, source_domain, split)` cell: an independent
    group-safe split of the real population and each arm's synthetic
    population, followed by ONE joint balancing call across all three arms
    (`balance_report`, `synthetic_by_arm={"RND": ..., "DET": ..., "LLM": ...}`)
    — never a per-arm balance with the other two arms filled with `[]`, which
    is exactly the defect this task fixes (an empty arm forces `N=0`).

    Fails closed (raises) if any required cell resolves `N == 0`, if a
    group-safety check fails, or if the real subset selected for one arm
    differs from the real subset selected for another arm in the same cell
    (they MUST be identical: `balance_classes` orders and truncates the SAME
    real list once per cell, shared across arms by construction, but this is
    asserted again explicitly rather than trusted silently).
    """
    resolved_protocol = protocol if protocol is not None else load_protocol(repo)
    protocol_id = protocol_identity(repo)
    namespace = resolved_protocol["matched_source_split"]["split_hash_namespace"]
    probe_seeds = list(resolved_protocol["probe_seed_values"])
    domains = list(resolved_protocol["source_domains"])

    real_spoof, synthetic_by_arm = resolve_joint_populations(repo)

    cells: list[dict[str, Any]] = []
    for seed in probe_seeds:
        real_split = assign_splits(real_spoof, namespace=namespace, probe_seed=seed)
        verify_group_safe_split(real_split)
        synth_split_by_arm: dict[str, dict[str, list[PopulationRecord]]] = {}
        for arm in ARMS:
            split = assign_splits(synthetic_by_arm[arm], namespace=namespace, probe_seed=seed)
            verify_group_safe_split(split)
            synth_split_by_arm[arm] = split

        for domain in domains:
            for split_label in (TRAIN_LABEL, VALIDATION_LABEL):
                real_cell = [r for r in real_split[split_label] if r.source_domain == domain]
                synthetic_cell = {
                    arm: [r for r in synth_split_by_arm[arm][split_label]
                          if r.source_domain == domain]
                    for arm in ARMS}
                report = balance_report(
                    protocol_id=protocol_id, probe_seed=seed, split=split_label,
                    source_domain=domain, real_spoof=real_cell,
                    synthetic_by_arm=synthetic_cell)

                if report["n"] == 0:
                    raise SyntheticRealProbeError(
                        f"population plan cell (seed={seed}, domain={domain!r}, "
                        f"split={split_label!r}) resolves N=0; refusing to freeze an "
                        "empty scientific probe population. pre-balance counts: "
                        f"{report['pre_balance_counts']}")

                real_ids = sorted(r.sample_identity for r in report["selected_real"])
                selected_by_arm = {arm: sorted(r.sample_identity for r in
                                               report["selected_synthetic"][arm])
                                   for arm in ARMS}
                if any(len(selected_by_arm[arm]) != len(real_ids) for arm in ARMS):
                    raise SyntheticRealProbeError(
                        f"cell (seed={seed}, domain={domain!r}, split={split_label!r}): "
                        "a synthetic arm's selected count does not match the real "
                        "count; the balancing rule is 1:1 real:synthetic per arm")

                cells.append({
                    "probe_seed": seed, "source_domain": domain, "split": split_label,
                    "n": report["n"],
                    "pre_balance_counts": report["pre_balance_counts"],
                    "post_balance_counts": report["post_balance_counts"],
                    "unique_source_record_id_counts": report["unique_source_record_id_counts"],
                    "real_selected": [
                        {"sample_identity": r.sample_identity,
                         "stable_group_identity": r.stable_group_identity}
                        for r in sorted(report["selected_real"], key=lambda r: r.sample_identity)],
                    "synthetic_selected": {
                        arm: [{"sample_identity": r.sample_identity,
                              "stable_group_identity": r.stable_group_identity}
                             for r in sorted(report["selected_synthetic"][arm],
                                             key=lambda r: r.sample_identity)]
                        for arm in ARMS},
                })

    leakage_audit = _population_plan_leakage_audit(cells)
    if leakage_audit["leaked"]:
        raise SyntheticRealProbeError(
            f"population plan leakage audit failed: {leakage_audit['leaked']}")

    plan = {
        "schema_version": "c9-ba-sep-population-plan-v1",
        "protocol_identity": protocol_id,
        "split_hash_namespace": namespace,
        "probe_seed_values": probe_seeds,
        "source_domains": domains,
        "cells": cells,
        "leakage_audit": leakage_audit,
        "target_access": 0,
    }
    plan["population_plan_identity_sha256"] = population_plan_identity(plan)
    return plan


def _population_plan_leakage_audit(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """For every probe seed, the union of every TRAIN-split stable group
    identity (real and synthetic, every arm) must share nothing with the
    union of every VALIDATION-split stable group identity. Computed once
    more here, over the assembled plan, on top of the per-population
    `verify_group_safe_split` checks already performed while building it."""
    by_seed: dict[int, dict[str, set[str]]] = {}
    for cell in cells:
        seed = cell["probe_seed"]
        bucket = by_seed.setdefault(seed, {TRAIN_LABEL: set(), VALIDATION_LABEL: set()})
        groups = {entry["stable_group_identity"] for entry in cell["real_selected"]}
        for arm in ARMS:
            groups |= {entry["stable_group_identity"] for entry in cell["synthetic_selected"][arm]}
        bucket[cell["split"]] |= groups

    leaked: dict[str, list[str]] = {}
    for seed, bucket in by_seed.items():
        overlap = bucket[TRAIN_LABEL] & bucket[VALIDATION_LABEL]
        if overlap:
            leaked[str(seed)] = sorted(overlap)[:10]
    return {"checked_seeds": sorted(by_seed), "leaked": leaked}


# ==============================================================================
# 3. Populations — traced through the exact canonical readers C7/C8 use
# ==============================================================================

@dataclass(frozen=True)
class PopulationRecord:
    """One candidate sample's identity, domain and class, before any pixel
    is read. `label` is `REAL_SPOOF_CLASS` or `SYNTHETIC_SPOOF_CLASS`.

    Two DISTINCT identities, deliberately kept apart (the V2 pre-execution
    correction, `reports/readiness/C9_BA_SEP_OPTION1_V2_PREEXECUTION_CORRECTION.md`):

    `sample_identity` — the individual sample/candidate's own identity
        (`sample_id` for real rows, `synthetic_id` for synthetic rows). Used
        ONLY for deterministic selection order (`balance_classes`); never for
        the train/validation split.

    `stable_group_identity` — the underlying `source_record_id` of the real
        source video/record this sample is derived from (directly, for real
        rows; via `live_target_sample_id` for synthetic rows). Used ONLY for
        the train/validation split (`split_bucket`/`assign_splits`); using
        `sample_identity` there would let two samples sharing one source
        record straddle the split — exactly the V1 defect V2 corrects.
    """

    sample_identity: str
    stable_group_identity: str
    source_domain: str
    label: int


def _source_train_rows(repo: Path) -> Sequence[Mapping[str, Any]]:
    """The raw `source_train` manifest rows, via the exact canonical loader
    C8/M9Trainer itself uses (`configs/data/loader_m4.yaml`,
    `src/prism_fas/pipeline/adapters/c8.py:1361`) — never a second loader
    configuration invented for this module."""
    from prism_fas.data.loader.config import TRAINING_SPLIT, load_loader_config
    from prism_fas.data.loader.loose_dataset import CanonicalPackageDataset
    from prism_fas.pipeline.adapters import sources

    inputs = sources.verify_detector_inputs(repo)
    package_root = Path(repo) / inputs["package_root"]
    loader_config = load_loader_config(Path(repo) / "configs/data/loader_m4.yaml")
    dataset = CanonicalPackageDataset(package_root, TRAINING_SPLIT, loader_config, mode="training")
    return dataset.index.rows


def _source_record_id_by_sample_id(repo: Path, *,
                                   domains: Sequence[str] = ("casia_fasd", "msu_mfsd")
                                   ) -> dict[str, str]:
    """`sample_id -> source_record_id` for every `source_train` row in
    `domains`, real or spoof, live or not — the synthetic population's
    `live_target_sample_id` names a LIVE real sample, not a spoof one, so
    this lookup is built over the full unfiltered manifest, not just the
    spoof subset `resolve_real_spoof_population` returns.

    Fail-closed: a `sample_id` with an empty/missing `source_record_id`, or
    a `sample_id` that maps to more than one DISTINCT `source_record_id`
    (a manifest inconsistency), raises rather than being silently dropped
    or silently resolved to either value.
    """
    lookup: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in _source_train_rows(repo):
        if row["dataset"] not in domains:
            continue
        sample_id = str(row["sample_id"])
        source_record_id = str(row.get("source_record_id") or "").strip()
        if not source_record_id:
            raise SyntheticRealProbeError(
                f"source_train sample_id={sample_id!r} has no source_record_id; "
                "fail closed rather than treat an unresolvable group identity as safe")
        existing = lookup.get(sample_id)
        if existing is not None and existing != source_record_id:
            ambiguous.add(sample_id)
        lookup[sample_id] = source_record_id
    if ambiguous:
        raise SyntheticRealProbeError(
            f"sample_id -> source_record_id is ambiguous for {sorted(ambiguous)}; "
            "fail closed rather than pick one source_record_id arbitrarily")
    return lookup


def resolve_real_spoof_population(repo: Path, *,
                                  domains: Sequence[str] = ("casia_fasd", "msu_mfsd")
                                  ) -> list[PopulationRecord]:
    """Every `source_train` real-spoof row's identities and domain.

    Reads only the package manifest (`CanonicalPackageDataset.index.rows`,
    the same rows `M9TrainingDataset._real_pools` reads) — never an image.
    `label_live_spoof == "spoof"` only; `source_dev` and any target split
    are unreachable from this function by construction (`assert_source_only`
    in `detector.dataset` refuses them upstream). `stable_group_identity` is
    `source_record_id`, read directly off the manifest row; fail closed if
    empty (`_source_record_id_by_sample_id` raises before this can happen,
    since it is built over the same rows).
    """
    lookup = _source_record_id_by_sample_id(repo, domains=domains)
    records: list[PopulationRecord] = []
    for row in _source_train_rows(repo):
        if row["dataset"] not in domains or row["label_live_spoof"] != "spoof":
            continue
        sample_id = str(row["sample_id"])
        source_record_id = lookup.get(sample_id)
        if not source_record_id:
            raise SyntheticRealProbeError(
                f"real spoof sample_id={sample_id!r} has no resolvable source_record_id; "
                "fail closed")
        records.append(PopulationRecord(
            sample_identity=sample_id, stable_group_identity=source_record_id,
            source_domain=str(row["dataset"]), label=REAL_SPOOF_CLASS))
    return records


def resolve_synthetic_population(repo: Path, arm: str, *,
                                 domains: Sequence[str] = ("casia_fasd", "msu_mfsd")
                                 ) -> list[PopulationRecord]:
    """Every row of ARM's frozen C6 matched bank — the exact bank that arm's
    C8 rows trained against. Opened through `c6_bank.open_arm_bank`, the same
    call `c8.py`'s row executor makes before training
    (`src/prism_fas/pipeline/adapters/c8.py:1346-1351`) — never a second
    bank resolver.

    `sample_identity` is the bank row's own `synthetic_id`. `stable_group_identity`
    is resolved as `live_target_sample_id -> source_train sample_id ->
    source_record_id`, through the SAME lookup `resolve_real_spoof_population`
    uses — one implementation, never a second. Fail closed if
    `live_target_sample_id` does not map uniquely and exactly to a
    `source_train` row.
    """
    from prism_fas.detector.c6_bank import open_arm_bank
    from prism_fas.evaluation import c6_evidence
    from prism_fas.pipeline.adapters import sources

    if arm not in ARMS:
        raise SyntheticRealProbeError(f"unknown arm {arm!r}; the protocol covers {ARMS}")
    inputs = sources.verify_detector_inputs(repo, arms=(arm,))
    evidence = c6_evidence.verify_c6_evidence(repo).bank(arm)
    bank = open_arm_bank(
        repo, arm=arm, evidence=evidence,
        candidates_root=Path(repo) / inputs["candidates_root"],
        package_identity=inputs["package_identity"],
        recipe_bank_identity=inputs["recipe_bank_identity"])
    lookup = _source_record_id_by_sample_id(repo, domains=domains)

    records: list[PopulationRecord] = []
    unmapped: list[str] = []
    for row in bank.rows:
        if row["live_target_dataset"] not in domains:
            continue
        synthetic_id = str(row["synthetic_id"])
        live_target_sample_id = str(row.get("live_target_sample_id") or "").strip()
        source_record_id = lookup.get(live_target_sample_id) if live_target_sample_id else None
        if not source_record_id:
            unmapped.append(synthetic_id)
            continue
        records.append(PopulationRecord(
            sample_identity=synthetic_id, stable_group_identity=source_record_id,
            source_domain=str(row["live_target_dataset"]), label=SYNTHETIC_SPOOF_CLASS))
    if unmapped:
        raise SyntheticRealProbeError(
            f"arm {arm!r}: {len(unmapped)} synthetic candidate(s) have a "
            f"live_target_sample_id that does not map uniquely and exactly to a "
            f"source_train row; fail closed rather than drop or group them "
            f"arbitrarily. first offenders: {sorted(unmapped)[:5]}")
    return records


def resolve_arm_populations(repo: Path, arm: str) -> tuple[list[PopulationRecord],
                                                            list[PopulationRecord]]:
    """`(real_spoof, synthetic_spoof)` population records for one arm."""
    return resolve_real_spoof_population(repo), resolve_synthetic_population(repo, arm)


# ==============================================================================
# 4. Deterministic, group-safe, cross-arm-shared split (§8 of the freeze task)
# ==============================================================================

def _split_digest(namespace: str, probe_seed: int, source_domain: str,
                  stable_group_identity: str) -> int:
    material = f"{namespace}|{probe_seed}|{source_domain}|{stable_group_identity}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest(), 16)


def split_bucket(namespace: str, probe_seed: int, source_domain: str,
                 stable_group_identity: str, *, train_fraction: float = 0.8) -> str:
    """`"train"` or `"validation"`, deterministic from stable identity alone —
    never from file path or array index, and identical for the same identity
    across every arm (RND/DET/LLM never disagree about where a given source
    identity falls)."""
    digest = _split_digest(namespace, probe_seed, source_domain, stable_group_identity)
    return TRAIN_LABEL if (digest % 100) < round(train_fraction * 100) else VALIDATION_LABEL


def assign_splits(records: Sequence[PopulationRecord], *, namespace: str, probe_seed: int,
                  train_fraction: float = 0.8) -> dict[str, list[PopulationRecord]]:
    """Every record assigned to train/validation, group-safe by construction:
    the same `stable_group_identity` always maps to the same bucket for a
    given `(namespace, probe_seed, source_domain)`, so no derived sample can
    ever straddle the split."""
    out: dict[str, list[PopulationRecord]] = {TRAIN_LABEL: [], VALIDATION_LABEL: []}
    for record in records:
        bucket = split_bucket(namespace, probe_seed, record.source_domain,
                              record.stable_group_identity, train_fraction=train_fraction)
        out[bucket].append(record)
    return out


def verify_group_safe_split(split: Mapping[str, Sequence[PopulationRecord]]) -> None:
    """Raise unless train and validation share zero `stable_group_identity`
    values. The split rule (`split_bucket`) is group-safe by construction, so
    this is a defense-in-depth assertion — the V2 correction this project
    made once already (V1 declared `group_safe: true` without actually
    proving it); never skip proving it again."""
    train_groups = {record.stable_group_identity for record in split.get(TRAIN_LABEL, ())}
    validation_groups = {record.stable_group_identity for record in split.get(VALIDATION_LABEL, ())}
    leaked = train_groups & validation_groups
    if leaked:
        raise SyntheticRealProbeError(
            f"group-safety violated: {len(leaked)} stable_group_identity value(s) "
            f"appear in both train and validation: {sorted(leaked)[:5]}")


# ==============================================================================
# 5. Deterministic 1:1 class balance, shared real subset across arms (§9)
# ==============================================================================

def _selection_order_key(protocol_id: str, probe_seed: int, split: str, source_domain: str,
                         sample_identity: str) -> str:
    """The SAMPLE-selection order key — deliberately keyed on `sample_identity`,
    never `stable_group_identity`. The split (above) partitions on the GROUP
    identity; selecting WHICH of the already-split samples to keep is a
    separate axis and uses the finer-grained sample identity, per the V2
    pre-execution correction."""
    material = f"{protocol_id}|{probe_seed}|{split}|{source_domain}|{sample_identity}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def balance_classes(*, protocol_id: str, probe_seed: int, split: str, source_domain: str,
                    real_spoof: Sequence[PopulationRecord],
                    synthetic_by_arm: Mapping[str, Sequence[PopulationRecord]]
                    ) -> tuple[list[PopulationRecord], dict[str, list[PopulationRecord]]]:
    """Exactly `N` real-spoof and `N` synthetic-spoof (per arm) records for one
    `(probe_seed, split, source_domain)` cell, `N = min(...)` over the real
    pool and all three arms' synthetic pools — so RND/DET/LLM are compared
    at the SAME real-spoof subset and the SAME sample budget. Deterministic
    SHA-256 order over `sample_identity` (never `stable_group_identity` —
    selection order is a separate axis from the group-safe split), no
    replacement, no oversampling, no class weights.
    """
    counts = [len(real_spoof)] + [len(synthetic_by_arm[arm]) for arm in ARMS]
    n = min(counts)
    ordered_real = sorted(
        real_spoof, key=lambda item: _selection_order_key(
            protocol_id, probe_seed, split, source_domain, item.sample_identity))
    selected_real = ordered_real[:n]
    selected_synthetic = {
        arm: sorted(synthetic_by_arm[arm], key=lambda item: _selection_order_key(
            protocol_id, probe_seed, split, source_domain, item.sample_identity)
        )[:n]
        for arm in ARMS}
    return selected_real, selected_synthetic


def balance_report(*, protocol_id: str, probe_seed: int, split: str, source_domain: str,
                   real_spoof: Sequence[PopulationRecord],
                   synthetic_by_arm: Mapping[str, Sequence[PopulationRecord]]
                   ) -> dict[str, Any]:
    """`balance_classes` plus the group-count reporting the V2 correction
    adds: unique `source_record_id` (`stable_group_identity`) counts before
    and after balancing, per population, so a reviewer can see how much
    group-diversity balancing preserved without recomputing it by hand."""
    selected_real, selected_synthetic = balance_classes(
        protocol_id=protocol_id, probe_seed=probe_seed, split=split, source_domain=source_domain,
        real_spoof=real_spoof, synthetic_by_arm=synthetic_by_arm)

    def _groups(records: Sequence[PopulationRecord]) -> int:
        return len({record.stable_group_identity for record in records})

    return {
        "n": len(selected_real),
        "pre_balance_counts": {"real": len(real_spoof),
                               **{arm: len(synthetic_by_arm[arm]) for arm in ARMS}},
        "post_balance_counts": {"real": len(selected_real),
                                **{arm: len(selected_synthetic[arm]) for arm in ARMS}},
        "unique_source_record_id_counts": {
            "real_pre": _groups(real_spoof), "real_post": _groups(selected_real),
            **{f"{arm}_pre": _groups(synthetic_by_arm[arm]) for arm in ARMS},
            **{f"{arm}_post": _groups(selected_synthetic[arm]) for arm in ARMS},
        },
        "selected_real": selected_real,
        "selected_synthetic": selected_synthetic,
    }


# ==============================================================================
# 6. Evidence extraction — exactly [global_logit_G, p_global], nothing else
# ==============================================================================

def _evidence_scalar(value: Any) -> float:
    """One evidence field as a plain Python float — safe whether `value` is
    a CUDA tensor, a CPU tensor, a numpy array, or a plain Python number.

    `np.asarray` alone raises on a CUDA tensor (`TypeError: can't convert
    cuda:0 device type tensor to numpy. Use Tensor.cpu()...`) — the C8
    canonical evaluation path (`M9Trainer` cross-source evaluation) always
    converts with `.detach().float().cpu().numpy()` first; this does the
    same, so evidence extraction matches C8's device semantics exactly
    regardless of which device the scientific detector ran on.
    """
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def extract_evidence(model_output: Any) -> np.ndarray:
    """`[global_logit_G, p_global]` from one `ModelOutput`-shaped forward
    result. Reads `.global_logit` and `.p_global` ONLY — this function's own
    source is asserted, by a static regression test, to never reference any
    `FORBIDDEN_EVIDENCE_FIELDS` name.
    """
    global_logit = _evidence_scalar(model_output.global_logit)
    p_global = _evidence_scalar(model_output.p_global)
    return np.array([global_logit, p_global], dtype=np.float64)


def forward_checkpoint_evidence(model: Any, batch: Any) -> np.ndarray:
    """`extract_evidence` applied to one forward pass, under `no_grad`.

    Pure given an already-built `model` (anything callable as `model(batch)`
    and returning a `ModelOutput`-shaped result) and `batch` (anything the
    model accepts) — safe to unit-test with a fake model and a fake batch,
    no checkpoint weights or real images required.
    """
    import torch

    with torch.no_grad():
        output = model(batch)
    return extract_evidence(output)


def average_checkpoint_evidence(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """`e_A(x) = arithmetic_mean_k e_k(x)` over the checkpoints given — §4 of
    the freeze task. All checkpoints contribute; none is selected."""
    if not vectors:
        raise SyntheticRealProbeError("cannot average zero checkpoint evidence vectors")
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in vectors])
    if stacked.shape[1] != EVIDENCE_DIMENSION:
        raise SyntheticRealProbeError(
            f"evidence vector must be {EVIDENCE_DIMENSION}-D, got shape {stacked.shape}")
    return stacked.mean(axis=0)


# ==============================================================================
# 7. Normalization — train-only z-score (§11)
# ==============================================================================

@dataclass(frozen=True)
class Normalization:
    mean: np.ndarray
    std: np.ndarray
    epsilon: float = 1e-8


def fit_normalization(train_features: np.ndarray, *, epsilon: float = 1e-8) -> Normalization:
    features = np.asarray(train_features, dtype=np.float64)
    mean = features.mean(axis=0)
    std = features.std(axis=0)  # population std, ddof=0
    return Normalization(mean=mean, std=std, epsilon=epsilon)


def apply_normalization(features: np.ndarray, normalization: Normalization) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    return (features - normalization.mean) / np.maximum(normalization.std, normalization.epsilon)


# ==============================================================================
# 8. Linear probe — torch.nn.Linear(2,1), zero-init, LBFGS (§12)
# ==============================================================================

#: The frozen LBFGS configuration. Never tuned, never searched.
LBFGS_CONFIG: dict[str, Any] = {
    "lr": 1.0, "max_iter": 200, "max_eval": 250,
    "tolerance_grad": 1e-7, "tolerance_change": 1e-9,
    "history_size": 100, "line_search_fn": "strong_wolfe",
}
L2_LAMBDA = 1e-4
CLASSIFIER_THRESHOLD = 0.5


def fit_linear_probe(train_features: np.ndarray, train_labels: np.ndarray) -> Any:
    """`torch.nn.Linear(2, 1)`, zero-initialized, fit by full-batch LBFGS
    with the frozen hyperparameters, float64, CPU. Deterministic: zero
    initialization plus a non-stochastic full-batch optimizer over fixed
    data has no source of randomness. No minibatching, no validation-metric
    early stopping, no hyperparameter search — the closure below is the
    entire fitting procedure.
    """
    import torch

    features = torch.as_tensor(np.asarray(train_features, dtype=np.float64))
    labels = torch.as_tensor(np.asarray(train_labels, dtype=np.float64)).reshape(-1, 1)
    if features.shape[1] != EVIDENCE_DIMENSION:
        raise SyntheticRealProbeError(
            f"probe features must be {EVIDENCE_DIMENSION}-D, got {features.shape}")

    torch.manual_seed(0)   # irrelevant to the deterministic zero-init below;
                           # set anyway so no global RNG state leaks in
    probe = torch.nn.Linear(EVIDENCE_DIMENSION, 1).to(torch.float64)
    with torch.no_grad():
        probe.weight.zero_()
        probe.bias.zero_()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(probe.parameters(), **LBFGS_CONFIG)

    def closure() -> Any:
        optimizer.zero_grad()
        logits = probe(features)
        loss = loss_fn(logits, labels) + L2_LAMBDA * probe.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return probe


def predict_probability(probe: Any, features: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        logits = probe(torch.as_tensor(np.asarray(features, dtype=np.float64)))
        return torch.sigmoid(logits).reshape(-1).numpy()


# ==============================================================================
# 9. BA_sep — per probe seed, then the frozen arithmetic mean (§13)
# ==============================================================================

def compute_ba_sep_for_seed(train_features: np.ndarray, train_labels: np.ndarray,
                            validation_features: np.ndarray,
                            validation_labels: np.ndarray) -> dict[str, Any]:
    """One probe seed's BA_sep_arm_seed: fit on `train`, score on
    `validation`, at the frozen threshold 0.5. This is a pure numeric
    function — it never resolves a population, a checkpoint or a target
    path, so it is safe to unit-test with entirely synthetic fixture
    arrays. It is ALSO the function a real scientific run would call with
    real evidence; nothing here distinguishes the two, which is the point —
    the protocol is the same either way. This module simply never calls it
    with real evidence.
    """
    from prism_fas.train.metrics import balanced_accuracy

    normalization = fit_normalization(train_features)
    train_z = apply_normalization(train_features, normalization)
    validation_z = apply_normalization(validation_features, normalization)

    probe = fit_linear_probe(train_z, train_labels)
    probabilities = predict_probability(probe, validation_z)
    ba = balanced_accuracy(probabilities, np.asarray(validation_labels), CLASSIFIER_THRESHOLD)

    with __import__("torch").no_grad():
        weight = probe.weight.detach().reshape(-1).tolist()
        bias = float(probe.bias.detach().item())
    return {
        "balanced_accuracy": float(ba),
        "normalization": {"mean": normalization.mean.tolist(),
                          "std": normalization.std.tolist(),
                          "epsilon": normalization.epsilon},
        "probe_coefficients": {"weight": weight, "bias": bias},
        "train_count": int(len(train_labels)), "validation_count": int(len(validation_labels)),
        "classifier_threshold": CLASSIFIER_THRESHOLD,
    }


def aggregate_ba_sep(per_seed: Mapping[int, float]) -> float:
    """`BA_sep_arm = arithmetic mean of BA_sep_arm_seed` over EXACTLY the
    three frozen probe seeds — §13 of the freeze task, §3.1.1 of the spec."""
    from prism_fas.evaluation import detector_reliability

    values = [per_seed[seed] for seed in sorted(per_seed)]
    if len(values) != detector_reliability.BA_SEP_SEEDS_REQUIRED:
        raise SyntheticRealProbeError(
            f"BA_sep_arm requires exactly {detector_reliability.BA_SEP_SEEDS_REQUIRED} "
            f"probe-seed values, got {len(values)}")
    return float(sum(values) / len(values))


def hard_verdict(ba_sep_by_arm: Mapping[str, float]) -> dict[str, Any]:
    """The explicit all-arm hard verdict rule the V2 freeze adds (§10 of
    `C9_BA_SEP_OPTION1_V2_PREEXECUTION_CORRECTION.md`), frozen before any
    BA_sep value exists:

        PASS iff BA_sep_RND <= 0.75 AND BA_sep_DET <= 0.75 AND BA_sep_LLM <= 0.75

    A single arm above the ceiling fails the whole test — there is no
    partial-arm pass. This is the HARD reliability gate only; it is
    independent of `detector_reliability.C_H4_SUPPORT_RULE`, which a PASS
    here never implies.
    """
    from prism_fas.evaluation import detector_reliability

    missing = [arm for arm in ARMS if arm not in ba_sep_by_arm]
    if missing:
        raise SyntheticRealProbeError(
            f"hard_verdict requires a BA_sep value for every arm {ARMS}; missing {missing}")
    ceiling = detector_reliability.BA_SEP_CEILING
    per_arm_pass = {arm: bool(ba_sep_by_arm[arm] <= ceiling) for arm in ARMS}
    overall = all(per_arm_pass.values())
    return {
        "ba_sep_by_arm": {arm: float(ba_sep_by_arm[arm]) for arm in ARMS},
        "ba_ceiling": ceiling,
        "per_arm_pass": per_arm_pass,
        "failing_arms": sorted(arm for arm, ok in per_arm_pass.items() if not ok),
        "verdict": "PASS" if overall else "FAIL",
        "rule": ("PASS iff BA_sep_RND <= 0.75 AND BA_sep_DET <= 0.75 AND "
                 "BA_sep_LLM <= 0.75; a single arm above the ceiling fails "
                 "the whole test"),
    }


# ==============================================================================
# 10. Preflight — read-only validation, never a probe fit or a BA number
# ==============================================================================

def preflight(repo: Path) -> dict[str, Any]:
    """Validate the protocol, the expected checkpoint cardinality and the
    target firewall — never fits a probe, never computes a BA value, never
    writes a scientific artifact, never touches `state/` and never creates
    `DETECTOR_RELIABILITY_LOCK_C.json`. Safe under `--preflight-only`.
    """
    checks: dict[str, Any] = {
        "protocol_resolved": False, "protocol_identity": None, "protocol_error": "",
        "checkpoints_per_arm_expected": CHECKPOINTS_PER_ARM,
        "total_checkpoints_expected": TOTAL_CHECKPOINTS,
        "evidence_fields": list(EVIDENCE_FIELDS),
        "probe_seed_values_expected": None,
        "implementation_available": True,
        "probe_fit_executed": False,
        "ba_metric_computed": False,
        "scientific_artifacts_written": False,
        "state_modified": False,
        "detector_reliability_lock_created": False,
        "target_access": 0,
    }
    try:
        protocol = load_protocol(repo)
        checks["protocol_resolved"] = True
        checks["protocol_identity"] = protocol_identity(repo)
        checks["probe_seed_values_expected"] = list(protocol.get("probe_seed_values") or [])
    except SyntheticRealProbeError as error:
        checks["protocol_error"] = str(error)
    checks["ready_to_execute"] = checks["protocol_resolved"]
    return checks


# ==============================================================================
# 11. Real checkpoint construction and evidence forwarding.
#
# THIS SECTION IS REAL, PRODUCTION CODE — it is never exercised against real
# data on this development laptop (no runs/full/c8/, no source package), and
# every test of it in this repository uses a monkeypatched trainer or a
# fixture ModelOutput. It reuses, never reimplements, the exact canonical
# C8 construction path: `source_matrix.build_plan` (the row),
# `pipeline.adapters.c7.verify_detector_config_lock` (the frozen Track-G
# config lock), `detector.c6_bank.open_arm_bank` (the arm's frozen C6 bank —
# the SAME call `c8.py::_run_scientific_row` makes),
# `pipeline.adapters.c8._detector_config_for_row` (the row's frozen
# hyperparameters), `detector.trainer.M9Trainer` (the exact class and
# constructor arguments C8's row executor uses, pointed at the row's own
# ALREADY-EXISTING run directory) and `detector.checkpoint.load_checkpoint`
# / `apply_checkpoint` (strict, identity-checked load). Nothing here calls
# `.step()`, `.backward()`, `run_source_only_flow`, or `.save()`.
# ==============================================================================

def _row_for_checkpoint(binding: CheckpointBinding) -> Any:
    """The exact `SourceRow` a checkpoint binding names, from the one
    canonical plan — never reconstructed by hand from the binding's fields."""
    from prism_fas.evaluation.source_matrix import build_plan

    for row in build_plan().rows:
        if row.row_id == binding.row_id:
            return row
    raise SyntheticRealProbeError(
        f"{binding.row_id!r} is not a row of the current source matrix plan; "
        "the plan may have drifted since the binding was built")


def construct_row_trainer(repo: Path, binding: CheckpointBinding) -> Any:
    """Construct one checkpoint's trainer through the exact C8 row-execution
    path, then strict-load that checkpoint's weights. Never trains.

    Raises (fails closed) if the C7 lock does not verify, if the scientific
    device cannot be resolved (`c7._scientific_device` — never CUDA-or-CPU
    guessed here; a scientific probe on this laptop, which has no CUDA,
    refuses before any model is even built), if the checkpoint bytes on disk
    no longer hash to the bound SHA-256, or if
    `checkpoint.load_checkpoint`'s own identity check disagrees on any
    field — `expected_identity=trainer.identity` is the SAME `RunIdentity`
    this exact reconstruction deterministically re-derives from the row's
    real config/package/bank inputs, so any drift since the checkpoint was
    written surfaces as a refusal, not a silently-wrong load.
    """
    from prism_fas.detector import checkpoint as checkpoint_module
    from prism_fas.detector.c6_bank import open_arm_bank
    from prism_fas.detector.trainer import M9Trainer
    from prism_fas.evaluation import c6_evidence, source_evidence
    from prism_fas.pipeline.adapters import AdapterRequest, sources
    from prism_fas.pipeline.adapters.c7 import (SCIENTIFIC_CONFIG_LOCK_PATH,
                                                _scientific_device,
                                                verify_detector_config_lock)
    from prism_fas.pipeline.adapters.c8 import _detector_config_for_row
    from prism_fas.pipeline.profiles import load_profile

    repo = Path(repo)
    row = _row_for_checkpoint(binding)

    # The SAME resolver C8's own scientific row executor uses
    # (`c8.py::_run_scientific_row`) — never a duplicated CUDA-selection
    # policy, and never a silent CPU fallback for scientific inference.
    device = _scientific_device()

    verification = verify_detector_config_lock(repo, repo / SCIENTIFIC_CONFIG_LOCK_PATH)
    if not verification["valid"]:
        raise SyntheticRealProbeError(
            f"the C7 detector config lock does not verify: {verification.get('problems')}")
    lock = verification["payload"]

    inputs = sources.verify_detector_inputs(repo, arms=(binding.arm,))
    evidence = c6_evidence.verify_c6_evidence(repo).bank(binding.arm)
    bank = open_arm_bank(
        repo, arm=binding.arm, evidence=evidence,
        candidates_root=repo / inputs["candidates_root"],
        package_identity=inputs["package_identity"],
        recipe_bank_identity=inputs["recipe_bank_identity"])

    request = AdapterRequest(repo=repo, profile=load_profile("full", repo=repo))
    config, configs = _detector_config_for_row(
        request, row=row, lock=lock, bank=bank, run_id=row.row_id)

    run_root = source_evidence.row_directory(repo / source_evidence.C8_RUNS, row)
    trainer = M9Trainer(
        config=config, detector_config=configs["detector_config"],
        package_root=repo / inputs["package_root"], bank_root=repo / inputs["candidates_root"],
        recipe_bank_root=repo / inputs["recipe_bank_root"], run_root=run_root,
        cache_root=run_root / "cache", weight_root=repo / inputs["weight_root"],
        loader_config_path=repo / "configs/data/loader_m4.yaml",
        device=device, synthetic_bank=bank)

    checkpoint_path = trainer.checkpoint_path(binding.checkpoint_kind)
    on_disk_sha256 = checkpoint_module.sha256_file(checkpoint_path)
    if on_disk_sha256 != binding.checkpoint_sha256:
        raise SyntheticRealProbeError(
            f"{binding.row_id!r}: checkpoint bytes on disk no longer match the bound "
            f"SHA-256 ({on_disk_sha256} != {binding.checkpoint_sha256}); refusing to "
            "forward evidence through a checkpoint that moved")

    payload = checkpoint_module.load_checkpoint(checkpoint_path, expected_identity=trainer.identity)
    checkpoint_module.apply_checkpoint(payload, model=trainer.model)
    trainer.model.eval()
    return trainer


class _OneSampleEvidenceView:
    """A single row sliced out of a batched `ModelOutput` — lets
    `extract_evidence` (the one frozen, tested 2-field extraction rule) run
    unmodified over one sample of a batch, rather than this function
    duplicating what fields it reads."""
    __slots__ = ("global_logit", "p_global")

    def __init__(self, global_logit: Any, p_global: Any) -> None:
        self.global_logit = global_logit
        self.p_global = p_global


def forward_evidence_for_records(trainer: Any, records: Sequence[PopulationRecord]
                                 ) -> dict[str, np.ndarray]:
    """Canonical batched evaluation, matching C8's own cross-source
    evaluation semantics exactly
    (`pipeline.adapters.c8._cross_source_evaluation`): `model.eval()`
    (set once, by `construct_row_trainer`), `torch.no_grad()`, each batch
    moved with `.to(trainer.device)`, and `trainer.config.validation_batch_size`
    as the batch size — never one sample per forward call, and never a
    hard-coded batch size independent of the row's own frozen config.

    Real records resolve through `trainer.dataset._real_position`
    (`sample_id -> position` into the exact `CanonicalPackageDataset` every
    other resolver in this module reads); synthetic records resolve through
    the trainer's own bound arm bank's row order
    (`sample_identity == synthetic_id`). Fails closed if any requested
    `sample_identity` is not resolvable in this trainer's dataset. Batching
    is execution only: every sample gets exactly one evidence vector, keyed
    by its own `sample_identity`, regardless of chunk boundaries — no
    sample is reordered, dropped or merged with another.
    """
    import torch

    from prism_fas.detector.dataset import collate_items

    synthetic_position = {str(row["synthetic_id"]): position
                          for position, row in enumerate(trainer.dataset.bank.rows)}
    real_position = trainer.dataset._real_position

    items: list[Any] = []
    sample_identities: list[str] = []
    for record in records:
        if record.label == REAL_SPOOF_CLASS:
            position = real_position.get(record.sample_identity)
            if position is None:
                raise SyntheticRealProbeError(
                    f"real sample_identity {record.sample_identity!r} is not resolvable "
                    "in this trainer's source_train dataset; fail closed")
            items.append(trainer.dataset.real_item(position))
        else:
            position = synthetic_position.get(record.sample_identity)
            if position is None:
                raise SyntheticRealProbeError(
                    f"synthetic sample_identity {record.sample_identity!r} is not "
                    "resolvable in this trainer's arm bank; fail closed")
            items.append(trainer.dataset.synthetic_item(position))
        sample_identities.append(record.sample_identity)

    batch_size = int(trainer.config.validation_batch_size)
    evidence: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk_items = items[start:start + batch_size]
            chunk_ids = sample_identities[start:start + batch_size]
            batch = collate_items(chunk_items).to(trainer.device)
            output = trainer.model(batch)
            for offset, sample_identity in enumerate(chunk_ids):
                view = _OneSampleEvidenceView(
                    output.global_logit[offset], output.p_global[offset])
                evidence[sample_identity] = extract_evidence(view)
    return evidence


# ==============================================================================
# 12. The joint scientific execution — one invocation, all three arms.
# ==============================================================================

def execute_joint_probe(repo: Path, *, checkpoint_binding: Mapping[str, Any],
                        population_plan: Mapping[str, Any]) -> dict[str, Any]:
    """The real, joint, three-arm `synthetic_vs_real_spoof_probe` execution.

    Requires an ALREADY-BUILT checkpoint binding and population plan (from
    `--bind-only`, or freshly rebuilt by the caller) and re-verifies, before
    any forward pass: both are bound to the CURRENTLY active protocol
    identity, both verify their own recorded identity hash, the CURRENT
    source package identity and all three CURRENT C6 arm bank identities
    (`sources.verify_detector_inputs`) agree with what the execution binding
    recorded, and every checkpoint's bytes on disk still match the bound
    SHA-256 (re-checked a second time, per-checkpoint, inside
    `construct_row_trainer`). Fails closed (raises) on any disagreement — no
    scientific metric is ever computed over a stale or mismatched binding,
    and the package/bank check happens before ANY checkpoint is loaded, not
    only as an eventual `RunIdentity` mismatch deep inside the first one.

    Uses ONLY the frozen mechanics already implemented and tested above
    (`construct_row_trainer`, `forward_evidence_for_records`,
    `average_checkpoint_evidence`, `compute_ba_sep_for_seed`,
    `aggregate_ba_sep`, `hard_verdict`) — no new numeric rule is introduced
    here; this function is glue, not science.
    """
    protocol_id = protocol_identity(repo)
    if checkpoint_binding.get("protocol_identity") != protocol_id:
        raise SyntheticRealProbeError(
            "checkpoint binding is not bound to the active protocol identity")
    if population_plan.get("protocol_identity") != protocol_id:
        raise SyntheticRealProbeError(
            "population plan is not bound to the active protocol identity")
    if checkpoint_binding.get("checkpoint_binding_identity_sha256") != \
            checkpoint_binding_identity(checkpoint_binding):
        raise SyntheticRealProbeError("checkpoint binding fails its own identity check")
    if population_plan.get("population_plan_identity_sha256") != \
            population_plan_identity(population_plan):
        raise SyntheticRealProbeError("population plan fails its own identity check")

    # Explicit preregistered-input reverification, BEFORE the first model
    # construction or forward pass — never left to eventual per-checkpoint
    # RunIdentity rejection inside construct_row_trainer. A source package
    # or C6 bank that has moved since --bind-only ran must block here, with
    # zero model forwards and zero BA_sep, not surface as a confusing
    # failure partway through the checkpoint loop.
    from prism_fas.pipeline.adapters import sources

    current_inputs = sources.verify_detector_inputs(repo, arms=ARMS)
    if current_inputs["package_identity"] != checkpoint_binding.get("source_package_identity"):
        raise SyntheticRealProbeError(
            "current source package identity does not match the bound execution "
            "binding's source_package_identity; refusing to forward any evidence")
    bound_bank_identities = dict(checkpoint_binding.get("c6_bank_identities") or {})
    for arm in ARMS:
        current_bank_identity = str(current_inputs["c6"]["banks"][arm]["selected_set_sha256"])
        if current_bank_identity != bound_bank_identities.get(arm):
            raise SyntheticRealProbeError(
                f"current C6 {arm!r} bank identity does not match the bound execution "
                f"binding's c6_bank_identities[{arm!r}]; refusing to forward any evidence")

    checkpoints_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for item in checkpoint_binding["checkpoints"]:
        checkpoints_by_arm[str(item["arm"])].append(item)
    for arm in ARMS:
        if len(checkpoints_by_arm[arm]) != CHECKPOINTS_PER_ARM:
            raise SyntheticRealProbeError(
                f"checkpoint binding names {len(checkpoints_by_arm[arm])} checkpoints "
                f"for arm {arm!r}, expected {CHECKPOINTS_PER_ARM}")

    cells = list(population_plan["cells"])

    evidence_by_arm: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        needed: dict[str, PopulationRecord] = {}
        for cell in cells:
            for entry in cell["real_selected"]:
                needed[entry["sample_identity"]] = PopulationRecord(
                    sample_identity=entry["sample_identity"],
                    stable_group_identity=entry["stable_group_identity"],
                    source_domain=cell["source_domain"], label=REAL_SPOOF_CLASS)
            for entry in cell["synthetic_selected"][arm]:
                needed[entry["sample_identity"]] = PopulationRecord(
                    sample_identity=entry["sample_identity"],
                    stable_group_identity=entry["stable_group_identity"],
                    source_domain=cell["source_domain"], label=SYNTHETIC_SPOOF_CLASS)

        per_checkpoint: list[dict[str, np.ndarray]] = []
        for item in checkpoints_by_arm[arm]:
            binding = CheckpointBinding(
                arm=str(item["arm"]), seed=int(item["seed"]), row_id=str(item["row_id"]),
                run_identity=str(item["run_identity"]), config_identity=str(item["config_identity"]),
                checkpoint_sha256=str(item["checkpoint_sha256"]),
                checkpoint_path=str(item["checkpoint_relative_path"]),
                checkpoint_kind=Path(str(item["checkpoint_relative_path"])).stem,
                decision_logit_name=str(item["decision_logit_name"]),
                decision_graph_hash=str(item["decision_graph_hash"]))
            trainer = construct_row_trainer(repo, binding)
            per_checkpoint.append(forward_evidence_for_records(trainer, list(needed.values())))
        if len(per_checkpoint) != CHECKPOINTS_PER_ARM:
            raise SyntheticRealProbeError(
                f"arm {arm!r}: only {len(per_checkpoint)} checkpoints contributed "
                f"evidence, need exactly {CHECKPOINTS_PER_ARM}")

        merged: dict[str, np.ndarray] = {}
        for sample_identity in needed:
            vectors = []
            for one_checkpoint_evidence in per_checkpoint:
                if sample_identity not in one_checkpoint_evidence:
                    raise SyntheticRealProbeError(
                        f"arm {arm!r}: sample {sample_identity!r} is missing evidence "
                        "from one checkpoint; refusing to average an incomplete set")
                vectors.append(one_checkpoint_evidence[sample_identity])
            merged[sample_identity] = average_checkpoint_evidence(vectors)
        evidence_by_arm[arm] = merged

    per_seed_by_arm: dict[str, dict[int, float]] = {arm: {} for arm in ARMS}
    seed_details: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        by_seed: dict[int, dict[str, list[tuple[str, int]]]] = {}
        for cell in cells:
            seed = int(cell["probe_seed"])
            bucket = by_seed.setdefault(seed, {TRAIN_LABEL: [], VALIDATION_LABEL: []})
            bucket[cell["split"]].extend(
                (entry["sample_identity"], REAL_SPOOF_CLASS) for entry in cell["real_selected"])
            bucket[cell["split"]].extend(
                (entry["sample_identity"], SYNTHETIC_SPOOF_CLASS)
                for entry in cell["synthetic_selected"][arm])
        for seed, bucket in by_seed.items():
            train_features = np.array([evidence_by_arm[arm][sid] for sid, _ in bucket[TRAIN_LABEL]])
            train_labels = np.array([label for _, label in bucket[TRAIN_LABEL]])
            validation_features = np.array(
                [evidence_by_arm[arm][sid] for sid, _ in bucket[VALIDATION_LABEL]])
            validation_labels = np.array([label for _, label in bucket[VALIDATION_LABEL]])
            result = compute_ba_sep_for_seed(
                train_features, train_labels, validation_features, validation_labels)
            per_seed_by_arm[arm][seed] = result["balanced_accuracy"]
            seed_details[arm][seed] = result

    ba_sep_by_arm = {arm: aggregate_ba_sep(per_seed_by_arm[arm]) for arm in ARMS}
    verdict = hard_verdict(ba_sep_by_arm)

    return {
        "protocol_identity": protocol_id,
        "checkpoint_binding_identity": checkpoint_binding["checkpoint_binding_identity_sha256"],
        "population_plan_identity": population_plan["population_plan_identity_sha256"],
        "ba_sep_by_arm": ba_sep_by_arm,
        "per_seed_by_arm": {arm: dict(per_seed_by_arm[arm]) for arm in ARMS},
        "seed_details": seed_details,
        "verdict": verdict,
        "target_access": 0,
    }


def run_scientific_probe(repo: Path, arm: str) -> dict[str, Any]:
    """Retired single-arm entry point.

    The frozen balancing rule is JOINT across RND/DET/LLM (`balance_classes`
    requires all three arms' pools simultaneously): a per-arm call here
    would force `N = min(real, this_arm, 0, 0) = 0` for the other two arms,
    which is exactly the runner-integration defect
    `reports/readiness/C9_BA_SEP_OPTION1_V2_RUNNER_INTEGRATION_FIX.md`
    corrects. Use `execute_joint_probe(repo, checkpoint_binding=...,
    population_plan=...)` instead, via
    `prism_fas.evaluation.synthetic_real_probe_runner --execute`.
    """
    raise SyntheticRealProbeError(
        "run_scientific_probe(repo, arm) is retired: the frozen protocol balances "
        "jointly across RND/DET/LLM, so no single arm can be probed in isolation. "
        "Use execute_joint_probe(repo, checkpoint_binding=..., population_plan=...).")


__all__ = ["ARMS", "CHECKPOINTS_PER_ARM", "TOTAL_CHECKPOINTS", "EVIDENCE_FIELDS",
           "EVIDENCE_DIMENSION", "FORBIDDEN_EVIDENCE_FIELDS", "TRAIN_LABEL",
           "VALIDATION_LABEL", "REAL_SPOOF_CLASS", "SYNTHETIC_SPOOF_CLASS",
           "SyntheticRealProbeError", "load_protocol", "protocol_identity",
           "CheckpointBinding", "track_g_p3_rows", "resolve_checkpoint_set",
           "resolve_all_checkpoint_sets", "PopulationRecord",
           "resolve_real_spoof_population", "resolve_synthetic_population",
           "resolve_arm_populations", "split_bucket", "assign_splits",
           "verify_group_safe_split", "balance_classes", "balance_report",
           "extract_evidence", "forward_checkpoint_evidence",
           "average_checkpoint_evidence", "Normalization", "fit_normalization",
           "apply_normalization", "LBFGS_CONFIG", "L2_LAMBDA",
           "CLASSIFIER_THRESHOLD", "fit_linear_probe", "predict_probability",
           "compute_ba_sep_for_seed", "aggregate_ba_sep", "hard_verdict",
           "preflight", "run_scientific_probe"]
