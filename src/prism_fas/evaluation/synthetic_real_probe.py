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
and batch and is safe to unit-test with a fake model. `run_scientific_probe`
is the one function that would load real checkpoint weights and forward real
images — it deliberately raises `NotImplementedError` here (see its
docstring): wiring and running it is future work for a separate scientific
runner (§16 of the protocol-freeze task), on a host that has the GPU C8
artifacts. No test in this repository calls it, and it is never called by
anything reachable from `--preflight-only`.
"""
from __future__ import annotations

import hashlib
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

@dataclass(frozen=True)
class CheckpointBinding:
    """One resolved, hash-verified P3-ready Track-G checkpoint."""

    arm: str
    seed: int
    row_id: str
    run_identity: str
    checkpoint_sha256: str
    checkpoint_path: str


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
    PASS, and byte-verified; never falls back to a partial set and never
    selects among available checkpoints by any criterion.
    """
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
        bindings.append(CheckpointBinding(
            arm=arm, seed=row.seed, row_id=row_id, run_identity=item.run_identity,
            checkpoint_sha256=str(item.checkpoint_sha256),
            checkpoint_path=source_evidence.row_directory(
                Path(repo) / source_evidence.C8_RUNS, row
            ).relative_to(Path(repo)).as_posix() + "/checkpoint.pt"))
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

def extract_evidence(model_output: Any) -> np.ndarray:
    """`[global_logit_G, p_global]` from one `ModelOutput`-shaped forward
    result. Reads `.global_logit` and `.p_global` ONLY — this function's own
    source is asserted, by a static regression test, to never reference any
    `FORBIDDEN_EVIDENCE_FIELDS` name.
    """
    global_logit = float(np.asarray(model_output.global_logit).reshape(-1)[0])
    p_global = float(np.asarray(model_output.p_global).reshape(-1)[0])
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
# 11. The scientific runner — NOT WIRED. Deliberately.
# ==============================================================================

def run_scientific_probe(repo: Path, arm: str) -> dict[str, Any]:
    """Would compute a real `BA_sep_arm` for one arm, end to end.

    NOT IMPLEMENTED, ON PURPOSE, IN THIS PROTOCOL-FREEZE TASK. Everything up
    to this function is real and reusable: `resolve_checkpoint_set` and
    `resolve_arm_populations` above already resolve the true checkpoints and
    populations through the canonical readers, and
    `forward_checkpoint_evidence` / `compute_ba_sep_for_seed` /
    `aggregate_ba_sep` already implement the frozen probe exactly. What is
    missing is the glue that loads each checkpoint's weights into a real
    `PRISMDetector` and forwards real image batches through it — the same
    construction `M9Trainer` and `M9TrainingDataset`/`M9ValidationDataset`
    already do for C8's own rows and cross-source diagnostics
    (`src/prism_fas/pipeline/adapters/c8.py::_run_scientific_row`,
    `_cross_source_evaluation`). Wiring it here, untested against real data,
    on a machine that has none, is exactly the kind of unverifiable
    integration code this project's own conventions refuse to ship. A
    separate scientific runner, built and tested on the GPU host that
    possesses the real C8 checkpoints and the real source package, is the
    correct place for it (§16/§17 of the protocol-freeze task).
    """
    raise NotImplementedError(
        "run_scientific_probe is intentionally unwired: this codebase freezes and "
        "implements the C9_DETECTOR_BA_SEP_OPTION1_V2 protocol's mechanics, it "
        "does not execute them. See this function's docstring.")


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
