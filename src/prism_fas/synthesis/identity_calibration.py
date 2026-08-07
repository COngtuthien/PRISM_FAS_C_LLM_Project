"""M8 identity calibration v2: a source_train-only cross-observation protocol.

v1 calibrated `tau_id` from the SAME image under ±2 % brightness/contrast and
noise std 0.002. That population measures encoder sensitivity to a photometric
nudge, not how an AdaFace embedding actually moves between two different
observations of the same person, so it produced a threshold an M7 physics repaint
cannot meet.

v2 calibrates from real pairs:

- genuine  — same dataset, same identity, DIFFERENT `source_record_id`
- impostor — same dataset, DIFFERENT identity

and freezes `tau_id_v2 = max(tau_genuine, tau_impostor)` by a rule declared
before any candidate is re-evaluated. Nothing here reads a generated candidate,
`source_dev`, `target_test` or a raw dataset path. Never imports modal.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Callable
import numpy as np
import yaml
from prism_fas.utils.core import atomic_json_write
from .m8_pipeline import SampleStore, SourceOnlyAudit
from .pair_plan import ALLOWED_DATASETS, SOURCE_SPLIT, SourceRow, load_source_train_rows
from .quality_gate import Thresholds

IDENTITY_CALIBRATION_VERSION = "m8-identity-calibration-v2"
IDENTITY_CALIBRATION_SCHEMA_VERSION = "m8-identity-calibration-v2"
CALIBRATION_SEED = 20260806
LIVE_LABEL = "live"
# Declared numerical tolerance for the "run calibration twice" requirement. Two
# runs on the same device are expected to be bit-identical; the tolerance exists
# so a genuine device change is reported rather than silently accepted.
COSINE_TOLERANCE = 1.0e-6
# Minimum structure the dataset must actually have. Falling back to same-record
# frames would reintroduce exactly the defect v2 exists to remove, so a shortfall
# stops the protocol instead.
MINIMUM_IDENTITIES = 10
MINIMUM_GENUINE_PAIRS = 100
REQUIRED_DATASETS = ALLOWED_DATASETS
# Provenance kept out of the calibration content identity, for the same reason as
# the pair plan: parquet bytes and wall clocks are not portable.
IDENTITY_EXCLUDED_FIELDS = ("calibration_content_identity_sha256", "identity_excluded_fields",
                            "created_at", "provenance", "parquet_byte_hashes", "device_report")


class IdentityCalibrationError(RuntimeError):
    """The source-only identity calibration cannot be performed as declared."""


# --- identity keys -----------------------------------------------------------

def normalize_subject_id(value: Any) -> str:
    """Whitespace-stripped, case-folded subject id. Never crosses datasets."""
    text = "" if value is None else str(value).strip().lower()
    if not text: raise IdentityCalibrationError("subject_id is empty; the sample cannot be keyed")
    return text


def identity_key(dataset: str, subject_id: Any) -> str:
    """`dataset::subject` — CASIA subject "5" and MSU subject "5" are two people,
    so the dataset prefix is part of the key, not decoration."""
    if dataset not in ALLOWED_DATASETS: raise IdentityCalibrationError(f"dataset {dataset!r} is not allowed")
    return f"{dataset}::{normalize_subject_id(subject_id)}"


def identity_key_hash(key: str) -> str:
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _rank_key(digest: str) -> int:
    return int(digest[:16], 16)


def live_rows(package_root: Path, audit: SourceOnlyAudit | None = None) -> list[SourceRow]:
    """`source_train` live rows only, from the package manifest alone."""
    audit = audit or SourceOnlyAudit()
    audit.record(f"manifests/{SOURCE_SPLIT}.parquet")
    rows = [row for row in load_source_train_rows(Path(package_root))
            if row.label == LIVE_LABEL and row.project_split == SOURCE_SPLIT]
    if not rows: raise IdentityCalibrationError(f"{SOURCE_SPLIT} holds no live samples")
    return sorted(rows, key=lambda row: row.sample_id)


def group_by_identity(rows: list[SourceRow]) -> dict[str, list[SourceRow]]:
    groups: dict[str, list[SourceRow]] = {}
    for row in rows:
        if not row.subject_id: continue
        groups.setdefault(identity_key(row.dataset, row.subject_id), []).append(row)
    return {key: sorted(value, key=lambda row: row.sample_id) for key, value in sorted(groups.items())}


# --- Phase 1: structure audit ------------------------------------------------

def audit_identity_structure(package_root: Path) -> dict[str, Any]:
    """Prove, from the manifest alone, that real cross-record same-identity
    calibration is possible before any threshold logic is written."""
    audit = SourceOnlyAudit()
    rows = live_rows(package_root, audit)
    groups = group_by_identity(rows)
    skipped = [row.sample_id for row in rows if not row.subject_id]

    def per_dataset(counter: Callable[[str], Any]) -> dict[str, Any]:
        return {dataset: counter(dataset) for dataset in ALLOWED_DATASETS}

    records_by_identity = {key: sorted({row.source_record_id for row in value}) for key, value in groups.items()}
    samples_by_record: dict[str, int] = {}
    for row in rows: samples_by_record[row.source_record_id] = samples_by_record.get(row.source_record_id, 0) + 1
    valid = {key for key, records in records_by_identity.items() if len(records) >= 2}
    genuine_available = {key: _count_cross_record_pairs(groups[key]) for key in sorted(valid)}
    impostor_available = per_dataset(lambda dataset: _count_impostor_pairs(
        [row for row in rows if row.dataset == dataset], groups))

    report = {
        "identity_structure_schema_version": IDENTITY_CALIBRATION_SCHEMA_VERSION,
        "split": SOURCE_SPLIT, "label": LIVE_LABEL,
        "identity_key_definition": 'dataset + "::" + normalized subject_id',
        "cross_dataset_subject_ids_are_different_people": True,
        "live_count": len(rows),
        "live_count_by_dataset": per_dataset(lambda dataset: sum(1 for row in rows if row.dataset == dataset)),
        "identity_count": len(groups),
        "identity_count_by_dataset": per_dataset(
            lambda dataset: sum(1 for key in groups if key.startswith(f"{dataset}::"))),
        "source_record_count_by_identity": {identity_key_hash(key): len(records)
                                            for key, records in sorted(records_by_identity.items())},
        "source_record_count_histogram": _histogram([len(records) for records in records_by_identity.values()]),
        "sample_count_by_source_record_histogram": _histogram(list(samples_by_record.values())),
        "sample_count_by_identity_histogram": _histogram([len(value) for value in groups.values()]),
        "identities_with_at_least_two_source_records": len(valid),
        "identities_with_at_least_two_source_records_by_dataset": per_dataset(
            lambda dataset: sum(1 for key in valid if key.startswith(f"{dataset}::"))),
        "potential_genuine_pairs": int(sum(genuine_available.values())),
        "potential_genuine_pairs_by_dataset": per_dataset(
            lambda dataset: int(sum(count for key, count in genuine_available.items()
                                    if key.startswith(f"{dataset}::")))),
        "potential_impostor_pairs_by_dataset": impostor_available,
        "skipped_identities": sorted({identity_key_hash(key) for key in set(groups) - valid}),
        "skipped_identity_reasons": {identity_key_hash(key): "fewer than two distinct source_record_id values"
                                     for key in sorted(set(groups) - valid)},
        "skipped_samples_without_subject_id": len(skipped),
        "source_isolation": audit.report()}
    report["requirements"] = {
        "both_datasets_represented": all(
            report["identity_count_by_dataset"][dataset] > 0 for dataset in REQUIRED_DATASETS),
        "at_least_10_valid_identities": len(valid) >= MINIMUM_IDENTITIES,
        "at_least_100_valid_genuine_pairs": report["potential_genuine_pairs"] >= MINIMUM_GENUINE_PAIRS,
        "every_genuine_pair_uses_distinct_source_records": True}
    report["supports_cross_record_identity_calibration"] = all(report["requirements"].values())
    return report


def _count_cross_record_pairs(rows: list[SourceRow]) -> int:
    return sum(1 for index, left in enumerate(rows) for right in rows[index + 1:]
               if left.source_record_id != right.source_record_id)


def _count_impostor_pairs(rows: list[SourceRow], groups: dict[str, list[SourceRow]]) -> int:
    keys = {row.sample_id: identity_key(row.dataset, row.subject_id) for row in rows if row.subject_id}
    ordered = [row for row in rows if row.sample_id in keys]
    return sum(1 for index, left in enumerate(ordered) for right in ordered[index + 1:]
               if keys[left.sample_id] != keys[right.sample_id])


def _histogram(values: list[int]) -> dict[str, int]:
    out: dict[int, int] = {}
    for value in values: out[value] = out.get(value, 0) + 1
    return {str(key): out[key] for key in sorted(out)}


# --- Phase 2: configuration ---------------------------------------------------

def load_identity_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise IdentityCalibrationError("v2 config must be a YAML mapping")
    block = payload.get("identity_calibration")
    if not isinstance(block, dict): raise IdentityCalibrationError("v2 config needs an identity_calibration block")
    if block.get("version") != IDENTITY_CALIBRATION_VERSION:
        raise IdentityCalibrationError(f"v2 config version must be {IDENTITY_CALIBRATION_VERSION!r}")
    if not block.get("require_source_train_live", False):
        raise IdentityCalibrationError("v2 config must require source_train live samples")
    if not block.get("require_different_source_record", False):
        raise IdentityCalibrationError("v2 config must require different source records for genuine pairs")
    if not block.get("require_same_dataset_for_impostor", False):
        raise IdentityCalibrationError("v2 config must require same-dataset impostor pairs")
    # A forbidden split may be named ONLY inside a `forbidden_splits` declaration.
    # Those declarations are stripped at any depth before the scan, so the config
    # cannot smuggle a reference in past a top-level-only check.
    from .quality_calibration import strip_key
    text = json.dumps(strip_key(payload, "forbidden_splits"), sort_keys=True).lower()
    for forbidden in ("source_dev", "target_test", "siw"):
        if forbidden in text:
            raise IdentityCalibrationError(f"v2 config references {forbidden!r} outside its forbidden list")
    return payload


def config_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# --- Phase 3: genuine pairs ---------------------------------------------------

GENUINE_FIELDS = ("pair_id", "dataset", "sample_id_a", "sample_id_b", "source_record_id_a",
                  "source_record_id_b", "identity_key_hash", "rank")
IMPOSTOR_FIELDS = ("pair_id", "dataset", "sample_id_a", "sample_id_b", "identity_key_hash_a",
                   "identity_key_hash_b", "rank")


def build_genuine_pairs(rows: list[SourceRow], *, seed: int = CALIBRATION_SEED,
                        maximum_per_identity: int = 20) -> list[dict[str, Any]]:
    """Same identity, same dataset, DIFFERENT `source_record_id`.

    Two frames of one canonical record are two views of one observation, so they
    would understate real embedding variation; they are excluded, not down-weighted.
    """
    groups = group_by_identity(rows)
    pairs: list[dict[str, Any]] = []
    for key, samples in groups.items():
        dataset, subject = key.split("::", 1)
        candidates: list[dict[str, Any]] = []
        for index, left in enumerate(samples):
            for right in samples[index + 1:]:
                if left.source_record_id == right.source_record_id: continue
                first, second = sorted((left, right), key=lambda row: row.sample_id)
                digest = _digest(IDENTITY_CALIBRATION_VERSION, seed, dataset, subject,
                                 first.sample_id, second.sample_id)
                candidates.append({"pair_id": "genuine_" + digest[:20], "dataset": dataset,
                                   "sample_id_a": first.sample_id, "sample_id_b": second.sample_id,
                                   "source_record_id_a": first.source_record_id,
                                   "source_record_id_b": second.source_record_id,
                                   "identity_key_hash": identity_key_hash(key), "_order": digest})
        candidates.sort(key=lambda row: (_rank_key(row["_order"]), row["sample_id_a"], row["sample_id_b"]))
        for rank, row in enumerate(candidates[:max(0, int(maximum_per_identity))]):
            pairs.append({**{name: row[name] for name in GENUINE_FIELDS if name != "rank"}, "rank": rank})
    pairs.sort(key=lambda row: row["pair_id"])
    _assert_unique(pairs)
    return pairs


# --- Phase 4: impostor pairs --------------------------------------------------

def build_impostor_pairs(rows: list[SourceRow], *, seed: int = CALIBRATION_SEED,
                         maximum_total: int = 20000) -> list[dict[str, Any]]:
    """Different identity, SAME dataset.

    Cross-dataset impostors are excluded: a CASIA/MSU pair differs in capture
    domain as well as identity, which would make the impostor problem artificially
    easy and push `tau_impostor` down.

    Selection is round-robin over identity PARTICIPATION so an identity with many
    samples cannot dominate the tail the 99.9th percentile is read from. Filing
    each pair under one owner and cycling owners is not enough: an identity that
    owns few pairs still appears in many, so the round-robin walks every identity
    a pair participates in.
    """
    groups = group_by_identity(rows)
    keys = {row.sample_id: identity_key(row.dataset, row.subject_id) for row in rows if row.subject_id}
    per_dataset_target = _dataset_targets(rows, groups, keys, maximum_total)
    selected: list[dict[str, Any]] = []
    for dataset in ALLOWED_DATASETS:
        buckets = _impostor_buckets(rows, keys, dataset, seed)
        selected.extend(_round_robin(buckets, per_dataset_target[dataset]))
    selected.sort(key=lambda row: row["pair_id"])
    pairs = [{**{name: row[name] for name in IMPOSTOR_FIELDS if name != "rank"}, "rank": rank}
             for rank, row in enumerate(selected)]
    _assert_unique(pairs)
    return pairs


def _impostor_buckets(rows: list[SourceRow], keys: dict[str, str], dataset: str,
                      seed: int) -> dict[str, list[dict[str, Any]]]:
    """Every unordered different-identity pair, listed under BOTH identities it
    involves, each list deterministically ordered."""
    ordered = sorted([row for row in rows if row.dataset == dataset and row.sample_id in keys],
                     key=lambda row: row.sample_id)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            key_a, key_b = keys[left.sample_id], keys[right.sample_id]
            if key_a == key_b: continue
            hash_a, hash_b = identity_key_hash(key_a), identity_key_hash(key_b)
            digest = _digest(IDENTITY_CALIBRATION_VERSION, seed, dataset, hash_a, hash_b,
                             left.sample_id, right.sample_id)
            pair = {"pair_id": "impostor_" + digest[:20], "dataset": dataset,
                    "sample_id_a": left.sample_id, "sample_id_b": right.sample_id,
                    "identity_key_hash_a": hash_a, "identity_key_hash_b": hash_b, "_order": digest}
            buckets.setdefault(hash_a, []).append(pair)
            buckets.setdefault(hash_b, []).append(pair)
    for owner in buckets:
        buckets[owner].sort(key=lambda row: (_rank_key(row["_order"]), row["sample_id_a"], row["sample_id_b"]))
    return dict(sorted(buckets.items()))


def _round_robin(buckets: dict[str, list[dict[str, Any]]], target: int) -> list[dict[str, Any]]:
    """One pair per identity per cycle, skipping pairs another identity already
    contributed, so participation is as even as the data allows."""
    taken: list[dict[str, Any]] = []
    chosen: set[str] = set()
    cursors = {owner: 0 for owner in buckets}
    while len(taken) < target:
        progressed = False
        for owner in sorted(buckets):
            if len(taken) >= target: break
            entries = buckets[owner]
            index = cursors[owner]
            while index < len(entries) and entries[index]["pair_id"] in chosen: index += 1
            cursors[owner] = index
            if index < len(entries):
                taken.append(entries[index]); chosen.add(entries[index]["pair_id"])
                cursors[owner] = index + 1; progressed = True
        if not progressed: break
    return taken


def _dataset_targets(rows: list[SourceRow], groups: dict[str, list[SourceRow]], keys: dict[str, str],
                     maximum_total: int) -> dict[str, int]:
    """Balance CASIA and MSU as closely as availability permits, under the cap."""
    available = {dataset: _count_impostor_pairs([row for row in rows if row.dataset == dataset], groups)
                 for dataset in ALLOWED_DATASETS}
    balanced = min(min(available.values()), max(1, int(maximum_total) // len(ALLOWED_DATASETS)))
    return {dataset: min(available[dataset], balanced) for dataset in ALLOWED_DATASETS}


def _assert_unique(pairs: list[dict[str, Any]]) -> None:
    ids = [row["pair_id"] for row in pairs]
    if len(set(ids)) != len(ids): raise IdentityCalibrationError("duplicate pair ids")
    unordered = {(row["dataset"], row["sample_id_a"], row["sample_id_b"]) for row in pairs}
    if len(unordered) != len(pairs): raise IdentityCalibrationError("duplicate unordered pairs")
    for row in pairs:
        if row["sample_id_a"] == row["sample_id_b"]: raise IdentityCalibrationError("a pair reuses one sample")


def pairs_digest(pairs: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    """Logical pair-plan identity: canonical rows, never parquet bytes, so it is
    portable across pyarrow versions."""
    canonical = json.dumps([{name: row[name] for name in fields}
                            for row in sorted(pairs, key=lambda row: row["pair_id"])],
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_pairs_parquet(path: Path, pairs: list[dict[str, Any]], fields: tuple[str, ...],
                        cosines: dict[str, float] | None = None) -> str:
    import pyarrow as pa, pyarrow.parquet as pq
    ordered = sorted(pairs, key=lambda row: row["pair_id"])
    schema_fields = [(name, pa.int32() if name == "rank" else pa.string()) for name in fields]
    columns = {name: [row[name] for row in ordered] for name, _ in schema_fields}
    if cosines is not None:
        schema_fields.append(("adaface_cosine", pa.float64()))
        columns["adaface_cosine"] = [float(cosines[row["pair_id"]]) for row in ordered]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(columns, schema=pa.schema(schema_fields)), Path(path),
                   compression="none", version="2.6", write_statistics=False)
    return pairs_digest(ordered, fields)


# --- Phase 5: pinned AdaFace evaluation ---------------------------------------

GENUINE_PERCENTILE = 1.0
IMPOSTOR_PERCENTILE = 99.9


def preprocessing_contract_identity() -> str:
    """The M8 production AdaFace preprocessing, hashed.

    v2 reuses the exact wrapper the gate and v1 calibration used — RGB->BGR flip,
    bicubic resize to 112, (x-0.5)/0.5, L2-normalized 512-d — so a v2 cosine is
    directly comparable to a candidate's `identity_cosine`. No alternate resize,
    alignment or normalization path is introduced.
    """
    from .quality_models import PINNED
    spec = PINNED["identity"]
    payload = {"backend": spec["backend"], "repo_id": spec["repo_id"], "revision": spec["revision"],
               "sha256": spec["sha256"], "input_size": spec["input_size"],
               "embedding_dim": spec["embedding_dim"], "input_color": spec["input_color"],
               "resize": "bicubic", "align_corners": False, "normalization_mean": 0.5,
               "normalization_std": 0.5, "l2_normalized": True,
               "wrapper": "prism_fas.synthesis.quality_models.DifferentiableAdaFace"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class IdentityBackend:
    """The pinned AdaFace only.

    `QualityBackends` also loads SCRFD and the 1.1 GB FaceXFormer, neither of
    which participates in identity calibration. This resolves the identity role
    through the same registry and the same `DifferentiableAdaFace` wrapper, so the
    numerics are the production ones, and exposes the same `embed`/`manifest`
    surface the calibration code expects.
    """

    def __init__(self, weight_root: Path, *, device: str = "cpu"):
        from .quality_models import QualityModelRegistry
        self.registry = QualityModelRegistry.resolve(Path(weight_root), roles=("identity",))
        self.device = device
        self.identity = self.registry.adaface(device)
        self.identity.eval()

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        import torch
        stacked = torch.from_numpy(np.stack(images).astype(np.float32)).to(self.device)
        with torch.no_grad():
            return self.identity(stacked).cpu().numpy().astype(np.float32)

    def manifest(self) -> dict[str, Any]: return self.registry.manifest()


class EmbeddingCache:
    """One embedding per unique processed image, keyed by image CONTENT.

    Keying by content rather than by path means the cache cannot depend on where
    the package is mounted, and two samples with identical pixels share one
    forward pass.
    """

    def __init__(self, store: SampleStore, backends: Any):
        self.store, self.backends = store, backends
        self._by_content: dict[str, np.ndarray] = {}
        self._by_sample: dict[str, str] = {}
        self.forward_passes = 0

    def content_identity(self, sample_id: str) -> str:
        from .contracts import array_hash
        if sample_id not in self._by_sample:
            image, _ = self.store.load(sample_id)
            self._by_sample[sample_id] = array_hash(image)
        return self._by_sample[sample_id]

    def embedding(self, sample_id: str) -> np.ndarray:
        content = self.content_identity(sample_id)
        cached = self._by_content.get(content)
        if cached is not None: return cached
        image, _ = self.store.load(sample_id)
        vector = np.asarray(self.backends.embed([image])[0], dtype=np.float64)
        if not np.isfinite(vector).all(): raise IdentityCalibrationError(f"{sample_id}: embedding is not finite")
        norm = float(np.linalg.norm(vector))
        if abs(norm - 1.0) > 1.0e-4: raise IdentityCalibrationError(f"{sample_id}: embedding is not L2-normalized ({norm})")
        self.forward_passes += 1
        self._by_content[content] = vector
        return vector

    def report(self) -> dict[str, Any]:
        return {"unique_samples": len(self._by_sample), "unique_image_contents": len(self._by_content),
                "forward_passes": self.forward_passes, "cache_key": "processed_image_content_identity",
                "cached_by_absolute_path": False}


def score_pairs(pairs: list[dict[str, Any]], cache: EmbeddingCache,
                progress: Callable[[dict[str, Any]], None] | None = None,
                label: str = "pairs") -> dict[str, float]:
    scores: dict[str, float] = {}
    for position, row in enumerate(pairs, 1):
        left = cache.embedding(row["sample_id_a"])
        right = cache.embedding(row["sample_id_b"])
        value = float(np.dot(left, right))
        if not np.isfinite(value): raise IdentityCalibrationError(f"{row['pair_id']}: cosine is not finite")
        scores[row["pair_id"]] = value
        if progress and (position == 1 or position % 2000 == 0 or position == len(pairs)):
            progress({"stage": label, "done": position, "total": len(pairs)})
    return scores


def _distribution(values: list[float]) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    if not series.size: return {}
    return {"count": int(series.size), "min": float(series.min()),
            "p01": float(np.percentile(series, 1.0)), "p05": float(np.percentile(series, 5.0)),
            "median": float(np.median(series)), "mean": float(series.mean()),
            "p95": float(np.percentile(series, 95.0)), "p99": float(np.percentile(series, 99.0)),
            "p999": float(np.percentile(series, 99.9)), "max": float(series.max()),
            "std": float(series.std(ddof=0))}


# --- Phase 6: freeze tau_id_v2 -------------------------------------------------

def calibrate_identity(package_root: Path, config: dict[str, Any], backends: Any, *,
                       v1_thresholds: Thresholds, progress: Callable[[dict[str, Any]], None] | None = None,
                       device_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit `tau_id_v2` from real cross-observation pairs and freeze it by the
    pre-declared rule `max(tau_genuine, tau_impostor)`.

    No generated candidate, no `source_dev`, no target and no raw dataset path
    participates, so the threshold cannot have been tuned on an acceptance count.
    """
    block = config["identity_calibration"]
    seed = int(block.get("seed", CALIBRATION_SEED))
    audit = SourceOnlyAudit()
    store = SampleStore.open(Path(package_root), audit)
    rows = live_rows(package_root, audit)
    structure = audit_identity_structure(package_root)
    if not structure["supports_cross_record_identity_calibration"]:
        raise IdentityCalibrationError(f"source_train cannot support v2 calibration: {structure['requirements']}")

    genuine = build_genuine_pairs(rows, seed=seed,
                                  maximum_per_identity=int(block["maximum_genuine_pairs_per_identity"]))
    impostor = build_impostor_pairs(rows, seed=seed,
                                    maximum_total=int(block["maximum_impostor_pairs_total"]))
    if len(genuine) < MINIMUM_GENUINE_PAIRS:
        raise IdentityCalibrationError(f"only {len(genuine)} genuine pairs, need >= {MINIMUM_GENUINE_PAIRS}")

    cache = EmbeddingCache(store, backends)
    genuine_scores = score_pairs(genuine, cache, progress, "genuine")
    impostor_scores = score_pairs(impostor, cache, progress, "impostor")
    genuine_values = [genuine_scores[row["pair_id"]] for row in genuine]
    impostor_values = [impostor_scores[row["pair_id"]] for row in impostor]

    genuine_percentile = float(block.get("genuine_percentile", GENUINE_PERCENTILE))
    impostor_percentile = float(block.get("impostor_percentile", IMPOSTOR_PERCENTILE))
    tau_genuine = float(np.percentile(genuine_values, genuine_percentile))
    tau_impostor = float(np.percentile(impostor_values, impostor_percentile))
    # Pre-declared rule. The max is a floor, not a choice: it keeps the impostor
    # false-match rate at or below (100 - impostor_percentile)% even when the
    # genuine distribution is loose.
    tau_id_v2 = float(max(tau_genuine, tau_impostor))

    thresholds = Thresholds(tau_fd=v1_thresholds.tau_fd, tau_id=tau_id_v2, tau_lm=v1_thresholds.tau_lm,
                            tau_parse=v1_thresholds.tau_parse, tau_out=v1_thresholds.tau_out,
                            tau_fp=v1_thresholds.tau_fp)
    genuine_accepted = int(sum(1 for value in genuine_values if value >= tau_id_v2))
    impostor_matched = int(sum(1 for value in impostor_values if value >= tau_id_v2))
    overlap = float(np.mean([value <= tau_impostor for value in genuine_values]))

    return {
        "identity_calibration_schema_version": IDENTITY_CALIBRATION_SCHEMA_VERSION,
        "calibration_version": IDENTITY_CALIBRATION_VERSION, "seed": seed,
        "split": SOURCE_SPLIT, "label": LIVE_LABEL,
        "identity_structure": structure,
        "populations": {
            "live_samples": len(rows), "identities": structure["identity_count"],
            "genuine_pairs": len(genuine), "impostor_pairs": len(impostor),
            "genuine_pairs_by_dataset": _counts(genuine, "dataset"),
            "impostor_pairs_by_dataset": _counts(impostor, "dataset"),
            "genuine_pairs_by_identity": _counts(genuine, "identity_key_hash"),
            "maximum_genuine_pairs_per_identity": int(block["maximum_genuine_pairs_per_identity"]),
            "maximum_impostor_pairs_total": int(block["maximum_impostor_pairs_total"]),
            "population_sha256": _digest(*sorted(row.sample_id for row in rows))},
        "genuine_pair_plan_identity_sha256": pairs_digest(genuine, GENUINE_FIELDS),
        "impostor_pair_plan_identity_sha256": pairs_digest(impostor, IMPOSTOR_FIELDS),
        "genuine_distribution": _distribution(genuine_values),
        "impostor_distribution": _distribution(impostor_values),
        "genuine_percentile": genuine_percentile, "impostor_percentile": impostor_percentile,
        "tau_genuine": tau_genuine, "tau_impostor": tau_impostor,
        "threshold_rule": "tau_id_v2 = max(tau_genuine, tau_impostor)",
        "tau_id_v2": tau_id_v2,
        "genuine_acceptance_rate_at_tau": genuine_accepted / len(genuine_values),
        "genuine_accepted_pairs": genuine_accepted,
        "impostor_false_match_rate_at_tau": impostor_matched / len(impostor_values),
        "impostor_false_matches": impostor_matched,
        "genuine_fraction_at_or_below_tau_impostor": overlap,
        "distributions_are_well_separated": bool(tau_genuine > tau_impostor),
        "separation_note": ("tau_genuine > tau_impostor: the genuine 1st percentile is above the impostor "
                            "99.9th percentile, so the rule is bound by the genuine population"
                            if tau_genuine > tau_impostor else
                            "tau_impostor >= tau_genuine: the genuine and impostor distributions overlap at "
                            "these percentiles, so the rule is bound by the impostor floor and part of the "
                            "genuine population falls below the threshold"),
        "v1_tau_id_informational_only": float(v1_thresholds.tau_id),
        "thresholds": thresholds.as_dict(), "threshold_sha256": thresholds.sha256(),
        "unchanged_from_v1": {name: getattr(v1_thresholds, name)
                              for name in ("tau_fd", "tau_lm", "tau_parse", "tau_out", "tau_fp")},
        "preprocessing_contract_identity_sha256": preprocessing_contract_identity(),
        "quality_models": backends.manifest(),
        "embedding_cache": cache.report(),
        "device_report": device_report or {"device": getattr(backends, "device", "unknown")},
        "used_generated_candidates": False, "used_source_dev": False, "used_target": False,
        "used_raw_dataset_paths": False,
        "source_isolation": audit.report(),
        "_genuine_pairs": genuine, "_impostor_pairs": impostor,
        "_genuine_scores": genuine_scores, "_impostor_scores": impostor_scores}


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows: out[str(row[key])] = out.get(str(row[key]), 0) + 1
    return dict(sorted(out.items()))


def build_lock(payload: dict[str, Any], *, package_identity: str, config: dict[str, Any]) -> dict[str, Any]:
    """The v2 calibration lock. Content identity binds only portable facts."""
    models = payload["quality_models"]["models"]["identity"]
    lock: dict[str, Any] = {
        "identity_calibration_lock_schema_version": IDENTITY_CALIBRATION_SCHEMA_VERSION,
        "calibration_version": IDENTITY_CALIBRATION_VERSION,
        "seed": payload["seed"],
        "package_identity": package_identity,
        "source_population_sha256": payload["populations"]["population_sha256"],
        "split": SOURCE_SPLIT, "label": LIVE_LABEL,
        "adaface_backend": models["backend"], "adaface_repo_id": models["repo_id"],
        "adaface_revision": models["revision"], "adaface_weight_sha256": models["verified_sha256"],
        "preprocessing_contract_identity_sha256": payload["preprocessing_contract_identity_sha256"],
        "config_sha256": config_sha256(config),
        "genuine_pair_plan_identity_sha256": payload["genuine_pair_plan_identity_sha256"],
        "impostor_pair_plan_identity_sha256": payload["impostor_pair_plan_identity_sha256"],
        "genuine_pairs": payload["populations"]["genuine_pairs"],
        "impostor_pairs": payload["populations"]["impostor_pairs"],
        "genuine_percentile": payload["genuine_percentile"],
        "impostor_percentile": payload["impostor_percentile"],
        "tau_genuine": payload["tau_genuine"], "tau_impostor": payload["tau_impostor"],
        "threshold_rule": payload["threshold_rule"], "tau_id_v2": payload["tau_id_v2"],
        "unchanged_from_v1": payload["unchanged_from_v1"],
        "thresholds": payload["thresholds"], "threshold_sha256": payload["threshold_sha256"],
        "used_generated_candidates": False, "used_source_dev": False, "used_target": False,
        "source_isolation": payload["source_isolation"]}
    lock["calibration_content_identity_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in lock.items() if key not in IDENTITY_EXCLUDED_FIELDS},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    lock["identity_excluded_fields"] = list(IDENTITY_EXCLUDED_FIELDS)
    return lock


def public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """The serializable calibration report: private working lists removed."""
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def write_calibration_artifacts(output_root: Path, payload: dict[str, Any], *, package_identity: str,
                                config: dict[str, Any]) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    genuine_digest = write_pairs_parquet(root / "identity_genuine_pairs_v2.parquet", payload["_genuine_pairs"],
                                         GENUINE_FIELDS, payload["_genuine_scores"])
    impostor_digest = write_pairs_parquet(root / "identity_impostor_pairs_v2.parquet", payload["_impostor_pairs"],
                                          IMPOSTOR_FIELDS, payload["_impostor_scores"])
    if genuine_digest != payload["genuine_pair_plan_identity_sha256"]:
        raise IdentityCalibrationError("genuine pair-plan identity changed when written")
    if impostor_digest != payload["impostor_pair_plan_identity_sha256"]:
        raise IdentityCalibrationError("impostor pair-plan identity changed when written")
    public = public_payload(payload)
    lock = build_lock(payload, package_identity=package_identity, config=config)
    atomic_json_write(root / "quality_calibration_v2.json", public)
    atomic_json_write(root / "identity_calibration_v2_summary.json", _summary(public))
    atomic_json_write(root / "IDENTITY_CALIBRATION_V2_LOCK.json", lock)
    return {"lock": lock, "calibration": public,
            "written": ["identity_genuine_pairs_v2.parquet", "identity_impostor_pairs_v2.parquet",
                        "quality_calibration_v2.json", "identity_calibration_v2_summary.json",
                        "IDENTITY_CALIBRATION_V2_LOCK.json"]}


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in (
        "identity_calibration_schema_version", "calibration_version", "seed", "populations",
        "genuine_pair_plan_identity_sha256", "impostor_pair_plan_identity_sha256",
        "genuine_distribution", "impostor_distribution", "genuine_percentile", "impostor_percentile",
        "tau_genuine", "tau_impostor", "threshold_rule", "tau_id_v2",
        "genuine_acceptance_rate_at_tau", "genuine_accepted_pairs",
        "impostor_false_match_rate_at_tau", "impostor_false_matches",
        "genuine_fraction_at_or_below_tau_impostor", "distributions_are_well_separated",
        "separation_note", "v1_tau_id_informational_only", "thresholds", "threshold_sha256",
        "unchanged_from_v1", "preprocessing_contract_identity_sha256", "embedding_cache",
        "device_report", "source_isolation") if key in payload}


def build_quality_gate_v2(v1_payload: dict[str, Any], v2_payload: dict[str, Any],
                          lock: dict[str, Any]) -> dict[str, Any]:
    """The gate artifact the bank actually loads: v2 thresholds on top of the
    v1 calibration.

    Only `tau_id` is re-fitted, so the fingerprint reference, the pinned model
    manifest and every other v1 threshold are carried over verbatim. Merging them
    into one file keeps a single loadable artifact whose SHA-256 is the bank's
    `quality_calibration_sha256`.
    """
    merged = {key: value for key, value in v1_payload.items() if not key.startswith("_")}
    merged["thresholds"] = dict(v2_payload["thresholds"])
    merged["threshold_sha256"] = v2_payload["threshold_sha256"]
    merged["quality_gate_version"] = "m8-quality-gate-v2"
    merged["identity_calibration_version"] = IDENTITY_CALIBRATION_VERSION
    merged["identity_calibration_content_identity_sha256"] = lock["calibration_content_identity_sha256"]
    merged["genuine_pair_plan_identity_sha256"] = v2_payload["genuine_pair_plan_identity_sha256"]
    merged["impostor_pair_plan_identity_sha256"] = v2_payload["impostor_pair_plan_identity_sha256"]
    merged["tau_genuine"] = v2_payload["tau_genuine"]
    merged["tau_impostor"] = v2_payload["tau_impostor"]
    merged["tau_id_v2"] = v2_payload["tau_id_v2"]
    merged["threshold_rule"] = v2_payload["threshold_rule"]
    merged["tau_id_v1_superseded"] = v1_payload["thresholds"]["tau_id"]
    merged["threshold_sha256_v1_superseded"] = v1_payload["threshold_sha256"]
    merged["inherited_from_v1"] = ["tau_fd", "tau_lm", "tau_parse", "tau_out", "tau_fp",
                                   "fingerprint", "quality_models"]
    merged["used_generated_candidates"] = False
    return merged


def compare_calibrations(first: dict[str, Any], second: dict[str, Any], *,
                         tolerance: float = COSINE_TOLERANCE) -> dict[str, Any]:
    """The two-run determinism requirement."""
    mismatches: list[dict[str, Any]] = []
    for name in ("genuine_pair_plan_identity_sha256", "impostor_pair_plan_identity_sha256",
                 "tau_genuine", "tau_impostor", "tau_id_v2", "threshold_sha256"):
        if first.get(name) != second.get(name):
            mismatches.append({"field": name, "first": first.get(name), "second": second.get(name)})
    for label in ("_genuine_scores", "_impostor_scores"):
        left, right = first.get(label, {}), second.get(label, {})
        if sorted(left) != sorted(right):
            mismatches.append({"field": f"{label}:pair_ids", "first": len(left), "second": len(right)})
            continue
        worst = max((abs(left[key] - right[key]) for key in left), default=0.0)
        if worst > tolerance:
            mismatches.append({"field": f"{label}:cosine", "max_abs_difference": worst, "tolerance": tolerance})
    return {"identical": not mismatches, "mismatches": mismatches, "mismatch_count": len(mismatches),
            "cosine_tolerance": tolerance}

