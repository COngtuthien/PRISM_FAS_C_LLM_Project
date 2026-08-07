"""M8 structural robustness calibration v3: a source_train-only, same-image,
localized benign appearance protocol for `tau_lm` and `tau_parse`.

v1 fitted both thresholds from the SAME image under +/-2 % brightness/contrast and
noise std 0.002. That population measures detector jitter under a near-identity
photometric nudge, not stability under a non-geometric, identity-preserving LOCAL
face appearance edit — which is what a synthetic candidate actually is. `tau_id`
carried the identical defect and was versioned in v2; the retained v2 run then
measured the consequence for the two thresholds v2 left alone (`landmark` 467 and
`parsing_dice` 233 rejections against `identity` 0).

v3 therefore compares each `source_train` live image against ITSELF after a fixed
deterministic appearance-only transform applied inside one deterministic semantic
region, and freezes

    tau_lm_v3    = 99th percentile of valid same-image landmark NME
    tau_parse_v3 = 1st  percentile of valid outside-support parsing Dice

by rules declared in `docs/M8_STRUCTURAL_CALIBRATION_V3.md` before any candidate is
re-evaluated.

The 560 v2 cross-record genuine pairs are computed here as a DIAGNOSTIC ONLY and
may never set a structural threshold: two source records differ in real pose,
expression, head-geometry projection, landmark position, occlusion and crop, which
is scene variation rather than detector jitter caused by an appearance-only edit.

Nothing here reads a generated candidate, `source_dev`, `target_test` or a raw
dataset path, and no M7 physics operator or GPAT checkpoint participates. Never
imports modal.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Callable
import numpy as np
import yaml
from prism_fas.utils.core import atomic_json_write
from .m8_pipeline import SampleStore, SourceOnlyAudit
from .masks import REGION_ORDER
from .pair_plan import ALLOWED_DATASETS, SOURCE_SPLIT
from .quality_gate import Thresholds, landmark_nme, parsing_dice
from .quality_models import PINNED

STRUCTURAL_CALIBRATION_VERSION = "m8-structural-calibration-v3"
STRUCTURAL_CALIBRATION_SCHEMA_VERSION = "m8-structural-calibration-v3"
CALIBRATION_SEED = 20260806
LIVE_LABEL = "live"

# --- the FIXED transform suite ------------------------------------------------
# Frozen before any v3 result exists and never changed after seeing calibration or
# candidate results. JPEG is deliberately excluded: cv2 JPEG encoding is not
# provably byte-deterministic across OpenCV builds.
TRANSFORM_SUITE: tuple[dict[str, Any], ...] = (
    {"slot": 0, "name": "brightness_090", "kind": "brightness", "parameter": 0.90},
    {"slot": 1, "name": "brightness_110", "kind": "brightness", "parameter": 1.10},
    {"slot": 2, "name": "contrast_090", "kind": "contrast", "parameter": 0.90},
    {"slot": 3, "name": "contrast_110", "kind": "contrast", "parameter": 1.10},
    {"slot": 4, "name": "gamma_090", "kind": "gamma", "parameter": 0.90},
    {"slot": 5, "name": "gamma_110", "kind": "gamma", "parameter": 1.10},
    {"slot": 6, "name": "noise_0005", "kind": "noise", "parameter": 0.005},
    {"slot": 7, "name": "blur_075", "kind": "blur", "parameter": 0.75})
TRANSFORM_SLOTS = len(TRANSFORM_SUITE)
EXPECTED_LIVE_SAMPLES = 280
EXPECTED_OBSERVATIONS = EXPECTED_LIVE_SAMPLES * TRANSFORM_SLOTS

TAU_LM_PERCENTILE = 99.0
TAU_PARSE_PERCENTILE = 1.0
# A threshold fitted on a heavily censored population would not describe the
# population it claims to, so a shortfall stops the protocol.
MINIMUM_VALID_LANDMARK_FRACTION = 0.95
# Declared before comparing. Two runs on one device are expected to be
# bit-identical; the tolerance exists so a real device change is reported rather
# than silently absorbed. It NEVER applies to a derived threshold.
METRIC_TOLERANCE = 1.0e-6
BLUR_TRUNCATE = 4.0

OBSERVATION_FIELDS = ("observation_id", "dataset", "sample_id", "transform_slot", "transform_name",
                      "transform_kind", "transform_parameter", "region")
IDENTITY_EXCLUDED_FIELDS = ("calibration_content_identity_sha256", "identity_excluded_fields",
                            "created_at", "provenance", "parquet_byte_hashes", "device_report")


class StructuralCalibrationError(RuntimeError):
    """The source-only structural calibration cannot be performed as declared."""


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def config_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_structural_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen v3 config."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise StructuralCalibrationError("v3 config must be a YAML mapping")
    block = payload.get("structural_calibration")
    if not isinstance(block, dict):
        raise StructuralCalibrationError("v3 config needs a structural_calibration block")
    if block.get("version") != STRUCTURAL_CALIBRATION_VERSION:
        raise StructuralCalibrationError(f"v3 config version must be {STRUCTURAL_CALIBRATION_VERSION!r}")
    for flag in ("require_source_train_live", "require_same_image_comparison", "require_all_nine_regions",
                 "require_no_geometric_transform", "cross_record_pairs_are_diagnostic_only"):
        if not block.get(flag, False): raise StructuralCalibrationError(f"v3 config must declare {flag}")
    for flag in ("uses_m7_physics_operators", "uses_gpat", "uses_generated_candidates"):
        if block.get(flag, True): raise StructuralCalibrationError(f"v3 config must declare {flag} false")
    declared = [{key: entry[key] for key in ("slot", "name", "kind", "parameter")}
                for entry in payload.get("transform_suite", [])]
    frozen = [{key: entry[key] for key in ("slot", "name", "kind", "parameter")} for entry in TRANSFORM_SUITE]
    if declared != frozen:
        raise StructuralCalibrationError("v3 config transform suite does not match the frozen suite")
    if list(payload.get("region_assignment", {}).get("regions", [])) != list(REGION_ORDER):
        raise StructuralCalibrationError("v3 config regions must be the nine canonical M7/M8 regions")
    # A forbidden split may be NAMED only inside a `forbidden_splits` declaration.
    # Those declarations are stripped at any depth first, so a config is judged on
    # what it opens rather than on which words it contains.
    from .quality_calibration import strip_key
    text = json.dumps(strip_key(payload, "forbidden_splits"), sort_keys=True).lower()
    for forbidden in ("source_dev", "target_test", "siw"):
        if forbidden in text:
            raise StructuralCalibrationError(f"v3 config references {forbidden!r} outside its forbidden list")
    return payload


# --- the fixed appearance transforms -----------------------------------------

def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = int(np.ceil(BLUR_TRUNCATE * float(sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-(offsets * offsets) / (2.0 * float(sigma) ** 2))
    return (weights / weights.sum()).astype(np.float32)


def _blur_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Separable Gaussian pass with reflect padding.

    An explicit shifted accumulation in a fixed order, so the result does not
    depend on a BLAS or OpenCV build. The kernel is symmetric and moves no
    coordinate: this is a low-pass appearance change, not a geometric resample.
    """
    radius = (len(kernel) - 1) // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="reflect")
    out = np.zeros_like(values, dtype=np.float32)
    for index in range(len(kernel)):
        window = [slice(None)] * values.ndim
        window[axis] = slice(index, index + values.shape[axis])
        out = out + np.float32(kernel[index]) * padded[tuple(window)]
    return out.astype(np.float32)


def apply_transform(image: np.ndarray, transform: dict[str, Any], *, noise_seed: int) -> np.ndarray:
    """One fixed appearance transform over the WHOLE `[3,H,W]` float image in [0,1].

    Defined independently of the region it is later restricted to, so the
    transform is a pure function of the image and the frozen parameter. No
    rotation, translation, scaling, affine, perspective, crop change, landmark
    movement or face replacement: dimensions and pixel coordinates are preserved.
    """
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 3:
        raise StructuralCalibrationError(f"expected a [3,H,W] float image, got {values.shape}")
    kind, parameter = str(transform["kind"]), float(transform["parameter"])
    if kind == "brightness":
        out = values * np.float32(parameter)
    elif kind == "contrast":
        mean = np.float32(values.mean())
        out = (values - mean) * np.float32(parameter) + mean
    elif kind == "gamma":
        out = np.power(np.clip(values, 0.0, 1.0), np.float32(parameter)).astype(np.float32)
    elif kind == "noise":
        rng = np.random.Generator(np.random.PCG64(int(noise_seed) % (2 ** 32)))
        out = values + rng.normal(0.0, parameter, size=values.shape).astype(np.float32)
    elif kind == "blur":
        kernel = _gaussian_kernel(parameter)
        out = _blur_axis(_blur_axis(values, kernel, axis=1), kernel, axis=2)
    else:
        raise StructuralCalibrationError(f"unknown calibration transform kind {kind!r}")
    result = np.clip(out, 0.0, 1.0).astype(np.float32)
    if result.shape != values.shape:
        raise StructuralCalibrationError("a calibration transform changed the image dimensions")
    return np.ascontiguousarray(result)


# --- the deterministic observation plan ---------------------------------------

def observation_digest(sample_id: str, slot: int, *, package_identity: str, seed: int) -> str:
    """`SHA256(calibration_version | source_package_identity | sample_id | transform_slot | seed)`."""
    return _digest(STRUCTURAL_CALIBRATION_VERSION, package_identity, sample_id, int(slot), int(seed))


def assign_region(digest: str) -> str:
    return REGION_ORDER[int(digest[:16], 16) % len(REGION_ORDER)]


def noise_seed(digest: str) -> int:
    """Domain-separated from the region assignment so one draw cannot bias the other."""
    return int.from_bytes(hashlib.sha256(f"structural_noise|{digest}".encode("utf-8")).digest()[:8], "big")


def build_observation_plan(rows: list[Any], *, package_identity: str,
                           seed: int = CALIBRATION_SEED) -> list[dict[str, Any]]:
    """Exactly 8 observations per live sample, each with a deterministic region."""
    plan: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item.sample_id):
        for transform in TRANSFORM_SUITE:
            digest = observation_digest(row.sample_id, transform["slot"],
                                        package_identity=package_identity, seed=seed)
            plan.append({"observation_id": "obs_" + digest[:20], "dataset": row.dataset,
                         "sample_id": row.sample_id, "transform_slot": int(transform["slot"]),
                         "transform_name": str(transform["name"]), "transform_kind": str(transform["kind"]),
                         "transform_parameter": float(transform["parameter"]),
                         "region": assign_region(digest), "_digest": digest})
    identifiers = [row["observation_id"] for row in plan]
    if len(set(identifiers)) != len(identifiers):
        raise StructuralCalibrationError("duplicate calibration observation ids")
    plan.sort(key=lambda row: row["observation_id"])
    return plan


def plan_digest(plan: list[dict[str, Any]]) -> str:
    """Logical assignment identity: canonical rows, never parquet bytes."""
    canonical = json.dumps([{name: row[name] for name in OBSERVATION_FIELDS}
                            for row in sorted(plan, key=lambda row: row["observation_id"])],
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_observation_plan(plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Prove the assignment really covers what the protocol declares."""
    regions = sorted({row["region"] for row in plan})
    slots = sorted({row["transform_slot"] for row in plan})
    per_slot_datasets = {int(slot): sorted({row["dataset"] for row in plan if row["transform_slot"] == slot})
                         for slot in slots}
    requirements = {
        "exactly_eight_transform_slots": len(slots) == TRANSFORM_SLOTS and slots == list(range(TRANSFORM_SLOTS)),
        "expected_observation_count": len(plan) == EXPECTED_OBSERVATIONS,
        "all_nine_regions_represented": len(regions) == len(REGION_ORDER),
        "both_datasets_in_every_transform": all(
            sorted(ALLOWED_DATASETS) == values for values in per_slot_datasets.values()),
        "every_sample_has_every_slot": all(
            len({row["transform_slot"] for row in plan if row["sample_id"] == sample}) == TRANSFORM_SLOTS
            for sample in {row["sample_id"] for row in plan})}
    return {"observations": len(plan), "live_samples": len({row["sample_id"] for row in plan}),
            "transform_slots": slots, "regions": regions, "region_count": len(regions),
            "observations_by_region": _counts(plan, "region"),
            "observations_by_transform": _counts(plan, "transform_name"),
            "observations_by_dataset": _counts(plan, "dataset"),
            "datasets_by_transform_slot": {str(key): value for key, value in per_slot_datasets.items()},
            "observation_plan_identity_sha256": plan_digest(plan),
            "requirements": requirements, "supports_structural_calibration": all(requirements.values())}


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows: out[str(row[key])] = out.get(str(row[key]), 0) + 1
    return dict(sorted(out.items()))


# --- the calibration edit ------------------------------------------------------

def region_support(store: SampleStore, sample_id: str, region: str) -> np.ndarray:
    """The support mask, from the exact M7/M8 `RegionMaskBuilder`.

    Same parsing-first, geometry-fallback policy and the same validated `crop_box`
    mapping the physics engine and the candidate pipeline use. Coverage is 1.0, so
    the support is the whole canonical region and no sub-sampling seed applies.
    """
    result = store.mask_builder(sample_id).build([region], geometry_shape="flat", coverage=1.0, seed=0)
    return np.asarray(result.operator_support_mask, dtype=np.float32)[0].astype(bool)


def build_calibration_pair(image: np.ndarray, support: np.ndarray, transform: dict[str, Any], *,
                           seed: int) -> dict[str, Any]:
    """`(original_uint8, edited_uint8)` under the M8 discrete convention.

    Outside the support the EXACT original uint8 pixels are copied, and the
    outside-support error is asserted to be exactly 0 after the final conversion.
    This is not a synthetic spoof candidate: it carries no spoof label, no recipe
    id, no GPAT source and no M7 physics artifact.
    """
    from .synthetic_bank import to_uint8
    mask = np.asarray(support).astype(bool)
    original_uint8 = to_uint8(image)
    transformed_uint8 = to_uint8(apply_transform(image, transform, noise_seed=seed))
    edited_uint8 = np.where(mask[:, :, None], transformed_uint8, original_uint8).astype(np.uint8)
    outside = ~mask
    error = int(np.abs(edited_uint8.astype(np.int32) - original_uint8.astype(np.int32))[outside].max()) \
        if outside.any() else 0
    if error != 0:
        raise StructuralCalibrationError(f"outside-support uint8 error {error} is not exactly zero")
    changed = np.any(edited_uint8 != original_uint8, axis=2)
    return {"original_uint8": original_uint8, "edited_uint8": edited_uint8, "support": mask,
            "exact_changed": changed, "outside_support_max_error": error,
            "support_pixels": int(mask.sum()), "changed_pixels": int(changed.sum())}


# --- measurement ---------------------------------------------------------------

class StructuralBackends:
    """SCRFD + FaceXFormer, resolved through the same pinned registry the gate uses.

    `QualityBackends` also loads the 174 MB AdaFace, which takes no part in a
    structural measurement. This exposes the same `detect`/`parse`/`manifest`
    surface with the identity role resolved only for the lock's model manifest.
    """

    def __init__(self, weight_root: Path, *, device: str = "cpu"):
        from .quality_calibration import QualityBackends
        self._inner = QualityBackends(Path(weight_root), device=device)
        self.device = device

    def detect(self, image: np.ndarray) -> tuple[float, np.ndarray | None]: return self._inner.detect(image)
    def parse(self, images: list[np.ndarray]) -> list[np.ndarray]: return self._inner.parse(images)
    def manifest(self) -> dict[str, Any]: return self._inner.manifest()


def preprocessing_contract_identity(role: str) -> str:
    """The pinned M8 preprocessing for one quality role, hashed.

    v3 introduces no alternate resize, alignment or normalization path, so a
    calibration measurement is directly comparable to a candidate's metric.
    """
    spec = PINNED[role]
    if role == "detector":
        payload = {"backend": spec["backend"], "sha256": spec["sha256"], "input_size": spec["input_size"],
                   "threshold": spec["threshold"], "policy": spec["policy"],
                   "input_color": "bgr_uint8", "selection": "highest_scoring_detection",
                   "coordinates": "crop_pixel_space", "landmarks": 5,
                   "normalization": "inter_ocular_distance", "epsilon": 1.0e-6}
    elif role == "parsing":
        payload = {"backend": spec["backend"], "repo_id": spec["repo_id"], "revision": spec["revision"],
                   "sha256": spec["sha256"], "input_size": spec["input_size"],
                   "num_classes": spec["num_classes"], "input_color": "bgr_uint8", "resize": "bicubic",
                   "normalization": "imagenet_mean_std", "task_token": 0, "labels": "lapa_11",
                   "wrapper": "prism_fas.data.package.model_priors.FaceXFormerBackend"}
    else:
        raise StructuralCalibrationError(f"no preprocessing contract declared for role {role!r}")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class ReferenceCache:
    """One detection and one parse per live sample, keyed by image CONTENT.

    280 live samples stand behind 2240 observations and behind the cross-record
    diagnostic, and a reference is a pure function of the sample, so keying by
    content rather than by path means the cache cannot depend on where the package
    is mounted and can only change timing, never values.
    """

    def __init__(self, store: SampleStore, backends: Any):
        self.store, self.backends = store, backends
        self._by_content: dict[str, dict[str, Any]] = {}
        self._by_sample: dict[str, str] = {}
        self.detections = 0
        self.parses = 0

    def reference(self, sample_id: str) -> dict[str, Any]:
        from .contracts import array_hash
        from .synthetic_bank import from_uint8, to_uint8
        if sample_id not in self._by_sample:
            image, _ = self.store.load(sample_id)
            self._by_sample[sample_id] = array_hash(image)
        content = self._by_sample[sample_id]
        cached = self._by_content.get(content)
        if cached is not None: return cached
        image, _ = self.store.load(sample_id)
        # The candidate evaluator builds its reference from the uint8 round trip of
        # the original, so the calibration reference is built the same way.
        original_uint8 = to_uint8(image)
        as_float = from_uint8(original_uint8)
        score, landmarks = self.backends.detect(as_float)
        entry = {"detection_score": float(score), "landmarks": landmarks,
                 "parsing": self.backends.parse([as_float])[0], "original_uint8": original_uint8}
        self.detections += 1
        self.parses += 1
        self._by_content[content] = entry
        return entry

    def report(self) -> dict[str, Any]:
        return {"unique_samples": len(self._by_sample), "unique_image_contents": len(self._by_content),
                "reference_detections": self.detections, "reference_parses": self.parses,
                "cache_key": "processed_image_content_identity", "cached_by_absolute_path": False}


def _distribution(values: list[float]) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    if not series.size: return {}
    return {"count": int(series.size), "min": float(series.min()),
            "p01": float(np.percentile(series, 1.0)), "p05": float(np.percentile(series, 5.0)),
            "p50": float(np.percentile(series, 50.0)), "mean": float(series.mean()),
            "p90": float(np.percentile(series, 90.0)), "p95": float(np.percentile(series, 95.0)),
            "p99": float(np.percentile(series, 99.0)), "max": float(series.max()),
            "std": float(series.std(ddof=0))}


def _grouped(rows: list[dict[str, Any]], key: str, metric: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None: continue
        groups.setdefault(str(row[key]), []).append(float(value))
    return {name: _distribution(values) for name, values in sorted(groups.items())}


# --- the protocol --------------------------------------------------------------

def calibrate_structural(package_root: Path, config: dict[str, Any], backends: Any, *,
                         v2_thresholds: Thresholds, progress: Callable[[dict[str, Any]], None] | None = None,
                         device_report: dict[str, Any] | None = None,
                         limit_samples: int | None = None,
                         cross_record_pairs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Fit `tau_lm_v3` and `tau_parse_v3` from same-image localized benign edits and
    freeze them by the pre-declared percentile rules.

    No generated candidate, no `source_dev`, no target, no raw dataset path, no M7
    physics operator and no GPAT checkpoint participates, so the thresholds cannot
    have been tuned on an acceptance count.
    """
    from .identity_calibration import live_rows
    block = config["structural_calibration"]
    seed = int(block.get("seed", CALIBRATION_SEED))
    audit = SourceOnlyAudit()
    store = SampleStore.open(Path(package_root), audit)
    rows = live_rows(package_root, audit)
    if limit_samples is not None: rows = rows[:limit_samples]
    package_identity = json.loads(
        (Path(package_root) / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))["content_identity_sha256"]

    plan = build_observation_plan(rows, package_identity=package_identity, seed=seed)
    plan_audit = audit_observation_plan(plan)
    if limit_samples is None and not plan_audit["supports_structural_calibration"]:
        raise StructuralCalibrationError(
            f"the observation plan does not satisfy the declared requirements: {plan_audit['requirements']}")

    cache = ReferenceCache(store, backends)
    measured: list[dict[str, Any]] = []
    landmark_values: list[float] = []
    dice_values: list[float] = []
    detection_failures: list[dict[str, Any]] = []
    mask_failures: list[dict[str, Any]] = []
    for position, entry in enumerate(plan, 1):
        reference = cache.reference(entry["sample_id"])
        image, _ = store.load(entry["sample_id"])
        row: dict[str, Any] = {name: entry[name] for name in OBSERVATION_FIELDS}
        try:
            support = region_support(store, entry["sample_id"], entry["region"])
        except Exception as error:                      # a region that cannot be built is recorded, not dropped
            mask_failures.append({"observation_id": entry["observation_id"], "region": entry["region"],
                                  "reason": type(error).__name__})
            row.update({"support_pixels": 0, "changed_pixels": 0, "outside_support_max_error": 0,
                        "landmark_nme": None, "landmark_valid": False, "landmark_failure": "region_mask",
                        "outside_support_parsing_dice": None, "outside_exact_parsing_dice": None})
            measured.append(row)
            continue
        pair = build_calibration_pair(image, support, dict(TRANSFORM_SUITE[entry["transform_slot"]]),
                                      seed=noise_seed(entry["_digest"]))
        from .synthetic_bank import from_uint8
        edited = from_uint8(pair["edited_uint8"])
        score, landmarks = backends.detect(edited)
        parsed = backends.parse([edited])[0]
        valid = landmarks is not None and reference["landmarks"] is not None
        failure = None if valid else ("edited_no_detection" if reference["landmarks"] is not None
                                      else "original_no_detection")
        if not valid:
            detection_failures.append({"observation_id": entry["observation_id"], "reason": failure,
                                       "edited_detection_score": float(score),
                                       "reference_detection_score": float(reference["detection_score"])})
        nme = landmark_nme(landmarks, reference["landmarks"]) if valid else None
        outside_support = ~pair["support"]
        outside_exact = ~pair["exact_changed"]
        dice = parsing_dice(parsed, reference["parsing"], outside_support)
        row.update({
            "support_pixels": pair["support_pixels"], "changed_pixels": pair["changed_pixels"],
            "outside_support_max_error": pair["outside_support_max_error"],
            "landmark_nme": None if nme is None else float(nme), "landmark_valid": bool(valid),
            "landmark_failure": failure,
            "edited_detection_score": float(score),
            "reference_detection_score": float(reference["detection_score"]),
            "outside_support_parsing_dice": float(dice),
            # Declared diagnostic: the candidate gate scores Dice outside the EXACT
            # changed-pixel mask, which also contains the in-support pixels that
            # survived quantization unchanged. Measured, not assumed.
            "outside_exact_parsing_dice": float(parsing_dice(parsed, reference["parsing"], outside_exact))})
        measured.append(row)
        if nme is not None and np.isfinite(nme): landmark_values.append(float(nme))
        if np.isfinite(dice): dice_values.append(float(dice))
        if progress and (position == 1 or position % 100 == 0 or position == len(plan)):
            progress({"stage": "structural", "done": position, "total": len(plan)})

    if not landmark_values: raise StructuralCalibrationError("the calibration produced no valid landmark comparison")
    if not dice_values: raise StructuralCalibrationError("the calibration produced no valid parsing comparison")
    valid_fraction = len(landmark_values) / len(plan)
    minimum_fraction = float(block.get("minimum_valid_landmark_fraction", MINIMUM_VALID_LANDMARK_FRACTION))
    if limit_samples is None and valid_fraction < minimum_fraction:
        raise StructuralCalibrationError(
            f"only {valid_fraction:.4f} of observations produced a valid landmark comparison, "
            f"below the declared minimum {minimum_fraction}")

    landmark_percentile = float(block.get("landmark_percentile", TAU_LM_PERCENTILE))
    parsing_percentile = float(block.get("parsing_percentile", TAU_PARSE_PERCENTILE))
    tau_lm_v3 = float(np.percentile(landmark_values, landmark_percentile))
    tau_parse_v3 = float(np.percentile(dice_values, parsing_percentile))
    thresholds = Thresholds(tau_fd=v2_thresholds.tau_fd, tau_id=v2_thresholds.tau_id, tau_lm=tau_lm_v3,
                            tau_parse=tau_parse_v3, tau_out=v2_thresholds.tau_out, tau_fp=v2_thresholds.tau_fp)

    diagnostic = cross_record_diagnostic(cross_record_pairs or [], cache, backends)
    public_rows = [{key: value for key, value in row.items()} for row in measured]
    return {
        "structural_rows_sha256": rows_digest(public_rows),
        "structural_calibration_schema_version": STRUCTURAL_CALIBRATION_SCHEMA_VERSION,
        "calibration_version": STRUCTURAL_CALIBRATION_VERSION, "seed": seed,
        "split": SOURCE_SPLIT, "label": LIVE_LABEL,
        "observation_plan": plan_audit,
        "populations": {"live_samples": len(rows), "observations": len(plan),
                        "transform_slots": TRANSFORM_SLOTS,
                        "live_by_dataset": _counts([{"dataset": row.dataset, "sample_id": row.sample_id}
                                                    for row in rows], "dataset"),
                        "population_sha256": _digest(*sorted(row.sample_id for row in rows))},
        "transform_suite": [dict(entry) for entry in TRANSFORM_SUITE],
        "transform_suite_sha256": _digest(json.dumps([dict(entry) for entry in TRANSFORM_SUITE],
                                                     sort_keys=True, separators=(",", ":"))),
        "observation_plan_identity_sha256": plan_audit["observation_plan_identity_sha256"],
        "landmark": {
            "total_observations": len(plan), "valid_comparisons": len(landmark_values),
            "face_detection_failures": len(detection_failures),
            "region_mask_failures": len(mask_failures),
            "valid_fraction": valid_fraction, "minimum_valid_fraction": minimum_fraction,
            "percentile": landmark_percentile, "distribution": _distribution(landmark_values),
            "by_dataset": _grouped(measured, "dataset", "landmark_nme"),
            "by_transform": _grouped(measured, "transform_name", "landmark_nme"),
            "by_region": _grouped(measured, "region", "landmark_nme"),
            "detection_failures": detection_failures[:50]},
        "parsing": {
            "total_observations": len(plan), "valid_comparisons": len(dice_values),
            "percentile": parsing_percentile, "distribution": _distribution(dice_values),
            "by_dataset": _grouped(measured, "dataset", "outside_support_parsing_dice"),
            "by_transform": _grouped(measured, "transform_name", "outside_support_parsing_dice"),
            "by_region": _grouped(measured, "region", "outside_support_parsing_dice"),
            "outside_exact_diagnostic_distribution": _distribution(
                [float(row["outside_exact_parsing_dice"]) for row in measured
                 if row.get("outside_exact_parsing_dice") is not None]),
            "outside_exact_is_a_diagnostic_only": True},
        "discrete_invariants": {
            "outside_support_max_error_max": max(
                [int(row.get("outside_support_max_error", 0)) for row in measured] or [0]),
            "observations_with_a_nonzero_outside_error": sum(
                1 for row in measured if int(row.get("outside_support_max_error", 0)) != 0),
            "observations_with_an_empty_change": sum(
                1 for row in measured if int(row.get("changed_pixels", 0)) == 0)},
        "landmark_percentile": landmark_percentile, "parsing_percentile": parsing_percentile,
        "tau_lm_v3": tau_lm_v3, "tau_parse_v3": tau_parse_v3,
        "threshold_rule": ("tau_lm_v3 = p99(same-image structural landmark NME); "
                           "tau_parse_v3 = p01(outside-support parsing Dice)"),
        "tau_lm_v1_superseded": 0.002135227532959269,
        "tau_parse_v1_superseded": 0.8747814437904173,
        "cross_record_diagnostic": diagnostic,
        "cross_record_pairs_set_no_threshold": True,
        "thresholds": thresholds.as_dict(), "threshold_sha256": thresholds.sha256(),
        "unchanged_from_v2": {"tau_fd": v2_thresholds.tau_fd, "tau_id": v2_thresholds.tau_id,
                              "tau_out": v2_thresholds.tau_out, "tau_fp": v2_thresholds.tau_fp},
        "preprocessing_contract_identity_sha256": {
            "detector": preprocessing_contract_identity("detector"),
            "parsing": preprocessing_contract_identity("parsing")},
        "quality_models": backends.manifest(),
        "reference_cache": cache.report(),
        "device_report": device_report or {"device": getattr(backends, "device", "unknown")},
        "uses_m7_physics_operators": False, "uses_gpat": False,
        "used_generated_candidates": False, "used_source_dev": False, "used_target": False,
        "used_raw_dataset_paths": False,
        "source_isolation": audit.report(),
        "_rows": public_rows}


def cross_record_diagnostic(pairs: list[dict[str, Any]], cache: ReferenceCache, backends: Any) -> dict[str, Any]:
    """Landmark NME and parsing Dice between DIFFERENT source records of one identity.

    Reported so the size of the gap against same-image jitter is a measurement
    rather than a claim. **It sets no threshold**: two source records differ in
    real pose, expression, head-geometry projection, landmark position, occlusion
    and crop, which is scene variation, not detector jitter caused by an
    appearance-only edit of one input.
    """
    if not pairs:
        return {"pairs": 0, "computed": False,
                "note": "no cross-record pair plan was supplied; the diagnostic is optional and sets no threshold",
                "sets_any_threshold": False}
    values: list[float] = []
    dice: list[float] = []
    skipped = 0
    for pair in pairs:
        left = cache.reference(pair["sample_id_a"])
        right = cache.reference(pair["sample_id_b"])
        if left["landmarks"] is None or right["landmarks"] is None:
            skipped += 1
            continue
        values.append(landmark_nme(left["landmarks"], right["landmarks"]))
        dice.append(parsing_dice(left["parsing"], right["parsing"],
                                 np.ones_like(right["parsing"], dtype=bool)))
    return {"pairs": len(pairs), "computed": True, "valid_comparisons": len(values), "skipped": skipped,
            "landmark_nme_distribution": _distribution(values),
            "whole_image_parsing_dice_distribution": _distribution(dice),
            "sets_any_threshold": False,
            "why_excluded": ("different source records differ in real pose, expression, head-geometry "
                             "projection, landmark position, occlusion and crop/alignment; that is scene "
                             "variation, not detector jitter under an appearance-only edit, so using it as "
                             "tau_lm would make the geometric gate conceptually far too permissive")}


# --- artifacts -----------------------------------------------------------------

PARQUET_FIELDS: tuple[tuple[str, str], ...] = (
    ("observation_id", "string"), ("dataset", "string"), ("sample_id", "string"),
    ("transform_slot", "int32"), ("transform_name", "string"), ("transform_kind", "string"),
    ("transform_parameter", "double"), ("region", "string"), ("support_pixels", "int32"),
    ("changed_pixels", "int32"), ("outside_support_max_error", "int32"),
    ("landmark_nme", "double"), ("landmark_valid", "bool"),
    ("outside_support_parsing_dice", "double"), ("outside_exact_parsing_dice", "double"))


def rows_digest(rows: list[dict[str, Any]]) -> str:
    """Logical row identity over the canonical fields, never over parquet bytes,
    so it is portable across pyarrow versions."""
    canonical = [{name: _canonical(row.get(name), kind) for name, kind in PARQUET_FIELDS}
                 for row in sorted(rows, key=lambda row: row["observation_id"])]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _canonical(value: Any, kind: str) -> Any:
    if value is None: return None
    if kind == "string": return str(value)
    if kind == "int32": return int(value)
    if kind == "bool": return bool(value)
    return float(value)


def write_rows_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    import pyarrow as pa, pyarrow.parquet as pq
    kinds = {"string": pa.string(), "int32": pa.int32(), "double": pa.float64(), "bool": pa.bool_()}
    ordered = sorted(rows, key=lambda row: row["observation_id"])
    schema = pa.schema([(name, kinds[kind]) for name, kind in PARQUET_FIELDS])
    columns = {name: [_canonical(row.get(name), kind) for row in ordered] for name, kind in PARQUET_FIELDS}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), Path(path),
                   compression="none", version="2.6", write_statistics=False)
    return rows_digest(ordered)


def build_lock(payload: dict[str, Any], *, package_identity: str, config: dict[str, Any]) -> dict[str, Any]:
    """The v3 calibration lock. Content identity binds only portable facts."""
    models = payload["quality_models"]["models"]
    lock: dict[str, Any] = {
        "structural_calibration_lock_schema_version": STRUCTURAL_CALIBRATION_SCHEMA_VERSION,
        "calibration_version": STRUCTURAL_CALIBRATION_VERSION,
        "seed": payload["seed"],
        "package_identity": package_identity,
        "source_population_sha256": payload["populations"]["population_sha256"],
        "split": SOURCE_SPLIT, "label": LIVE_LABEL,
        "live_samples": payload["populations"]["live_samples"],
        "observations": payload["populations"]["observations"],
        "scrfd_backend": models["detector"]["backend"],
        "scrfd_weight_sha256": models["detector"]["verified_sha256"],
        "scrfd_input_size": models["detector"]["input_size"],
        "scrfd_threshold": models["detector"]["threshold"],
        "facexformer_backend": models["parsing"]["backend"],
        "facexformer_repo_id": models["parsing"]["repo_id"],
        "facexformer_revision": models["parsing"]["revision"],
        "facexformer_weight_sha256": models["parsing"]["verified_sha256"],
        "preprocessing_contract_identity_sha256": payload["preprocessing_contract_identity_sha256"],
        "config_sha256": config_sha256(config),
        "transform_suite": payload["transform_suite"],
        "transform_suite_sha256": payload["transform_suite_sha256"],
        "region_assignment_identity_sha256": payload["observation_plan_identity_sha256"],
        "regions": list(REGION_ORDER),
        "landmark_percentile": payload["landmark_percentile"],
        "parsing_percentile": payload["parsing_percentile"],
        "tau_lm_v3": payload["tau_lm_v3"], "tau_parse_v3": payload["tau_parse_v3"],
        "threshold_rule": payload["threshold_rule"],
        "inherited_tau_id_v2": payload["unchanged_from_v2"]["tau_id"],
        "inherited_from_v2": payload["unchanged_from_v2"],
        "thresholds": payload["thresholds"], "threshold_sha256": payload["threshold_sha256"],
        "structural_rows_sha256": payload["structural_rows_sha256"],
        "uses_m7_physics_operators": False, "uses_gpat": False,
        "used_generated_candidates": False, "used_source_dev": False, "used_target": False,
        "cross_record_pairs_set_no_threshold": True,
        "source_isolation": payload["source_isolation"]}
    lock["calibration_content_identity_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in lock.items() if key not in IDENTITY_EXCLUDED_FIELDS},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    lock["identity_excluded_fields"] = list(IDENTITY_EXCLUDED_FIELDS)
    return lock


def public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in (
        "structural_calibration_schema_version", "calibration_version", "seed", "populations",
        "observation_plan", "transform_suite", "transform_suite_sha256",
        "observation_plan_identity_sha256", "landmark", "parsing", "discrete_invariants",
        "landmark_percentile", "parsing_percentile", "tau_lm_v3", "tau_parse_v3", "threshold_rule",
        "tau_lm_v1_superseded", "tau_parse_v1_superseded", "cross_record_diagnostic",
        "cross_record_pairs_set_no_threshold", "thresholds", "threshold_sha256", "unchanged_from_v2",
        "preprocessing_contract_identity_sha256", "structural_rows_sha256", "reference_cache",
        "device_report", "source_isolation") if key in payload}


def write_calibration_artifacts(output_root: Path, payload: dict[str, Any], *, package_identity: str,
                                config: dict[str, Any]) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    digest = write_rows_parquet(root / "structural_calibration_v3.parquet", payload["_rows"])
    if digest != payload["structural_rows_sha256"]:
        raise StructuralCalibrationError("structural row identity changed when written")
    public = public_payload(payload)
    lock = build_lock(payload, package_identity=package_identity, config=config)
    atomic_json_write(root / "quality_calibration_v3.json", public)
    atomic_json_write(root / "structural_calibration_v3_summary.json", _summary(public))
    atomic_json_write(root / "STRUCTURAL_CALIBRATION_V3_LOCK.json", lock)
    return {"lock": lock, "calibration": public,
            "written": ["structural_calibration_v3.parquet", "quality_calibration_v3.json",
                        "structural_calibration_v3_summary.json", "STRUCTURAL_CALIBRATION_V3_LOCK.json"]}


def compare_calibrations(first: dict[str, Any], second: dict[str, Any], *,
                         tolerance: float = METRIC_TOLERANCE) -> dict[str, Any]:
    """The two-run determinism requirement.

    Derived thresholds, the threshold SHA and the logical identities must match
    EXACTLY; the tolerance applies only to per-observation metric values, so a real
    device change is reported rather than silently absorbed.
    """
    mismatches: list[dict[str, Any]] = []
    for name in ("observation_plan_identity_sha256", "structural_rows_sha256", "transform_suite_sha256",
                 "tau_lm_v3", "tau_parse_v3", "threshold_sha256"):
        if first.get(name) != second.get(name):
            mismatches.append({"field": name, "first": first.get(name), "second": second.get(name)})
    left = {row["observation_id"]: row for row in first.get("_rows", [])}
    right = {row["observation_id"]: row for row in second.get("_rows", [])}
    if sorted(left) != sorted(right):
        mismatches.append({"field": "observation_ids", "first": len(left), "second": len(right)})
    else:
        for field in ("region", "transform_name", "support_pixels", "changed_pixels", "landmark_valid"):
            differing = [key for key in left if left[key].get(field) != right[key].get(field)]
            if differing:
                mismatches.append({"field": f"rows:{field}", "differing": len(differing),
                                   "examples": sorted(differing)[:5]})
        for field in ("landmark_nme", "outside_support_parsing_dice"):
            worst, worst_id = 0.0, None
            for key in left:
                a, b = left[key].get(field), right[key].get(field)
                if a is None or b is None:
                    if a is not b: mismatches.append({"field": f"rows:{field}:presence", "observation_id": key})
                    continue
                difference = abs(float(a) - float(b))
                if difference > worst: worst, worst_id = difference, key
            if worst > tolerance:
                mismatches.append({"field": f"rows:{field}", "max_abs_difference": worst,
                                   "observation_id": worst_id, "tolerance": tolerance})
    return {"identical": not mismatches, "mismatches": mismatches[:20], "mismatch_count": len(mismatches),
            "metric_tolerance": tolerance, "thresholds_compared_exactly": True}


def build_quality_gate_v3(v1_payload: dict[str, Any], v2_payload: dict[str, Any], v3_payload: dict[str, Any],
                          v2_lock: dict[str, Any], v3_lock: dict[str, Any]) -> dict[str, Any]:
    """The gate artifact the bank actually loads: v3 thresholds on top of the v1
    calibration and the v2 identity threshold.

    Only `tau_lm` and `tau_parse` are re-fitted, so the fingerprint reference, the
    pinned model manifest, `tau_fd`, `tau_out`, `tau_fp` and `tau_id_v2` are carried
    over verbatim. Merging them into one file keeps a single loadable artifact whose
    SHA-256 is the bank's `quality_calibration_sha256`.
    """
    merged = {key: value for key, value in v1_payload.items() if not key.startswith("_")}
    merged["thresholds"] = dict(v3_payload["thresholds"])
    merged["threshold_sha256"] = v3_payload["threshold_sha256"]
    merged["quality_gate_version"] = "m8-quality-gate-v3"
    merged["identity_calibration_version"] = "m8-identity-calibration-v2"
    merged["identity_calibration_content_identity_sha256"] = v2_lock["calibration_content_identity_sha256"]
    merged["structural_calibration_version"] = STRUCTURAL_CALIBRATION_VERSION
    merged["structural_calibration_content_identity_sha256"] = v3_lock["calibration_content_identity_sha256"]
    merged["observation_plan_identity_sha256"] = v3_payload["observation_plan_identity_sha256"]
    merged["structural_rows_sha256"] = v3_payload["structural_rows_sha256"]
    merged["transform_suite_sha256"] = v3_payload["transform_suite_sha256"]
    merged["tau_id_v2"] = v2_payload["tau_id_v2"]
    merged["tau_lm_v3"] = v3_payload["tau_lm_v3"]
    merged["tau_parse_v3"] = v3_payload["tau_parse_v3"]
    merged["threshold_rule"] = v3_payload["threshold_rule"]
    merged["tau_id_v1_superseded"] = v1_payload["thresholds"]["tau_id"]
    merged["tau_lm_v1_superseded"] = v1_payload["thresholds"]["tau_lm"]
    merged["tau_parse_v1_superseded"] = v1_payload["thresholds"]["tau_parse"]
    merged["threshold_sha256_v1_superseded"] = v1_payload["threshold_sha256"]
    merged["threshold_sha256_v2_superseded"] = v2_payload["threshold_sha256"]
    merged["inherited_from_v1"] = ["tau_fd", "tau_out", "tau_fp", "fingerprint", "quality_models"]
    merged["inherited_from_v2"] = ["tau_id"]
    merged["used_generated_candidates"] = False
    return merged
