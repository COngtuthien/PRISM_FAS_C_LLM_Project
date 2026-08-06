from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Callable
import numpy as np
import yaml
from prism_fas.utils.core import atomic_json_write
from .fingerprint import fingerprint_features, fit_fingerprint_reference
from .m8_pipeline import SampleStore, SourceOnlyAudit
from .pair_plan import SOURCE_SPLIT
from .quality_gate import Thresholds, landmark_nme, parsing_dice
from .quality_models import PINNED, QualityModelRegistry

CALIBRATION_SCHEMA_VERSION = "m8-quality-calibration-v1"
# Mild, fixed before calibration. JPEG Q=95 is deliberately excluded: cv2 JPEG
# encoding is not guaranteed byte-identical across OpenCV builds, so it cannot
# be proven deterministic here. No M7 spoof-producing operator is used.
BENIGN_VARIANTS = ({"name": "brightness_098", "brightness": 0.98, "contrast": 1.0},
                   {"name": "brightness_102", "brightness": 1.02, "contrast": 1.0},
                   {"name": "contrast_098", "brightness": 1.0, "contrast": 0.98},
                   {"name": "contrast_102", "brightness": 1.0, "contrast": 1.02})
BENIGN_NOISE_STD = 0.002
TAU_ID_PERCENTILE = 1.0
TAU_LM_PERCENTILE = 99.0
TAU_PARSE_PERCENTILE = 1.0


class CalibrationError(RuntimeError):
    """Source-only quality calibration could not be completed as declared."""


def load_quality_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise CalibrationError("quality config must be a YAML mapping")
    population = payload.get("calibration_population", {})
    if population.get("split") != SOURCE_SPLIT:
        raise CalibrationError(f"calibration population split must be {SOURCE_SPLIT!r}")
    for forbidden in ("source_dev", "target_test"):
        if forbidden in json.dumps(population): raise CalibrationError(f"calibration config references {forbidden!r}")
    return payload


def config_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def benign_variant(image: np.ndarray, variant: dict[str, Any], *, sample_id: str, noise_std: float) -> np.ndarray:
    """Deterministic mild perturbation of a real live crop.

    Seeded per (sample, variant) with a local PCG64, so the benign population is
    reproducible and never depends on global RNG state.
    """
    values = np.asarray(image, dtype=np.float32)
    out = values * np.float32(variant["brightness"])
    mean = float(out.mean())
    out = (out - mean) * np.float32(variant["contrast"]) + mean
    seed = int.from_bytes(hashlib.sha256(f"benign|{sample_id}|{variant['name']}".encode("utf-8")).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed % (2 ** 32)))
    out = out + rng.normal(0.0, float(noise_std), size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(image, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb.transpose(1, 2, 0)[:, :, ::-1])


class QualityBackends:
    """The three pinned models, loaded once and reused. Never substituted."""
    def __init__(self, weight_root: Path, *, device: str = "cpu"):
        from prism_fas.data.preprocess_m2 import SCRFDDetector
        from prism_fas.data.package.model_priors import FaceXFormerBackend
        from .quality_models import resolve_weight
        self.registry = QualityModelRegistry.resolve(weight_root, roles=("identity", "parsing", "detector"))
        self.device = device
        provider = "CUDAExecutionProvider" if device.startswith("cuda") else "CPUExecutionProvider"
        try:
            self.detector = SCRFDDetector(resolve_weight(weight_root, "detector"),
                                          int(PINNED["detector"]["input_size"]), provider)
        except Exception:
            self.detector = SCRFDDetector(resolve_weight(weight_root, "detector"),
                                          int(PINNED["detector"]["input_size"]), "CPUExecutionProvider")
        self.parsing = FaceXFormerBackend(weight_path=resolve_weight(weight_root, "parsing"),
                                          code_root=Path(weight_root) / "code" / "facexformer", device=device)
        self.identity = self.registry.adaface(device)

    def detect(self, image: np.ndarray) -> tuple[float, np.ndarray | None]:
        """Best detection score plus its 5 landmarks in crop pixel coordinates."""
        detections = self.detector.detect(_to_bgr_uint8(image))
        if not detections: return 0.0, None
        best = max(detections, key=lambda item: item.score)
        return float(best.score), np.asarray(best.landmarks, dtype=np.float32)

    def parse(self, images: list[np.ndarray]) -> list[np.ndarray]:
        return [result["parsing_labels"] for result in self.parsing.infer([_to_bgr_uint8(image) for image in images])]

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        import torch
        stacked = torch.from_numpy(np.stack(images).astype(np.float32)).to(self.device)
        with torch.no_grad():
            return self.identity(stacked).cpu().numpy().astype(np.float32)

    def manifest(self) -> dict[str, Any]: return self.registry.manifest()


def calibrate(package_root: Path, config: dict[str, Any], backends: QualityBackends, *,
              limit_live: int | None = None, limit_spoof: int | None = None,
              progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Fit every M8 threshold from `source_train` alone."""
    audit = SourceOnlyAudit()
    store = SampleStore.open(package_root, audit)
    rows = sorted(store._rows.values(), key=lambda row: row["sample_id"])
    live = [row for row in rows if row["label_live_spoof"] == "live"]
    spoof = [row for row in rows if row["label_live_spoof"] == "spoof"]
    if limit_live is not None: live = live[:limit_live]
    if limit_spoof is not None: spoof = spoof[:limit_spoof]
    if not live or not spoof: raise CalibrationError("source_train must supply both live and spoof populations")
    noise_std = float(config["benign"]["gaussian_noise_std"])
    if noise_std > 0.002 + 1e-12: raise CalibrationError("benign Gaussian noise std must stay <= 0.002")

    identity_cosines: list[float] = []
    landmark_errors: list[float] = []
    dice_scores: list[float] = []
    for position, row in enumerate(live, 1):
        image, _ = store.load(row["sample_id"])
        score, landmarks = backends.detect(image)
        reference_parse = backends.parse([image])[0]
        reference_embedding = backends.embed([image])[0]
        variants = [benign_variant(image, variant, sample_id=row["sample_id"], noise_std=noise_std)
                    for variant in BENIGN_VARIANTS]
        embeddings = backends.embed(variants)
        parses = backends.parse(variants)
        for index, variant_image in enumerate(variants):
            identity_cosines.append(float(np.dot(embeddings[index], reference_embedding)))
            dice_scores.append(parsing_dice(parses[index], reference_parse, np.ones_like(reference_parse, dtype=bool)))
            _, variant_landmarks = backends.detect(variant_image)
            if landmarks is not None and variant_landmarks is not None:
                landmark_errors.append(landmark_nme(variant_landmarks, landmarks))
        if progress and (position == 1 or position % 25 == 0 or position == len(live)):
            progress({"stage": "benign", "done": position, "total": len(live)})
    if not identity_cosines or not dice_scores: raise CalibrationError("benign population produced no metrics")
    if not landmark_errors: raise CalibrationError("benign population produced no landmark pairs")

    vectors, domains, records = [], [], []
    for position, row in enumerate(spoof, 1):
        image, _ = store.load(row["sample_id"])
        vectors.append(fingerprint_features(image))
        domains.append(row["dataset"]); records.append(row["source_record_id"])
        if progress and (position == 1 or position % 200 == 0 or position == len(spoof)):
            progress({"stage": "fingerprint", "done": position, "total": len(spoof)})
    reference = fit_fingerprint_reference(np.stack(vectors), domains, records)

    thresholds = Thresholds(
        tau_fd=float(PINNED["detector"]["threshold"]),      # pinned production threshold, never fitted
        tau_id=float(np.percentile(identity_cosines, TAU_ID_PERCENTILE)),
        tau_lm=float(np.percentile(landmark_errors, TAU_LM_PERCENTILE)),
        tau_parse=float(np.percentile(dice_scores, TAU_PARSE_PERCENTILE)),
        tau_out=0.0,
        tau_fp=float(reference["tau_fp"]))
    population_hash = hashlib.sha256("|".join(sorted(row["sample_id"] for row in live + spoof)).encode("utf-8")).hexdigest()
    return {
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "device": backends.device,
        "populations": {"live": len(live), "spoof": len(spoof),
                        "benign": len(live) * len(BENIGN_VARIANTS),
                        "live_per_dataset": _counts(live), "spoof_per_dataset": _counts(spoof),
                        "split": SOURCE_SPLIT, "population_sha256": population_hash},
        "benign_policy": {"variants": [variant["name"] for variant in BENIGN_VARIANTS],
                          "gaussian_noise_std": noise_std, "jpeg_q95_used": False,
                          "jpeg_exclusion_reason": "cv2 JPEG encoding is not provably byte-deterministic across builds",
                          "uses_m7_spoof_operators": False, "deterministic": True},
        "raw_percentiles": {"identity_cosine_p1": float(np.percentile(identity_cosines, TAU_ID_PERCENTILE)),
                            "landmark_nme_p99": float(np.percentile(landmark_errors, TAU_LM_PERCENTILE)),
                            "parsing_dice_p1": float(np.percentile(dice_scores, TAU_PARSE_PERCENTILE)),
                            "identity_cosine_min": float(np.min(identity_cosines)),
                            "landmark_nme_max": float(np.max(landmark_errors)),
                            "parsing_dice_min": float(np.min(dice_scores)),
                            "landmark_pairs": len(landmark_errors)},
        "thresholds": thresholds.as_dict(), "threshold_sha256": thresholds.sha256(),
        "tau_fd_source": "pinned SCRFD production threshold (scrfd_source_policy_v1); not fitted on generated images",
        "fingerprint": reference,
        "quality_models": backends.manifest(),
        "source_isolation": audit.report(),
        "used_source_dev": False, "used_target": False, "used_generated_candidates": False}


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows: out[row["dataset"]] = out.get(row["dataset"], 0) + 1
    return dict(sorted(out.items()))


def write_calibration(path: Path, payload: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    enriched = {**payload, "calibration_config_sha256": config_sha(config)}
    atomic_json_write(Path(path), enriched)
    return enriched


def load_thresholds(path: Path) -> Thresholds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Thresholds.from_dict(payload["thresholds"])
