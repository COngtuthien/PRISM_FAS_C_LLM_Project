"""M8 synthetic bank generation: discrete uint8 finalization, the two generation
routes and the resume-safe candidate pipeline.

Source-only. This module opens the `source_train` manifest, `source_train` images
and priors, the frozen M7 recipe bank, the frozen GPAT checkpoint and the pinned
quality weights — nothing else. It must never import modal.
"""
from __future__ import annotations
import hashlib, io, json, zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import numpy as np
from prism_fas.recipes.compile import compile_recipe
from prism_fas.recipes.conditioning import conditioning_vector
from prism_fas.utils.core import atomic_json_write
# `_atomic_bytes` is the M7 atomic binary writer; reused rather than duplicated.
from .audit import _atomic_bytes as atomic_bytes_write
from .candidate_plan import load_candidate_plan
from .fingerprint import fingerprint_features, fingerprint_score
from .m8_pipeline import SampleStore, SourceOnlyAudit
from .physics import PHYSICS_ENGINE_VERSION, PhysicsEngine
from .quality_gate import Thresholds, evaluate, landmark_nme, parsing_dice, support_overlap

SYNTHETIC_BANK_SCHEMA_VERSION = "m8-synthetic-bank-v1"
CANDIDATE_RECORD_SCHEMA_VERSION = "m8-synthetic-candidate-v1"
GENERATOR_VERSION = "m8-synthetic-generator-v1"
IMAGE_SIZE = 224
ARTIFACT_MAP_KEY = "artifact_map"
# The single documented float -> uint8 convention, applied identically to the
# generated image and to the original live image.
DISCRETE_CONVENTION = "round_half_up_clip_0_255"
TERMINAL_STATES = ("accepted", "rejected", "failed_generation")
# SCRFD found no face on the generated image: the landmark error is undefined, so
# a finite sentinel of one inter-ocular distance is recorded. Every hard gate that
# depends on detection already fails in that case.
LANDMARK_NME_NO_DETECTION = 1.0
ROUTES = ("physics", "gpat")


class SyntheticBankError(RuntimeError):
    """A synthetic-bank stage could not be completed as declared."""


# --- discrete uint8 output -------------------------------------------------

def to_uint8(image: np.ndarray) -> np.ndarray:
    """`[3,H,W]` float in [0,1] -> `[H,W,3]` uint8 RGB, round-half-up then clip.

    This is the ONE conversion used for both the generated image and the original
    live image, so an unchanged pixel is byte-identical by construction.
    """
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 3: raise SyntheticBankError(f"expected [3,H,W], got {values.shape}")
    if not np.isfinite(values).all(): raise SyntheticBankError("image contains non-finite values")
    scaled = np.floor(np.clip(values, 0.0, 1.0).astype(np.float64) * 255.0 + 0.5)
    return np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8).transpose(1, 2, 0))


def from_uint8(image: np.ndarray) -> np.ndarray:
    """`[H,W,3]` uint8 RGB -> `[3,H,W]` float32 in [0,1]."""
    return np.ascontiguousarray(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0))


def encode_png(image: np.ndarray) -> bytes:
    """Lossless PNG. `[H,W,3]` uint8 RGB or `[H,W]` uint8 grey."""
    import cv2
    values = np.asarray(image, dtype=np.uint8)
    payload = values[:, :, ::-1] if values.ndim == 3 else values
    ok, buffer = cv2.imencode(".png", payload, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    if not ok: raise SyntheticBankError("PNG encoding failed")
    return bytes(buffer.tobytes())


def decode_png(data: bytes) -> np.ndarray:
    import cv2
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None: raise SyntheticBankError("PNG decoding failed")
    return np.ascontiguousarray(decoded[:, :, ::-1]) if decoded.ndim == 3 else np.ascontiguousarray(decoded)


def encode_npz(arrays: dict[str, np.ndarray]) -> bytes:
    """Deterministic compressed NPZ.

    `np.savez_compressed` stamps the local wall clock into every zip entry, which
    would make byte-identical reruns impossible, so the archive is written with a
    fixed member date and mode.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.getvalue())
    return buffer.getvalue()


def decode_npz(data: bytes, key: str = ARTIFACT_MAP_KEY) -> np.ndarray:
    with np.load(io.BytesIO(data), allow_pickle=False) as handle:
        return np.asarray(handle[key])


@dataclass(frozen=True)
class DiscreteResult:
    """The finalized, byte-exact candidate payload."""
    image_uint8: np.ndarray            # [H,W,3] RGB
    original_uint8: np.ndarray         # [H,W,3] RGB
    exact_edit_mask: np.ndarray        # [H,W] bool — ACTUAL changed uint8 pixels
    artifact_map: np.ndarray           # [1,H,W] float16, exactly 0 outside the exact mask
    image_png: bytes
    mask_png: bytes
    artifact_map_npz: bytes
    outside_mask_max_error: int
    requested_support_pixels: int
    exact_mask_pixels: int

    @property
    def image_sha256(self) -> str: return hashlib.sha256(self.image_png).hexdigest()
    @property
    def mask_sha256(self) -> str: return hashlib.sha256(self.mask_png).hexdigest()
    @property
    def artifact_map_sha256(self) -> str: return hashlib.sha256(self.artifact_map_npz).hexdigest()


def finalize_discrete(generated: np.ndarray, original: np.ndarray, requested_support: np.ndarray,
                      artifact_map: np.ndarray) -> DiscreteResult:
    """The frozen finalization procedure of `docs/M8_SYNTHETIC_BANK_CONTRACT.md` §3.

    The exact mask is the set of pixels that really differ AFTER quantization, not
    the requested support, so an artifact too weak to survive uint8 rounding
    produces an empty mask and is rejected rather than silently counted.
    """
    values = np.asarray(generated, dtype=np.float32)
    if not np.isfinite(values).all(): raise SyntheticBankError("generated image is not finite")
    values = np.clip(values, 0.0, 1.0).astype(np.float32)
    if values.shape != (3, IMAGE_SIZE, IMAGE_SIZE): raise SyntheticBankError(f"generated image shape {values.shape}")
    support = np.asarray(requested_support).astype(bool).reshape(IMAGE_SIZE, IMAGE_SIZE)

    generated_uint8 = to_uint8(values)                                   # 1
    original_uint8 = to_uint8(original)                                  # 2
    composited = np.where(support[:, :, None], generated_uint8, original_uint8).astype(np.uint8)   # 3
    changed = np.any(composited != original_uint8, axis=2)               # 4
    exact = changed                                                      # 5
    final = np.where(exact[:, :, None], composited, original_uint8).astype(np.uint8)               # 6

    strength = np.asarray(artifact_map, dtype=np.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE)
    if not np.isfinite(strength).all(): raise SyntheticBankError("artifact map is not finite")
    strength = np.where(exact[None], np.clip(strength, 0.0, 1.0), np.float32(0.0)).astype(np.float16)
    if not np.isfinite(strength.astype(np.float32)).all(): raise SyntheticBankError("artifact map is not finite after cast")

    image_png = encode_png(final)                                        # 7
    decoded = decode_png(image_png)                                      # 8
    if decoded.shape != final.shape or not np.array_equal(decoded, final):
        raise SyntheticBankError("saved PNG did not decode back to the finalized image")
    outside = ~exact                                                     # 9
    error = int(np.abs(decoded.astype(np.int32) - original_uint8.astype(np.int32))[outside].max()) if outside.any() else 0
    if error != 0: raise SyntheticBankError(f"outside-mask uint8 error {error} is not exactly zero")
    mask_png = encode_png((exact.astype(np.uint8) * np.uint8(255)))
    mask_decoded = decode_png(mask_png)
    if set(np.unique(mask_decoded).tolist()) - {0, 255}: raise SyntheticBankError("mask PNG holds values other than 0/255")
    npz = encode_npz({ARTIFACT_MAP_KEY: strength})
    reloaded = decode_npz(npz)
    if reloaded.dtype != np.float16 or reloaded.shape != (1, IMAGE_SIZE, IMAGE_SIZE):
        raise SyntheticBankError("artifact map did not round-trip as float16 [1,224,224]")
    if float(np.abs(reloaded[0][outside]).max() if outside.any() else 0.0) != 0.0:
        raise SyntheticBankError("artifact map is not exactly zero outside the exact mask")
    return DiscreteResult(image_uint8=final, original_uint8=original_uint8, exact_edit_mask=exact,
                          artifact_map=strength, image_png=image_png, mask_png=mask_png,
                          artifact_map_npz=npz, outside_mask_max_error=error,
                          requested_support_pixels=int(support.sum()), exact_mask_pixels=int(exact.sum()))


# --- frozen calibration ----------------------------------------------------

@dataclass(frozen=True)
class FrozenCalibration:
    """The frozen quality policy. Never refitted, never adjusted after seeing
    candidate acceptance counts."""
    thresholds: Thresholds
    references: dict[str, dict[str, list[float]]]
    threshold_sha256: str
    fingerprint_reference_sha256: str
    calibration_config_sha256: str
    calibration_sha256: str
    quality_models: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "FrozenCalibration":
        raw = Path(path).read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        thresholds = Thresholds.from_dict(payload["thresholds"])
        if thresholds.sha256() != payload["threshold_sha256"]:
            raise SyntheticBankError("calibration threshold hash does not match its own thresholds")
        return cls(thresholds=thresholds, references=payload["fingerprint"]["references"],
                   threshold_sha256=str(payload["threshold_sha256"]),
                   fingerprint_reference_sha256=str(payload["fingerprint"]["reference_sha256"]),
                   calibration_config_sha256=str(payload.get("calibration_config_sha256", "")),
                   calibration_sha256=hashlib.sha256(raw).hexdigest(),
                   quality_models=payload.get("quality_models", {}))


def generation_config_sha256(bank_config: dict[str, Any], quality_config: dict[str, Any]) -> str:
    payload = {"synthetic_bank": bank_config, "quality_gate": quality_config,
               "generator_version": GENERATOR_VERSION, "discrete_convention": DISCRETE_CONVENTION,
               "schema_version": SYNTHETIC_BANK_SCHEMA_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# --- generation routes -----------------------------------------------------

@dataclass
class RouteOutput:
    """One route's float result before discrete finalization."""
    image: np.ndarray                 # [3,224,224] float32 in [0,1]
    artifact_map: np.ndarray          # [1,224,224] float32 in [0,1]
    requested_support: np.ndarray     # [224,224] bool
    requested_region_pixels: int
    requested_coverage: float
    achieved_coverage: float
    binding: str
    trace: dict[str, Any] = field(default_factory=dict)


def _support_masks(store: SampleStore, sample_id: str, graph: Any) -> Any:
    """The M7 region masks for one (sample, recipe), built exactly as the physics
    engine and the GPAT trainer build them.

    Memoized on the store: mask building is a pure function of
    (sample, recipe) and is the dominant CPU cost of a generation pass, so the
    cache can only change timing, never values.
    """
    cache = getattr(store, "_region_mask_results", None)
    if cache is None:
        cache = {}
        store._region_mask_results = cache          # type: ignore[attr-defined]
    key = (sample_id, graph.recipe_id)
    hit = cache.get(key)
    if hit is not None: return hit
    policy = graph.region_mask_policy
    result = store.mask_builder(sample_id).build(
        list(graph.requested_regions), geometry_shape=str(policy.get("geometry_shape", "flat")),
        coverage=float(policy.get("requested_coverage", 1.0)),
        seed=graph.node_seed(graph.nodes[0], f"{sample_id}|region_mask"),
        coverage_tolerance=float(policy.get("coverage_tolerance", 0.05)))
    cache[key] = result
    return result


def applied_strength_map(graph: Any, result: Any) -> np.ndarray:
    """The physics route's artifact map, in REQUESTED-STRENGTH units.

    `PhysicsResult.artifact_strength_map` is M7's preview map: the per-pixel RGB
    difference divided by its own peak. Its masked mean measures contrast, not
    strength, so comparing it against `a_recipe` is a units error — a recipe asking
    for 0.09 measured 0.78 in the correctness pilot.

    M7's own per-operator `strength_map` is `support * strength`, which IS in
    recipe-strength units, and the GPAT head is trained against `M_t * a_recipe` in
    those same units. The M8 saved map therefore averages the per-operator applied
    strengths over the graph, so `masked_mean == mean(strengths) == a_recipe` when
    the operators share a support. M7 is not modified: this is an M8-side artifact.
    """
    layers: list[np.ndarray] = []
    for node in graph.nodes:
        support = result.per_operator_support_masks.get(node.operator_name)
        if support is None: continue
        layers.append(np.asarray(support, dtype=np.float32)[0] * np.float32(node.strength))
    if not layers: raise SyntheticBankError(f"{graph.recipe_id}: no operator support to build an artifact map from")
    stacked = np.clip(np.mean(np.stack(layers), axis=0), 0.0, 1.0).astype(np.float32)
    return np.ascontiguousarray(stacked[None])


class PhysicsRoute:
    """The exact final M7 `PhysicsEngine`. No GPAT checkpoint participates."""
    binding = PHYSICS_ENGINE_VERSION

    def __init__(self) -> None:
        self.engine = PhysicsEngine()

    def generate(self, store: SampleStore, bank: dict[str, Any], row: dict[str, Any]) -> RouteOutput:
        recipe = _recipe(bank, row["recipe_id"])
        graph = compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"])
        sample_id = row["live_target_sample_id"]
        image, arrays = store.load(sample_id)
        masks = _support_masks(store, sample_id, graph)
        result = self.engine.apply(image, arrays["parsing_labels"], arrays["landmarks"], arrays["bbox"],
                                   graph, sample_id, crop_box=arrays["crop_box"])
        return RouteOutput(image=np.asarray(result.synthetic_image, dtype=np.float32),
                           artifact_map=applied_strength_map(graph, result),
                           requested_support=np.asarray(masks.operator_support_mask)[0].astype(bool),
                           requested_region_pixels=int(np.asarray(masks.requested_region_mask)[0].astype(bool).sum()),
                           requested_coverage=float(masks.requested_coverage),
                           achieved_coverage=float(masks.achieved_coverage),
                           binding=self.binding,
                           trace={"engine_version": PHYSICS_ENGINE_VERSION,
                                  "operator_order": list(graph.operator_names()),
                                  "graph_hash": graph.graph_hash, "recipe_hash": graph.recipe_hash})


class GPATRoute:
    """The frozen best GPAT checkpoint, bound by its exact SHA-256.

    Inference is fp32, batch-1 and cuDNN-deterministic so a 32-candidate rerun
    reproduces the bytes of the full run exactly.
    """

    def __init__(self, checkpoint_path: Path, config: dict[str, Any], *, expected_sha256: str,
                 expected_identity: dict[str, str], device: str = "cpu",
                 conditioning_control: Any = None):
        """`conditioning_control` is the ONE declared A02 scientific control, or None.

        `None` — every ordinary M8/M9/B08/M10 path — checks the full identity,
        `recipe_bank_identity` included, and fails closed on a mismatch. The control
        is a typed, self-validating object, never a boolean, so a caller cannot ask
        for the exemption without naming the exact frozen identities it applies to.
        """
        import torch
        from .conditioning_control import expected_identity_for
        from .gpat_checkpoint import load_checkpoint, sha256_file
        from .gpat_model import build_gpat_model
        actual = sha256_file(Path(checkpoint_path))
        if actual != expected_sha256:
            raise SyntheticBankError(f"GPAT checkpoint SHA {actual} != frozen {expected_sha256}")
        self.checkpoint_sha256 = actual
        self.binding = actual
        self.device = device
        self.conditioning_control = conditioning_control
        self.model = build_gpat_model(config).to(device)
        payload = load_checkpoint(
            Path(checkpoint_path),
            expected_identity=expected_identity_for(expected_identity, control=conditioning_control))
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        self.architecture_hash = self.model.architecture_hash()
        self.checkpoint_identity = dict(payload.get("identity") or {})
        self.checkpoint_epoch = int(payload.get("epoch", -1))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def generate(self, store: SampleStore, bank: dict[str, Any], row: dict[str, Any]) -> RouteOutput:
        import torch
        recipe = _recipe(bank, row["recipe_id"])
        graph = compile_recipe(recipe, bank["ontology"], bank_id=bank["bank_id"])
        live_id, spoof_id = row["live_target_sample_id"], row["spoof_source_sample_id"]
        live_image, _ = store.load(live_id)
        spoof_image, _ = store.load(spoof_id)
        masks = _support_masks(store, live_id, graph)
        support = np.asarray(masks.operator_support_mask, dtype=np.float32)
        style = store.cached_mask(spoof_id, "spoof", graph, coverage=1.0, seed_scope="style_mask", use_support=False)
        tensor = lambda array: torch.from_numpy(np.asarray(array, dtype=np.float32)[None]).to(self.device)
        with torch.no_grad():
            output = self.model(tensor(live_image), tensor(spoof_image),
                                torch.from_numpy(conditioning_vector(recipe, bank["ontology"])[None]).to(self.device),
                                tensor(support), tensor(style))
            output.validate(tensor(live_image))
        return RouteOutput(image=output.synthetic_image[0].detach().cpu().numpy().astype(np.float32),
                           artifact_map=output.artifact_map[0].detach().cpu().numpy().astype(np.float32),
                           requested_support=support[0].astype(bool),
                           requested_region_pixels=int(np.asarray(masks.requested_region_mask)[0].astype(bool).sum()),
                           requested_coverage=float(masks.requested_coverage),
                           achieved_coverage=float(masks.achieved_coverage),
                           binding=self.checkpoint_sha256,
                           trace={"checkpoint_sha256": self.checkpoint_sha256,
                                  "architecture_hash": self.architecture_hash,
                                  # Present only under the declared A02 control, so a
                                  # sample generated out of the training conditioning
                                  # distribution says so in its own record.
                                  **({"conditioning_control": self.conditioning_control.policy_payload()}
                                     if self.conditioning_control is not None else {}),
                                  "ll_invariant_max_abs_error": output.ll_invariant_error(),
                                  "delta_high_abs_max": float(output.trace["delta_high_abs_max"]),
                                  "graph_hash": graph.graph_hash, "recipe_hash": graph.recipe_hash})


def _recipe(bank: dict[str, Any], recipe_id: str) -> Any:
    for recipe in bank["recipes"]:
        if recipe.recipe_id == recipe_id: return recipe
    raise SyntheticBankError(f"recipe {recipe_id!r} is not in the frozen bank")


# --- per-candidate quality evaluation --------------------------------------

class CandidateEvaluator:
    """Computes every declared quality metric for one finalized candidate.

    Reference detections, parses and embeddings are cached per live sample: there
    are 280 live targets behind 1120 candidates, and a reference is a pure
    function of the sample.
    """

    def __init__(self, backends: Any, calibration: FrozenCalibration):
        self.backends = backends
        self.calibration = calibration
        self._reference: dict[str, dict[str, Any]] = {}

    def reference(self, sample_id: str, original_uint8: np.ndarray) -> dict[str, Any]:
        cached = self._reference.get(sample_id)
        if cached is not None: return cached
        image = from_uint8(original_uint8)
        score, landmarks = self.backends.detect(image)
        entry = {"detection_score": float(score), "landmarks": landmarks,
                 "parsing": self.backends.parse([image])[0], "embedding": self.backends.embed([image])[0]}
        self._reference[sample_id] = entry
        return entry

    def evaluate(self, discrete: DiscreteResult, *, live_target_sample_id: str, requested_strength: float,
                 requested_support: np.ndarray) -> dict[str, Any]:
        reference = self.reference(live_target_sample_id, discrete.original_uint8)
        generated = from_uint8(discrete.image_uint8)
        score, landmarks = self.backends.detect(generated)
        parsed = self.backends.parse([generated])[0]
        embedding = self.backends.embed([generated])[0]
        outside = ~discrete.exact_edit_mask
        exact = discrete.exact_edit_mask
        measured = float(np.asarray(discrete.artifact_map[0], dtype=np.float64)[exact].mean()) if exact.any() else 0.0
        detected = landmarks is not None and reference["landmarks"] is not None
        metrics = {
            "face_detection_score": float(score),
            "identity_cosine": float(np.dot(np.asarray(embedding, dtype=np.float64),
                                            np.asarray(reference["embedding"], dtype=np.float64))),
            "landmark_nme": (landmark_nme(landmarks, reference["landmarks"]) if detected
                             else LANDMARK_NME_NO_DETECTION),
            "outside_mask_parsing_dice": parsing_dice(parsed, reference["parsing"], outside),
            "outside_mask_max_error": float(discrete.outside_mask_max_error),
            "measured_artifact_strength": measured,
            "requested_artifact_strength": float(requested_strength),
            "fingerprint_score": fingerprint_score(fingerprint_features(generated), self.calibration.references),
            "support_overlap": support_overlap(exact, np.asarray(requested_support).astype(bool)),
            "reference_detection_score": float(reference["detection_score"]),
            "landmark_detected": bool(detected)}
        return evaluate(metrics, self.calibration.thresholds)


# --- candidate records -----------------------------------------------------

def _relative_paths(synthetic_id: str) -> dict[str, str]:
    return {"image_relative_path": f"images/{synthetic_id}.png",
            "mask_relative_path": f"masks/{synthetic_id}.png",
            "artifact_map_relative_path": f"artifact_maps/{synthetic_id}.npz"}


# A candidate's identity splits in two. The generation half determines the PAYLOAD
# BYTES; the evaluation half determines only the DECISION. Re-calibrating the
# quality gate changes the second and leaves the first untouched, which is what
# makes a v1 payload reusable under a v2 threshold.
GENERATION_IDENTITY_FIELDS = ("package_identity", "recipe_bank_identity", "candidate_plan_identity",
                              "gpat_checkpoint_sha256", "physics_engine_version", "generator_version",
                              "discrete_convention")
EVALUATION_IDENTITY_FIELDS = ("threshold_sha256", "fingerprint_reference_sha256", "calibration_sha256",
                              "generation_config_sha256")


def generation_identity(identity: dict[str, str]) -> dict[str, str]:
    return {name: identity[name] for name in GENERATION_IDENTITY_FIELDS if name in identity}


def discrete_from_saved(image_png: bytes, mask_png: bytes, artifact_map_npz: bytes,
                        original: np.ndarray) -> DiscreteResult:
    """Rebuild a `DiscreteResult` from saved payload bytes, RE-PROVING every
    invariant instead of trusting the run that wrote them."""
    image = decode_png(image_png)
    if image.ndim != 3 or image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise SyntheticBankError("saved image is not uint8 RGB 224x224")
    original_uint8 = to_uint8(original)
    mask = decode_png(mask_png)
    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE) or set(np.unique(mask).tolist()) - {0, 255}:
        raise SyntheticBankError("saved mask is not a 0/255 224x224 map")
    exact = mask == 255
    changed = np.any(image != original_uint8, axis=2)
    if not np.array_equal(changed, exact):
        raise SyntheticBankError("saved mask does not equal the actual changed uint8 pixels")
    outside = ~exact
    error = int(np.abs(image.astype(np.int32) - original_uint8.astype(np.int32))[outside].max()) if outside.any() else 0
    if error != 0: raise SyntheticBankError(f"saved outside-mask uint8 error {error} is not exactly zero")
    strength = decode_npz(artifact_map_npz)
    if strength.dtype != np.float16 or strength.shape != (1, IMAGE_SIZE, IMAGE_SIZE):
        raise SyntheticBankError("saved artifact map is not float16 [1,224,224]")
    values = np.asarray(strength, dtype=np.float32)
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise SyntheticBankError("saved artifact map is not finite in [0,1]")
    if outside.any() and float(np.abs(values[0][outside]).max()) != 0.0:
        raise SyntheticBankError("saved artifact map is not exactly zero outside the exact mask")
    return DiscreteResult(image_uint8=image, original_uint8=original_uint8, exact_edit_mask=exact,
                          artifact_map=strength, image_png=image_png, mask_png=mask_png,
                          artifact_map_npz=artifact_map_npz, outside_mask_max_error=error,
                          requested_support_pixels=0, exact_mask_pixels=int(exact.sum()))


def payload_is_reusable(record: dict[str, Any], generation: dict[str, str], root: Path) -> tuple[bool, str]:
    """A v1 payload may be reused only when every GENERATION input matches and
    every stored hash still validates."""
    if record.get("schema_version") != CANDIDATE_RECORD_SCHEMA_VERSION: return False, "schema_version"
    if record.get("terminal_state") != "accepted": return False, "no_retained_payload"
    stored = generation_identity(record.get("identity") or {})
    for key, value in generation.items():
        if stored.get(key) != value: return False, f"generation_identity:{key}"
    outputs = record.get("outputs") or {}
    for name, digest_key in (("image_relative_path", "image_sha256"), ("mask_relative_path", "mask_sha256"),
                             ("artifact_map_relative_path", "artifact_map_sha256")):
        path = Path(root) / str(outputs.get(name, ""))
        if not path.is_file(): return False, f"missing:{name}"
        if hashlib.sha256(path.read_bytes()).hexdigest() != outputs.get(digest_key): return False, f"hash:{name}"
    return True, "reused"


def record_is_reusable(record: dict[str, Any], identity: dict[str, str], root: Path) -> tuple[bool, str]:
    """A completed candidate is reused only when every identity input and every
    output hash still validates. A mismatched partial candidate is rebuilt."""
    if record.get("schema_version") != CANDIDATE_RECORD_SCHEMA_VERSION: return False, "schema_version"
    if record.get("terminal_state") not in TERMINAL_STATES: return False, "terminal_state"
    stored = record.get("identity") or {}
    for key, value in identity.items():
        if stored.get(key) != value: return False, f"identity:{key}"
    if record["terminal_state"] != "accepted": return True, "reused"
    outputs = record.get("outputs") or {}
    for name, digest_key in (("image_relative_path", "image_sha256"), ("mask_relative_path", "mask_sha256"),
                             ("artifact_map_relative_path", "artifact_map_sha256")):
        path = Path(root) / str(outputs.get(name, ""))
        if not path.is_file(): return False, f"missing:{name}"
        if hashlib.sha256(path.read_bytes()).hexdigest() != outputs.get(digest_key): return False, f"hash:{name}"
    return True, "reused"


# --- the generation pipeline ----------------------------------------------

@dataclass
class SyntheticBankGenerator:
    """Deterministic, atomic, resume-safe generation of the frozen candidate plan.

    Every planned candidate ends in exactly one terminal state; nothing is
    resampled to reach an acceptance minimum and no rejected row is dropped.
    """
    package_root: Path
    bank_root: Path
    work_root: Path
    plan_path: Path
    calibration_path: Path
    bank_config: dict[str, Any]
    quality_config: dict[str, Any]
    backends: Any = None
    gpat_checkpoint_path: Path | None = None
    gpat_config: dict[str, Any] | None = None
    device: str = "cpu"
    expected_gpat_sha256: str | None = None
    expected_pair_plan_identity: str | None = None
    # An earlier run's work root whose payloads may be reused when the GENERATION
    # identity matches. Re-calibrating the gate does not change a payload, so a
    # re-evaluation reuses bytes instead of regenerating them.
    reuse_root: Path | None = None
    # A bank built under a different calibration policy must not carry the v1
    # name. The prefix is part of the declared layout, not cosmetic.
    bank_id_prefix: str | None = None
    # Extra calibration artifacts shipped inside the bank, as {file name: path}.
    calibration_files: dict[str, Path] = field(default_factory=dict)
    # The ONE declared A02 scientific control, or None. `None` keeps the full GPAT
    # identity guard, `recipe_bank_identity` included; see `conditioning_control`.
    conditioning_control: Any = None

    def __post_init__(self) -> None:
        from prism_fas.recipes.bank import load_bank
        self.package_root = Path(self.package_root); self.bank_root = Path(self.bank_root)
        self.work_root = Path(self.work_root)
        self.audit = SourceOnlyAudit()
        self.store = SampleStore.open(self.package_root, self.audit)
        self.bank = load_bank(self.bank_root)
        self.package_identity = json.loads((self.package_root / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))["content_identity_sha256"]
        self.recipe_bank_identity = str(self.bank["lock"]["bank_content_identity_sha256"])
        self.plan_rows = load_candidate_plan(Path(self.plan_path))
        lock_path = Path(self.plan_path).parent / "CANDIDATE_PLAN_LOCK.json"
        self.plan_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.candidate_plan_identity = str(self.plan_lock["candidate_plan_identity_sha256"])
        self.calibration = FrozenCalibration.load(Path(self.calibration_path))
        self.generation_config_sha256 = generation_config_sha256(self.bank_config, self.quality_config)
        self.physics = PhysicsRoute()
        self.gpat: GPATRoute | None = None
        # Why each candidate's payload was or was not taken from `reuse_root`.
        self.reuse_reasons: dict[str, int] = {}
        # Derived from the config alone, so a completed rerun that reuses every
        # record still writes the same architecture hash into BANK_LOCK.
        self.gpat_architecture_hash = self._architecture_hash()
        self.evaluator = CandidateEvaluator(self.backends, self.calibration) if self.backends is not None else None
        self._verify_plan()

    def _architecture_hash(self) -> str:
        if self.gpat_config is None: return ""
        from .gpat_model import build_gpat_model
        return build_gpat_model(self.gpat_config).architecture_hash()

    def _verify_plan(self) -> None:
        if self.plan_lock["package_identity"] != self.package_identity:
            raise SyntheticBankError("candidate plan was built against a different source package")
        if self.plan_lock["recipe_bank_identity"] != self.recipe_bank_identity:
            raise SyntheticBankError("candidate plan was built against a different recipe bank")
        from .candidate_plan import rows_digest
        if rows_digest(self.plan_rows) != self.plan_lock["candidate_rows_sha256"]:
            raise SyntheticBankError("candidate plan rows do not match their lock digest")
        if self.expected_gpat_sha256 and self.plan_lock["gpat_checkpoint_sha256"] != self.expected_gpat_sha256:
            raise SyntheticBankError("candidate plan binds a different GPAT checkpoint")

    # --- identity ---------------------------------------------------------
    def identity(self) -> dict[str, str]:
        body = {"package_identity": self.package_identity,
                "recipe_bank_identity": self.recipe_bank_identity,
                "candidate_plan_identity": self.candidate_plan_identity,
                "threshold_sha256": self.calibration.threshold_sha256,
                "fingerprint_reference_sha256": self.calibration.fingerprint_reference_sha256,
                "calibration_sha256": self.calibration.calibration_sha256,
                "generation_config_sha256": self.generation_config_sha256,
                "gpat_checkpoint_sha256": str(self.plan_lock["gpat_checkpoint_sha256"]),
                "physics_engine_version": PHYSICS_ENGINE_VERSION,
                "generator_version": GENERATOR_VERSION,
                "discrete_convention": DISCRETE_CONVENTION}
        # Present ONLY under the declared A02 control, and absent otherwise, so the
        # M8 v3 bank's identity is unchanged while any artifact built under the
        # control carries the policy — including the bank the generator was TRAINED
        # on and the out-of-distribution caveat — inside its own identity.
        if self.conditioning_control is not None:
            body["conditioning_control_identity"] = self.conditioning_control.identity()
            body["gpat_trained_on_recipe_bank_identity"] = \
                self.conditioning_control.trained_on_recipe_bank_identity
        return body

    def _route(self, name: str) -> Any:
        if name == "physics": return self.physics
        if self.gpat is None:
            if self.gpat_checkpoint_path is None or self.gpat_config is None:
                raise SyntheticBankError("the gpat route needs a checkpoint path and the gpat config")
            expected = {"package_identity": self.package_identity,
                        "recipe_bank_identity": self.recipe_bank_identity}
            if self.expected_pair_plan_identity: expected["pair_plan_identity"] = self.expected_pair_plan_identity
            self.gpat = GPATRoute(Path(self.gpat_checkpoint_path), self.gpat_config,
                                  expected_sha256=str(self.plan_lock["gpat_checkpoint_sha256"]),
                                  expected_identity=expected, device=self.device,
                                  conditioning_control=self.conditioning_control)
        return self.gpat

    # --- one candidate ----------------------------------------------------
    def build_candidate(self, row: dict[str, Any]) -> dict[str, Any]:
        """Generate, finalize, quality-gate and atomically persist one candidate."""
        synthetic_id = row["synthetic_id"]
        identity = self.identity()
        recipe = _recipe(self.bank, row["recipe_id"])
        graph = compile_recipe(recipe, self.bank["ontology"], bank_id=self.bank["bank_id"])
        requested_strength = float(np.mean([spec.strength for spec in recipe.artifacts]))
        stage = "generate"
        try:
            original, _ = self.store.load(row["live_target_sample_id"])
            masks = _support_masks(self.store, row["live_target_sample_id"], graph)
            support = np.asarray(masks.operator_support_mask)[0].astype(bool)
            geometry = {"requested_region_pixels": int(np.asarray(masks.requested_region_mask)[0].astype(bool).sum()),
                        "requested_coverage": float(masks.requested_coverage),
                        "achieved_coverage": float(masks.achieved_coverage)}
            reused = self._reuse_payload(row, original)
            if reused is not None:
                discrete, payload_source, trace = reused, "reused_prior_payload", {"payload_reused": True}
            else:
                route = self._route(row["route"])
                produced = route.generate(self.store, self.bank, row)
                if produced.binding != row["generator_binding"]:
                    raise SyntheticBankError(f"{synthetic_id}: route binding {produced.binding} != plan {row['generator_binding']}")
                stage = "finalize"
                discrete = finalize_discrete(produced.image, original, produced.requested_support,
                                             produced.artifact_map)
                payload_source, trace = "generated", produced.trace
                support = produced.requested_support
                geometry = {"requested_region_pixels": produced.requested_region_pixels,
                            "requested_coverage": produced.requested_coverage,
                            "achieved_coverage": produced.achieved_coverage}
            stage = "quality_gate"
            if self.evaluator is None: raise SyntheticBankError("quality backends are required to gate a candidate")
            quality = self.evaluator.evaluate(discrete, live_target_sample_id=row["live_target_sample_id"],
                                              requested_strength=requested_strength,
                                              requested_support=support)
        except Exception as error:                       # noqa: BLE001 - every stage failure is a terminal state
            stage_name, reason = stage, _sanitize(str(error))
            record = {"schema_version": CANDIDATE_RECORD_SCHEMA_VERSION, "synthetic_id": synthetic_id,
                      "route": row["route"], "terminal_state": "failed_generation", "identity": identity,
                      "plan": _plan_fields(row),
                      "failure": {"stage": stage_name, "exception_type": type(error).__name__, "reason": reason}}
            self._write_record(record)
            return record

        paths = _relative_paths(synthetic_id)
        record = {
            "schema_version": CANDIDATE_RECORD_SCHEMA_VERSION, "synthetic_id": synthetic_id,
            "route": row["route"], "terminal_state": "accepted" if quality["accepted"] else "rejected",
            "identity": identity, "plan": _plan_fields(row),
            "recipe": {"recipe_id": recipe.recipe_id, "recipe_hash": graph.recipe_hash, "graph_hash": graph.graph_hash,
                       "artifact_types": sorted({spec.name for spec in recipe.artifacts}),
                       "regions": sorted(graph.requested_regions),
                       "requested_artifact_strength": round(requested_strength, 8)},
            "geometry": {"exact_mask_pixels": discrete.exact_mask_pixels,
                         "requested_support_pixels": int(np.asarray(support).astype(bool).sum()),
                         "requested_region_pixels": geometry["requested_region_pixels"],
                         "requested_coverage": round(geometry["requested_coverage"], 8),
                         "achieved_coverage": round(geometry["achieved_coverage"], 8)},
            "payload_source": payload_source,
            "quality": quality, "generation_trace": trace}
        if quality["accepted"]:
            record["outputs"] = {**paths, "image_sha256": discrete.image_sha256,
                                 "mask_sha256": discrete.mask_sha256,
                                 "artifact_map_sha256": discrete.artifact_map_sha256}
            atomic_bytes_write(self.work_root / paths["image_relative_path"], discrete.image_png)
            atomic_bytes_write(self.work_root / paths["mask_relative_path"], discrete.mask_png)
            atomic_bytes_write(self.work_root / paths["artifact_map_relative_path"], discrete.artifact_map_npz)
            atomic_json_write(self.work_root / "metadata" / f"{synthetic_id}.json", _sample_metadata(record))
        self._write_record(record)
        return record

    def _reuse_payload(self, row: dict[str, Any], original: np.ndarray) -> DiscreteResult | None:
        """Load this exact candidate's payload from an earlier run, if that run
        retained it and every generation input and hash still validates.

        Returns the bytes rebuilt into a `DiscreteResult` with every discrete
        invariant re-proven — nothing about the earlier run is trusted.
        """
        if self.reuse_root is None: return None
        path = Path(self.reuse_root) / "records" / f"{row['synthetic_id']}.json"
        if not path.is_file():
            self.reuse_reasons["absent"] = self.reuse_reasons.get("absent", 0) + 1
            return None
        try: record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.reuse_reasons["unreadable_record"] = self.reuse_reasons.get("unreadable_record", 0) + 1
            return None
        usable, reason = payload_is_reusable(record, generation_identity(self.identity()), self.reuse_root)
        if not usable:
            self.reuse_reasons[reason] = self.reuse_reasons.get(reason, 0) + 1
            return None
        outputs = record["outputs"]
        root = Path(self.reuse_root)
        try:
            discrete = discrete_from_saved((root / outputs["image_relative_path"]).read_bytes(),
                                           (root / outputs["mask_relative_path"]).read_bytes(),
                                           (root / outputs["artifact_map_relative_path"]).read_bytes(),
                                           original)
        except SyntheticBankError as error:
            self.reuse_reasons[f"invalid_payload:{_sanitize(str(error))[:40]}"] = \
                self.reuse_reasons.get(f"invalid_payload:{_sanitize(str(error))[:40]}", 0) + 1
            return None
        self.reuse_reasons["reused"] = self.reuse_reasons.get("reused", 0) + 1
        return discrete

    def _write_record(self, record: dict[str, Any]) -> None:
        """The record is written last, so an interrupted candidate leaves no
        terminal marker and is rebuilt on resume."""
        atomic_json_write(self.work_root / "records" / f"{record['synthetic_id']}.json", record)

    def load_record(self, synthetic_id: str) -> dict[str, Any] | None:
        path = self.work_root / "records" / f"{synthetic_id}.json"
        if not path.is_file(): return None
        try: return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError: return None

    # --- the whole plan ---------------------------------------------------
    def run(self, *, rows: list[dict[str, Any]] | None = None, resume: bool = True,
            progress: Callable[[dict[str, Any]], None] | None = None,
            stop_after: int | None = None) -> dict[str, Any]:
        planned = list(rows if rows is not None else self.plan_rows)
        identity = self.identity()
        examined = reused = rebuilt = 0
        records: list[dict[str, Any]] = []
        reuse_reasons: dict[str, int] = {}
        for position, row in enumerate(planned, 1):
            examined += 1
            existing = self.load_record(row["synthetic_id"]) if resume else None
            if existing is not None:
                usable, reason = record_is_reusable(existing, identity, self.work_root)
                if usable:
                    records.append(existing); reused += 1
                    reuse_reasons["reused"] = reuse_reasons.get("reused", 0) + 1
                    if progress and position % 50 == 0: progress({"stage": "generate", "done": position,
                                                                  "total": len(planned), "reused": reused})
                    continue
                reuse_reasons[reason] = reuse_reasons.get(reason, 0) + 1
            records.append(self.build_candidate(row))
            rebuilt += 1
            if progress and (position == 1 or position % 25 == 0 or position == len(planned)):
                progress({"stage": "generate", "done": position, "total": len(planned),
                          "reused": reused, "rebuilt": rebuilt})
            if stop_after is not None and rebuilt >= stop_after:
                return {"status": "interrupted", "planned": len(planned), "examined": examined,
                        "reused": reused, "rebuilt": rebuilt, "records": records,
                        "reuse_reasons": dict(sorted(reuse_reasons.items()))}
        return {"status": "completed", "planned": len(planned), "examined": examined,
                "reused": reused, "rebuilt": rebuilt, "records": records,
                "reuse_reasons": dict(sorted(reuse_reasons.items())),
                "payload_sources": _counts([{"payload_source": record.get("payload_source", "unknown"),
                                             "synthetic_id": record["synthetic_id"]} for record in records],
                                           "payload_source"),
                "payload_reuse_reasons": dict(sorted(self.reuse_reasons.items())),
                "source_isolation": self.audit.report()}


PLAN_FIELDS = ("synthetic_id", "route", "slot", "live_target_sample_id", "live_target_dataset",
               "spoof_source_sample_id", "spoof_source_dataset", "domain_relation", "recipe_id",
               "candidate_seed", "generator_binding", "gpat_checkpoint_sha256", "physics_engine_version")


def _plan_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in PLAN_FIELDS}


def _sanitize(reason: str) -> str:
    """A failure reason never carries an absolute path or a raw dataset path."""
    import re
    text = re.sub(r"[A-Za-z]:[\\/][^\s'\"]*", "<path>", str(reason))
    text = re.sub(r"(/[^\s'\"/]+)+/", "<path>/", text)
    return text[:400]


def _sample_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """The per-sample JSON shipped inside a shard. Carries no path, subject or
    target field."""
    return {"schema_version": CANDIDATE_RECORD_SCHEMA_VERSION,
            "synthetic_id": record["synthetic_id"], "route": record["route"],
            "live_target_sample_id": record["plan"]["live_target_sample_id"],
            "live_target_dataset": record["plan"]["live_target_dataset"],
            "spoof_source_sample_id": record["plan"]["spoof_source_sample_id"],
            "spoof_source_dataset": record["plan"]["spoof_source_dataset"],
            "domain_relation": record["plan"]["domain_relation"],
            "recipe_id": record["recipe"]["recipe_id"], "recipe_hash": record["recipe"]["recipe_hash"],
            "graph_hash": record["recipe"]["graph_hash"],
            "artifact_types": record["recipe"]["artifact_types"], "regions": record["recipe"]["regions"],
            "gpat_checkpoint_sha256": record["plan"]["gpat_checkpoint_sha256"],
            "physics_engine_version": record["plan"]["physics_engine_version"],
            "q": record["quality"]["q"], "quality_components": record["quality"]["quality_components"],
            "metrics": record["quality"]["metrics"], "recipe_match": record["quality"]["recipe_match"],
            "threshold_hash": record["quality"]["threshold_hash"],
            "geometry": record["geometry"], "outputs": record.get("outputs", {}),
            "identity": record["identity"]}


# --- bank assembly ---------------------------------------------------------

BANK_LOCK_SCHEMA_VERSION = "m8-bank-lock-v1"
BANK_ID_PREFIX = "prism_synthetic_bank_m8_v1"
IDENTITY_SHORT_LENGTH = 12
# Provenance that MUST NOT reach the bank content identity: it changes between
# machines and runs without changing a single output byte.
LOCK_IDENTITY_EXCLUDED = ("bank_id", "bank_content_identity_sha256", "identity_excluded_fields",
                          "created_at", "provenance", "parquet_byte_hashes", "export")

_METRIC_FIELDS = ("face_detection_score", "identity_cosine", "landmark_nme", "outside_mask_parsing_dice",
                  "outside_mask_max_error", "measured_artifact_strength", "requested_artifact_strength",
                  "fingerprint_score", "support_overlap")
_COMPONENT_FIELDS = ("q_fd", "q_id", "q_lm", "q_parse", "q_strength", "q_fp", "q_support")


def _accepted_row(record: dict[str, Any]) -> dict[str, Any]:
    plan, quality, geometry, recipe = record["plan"], record["quality"], record["geometry"], record["recipe"]
    metrics, outputs, identity = quality["metrics"], record["outputs"], record["identity"]
    row = {"synthetic_id": record["synthetic_id"], "route": record["route"],
           "live_target_sample_id": plan["live_target_sample_id"],
           "spoof_source_sample_id": plan["spoof_source_sample_id"],
           "live_target_dataset": plan["live_target_dataset"],
           "spoof_source_dataset": plan["spoof_source_dataset"],
           "domain_relation": plan["domain_relation"],
           "recipe_id": recipe["recipe_id"], "recipe_hash": recipe["recipe_hash"],
           "graph_hash": recipe["graph_hash"],
           "artifact_types": "|".join(recipe["artifact_types"]), "regions": "|".join(recipe["regions"]),
           "gpat_checkpoint_sha256": plan["gpat_checkpoint_sha256"],
           "physics_engine_version": plan["physics_engine_version"],
           "image_relative_path": outputs["image_relative_path"], "image_sha256": outputs["image_sha256"],
           "mask_relative_path": outputs["mask_relative_path"], "mask_sha256": outputs["mask_sha256"],
           "artifact_map_relative_path": outputs["artifact_map_relative_path"],
           "artifact_map_sha256": outputs["artifact_map_sha256"],
           "q": float(quality["q"]), "recipe_match": quality["recipe_match"],
           "threshold_hash": quality["threshold_hash"],
           "exact_mask_pixels": int(geometry["exact_mask_pixels"]),
           "requested_support_pixels": int(geometry["requested_support_pixels"]),
           "requested_region_pixels": int(geometry["requested_region_pixels"]),
           "requested_coverage": float(geometry["requested_coverage"]),
           "achieved_coverage": float(geometry["achieved_coverage"]),
           "package_identity": identity["package_identity"],
           "recipe_bank_identity": identity["recipe_bank_identity"],
           "candidate_plan_identity": identity["candidate_plan_identity"],
           "calibration_hash": identity["calibration_sha256"],
           "fingerprint_reference_sha256": identity["fingerprint_reference_sha256"],
           "generation_config_hash": identity["generation_config_sha256"]}
    row.update({name: float(metrics[name]) for name in _METRIC_FIELDS})
    row.update({name: float(quality["quality_components"][name]) for name in _COMPONENT_FIELDS})
    return row


def _rejected_row(record: dict[str, Any]) -> dict[str, Any]:
    plan, quality = record["plan"], record["quality"]
    row = {"synthetic_id": record["synthetic_id"], "route": record["route"],
           "live_target_sample_id": plan["live_target_sample_id"],
           "spoof_source_sample_id": plan["spoof_source_sample_id"],
           "live_target_dataset": plan["live_target_dataset"],
           "spoof_source_dataset": plan["spoof_source_dataset"],
           "recipe_id": record["recipe"]["recipe_id"],
           "failed_gates": "|".join(quality["failed_gates"]),
           "failed_gate_count": len(quality["failed_gates"]),
           "q": float(quality["q"]), "threshold_hash": quality["threshold_hash"]}
    row.update({name: float(quality["metrics"][name]) for name in _METRIC_FIELDS})
    return row


def _failure_row(record: dict[str, Any]) -> dict[str, Any]:
    failure = record["failure"]
    return {"synthetic_id": record["synthetic_id"], "route": record["route"],
            "failed_stage": failure["stage"], "exception_type": failure["exception_type"],
            "reason": failure["reason"]}


_ACCEPTED_SCHEMA = ([("synthetic_id", "string"), ("route", "string"), ("live_target_sample_id", "string"),
                     ("spoof_source_sample_id", "string"), ("live_target_dataset", "string"),
                     ("spoof_source_dataset", "string"), ("domain_relation", "string"),
                     ("recipe_id", "string"), ("recipe_hash", "string"), ("graph_hash", "string"),
                     ("artifact_types", "string"), ("regions", "string"),
                     ("gpat_checkpoint_sha256", "string"), ("physics_engine_version", "string"),
                     ("image_relative_path", "string"), ("image_sha256", "string"),
                     ("mask_relative_path", "string"), ("mask_sha256", "string"),
                     ("artifact_map_relative_path", "string"), ("artifact_map_sha256", "string"),
                     ("q", "double"), ("recipe_match", "string"), ("threshold_hash", "string"),
                     ("exact_mask_pixels", "int64"), ("requested_support_pixels", "int64"),
                     ("requested_region_pixels", "int64"), ("requested_coverage", "double"),
                     ("achieved_coverage", "double"), ("package_identity", "string"),
                     ("recipe_bank_identity", "string"), ("candidate_plan_identity", "string"),
                     ("calibration_hash", "string"), ("fingerprint_reference_sha256", "string"),
                     ("generation_config_hash", "string")]
                    + [(name, "double") for name in _METRIC_FIELDS]
                    + [(name, "double") for name in _COMPONENT_FIELDS])
_REJECTED_SCHEMA = ([("synthetic_id", "string"), ("route", "string"), ("live_target_sample_id", "string"),
                     ("spoof_source_sample_id", "string"), ("live_target_dataset", "string"),
                     ("spoof_source_dataset", "string"), ("recipe_id", "string"),
                     ("failed_gates", "string"), ("failed_gate_count", "int64"),
                     ("q", "double"), ("threshold_hash", "string")]
                    + [(name, "double") for name in _METRIC_FIELDS])
_FAILURE_SCHEMA = [("synthetic_id", "string"), ("route", "string"), ("failed_stage", "string"),
                   ("exception_type", "string"), ("reason", "string")]
MANIFEST_SCHEMAS = {"manifest": _ACCEPTED_SCHEMA, "rejected": _REJECTED_SCHEMA, "failures": _FAILURE_SCHEMA}


def _arrow_schema(fields: list[tuple[str, str]]) -> Any:
    import pyarrow as pa
    mapping = {"string": pa.string(), "double": pa.float64(), "int64": pa.int64()}
    return pa.schema([(name, mapping[kind]) for name, kind in fields])


def write_manifest(path: Path, rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> str:
    """Write one manifest parquet and return the LOGICAL row digest.

    Parquet bytes are not portable across pyarrow versions, so identity binds the
    logical rows; the byte hash is recorded separately as provenance.
    """
    import pyarrow as pa, pyarrow.parquet as pq
    ordered = sorted(rows, key=lambda row: row["synthetic_id"])
    table = pa.Table.from_pydict({name: [row.get(name) for row in ordered] for name, _ in fields},
                                 schema=_arrow_schema(fields))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, Path(path), compression="none", version="2.6", write_statistics=False)
    return rows_logical_digest(ordered, fields)


def rows_logical_digest(rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> str:
    canonical = json.dumps([{name: _canonical(row.get(name), kind) for name, kind in fields}
                            for row in sorted(rows, key=lambda item: item["synthetic_id"])],
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical(value: Any, kind: str) -> Any:
    if value is None: return None
    if kind == "double": return format(float(value), ".12g")
    if kind == "int64": return int(value)
    return str(value)


def load_manifest(path: Path, fields: list[tuple[str, str]]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    table = pq.read_table(Path(path)).to_pydict()
    count = len(table["synthetic_id"]) if "synthetic_id" in table else 0
    return [{name: table[name][index] for name, _ in fields} for index in range(count)]


def coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts, regions = set(), set()
    for row in rows:
        artifacts.update(value for value in str(row["artifact_types"]).split("|") if value)
        regions.update(value for value in str(row["regions"]).split("|") if value)
    return {"artifact_types": sorted(artifacts), "artifact_type_count": len(artifacts),
            "regions": sorted(regions), "region_count": len(regions)}


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value is None: continue
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def check_operational_minimums(accepted: list[dict[str, Any]], candidates: int,
                               minimums: dict[str, Any]) -> dict[str, Any]:
    """The pre-declared minimums. Never adjusted after seeing the counts."""
    routes = _counts(accepted, "route")
    domains = _counts(accepted, "live_target_dataset")
    coverage = coverage_summary(accepted)
    relations = {str(row["domain_relation"]) for row in accepted if row["route"] == "gpat"}
    checks = {
        "candidates": candidates == int(minimums["candidates"]),
        "accepted_total": len(accepted) >= int(minimums["accepted_total"]),
        "accepted_physics": routes.get("physics", 0) >= int(minimums["accepted_physics"]),
        "accepted_gpat": routes.get("gpat", 0) >= int(minimums["accepted_gpat"]),
        "accepted_live_casia": domains.get("casia_fasd", 0) >= int(minimums["accepted_live_casia"]),
        "accepted_live_msu": domains.get("msu_mfsd", 0) >= int(minimums["accepted_live_msu"]),
        "all_artifact_types": coverage["artifact_type_count"] >= int(minimums["require_all_artifact_types"]),
        "all_regions": coverage["region_count"] >= int(minimums["require_all_regions"]),
        "same_and_cross_domain_gpat": (not bool(minimums["require_same_and_cross_domain_gpat"])
                                       or {"same_domain", "cross_domain"} <= relations)}
    return {"passed": all(checks.values()), "checks": checks,
            "observed": {"candidates": candidates, "accepted_total": len(accepted),
                         "accepted_route_counts": routes, "accepted_live_target_datasets": domains,
                         "accepted_gpat_domain_relations": sorted(relations), **coverage},
            "declared_minimums": {key: minimums[key] for key in sorted(minimums)}}


def quality_summary(accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(row["q"]) for row in accepted], dtype=np.float64)
    reasons: dict[str, int] = {}
    for row in rejected:
        for name in str(row["failed_gates"]).split("|"):
            if name: reasons[name] = reasons.get(name, 0) + 1
    def distribution(rows: list[dict[str, Any]], name: str) -> dict[str, float]:
        series = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not series.size: return {}
        return {"min": float(series.min()), "p05": float(np.percentile(series, 5)),
                "median": float(np.median(series)), "mean": float(series.mean()),
                "p95": float(np.percentile(series, 95)), "max": float(series.max())}
    return {"quality_gate_schema_version": SYNTHETIC_BANK_SCHEMA_VERSION,
            "accepted": len(accepted), "rejected": len(rejected),
            "q_statistics": ({"min": float(values.min()), "p05": float(np.percentile(values, 5)),
                              "median": float(np.median(values)), "mean": float(values.mean()),
                              "p95": float(np.percentile(values, 95)), "max": float(values.max())}
                             if values.size else {}),
            "accepted_metric_distributions": {name: distribution(accepted, name) for name in _METRIC_FIELDS},
            "rejected_metric_distributions": {name: distribution(rejected, name) for name in _METRIC_FIELDS},
            "dominant_rejection_reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
            "recipe_match": "not_applicable",
            "accepted_route_counts": _counts(accepted, "route"),
            "accepted_live_target_datasets": _counts(accepted, "live_target_dataset"),
            "accepted_coverage": coverage_summary(accepted)}


def assemble_bank(generator: SyntheticBankGenerator, records: list[dict[str, Any]], *, pairs_root: Path,
                  build_root: Path | None = None, max_samples_per_shard: int | None = None) -> dict[str, Any]:
    """Build the versioned bank directory from the terminal candidate records.

    Nothing here can invent a candidate: the accepted, rejected and failure rows
    are exactly the terminal records of the frozen plan.
    """
    import shutil
    from .synthetic_shards import build_shards, index_digest, write_shards_index, MAX_SAMPLES_PER_SHARD
    work = Path(generator.work_root)
    states = {state: [record for record in records if record["terminal_state"] == state] for state in TERMINAL_STATES}
    total = sum(len(rows) for rows in states.values())
    if total != len(records) or len(records) != len(generator.plan_rows):
        raise SyntheticBankError(f"terminal accounting is {total}/{len(records)} against a plan of {len(generator.plan_rows)}")
    ids = [record["synthetic_id"] for record in records]
    if len(set(ids)) != len(ids): raise SyntheticBankError("duplicate synthetic ids in the terminal records")

    accepted = [_accepted_row(record) for record in states["accepted"]]
    rejected = [_rejected_row(record) for record in states["rejected"]]
    failures = [_failure_row(record) for record in states["failed_generation"]]
    minimums = check_operational_minimums(accepted, len(records), generator.bank_config["operational_minimums"])
    summary_quality = quality_summary(accepted, rejected)

    build = Path(build_root or (work / "_build"))
    if build.exists(): shutil.rmtree(build)
    for name in ("images", "masks", "artifact_maps", "metadata", "manifests", "calibration", "shards"):
        (build / name).mkdir(parents=True, exist_ok=True)
    for record in states["accepted"]:
        synthetic_id = record["synthetic_id"]
        for relative in _relative_paths(synthetic_id).values():
            shutil.copyfile(work / relative, build / relative)
        shutil.copyfile(work / "metadata" / f"{synthetic_id}.json", build / "metadata" / f"{synthetic_id}.json")

    digests = {
        "manifest": write_manifest(build / "manifests" / "manifest.parquet", accepted, _ACCEPTED_SCHEMA),
        "rejected": write_manifest(build / "manifests" / "rejected.parquet", rejected, _REJECTED_SCHEMA),
        "failures": write_manifest(build / "manifests" / "failures.parquet", failures, _FAILURE_SCHEMA)}
    candidate_manifest = _write_candidate_manifest(build / "manifests" / "candidate_manifest.parquet",
                                                   generator.plan_rows, records)
    for name in ("pair_manifest_train.parquet", "pair_manifest_validation.parquet"):
        source = Path(pairs_root) / name
        if not source.is_file(): raise SyntheticBankError(f"pair manifest {name} is missing")
        shutil.copyfile(source, build / "manifests" / name)
    primary = Path(generator.calibration_path)
    shutil.copyfile(primary, build / "calibration" / primary.name)
    for name, path in sorted(getattr(generator, "calibration_files", {}).items()):
        shutil.copyfile(Path(path), build / "calibration" / name)

    shard_limit = int(max_samples_per_shard or generator.bank_config.get("shards", {}).get("max_samples_per_shard",
                                                                                           MAX_SAMPLES_PER_SHARD))
    shard_index = build_shards(build, accepted, max_samples=shard_limit)
    write_shards_index(build, shard_index)

    generation = {"generation_summary_schema_version": SYNTHETIC_BANK_SCHEMA_VERSION,
                  "candidates": len(records), "terminal_counts": {state: len(states[state]) for state in TERMINAL_STATES},
                  "route_counts": _counts([{"route": record["route"], "synthetic_id": record["synthetic_id"]}
                                           for record in records], "route"),
                  "accepted_route_counts": _counts(accepted, "route"),
                  "rejected_route_counts": _counts(rejected, "route"),
                  "failure_stages": _counts(failures, "failed_stage"),
                  "payload_sources": _counts([{"payload_source": record.get("payload_source", "unknown"),
                                               "synthetic_id": record["synthetic_id"]} for record in records],
                                             "payload_source"),
                  "operational_minimums": minimums,
                  "generator_version": GENERATOR_VERSION, "discrete_convention": DISCRETE_CONVENTION,
                  "identity": generator.identity(),
                  "source_isolation": generator.audit.report()}
    atomic_json_write(build / "quality_summary.json", summary_quality)
    atomic_json_write(build / "generation_summary.json", generation)

    lock = _bank_lock(generator, records=records, accepted=accepted, rejected=rejected, failures=failures,
                      digests=digests, candidate_manifest=candidate_manifest, shard_index=shard_index,
                      shard_index_digest=index_digest(shard_index), quality=summary_quality,
                      generation=generation, minimums=minimums, build=build)
    atomic_json_write(build / "BANK_LOCK.json", lock)

    bank_id = lock["bank_id"]
    destination = work / bank_id
    if destination.exists():
        existing = json.loads((destination / "BANK_LOCK.json").read_text(encoding="utf-8"))
        if existing.get("bank_content_identity_sha256") != lock["bank_content_identity_sha256"]:
            raise SyntheticBankError(f"{bank_id} already exists with a different content identity")
        shutil.rmtree(build)
        return {"status": "reused", "bank_id": bank_id, "bank_root": str(destination), "lock": existing,
                "operational_minimums": minimums, "quality_summary": summary_quality,
                "generation_summary": generation}
    build.rename(destination)
    return {"status": "created", "bank_id": bank_id, "bank_root": str(destination), "lock": lock,
            "operational_minimums": minimums, "quality_summary": summary_quality,
            "generation_summary": generation}


def _write_candidate_manifest(path: Path, plan_rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, str]:
    """The frozen plan plus each candidate's terminal state.

    The extra column does not disturb `candidate_rows_sha256`, which is computed
    over the plan fields alone.
    """
    import pyarrow as pa, pyarrow.parquet as pq
    from .candidate_plan import _FIELDS, rows_digest
    states = {record["synthetic_id"]: record["terminal_state"] for record in records}
    ordered = sorted(plan_rows, key=lambda row: row["synthetic_id"])
    schema = pa.schema(list(_FIELDS) + [("terminal_state", pa.string())])
    columns = {name: [row[name] for row in ordered] for name, _ in _FIELDS}
    columns["terminal_state"] = [states.get(row["synthetic_id"], "missing") for row in ordered]
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), Path(path),
                   compression="none", version="2.6", write_statistics=False)
    return {"candidate_rows_sha256": rows_digest(ordered),
            "terminal_state_sha256": hashlib.sha256("|".join(
                f"{row['synthetic_id']}={states.get(row['synthetic_id'], 'missing')}" for row in ordered
            ).encode("utf-8")).hexdigest()}


def _bank_lock(generator: SyntheticBankGenerator, *, records: list[dict[str, Any]], accepted: list[dict[str, Any]],
               rejected: list[dict[str, Any]], failures: list[dict[str, Any]], digests: dict[str, str],
               candidate_manifest: dict[str, str], shard_index: list[dict[str, Any]], shard_index_digest: str,
               quality: dict[str, Any], generation: dict[str, Any], minimums: dict[str, Any],
               build: Path) -> dict[str, Any]:
    from prism_fas.utils.core import utc_timestamp
    identity = generator.identity()
    pair_plan_identity = generator.expected_pair_plan_identity or ""
    lock: dict[str, Any] = {
        "bank_lock_schema_version": BANK_LOCK_SCHEMA_VERSION,
        "synthetic_bank_schema_version": SYNTHETIC_BANK_SCHEMA_VERSION,
        # Never unconditional: a hard-coded "validated" would make both this field
        # and the validator's status check vacuous. A bank that misses a
        # pre-declared operational minimum is retained and labelled, not relabelled.
        "status": "validated" if minimums["passed"] else "operational_minimums_failed",
        "seed": int(generator.bank_config["seed"]),
        "package_identity": identity["package_identity"],
        "recipe_bank_identity": identity["recipe_bank_identity"],
        "gpat_pair_plan_identity": pair_plan_identity,
        "candidate_plan_identity": identity["candidate_plan_identity"],
        "gpat_architecture_hash": generator.gpat_architecture_hash,
        "gpat_checkpoint_sha256": identity["gpat_checkpoint_sha256"],
        "physics_engine_version": PHYSICS_ENGINE_VERSION,
        "quality_calibration_sha256": identity["calibration_sha256"],
        "fingerprint_reference_sha256": identity["fingerprint_reference_sha256"],
        "threshold_sha256": identity["threshold_sha256"],
        "generation_config_sha256": identity["generation_config_sha256"],
        "generator_version": GENERATOR_VERSION, "discrete_convention": DISCRETE_CONVENTION,
        "candidate_count": len(records),
        "accepted_count": len(accepted), "rejected_count": len(rejected), "failed_count": len(failures),
        "route_counts": generation["route_counts"],
        "accepted_route_counts": _counts(accepted, "route"),
        "accepted_live_target_datasets": _counts(accepted, "live_target_dataset"),
        "accepted_domain_relations": _counts([row for row in accepted if row["route"] == "gpat"], "domain_relation"),
        "accepted_coverage": coverage_summary(accepted),
        "manifest_digests": {**digests, **candidate_manifest,
                             "pair_manifest_train_sha256": hashlib.sha256(
                                 (build / "manifests" / "pair_manifest_train.parquet").read_bytes()).hexdigest(),
                             "pair_manifest_validation_sha256": hashlib.sha256(
                                 (build / "manifests" / "pair_manifest_validation.parquet").read_bytes()).hexdigest()},
        "quality_summary_sha256": _json_digest(quality),
        "generation_summary_sha256": _json_digest(generation),
        "shards_index_sha256": shard_index_digest,
        "shard_count": len(shard_index),
        "shards": [{"shard_name": row["shard_name"], "row_count": int(row["row_count"]),
                    "byte_size": int(row["byte_size"]), "sha256": row["sha256"]} for row in shard_index],
        "operational_minimums": minimums,
        "quality_models": generator.calibration.quality_models,
        "operational_minimums_passed": bool(minimums["passed"])}
    lock["bank_content_identity_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in lock.items() if key not in LOCK_IDENTITY_EXCLUDED},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    prefix = (getattr(generator, "bank_id_prefix", None)
              or generator.bank_config.get("bank_id_prefix") or BANK_ID_PREFIX)
    lock["bank_id"] = f"{prefix}_{lock['bank_content_identity_sha256'][:IDENTITY_SHORT_LENGTH]}"
    lock["identity_excluded_fields"] = list(LOCK_IDENTITY_EXCLUDED)
    # Informational only, explicitly outside the content identity.
    lock["created_at"] = utc_timestamp()
    return lock


def _json_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
