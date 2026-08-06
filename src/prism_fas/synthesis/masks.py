from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .contracts import MaskBuildError, RegionMaskResult, mask_hash
from .operators.base import binary_dilate, boundary_band

MASK_POLICY_VERSION = "m7-mask-v1"
# FaceXFormer / LaPa 11-class ordering as stored by M3B. Verified empirically
# from class centroids on real source_train live priors; see
# docs/M7_REGION_MASK_MAPPING.md. Left/right are image-space (viewer) sides.
PARSING_LABELS = {"background": 0, "skin": 1, "left_eyebrow": 2, "right_eyebrow": 3, "left_eye": 4,
                  "right_eye": 5, "nose": 6, "upper_lip": 7, "inner_mouth": 8, "lower_lip": 9, "hair": 10}
FACE_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
BROW_AND_EYE_CLASSES = (2, 3, 4, 5)
# Parsing classes that define a region directly. Regions absent from this map
# are always geometric (forehead, cheeks) or morphological (boundary, context).
REGION_PARSING_CLASSES = {"left_eye": (4,), "right_eye": (5,), "nose": (6,), "mouth": (7, 8, 9)}
# A parsing region smaller than this is treated as missing and the documented
# geometry fallback is used instead. Real priors do drop classes: inner_mouth is
# absent on most samples and one eye can be missing under yaw.
MIN_PARSING_PIXELS = 24
LANDMARK_INDEX = {"left_eye": 0, "right_eye": 1, "nose": 2, "mouth_left": 3, "mouth_right": 4}
REGION_ORDER = ("left_eye", "right_eye", "nose", "mouth", "forehead", "left_cheek", "right_cheek",
                "face_boundary", "context")


def _ellipse(height: int, width: int, centre_x: float, centre_y: float, radius_x: float, radius_y: float) -> np.ndarray:
    rows = np.arange(height, dtype=np.float32)[:, None] + np.float32(0.5)
    columns = np.arange(width, dtype=np.float32)[None, :] + np.float32(0.5)
    dx = (columns - np.float32(centre_x)) / np.float32(max(radius_x, 1e-3))
    dy = (rows - np.float32(centre_y)) / np.float32(max(radius_y, 1e-3))
    return (dx * dx + dy * dy) <= np.float32(1.0)


def _band(height: int, width: int, top: float, bottom: float) -> np.ndarray:
    rows = np.arange(height, dtype=np.float32)[:, None] + np.float32(0.5)
    return np.broadcast_to((rows >= np.float32(top)) & (rows < np.float32(bottom)), (height, width)).copy()


@dataclass(frozen=True)
class RegionMaskBuilder:
    """Deterministic parsing-first, geometry-fallback semantic region masks.

    `landmarks`/`bbox` are stored by M3B in original-frame coordinates while the
    parsing mask and image are the 224x224 crop, so a `crop_box` is required to
    map between them. When `crop_box` is None the points are assumed to already
    live in crop pixel space (unit-test fixtures do this).
    """
    height: int
    width: int
    parsing: np.ndarray | None
    landmarks: np.ndarray
    bbox: np.ndarray
    crop_box: np.ndarray | None = None

    # --- coordinate mapping -------------------------------------------------
    def _scale(self) -> tuple[float, float, float, float]:
        if self.crop_box is None: return 0.0, 0.0, 1.0, 1.0
        box = np.asarray(self.crop_box, dtype=np.float32).reshape(-1)
        if box.shape != (4,): raise MaskBuildError("crop_box must hold four values")
        span_x = float(box[2] - box[0]); span_y = float(box[3] - box[1])
        if span_x <= 0 or span_y <= 0: raise MaskBuildError("crop_box has a non-positive span")
        return float(box[0]), float(box[1]), self.width / span_x, self.height / span_y

    def _point(self, x: float, y: float) -> tuple[float, float]:
        ox, oy, sx, sy = self._scale()
        return (float(x) - ox) * sx, (float(y) - oy) * sy

    def landmark(self, name: str) -> tuple[float, float]:
        points = np.asarray(self.landmarks, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 5: raise MaskBuildError("landmarks must hold five points")
        row = points[LANDMARK_INDEX[name]]
        return self._point(float(row[0]), float(row[1]))

    def face_box(self) -> tuple[float, float, float, float]:
        box = np.asarray(self.bbox, dtype=np.float32).reshape(-1)
        if box.shape != (4,): raise MaskBuildError("bbox must hold four values")
        x1, y1 = self._point(float(box[0]), float(box[1]))
        x2, y2 = self._point(float(box[2]), float(box[3]))
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1 = float(np.clip(x1, 0.0, self.width)); x2 = float(np.clip(x2, 0.0, self.width))
        y1 = float(np.clip(y1, 0.0, self.height)); y2 = float(np.clip(y2, 0.0, self.height))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:  # degenerate detection: fall back to the whole crop
            return 0.0, 0.0, float(self.width), float(self.height)
        return x1, y1, x2, y2

    # --- base masks ---------------------------------------------------------
    def _parsing_mask(self, classes: tuple[int, ...]) -> np.ndarray | None:
        if self.parsing is None: return None
        parsing = np.asarray(self.parsing)
        if parsing.shape != (self.height, self.width): raise MaskBuildError(
            f"parsing shape {parsing.shape} != ({self.height}, {self.width})")
        mask = np.isin(parsing, classes)
        return mask if int(mask.sum()) >= MIN_PARSING_PIXELS else None

    def face_mask(self) -> tuple[np.ndarray, str]:
        parsed = self._parsing_mask(FACE_CLASSES)
        if parsed is not None: return parsed, "parsing"
        x1, y1, x2, y2 = self.face_box()
        return _ellipse(self.height, self.width, (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                        (x2 - x1) / 2.0, (y2 - y1) / 2.0), "bbox_geometry"

    def _eye_line(self) -> float:
        brows = self._parsing_mask(BROW_AND_EYE_CLASSES)
        if brows is not None:
            rows = np.nonzero(brows.any(axis=1))[0]
            if rows.size: return float(rows.min())
        left = self.landmark("left_eye"); right = self.landmark("right_eye")
        x1, y1, x2, y2 = self.face_box()
        return float(np.clip((left[1] + right[1]) / 2.0 - 0.12 * (y2 - y1), 0.0, self.height))

    def _mouth_line(self) -> float:
        mouth = self._parsing_mask(REGION_PARSING_CLASSES["mouth"])
        if mouth is not None:
            rows = np.nonzero(mouth.any(axis=1))[0]
            if rows.size: return float(rows.min())
        left = self.landmark("mouth_left"); right = self.landmark("mouth_right")
        return float(np.clip((left[1] + right[1]) / 2.0, 0.0, self.height))

    def region(self, name: str) -> tuple[np.ndarray, str]:
        """One canonical region mask plus the source it came from."""
        face, face_source = self.face_mask()
        x1, y1, x2, y2 = self.face_box()
        face_width, face_height = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
        if name in ("left_eye", "right_eye", "nose", "mouth"):
            parsed = self._parsing_mask(REGION_PARSING_CLASSES[name])
            if parsed is not None: return parsed, "parsing"
            if name == "mouth":
                left = self.landmark("mouth_left"); right = self.landmark("mouth_right")
                centre_x, centre_y = (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0
                radius_x = max(abs(right[0] - left[0]) * 0.75, face_width * 0.12)
                return _ellipse(self.height, self.width, centre_x, centre_y, radius_x, face_height * 0.07), "landmark_geometry"
            key = {"left_eye": "left_eye", "right_eye": "right_eye", "nose": "nose"}[name]
            centre_x, centre_y = self.landmark(key)
            if name == "nose":
                return _ellipse(self.height, self.width, centre_x, centre_y, face_width * 0.11, face_height * 0.14), "landmark_geometry"
            return _ellipse(self.height, self.width, centre_x, centre_y, face_width * 0.11, face_height * 0.07), "landmark_geometry"
        if name == "forehead":
            top = max(y1, 0.0)
            mask = face & _band(self.height, self.width, top, self._eye_line())
            if int(mask.sum()) >= MIN_PARSING_PIXELS: return mask, f"{face_source}+geometry"
            mask = _ellipse(self.height, self.width, (x1 + x2) / 2.0, y1 + face_height * 0.18,
                            face_width * 0.36, face_height * 0.14)
            return mask, "bbox_geometry"
        if name in ("left_cheek", "right_cheek"):
            nose_x, _ = self.landmark("nose")
            top = self._eye_line() + 0.10 * face_height
            bottom = min(self._mouth_line() + 0.05 * face_height, y2)
            columns = np.arange(self.width, dtype=np.float32)[None, :] + np.float32(0.5)
            side = (columns < np.float32(nose_x - 0.06 * face_width)) if name == "left_cheek" \
                else (columns > np.float32(nose_x + 0.06 * face_width))
            mask = face & _band(self.height, self.width, top, max(bottom, top + 2.0)) & np.broadcast_to(side, (self.height, self.width))
            if int(mask.sum()) >= MIN_PARSING_PIXELS: return mask, f"{face_source}+geometry"
            centre_x = nose_x - 0.22 * face_width if name == "left_cheek" else nose_x + 0.22 * face_width
            mask = _ellipse(self.height, self.width, centre_x, (top + max(bottom, top + 2.0)) / 2.0,
                            face_width * 0.15, face_height * 0.14)
            return mask, "bbox_geometry"
        if name == "face_boundary":
            radius = int(max(2, round(0.04 * min(face_width, face_height))))
            return boundary_band(face, radius), f"{face_source}+morphology"
        if name == "context":
            radius = int(max(3, round(0.16 * min(face_width, face_height))))
            mask = binary_dilate(face, radius) & ~face
            if int(mask.sum()) >= MIN_PARSING_PIXELS: return mask, f"{face_source}+morphology"
            outer = _ellipse(self.height, self.width, (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                             face_width * 0.75, face_height * 0.75)
            return outer & ~face, "bbox_geometry"
        raise MaskBuildError(f"unknown canonical region {name!r}")

    # --- coverage -----------------------------------------------------------
    @staticmethod
    def _coverage_field(height: int, width: int, seed: int) -> np.ndarray:
        """Smooth deterministic scalar field: a small seeded lattice upsampled
        with bilinear weights, so the coverage sub-mask forms blobs rather than
        salt-and-pepper noise."""
        rng = np.random.Generator(np.random.PCG64(int(seed) % (2 ** 32)))
        lattice = rng.random((8, 8)).astype(np.float32)
        rows = np.linspace(0.0, 7.0, height, dtype=np.float32)
        columns = np.linspace(0.0, 7.0, width, dtype=np.float32)
        r0 = np.clip(np.floor(rows), 0, 7).astype(np.int64); r1 = np.clip(r0 + 1, 0, 7)
        c0 = np.clip(np.floor(columns), 0, 7).astype(np.int64); c1 = np.clip(c0 + 1, 0, 7)
        wr = (rows - r0.astype(np.float32))[:, None]; wc = (columns - c0.astype(np.float32))[None, :]
        top = lattice[np.ix_(r0, c0)] * (1 - wc) + lattice[np.ix_(r0, c1)] * wc
        bottom = lattice[np.ix_(r1, c0)] * (1 - wc) + lattice[np.ix_(r1, c1)] * wc
        return (top * (1 - wr) + bottom * wr).astype(np.float32)

    def _apply_coverage(self, base: np.ndarray, coverage: float, seed: int) -> tuple[np.ndarray, float]:
        area = int(base.sum())
        if area == 0: raise MaskBuildError("requested region union is empty")
        coverage = float(np.clip(coverage, 0.0, 1.0))
        if coverage >= 1.0: return base.copy(), 1.0
        keep = int(max(1, round(coverage * area)))
        if keep >= area: return base.copy(), 1.0
        field = self._coverage_field(self.height, self.width, seed)
        flat_index = np.nonzero(base.reshape(-1))[0]
        values = field.reshape(-1)[flat_index]
        # stable sort with the flat index as tie-break keeps this deterministic
        order = np.lexsort((flat_index, -values))
        chosen = flat_index[order[:keep]]
        mask = np.zeros(self.height * self.width, dtype=bool)
        mask[chosen] = True
        return mask.reshape(self.height, self.width), float(keep) / float(area)

    # --- public API ---------------------------------------------------------
    def build(self, regions: list[str], *, geometry_shape: str, coverage: float, seed: int,
              coverage_tolerance: float = 0.05) -> RegionMaskResult:
        if not regions: raise MaskBuildError("at least one canonical region is required")
        unknown = [name for name in regions if name not in REGION_ORDER]
        if unknown: raise MaskBuildError(f"unknown canonical regions {unknown}")
        per_region: dict[str, np.ndarray] = {}
        sources: dict[str, str] = {}
        union = np.zeros((self.height, self.width), dtype=bool)
        for name in [item for item in REGION_ORDER if item in regions]:
            mask, source = self.region(name)
            mask = np.asarray(mask, dtype=bool)
            if not mask.any(): raise MaskBuildError(f"region {name!r} produced an empty mask")
            per_region[name] = mask
            sources[name] = source
            union |= mask
        if not union.any(): raise MaskBuildError("requested region union is empty")
        support, achieved = self._apply_coverage(union, coverage, seed)
        if not support.any(): raise MaskBuildError("coverage selection produced an empty support mask")
        result = RegionMaskResult(
            requested_region_mask=union.astype(np.float32)[None],
            operator_support_mask=support.astype(np.float32)[None],
            per_region_masks={name: mask.astype(np.float32)[None] for name, mask in per_region.items()},
            region_sources=sources, requested_coverage=float(coverage), achieved_coverage=float(achieved),
            mask_hash=mask_hash(support),
            metadata={"mask_policy_version": MASK_POLICY_VERSION, "policy": "parsing_first_geometry_fallback",
                      "geometry_shape": geometry_shape, "seed": int(seed),
                      "coverage_tolerance": float(coverage_tolerance),
                      "coverage_error": round(abs(float(achieved) - float(coverage)), 6),
                      "coverage_within_tolerance": bool(abs(float(achieved) - float(coverage)) <= float(coverage_tolerance)),
                      "requested_pixels": int(union.sum()), "support_pixels": int(support.sum()),
                      "parsing_available": self.parsing is not None,
                      "per_region_pixels": {name: int(mask.sum()) for name, mask in sorted(per_region.items())},
                      "requested_region_mask_sha256": mask_hash(union)})
        return result.validate()


def builder_from_geometry(geometry: Any, *, height: int = 224, width: int = 224) -> RegionMaskBuilder:
    """Build from a `CanonicalGeometry` (the M4 loader contract)."""
    return RegionMaskBuilder(height=height, width=width, parsing=np.asarray(geometry.parsing_labels),
                             landmarks=np.asarray(geometry.landmarks, dtype=np.float32),
                             bbox=np.asarray(geometry.bbox, dtype=np.float32),
                             crop_box=np.asarray(geometry.crop_box, dtype=np.float32))
