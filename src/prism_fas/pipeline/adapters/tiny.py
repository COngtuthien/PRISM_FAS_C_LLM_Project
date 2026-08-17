"""Deterministic tiny fixtures for the C4-C13 engineering smoke.

Everything here is scaffolding. It exists so the real code paths can run without
the real data, and it is built to be obviously not the real data: noise fields
instead of faces, a handful of samples instead of thousands, no labels that mean
anything. Nothing produced here may enter a scientific artifact, and every
adapter that consumes it stamps `fixture_backed: true` on what it writes.

Two rules keep the fixtures honest.

**Deterministic.** Every builder takes a seed and produces the same bytes on
every machine, so a smoke failure is reproducible and a smoke identity is stable
across a resume. A fixture that varied run to run would make the idempotency
check meaningless.

**Contract-satisfying, not contract-bypassing.** The face fixture really does
carry every parsing class the region builder looks for; the batch fixtures really
do satisfy the frozen tensor contracts. If a fixture had to be waved past a
validator, the validator would no longer be under test — which is the one thing
the smoke is for.
"""
from __future__ import annotations

from typing import Any

#: The LaPa parsing classes the region mask builder resolves against. Imported
#: from the canonical map at call time rather than restated as integers here.
FIXTURE_SIZE = 64


def face_arrays(size: int = FIXTURE_SIZE) -> tuple[Any, Any, Any]:
    """A parsing map, five landmarks and a bounding box for one synthetic face.

    Every region the mask builder can be asked for is present, so
    `RegionMaskBuilder` resolves through its parsing path rather than silently
    falling back to landmark geometry for everything. Testing the fallback is
    useful; testing *only* the fallback would not be.
    """
    import numpy as np

    # The canonical class map lives beside the mask builder that reads it, so
    # the fixture is painted with the same labels the builder resolves against.
    from prism_fas.synthesis.masks import PARSING_LABELS

    def label(name: str, default: int) -> int:
        return int(PARSING_LABELS.get(name, default))

    scale = size / 64.0

    def box(y0: int, y1: int, x0: int, x1: int) -> tuple[slice, slice]:
        return (slice(int(y0 * scale), int(y1 * scale)),
                slice(int(x0 * scale), int(x1 * scale)))

    parsing = np.zeros((size, size), dtype=np.uint8)
    parsing[box(10, 56, 12, 52)] = label("skin", 1)
    parsing[box(4, 12, 12, 52)] = label("hair", 10)
    parsing[box(20, 24, 16, 26)] = label("left_eyebrow", 4)
    parsing[box(20, 24, 38, 48)] = label("right_eyebrow", 5)
    parsing[box(25, 31, 17, 27)] = label("left_eye", 2)
    parsing[box(25, 31, 37, 47)] = label("right_eye", 3)
    parsing[box(30, 40, 28, 36)] = label("nose", 6)
    parsing[box(44, 47, 24, 40)] = label("upper_lip", 7)
    parsing[box(47, 49, 26, 38)] = label("inner_mouth", 8)
    parsing[box(49, 52, 24, 40)] = label("lower_lip", 9)

    landmarks = np.asarray([[22.0, 28.0], [42.0, 28.0], [32.0, 35.0],
                            [26.0, 48.0], [38.0, 48.0]], dtype=np.float32) * scale
    bbox = np.asarray([12.0, 10.0, 52.0, 56.0], dtype=np.float32) * scale
    return parsing, landmarks, bbox


def face_image(size: int = FIXTURE_SIZE, *, seed: int = 20260806) -> Any:
    """A CHW float32 image in [0,1]. Noise, not a face — and it says so."""
    import numpy as np

    generator = np.random.default_rng(seed)
    return generator.random((3, size, size), dtype=np.float32)


def frozen_recipes(repo: Any, arm: str, count: int) -> list[Any]:
    """`count` parsed recipes from a frozen C3 scientific bank.

    Real recipes on purpose. The compiler, the route policy and the conditioning
    encoder are what C5 has to exercise, and feeding them invented payloads would
    only prove the fixtures parse.
    """
    import json
    from pathlib import Path

    from prism_fas.recipes.schema import parse_recipe

    path = Path(repo) / "assets/recipe_banks/c3" / arm.lower() / "recipes.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [parse_recipe(json.loads(line)) for line in lines[:count]]


def gate_metrics(*, accepted: bool, requested_strength: float = 0.4,
                 seed: int = 0) -> dict[str, Any]:
    """One quality-gate metric row, built to pass or to fail on purpose.

    The failing variant fails on a *named* gate rather than on garbage, so the
    rejection-reason bookkeeping is exercised rather than merely triggered.
    """
    measured = requested_strength if accepted else requested_strength * 0.05
    return {
        "face_detection_score": 0.95 if accepted else 0.10,
        "identity_cosine": 0.92 if accepted else 0.20,
        "landmark_nme": 0.02 if accepted else 0.90,
        "outside_mask_parsing_dice": 0.98 if accepted else 0.30,
        "outside_mask_max_error": 0.0,
        "measured_artifact_strength": measured,
        "requested_artifact_strength": requested_strength,
        "fingerprint_score": 0.10 if accepted else 0.99,
        "support_overlap": 0.99 if accepted else 0.10,
        "fixture_seed": seed,
    }


#: A NOMINAL threshold set for the engineering rehearsal only.
#:
#: The scientific NOMINAL is fitted from the source_train benign population at
#: C6 (percentiles declared in `configs/synthesis/quality_gate_m8.yaml`), and
#: that calibration artifact does not exist on this machine. These values let the
#: §11.4 derivation and selection logic execute; they are not thresholds, and the
#: full profile refuses to run without the real calibration.
ENGINEERING_NOMINAL: dict[str, float] = {
    "tau_fd": 0.50, "tau_id": 0.80, "tau_lm": 0.08,
    "tau_parse": 0.90, "tau_out": 0.0, "tau_fp": 0.50,
}


def prediction_rows(count: int, *, threshold: float = 0.5,
                    seed: int = 20260806) -> list[dict[str, Any]]:
    """Fake target prediction rows: no label, no family, no path, no taxonomy.

    The absence is the point. C11's contract is that a prediction carries no
    ground truth, attack family, raw path or subject/session taxonomy, and a
    fixture that carried any of them would let the firewall audit pass on a
    payload the real one would reject.
    """
    import random

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        video = f"fixture_video_{index // 4:03d}"
        rows.append({
            "sample_id": f"fixture_sample_{index:04d}",
            "video_id": video,
            "frame_id": index % 4,
            "p_global": round(rng.random(), 6),
            "threshold": threshold,
        })
    return rows


def evaluation_labels(rows: list[dict[str, Any]], *, seed: int = 7) -> dict[str, int]:
    """Fake video-level labels, produced separately from the predictions.

    Built from the sorted video ids alone and never from the scores, so the
    fixture cannot encode the answer into the thing being scored.

    Deliberately balanced by alternating rather than sampled at random. A random
    draw over a handful of videos can easily produce zero attack presentations,
    and the canonical metrics correctly refuse to compute APCER/ACER on such a
    population — so a random fixture would fail for a reason that has nothing to
    do with the code under test. `seed` is accepted and unused; it keeps the
    signature stable for callers that vary it.
    """
    videos = sorted({row["video_id"] for row in rows})
    return {video: index % 2 for index, video in enumerate(videos)}


__all__ = ["FIXTURE_SIZE", "face_arrays", "face_image", "frozen_recipes", "gate_metrics",
           "ENGINEERING_NOMINAL", "prediction_rows", "evaluation_labels"]
