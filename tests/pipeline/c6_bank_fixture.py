"""A production-shaped C5 candidate tree and C6 closure, built in a tmp_path.

This is engineering scaffolding for the C7/C8 rehearsals and it is deliberately
NOT a mock. Every artifact it writes is produced by the module that produces the
real one — `c6_matched_bank.selected_set_digest` and `selector_identity` for the
identities, `c6_scientific.bank_lock_payload` for the lock body, the canonical
`synthetic_bank` codecs for the payload bytes, `c5_raw_generation.CandidateRecord`
for the records — so the rehearsal exercises the real readers against the real
schema rather than against a hand-written dictionary that agrees with the code
only until one of them changes.

What it is not: it is tiny (a handful of candidates per arm, not 1024), and it
lives under a temporary directory. Nothing it writes may enter a scientific
namespace, and no number derived from it is a measurement.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from prism_fas.pipeline.state import atomic_write_json
from prism_fas.synthesis import c5_raw_generation as raw
from prism_fas.synthesis import c6_matched_bank as selector
from prism_fas.synthesis import c6_scientific as science
from prism_fas.synthesis.synthetic_bank import encode_npz, encode_png

ARMS: tuple[str, ...] = selector.ARMS
DOMAINS: tuple[str, ...] = ("casia_fasd", "msu_mfsd")
# The detector's `SyntheticSample.validate` fixes 224x224, so the fixture uses
# the real size rather than a smaller one: a sample that would not validate is
# not a sample the dataset could ever have loaded.
IMAGE_SIZE = 224

PACKAGE_IDENTITY = "f" * 64
RECIPE_BANK_IDENTITY = "e" * 64
POOL_LOCK = "d" * 64
PROFILE = "NOMINAL"
THRESHOLD_IDENTITY = "c" * 64


def _payload_bytes(seed: int) -> tuple[bytes, bytes, bytes, int]:
    """One candidate's three payloads, encoded exactly as C5 encodes them."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    artifact = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    artifact[0, 4:12, 4:12] = np.float32(0.5)
    return (encode_png(image), encode_png(mask),
            encode_npz({"artifact_map": artifact}), int((mask == 255).sum()))


def recipes(count: int) -> list[dict[str, Any]]:
    """C3-shaped recipes. `artifact_types` and `regions` come from these."""
    return [{"recipe_id": f"R-{index:06d}", "schema_version": "1.1",
             "artifacts": [{"name": "blur", "parameters": {}, "strength": 0.4},
                           {"name": "color_shift", "parameters": {}, "strength": 0.2}],
             "regions": ["forehead", "right_cheek"],
             "generator_route": ["physics", "gpat"],
             "seed": 1000 + index} for index in range(count)]


def write_candidate(root: Path, *, arm: str, candidate_id: str, route: str,
                    recipe_id: str, recipe_ordinal: int, position: int,
                    live_target_sample_id: str, seed: int,
                    status: str = raw.GENERATED) -> dict[str, Any]:
    """One C5 candidate directory, written the way `render_arm` writes one."""
    directory = raw.candidate_dir(root, arm, candidate_id)
    directory.mkdir(parents=True, exist_ok=True)
    identity = raw.GenerationIdentity(
        candidate_id=candidate_id, arm=arm, arm_plan_identity="a" * 64,
        source_pair_plan_identity="b" * 64, package_identity=PACKAGE_IDENTITY,
        recipe_bank_identity=RECIPE_BANK_IDENTITY, recipe_id=recipe_id,
        recipe_ordinal=recipe_ordinal, slot=position % 8, position=position,
        route=route, live_target_sample_id=live_target_sample_id,
        spoof_source_sample_id=(f"spoof_{position:04d}" if route == "gpat" else None),
        generator_binding="engineering_fixture", ontology_identity="0" * 64)

    if status == raw.FAILED_GENERATION:
        record = raw.CandidateRecord(
            identity=identity, status=raw.FAILED_GENERATION,
            failure={"stage": f"render_{route}",
                     "exception_type": "SemanticGenerationFailure",
                     "reason": "the artifact finalized to an empty exact mask"})
        raw.write_record(directory, record)
        return record.as_dict()

    image, mask, artifact, pixels = _payload_bytes(seed)
    hashes: dict[str, str] = {}
    for name, payload in ((raw.IMAGE_NAME, image), (raw.MASK_NAME, mask),
                          (raw.ARTIFACT_MAP_NAME, artifact)):
        (directory / name).write_bytes(payload)
        hashes[name] = raw._sha256_bytes(payload)
    record = raw.CandidateRecord(
        identity=identity, status=raw.GENERATED, payload_sha256=hashes,
        trace={"exact_mask_pixels": pixels, "requested_support_pixels": pixels,
               "outside_mask_max_error": 0})
    raw.write_record(directory, record)
    return record.as_dict()


def build_c6_closure(repo: Path, *, per_route: int = 2,
                     arms: Sequence[str] = ARMS) -> dict[str, Any]:
    """A complete, verifiable C6 closure over a tiny candidate tree.

    `per_route` stands in for `PER_ROUTE`; `verify_c6_evidence` is parameterized
    on the real constants, so a test that needs the strict verifier to PASS
    monkeypatches those two constants rather than weakening the verifier.
    """
    candidates_root = repo / "runs/full/c5/scientific/candidates"
    reports = repo / "reports/full/c6"
    reports.mkdir(parents=True, exist_ok=True)

    bank_recipes = recipes(per_route * len(DOMAINS))
    banks: dict[str, dict[str, Any]] = {}
    selected_by_arm: dict[str, list[dict[str, Any]]] = {}

    for arm_index, arm in enumerate(arms):
        rows: list[dict[str, Any]] = []
        position = 0
        for route in selector.ROUTES:
            for slot in range(per_route):
                domain = DOMAINS[slot % len(DOMAINS)]
                recipe = bank_recipes[slot % len(bank_recipes)]
                candidate_id = f"{arm.lower()}-{route}-{slot:04d}"
                write_candidate(
                    candidates_root, arm=arm, candidate_id=candidate_id, route=route,
                    recipe_id=recipe["recipe_id"], recipe_ordinal=slot,
                    position=position,
                    live_target_sample_id=f"{domain}:live:{slot:04d}",
                    seed=1000 * (arm_index + 1) + position)
                rows.append({
                    "candidate_id": candidate_id, "arm": arm, "route": route,
                    "source_domain": domain, "recipe_id": recipe["recipe_id"],
                    "recipe_ordinal": slot,
                    "live_target_sample_id": f"{domain}:live:{slot:04d}",
                    "base_position": position, "selection_step": len(rows) + 1,
                    "recipe_count_before": 0, "live_count_before": 0,
                    "source_domain_quota": per_route // len(DOMAINS) or 1,
                    "canonical_tie_hash": selector.canonical_tie_hash(
                        route=route, source_domain=domain, base_position=position,
                        live_target_sample_id=f"{domain}:live:{slot:04d}"),
                    "q": round(0.5 + 0.01 * slot, 6)})
                position += 1
        selected_by_arm[arm] = rows
        banks[arm] = {
            "size": len(rows),
            "by_route": {route: sum(1 for row in rows if row["route"] == route)
                         for route in selector.ROUTES},
            "exposure": selector.exposure_summary(rows),
            "selected": rows,
            "selected_set_sha256": selector.selected_set_digest(rows)}

    contract = selector.selector_identity(
        quality_profile_identity=THRESHOLD_IDENTITY,
        c5_pool_lock_sha256=POOL_LOCK,
        decision_set_sha256=selector.decision_set_digest([]))

    atomic_write_json(reports / "C6_PROFILE_SELECTION_LOCK.json", {
        "schema_version": "c6-profile-selection-lock-v1",
        "execution_profile": "full", "scientific_eligible": True,
        "is_scientific_lock": True, "fixture_backed": False,
        "selected_profile": PROFILE, "threshold_identity": THRESHOLD_IDENTITY,
        "ba_sep_not_used_for_profile_selection": True, "target_access": 0})

    atomic_write_json(reports / "C6_MATCHED_BANKS.json", {
        "schema_version": "c6-matched-banks-v1",
        "execution_profile": "full", "scientific_eligible": True,
        "selector": selector.SELECTOR_NAME, "selected_profile": PROFILE,
        "selector_identity": contract, "fixture_backed": False,
        "banks": {arm: {key: bank[key] for key in
                        ("size", "by_route", "exposure", "selected_set_sha256")}
                  for arm, bank in banks.items()}})

    for arm, bank in banks.items():
        closure = {"planned": bank["size"], "covered": bank["size"], "closed": True,
                   "unaccounted": []}
        atomic_write_json(reports / f"C6_BANK_LOCK_{arm}.json", {
            **science.bank_lock_payload(
                arm=arm, bank=bank, selector_contract=contract, profile=PROFILE,
                threshold_identity=THRESHOLD_IDENTITY, c5_pool_lock_sha256=POOL_LOCK,
                provenance=closure),
            "execution_profile": "full", "scientific_eligible": True,
            "generated_at_utc": "2026-08-24T00:00:00Z", "fixture_backed": False})

    return {"candidates_root": candidates_root, "reports": reports,
            "banks": banks, "selector_identity": contract,
            "recipes": bank_recipes, "per_route": per_route,
            "package_identity": PACKAGE_IDENTITY,
            "recipe_bank_identity": RECIPE_BANK_IDENTITY}


def install_c3_bank(repo: Path, arm: str, bank_recipes: Sequence[dict[str, Any]]) -> Path:
    """A frozen C3 arm bank, as `c5_arm_plan.load_arm_bank` expects one."""
    from prism_fas.synthesis.c5_arm_plan import C3_BANK_ROOT, RECIPES_PER_ARM

    root = repo / C3_BANK_ROOT / arm.lower()
    root.mkdir(parents=True, exist_ok=True)
    padded = list(bank_recipes)
    while len(padded) < RECIPES_PER_ARM:
        extra = dict(padded[len(padded) % len(bank_recipes)])
        extra["recipe_id"] = f"R-PAD-{len(padded):06d}"
        padded.append(extra)
    (root / "recipes.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True, separators=(",", ":"))
                  for item in padded[:RECIPES_PER_ARM]) + "\n", encoding="utf-8")
    atomic_write_json(root / "C3_BANK.json", {
        "arm": arm, "scientific_eligible": True,
        "bank_identity": RECIPE_BANK_IDENTITY,
        "selected_set_identity": "9" * 64, "ontology_identity": "0" * 64})
    return root


__all__ = ["ARMS", "DOMAINS", "IMAGE_SIZE", "PACKAGE_IDENTITY", "RECIPE_BANK_IDENTITY",
           "POOL_LOCK", "PROFILE", "THRESHOLD_IDENTITY", "recipes", "write_candidate",
           "build_c6_closure", "install_c3_bank"]
