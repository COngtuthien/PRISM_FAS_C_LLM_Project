"""Full validation of a built M8 synthetic bank.

Everything is re-derived from the bank on disk: hashes are recomputed, PNGs and
NPZs are decoded, the outside-mask invariant is re-proven against the real
`source_train` live crops and every accepted row is re-gated against the frozen
thresholds. Never imports modal.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from typing import Any
import numpy as np
from prism_fas.utils.core import atomic_json_write
from .audit import FORBIDDEN_AUDIT_PATTERNS, FORBIDDEN_AUDIT_TOKENS
from .quality_gate import HARD_GATES, Thresholds, evaluate
from .synthetic_bank import (BANK_LOCK_SCHEMA_VERSION, IMAGE_SIZE, MANIFEST_SCHEMAS, SyntheticBankError,
                             coverage_summary, check_operational_minimums, decode_npz, decode_png,
                             load_manifest, rows_logical_digest, to_uint8)
from .synthetic_shards import load_shards_index, index_digest, validate_shards

VALIDATION_SCHEMA_VERSION = "m8-synthetic-bank-validation-v1"
EXPECTED_CANDIDATES = 1120
# The frozen calibration artifact is copied byte-identically and pinned by SHA, so
# the split-count keys it legitimately carries are not treated as a leak; the
# validator instead asserts its explicit "not used" flags.
CALIBRATION_ALLOWED_TOKENS = ("source_dev", "target_test")


class BankValidationError(RuntimeError):
    """A built bank does not satisfy its declared contract."""


def scan_forbidden(text: str, *, allow: tuple[str, ...] = ()) -> list[str]:
    lowered = str(text).lower()
    hits = [token for token in FORBIDDEN_AUDIT_TOKENS if token not in allow and token.lower() in lowered]
    hits += [pattern for pattern in FORBIDDEN_AUDIT_PATTERNS if re.search(pattern, str(text))]
    return sorted(set(hits))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _live_originals(package_root: Path) -> Any:
    from .m8_pipeline import SampleStore, SourceOnlyAudit
    audit = SourceOnlyAudit()
    return SampleStore.open(Path(package_root), audit), audit


def validate_bank(bank_root: Path, *, package_root: Path | None = None, bank_root_name: str | None = None,
                  recipe_bank_root: Path | None = None, gpat_checkpoint_path: Path | None = None,
                  sample_limit: int | None = None, expected_candidates: int = EXPECTED_CANDIDATES) -> dict[str, Any]:
    """Re-derive and re-check everything the bank claims about itself."""
    root = Path(bank_root)
    errors: list[str] = []
    checks: dict[str, bool] = {}

    def require(name: str, condition: bool, detail: str = "") -> bool:
        checks[name] = bool(condition)
        if not condition: errors.append(f"{name}{': ' + detail if detail else ''}")
        return bool(condition)

    lock_path = root / "BANK_LOCK.json"
    if not lock_path.is_file(): raise BankValidationError(f"{root.name} has no BANK_LOCK.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require("lock_schema_version", lock.get("bank_lock_schema_version") == BANK_LOCK_SCHEMA_VERSION)
    require("lock_status_validated", lock.get("status") == "validated", str(lock.get("status")))
    require("bank_directory_matches_bank_id", (bank_root_name or root.name) == lock.get("bank_id"),
            f"{bank_root_name or root.name} != {lock.get('bank_id')}")

    recomputed = _recompute_identity(lock)
    require("bank_content_identity_reproducible", recomputed == lock.get("bank_content_identity_sha256"),
            f"{recomputed} != {lock.get('bank_content_identity_sha256')}")

    accepted = load_manifest(root / "manifests" / "manifest.parquet", MANIFEST_SCHEMAS["manifest"])
    rejected = load_manifest(root / "manifests" / "rejected.parquet", MANIFEST_SCHEMAS["rejected"])
    failures = load_manifest(root / "manifests" / "failures.parquet", MANIFEST_SCHEMAS["failures"])
    for name, rows in (("manifest", accepted), ("rejected", rejected), ("failures", failures)):
        require(f"{name}_digest", rows_logical_digest(rows, MANIFEST_SCHEMAS[name]) == lock["manifest_digests"][name])
    total = len(accepted) + len(rejected) + len(failures)
    require("candidate_count", int(lock["candidate_count"]) == expected_candidates, str(lock["candidate_count"]))
    require("terminal_accounting", total == int(lock["candidate_count"]), f"{total} != {lock['candidate_count']}")
    require("accepted_count", len(accepted) == int(lock["accepted_count"]))
    require("rejected_count", len(rejected) == int(lock["rejected_count"]))
    require("failed_count", len(failures) == int(lock["failed_count"]))
    ids = [row["synthetic_id"] for row in accepted + rejected + failures]
    require("no_duplicate_synthetic_ids", len(set(ids)) == len(ids), f"{len(ids) - len(set(ids))} duplicates")

    candidate_rows = _candidate_manifest_rows(root / "manifests" / "candidate_manifest.parquet")
    require("candidate_manifest_count", len(candidate_rows) == expected_candidates)
    require("candidate_manifest_covers_terminals",
            {row["synthetic_id"] for row in candidate_rows} == set(ids))

    thresholds = _thresholds(root)
    require("threshold_hash_matches_lock", thresholds.sha256() == lock["threshold_sha256"])
    gate_failures = 0
    for row in accepted:
        result = evaluate({name: row[name] for name in
                           ("face_detection_score", "identity_cosine", "landmark_nme", "outside_mask_parsing_dice",
                            "outside_mask_max_error", "measured_artifact_strength", "requested_artifact_strength",
                            "fingerprint_score", "support_overlap")}, thresholds)
        if not result["accepted"] or row["threshold_hash"] != thresholds.sha256(): gate_failures += 1
        if not (0.0 <= float(row["q"]) <= 1.0) or not np.isfinite(float(row["q"])): gate_failures += 1
    require("every_accepted_row_passes_every_hard_gate", gate_failures == 0, f"{gate_failures} rows")
    require("every_rejected_row_names_a_failed_gate",
            all(str(row["failed_gates"]).strip() and
                set(str(row["failed_gates"]).split("|")) <= set(HARD_GATES) for row in rejected))
    require("recipe_match_not_applicable", all(row["recipe_match"] == "not_applicable" for row in accepted))

    payload_report = _validate_payloads(root, accepted, package_root=package_root, sample_limit=sample_limit)
    require("accepted_files_exist", payload_report["missing_files"] == 0, str(payload_report["missing_files"]))
    require("accepted_hashes_match", payload_report["hash_mismatches"] == 0, str(payload_report["hash_mismatches"]))
    require("images_decode_rgb_224", payload_report["image_shape_errors"] == 0)
    require("masks_binary_0_255", payload_report["mask_value_errors"] == 0)
    require("artifact_maps_load_without_pickle", payload_report["npz_errors"] == 0)
    require("artifact_maps_finite_in_range", payload_report["map_range_errors"] == 0)
    require("artifact_maps_zero_outside_exact_mask", payload_report["map_outside_errors"] == 0)
    require("exact_mask_pixel_counts_match", payload_report["mask_pixel_mismatches"] == 0)
    if package_root is not None:
        require("saved_outside_mask_error_exactly_zero", payload_report["outside_mask_errors"] == 0,
                str(payload_report["outside_mask_errors"]))

    shard_index = load_shards_index(root / "shards_index.parquet")
    require("shards_index_digest", index_digest(shard_index) == lock["shards_index_sha256"])
    require("shard_count", len(shard_index) == int(lock["shard_count"]))
    locked_shards = {row["shard_name"]: row for row in lock["shards"]}
    require("shard_hashes_match_lock",
            all(locked_shards.get(row["shard_name"], {}).get("sha256") == row["sha256"] for row in shard_index))
    shard_report = validate_shards(root, shard_index, accepted)
    require("shards_validate", shard_report["passed"], f"{shard_report['error_count']} errors")

    minimums = check_operational_minimums(accepted, int(lock["candidate_count"]),
                                          lock["operational_minimums"]["declared_minimums"])
    require("operational_minimums", minimums["passed"], json.dumps(minimums["checks"]))

    leak = _leak_scan(root)
    require("no_target_or_private_fields", not leak["hits"], json.dumps(leak["hits"])[:300])
    require("calibration_declares_no_source_dev_or_target", leak["calibration_flags_clean"])
    require("source_only_isolation_evidence", leak["source_isolation_clean"],
            json.dumps(leak["source_isolation_evidence"]))

    parents = _parent_identities(lock, package_root=package_root, recipe_bank_root=recipe_bank_root,
                                 gpat_checkpoint_path=gpat_checkpoint_path)
    for name, value in parents["checks"].items(): require(name, value, parents["details"].get(name, ""))

    report = {
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "bank_id": lock.get("bank_id"),
        "bank_content_identity_sha256": lock.get("bank_content_identity_sha256"),
        "passed": not errors, "errors": errors[:80], "error_count": len(errors), "checks": checks,
        "counts": {"candidates": int(lock["candidate_count"]), "accepted": len(accepted),
                   "rejected": len(rejected), "failed_generation": len(failures)},
        "payload_report": payload_report, "shard_report": {key: shard_report[key] for key in
                                                           ("shards", "members_checked", "parity_ids",
                                                            "covers_every_accepted_row", "error_count", "passed")},
        "operational_minimums": minimums, "coverage": coverage_summary(accepted),
        "parent_identities": parents["identities"], "leak_scan": leak,
        "source_isolation": payload_report.get("source_isolation", {})}
    return report


def _recompute_identity(lock: dict[str, Any]) -> str:
    excluded = tuple(lock.get("identity_excluded_fields") or ())
    return hashlib.sha256(json.dumps({key: value for key, value in lock.items() if key not in excluded},
                                     sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _thresholds(root: Path) -> Thresholds:
    payload = json.loads((Path(root) / "calibration" / "quality_gate.json").read_text(encoding="utf-8"))
    return Thresholds.from_dict(payload["thresholds"])


def _candidate_manifest_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    table = pq.read_table(Path(path)).to_pydict()
    return [{"synthetic_id": table["synthetic_id"][index], "route": table["route"][index],
             "terminal_state": table["terminal_state"][index]} for index in range(len(table["synthetic_id"]))]


def _validate_payloads(root: Path, accepted: list[dict[str, Any]], *, package_root: Path | None,
                       sample_limit: int | None) -> dict[str, Any]:
    """Decode every accepted payload and re-prove the discrete invariants."""
    rows = sorted(accepted, key=lambda row: row["synthetic_id"])
    if sample_limit is not None: rows = rows[:sample_limit]
    store = audit = None
    if package_root is not None: store, audit = _live_originals(package_root)
    report = {"checked": len(rows), "missing_files": 0, "hash_mismatches": 0, "image_shape_errors": 0,
              "mask_value_errors": 0, "npz_errors": 0, "map_range_errors": 0, "map_outside_errors": 0,
              "mask_pixel_mismatches": 0, "outside_mask_errors": 0, "outside_mask_checked": 0,
              "examples": []}
    for row in rows:
        paths = {name: Path(root) / row[f"{name}_relative_path"] for name in ("image", "mask", "artifact_map")}
        if any(not path.is_file() for path in paths.values()):
            report["missing_files"] += 1
            continue
        for name, path in paths.items():
            if _sha256(path) != row[f"{name}_sha256"]: report["hash_mismatches"] += 1
        image = decode_png(paths["image"].read_bytes())
        if image.ndim != 3 or image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
            report["image_shape_errors"] += 1
            continue
        mask = decode_png(paths["mask"].read_bytes())
        if mask.ndim != 2 or mask.shape != (IMAGE_SIZE, IMAGE_SIZE) or set(np.unique(mask).tolist()) - {0, 255}:
            report["mask_value_errors"] += 1
            continue
        binary = mask == 255
        if int(binary.sum()) != int(row["exact_mask_pixels"]): report["mask_pixel_mismatches"] += 1
        try:
            artifact_map = decode_npz(paths["artifact_map"].read_bytes())
        except Exception:                                   # noqa: BLE001 - any decode failure is an npz error
            report["npz_errors"] += 1
            continue
        values = np.asarray(artifact_map, dtype=np.float32)
        if artifact_map.dtype != np.float16 or values.shape != (1, IMAGE_SIZE, IMAGE_SIZE) \
                or not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
            report["map_range_errors"] += 1
        else:
            outside_map = values[0][~binary]
            if outside_map.size and float(np.abs(outside_map).max()) != 0.0: report["map_outside_errors"] += 1
        if store is not None:
            original, _ = store.load(row["live_target_sample_id"])
            original_uint8 = to_uint8(original)
            outside = ~binary
            report["outside_mask_checked"] += 1
            if outside.any():
                error = int(np.abs(image.astype(np.int32) - original_uint8.astype(np.int32))[outside].max())
                if error != 0:
                    report["outside_mask_errors"] += 1
                    if len(report["examples"]) < 5: report["examples"].append(
                        {"synthetic_id": row["synthetic_id"], "outside_mask_max_error": error})
    if audit is not None: report["source_isolation"] = audit.report()
    return report


def _leak_scan(root: Path) -> dict[str, Any]:
    """No manifest, metadata, summary or lock may carry a target token, a private
    field or an absolute path."""
    hits: dict[str, list[str]] = {}
    generation = json.loads((Path(root) / "generation_summary.json").read_text(encoding="utf-8"))
    # The isolation block is the REQUIRED evidence that the forbidden splits were
    # never opened, so its key names legitimately contain those tokens. It is
    # asserted field by field below instead of being pattern-matched.
    isolation = generation.pop("source_isolation", {})
    for relative, payload in (("BANK_LOCK.json", None), ("quality_summary.json", None),
                              ("generation_summary.json", generation)):
        path = Path(root) / relative
        if not path.is_file(): continue
        found = scan_forbidden(json.dumps(payload, sort_keys=True, default=str) if payload is not None
                               else path.read_text(encoding="utf-8"))
        if found: hits[relative] = found
    isolation_required = {"source_train_opened": True, "source_dev_opened": False, "target_test_opened": False,
                          "target_label_artifact_opened": False, "raw_dataset_path_opened": False}
    isolation_clean = all(isolation.get(key) is value for key, value in isolation_required.items())
    metadata = sorted((Path(root) / "metadata").glob("*.json"))
    for path in metadata[:200]:
        found = scan_forbidden(path.read_text(encoding="utf-8"))
        if found: hits[f"metadata/{path.name}"] = found
    for name in ("manifest", "rejected", "failures"):
        rows = load_manifest(Path(root) / "manifests" / f"{name}.parquet", MANIFEST_SCHEMAS[name])
        found = scan_forbidden(json.dumps(rows, default=str))
        if found: hits[f"manifests/{name}.parquet"] = found
    calibration = json.loads((Path(root) / "calibration" / "quality_gate.json").read_text(encoding="utf-8"))
    calibration_hits = scan_forbidden(json.dumps(calibration, default=str), allow=CALIBRATION_ALLOWED_TOKENS)
    if calibration_hits: hits["calibration/quality_gate.json"] = calibration_hits
    clean = (calibration.get("used_source_dev") is False and calibration.get("used_target") is False
             and calibration.get("used_generated_candidates") is False
             and calibration.get("source_isolation", {}).get("source_dev_opened") is False
             and calibration.get("source_isolation", {}).get("target_test_opened") is False)
    return {"hits": hits, "metadata_files_scanned": min(len(metadata), 200),
            "calibration_allowed_tokens": list(CALIBRATION_ALLOWED_TOKENS),
            "calibration_allowed_reason": "package split-count keys and explicit not-used flags, pinned by SHA",
            "calibration_flags_clean": bool(clean),
            "source_isolation_evidence": {key: isolation.get(key) for key in sorted(isolation_required)},
            "source_isolation_clean": bool(isolation_clean)}


def _parent_identities(lock: dict[str, Any], *, package_root: Path | None, recipe_bank_root: Path | None,
                       gpat_checkpoint_path: Path | None) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, str] = {}
    identities = {"package_identity": lock["package_identity"], "recipe_bank_identity": lock["recipe_bank_identity"],
                  "candidate_plan_identity": lock["candidate_plan_identity"],
                  "gpat_pair_plan_identity": lock["gpat_pair_plan_identity"],
                  "gpat_checkpoint_sha256": lock["gpat_checkpoint_sha256"],
                  "quality_calibration_sha256": lock["quality_calibration_sha256"],
                  "threshold_sha256": lock["threshold_sha256"],
                  "fingerprint_reference_sha256": lock["fingerprint_reference_sha256"],
                  "generation_config_sha256": lock["generation_config_sha256"]}
    if package_root is not None:
        actual = json.loads((Path(package_root) / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))["content_identity_sha256"]
        checks["source_package_unchanged"] = actual == lock["package_identity"]
        details["source_package_unchanged"] = f"{actual} != {lock['package_identity']}"
    if recipe_bank_root is not None:
        actual = json.loads((Path(recipe_bank_root) / "BANK_LOCK.json").read_text(encoding="utf-8"))["bank_content_identity_sha256"]
        checks["recipe_bank_unchanged"] = actual == lock["recipe_bank_identity"]
        details["recipe_bank_unchanged"] = f"{actual} != {lock['recipe_bank_identity']}"
    if gpat_checkpoint_path is not None:
        actual = _sha256(Path(gpat_checkpoint_path))
        checks["gpat_checkpoint_hash_matches"] = actual == lock["gpat_checkpoint_sha256"]
        details["gpat_checkpoint_hash_matches"] = f"{actual} != {lock['gpat_checkpoint_sha256']}"
    return {"checks": checks, "details": details, "identities": identities}


def write_validation_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    atomic_json_write(Path(path), report)
    return report


def compare_banks(left: Path, right: Path) -> dict[str, Any]:
    """Identity comparison of two copies of the same bank (remote vs downloaded)."""
    locks = [json.loads((Path(root) / "BANK_LOCK.json").read_text(encoding="utf-8")) for root in (left, right)]
    fields = ("bank_id", "bank_content_identity_sha256", "candidate_count", "accepted_count",
              "rejected_count", "failed_count", "shards_index_sha256", "manifest_digests")
    differences = {name: [lock.get(name) for lock in locks] for name in fields if locks[0].get(name) != locks[1].get(name)}
    return {"identical": not differences, "differences": differences,
            "bank_id": locks[0].get("bank_id"),
            "bank_content_identity_sha256": locks[0].get("bank_content_identity_sha256")}
