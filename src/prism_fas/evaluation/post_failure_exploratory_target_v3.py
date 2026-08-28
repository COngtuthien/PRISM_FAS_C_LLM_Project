"""POST_FAILURE_EXPLORATORY_TARGET_V3 — Phase E1, final pre-target
provenance/access/statistics hardening.

V2 (`configs/evaluation/post_failure_exploratory_target_v2.yaml`, identity
`2f1beb0b95f01051e06c0ef8a82d06a759d0fe8f81f693c5d3a4d777845196a9`) was
never scientifically executed. This module corrects the remaining
pre-target defects found by a final audit:

  A. The per-row binding never actually carried
     `target_feature_package_identity` — `predict_one_row` read
     `binding.get("target_feature_package_identity", "")`, which always
     resolved to an empty string. Corrected: every row binds the real,
     top-level VERIFIED package identity; inference-config hashing, the
     per-row lock, and the validator all require exact equality with it.
  B. No code commit was ever bound to a prediction — `build_prediction_lock`
     already accepts a `code_commit` parameter, but V2 never populated it.
     Corrected: the commit is read ONCE at execution start
     (`prediction_execution_code_commit`) and threaded through every row and
     the overall lockset.
  C. `target_access: 0` conflated three different accesses. Corrected:
     `target_access_state` — three explicit booleans plus counts.
  F. Partial-result detection now also rejects extra run directories that
     do not belong to the frozen 24-row set.
  (Section 14) Crash safety: inference now writes to a disposable staging
  namespace first, promoting into the final scientific row directories only
  after all 24 rows succeed and validate; the overall lock is written last.

Reuses V1's and V2's own resolution functions verbatim (imported, never
duplicated) and the entire legacy M10 prediction machinery unchanged except
for finally populating `build_prediction_lock`'s existing `code_commit`
parameter. Nothing here can resolve a target label.

IMPLEMENTATION RECONCILIATION (no protocol-identity change; the frozen
`post_failure_exploratory_target_v3.yaml` is untouched): `promote_staged_rows`
is now crash-recoverable. A `PREDICTION_PROMOTION_TRANSACTION_<execution_id>.json`
manifest is written (state `READY_TO_PROMOTE`) BEFORE any row is renamed,
binding the exact 24 row IDs and their staged artifact hashes. A crash
mid-promotion is recovered by validating already-promoted and still-staged
rows against those recorded hashes and resuming file-renames only — zero
model inference — then writing the overall lock and marking the
transaction `COMPLETE`.

SEMANTIC FIX (no protocol-identity change; the frozen
`post_failure_exploratory_target_v3.yaml` is untouched): the target feature
package identity check previously compared a whole-file-tree digest
(V1's `compute_target_feature_package_identity`, modeled on a DIFFERENT
adapter's synthetic-fixture algorithm) against the frozen pin
`c3a29e695ad08c4b31e01533f1d12374f4e30c51f0167c6622cf8168792e48a8` — a hash
of the wrong TYPE, since that pin is actually
`PACKAGE_LOCK.json["content_identity_sha256"]` (a stable JSON hash of the
lock's own declared metadata, per Version B `modal_m10.py::m10_verify_target_features`
and `data.package.builder.finalize_lock`). Corrected:
`verify_target_feature_package_expected_v3`/`_required_v3` and the low-level
`verify_locked_target_feature_package` recompute and compare the LOCK's
own content identity, verify every manifest/shard byte the lock declares,
reuse `data.package.validator.validate_package` for structural checks and
`data.target_eval.assert_features_label_free`/`assert_no_target_identity`
for target isolation, and never open `target_label_root`.
`build_prediction_plan_binding` and `_preflight` now call these V3-only
verifiers directly; V1's and V2's own (defective) verifiers are left
completely unmodified and are no longer called from this module.

EXECUTION FIX (technical, not scientific; no protocol-identity change): a
real GPU `--predict` attempt failed before `target_batches`/`predict_target`
with `VariantError: unknown flags ['recipe_arm']`. `predict_one_row_to_staging`
called `VariantCapabilities.from_flags(binding["flags"])`, resolving a
variant from the FULL bound flags — including `recipe_arm`, C8 treatment/
bank metadata (`source_matrix._track_g_flags`/`_track_r_flags`) that is
deliberately NOT a `detector.variant.FLAG_KEYS` vocabulary entry, and that
the canonical C8 path (`pipeline.adapters.c8._detector_config_for_row`,
reused verbatim by `synthetic_real_probe.construct_row_trainer`) already
strips before resolving a variant. Corrected:
`resolve_verified_row_capabilities` derives capabilities from the variant
the canonical trainer reconstruction already resolved
(`trainer.config.variant`), cross-checked — before any target batch is
created — against the variant portion of the frozen binding's own flags;
`recipe_arm` is never removed from the binding, stays fully bound into
`inference_config_hash`, and `ResolvedExperimentVariant.resolve()` is never
weakened to tolerate it or any other unknown flag globally.

EXECUTION FIX 2 — `E1_TECHNICAL_TARGET_LOADER_POLICY_FAILURE` (technical,
not scientific; no protocol-identity change): a real GPU `--predict`
attempt loaded weights, passed the `recipe_arm` fix, then failed inside
`target_batches`'s `open_package()` with `PackageContractError: package id
'prism_target_eval_v2' != expected 'prism_data_v1_m3b'`.
`construct_row_trainer` correctly reconstructs the SOURCE trainer, so
`trainer.loader_config` still declares the SOURCE package policy
(`configs/data/loader_m4.yaml`); passing it straight into `target_batches`
made `open_package()` correctly reject the valid TARGET package. Corrected:
`build_verified_target_loader_config` rebinds ONLY the two package-policy
fields (`package.expected_package_id`, `package.expected_content_identity_sha256`)
onto an otherwise byte-identical copy of the source config (via
`model_dump` + `LoaderConfig.model_validate`, never mutating
`trainer.loader_config` in place); `resolve_frozen_target_package_reference`
sources the target package id/identity SOLELY from the already-frozen
`PREDICTION_PLAN_BINDING.json`, fail-closed on any inconsistency, before
any target sample is read. `target_prediction.target_batches` itself, and
every other package-validation rule, are unchanged — this fixes the V3
caller, never the generic package contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from prism_fas.evaluation import post_failure_exploratory_target as v1
from prism_fas.evaluation import post_failure_exploratory_target_v2 as v2
from prism_fas.evaluation.contracts import stable_identity

EXIT_PASS, EXIT_BLOCKED, EXIT_USAGE = 0, 2, 3

DIAGNOSTICS_DIR = "reports/full/exploratory_target_v3"
RUN_DIR = "runs/exploratory_target_v3"
STAGING_ROOT = f"{RUN_DIR}/.staging"
PROTOCOL_CONFIG_PATH = "configs/evaluation/post_failure_exploratory_target_v3.yaml"

PREDICTION_PLAN_BINDING_PATH = f"{DIAGNOSTICS_DIR}/PREDICTION_PLAN_BINDING.json"
PREDICTION_LOCK_PATH = f"{DIAGNOSTICS_DIR}/TARGET_PREDICTION_LOCK.json"

EXPECTED_TOTAL_ROWS = 24
STAGING_MARKER = "ENGINEERING_STAGING_NOT_SCIENTIFICALLY_LOCKED"
BINDING_REQUIRED_ROW_FIELDS: tuple[str, ...] = v2.BINDING_REQUIRED_ROW_FIELDS + (
    "target_feature_package_identity",
)


class ExploratoryTargetV3Error(RuntimeError):
    """The V3 exploratory target protocol cannot proceed with the inputs given."""


def _access_state(*, feature_identity: bool = False, prediction_features: bool = False,
                  labels: bool = False, feature_count: int = 0, label_count: int = 0
                  ) -> dict[str, Any]:
    return {"target_feature_identity_accessed": bool(feature_identity),
           "target_prediction_features_accessed": bool(prediction_features),
           "target_labels_accessed": bool(labels),
           "target_feature_access_count": int(feature_count),
           "target_label_access_count": int(label_count)}


# ==============================================================================
# 1. Protocol
# ==============================================================================

def load_protocol(repo: Path) -> dict[str, Any]:
    import yaml

    path = Path(repo) / PROTOCOL_CONFIG_PATH
    if not path.is_file():
        raise ExploratoryTargetV3Error(
            f"POST_FAILURE_EXPLORATORY_TARGET_V3 is not frozen (expected {PROTOCOL_CONFIG_PATH} "
            "to exist and declare status: FROZEN_NOT_RUN)")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, Exception) as error:                # noqa: BLE001
        raise ExploratoryTargetV3Error(f"{PROTOCOL_CONFIG_PATH} did not parse: {error}") from error
    if not isinstance(payload, dict) or payload.get("status") != "FROZEN_NOT_RUN":
        raise ExploratoryTargetV3Error(f"{PROTOCOL_CONFIG_PATH} does not declare status: FROZEN_NOT_RUN")
    if payload.get("target_labels_revealed") is not False or payload.get("target_labels_opened") is not False:
        raise ExploratoryTargetV3Error(
            "the V3 protocol must declare target_labels_revealed: false and "
            "target_labels_opened: false; refusing to load a protocol that starts opened")
    state = payload.get("target_access_state") or {}
    if any(state.get(key) is not False for key in
          ("target_feature_identity_accessed", "target_prediction_features_accessed",
           "target_labels_accessed")):
        raise ExploratoryTargetV3Error("the V3 protocol must start with every target_access_state flag false")
    return payload


_PROTOCOL_IDENTITY_EXCLUDED_KEYS = frozenset({
    "frozen_on", "approved_by", "status", "schema_version", "decision_id", "document_kind",
})


def protocol_identity(protocol: Mapping[str, Any]) -> str:
    material = {key: value for key, value in protocol.items()
               if key not in _PROTOCOL_IDENTITY_EXCLUDED_KEYS}
    return hashlib.sha256(json.dumps(
        material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def active_protocol_identity(repo: Path) -> str:
    return protocol_identity(load_protocol(repo))


def current_code_commit(repo: Path) -> str:
    """The current git commit, determined ONCE at execution start. Never a
    timestamp; reused verbatim from `detector.checkpoint.git_commit`."""
    from prism_fas.detector.checkpoint import git_commit

    return git_commit(Path(repo))


# ==============================================================================
# 1b. Locked target feature package verification (fixes the V1/V2 semantic
#     bug: neither ever compared a whole-file-tree digest against
#     `PACKAGE_LOCK.json["content_identity_sha256"]` — a hash of a different
#     TYPE entirely). V1's `compute_target_feature_package_identity` modeled
#     itself on `pipeline.adapters.c10._package_identity`, a whole-tree digest
#     used for that adapter's OWN synthetic rehearsal fixture — never the real
#     contract the target feature package actually carries.
#
#     The real contract (Version B `modal_m10.py::m10_verify_target_features`,
#     and `data.package.builder.finalize_lock`, both frozen historical
#     evidence): `content_identity_sha256` is a stable JSON hash of the
#     PACKAGE_LOCK body with exactly five volatile fields excluded
#     (`created_at`, `git_commit`, `content_identity_sha256`,
#     `build_seconds`, `environment`) — a hash of DECLARED METADATA, never a
#     second whole-tree digest of the bytes on disk.
# ==============================================================================

V3_PACKAGE_LOCK_IDENTITY_EXCLUDED_FIELDS: tuple[str, ...] = (
    "created_at", "git_commit", "content_identity_sha256", "build_seconds", "environment")


def recompute_package_lock_content_identity(lock: Mapping[str, Any]) -> str:
    """The EXACT historical algorithm, reused verbatim in spirit (never
    reimplemented differently): a stable JSON hash of the lock body with the
    five volatile fields excluded. Matches `modal_m10.py`'s
    `IDENTITY_EXCLUDED_FIELDS` and `data.package.builder.finalize_lock` byte
    for byte."""
    body = {key: value for key, value in lock.items()
           if key not in V3_PACKAGE_LOCK_IDENTITY_EXCLUDED_FIELDS}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def verify_locked_target_feature_package(package_root: Path, *, expected_package_id: str,
                                         expected_content_identity: str) -> dict[str, Any]:
    """Low-level, read-only verifier over an EXPLICIT package root. Takes no
    protocol and resolves no root itself — the caller decides what path this
    checks, so a test can point it at a disposable fixture without touching
    any frozen configuration. Never opens a target label; nothing here reads
    `target_label_root` under any name.

    Fail-closed on every mismatch once the package claims to be present: a
    missing lock, a wrong `package_id`, a non-`validated` status, a lock that
    does not hash to its own recorded `content_identity_sha256`, a lock
    identity that disagrees with the frozen expected pin, a manifest byte
    mismatch, or a shard byte/size mismatch each raise
    `ExploratoryTargetV3Error` rather than return a soft failure. Only an
    absent package root is a quiet, non-raising `NOT_PRESENT_ON_THIS_HOST` —
    matching V1's original preflight-friendly contract.

    `computed_identity` in the return value is ALWAYS the verified lock's own
    `content_identity_sha256` — never a whole-tree digest.
    """
    root = Path(package_root)
    if not root.is_dir():
        return {"present_on_this_host": False, "verified": False,
               "expected_identity": expected_content_identity, "computed_identity": None,
               "reason": "NOT_PRESENT_ON_THIS_HOST"}

    lock_path = root / "PACKAGE_LOCK.json"
    if not lock_path.is_file():
        raise ExploratoryTargetV3Error(
            f"{root}: target feature package has no PACKAGE_LOCK.json; cannot verify")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    if lock.get("package_id") != expected_package_id:
        raise ExploratoryTargetV3Error(
            f"target feature package_id is {lock.get('package_id')!r}, expected {expected_package_id!r}; "
            "fail closed rather than bind the wrong package")
    if lock.get("status") != "validated":
        raise ExploratoryTargetV3Error(
            f"target feature package status is {lock.get('status')!r}, not 'validated'; refusing to bind")

    recomputed = recompute_package_lock_content_identity(lock)
    lock_content_identity = lock.get("content_identity_sha256")
    if recomputed != lock_content_identity:
        raise ExploratoryTargetV3Error(
            "the target feature PACKAGE_LOCK does not hash to its own recorded content_identity_sha256 "
            f"(recomputed {recomputed!r} != lock {lock_content_identity!r}); fail closed rather than "
            "trust a tampered or corrupted lock")
    if lock_content_identity != expected_content_identity:
        raise ExploratoryTargetV3Error(
            f"target feature package content identity mismatch: expected {expected_content_identity!r}, "
            f"lock declares {lock_content_identity!r}; fail closed rather than bind a drifted target package")

    manifest_problems: list[str] = []
    for name, expected_sha in (lock.get("manifest_sha256") or {}).items():
        manifest_path = root / "manifests" / f"{name}.parquet"
        if not manifest_path.is_file():
            manifest_problems.append(f"missing manifest {name}")
        elif hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_sha:
            manifest_problems.append(f"manifest hash mismatch: {name}")
    if manifest_problems:
        raise ExploratoryTargetV3Error(
            f"target feature package manifest integrity failed: {manifest_problems}")

    shard_problems: list[str] = []
    shard_results: list[dict[str, Any]] = []
    for entry in lock.get("shards") or []:
        shard_path = root / "shards" / entry["shard_filename"]
        if not shard_path.is_file():
            shard_problems.append(f"missing shard {entry['shard_filename']}")
            continue
        digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        size = shard_path.stat().st_size
        matches = digest == entry["sha256"] and size == entry["byte_size"]
        if not matches:
            shard_problems.append(f"shard mismatch: {entry['shard_filename']}")
        shard_results.append({"shard": entry["shard_filename"], "rows": entry.get("row_count"),
                              "sha256_matches": digest == entry["sha256"],
                              "byte_size_matches": size == entry["byte_size"]})
    if shard_problems:
        raise ExploratoryTargetV3Error(f"target feature package shard integrity failed: {shard_problems}")

    from prism_fas.data.package.validator import validate_package

    structural = validate_package(root, require_validated_status=True)
    if not structural.get("passed"):
        raise ExploratoryTargetV3Error(
            f"target feature package failed structural validation: {structural.get('errors')}")

    from prism_fas.data.package.manifests import read_manifest
    from prism_fas.data.target_eval import assert_features_label_free, assert_no_target_identity

    feature_rows = read_manifest(root / "manifests" / "target_test_features.parquet")
    label_free_proof = assert_features_label_free(feature_rows)
    no_identity_proof = assert_no_target_identity(root)

    return {"present_on_this_host": True, "verified": True,
           "expected_identity": expected_content_identity, "computed_identity": lock_content_identity,
           "package_id": lock.get("package_id"), "status": lock.get("status"),
           "identity_self_consistent": True, "identity_matches_pin": True,
           "manifest_problems": manifest_problems, "shards": shard_results,
           "structural_validation_passed": True,
           "label_free": label_free_proof, "no_target_identity": no_identity_proof, "reason": ""}


def verify_target_feature_package_expected_v3(repo: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Protocol wrapper: resolves ONLY the target feature root and the
    expected package id/content identity from the frozen V3 protocol, then
    delegates entirely to `verify_locked_target_feature_package`. Never
    resolves or opens `target_label_root` under any name."""
    declared = protocol["target_feature_package"]
    root = Path(repo) / declared["target_feature_root"]
    return verify_locked_target_feature_package(
        root, expected_package_id=str(declared["package_id"]),
        expected_content_identity=str(declared["expected_identity"]))


def verify_target_feature_package_required_v3(repo: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """`--bind-prediction-plan` requires `present_on_this_host AND verified`,
    else fails closed — mirrors V2's `verify_target_feature_package_required`
    enforcement contract exactly, but is built entirely on the CORRECTED V3
    locked-package verifier above; it never calls V1's or V2's defective
    whole-tree-hash check."""
    check = verify_target_feature_package_expected_v3(repo, protocol)
    if not (check.get("present_on_this_host") and check.get("verified")):
        raise ExploratoryTargetV3Error(
            f"target feature package is not present-and-verified on this host ({check}); "
            "--bind-prediction-plan requires verification before binding")
    return {**check, "target_feature_package_identity_verified": True, "target_label_access": 0}


# ==============================================================================
# 2. Row bindings — reuses V1/V2 verbatim, adds target_feature_package_identity
# ==============================================================================

def resolve_all_row_bindings_v3(repo: Path, rows: Sequence[Any], *,
                                target_feature_package_identity: str) -> dict[str, dict[str, Any]]:
    """V2's `resolve_all_row_bindings_v2`, reused verbatim, then annotated
    with the REAL, verified `target_feature_package_identity` (Defect A) —
    never an empty string, never a per-row re-derivation."""
    if not target_feature_package_identity:
        raise ExploratoryTargetV3Error(
            "cannot bind rows without a non-empty, already-verified target_feature_package_identity")
    bindings = v2.resolve_all_row_bindings_v2(repo, rows)
    for binding in bindings.values():
        binding["target_feature_package_identity"] = target_feature_package_identity
    return bindings


# ==============================================================================
# 3. The prediction-plan binding
# ==============================================================================

def build_prediction_plan_binding(repo: Path) -> dict[str, Any]:
    protocol = load_protocol(repo)
    protocol_id = protocol_identity(protocol)
    rows = v1.resolve_target_matrix(repo)
    matrix_id = v1.target_matrix_identity(rows)
    c8_matrix = v1.bind_c8_matrix_identity(repo)
    package_check = verify_target_feature_package_required_v3(repo, protocol)
    package_identity = str(package_check["computed_identity"])
    row_bindings = resolve_all_row_bindings_v3(repo, rows, target_feature_package_identity=package_identity)
    label_seal = v1.verify_target_label_root_sealed(repo, protocol)

    for row_id, binding in row_bindings.items():
        missing = [field for field in BINDING_REQUIRED_ROW_FIELDS if field not in binding]
        if missing:
            raise ExploratoryTargetV3Error(f"{row_id}: binding is missing required fields {missing}")

    binding = {
        "schema_version": "post-failure-exploratory-target-v3-prediction-plan-binding-v1",
        "protocol_identity": protocol_id,
        "target_matrix_identity": matrix_id,
        "c8_matrix_identity": c8_matrix["c8_matrix_identity"],
        "target_feature_package_identity": package_identity,
        "row_count": len(rows),
        "rows": {row_id: dict(sorted(binding.items())) for row_id, binding in sorted(row_bindings.items())},
        "target_feature_package": package_check,
        "target_label_root_seal": label_seal,
        "immutable_upstream_state": dict(protocol["immutable_upstream_state"]),
        "target_labels_opened": False,
        "access_state": _access_state(feature_identity=True, feature_count=1),
    }
    binding["prediction_plan_binding_identity"] = stable_identity(
        {key: value for key, value in binding.items() if key != "prediction_plan_binding_identity"})
    return binding


def verify_binding_unchanged(repo: Path, frozen_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only recomputation, required to match the frozen binding EXACTLY
    before `--predict` may proceed — reused design from V2, now over the
    V3 binding shape (which also includes `target_feature_package_identity`
    per row and at top level)."""
    fresh = build_prediction_plan_binding(repo)
    if fresh != dict(frozen_binding):
        return {"unchanged": False,
               "fresh_prediction_plan_binding_identity": fresh.get("prediction_plan_binding_identity"),
               "frozen_prediction_plan_binding_identity":
                   frozen_binding.get("prediction_plan_binding_identity")}
    return {"unchanged": True}


# ==============================================================================
# 4. Phase E1 real inference driver — staging, then atomic promotion
# ==============================================================================

def execution_identity(*, plan_binding_identity: str, code_commit: str) -> str:
    return stable_identity({"plan_binding_identity": plan_binding_identity, "code_commit": code_commit})[:16]


# The only non-variant metadata flag the frozen C8 target rows carry.
# `source_matrix._track_g_flags`/`_track_r_flags` deliberately add it as
# treatment/provenance metadata; `pipeline.adapters.c8._detector_config_for_row`
# strips it before resolving a variant, and this module must do the same —
# never widen `ResolvedExperimentVariant.resolve()` to tolerate it globally.
KNOWN_NON_VARIANT_ROW_METADATA_FLAGS: frozenset[str] = frozenset({"recipe_arm"})


def resolve_verified_row_capabilities(binding: Mapping[str, Any], *, trainer_variant: Any) -> Any:
    """Derive `VariantCapabilities` from the variant the CANONICAL trainer
    reconstruction (`synthetic_real_probe.construct_row_trainer`, which
    reuses `pipeline.adapters.c8._detector_config_for_row` verbatim)
    actually resolved — never from `VariantCapabilities.from_flags(binding["flags"])`,
    which passes the FULL bound flags (including `recipe_arm`, C8
    treatment/bank metadata, never a `ResolvedExperimentVariant` vocabulary
    key) straight into `ResolvedExperimentVariant.resolve()` and fails with
    `VariantError: unknown flags ['recipe_arm']`.

    Fails closed, before any target batch is created, if: a required
    `detector.variant.FLAG_KEYS` entry is missing from the binding; the
    binding's flags carry any non-variant metadata beyond the one known key
    (`recipe_arm`) — an arbitrary unexpected extra is never silently
    accepted; `recipe_arm` disagrees with the row's own bound `arm`; or the
    canonical trainer's reconstructed variant disagrees with the variant
    portion of the frozen binding's own flags. `recipe_arm` itself is never
    removed from `binding["flags"]` — it stays exactly as bound for
    `inference_config_hash` and every other provenance use."""
    from prism_fas.detector.variant import FLAG_KEYS
    from prism_fas.evaluation.target_prediction import VariantCapabilities

    row_id = str(binding.get("row_id", "<unknown row>"))
    flags = binding["flags"]

    missing = [key for key in FLAG_KEYS if key not in flags]
    if missing:
        raise ExploratoryTargetV3Error(
            f"{row_id}: binding flags are missing required variant flag(s) {missing}")

    unexpected = sorted(set(flags) - set(FLAG_KEYS) - KNOWN_NON_VARIANT_ROW_METADATA_FLAGS)
    if unexpected:
        raise ExploratoryTargetV3Error(
            f"{row_id}: binding flags carry unexpected non-variant metadata {unexpected}; only "
            f"{sorted(KNOWN_NON_VARIANT_ROW_METADATA_FLAGS)} is known non-variant metadata in the "
            "frozen C8 target rows")

    if flags.get("recipe_arm") != binding["arm"]:
        raise ExploratoryTargetV3Error(
            f"{row_id}: binding flags recipe_arm={flags.get('recipe_arm')!r} disagrees with the "
            f"row's own bound arm={binding['arm']!r}; refusing to trust a drifted binding")

    variant_flags = {key: flags[key] for key in FLAG_KEYS}
    trainer_variant_flags = trainer_variant.flags()
    if trainer_variant_flags != variant_flags:
        raise ExploratoryTargetV3Error(
            f"{row_id}: the canonical trainer's reconstructed variant ({trainer_variant_flags}) "
            f"disagrees with the frozen binding's variant flags ({variant_flags}); refusing to "
            "resolve prediction capabilities from an unverified variant")

    return VariantCapabilities.from_variant(trainer_variant)


def resolve_frozen_target_package_reference(frozen_binding: Mapping[str, Any]) -> tuple[str, str]:
    """The V3 target package reference (`target_package_id`,
    `target_content_identity`) comes SOLELY from the already-frozen
    `PREDICTION_PLAN_BINDING.json` — never from whatever live package
    happens to be on disk. Fails closed, before any target sample is ever
    read, unless: the binding names a non-empty package id; the top-level
    frozen `target_feature_package_identity` is non-empty; the nested
    `target_feature_package` verification record's `computed_identity` AND
    `expected_identity` both agree with it; and EVERY row's own
    `target_feature_package_identity` agrees with it too."""
    nested = frozen_binding.get("target_feature_package") or {}
    package_id = str(nested.get("package_id") or "")
    if not package_id:
        raise ExploratoryTargetV3Error("the frozen binding does not name a non-empty target package_id")

    content_identity = str(frozen_binding.get("target_feature_package_identity") or "")
    if not content_identity:
        raise ExploratoryTargetV3Error(
            "the frozen binding does not carry a non-empty target_feature_package_identity")

    for field in ("computed_identity", "expected_identity"):
        nested_value = str(nested.get(field) or "")
        if nested_value != content_identity:
            raise ExploratoryTargetV3Error(
                f"the frozen binding's target_feature_package.{field} ({nested_value!r}) disagrees "
                f"with the top-level frozen target_feature_package_identity ({content_identity!r})")

    rows = frozen_binding.get("rows") or {}
    disagreeing = sorted(row_id for row_id, row in rows.items()
                         if str(row.get("target_feature_package_identity") or "") != content_identity)
    if disagreeing:
        raise ExploratoryTargetV3Error(
            f"row(s) {disagreeing} carry a target_feature_package_identity that disagrees with the "
            f"frozen binding's top-level identity {content_identity!r}")

    return package_id, content_identity


def build_verified_target_loader_config(source_loader_config: Any, *, target_package_id: str,
                                        target_content_identity: str) -> Any:
    """`construct_row_trainer` reconstructs the original SOURCE trainer, so
    `trainer.loader_config` still declares the SOURCE package policy
    (`configs/data/loader_m4.yaml`: `expected_package_id: prism_data_v1_m3b`).
    `target_batches` opens the TARGET feature package through that same
    typed policy, so passing the source-bound config straight through makes
    `open_package()` correctly reject the valid target package.

    Rebinds ONLY the two package-policy fields the target reader needs —
    `package.expected_package_id` and `package.expected_content_identity_sha256`
    — onto an otherwise byte-identical copy of the source config, built via
    `model_dump` + `LoaderConfig.model_validate` rather than mutating
    `source_loader_config` in place, so the trainer's own object is never
    touched. Every other field — image decode/color/size/dtype/range/
    channels_first, `label_mapping`, `sampler`, `backends`, `dataloader`,
    `package.require_validated_status`, `package.integrity_verification` —
    is preserved verbatim, so target package validation is never weakened:
    an exact package-id match, a `validated` status and (since
    `target_content_identity` is always supplied here, never `None`) an
    exact content-identity match are all still required."""
    from prism_fas.data.loader.config import LoaderConfig

    if not target_package_id:
        raise ExploratoryTargetV3Error(
            "cannot build a target loader config without a non-empty target_package_id")
    if not target_content_identity:
        raise ExploratoryTargetV3Error(
            "cannot build a target loader config without a non-empty target_content_identity")

    payload = source_loader_config.model_dump(mode="python")
    payload["package"] = {**payload["package"], "expected_package_id": target_package_id,
                          "expected_content_identity_sha256": target_content_identity}
    return LoaderConfig.model_validate(payload)


def predict_one_row_to_staging(repo: Path, binding: Mapping[str, Any], *, package_root: Path,
                               firewall: Any, staging_root: Path, code_commit: str,
                               target_package_id: str) -> dict[str, Any]:
    """Real, label-free target inference for ONE row, writing ONLY into the
    disposable staging namespace. Reuses `synthetic_real_probe.construct_row_trainer`
    and `target_prediction.target_batches`/`predict_target`/`write_predictions`/
    `build_prediction_lock` verbatim — now finally passing `code_commit` and
    the real `target_feature_package_identity` (Defects A, B)."""
    from prism_fas.evaluation.synthetic_real_probe import CheckpointBinding, construct_row_trainer
    from prism_fas.evaluation.target_prediction import (PREDICTION_LOCK_FILE,
                                                         build_prediction_lock,
                                                         inference_config_hash,
                                                         predict_target, target_batches,
                                                         write_prediction_lock, write_predictions)

    checkpoint_binding = CheckpointBinding(
        arm=str(binding["arm"]), seed=int(binding["seed"]), row_id=str(binding["row_id"]),
        run_identity=str(binding["run_identity"]), config_identity=str(binding["config_identity"]),
        checkpoint_sha256=str(binding["checkpoint_sha256"]),
        checkpoint_path=str(binding["checkpoint_relative_path"]),
        checkpoint_kind=str(binding["checkpoint_kind"]),
        decision_logit_name=str(binding["decision_logit_name"]),
        decision_graph_hash=str(binding["decision_graph_hash"]))
    trainer = construct_row_trainer(repo, checkpoint_binding)
    # Binding/variant consistency is verified BEFORE any target batch is
    # created, any target image/feature is read, or predict_target() runs.
    capabilities = resolve_verified_row_capabilities(binding, trainer_variant=trainer.config.variant)
    variant = str(binding["prediction_variant_id"])
    package_identity = str(binding["target_feature_package_identity"])

    # The SOURCE trainer's own loader config is source-package-bound; the
    # target reader needs the SAME image/sampler/backend/dataloader policy
    # but bound to the TARGET package — never trainer.loader_config as-is.
    target_loader_config = build_verified_target_loader_config(
        trainer.loader_config, target_package_id=target_package_id, target_content_identity=package_identity)
    batches = target_batches(Path(package_root), target_loader_config, cache_root=trainer.cache_root,
                             firewall=firewall)

    config_hash = inference_config_hash(
        variant=variant, flags=binding["flags"], threshold=binding["threshold"],
        unknown_threshold=None, temperature=binding["temperature"],
        package_identity=package_identity, architecture_identity=trainer.model.architecture_identity())

    rows = predict_target(
        trainer.model, batches, capabilities=capabilities, threshold=binding["threshold"],
        unknown_threshold=None, temperature=binding["temperature"],
        checkpoint_hash=binding["checkpoint_sha256"], calibration_hash=binding["calibration_hash"],
        inference_config_hash=config_hash, variant=variant, device=trainer.device)

    row_staging_dir = Path(staging_root) / binding["row_id"]
    row_staging_dir.mkdir(parents=True, exist_ok=True)
    (row_staging_dir / "STAGING_MARKER.json").write_text(
        json.dumps({"marker": STAGING_MARKER, "row_id": binding["row_id"]}), encoding="utf-8")
    prediction_path = row_staging_dir / "target_predictions.parquet"
    write_predictions(prediction_path, rows, variant=variant, firewall=firewall)
    prediction_file_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

    lock = build_prediction_lock(
        experiment_id=binding["experiment_id"], variant=variant, seed=binding["seed"],
        rows=rows, checkpoint_sha256=binding["checkpoint_sha256"],
        source_calibration_sha256=binding["calibration_hash"], calibration_hash=binding["calibration_hash"],
        inference_config_hash=config_hash, target_feature_package_identity=package_identity,
        target_package_id=target_package_id, threshold=binding["threshold"],
        unknown_threshold=None, scientific_config_hash=binding["config_identity"],
        source_matrix_lock_identity=binding["run_identity"], code_commit=code_commit)
    write_prediction_lock(row_staging_dir / PREDICTION_LOCK_FILE, lock)
    return {"row_id": binding["row_id"], "staging_dir": str(row_staging_dir),
           "prediction_file_sha256": prediction_file_sha256, "row_count": len(rows), "lock": lock}


PROMOTION_TRANSACTION_SCHEMA_VERSION = "post-failure-exploratory-target-v3-promotion-transaction-v1"


def _promotion_transaction_path(repo: Path, execution_id: str) -> Path:
    """Lives one level INSIDE `.staging/` (a sibling of the per-execution
    row-staging subdirectory, never inside it) so it is invisible to every
    RUN_DIR-level scan that already excludes the whole `.staging` entry, and
    survives even after the per-execution staging subdirectory is emptied
    and removed."""
    return Path(repo) / RUN_DIR / ".staging" / f"PREDICTION_PROMOTION_TRANSACTION_{execution_id}.json"


def build_promotion_transaction(*, protocol_identity: str, plan_binding_identity: str,
                                execution_id: str, code_commit: str,
                                row_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Defect A: the manifest binding exactly what promotion is allowed to
    move, computed from the ALREADY-STAGED artifacts (all 24 rows already
    predicted; no inference happens after this point). `transaction_identity`
    hashes everything except the mutable `state` field, which is appended
    only after the identity is computed."""
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE

    row_ids = sorted(row_results)
    staged_artifacts: dict[str, Any] = {}
    for row_id in row_ids:
        result = row_results[row_id]
        lock_path = Path(result["staging_dir"]) / PREDICTION_LOCK_FILE
        staged_artifacts[row_id] = {
            "prediction_file_sha256": result["prediction_file_sha256"],
            "prediction_lock_identity": result["lock"]["prediction_lock_identity"],
            "lock_file_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        }
    body = {
        "schema_version": PROMOTION_TRANSACTION_SCHEMA_VERSION,
        "protocol_identity": protocol_identity,
        "plan_binding_identity": plan_binding_identity,
        "execution_identity": execution_id,
        "code_commit": code_commit,
        "row_ids": row_ids,
        "staged_artifacts": staged_artifacts,
    }
    body["transaction_identity"] = stable_identity(body)
    body["state"] = "READY_TO_PROMOTE"
    return body


def _validate_row_artifacts_against_transaction(directory: Path, staged_record: Mapping[str, Any],
                                                 row_id: str) -> None:
    """Pure file hashing — zero model inference, zero re-derivation."""
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE

    prediction_path = directory / "target_predictions.parquet"
    if not prediction_path.is_file():
        raise ExploratoryTargetV3Error(f"{row_id}: prediction file missing at {directory}")
    if hashlib.sha256(prediction_path.read_bytes()).hexdigest() != staged_record["prediction_file_sha256"]:
        raise ExploratoryTargetV3Error(f"{row_id}: prediction file hash disagrees with the promotion transaction")
    lock_path = directory / PREDICTION_LOCK_FILE
    if not lock_path.is_file():
        raise ExploratoryTargetV3Error(f"{row_id}: lock file missing at {directory}")
    if hashlib.sha256(lock_path.read_bytes()).hexdigest() != staged_record["lock_file_sha256"]:
        raise ExploratoryTargetV3Error(f"{row_id}: lock file hash disagrees with the promotion transaction")


def promote_staged_rows(repo: Path, staging_root: Path, row_ids: Sequence[str], *,
                        protocol_identity: str, plan_binding_identity: str, execution_identity: str,
                        code_commit: str, row_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Defect A: crash-recoverable promotion. Writes the transaction manifest
    (state=READY_TO_PROMOTE) BEFORE moving anything. Each row is then
    file-renamed exactly once; a row already sitting in its final directory
    (because a prior invocation crashed after promoting it) is validated
    in place against the transaction's recorded hashes and left untouched —
    never re-inferred, never re-promoted, never silently trusted without a
    hash check. The overall lock is built and written by the CALLER after
    this function returns, and the transaction is marked COMPLETE only
    after every row has been confirmed promoted."""
    from prism_fas.pipeline.state import atomic_write_json

    transaction_path = _promotion_transaction_path(repo, execution_identity)
    transaction = _read_json(transaction_path)
    if transaction is None:
        transaction = build_promotion_transaction(
            protocol_identity=protocol_identity, plan_binding_identity=plan_binding_identity,
            execution_id=execution_identity, code_commit=code_commit, row_results=row_results)
        if sorted(transaction["row_ids"]) != sorted(row_ids):
            raise ExploratoryTargetV3Error("promotion transaction row_ids do not match the requested row set")
        atomic_write_json(transaction_path, transaction)
    elif transaction.get("state") not in ("READY_TO_PROMOTE", "COMPLETE"):
        raise ExploratoryTargetV3Error(f"unrecognized promotion transaction state {transaction.get('state')!r}")
    elif sorted(transaction["row_ids"]) != sorted(row_ids):
        raise ExploratoryTargetV3Error(
            "an existing promotion transaction does not match the requested row set; refusing to proceed")

    run_root = Path(repo) / RUN_DIR
    for row_id in transaction["row_ids"]:
        staged_record = transaction["staged_artifacts"][row_id]
        final_dir = run_root / row_id
        staging_dir = Path(staging_root) / row_id
        if final_dir.is_dir():
            _validate_row_artifacts_against_transaction(final_dir, staged_record, row_id)
            continue
        if not staging_dir.is_dir():
            raise ExploratoryTargetV3Error(f"{row_id}: neither staged nor promoted; transaction cannot recover")
        _validate_row_artifacts_against_transaction(staging_dir, staged_record, row_id)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(final_dir)
        (final_dir / "STAGING_MARKER.json").unlink(missing_ok=True)

    transaction["state"] = "COMPLETE"
    atomic_write_json(transaction_path, transaction)
    return transaction


def _row_result_from_recovered_artifacts(repo: Path, staging_root: Path, row_id: str,
                                         staged_record: Mapping[str, Any]) -> dict[str, Any]:
    """Recovery-only reconstruction of a `predict_one_row_to_staging` result
    from artifacts already on disk (either still staged, or already
    promoted by a prior, interrupted invocation). Reads JSON only — zero
    model inference, zero checkpoint load."""
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE

    final_dir = Path(repo) / RUN_DIR / row_id
    staging_dir = Path(staging_root) / row_id
    directory = final_dir if final_dir.is_dir() else staging_dir
    lock = _read_json(directory / PREDICTION_LOCK_FILE)
    if lock is None:
        raise ExploratoryTargetV3Error(f"{row_id}: cannot recover — no lock file at {directory}")
    return {"row_id": row_id, "staging_dir": str(staging_dir),
           "prediction_file_sha256": staged_record["prediction_file_sha256"],
           "row_count": lock["row_count"], "lock": lock}


# ==============================================================================
# 5. V3 prediction lockset — row_id-keyed, binds code_commit + package identity
# ==============================================================================

LOCKSET_SCHEMA_VERSION = "post-failure-exploratory-target-v3-prediction-lockset-v1"


def build_v3_prediction_lockset(*, protocol_id: str, matrix_id: str, c8_matrix_id: str,
                                package_identity: str, plan_binding_identity: str, code_commit: str,
                                row_bindings: Mapping[str, Mapping[str, Any]],
                                row_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if len(row_results) != EXPECTED_TOTAL_ROWS:
        raise ExploratoryTargetV3Error(
            f"expected exactly {EXPECTED_TOTAL_ROWS} row results, got {len(row_results)}")
    entries: dict[str, Any] = {}
    for row_id, result in row_results.items():
        binding = row_bindings[row_id]
        lock = result["lock"]
        if lock.get("target_feature_package_identity") != package_identity:
            raise ExploratoryTargetV3Error(f"{row_id}: per-row lock package identity disagrees with the top-level one")
        if lock.get("code_commit") != code_commit:
            raise ExploratoryTargetV3Error(f"{row_id}: per-row lock code_commit disagrees with the execution commit")
        entries[row_id] = {
            "row_id": row_id, "experiment_id": binding["experiment_id"], "track": binding["track"],
            "arm": binding["arm"], "seed": int(binding["seed"]),
            "prediction_variant_id": binding["prediction_variant_id"],
            "checkpoint_sha256": binding["checkpoint_sha256"], "calibration_hash": binding["calibration_hash"],
            "threshold": float(binding["threshold"]),
            "temperature": (None if binding["temperature"] is None else float(binding["temperature"])),
            "target_feature_package_identity": package_identity, "code_commit": code_commit,
            "inference_config_hash": lock["inference_config_hash"],
            "prediction_logical_identity": lock["prediction_logical_identity"],
            "prediction_file_sha256": result["prediction_file_sha256"],
            "prediction_lock_identity": lock["prediction_lock_identity"],
            "row_count": lock["row_count"], "video_count": lock["video_count"],
        }
    body = {
        "lockset_schema_version": LOCKSET_SCHEMA_VERSION,
        "protocol_identity": protocol_id, "target_matrix_identity": matrix_id,
        "c8_matrix_identity": c8_matrix_id, "target_feature_package_identity": package_identity,
        "prediction_plan_binding_identity": plan_binding_identity,
        "prediction_execution_code_commit": code_commit,
        "entries": dict(sorted(entries.items())), "entry_count": len(entries),
        "frame_rows_total": sum(int(e["row_count"]) for e in entries.values()),
        "access_state": _access_state(feature_identity=True, prediction_features=True, feature_count=1),
        "target_labels_opened": False,
        "ba_sep_observed_verdict": "FAIL",
        "detector_reliability_lock_c_observed_overall": "FAILED",
        "post_failure_diagnostics_v2": "FAIL",
        "c9_original_confirmatory_path": "BLOCKED",
        "exploratory_target_status": "POST_FAILURE_EXPLORATORY",
        "status": "FROZEN",
    }
    return {**body, "lockset_identity": stable_identity(body)}


# ==============================================================================
# 6. Existing-result validation (Defect E hardened, section 13)
# ==============================================================================

def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def validate_existing_exploratory_prediction_result_v3(repo: Path) -> dict[str, Any]:
    """Comprehensive validation, hardened per section 13: every per-row
    field the frozen binding and lock jointly declare is cross-checked,
    including `target_feature_package_identity` and `code_commit`, and
    the run-directory set may not carry unexpected extras."""
    problems: list[str] = []
    repo = Path(repo)

    binding = _read_json(repo / PREDICTION_PLAN_BINDING_PATH)
    if binding is None:
        return {"valid": False, "problems": ["no PREDICTION_PLAN_BINDING.json on disk"], "lockset": None}
    lockset = _read_json(repo / PREDICTION_LOCK_PATH)
    if lockset is None:
        return {"valid": False, "problems": ["no TARGET_PREDICTION_LOCK.json on disk"], "lockset": None}

    try:
        active_id = active_protocol_identity(repo)
    except ExploratoryTargetV3Error as error:
        return {"valid": False, "problems": [f"active protocol unresolvable: {error}"], "lockset": lockset}

    if lockset.get("status") != "FROZEN":
        problems.append(f"lockset status is {lockset.get('status')!r}, not FROZEN")
    if lockset.get("protocol_identity") != active_id or binding.get("protocol_identity") != active_id:
        problems.append("lockset or binding protocol_identity does not match the active V3 protocol identity")
    if lockset.get("prediction_plan_binding_identity") != binding.get("prediction_plan_binding_identity"):
        problems.append("lockset.prediction_plan_binding_identity does not match the bound plan")
    if lockset.get("target_matrix_identity") != binding.get("target_matrix_identity"):
        problems.append("lockset.target_matrix_identity does not match the bound plan")
    if lockset.get("c8_matrix_identity") != binding.get("c8_matrix_identity"):
        problems.append("lockset.c8_matrix_identity does not match the bound plan")
    if lockset.get("target_feature_package_identity") != binding.get("target_feature_package_identity"):
        problems.append("lockset.target_feature_package_identity does not match the bound plan")
    if not lockset.get("prediction_execution_code_commit"):
        problems.append("lockset does not record a prediction_execution_code_commit")
    if int(lockset.get("entry_count", -1)) != EXPECTED_TOTAL_ROWS:
        problems.append(f"lockset.entry_count is {lockset.get('entry_count')}, expected {EXPECTED_TOTAL_ROWS}")
    if lockset.get("target_labels_opened") is not False:
        problems.append("lockset.target_labels_opened is not False")
    for field, expected in (("ba_sep_observed_verdict", "FAIL"),
                            ("detector_reliability_lock_c_observed_overall", "FAILED"),
                            ("post_failure_diagnostics_v2", "FAIL"),
                            ("c9_original_confirmatory_path", "BLOCKED")):
        if lockset.get(field) != expected:
            problems.append(f"lockset.{field} is not {expected!r}")

    entries = dict(lockset.get("entries") or {})
    binding_rows = dict(binding.get("rows") or {})
    if set(entries) != set(binding_rows):
        problems.append("lockset row_ids do not exactly match the bound plan's row_ids")

    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE, read_predictions, validate_predictions

    code_commit = lockset.get("prediction_execution_code_commit")
    package_identity = lockset.get("target_feature_package_identity")
    for row_id, entry in sorted(entries.items()):
        row_binding = binding_rows.get(row_id)
        if row_binding is None:
            continue
        for field in ("experiment_id", "track", "arm", "prediction_variant_id"):
            if entry.get(field) != row_binding.get(field):
                problems.append(f"{row_id}: lockset.{field} disagrees with the bound row")
        if int(entry.get("seed", -1)) != int(row_binding.get("seed", -2)):
            problems.append(f"{row_id}: lockset.seed disagrees with the bound row")
        for field in ("checkpoint_sha256", "calibration_hash"):
            if entry.get(field) != row_binding.get(field):
                problems.append(f"{row_id}: {field} mismatch")
        if float(entry.get("threshold", -1)) != float(row_binding.get("threshold", -2)):
            problems.append(f"{row_id}: threshold mismatch")
        if entry.get("temperature") != row_binding.get("temperature"):
            problems.append(f"{row_id}: temperature mismatch")
        if entry.get("target_feature_package_identity") != package_identity:
            problems.append(f"{row_id}: entry target_feature_package_identity != lockset value")
        if entry.get("code_commit") != code_commit:
            problems.append(f"{row_id}: entry code_commit != lockset execution commit")

        row_dir = repo / RUN_DIR / row_id
        prediction_path = row_dir / "target_predictions.parquet"
        if not prediction_path.is_file():
            problems.append(f"{row_id}: target_predictions.parquet is missing")
            continue
        real_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        if real_hash != entry.get("prediction_file_sha256"):
            problems.append(f"{row_id}: prediction file sha256 no longer matches the locked value")
        try:
            rows = read_predictions(prediction_path)
            validate_predictions(rows)
            recomputed_logical = _recompute_prediction_logical_identity(rows)
            if recomputed_logical != entry.get("prediction_logical_identity"):
                problems.append(f"{row_id}: prediction_logical_identity recomputed from the file does not match")
        except Exception as error:                        # noqa: BLE001
            problems.append(f"{row_id}: prediction file failed schema validation: {error}")

        row_lock = _read_json(row_dir / PREDICTION_LOCK_FILE)
        if row_lock is None:
            problems.append(f"{row_id}: per-row PREDICTION_LOCK.json is missing")
        else:
            for field in ("prediction_lock_identity", "prediction_logical_identity", "variant",
                         "seed", "checkpoint_sha256", "calibration_hash",
                         "target_feature_package_identity", "code_commit"):
                if row_lock.get(field) != entry.get(field, row_lock.get(field)):
                    if field in ("prediction_lock_identity", "prediction_logical_identity") and \
                            row_lock.get(field) != entry.get(field):
                        problems.append(f"{row_id}: per-row lock.{field} mismatch")
                    elif field == "variant" and row_lock.get(field) != entry.get("prediction_variant_id"):
                        problems.append(f"{row_id}: per-row lock.variant mismatch")
                    elif field in ("seed", "checkpoint_sha256", "calibration_hash",
                                  "target_feature_package_identity", "code_commit") and \
                            row_lock.get(field) != entry.get(field):
                        problems.append(f"{row_id}: per-row lock.{field} mismatch")

    real_row_dirs = ({p.name for p in (repo / RUN_DIR).iterdir() if p.is_dir() and p.name != ".staging"}
                     if (repo / RUN_DIR).is_dir() else set())
    extra = real_row_dirs - set(entries)
    if extra:
        problems.append(f"unexpected run directories not in the frozen row set: {sorted(extra)}")

    recomputed_identity = stable_identity(
        {key: value for key, value in lockset.items() if key != "lockset_identity"})
    if recomputed_identity != lockset.get("lockset_identity"):
        problems.append("lockset does not hash to its own recorded lockset_identity")

    return {"valid": not problems, "problems": problems, "lockset": lockset}


def _recompute_prediction_logical_identity(rows: Sequence[dict[str, Any]]) -> str:
    from prism_fas.evaluation.target_prediction import prediction_logical_identity

    return prediction_logical_identity(rows)


# ==============================================================================
# 7. CLI
# ==============================================================================

def _preflight(repo: Path) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {
        "protocol_resolved": False, "protocol_identity": None,
        "matrix_resolved": False, "row_count": None, "rows_bindable": False,
        "c8_matrix_identity_resolvable": False, "target_feature_package": None,
        "target_label_root_sealed": None, "checkpoint_weights_loaded": False,
        "images_forwarded": False, "access_state": _access_state(),
    }
    try:
        protocol = load_protocol(repo)
        report["protocol_resolved"] = True
        report["protocol_identity"] = protocol_identity(protocol)
    except ExploratoryTargetV3Error as error:
        report["protocol_error"] = str(error)
        return EXIT_BLOCKED, report

    try:
        rows = v1.resolve_target_matrix(repo)
        report["matrix_resolved"] = True
        report["row_count"] = len(rows)
    except v1.ExploratoryTargetError as error:
        report["matrix_error"] = str(error)
        return EXIT_BLOCKED, report

    # Defect: `access_state` must report a REAL access truthfully, including
    # a fail-closed one. `verify_locked_target_feature_package` (and its
    # protocol wrapper here) has exactly one quiet, non-raising outcome —
    # `present_on_this_host: False` when the package root does not exist —
    # every other path either returns `present_on_this_host: True` or RAISES
    # `ExploratoryTargetV3Error`, which by construction only happens AFTER
    # the root was found present and real package content (the lock,
    # manifests, shards) was opened to check it. So a raise here is itself
    # proof of a real, attempted feature-identity access — never a reason to
    # under-report it, even though the verification failed closed.
    package_content_accessed = False
    try:
        report["target_feature_package"] = verify_target_feature_package_expected_v3(repo, protocol)
        package_content_accessed = bool(report["target_feature_package"].get("present_on_this_host"))
    except ExploratoryTargetV3Error as error:
        report["target_feature_package"] = {"verified": False, "error": str(error)}
        package_content_accessed = True
    report["access_state"] = _access_state(
        feature_identity=package_content_accessed, feature_count=(1 if package_content_accessed else 0))

    if isinstance(report["target_feature_package"], dict) and report["target_feature_package"].get("verified"):
        try:
            resolve_all_row_bindings_v3(
                repo, rows, target_feature_package_identity=report["target_feature_package"]["computed_identity"])
            report["rows_bindable"] = True
        except (v1.ExploratoryTargetError, ExploratoryTargetV3Error) as error:
            report["rows_binding_error"] = str(error)
    else:
        report["rows_binding_error"] = "target feature package not present-and-verified"

    try:
        c8_matrix = v1.bind_c8_matrix_identity(repo)
        report["c8_matrix_identity_resolvable"] = True
        report["c8_matrix_identity"] = c8_matrix["c8_matrix_identity"]
    except v1.ExploratoryTargetError as error:
        report["c8_matrix_identity_error"] = str(error)

    try:
        report["target_label_root_sealed"] = v1.verify_target_label_root_sealed(repo, protocol)
    except Exception as error:                            # noqa: BLE001
        report["target_label_root_sealed"] = {"error": f"{type(error).__name__}: {error}"}

    report["ready_for_bind"] = bool(
        report["rows_bindable"] and report["c8_matrix_identity_resolvable"]
        and isinstance(report["target_feature_package"], dict)
        and report["target_feature_package"].get("present_on_this_host")
        and report["target_feature_package"].get("verified")
        and isinstance(report["target_label_root_sealed"], dict)
        and report["target_label_root_sealed"].get("target_labels_opened") is False)
    exit_code = EXIT_PASS if report["ready_for_bind"] else EXIT_BLOCKED
    return exit_code, report


def _bind_prediction_plan(repo: Path) -> tuple[int, dict[str, Any]]:
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"bound": False, "reused": False}
    try:
        binding = build_prediction_plan_binding(repo)
    except (ExploratoryTargetV3Error, v1.ExploratoryTargetError) as error:
        report["error"] = str(error)
        report["access_state"] = _access_state()
        return EXIT_BLOCKED, report

    path = Path(repo) / PREDICTION_PLAN_BINDING_PATH
    existing = _read_json(path)
    if existing is not None:
        if existing != binding:
            report["error"] = ("an existing prediction plan binding differs from the one just "
                               "resolved; refusing to silently overwrite a prior preregistration")
            return EXIT_BLOCKED, report
        report.update({"bound": True, "reused": True, "protocol_identity": binding["protocol_identity"],
                      "target_matrix_identity": binding["target_matrix_identity"], "row_count": binding["row_count"],
                      "target_feature_package_identity": binding["target_feature_package_identity"],
                      "access_state": binding["access_state"]})
        return EXIT_PASS, report

    atomic_write_json(path, binding)
    report.update({"bound": True, "reused": False, "protocol_identity": binding["protocol_identity"],
                  "target_matrix_identity": binding["target_matrix_identity"], "row_count": binding["row_count"],
                  "target_feature_package_identity": binding["target_feature_package_identity"],
                  "access_state": binding["access_state"], "binding_path": PREDICTION_PLAN_BINDING_PATH})
    return EXIT_PASS, report


def _status(repo: Path) -> tuple[int, dict[str, Any]]:
    binding = _read_json(Path(repo) / PREDICTION_PLAN_BINDING_PATH)
    lock_present = (Path(repo) / PREDICTION_LOCK_PATH).is_file()
    report: dict[str, Any] = {
        "prediction_plan_bound": binding is not None,
        "prediction_lock_exists": lock_present,
        "access_state": (binding or {}).get("access_state", _access_state()),
    }
    if binding is None:
        report["reason"] = "NO_PREDICTION_PLAN_BINDING"
        return EXIT_BLOCKED, report
    if not lock_present:
        report["reason"] = "NO_PREDICTION_LOCK_YET"
        return EXIT_BLOCKED, report

    validation = validate_existing_exploratory_prediction_result_v3(repo)
    report["existing_result_validation"] = {"valid": validation["valid"], "problems": validation["problems"]}
    if not validation["valid"]:
        report["reason"] = "EXISTING_RESULT_FAILED_VALIDATION"
        return EXIT_BLOCKED, report
    report["prediction_lock_status"] = validation["lockset"]["status"]
    report["access_state"] = validation["lockset"]["access_state"]
    return EXIT_PASS, report


def _detect_partial_state(repo: Path, expected_rows: int) -> str | None:
    from prism_fas.evaluation.target_prediction import PREDICTION_LOCK_FILE

    run_root = Path(repo) / RUN_DIR
    row_dirs = ([p for p in run_root.iterdir() if p.is_dir() and p.name != ".staging"]
               if run_root.is_dir() else [])
    prediction_files = [d / "target_predictions.parquet" for d in row_dirs
                        if (d / "target_predictions.parquet").is_file()]
    lock_files = [d / PREDICTION_LOCK_FILE for d in row_dirs if (d / PREDICTION_LOCK_FILE).is_file()]
    overall_lock_present = (Path(repo) / PREDICTION_LOCK_PATH).is_file()

    if not prediction_files and not lock_files and not overall_lock_present:
        return None
    if 0 < len(prediction_files) < expected_rows or 0 < len(lock_files) < expected_rows:
        return (f"PARTIAL_SCIENTIFIC_RESULT_SET: {len(prediction_files)} prediction file(s), "
               f"{len(lock_files)} per-row lock(s), expected {expected_rows} of each")
    if len(prediction_files) == expected_rows and len(lock_files) == expected_rows \
            and not overall_lock_present:
        return "all row artifacts present but the overall TARGET_PREDICTION_LOCK.json is missing"
    if overall_lock_present and (len(prediction_files) < expected_rows or len(lock_files) < expected_rows):
        return (f"overall TARGET_PREDICTION_LOCK.json exists but only {len(prediction_files)} prediction "
               f"file(s) / {len(lock_files)} per-row lock(s) of {expected_rows} exist")
    return None


def _predict(repo: Path) -> tuple[int, dict[str, Any]]:
    """Phase E1 execution. NEVER invoked on this laptop for real. Writes to
    a disposable staging namespace first; promotes to the final scientific
    row directories only after all 24 rows succeed, then writes the overall
    lockset last."""
    from prism_fas.pipeline.state import atomic_write_json

    report: dict[str, Any] = {"executed": False}
    binding_path = Path(repo) / PREDICTION_PLAN_BINDING_PATH
    frozen_binding = _read_json(binding_path)
    if frozen_binding is None:
        report["error"] = "no prediction plan binding on disk; run --bind-prediction-plan first"
        report["access_state"] = _access_state()
        return EXIT_BLOCKED, report

    lock_path = Path(repo) / PREDICTION_LOCK_PATH
    if lock_path.is_file():
        validation = validate_existing_exploratory_prediction_result_v3(repo)
        if not validation["valid"]:
            report.update({"error": "EXISTING_RESULT_FAILED_VALIDATION", "problems": validation["problems"]})
            return EXIT_BLOCKED, report
        report.update({"executed": True, "reused_existing_lock": True,
                      "checkpoint_weights_loaded": False, "images_forwarded": False,
                      "prediction_recomputed": False, "lock_status": validation["lockset"]["status"],
                      "access_state": validation["lockset"]["access_state"],
                      "code_lineage_reverified_not_rerun": True,
                      "prediction_execution_code_commit": validation["lockset"]["prediction_execution_code_commit"]})
        return EXIT_PASS, report

    code_commit = current_code_commit(repo)
    execution_id = execution_identity(
        plan_binding_identity=frozen_binding["prediction_plan_binding_identity"], code_commit=code_commit)
    staging_root = Path(repo) / STAGING_ROOT / execution_id
    transaction = _read_json(_promotion_transaction_path(repo, execution_id))
    if transaction is not None and transaction.get("code_commit") != code_commit:
        report["error"] = ("a promotion transaction exists at this execution identity but its recorded "
                           "code_commit disagrees with the current one; refusing to recover")
        return EXIT_BLOCKED, report

    partial_problem = _detect_partial_state(repo, int(frozen_binding["row_count"]))
    if partial_problem is not None and transaction is None:
        report["error"] = partial_problem
        return EXIT_BLOCKED, report

    unchanged = verify_binding_unchanged(repo, frozen_binding)
    if not unchanged["unchanged"]:
        report.update({"error": "PREDICTION_PLAN_BINDING_DRIFTED", **unchanged})
        return EXIT_BLOCKED, report

    recovered_from_promotion_transaction = transaction is not None
    try:
        protocol = load_protocol(repo)
        firewall = v1.build_firewall(repo, protocol)
        package_root = Path(repo) / protocol["roots"]["target_feature_root"]
        row_bindings = frozen_binding["rows"]
        # The target package reference comes SOLELY from the frozen binding
        # — resolved and fail-closed-verified before any target sample is
        # read, whether this run predicts fresh or recovers from a
        # promotion transaction.
        target_package_id, _ = resolve_frozen_target_package_reference(frozen_binding)

        if recovered_from_promotion_transaction:
            row_results: dict[str, Any] = {
                row_id: _row_result_from_recovered_artifacts(
                    repo, staging_root, row_id, transaction["staged_artifacts"][row_id])
                for row_id in sorted(row_bindings)}
        else:
            row_results = {}
            for row_id in sorted(row_bindings):
                row_results[row_id] = predict_one_row_to_staging(
                    repo, row_bindings[row_id], package_root=package_root, firewall=firewall,
                    staging_root=staging_root, code_commit=code_commit,
                    target_package_id=target_package_id)

        lockset = build_v3_prediction_lockset(
            protocol_id=frozen_binding["protocol_identity"], matrix_id=frozen_binding["target_matrix_identity"],
            c8_matrix_id=frozen_binding["c8_matrix_identity"],
            package_identity=frozen_binding["target_feature_package_identity"], code_commit=code_commit,
            plan_binding_identity=frozen_binding["prediction_plan_binding_identity"],
            row_bindings=row_bindings, row_results=row_results)
        promote_staged_rows(
            repo, staging_root, sorted(row_results),
            protocol_identity=frozen_binding["protocol_identity"],
            plan_binding_identity=frozen_binding["prediction_plan_binding_identity"],
            execution_identity=execution_id, code_commit=code_commit, row_results=row_results)
        if staging_root.is_dir() and not any(staging_root.iterdir()):
            staging_root.rmdir()
    except Exception as error:                            # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_BLOCKED, report

    atomic_write_json(lock_path, lockset)
    report.update({"executed": True, "reused_existing_lock": False, "row_count": len(row_results),
                  "lock_path": PREDICTION_LOCK_PATH, "prediction_execution_code_commit": code_commit,
                  "access_state": lockset["access_state"],
                  "recovered_from_promotion_transaction": recovered_from_promotion_transaction,
                  "model_inference_performed": not recovered_from_promotion_transaction})
    return EXIT_PASS, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism_fas.evaluation.post_failure_exploratory_target_v3",
        description="POST_FAILURE_EXPLORATORY_TARGET_V3 — Phase E1 blind target "
                    "prediction, final pre-target hardened. Never a C9 pass path.")
    parser.add_argument("--repo", default=".", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--bind-prediction-plan", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--predict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.preflight_only:
        exit_code, payload = _preflight(args.repo)
    elif args.bind_prediction_plan:
        exit_code, payload = _bind_prediction_plan(args.repo)
    elif args.status:
        exit_code, payload = _status(args.repo)
    else:
        exit_code, payload = _predict(args.repo)

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DIAGNOSTICS_DIR", "RUN_DIR", "STAGING_ROOT", "PROTOCOL_CONFIG_PATH",
    "PREDICTION_PLAN_BINDING_PATH", "PREDICTION_LOCK_PATH", "EXPECTED_TOTAL_ROWS",
    "STAGING_MARKER", "BINDING_REQUIRED_ROW_FIELDS", "LOCKSET_SCHEMA_VERSION",
    "ExploratoryTargetV3Error", "load_protocol", "protocol_identity", "active_protocol_identity",
    "current_code_commit",
    "V3_PACKAGE_LOCK_IDENTITY_EXCLUDED_FIELDS", "recompute_package_lock_content_identity",
    "verify_locked_target_feature_package", "verify_target_feature_package_expected_v3",
    "verify_target_feature_package_required_v3",
    "resolve_all_row_bindings_v3", "build_prediction_plan_binding",
    "verify_binding_unchanged", "execution_identity",
    "KNOWN_NON_VARIANT_ROW_METADATA_FLAGS", "resolve_verified_row_capabilities",
    "resolve_frozen_target_package_reference", "build_verified_target_loader_config",
    "predict_one_row_to_staging",
    "promote_staged_rows", "build_promotion_transaction", "PROMOTION_TRANSACTION_SCHEMA_VERSION",
    "build_v3_prediction_lockset",
    "validate_existing_exploratory_prediction_result_v3",
    "EXIT_PASS", "EXIT_BLOCKED", "EXIT_USAGE",
]
