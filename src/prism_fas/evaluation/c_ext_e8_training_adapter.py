"""PRISM-FAS-C EXT-Q1Q2 -- E8 Track-G training adapter.

Filters the historical, frozen Flow-1/C6 NOMINAL bank-lock ``selected`` lists
(``reports/full/c6/C6_BANK_LOCK_<ARM>.json``) down to the frozen E8 QMATCH-v1
membership (``reports/c_ext_q1q2_v1/e8_qmatched/membership/``), then exposes
the result through the EXISTING, UNMODIFIED
``prism_fas.detector.c6_bank.C6MatchedBankReader`` -- never a new reader,
never a new sampler, never a new image loader.

Design, verified against the real reader before writing this module (see the
E8 training-integration audit, ``reports/c_ext_q1q2_v1/e8_qmatched/training/
E8_TRAINING_INTEGRATION_AUDIT.{json,md}``):

    frozen E8 membership
        |  (read-only, SHA256-verified)
        v
    this adapter
        |  deterministic filter: SET MEMBERSHIP from E8, RECORD CONTENT and
        |  RELATIVE ORDER from the historical C6 bank lock -- nothing
        |  recomputed, nothing mutated
        v
    filtered bank_lock (in-memory dict, same shape as the historical lock)
        |
        v
    prism_fas.detector.c6_bank.C6MatchedBankReader.open(...)   <- UNCHANGED
        |
        v
    existing Track-G dataset/sampler (M9TrainingDataset's ``bank=`` seam)   <- UNCHANGED

``C6MatchedBankReader.open()`` takes ``bank_lock`` as an in-memory mapping
(see ``prism_fas.detector.c6_bank.open_arm_bank``, which reads the historical
JSON into a dict before passing it in the same way) -- no physical "bank
view" file is required by the reader's own API, so this module does not
write one; see ``E8_TRAINING_ADAPTER_BINDING.json`` for the explicit note.

This module never opens SigLIP2, never trains, never touches GPU, never
accesses target labels, and never regenerates E8 membership or any C6/C5
asset. It is read-only over already-frozen artifacts plus a pure, in-memory
filter.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism_fas.evaluation import c_ext_common as cc  # noqa: E402
from prism_fas.evaluation.c_ext_e8_qmatched import (  # noqa: E402
    INPUT_BINDING_RULE_IDENTITY, PRIMARY_FOLD, QUALITY_PROFILE,
)

# --------------------------------------------------------------------------- #
# Frozen constants (ratified upstream; never redefine, never recompute here)
# --------------------------------------------------------------------------- #

ARMS: tuple[str, ...] = ("RND", "DET", "LLM")
CONDITION_BY_ARM: dict[str, str] = {"RND": "G-RND-QMATCH", "DET": "G-DET-QMATCH", "LLM": "G-LLM-QMATCH"}
ROUTES: tuple[str, ...] = ("physics", "gpat")
ALLOWED_SOURCE_DOMAINS = frozenset({"casia_fasd", "msu_mfsd"})

EXPECTED_COUNT_PER_ARM = 818
EXPECTED_PHYSICS_PER_ARM = 354
EXPECTED_GPAT_PER_ARM = 464
EXPECTED_GRAND_TOTAL = 2454

FROZEN_MEMBERSHIP_PARQUET_SHA256 = "d2f91738c8a250997491eaabf869a2901f3d759ad19b2fb194df806becba656d"
FROZEN_MEMBERSHIP_JSONL_SHA256 = "0122a47129dbaeac8950e47fb9a77e3bf884a45149433cf20a16a50ca2c27070"
FROZEN_MEMBERSHIP_LOCK_SHA256 = "16a25b0d777c37c9dc9ec775e7581e45d34c3b6ced77e328b4d94f72ebe33c07"
SELECTOR_RULE_IDENTITY = "95d60b94e3ff69be420eaf9d7e9cfd636cfe0adc667b30db05aadaeabe430126"
TRACK_G_VARIANT_IDENTITY = "1e26c31bba8922bff9fc15ce2e163d2f82f5a353614c55b52bc0db33b39fac8d"
C7_WINNER_CONFIG_SHA256 = "97d32c36745e1f4758cbc342b5f83f2fa9c87d69f4ba91605678164d32b5b5dd"
FROZEN_QUALITY_THRESHOLD_IDENTITY = "8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19"

MEMBERSHIP_PARQUET_RELATIVE_PATH = "reports/c_ext_q1q2_v1/e8_qmatched/membership/E8_QMATCH_SELECTED_MEMBERSHIP.parquet"
MEMBERSHIP_LOCK_RELATIVE_PATH = "reports/c_ext_q1q2_v1/e8_qmatched/membership/E8_QMATCH_MEMBERSHIP_LOCK.json"
C6_BANK_LOCK_RELATIVE_PATH_TEMPLATE = "reports/full/c6/C6_BANK_LOCK_{arm}.json"

IDENTIFIER_FIELD_CONTRACT = "membership.sample_id == c6_bank_lock.selected[].candidate_id"
ADAPTER_RULE_NAME = "E8_TRACK_G_QMATCH_ADAPTER_V1"


class E8AdapterError(RuntimeError):
    """The E8 training adapter cannot construct a filtered bank view."""


@dataclass(frozen=True)
class E8ArmBinding:
    """Immutable record of everything one arm's filtered bank view is bound to."""
    arm: str
    condition: str
    fold_id: str
    profile: str
    identifier_field: str
    membership_count: int
    physics_count: int
    gpat_count: int
    membership_parquet_sha256: str
    membership_lock_sha256: str
    c6_bank_lock_sha256: str
    quality_threshold_identity: str


def _repo_root(root: Path | None) -> Path:
    return root or cc.repo_root()


# --------------------------------------------------------------------------- #
# 1. Read the frozen E8 membership (verified, read-only)
# --------------------------------------------------------------------------- #

def load_frozen_membership(root: Path | None = None):
    """Read-only load + SHA256 verification of the canonical E8 membership
    parquet. Fails closed on any hash mismatch -- never trusts an unverified
    table."""
    import pandas as pd
    repo = _repo_root(root)
    path = repo / MEMBERSHIP_PARQUET_RELATIVE_PATH
    actual = cc.sha256_file(path)
    if actual != FROZEN_MEMBERSHIP_PARQUET_SHA256:
        raise E8AdapterError(
            f"{path}: SHA256 {actual} != frozen {FROZEN_MEMBERSHIP_PARQUET_SHA256} -- "
            "refusing to trust an unverified E8 membership table"
        )
    return pd.read_parquet(path)


def verify_membership_lock(root: Path | None = None) -> dict:
    """Read-only SHA256-verified load of the E8 membership lock (for cross-
    checking identities recorded there against this module's frozen constants)."""
    repo = _repo_root(root)
    path = repo / MEMBERSHIP_LOCK_RELATIVE_PATH
    actual = cc.sha256_file(path)
    if actual != FROZEN_MEMBERSHIP_LOCK_SHA256:
        raise E8AdapterError(f"{path}: SHA256 {actual} != frozen {FROZEN_MEMBERSHIP_LOCK_SHA256}")
    return cc.read_json(path)


# --------------------------------------------------------------------------- #
# 2. Resolve one arm's membership identity set
# --------------------------------------------------------------------------- #

def resolve_arm_membership(membership_df, arm: str) -> tuple[str, ...]:
    """Exactly the frozen E8 candidate_id set for one arm, as an ordered
    tuple in the membership table's own row order.

    Hard-fails on: an unknown arm; a wrong fold binding; a duplicate
    identifier within the arm; a count or route split that disagrees with
    the frozen 818 / 354+464 contract.
    """
    if arm not in ARMS:
        raise E8AdapterError(f"unknown arm {arm!r}; expected one of {ARMS!r}")
    rows = membership_df.loc[membership_df["arm"] == arm]
    if rows.empty:
        raise E8AdapterError(f"no membership rows found for arm {arm!r}")
    bad_fold = set(rows["fold_id"].unique()) - {PRIMARY_FOLD}
    if bad_fold:
        raise E8AdapterError(f"arm {arm!r} membership carries unexpected fold_id(s) {bad_fold!r}")
    bad_condition = set(rows["condition"].unique()) - {CONDITION_BY_ARM[arm]}
    if bad_condition:
        raise E8AdapterError(f"arm {arm!r} membership carries unexpected condition(s) {bad_condition!r}")

    ids = rows["sample_id"].tolist()
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        dupes = sorted({i for i in ids if i in seen or seen.add(i)})
        raise E8AdapterError(f"duplicate sample_id within arm {arm!r} membership: {dupes!r}")
    if len(ids) != EXPECTED_COUNT_PER_ARM:
        raise E8AdapterError(
            f"arm {arm!r} membership count {len(ids)} != frozen expected {EXPECTED_COUNT_PER_ARM}"
        )

    phys = int((rows["route"] == "physics").sum())
    gpat = int((rows["route"] == "gpat").sum())
    if phys != EXPECTED_PHYSICS_PER_ARM or gpat != EXPECTED_GPAT_PER_ARM:
        raise E8AdapterError(
            f"arm {arm!r} route counts physics={phys}/gpat={gpat} != frozen expected "
            f"{EXPECTED_PHYSICS_PER_ARM}/{EXPECTED_GPAT_PER_ARM}"
        )
    return tuple(ids)


# --------------------------------------------------------------------------- #
# 3. Read the historical, frozen C6 bank lock for one arm
# --------------------------------------------------------------------------- #

def load_historical_bank_lock(arm: str, root: Path | None = None) -> tuple[dict, str]:
    """Read-only load of the real, historical, immutable
    ``reports/full/c6/C6_BANK_LOCK_<ARM>.json``. Returns ``(payload, sha256)``.

    Verifies ``quality_profile == 'NOMINAL'`` and ``quality_threshold_identity``
    match the frozen E8 input binding -- E8 is bound to NOMINAL only, never
    PERMISSIVE (E7-v1.1's unrelated reserve fallback)."""
    if arm not in ARMS:
        raise E8AdapterError(f"unknown arm {arm!r}; expected one of {ARMS!r}")
    repo = _repo_root(root)
    path = repo / C6_BANK_LOCK_RELATIVE_PATH_TEMPLATE.format(arm=arm)
    sha256 = cc.sha256_file(path)
    payload = cc.read_json(path)

    profile = str(payload.get("quality_profile") or "")
    if profile != QUALITY_PROFILE:
        raise E8AdapterError(
            f"{path}: quality_profile {profile!r} != frozen expected {QUALITY_PROFILE!r} -- "
            "E8 is bound to NOMINAL only"
        )
    threshold = str(payload.get("quality_threshold_identity") or "")
    if threshold != FROZEN_QUALITY_THRESHOLD_IDENTITY:
        raise E8AdapterError(
            f"{path}: quality_threshold_identity {threshold!r} != frozen expected "
            f"{FROZEN_QUALITY_THRESHOLD_IDENTITY!r}"
        )
    return payload, sha256


# --------------------------------------------------------------------------- #
# 4. Deterministic filter: E8 set membership x historical C6 record content
# --------------------------------------------------------------------------- #

def filter_bank_lock_to_e8(bank_lock: Mapping[str, Any], arm: str,
                           e8_ids: Sequence[str]) -> dict[str, Any]:
    """Deterministically filter one arm's historical C6 ``selected`` list
    down to exactly the frozen E8 identity set for that arm.

    * SET MEMBERSHIP comes from E8's frozen membership (``e8_ids``).
    * RECORD CONTENT and RELATIVE ORDER come from the historical C6 bank
      lock -- every retained entry is returned byte-for-byte as C6 recorded
      it (no field recomputed, no record mutated), and the filtered list is
      a SUBSEQUENCE of the original ``selected`` order (never re-sorted by
      q, qmatch_hash, or any other E8-side criterion here).

    ``C6MatchedBankReader.open()`` itself subsequently sorts its assembled
    rows by ``synthetic_id`` (== candidate_id) unconditionally -- that is
    the reader's own, pre-existing, already-frozen canonical ordering, not
    something this filter imposes or needs to duplicate.

    Hard-fails on: an unknown arm; a duplicate id in ``e8_ids``; a
    duplicate ``candidate_id`` already present in the historical lock
    itself; any E8 id absent from the historical lock; a forbidden
    (non source-only) ``source_domain`` on any retained entry; or a
    filtered count that disagrees with the requested identity-set size.
    """
    if arm not in ARMS:
        raise E8AdapterError(f"unknown arm {arm!r}; expected one of {ARMS!r}")

    e8_id_list = list(e8_ids)
    e8_id_set = frozenset(e8_id_list)
    if len(e8_id_set) != len(e8_id_list):
        raise E8AdapterError(f"arm {arm!r}: e8_ids contains a duplicate identifier")

    original_selected = list(bank_lock.get("selected") or ())
    original_ids = [str(entry.get("candidate_id") or "") for entry in original_selected]
    if len(set(original_ids)) != len(original_ids):
        raise E8AdapterError(
            f"arm {arm!r}: the historical C6 bank lock itself contains a duplicate "
            "candidate_id -- refusing to filter an inconsistent lock"
        )

    filtered = [entry for entry in original_selected
                if str(entry.get("candidate_id") or "") in e8_id_set]
    filtered_ids = {str(entry.get("candidate_id") or "") for entry in filtered}

    missing = e8_id_set - filtered_ids
    if missing:
        raise E8AdapterError(
            f"arm {arm!r}: {len(missing)} E8-selected candidate_id(s) absent from the "
            f"historical C6 bank lock's 'selected' list, e.g. {sorted(missing)[:5]!r}"
        )
    if len(filtered) != len(e8_id_set):
        raise E8AdapterError(
            f"arm {arm!r}: filtered count {len(filtered)} != frozen E8 identity-set size "
            f"{len(e8_id_set)}"
        )

    bad_domains = sorted({str(entry.get("source_domain")) for entry in filtered} - ALLOWED_SOURCE_DOMAINS)
    if bad_domains:
        raise E8AdapterError(
            f"arm {arm!r}: retained entries carry forbidden source_domain(s) {bad_domains!r} -- "
            "target-domain rows are never permitted in an E8 bank view"
        )

    return {**{key: value for key, value in bank_lock.items() if key != "selected"},
           "selected": filtered}


# --------------------------------------------------------------------------- #
# 5. Open the filtered bank through the EXISTING, unmodified C6MatchedBankReader
# --------------------------------------------------------------------------- #

def open_e8_arm_bank(arm: str, *, candidates_root: Path, recipes: Sequence[Mapping[str, Any]],
                     package_identity: str, recipe_bank_identity: str,
                     root: Path | None = None):
    """Construct the E8-filtered bank_lock for one arm and open it through
    the EXISTING, unmodified ``prism_fas.detector.c6_bank.C6MatchedBankReader``
    -- the only way any code reaches a synthetic sample. No new reader, no
    new sampler, no new image loader is created here."""
    from prism_fas.detector.c6_bank import C6MatchedBankReader

    membership_df = load_frozen_membership(root)
    e8_ids = resolve_arm_membership(membership_df, arm)
    bank_lock, _ = load_historical_bank_lock(arm, root)
    filtered_lock = filter_bank_lock_to_e8(bank_lock, arm, e8_ids)
    return C6MatchedBankReader.open(
        candidates_root=candidates_root, arm=arm, bank_lock=filtered_lock,
        recipes=recipes, package_identity=package_identity,
        recipe_bank_identity=recipe_bank_identity,
        # the filtered lock's content differs from the historical selected_set_sha256
        # it was carved out of by construction, so that check does not apply here
        expected_selected_set_sha256=None,
    )


def resolve_arm_binding(arm: str, root: Path | None = None) -> E8ArmBinding:
    """Build the immutable binding record for one arm without opening any
    C5/C6 pixel payload."""
    membership_df = load_frozen_membership(root)
    e8_ids = resolve_arm_membership(membership_df, arm)
    rows = membership_df.loc[membership_df["arm"] == arm]
    bank_lock, bank_lock_sha256 = load_historical_bank_lock(arm, root)
    # Exercise the filter for validation even though the binding record itself
    # does not need the filtered rows -- any inconsistency must surface here.
    filter_bank_lock_to_e8(bank_lock, arm, e8_ids)
    return E8ArmBinding(
        arm=arm, condition=CONDITION_BY_ARM[arm], fold_id=PRIMARY_FOLD, profile=QUALITY_PROFILE,
        identifier_field=IDENTIFIER_FIELD_CONTRACT, membership_count=len(e8_ids),
        physics_count=int((rows["route"] == "physics").sum()),
        gpat_count=int((rows["route"] == "gpat").sum()),
        membership_parquet_sha256=FROZEN_MEMBERSHIP_PARQUET_SHA256,
        membership_lock_sha256=FROZEN_MEMBERSHIP_LOCK_SHA256,
        c6_bank_lock_sha256=bank_lock_sha256,
        quality_threshold_identity=FROZEN_QUALITY_THRESHOLD_IDENTITY,
    )


# --------------------------------------------------------------------------- #
# Adapter rule identity (frozen before any scientific execution)
# --------------------------------------------------------------------------- #

def build_adapter_rule_payload(per_arm_bank_lock_sha256: Mapping[str, str]) -> dict:
    """Canonical, timestamp-free, path-free, outcome-free payload identifying
    this adapter's frozen rule set."""
    return {
        "rule_name": ADAPTER_RULE_NAME,
        "fold": PRIMARY_FOLD,
        "input_population": "FROZEN_F1_C6_NOMINAL",
        "profile": QUALITY_PROFILE,
        "membership_parquet_sha256": FROZEN_MEMBERSHIP_PARQUET_SHA256,
        "membership_lock_sha256": FROZEN_MEMBERSHIP_LOCK_SHA256,
        "selector_rule_identity": SELECTOR_RULE_IDENTITY,
        "input_binding_rule_identity": INPUT_BINDING_RULE_IDENTITY,
        "track_g_variant_identity": TRACK_G_VARIANT_IDENTITY,
        "c7_winner_config_sha256": C7_WINNER_CONFIG_SHA256,
        "per_arm_c6_bank_lock_sha256": dict(sorted(per_arm_bank_lock_sha256.items())),
        "identifier_field_contract": IDENTIFIER_FIELD_CONTRACT,
        "ordering_policy": "filtered selected list is a subsequence of the historical C6 selected "
                          "order; C6MatchedBankReader.open() subsequently sorts by synthetic_id "
                          "unconditionally (pre-existing, unmodified reader behavior)",
        "filter_semantics": "SET MEMBERSHIP from frozen E8 membership; RECORD CONTENT and RELATIVE "
                           "ORDER from the historical C6 bank lock -- no field recomputed, no "
                           "record mutated",
        "target_access": False,
    }


def adapter_rule_identity(root: Path | None = None) -> str:
    """SHA-256 over the canonical JSON of :func:`build_adapter_rule_payload`,
    with per-arm C6 bank-lock hashes read fresh from the historical,
    immutable lock files (never hardcoded, so the identity can never drift
    from what is actually on disk)."""
    per_arm = {arm: load_historical_bank_lock(arm, root)[1] for arm in ARMS}
    return cc.sha256_bytes(cc.canonical_json_bytes(build_adapter_rule_payload(per_arm)))
