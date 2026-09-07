"""PRISM-FAS-C EXT-Q1Q2 -- E8: frozen-F1 QMATCH-v1 quality-matched selector.

Scope of this module: PURE, deterministic q-bin/count/select logic for the E8
quality-matched synthetic-bank ablation, plus the frozen E8_QMATCH_V1
selector-protocol payload/identity. It never trains a detector, never touches
GPU, never accesses target labels, never calls an LLM, and never mutates any
E7 or Flow-1/Flow-2 artifact.

Ratified input binding (frozen; do not change without a new amendment):

    reports/c_ext_q1q2_v1/e8_qmatched/E8_FROZEN_F1_INPUT_BINDING_AMENDMENT.json
    e8_input_binding_rule_identity =
        8223df2d5acb38483b7cfe94bbd0a293489a9200b20fd309fb679ea6621cc7d8
    input population = FROZEN_F1_C6_NOMINAL (the historical Flow-1/C6 NOMINAL
        final-selected bank, 1024 samples/arm, 3072 total)
    q table = reports/c_ext_q1q2_v1/e2_quality/reconstructed_q/C6_Q_RECONSTRUCTED.parquet
        sha256 = 87fdc8ea594a487bfef5206c5a0f0c763c7d19451a5f915e67e2237fd7431cd9

This module implements the SELECTOR only. It does not write any scientific
selected-bank membership artifact -- callers decide when/whether to persist
the output of ``select_qmatch_bank``.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism_fas.evaluation.c_ext_common import (  # noqa: E402
    canonical_json_bytes, sha256_bytes, sha256_file,
)

# --------------------------------------------------------------------------- #
# Frozen constants (ratified; never change without a new dated amendment)
# --------------------------------------------------------------------------- #

PROTOCOL_NAME = "E8_QMATCH_V1"
QMATCH_LITERAL = "QMATCH-v1"
INPUT_BINDING_RULE_IDENTITY = "8223df2d5acb38483b7cfe94bbd0a293489a9200b20fd309fb679ea6621cc7d8"
INPUT_POPULATION = "FROZEN_F1_C6_NOMINAL"
PRIMARY_FOLD = "EXT-F1"
QUALITY_PROFILE = "NOMINAL"
QUALITY_THRESHOLD_IDENTITY = "8fa2648643cd526730497ae2d717e17684dda3ecea361fc84929db07ac03bb19"
INPUT_Q_TABLE_RELATIVE_PATH = "reports/c_ext_q1q2_v1/e2_quality/reconstructed_q/C6_Q_RECONSTRUCTED.parquet"
INPUT_Q_TABLE_SHA256 = "87fdc8ea594a487bfef5206c5a0f0c763c7d19451a5f915e67e2237fd7431cd9"

ALLOWED_ARMS: tuple[str, ...] = ("RND", "DET", "LLM")
ALLOWED_ROUTES: tuple[str, ...] = ("physics", "gpat")
N_BINS = 10
Q_BIN_EDGES: tuple[float, ...] = tuple(round(i * 0.1, 1) for i in range(N_BINS + 1))  # 0.0 .. 1.0

EXPECTED_PHYSICS_COMMON_SUPPORT_TOTAL = 354
EXPECTED_GPAT_COMMON_SUPPORT_TOTAL = 464
EXPECTED_TOTAL_UNIQUE_PER_ARM = 818


class E8InputError(ValueError):
    """Raised when the candidate population violates a required invariant."""


# --------------------------------------------------------------------------- #
# 3. Pure q-bin function
# --------------------------------------------------------------------------- #

def q_bin_index(q: float) -> int:
    """Map q in [0.0, 1.0] to one of 10 frozen bins.

    Bins 0-8 are half-open ``[k*0.1, (k+1)*0.1)``; bin 9 is closed on both
    ends, ``[0.9, 1.0]`` -- this is the frozen spec notation verbatim, so
    q == 1.0 maps to bin 9, never dropped.

    NaN, +-inf, q < 0 and q > 1 are rejected (never silently clamped).  Exact
    decimal boundaries (0.1, 0.2, ..., 0.9) are resolved via ``Decimal(repr(q))``
    rather than raw float multiplication, so binary floating-point noise
    (e.g. ``0.3 * 10 == 2.9999999999999996``) can never misclassify a value
    that is genuinely equal to a bin edge.
    """
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise TypeError(f"q must be a real number, got {type(q).__name__}: {q!r}")
    qf = float(q)
    if math.isnan(qf) or math.isinf(qf):
        raise ValueError(f"q must be finite, got {q!r}")
    if qf < 0.0 or qf > 1.0:
        raise ValueError(f"q must satisfy 0.0 <= q <= 1.0, got {q!r}")
    if qf >= 0.9:
        return 9
    scaled = Decimal(repr(qf)) * 10
    idx = int(scaled.to_integral_value(rounding=ROUND_FLOOR))
    if idx < 0:
        idx = 0
    if idx > 8:
        idx = 8
    return idx


# --------------------------------------------------------------------------- #
# 4. Normalized input record
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CandidateRow:
    sample_id: str
    arm: str
    route: str
    q: float
    q_bin: int
    source_domain: str | None = None
    input_row_identity: str | None = None


def normalize_candidates(rows: Iterable[Mapping]) -> tuple[CandidateRow, ...]:
    """Validate and type raw candidate rows into ``CandidateRow`` tuples.

    Rejects (never silently drops or deduplicates):
    * an ``arm`` not in {RND, DET, LLM};
    * a ``route`` not in {physics, gpat};
    * an empty/non-string ``sample_id``;
    * an invalid ``q`` (see ``q_bin_index``);
    * a duplicate ``(arm, sample_id)`` pair -- this also catches the same
      ``sample_id`` reappearing under a different ``route`` for the same
      ``arm``, which the frozen source format defines as impossible.
    """
    seen: dict[tuple[str, str], int] = {}
    out: list[CandidateRow] = []
    for i, r in enumerate(rows):
        arm = r["arm"]
        route = r["route"]
        sample_id = r["sample_id"]
        q = r["q"]
        if arm not in ALLOWED_ARMS:
            raise E8InputError(f"row {i}: arm {arm!r} not in {ALLOWED_ARMS!r}")
        if route not in ALLOWED_ROUTES:
            raise E8InputError(f"row {i}: route {route!r} not in {ALLOWED_ROUTES!r}")
        if not isinstance(sample_id, str) or not sample_id:
            raise E8InputError(f"row {i}: sample_id must be a non-empty string, got {sample_id!r}")
        key = (arm, sample_id)
        seen[key] = seen.get(key, 0) + 1
        q_bin = q_bin_index(q)
        out.append(CandidateRow(
            sample_id=sample_id, arm=arm, route=route, q=float(q), q_bin=q_bin,
            source_domain=r.get("source_domain"),
            input_row_identity=r.get("input_row_identity") or r.get("source_artifact_identity"),
        ))
    dupes = sorted(k for k, n in seen.items() if n > 1)
    if dupes:
        raise E8InputError(
            "duplicate (arm, sample_id) pairs found -- this is impossible under the frozen "
            f"source format and must never be silently deduplicated: {dupes!r}"
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# 5. Frozen stable hash
# --------------------------------------------------------------------------- #

def qmatch_hash(sample_id: str) -> str:
    """SHA-256 hex digest of ``UTF8(sample_id + 'QMATCH-v1')`` -- direct
    concatenation, no separator, no arm/route/q/bin/fold/timestamp/seed
    material. Pure function; no RNG anywhere in selector membership."""
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"sample_id must be a non-empty string, got {sample_id!r}")
    return hashlib.sha256((sample_id + QMATCH_LITERAL).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 6. Counting phase (route x q_bin x arm) -- no selection here
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FeasibilityRow:
    route: str
    q_bin: int
    q_lower: float
    q_upper: float
    q_upper_inclusive: bool
    count_RND: int
    count_DET: int
    count_LLM: int
    n_b: int
    limiting_arm: tuple[str, ...]
    empty_common_support: bool


def compute_feasibility_table(candidates: Sequence[CandidateRow]) -> tuple[FeasibilityRow, ...]:
    """Pure grouping/counting function. Performs NO sample selection."""
    counts: dict[tuple[str, int, str], int] = {}
    for c in candidates:
        key = (c.route, c.q_bin, c.arm)
        counts[key] = counts.get(key, 0) + 1

    rows: list[FeasibilityRow] = []
    for route in ALLOWED_ROUTES:
        for b in range(N_BINS):
            arm_counts = {arm: counts.get((route, b, arm), 0) for arm in ALLOWED_ARMS}
            n_b = min(arm_counts.values())
            limiting = tuple(sorted(a for a, v in arm_counts.items() if v == n_b))
            rows.append(FeasibilityRow(
                route=route, q_bin=b, q_lower=Q_BIN_EDGES[b], q_upper=Q_BIN_EDGES[b + 1],
                q_upper_inclusive=(b == N_BINS - 1),
                count_RND=arm_counts["RND"], count_DET=arm_counts["DET"], count_LLM=arm_counts["LLM"],
                n_b=n_b, limiting_arm=limiting, empty_common_support=(n_b == 0),
            ))
    return tuple(rows)


def common_support_totals(feasibility: Sequence[FeasibilityRow]) -> dict[str, int]:
    physics_total = sum(f.n_b for f in feasibility if f.route == "physics")
    gpat_total = sum(f.n_b for f in feasibility if f.route == "gpat")
    return {
        "physics": physics_total,
        "gpat": gpat_total,
        "total_per_arm": physics_total + gpat_total,
    }


# --------------------------------------------------------------------------- #
# 7. Selection phase
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SelectedRow:
    sample_id: str
    arm: str
    route: str
    q: float
    q_bin: int
    qmatch_hash: str
    source_domain: str | None
    input_row_identity: str | None


def select_qmatch_bank(
    candidates: Sequence[CandidateRow],
    feasibility: Sequence[FeasibilityRow] | None = None,
) -> tuple[SelectedRow, ...]:
    """Pure selector. For each route x q_bin x arm, sorts that group's
    candidates by ``(qmatch_hash(sample_id), sample_id)`` ascending and takes
    the first ``n_b`` -- never reordering by q magnitude within a bin, never
    using any RNG. Every arm's bucket in a given route/bin is guaranteed by
    ``compute_feasibility_table`` to contain at least ``n_b`` candidates, so
    no candidate can be selected twice and every arm's selected count for a
    given route/bin is exactly ``n_b``."""
    if feasibility is None:
        feasibility = compute_feasibility_table(candidates)
    n_b_by_key = {(f.route, f.q_bin): f.n_b for f in feasibility}

    groups: dict[tuple[str, int, str], list[CandidateRow]] = {}
    for c in candidates:
        groups.setdefault((c.route, c.q_bin, c.arm), []).append(c)

    selected: list[SelectedRow] = []
    for route in ALLOWED_ROUTES:
        for b in range(N_BINS):
            n_b = n_b_by_key.get((route, b), 0)
            if n_b == 0:
                continue
            for arm in ALLOWED_ARMS:
                bucket = groups.get((route, b, arm), [])
                ordered = sorted(bucket, key=lambda c: (qmatch_hash(c.sample_id), c.sample_id))
                for c in ordered[:n_b]:
                    selected.append(SelectedRow(
                        sample_id=c.sample_id, arm=c.arm, route=c.route, q=c.q, q_bin=c.q_bin,
                        qmatch_hash=qmatch_hash(c.sample_id),
                        source_domain=c.source_domain, input_row_identity=c.input_row_identity,
                    ))
    return tuple(selected)


# --------------------------------------------------------------------------- #
# 9. Frozen E8_QMATCH_V1 selector-protocol payload/identity
# --------------------------------------------------------------------------- #

def build_selector_protocol_payload() -> dict:
    """Canonical, timestamp-free, path-free, hostname-free, target-metric-free
    protocol payload for the E8 QMATCH-v1 selector implementation."""
    return {
        "protocol_name": PROTOCOL_NAME,
        "e8_input_binding_rule_identity": INPUT_BINDING_RULE_IDENTITY,
        "input_population": INPUT_POPULATION,
        "primary_fold": PRIMARY_FOLD,
        "profile": QUALITY_PROFILE,
        "routes": list(ALLOWED_ROUTES),
        "q_bin_edges": list(Q_BIN_EDGES),
        "q_equals_one_rule": "FINAL_BIN_INCLUSIVE",
        "count_rule": "MIN_ACROSS_RND_DET_LLM_PER_ROUTE_BIN",
        "hash_rule": "SHA256_UTF8_SAMPLE_ID_PLUS_LITERAL_QMATCH_V1",
        "hash_literal": QMATCH_LITERAL,
        "selection_order": "HASH_ASC_THEN_SAMPLE_ID_ASC",
        "target_access_allowed": False,
        "llm_api_calls_allowed": False,
    }


def selector_protocol_identity() -> str:
    """``e8_qmatch_selector_rule_identity`` -- SHA-256 of the canonical JSON
    of ``build_selector_protocol_payload()``. Deterministic across process
    runs; frozen before any scientific membership generation."""
    return sha256_bytes(canonical_json_bytes(build_selector_protocol_payload()))


# --------------------------------------------------------------------------- #
# 12. Future selected-membership output schema (defined, never written here)
# --------------------------------------------------------------------------- #

MEMBERSHIP_SCHEMA_VERSION = "ext-q1q2-e8-qmatch-selected-membership-v1"
MEMBERSHIP_CANONICAL_SORT_KEYS: tuple[str, ...] = ("arm", "route", "q_bin", "qmatch_hash", "sample_id")


def build_membership_row(
    selected: SelectedRow,
    *,
    fold_id: str,
    condition: str,
    input_q_table_sha256: str,
    e8_input_binding_rule_identity: str,
    e8_qmatch_selector_rule_identity: str,
) -> dict:
    """Defines the future selected-membership row schema. Pure function;
    callers decide whether/when to persist the result -- this module never
    writes it."""
    return {
        "schema_version": MEMBERSHIP_SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "fold_id": fold_id,
        "condition": condition,
        "arm": selected.arm,
        "route": selected.route,
        "sample_id": selected.sample_id,
        "q": selected.q,
        "q_bin": selected.q_bin,
        "qmatch_hash": selected.qmatch_hash,
        "source_domain": selected.source_domain,
        "source_locator": selected.input_row_identity,
        "input_q_table_sha256": input_q_table_sha256,
        "e8_input_binding_rule_identity": e8_input_binding_rule_identity,
        "e8_qmatch_selector_rule_identity": e8_qmatch_selector_rule_identity,
    }


def sort_membership_rows(rows: Sequence[Mapping]) -> list[Mapping]:
    return sorted(rows, key=lambda r: tuple(r[k] for k in MEMBERSHIP_CANONICAL_SORT_KEYS))


# --------------------------------------------------------------------------- #
# 11. Real-data read-only reproduction helper (no membership persisted)
# --------------------------------------------------------------------------- #

def load_candidates_from_parquet(path: Path) -> tuple[CandidateRow, ...]:
    """Read-only load of the ratified frozen-F1 q table. Verifies the file's
    SHA-256 against the ratified ``INPUT_Q_TABLE_SHA256`` before trusting any
    row -- fails closed on any mismatch. Requires pandas/pyarrow (not a
    bare-interpreter path)."""
    actual_sha256 = sha256_file(path)
    if actual_sha256 != INPUT_Q_TABLE_SHA256:
        raise E8InputError(
            f"{path}: SHA256 {actual_sha256} != ratified {INPUT_Q_TABLE_SHA256} -- "
            "refusing to trust an unverified input population"
        )
    import pandas as pd
    df = pd.read_parquet(path)
    rows = [
        {
            "sample_id": row.candidate_id,
            "arm": row.arm,
            "route": row.route,
            "q": row.q,
            "source_domain": None,
            "input_row_identity": row.source_artifact_identity,
        }
        for row in df.itertuples(index=False)
    ]
    return normalize_candidates(rows)
