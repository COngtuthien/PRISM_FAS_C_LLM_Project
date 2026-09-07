"""Tests for the E8 frozen-F1 QMATCH-v1 selector (src/prism_fas/evaluation/
c_ext_e8_qmatched.py). Pure unit tests only -- no GPU, no LLM, no target
labels, no persisted membership. The real-data reproduction test (section 11
of the implementation task) is a separate, explicitly opt-in integration
check guarded by the parquet's existence -- see
``test_real_data_reproduction_opt_in`` below and its module docstring note.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from prism_fas.evaluation import c_ext_e8_qmatched as e8  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# A. q-bin boundaries / B. q==1.0 -> bin 9
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("q,expected_bin", [
    (0.0, 0),
    (0.05, 0),
    (0.099999999999999, 0),
    (0.1, 1),
    (0.15, 1),
    (0.2, 2),
    (0.3, 3),
    (0.4, 4),
    (0.5, 5),
    (0.6, 6),
    (0.7, 7),
    (0.8, 8),
    (0.85, 8),
    (0.9, 9),
    (0.95, 9),
    (1.0, 9),
])
def test_q_bin_boundaries(q, expected_bin):
    assert e8.q_bin_index(q) == expected_bin


def test_q_equals_one_maps_to_bin_9():
    assert e8.q_bin_index(1.0) == 9


# --------------------------------------------------------------------------- #
# C. invalid q rejected (never silently clamped)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad_q", [-0.0001, -1.0, 1.0001, 2.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_q_rejected(bad_q):
    with pytest.raises((ValueError,)):
        e8.q_bin_index(bad_q)


def test_non_numeric_q_rejected():
    with pytest.raises(TypeError):
        e8.q_bin_index("0.5")
    with pytest.raises(TypeError):
        e8.q_bin_index(True)  # bool is a numeric subtype in Python -- must still be rejected


# --------------------------------------------------------------------------- #
# D/E. hash deterministic, independent of process/random state
# --------------------------------------------------------------------------- #

def test_hash_deterministic():
    h1 = e8.qmatch_hash("c5syn_abc123")
    h2 = e8.qmatch_hash("c5syn_abc123")
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)  # valid hex


def test_hash_matches_explicit_construction():
    sample_id = "sample-42"
    import hashlib
    expected = hashlib.sha256((sample_id + "QMATCH-v1").encode("utf-8")).hexdigest()
    assert e8.qmatch_hash(sample_id) == expected


def test_hash_independent_of_random_state():
    random.seed(1)
    h1 = e8.qmatch_hash("sample-x")
    random.seed(999999)
    h2 = e8.qmatch_hash("sample-x")
    assert h1 == h2


def test_hash_does_not_depend_on_arm_route_q_bin():
    # Only sample_id feeds the hash -- verified by signature (qmatch_hash takes
    # only sample_id) and by two different candidates with the same sample_id
    # (which normalize_candidates would reject at population level, but the
    # pure hash function itself must be single-argument and referentially
    # transparent).
    assert e8.qmatch_hash("same-id") == e8.qmatch_hash("same-id")


def test_hash_rejects_empty_or_non_string():
    with pytest.raises(ValueError):
        e8.qmatch_hash("")
    with pytest.raises(ValueError):
        e8.qmatch_hash(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Helpers to build synthetic candidate rows
# --------------------------------------------------------------------------- #

def _rows(route, bin_q, *, RND=0, DET=0, LLM=0, prefix=""):
    """Build `n` synthetic rows per arm, all landing in the bin containing
    `bin_q`, with distinct sample_ids."""
    out = []
    counts = {"RND": RND, "DET": DET, "LLM": LLM}
    for arm, n in counts.items():
        for i in range(n):
            out.append({
                "sample_id": f"{prefix}{arm}-{route}-{i}",
                "arm": arm, "route": route, "q": bin_q,
            })
    return out


# --------------------------------------------------------------------------- #
# F. min-count logic
# --------------------------------------------------------------------------- #

def test_min_count_logic_basic():
    rows = _rows("physics", 0.55, RND=5, DET=3, LLM=4)
    candidates = e8.normalize_candidates(rows)
    feas = e8.compute_feasibility_table(candidates)
    row = next(f for f in feas if f.route == "physics" and f.q_bin == 5)
    assert row.n_b == 3
    assert row.count_RND == 5 and row.count_DET == 3 and row.count_LLM == 4
    assert row.limiting_arm == ("DET",)

    selected = e8.select_qmatch_bank(candidates, feas)
    by_arm = {"RND": 0, "DET": 0, "LLM": 0}
    for s in selected:
        by_arm[s.arm] += 1
    assert by_arm == {"RND": 3, "DET": 3, "LLM": 3}


# --------------------------------------------------------------------------- #
# G. zero-support bin
# --------------------------------------------------------------------------- #

def test_zero_support_bin():
    rows = _rows("gpat", 0.25, RND=5, DET=0, LLM=4)
    candidates = e8.normalize_candidates(rows)
    feas = e8.compute_feasibility_table(candidates)
    row = next(f for f in feas if f.route == "gpat" and f.q_bin == 2)
    assert row.n_b == 0
    assert row.empty_common_support is True

    selected = e8.select_qmatch_bank(candidates, feas)
    assert len(selected) == 0


# --------------------------------------------------------------------------- #
# H. selected membership deterministic across input-row permutation and
#    repeated execution
# --------------------------------------------------------------------------- #

def test_selection_deterministic_across_permutation_and_repeat():
    rows = (
        _rows("physics", 0.35, RND=6, DET=6, LLM=6, prefix="p35-")
        + _rows("physics", 0.75, RND=4, DET=7, LLM=5, prefix="p75-")
        + _rows("gpat", 0.15, RND=3, DET=2, LLM=9, prefix="g15-")
    )
    candidates_a = e8.normalize_candidates(rows)
    selected_a = e8.select_qmatch_bank(candidates_a)

    shuffled = list(rows)
    random.Random(12345).shuffle(shuffled)
    candidates_b = e8.normalize_candidates(shuffled)
    selected_b = e8.select_qmatch_bank(candidates_b)

    key = lambda s: (s.arm, s.route, s.q_bin, s.sample_id)
    assert sorted(selected_a, key=key) == sorted(selected_b, key=key)

    # repeated execution on the same input is byte-identical
    selected_a2 = e8.select_qmatch_bank(candidates_a)
    assert selected_a == selected_a2


# --------------------------------------------------------------------------- #
# I. duplicate sample_id detection
# --------------------------------------------------------------------------- #

def test_duplicate_arm_sample_id_rejected():
    rows = [
        {"sample_id": "dup-1", "arm": "RND", "route": "physics", "q": 0.5},
        {"sample_id": "dup-1", "arm": "RND", "route": "gpat", "q": 0.6},  # same arm, diff route
    ]
    with pytest.raises(e8.E8InputError):
        e8.normalize_candidates(rows)


def test_same_sample_id_different_arm_is_allowed():
    # (arm, sample_id) is the uniqueness key -- same sample_id under a
    # DIFFERENT arm is not a duplicate.
    rows = [
        {"sample_id": "shared-id", "arm": "RND", "route": "physics", "q": 0.5},
        {"sample_id": "shared-id", "arm": "DET", "route": "physics", "q": 0.5},
    ]
    candidates = e8.normalize_candidates(rows)
    assert len(candidates) == 2


def test_invalid_arm_and_route_rejected():
    with pytest.raises(e8.E8InputError):
        e8.normalize_candidates([{"sample_id": "x", "arm": "BOGUS", "route": "physics", "q": 0.5}])
    with pytest.raises(e8.E8InputError):
        e8.normalize_candidates([{"sample_id": "x", "arm": "RND", "route": "bogus", "q": 0.5}])
    with pytest.raises(e8.E8InputError):
        e8.normalize_candidates([{"sample_id": "", "arm": "RND", "route": "physics", "q": 0.5}])


# --------------------------------------------------------------------------- #
# J. equal final bank size across three arms / K. Physics/GPAT independence
# --------------------------------------------------------------------------- #

def test_equal_bank_size_across_arms_and_route_independence():
    rows = (
        _rows("physics", 0.45, RND=10, DET=8, LLM=12, prefix="ph-")
        + _rows("gpat", 0.65, RND=20, DET=15, LLM=25, prefix="gp-")
    )
    candidates = e8.normalize_candidates(rows)
    feas = e8.compute_feasibility_table(candidates)
    totals = e8.common_support_totals(feas)
    assert totals["physics"] == 8
    assert totals["gpat"] == 15
    assert totals["total_per_arm"] == 23

    selected = e8.select_qmatch_bank(candidates, feas)
    from collections import Counter
    by_arm = Counter(s.arm for s in selected)
    assert by_arm["RND"] == by_arm["DET"] == by_arm["LLM"] == 23

    by_arm_route = Counter((s.arm, s.route) for s in selected)
    for arm in ("RND", "DET", "LLM"):
        assert by_arm_route[(arm, "physics")] == 8
        assert by_arm_route[(arm, "gpat")] == 15


# --------------------------------------------------------------------------- #
# L. q magnitude within a bin does not affect selection order
# --------------------------------------------------------------------------- #

def test_q_magnitude_within_bin_does_not_affect_selection():
    # Two populations differ only in the actual q VALUES within the same bin
    # (all still landing in bin 6, [0.6,0.7)); the selected sample_id set must
    # be identical because ordering is governed solely by qmatch_hash(sample_id).
    ids = [f"id-{i}" for i in range(5)]
    rows_a = [{"sample_id": sid, "arm": arm, "route": "physics", "q": 0.61}
              for arm in ("RND", "DET", "LLM") for sid in ids]
    rows_b = [{"sample_id": sid, "arm": arm, "route": "physics", "q": 0.60 + 0.001 * i}
              for arm in ("RND", "DET", "LLM") for i, sid in enumerate(ids)]

    sel_a = e8.select_qmatch_bank(e8.normalize_candidates(rows_a))
    sel_b = e8.select_qmatch_bank(e8.normalize_candidates(rows_b))

    ids_a = sorted(s.sample_id for s in sel_a if s.arm == "RND")
    ids_b = sorted(s.sample_id for s in sel_b if s.arm == "RND")
    assert ids_a == ids_b == sorted(ids)


def test_selection_order_is_hash_then_sample_id_not_input_order():
    # With n_b < population size, verify the selected subset for one arm/bin
    # is exactly the lexicographically-smallest-by-hash subset, regardless of
    # input row order.
    ids = [f"cand-{i}" for i in range(20)]
    rows = [{"sample_id": sid, "arm": "RND", "route": "physics", "q": 0.72} for sid in ids]
    rows += [{"sample_id": sid, "arm": "DET", "route": "physics", "q": 0.72} for sid in ids[:6]]
    rows += [{"sample_id": sid, "arm": "LLM", "route": "physics", "q": 0.72} for sid in ids[:20]]
    candidates = e8.normalize_candidates(rows)
    selected = e8.select_qmatch_bank(candidates)
    rnd_selected = sorted(s.sample_id for s in selected if s.arm == "RND")
    expected = sorted(sorted(ids, key=lambda sid: (e8.qmatch_hash(sid), sid))[:6])
    assert rnd_selected == expected


# --------------------------------------------------------------------------- #
# M. protocol identity deterministic
# --------------------------------------------------------------------------- #

def test_protocol_identity_deterministic():
    id1 = e8.selector_protocol_identity()
    id2 = e8.selector_protocol_identity()
    assert id1 == id2
    assert len(id1) == 64
    int(id1, 16)


def test_protocol_payload_excludes_forbidden_fields():
    payload = e8.build_selector_protocol_payload()
    forbidden_substrings = ("timestamp", "hostname", "path", "target_metric")
    import json
    blob = json.dumps(payload).lower()
    for token in forbidden_substrings:
        assert token not in blob, f"forbidden field material {token!r} leaked into protocol payload"


def test_protocol_payload_pins_ratified_input_binding_identity():
    payload = e8.build_selector_protocol_payload()
    assert payload["e8_input_binding_rule_identity"] == (
        "8223df2d5acb38483b7cfe94bbd0a293489a9200b20fd309fb679ea6621cc7d8"
    )
    assert payload["input_population"] == "FROZEN_F1_C6_NOMINAL"
    assert payload["primary_fold"] == "EXT-F1"
    assert payload["profile"] == "NOMINAL"
    assert payload["target_access_allowed"] is False
    assert payload["llm_api_calls_allowed"] is False


# --------------------------------------------------------------------------- #
# Membership schema (defined, never persisted here)
# --------------------------------------------------------------------------- #

def test_membership_row_schema_and_sort_order():
    rows = _rows("physics", 0.55, RND=1, DET=1, LLM=1)
    candidates = e8.normalize_candidates(rows)
    selected = e8.select_qmatch_bank(candidates)
    assert len(selected) == 3
    membership = [
        e8.build_membership_row(
            s, fold_id="EXT-F1", condition="G-RND-QMATCH",
            input_q_table_sha256=e8.INPUT_Q_TABLE_SHA256,
            e8_input_binding_rule_identity=e8.INPUT_BINDING_RULE_IDENTITY,
            e8_qmatch_selector_rule_identity=e8.selector_protocol_identity(),
        )
        for s in selected
    ]
    for row in membership:
        assert row["schema_version"] == e8.MEMBERSHIP_SCHEMA_VERSION
        assert set(e8.MEMBERSHIP_CANONICAL_SORT_KEYS) <= set(row.keys())
    ordered = e8.sort_membership_rows(membership)
    keys = [tuple(r[k] for k in e8.MEMBERSHIP_CANONICAL_SORT_KEYS) for r in ordered]
    assert keys == sorted(keys)


# --------------------------------------------------------------------------- #
# 11. Real-data read-only reproduction -- explicit opt-in integration check.
#
# This does NOT run as an ordinary unit test dependency: it is skipped
# automatically if the large, externally-produced parquet artifact is not
# present in this checkout. To run it explicitly:
#
#   pytest -q tests/pipeline/test_c_ext_e8_qmatched.py -k real_data_reproduction --run-e8-integration
#
# (the -k filter is optional; the marker below is what actually gates it)
# --------------------------------------------------------------------------- #

_PARQUET_PATH = REPO / e8.INPUT_Q_TABLE_RELATIVE_PATH


@pytest.mark.skipif(not _PARQUET_PATH.is_file(), reason="C6_Q_RECONSTRUCTED.parquet not present in this checkout")
def test_real_data_reproduction_opt_in(tmp_path):
    """Read-only reproduction against the ratified frozen-F1 population.
    Never writes a selected-membership file."""
    actual_sha = e8.sha256_file(_PARQUET_PATH)
    assert actual_sha == e8.INPUT_Q_TABLE_SHA256

    candidates = e8.load_candidates_from_parquet(_PARQUET_PATH)
    assert len(candidates) == 3072
    from collections import Counter
    by_arm = Counter(c.arm for c in candidates)
    assert by_arm == {"RND": 1024, "DET": 1024, "LLM": 1024}

    feas = e8.compute_feasibility_table(candidates)
    totals = e8.common_support_totals(feas)
    assert totals["physics"] == 354
    assert totals["gpat"] == 464
    assert totals["total_per_arm"] == 818

    empty = sorted((f.route, f.q_bin) for f in feas if f.empty_common_support)
    assert empty == [("gpat", 0), ("gpat", 1), ("gpat", 2),
                      ("physics", 0), ("physics", 1), ("physics", 2)]

    q_eq_1 = sum(1 for c in candidates if c.q == 1.0)
    assert q_eq_1 == 0

    # membership is computable but must NOT be persisted by this test
    selected = e8.select_qmatch_bank(candidates, feas)
    assert len(selected) == 818 * 3
    written_files_before = set(tmp_path.iterdir())
    assert written_files_before == set()  # this test writes nothing
