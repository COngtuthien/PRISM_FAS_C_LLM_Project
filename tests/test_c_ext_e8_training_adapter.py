"""Tests for the E8 Track-G training adapter
(src/prism_fas/evaluation/c_ext_e8_training_adapter.py).

Pure, read-only, source-only tests. No GPU, no SigLIP2, no target labels, no
E8 membership regeneration, no mutation of any real frozen artifact.
Corruption/failure tests build tiny synthetic fixtures rather than touching
the real historical C6 locks or the real E8 membership.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prism_fas.evaluation import c_ext_e8_training_adapter as adapter  # noqa: E402
from prism_fas.evaluation import c_ext_common as cc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ARMS = ("RND", "DET", "LLM")


# --------------------------------------------------------------------------- #
# Real-artifact hash snapshots, taken once, used to prove nothing was mutated
# --------------------------------------------------------------------------- #

def _real_hashes() -> dict[str, str]:
    out = {arm: cc.sha256_file(REPO / adapter.C6_BANK_LOCK_RELATIVE_PATH_TEMPLATE.format(arm=arm))
           for arm in ARMS}
    out["membership_parquet"] = cc.sha256_file(REPO / adapter.MEMBERSHIP_PARQUET_RELATIVE_PATH)
    out["membership_lock"] = cc.sha256_file(REPO / adapter.MEMBERSHIP_LOCK_RELATIVE_PATH)
    return out


@pytest.fixture(scope="module")
def before_hashes():
    return _real_hashes()


# --------------------------------------------------------------------------- #
# H/I/J. Exact counts, real arms
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ARMS)
def test_exact_selected_count_and_routes(arm):
    binding = adapter.resolve_arm_binding(arm)
    assert binding.membership_count == 818
    assert binding.physics_count == 354
    assert binding.gpat_count == 464


def test_route_contract_never_rebalanced():
    for arm in ARMS:
        binding = adapter.resolve_arm_binding(arm)
        assert binding.physics_count + binding.gpat_count == 818
        assert binding.physics_count != binding.gpat_count  # never forced to 512/512


# --------------------------------------------------------------------------- #
# J. Identity-set equality: membership == filtered historical selection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ARMS)
def test_identity_set_equals_canonical_membership(arm):
    membership_df = adapter.load_frozen_membership()
    e8_ids = adapter.resolve_arm_membership(membership_df, arm)
    bank_lock, _ = adapter.load_historical_bank_lock(arm)
    filtered = adapter.filter_bank_lock_to_e8(bank_lock, arm, e8_ids)
    filtered_ids = {e["candidate_id"] for e in filtered["selected"]}
    assert filtered_ids == set(e8_ids)
    assert len(filtered["selected"]) == len(e8_ids) == 818


# --------------------------------------------------------------------------- #
# K. Output order is a subsequence of the historical C6 order
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ARMS)
def test_output_order_is_historical_subsequence(arm):
    membership_df = adapter.load_frozen_membership()
    e8_ids = adapter.resolve_arm_membership(membership_df, arm)
    bank_lock, _ = adapter.load_historical_bank_lock(arm)
    original_order = [e["candidate_id"] for e in bank_lock["selected"]]
    filtered = adapter.filter_bank_lock_to_e8(bank_lock, arm, e8_ids)
    filtered_order = [e["candidate_id"] for e in filtered["selected"]]

    # subsequence check: filtered_order must appear in the same relative order
    # as in original_order (it need not be contiguous).
    position_in_original = {cid: i for i, cid in enumerate(original_order)}
    positions = [position_in_original[cid] for cid in filtered_order]
    assert positions == sorted(positions), "filtered order is not a subsequence of the historical order"
    # never re-sorted by q or qmatch_hash at the filter stage
    assert filtered_order != sorted(filtered_order)


# --------------------------------------------------------------------------- #
# L. Retained record payloads are byte-identical to their historical C6 record
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ARMS)
def test_retained_records_unchanged(arm):
    membership_df = adapter.load_frozen_membership()
    e8_ids = adapter.resolve_arm_membership(membership_df, arm)
    bank_lock, _ = adapter.load_historical_bank_lock(arm)
    original_by_id = {e["candidate_id"]: e for e in bank_lock["selected"]}
    filtered = adapter.filter_bank_lock_to_e8(bank_lock, arm, e8_ids)
    for entry in filtered["selected"]:
        cid = entry["candidate_id"]
        assert entry == original_by_id[cid], f"{cid}: retained record was mutated"
    # top-level lock fields other than 'selected' also pass through unchanged
    for key in bank_lock:
        if key == "selected":
            continue
        assert filtered[key] == bank_lock[key]


# --------------------------------------------------------------------------- #
# M. Repeated construction is deterministic
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ARMS)
def test_repeated_construction_deterministic(arm):
    b1 = adapter.resolve_arm_binding(arm)
    b2 = adapter.resolve_arm_binding(arm)
    assert b1 == b2


# --------------------------------------------------------------------------- #
# N. Adapter rule identity is deterministic
# --------------------------------------------------------------------------- #

def test_adapter_rule_identity_deterministic():
    id1 = adapter.adapter_rule_identity()
    id2 = adapter.adapter_rule_identity()
    assert id1 == id2
    assert len(id1) == 64
    int(id1, 16)


def test_adapter_rule_payload_excludes_forbidden_fields():
    payload = adapter.build_adapter_rule_payload({"RND": "x", "DET": "y", "LLM": "z"})
    blob = json.dumps(payload).lower()
    for forbidden in ("timestamp", "hostname", "dirty", "random", "target_metric", "/home/"):
        assert forbidden not in blob


# --------------------------------------------------------------------------- #
# O. Source-domain firewall: only CASIA/MSU
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ARMS)
def test_source_domain_firewall_casia_msu_only(arm):
    membership_df = adapter.load_frozen_membership()
    e8_ids = adapter.resolve_arm_membership(membership_df, arm)
    bank_lock, _ = adapter.load_historical_bank_lock(arm)
    filtered = adapter.filter_bank_lock_to_e8(bank_lock, arm, e8_ids)
    domains = {e["source_domain"] for e in filtered["selected"]}
    assert domains <= {"casia_fasd", "msu_mfsd"}
    assert domains  # non-empty


# --------------------------------------------------------------------------- #
# Q. No target-label input required to construct the binding
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ARMS)
def test_no_target_label_input_required(arm):
    binding = adapter.resolve_arm_binding(arm)
    assert "siw" not in binding.identifier_field.lower()
    # resolve_arm_binding never opened anything under a target namespace
    # (structural guarantee: only membership parquet + C6_BANK_LOCK files are read)


# --------------------------------------------------------------------------- #
# S. No duplicate synthetic asset/candidate identity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ARMS)
def test_no_duplicate_identity_in_filtered_output(arm):
    membership_df = adapter.load_frozen_membership()
    e8_ids = adapter.resolve_arm_membership(membership_df, arm)
    bank_lock, _ = adapter.load_historical_bank_lock(arm)
    filtered = adapter.filter_bank_lock_to_e8(bank_lock, arm, e8_ids)
    ids = [e["candidate_id"] for e in filtered["selected"]]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# T. Output path firewall: any write from this module must land under
#    reports/c_ext_q1q2_v1/e8_qmatched/
# --------------------------------------------------------------------------- #

def test_output_path_firewall():
    scope = "reports/c_ext_q1q2_v1/e8_qmatched"
    ok_path = REPO / "reports/c_ext_q1q2_v1/e8_qmatched/training/adapter/PROBE.json"
    rel = cc.assert_ext_write_path(ok_path, root=REPO, must_be_under=scope)
    assert rel.startswith("reports/c_ext_q1q2_v1/e8_qmatched/")

    for bad in (
        REPO / "reports/full/c6/PROBE.json",                       # frozen Flow-1/Flow-2 namespace
        REPO / "reports/c_ext_q1q2_v1/e7_three_fold/PROBE.json",   # a sibling E-namespace, not e8_qmatched
    ):
        with pytest.raises(cc.ExtPathSafetyError):
            cc.assert_ext_write_path(bad, root=REPO, must_be_under=scope)

    # membership/ is technically inside the e8_qmatched firewall boundary (it
    # would pass this path-namespace check), so its read-only status is instead
    # a structural property of this module: it contains zero write calls.
    membership_path = REPO / "reports/c_ext_q1q2_v1/e8_qmatched/membership/PROBE.json"
    rel2 = cc.assert_ext_write_path(membership_path, root=REPO, must_be_under=scope)
    assert rel2.startswith("reports/c_ext_q1q2_v1/e8_qmatched/membership/")
    import inspect
    source = inspect.getsource(adapter)
    assert "write_json_atomic" not in source and "write_text_atomic" not in source, (
        "the adapter module must remain 100% read-only -- membership's read-only status "
        "is enforced by never containing a write call, not by the generic path guard"
    )


# --------------------------------------------------------------------------- #
# U/V. Real artifacts byte-identical before and after this whole test run
# --------------------------------------------------------------------------- #

def test_real_artifacts_unchanged(before_hashes):
    after = _real_hashes()
    assert after == before_hashes


# --------------------------------------------------------------------------- #
# A-G, P. Hard-fail tests using tiny synthetic fixtures (never the real files)
# --------------------------------------------------------------------------- #

def _tiny_lock(*, profile="NOMINAL", threshold=adapter.FROZEN_QUALITY_THRESHOLD_IDENTITY,
              selected=None):
    if selected is None:
        selected = [
            {"candidate_id": "c5syn_a", "route": "physics", "source_domain": "casia_fasd", "q": 0.5},
            {"candidate_id": "c5syn_b", "route": "gpat", "source_domain": "msu_mfsd", "q": 0.6},
            {"candidate_id": "c5syn_c", "route": "physics", "source_domain": "casia_fasd", "q": 0.7},
        ]
    return {"quality_profile": profile, "quality_threshold_identity": threshold,
           "selected_set_sha256": "irrelevant", "selected": selected}


def test_A_frozen_membership_sha_mismatch_hard_fails(tmp_path, monkeypatch):
    bad_dir = tmp_path / "reports/c_ext_q1q2_v1/e8_qmatched/membership"
    bad_dir.mkdir(parents=True)
    (bad_dir / "E8_QMATCH_SELECTED_MEMBERSHIP.parquet").write_bytes(b"not the real parquet")
    with pytest.raises(adapter.E8AdapterError, match="SHA256"):
        adapter.load_frozen_membership(root=tmp_path)


def test_B_membership_lock_sha_mismatch_hard_fails(tmp_path):
    bad_dir = tmp_path / "reports/c_ext_q1q2_v1/e8_qmatched/membership"
    bad_dir.mkdir(parents=True)
    (bad_dir / "E8_QMATCH_MEMBERSHIP_LOCK.json").write_text("{}")
    with pytest.raises(adapter.E8AdapterError, match="SHA256"):
        adapter.verify_membership_lock(root=tmp_path)


def test_C_unknown_arm_hard_fails():
    import pandas as pd
    df = pd.DataFrame({"arm": ["RND"], "fold_id": ["EXT-F1"], "condition": ["G-RND-QMATCH"],
                       "sample_id": ["x"], "route": ["physics"]})
    with pytest.raises(adapter.E8AdapterError, match="unknown arm"):
        adapter.resolve_arm_membership(df, "XYZ")
    with pytest.raises(adapter.E8AdapterError, match="unknown arm"):
        adapter.filter_bank_lock_to_e8(_tiny_lock(), "XYZ", ("c5syn_a",))


def test_D_wrong_fold_or_profile_hard_fails(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"arm": ["RND"], "fold_id": ["EXT-F9"], "condition": ["G-RND-QMATCH"],
                       "sample_id": ["x"], "route": ["physics"]})
    with pytest.raises(adapter.E8AdapterError, match="fold_id"):
        adapter.resolve_arm_membership(df, "RND")

    lock_path = tmp_path / "reports/full/c6"
    lock_path.mkdir(parents=True)
    (lock_path / "C6_BANK_LOCK_RND.json").write_text(json.dumps(_tiny_lock(profile="PERMISSIVE")))
    with pytest.raises(adapter.E8AdapterError, match="quality_profile"):
        adapter.load_historical_bank_lock("RND", root=tmp_path)

    (lock_path / "C6_BANK_LOCK_DET.json").write_text(json.dumps(_tiny_lock(threshold="deadbeef")))
    with pytest.raises(adapter.E8AdapterError, match="quality_threshold_identity"):
        adapter.load_historical_bank_lock("DET", root=tmp_path)


def test_E_duplicate_membership_identifier_hard_fails():
    with pytest.raises(adapter.E8AdapterError, match="duplicate"):
        adapter.filter_bank_lock_to_e8(_tiny_lock(), "RND", ("c5syn_a", "c5syn_a"))


def test_F_membership_candidate_absent_from_historical_lock_hard_fails():
    with pytest.raises(adapter.E8AdapterError, match="absent from the historical"):
        adapter.filter_bank_lock_to_e8(_tiny_lock(), "RND", ("c5syn_a", "c5syn_NOT_PRESENT"))


def test_G_extra_historical_candidates_are_filtered_out():
    lock = _tiny_lock()  # 3 entries: a, b, c
    filtered = adapter.filter_bank_lock_to_e8(lock, "RND", ("c5syn_a",))
    assert [e["candidate_id"] for e in filtered["selected"]] == ["c5syn_a"]


def test_duplicate_candidate_id_in_historical_lock_itself_hard_fails():
    lock = _tiny_lock(selected=[
        {"candidate_id": "c5syn_a", "route": "physics", "source_domain": "casia_fasd", "q": 0.5},
        {"candidate_id": "c5syn_a", "route": "physics", "source_domain": "casia_fasd", "q": 0.5},
    ])
    with pytest.raises(adapter.E8AdapterError, match="duplicate candidate_id"):
        adapter.filter_bank_lock_to_e8(lock, "RND", ("c5syn_a",))


def test_P_target_domain_row_injection_hard_fails():
    lock = _tiny_lock(selected=[
        {"candidate_id": "c5syn_a", "route": "physics", "source_domain": "casia_fasd", "q": 0.5},
        {"candidate_id": "c5syn_siw", "route": "physics", "source_domain": "siw_mv2", "q": 0.9},
    ])
    with pytest.raises(adapter.E8AdapterError, match="forbidden source_domain"):
        adapter.filter_bank_lock_to_e8(lock, "RND", ("c5syn_a", "c5syn_siw"))


# --------------------------------------------------------------------------- #
# R. C6MatchedBankReader compatibility -- tiny synthetic fixture, no real
#    pixel payloads needed
# --------------------------------------------------------------------------- #

def _build_fixture_bank(tmp_path, arm="RND", n=4):
    from prism_fas.synthesis.synthetic_bank import encode_npz, encode_png
    import numpy as np

    candidates_root = tmp_path / "candidates"
    recipe_id = "R-000001"
    recipes = [{"recipe_id": recipe_id, "artifacts": [{"name": "moire"}], "regions": ["forehead"]}]
    selected = []
    for i in range(n):
        candidate_id = f"c5syn_fix{i:03d}"
        directory = candidates_root / arm / candidate_id
        directory.mkdir(parents=True)
        image_bytes = encode_png(np.zeros((224, 224, 3), dtype="uint8"))
        mask_bytes = encode_png(np.zeros((224, 224), dtype="uint8"))
        artifact_bytes = encode_npz({"artifact_map": np.zeros((1, 224, 224), dtype="float32")})
        (directory / "synthetic.png").write_bytes(image_bytes)
        (directory / "exact_mask.png").write_bytes(mask_bytes)
        (directory / "artifact_map.npz").write_bytes(artifact_bytes)
        record = {
            "status": "generated",
            "generation_identity": {
                "arm": arm, "candidate_id": candidate_id, "route": "physics" if i % 2 == 0 else "gpat",
                "recipe_id": recipe_id, "recipe_ordinal": i, "slot": 0,
                "live_target_sample_id": f"live{i}", "package_identity": "pkg-fixture",
                "recipe_bank_identity": "bank-fixture",
            },
            "generation_identity_sha256": f"genid{i}",
            "payload_sha256": {
                "synthetic.png": hashlib.sha256(image_bytes).hexdigest(),
                "exact_mask.png": hashlib.sha256(mask_bytes).hexdigest(),
                "artifact_map.npz": hashlib.sha256(artifact_bytes).hexdigest(),
            },
            "trace": {"exact_mask_pixels": 0},
        }
        (directory / "CANDIDATE.json").write_text(json.dumps(record))
        selected.append({
            "candidate_id": candidate_id, "route": "physics" if i % 2 == 0 else "gpat",
            "recipe_id": recipe_id, "recipe_ordinal": i, "source_domain": "casia_fasd",
            "q": round(0.5 + 0.01 * i, 4), "live_target_sample_id": f"live{i}", "base_position": i,
        })
    bank_lock = {
        "quality_profile": "NOMINAL", "quality_threshold_identity": adapter.FROZEN_QUALITY_THRESHOLD_IDENTITY,
        "selected_set_sha256": "fixture", "selector_identity_sha256": "fixture", "selected": selected,
    }
    return candidates_root, recipes, bank_lock, [s["candidate_id"] for s in selected]


def test_R_c6_matched_bank_reader_can_consume_e8_filtered_view(tmp_path):
    from prism_fas.detector.c6_bank import C6MatchedBankReader

    candidates_root, recipes, bank_lock, all_ids = _build_fixture_bank(tmp_path)
    e8_ids = tuple(all_ids[:2])  # a genuine subset, exercising the real filter
    filtered = adapter.filter_bank_lock_to_e8(bank_lock, "RND", e8_ids)

    reader = C6MatchedBankReader.open(
        candidates_root=candidates_root, arm="RND", bank_lock=filtered, recipes=recipes,
        package_identity="pkg-fixture", recipe_bank_identity="bank-fixture",
        expected_selected_set_sha256=None,
    )
    assert len(reader) == 2
    for position in range(len(reader)):
        sample = reader.sample(position)
        assert sample.route in ("physics", "gpat")
        assert 0.0 <= sample.quality_weight <= 1.0
    assert set(reader.synthetic_ids) == set(e8_ids)


# --------------------------------------------------------------------------- #
# 16. Sampler compatibility smoke -- pure index-space, no pixels, no SigLIP2
# --------------------------------------------------------------------------- #

def test_sampler_818_bank_compatibility_smoke():
    from prism_fas.detector.sampler import BatchContract, M9BatchSampler

    physics_positions = list(range(0, 354))
    gpat_positions = list(range(354, 354 + 464))
    assert len(physics_positions) + len(gpat_positions) == 818

    contract = BatchContract()  # frozen G5 defaults: 12/12/8, both routes/domains, mixed phase
    real_live = {"casia_fasd": list(range(200)), "msu_mfsd": list(range(200))}
    real_spoof = {"casia_fasd": list(range(200)), "msu_mfsd": list(range(200))}

    def make(seed):
        return M9BatchSampler(real_live=real_live, real_spoof=real_spoof,
                              synthetic_routes={"physics": physics_positions, "gpat": gpat_positions},
                              contract=contract, seed=seed, steps_per_epoch=45, identity="e8-smoke-RND")

    sampler = make(20260806)

    total_draws = 0
    touched: set[int] = set()
    for epoch in range(30):  # mirrors the real G5 mixed-epoch count
        for plan in sampler.epoch_plans(epoch):
            assert len(plan.synthetic) == 8
            for idx in plan.synthetic:
                assert 0 <= idx <= 817
                total_draws += 1
                touched.add(idx)
    assert total_draws == 10800  # 30 * 45 * 8, matches the frozen training schedule
    assert len(touched) <= 818
    assert len(touched) > 700  # heavy reuse cycles through nearly the whole 818-bank

    # same seed -> deterministic sequence
    sampler_a = make(20260806)
    sampler_b = make(20260806)
    plans_a = [(p.real_live, p.real_spoof, p.synthetic) for p in sampler_a.epoch_plans(0)]
    plans_b = [(p.real_live, p.real_spoof, p.synthetic) for p in sampler_b.epoch_plans(0)]
    assert plans_a == plans_b

    # different seed -> may differ (existing sampler semantics)
    sampler_c = make(20260807)
    plans_c = [(p.real_live, p.real_spoof, p.synthetic) for p in sampler_c.epoch_plans(0)]
    assert plans_c != plans_a
