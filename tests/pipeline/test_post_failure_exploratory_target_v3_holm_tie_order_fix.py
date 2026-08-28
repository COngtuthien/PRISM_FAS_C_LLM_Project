"""POST_FAILURE_EXPLORATORY_TARGET_V3 — TECHNICAL_FINAL_VALIDATOR_TIE_ORDER_DEFECT.

Real E2 scientific scoring completed on the GPU: 24/24 score rows, 7/7
exploratory comparisons, a real `TARGET_LABEL_REVEAL.json`, a real
`EXPLORATORY_TARGET_SCORE_RESULT.json`. Final validation failed ONLY with
"Holm-Bonferroni recomputed from the RECORDED randomization p-values does
not match the stored correction" — a validator defect, not a scoring
defect: all seven recorded randomization p-values are tied at exactly
`9.999000099990002e-05`. `EXPLORATORY_TARGET_SCORE_RESULT.json` is written
via `atomic_write_json(..., sort_keys=True)`, so `comparisons.items()`
iterates ALPHABETICALLY when read back — never the frozen hypothesis order
(`REQUIRED_MATCHED_SEEDS`'s own definition order:
`E-H1_RND_vs_DET, E-H1_RND_vs_LLM, E-H1_DET_vs_LLM, E-H2, E-H3, E-H4_DET,
E-H4_LLM`) the scientific scoring execution actually used.
`holm_bonferroni`'s `sorted(..., key=...)` is stable, so tied p-values keep
whatever order they were inserted in — reconstructing `recorded_p_values`
from `comparisons.items()` (alphabetical) therefore recomputes a DIFFERENT
rank/adjusted_alpha assignment than the frozen run actually produced, even
though not one p-value changed.

This file proves the corrected `validate_existing_exploratory_score_result_v3`
reconstructs `recorded_p_values` in `REQUIRED_MATCHED_SEEDS` order — only
after confirming the comparison set is exactly the frozen seven — and that
`holm_bonferroni` itself, and every other statistical/scoring function, is
untouched.

**FIXTURE / ENGINEERING ONLY.** No real GPU score result, prediction, or
label is read anywhere in this file — every fixture is a small, synthetic
JSON structure in `tmp_path`. No test calls `--predict`, `--score`, or any
label-build/reveal path. `holm_bonferroni` is called directly as the same,
completely unmodified, frozen statistical function the real scoring
execution used — never redefined or reimplemented here.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.evaluation import post_failure_exploratory_target_v3_scorer as v3s  # noqa: E402
from prism_fas.pipeline.state import atomic_write_json  # noqa: E402

# The exact tied p-value observed in the real GPU scoring result.
TIED_P_VALUE = 9.999000099990002e-05


def _comparisons_with_p_values(p_values_in_frozen_order: dict[str, float]) -> dict[str, Any]:
    """Built by iterating `REQUIRED_MATCHED_SEEDS` — the SAME frozen order
    `compute_exploratory_comparisons_v3` builds its own `comparisons` dict
    in. `atomic_write_json`'s `sort_keys=True` + a JSON round-trip is what
    turns this into alphabetical order, exactly reproducing the historical
    defect's exact mechanism."""
    comparisons: dict[str, Any] = {}
    for name in v3s.REQUIRED_MATCHED_SEEDS:
        comparisons[name] = {"matched_seeds": sorted(v3s.REQUIRED_MATCHED_SEEDS[name]),
                             "randomization": {"p_value_two_sided": p_values_in_frozen_order[name]},
                             "bootstrap_ci": {"used_for": "CI_ONLY"}}
    return comparisons


def _tied_fixture(p_value: float = TIED_P_VALUE) -> tuple[dict[str, Any], dict[str, Any]]:
    p_values = {name: p_value for name in v3s.REQUIRED_MATCHED_SEEDS}
    comparisons = _comparisons_with_p_values(p_values)
    # The "stored" Holm result — exactly what the real, frozen scoring
    # execution would have produced and written, since it builds p_values
    # by iterating its own comparisons dict, itself built in frozen order.
    stored_holm = v3s.holm_bonferroni(p_values)
    return comparisons, stored_holm


def _write_result(tmp_path: Path, *, comparisons: dict[str, Any], holm: dict[str, Any]) -> Path:
    """A deliberately MINIMAL EXPLORATORY_TARGET_SCORE_RESULT.json — only
    `exploratory_comparisons` is realistic. Every other field the validator
    checks is left absent; those checks correctly append their own
    (ignored, unrelated) problems, isolating these tests to ONLY the
    Holm-recomputation logic this fix changes."""
    result = {"exploratory_comparisons": {"comparisons": comparisons, "holm_bonferroni": holm}}
    path = tmp_path / v3s.SCORE_RESULT_PATH
    atomic_write_json(path, result)
    return path


def _install_minimal_validator_fixtures(monkeypatch) -> None:
    """An empty-entries fake lockset makes every row/score-file check
    trivially satisfied (both sides empty), leaving only the
    exploratory_comparisons/Holm section under test."""
    fake_lockset = {"entries": {}, "lockset_identity": "l" * 64,
                    "prediction_execution_code_commit": "c" * 40,
                    "target_feature_package_identity": "e" * 64}
    monkeypatch.setattr(v3s, "require_frozen_prediction_lockset", lambda repo: fake_lockset)


def _holm_problems(problems: list[str]) -> list[str]:
    return [p for p in problems if "Holm-Bonferroni" in p]


# ==============================================================================
# A. Seven identical p-values, frozen order, produce a stored Holm object
# ==============================================================================

def test_seven_identical_p_values_in_frozen_order_produce_a_stored_holm_object() -> None:
    _, stored_holm = _tied_fixture()
    frozen_order = list(v3s.REQUIRED_MATCHED_SEEDS)
    assert set(stored_holm) == set(v3s.REQUIRED_MATCHED_SEEDS)
    # All seven p-values are tied: holm_bonferroni's stable sort keeps the
    # FROZEN insertion order, so ranks 1..7 land exactly on that order.
    assert [stored_holm[name]["rank"] for name in frozen_order] == list(range(1, 8))
    assert all(isinstance(entry["significant"], bool) for entry in stored_holm.values())


# ==============================================================================
# B. JSON sort_keys=True round-trip still validates successfully
# ==============================================================================

def test_json_roundtrip_with_sort_keys_still_validates_successfully(monkeypatch, tmp_path) -> None:
    comparisons, stored_holm = _tied_fixture()
    path = _write_result(tmp_path, comparisons=comparisons, holm=stored_holm)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    on_disk_comparisons = on_disk["exploratory_comparisons"]["comparisons"]
    assert list(on_disk_comparisons) == sorted(on_disk_comparisons)   # confirms alphabetical JSON order

    _install_minimal_validator_fixtures(monkeypatch)
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert _holm_problems(validation["problems"]) == []


# ==============================================================================
# C. The historical mismatch is real: naive alphabetical reconstruction
#    reproduces it, proving this regression test genuinely exercises the bug
# ==============================================================================

def test_naive_alphabetical_reconstruction_reproduces_the_historical_mismatch(tmp_path) -> None:
    comparisons, stored_holm = _tied_fixture()
    path = _write_result(tmp_path, comparisons=comparisons, holm=stored_holm)

    on_disk_comparisons = json.loads(path.read_text(encoding="utf-8"))["exploratory_comparisons"]["comparisons"]
    assert list(on_disk_comparisons) == sorted(on_disk_comparisons)

    # The OLD, buggy reconstruction: comparisons.items() in JSON (alphabetical) order.
    naive_p_values = {name: entry["randomization"]["p_value_two_sided"]
                      for name, entry in on_disk_comparisons.items()}
    naive_holm = v3s.holm_bonferroni(naive_p_values)
    assert naive_holm != stored_holm


# ==============================================================================
# D. Validator recomputation using REQUIRED_MATCHED_SEEDS order exactly
#    matches the stored Holm output
# ==============================================================================

def test_frozen_order_reconstruction_exactly_matches_stored_holm(tmp_path) -> None:
    comparisons, stored_holm = _tied_fixture()
    path = _write_result(tmp_path, comparisons=comparisons, holm=stored_holm)

    on_disk_comparisons = json.loads(path.read_text(encoding="utf-8"))["exploratory_comparisons"]["comparisons"]
    # The FIXED reconstruction: REQUIRED_MATCHED_SEEDS order, not comparisons.items().
    recorded_p_values = {name: on_disk_comparisons[name]["randomization"]["p_value_two_sided"]
                         for name in v3s.REQUIRED_MATCHED_SEEDS}
    recomputed_holm = v3s.holm_bonferroni(recorded_p_values)
    assert recomputed_holm == stored_holm


def test_validator_no_longer_reports_a_holm_mismatch_for_tied_p_values(monkeypatch, tmp_path) -> None:
    comparisons, stored_holm = _tied_fixture()
    _write_result(tmp_path, comparisons=comparisons, holm=stored_holm)
    _install_minimal_validator_fixtures(monkeypatch)
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert _holm_problems(validation["problems"]) == []


# ==============================================================================
# E. Non-tied p-values continue validating identically
# ==============================================================================

def test_non_tied_p_values_validate_identically(monkeypatch, tmp_path) -> None:
    p_values = {name: 0.001 * (index + 1) for index, name in enumerate(v3s.REQUIRED_MATCHED_SEEDS)}
    comparisons = _comparisons_with_p_values(p_values)
    stored_holm = v3s.holm_bonferroni(p_values)
    _write_result(tmp_path, comparisons=comparisons, holm=stored_holm)

    _install_minimal_validator_fixtures(monkeypatch)
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert _holm_problems(validation["problems"]) == []


# ==============================================================================
# F/G. Missing or extra comparison names still fail closed (no crash)
# ==============================================================================

def test_missing_one_of_the_seven_comparisons_fails_closed(monkeypatch, tmp_path) -> None:
    comparisons, stored_holm = _tied_fixture()
    del comparisons["E-H3"]
    _write_result(tmp_path, comparisons=comparisons, holm=stored_holm)

    _install_minimal_validator_fixtures(monkeypatch)
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert validation["valid"] is False
    assert _holm_problems(validation["problems"]) == []   # the Holm block is skipped, not crashed


def test_extra_comparison_name_fails_closed(monkeypatch, tmp_path) -> None:
    comparisons, stored_holm = _tied_fixture()
    comparisons["BOGUS_EXTRA_HYPOTHESIS"] = {"matched_seeds": [],
                                             "randomization": {"p_value_two_sided": 0.5},
                                             "bootstrap_ci": {}}
    _write_result(tmp_path, comparisons=comparisons, holm=stored_holm)

    _install_minimal_validator_fixtures(monkeypatch)
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert validation["valid"] is False
    assert _holm_problems(validation["problems"]) == []   # the Holm block is skipped, not crashed


# ==============================================================================
# H. A genuinely altered p-value still causes validation failure
# ==============================================================================

def test_a_genuinely_altered_p_value_still_causes_validation_failure(monkeypatch, tmp_path) -> None:
    comparisons, stored_holm = _tied_fixture()
    comparisons["E-H3"]["randomization"]["p_value_two_sided"] = 0.5   # stored_holm NOT updated to match
    _write_result(tmp_path, comparisons=comparisons, holm=stored_holm)

    _install_minimal_validator_fixtures(monkeypatch)
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert len(_holm_problems(validation["problems"])) == 1


# ==============================================================================
# I. A genuinely altered stored Holm field still causes validation failure
# ==============================================================================

def test_a_genuinely_altered_stored_holm_field_still_causes_validation_failure(monkeypatch, tmp_path) -> None:
    comparisons, stored_holm = _tied_fixture()
    tampered_holm = {**stored_holm,
                     "E-H3": {**stored_holm["E-H3"], "significant": not stored_holm["E-H3"]["significant"]}}
    _write_result(tmp_path, comparisons=comparisons, holm=tampered_holm)

    _install_minimal_validator_fixtures(monkeypatch)
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert len(_holm_problems(validation["problems"])) == 1


def test_a_genuinely_altered_rank_field_still_causes_validation_failure(monkeypatch, tmp_path) -> None:
    comparisons, stored_holm = _tied_fixture()
    name = next(iter(stored_holm))
    tampered_holm = {**stored_holm, name: {**stored_holm[name], "rank": stored_holm[name]["rank"] + 1}}
    _write_result(tmp_path, comparisons=comparisons, holm=tampered_holm)

    _install_minimal_validator_fixtures(monkeypatch)
    validation = v3s.validate_existing_exploratory_score_result_v3(tmp_path)
    assert len(_holm_problems(validation["problems"])) == 1


# ==============================================================================
# J. Validation never rewrites the result artifact
# ==============================================================================

def test_validation_never_rewrites_the_result_artifact(monkeypatch, tmp_path) -> None:
    comparisons, stored_holm = _tied_fixture()
    path = _write_result(tmp_path, comparisons=comparisons, holm=stored_holm)
    before = path.read_bytes()

    _install_minimal_validator_fixtures(monkeypatch)
    v3s.validate_existing_exploratory_score_result_v3(tmp_path)

    after = path.read_bytes()
    assert before == after


# ==============================================================================
# Scorer/statistical functions and required-order source are unchanged
# ==============================================================================

def test_holm_bonferroni_source_is_unmodified_by_this_fix() -> None:
    source = inspect.getsource(v3s.holm_bonferroni)
    assert "sorted(p_values.items(), key=lambda item: item[1])" in source


def test_validator_source_reconstructs_from_required_matched_seeds_not_comparisons_items() -> None:
    source = inspect.getsource(v3s.validate_existing_exploratory_score_result_v3)
    assert "for name in REQUIRED_MATCHED_SEEDS" in source
    assert "for name, entry in comparisons.items()" not in source


def test_v1_v2_and_frozen_configs_untouched() -> None:
    import subprocess

    result = subprocess.run(["git", "diff", "--stat", "HEAD", "--",
                            "configs/evaluation/post_failure_exploratory_target_v1.yaml",
                            "configs/evaluation/post_failure_exploratory_target_v2.yaml",
                            "configs/evaluation/post_failure_exploratory_target_v3.yaml",
                            "src/prism_fas/evaluation/post_failure_exploratory_target.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_scorer.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v2.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v2_scorer.py",
                            "src/prism_fas/evaluation/post_failure_exploratory_target_v3.py"],
                            cwd=str(REPO), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
