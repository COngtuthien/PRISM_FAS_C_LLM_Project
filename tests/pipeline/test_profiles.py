"""Profile semantics (v1.5 Appendix L.2, L.12).

These tests exist because the profile config is the one file an impatient future
session would be most tempted to edit — making validate "just this once"
eligible, or giving full a small budget so a run finishes. Each edit is tried
here and must be refused at load time.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prism_fas.pipeline.profiles import PROFILE_NAMES, ProfileError, load_profile


def test_all_three_profiles_load(repo: Path) -> None:
    for name in PROFILE_NAMES:
        assert load_profile(name, repo=repo).name == name


def test_only_full_is_scientifically_eligible(repo: Path) -> None:
    """L.2's invariant, stated so adding a profile cannot weaken it.

    Written against the profile LIST rather than a hard-coded set: the point is
    that exactly one profile is eligible however many exist, and a test that
    enumerated three would have to be edited — and could be edited wrongly — every
    time a non-eligible profile is added.
    """
    eligible = {name: load_profile(name, repo=repo).scientific_eligible
                for name in PROFILE_NAMES}
    assert [name for name, value in eligible.items() if value] == ["full"]
    assert set(eligible) == set(PROFILE_NAMES)
    assert len(PROFILE_NAMES) >= 4, "validate, smoke, rehearsal and full"


def test_only_full_may_select_a_winner(repo: Path) -> None:
    for name in PROFILE_NAMES:
        profile = load_profile(name, repo=repo)
        assert profile.may_select_scientific_winner is (name == "full")


def test_full_declares_the_frozen_selector_as_its_only_winner_path(repo: Path) -> None:
    rule = load_profile("full", repo=repo).winner_selection_rule
    assert rule and "frozen source-only selector" in rule
    assert "before source freeze" in rule


def test_namespaces_are_disjoint_per_profile(repo: Path) -> None:
    """Every profile writes somewhere no other profile writes.

    This is what keeps a rehearsal from being mistaken for — or resumed as —
    scientific evidence, so it is asserted as disjointness rather than as a fixed
    list of three names.
    """
    reports = [load_profile(name, repo=repo).reports_namespace for name in PROFILE_NAMES]
    assert len(reports) == len(set(reports)), f"namespaces collide: {reports}"
    for name in PROFILE_NAMES:
        assert load_profile(name, repo=repo).reports_namespace == f"reports/{name}"
    assert "reports/rehearsal" in reports


def test_validate_claims_no_run_tree(repo: Path) -> None:
    assert load_profile("validate", repo=repo).runs_namespace is None


def test_non_eligible_profiles_forbid_provider_gpu_and_labels(repo: Path) -> None:
    for name in ("validate", "smoke", "rehearsal"):
        policy = load_profile(name, repo=repo).compute_policy
        assert policy.forbids_live_provider
        assert policy.forbids_target_labels
        assert policy.scientific_training is False
        assert policy.gpu in {"none", "fixture_only"}


def test_full_gates_the_provider_on_a_lock_and_limits_it_to_c3(repo: Path) -> None:
    policy = load_profile("full", repo=repo).compute_policy
    assert policy.live_provider == "gated"
    assert policy.live_provider_permitted_for_stages == ("C3",)
    assert policy.raw["live_provider_requires_approved_bank_lock"] is True


def test_full_confines_target_labels_to_the_c12_scorer(repo: Path) -> None:
    policy = load_profile("full", repo=repo).compute_policy
    assert policy.target_label_access == "c12_scorer_only"
    assert policy.target_metric_access == "c12_scorer_only"


def test_full_carries_no_engineering_budget(repo: Path) -> None:
    """L.12: a scientific budget is never shrunk because credit is low."""
    profile = load_profile("full", repo=repo)
    assert profile.engineering_budget is None
    assert profile.raw["silent_scientific_shrink_permitted"] is False


def test_smoke_may_reduce_only_the_four_authorized_dimensions(repo: Path) -> None:
    profile = load_profile("smoke", repo=repo)
    assert profile.reduction_permitted
    assert set(profile.raw["reducible_dimensions"]) == {"samples", "steps", "epochs", "seeds"}
    assert set(profile.preserved_under_reduction) == {
        "code_paths", "tensor_semantics", "model_topology", "active_loss_semantics",
        "target_firewall", "artifact_schemas"}


def test_profile_identity_tracks_the_config_bytes(repo: Path, tmp_path: Path) -> None:
    original = load_profile("validate", repo=repo)
    assert len(original.profile_identity) == 64

    mirror = tmp_path / "configs" / "execution"
    mirror.mkdir(parents=True)
    source = repo / "configs" / "execution" / "validate.yaml"
    (mirror / "validate.yaml").write_bytes(source.read_bytes() + b"\n# a comment\n")
    changed = load_profile("validate", repo=tmp_path)
    assert changed.profile_identity != original.profile_identity


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    target = tmp_path / "configs" / "execution"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{name}.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    return tmp_path


def _payload(repo: Path, name: str) -> dict:
    return yaml.safe_load((repo / "configs" / "execution" / f"{name}.yaml").read_text(
        encoding="utf-8"))


def test_validate_cannot_be_made_eligible_by_editing_its_config(
        repo: Path, tmp_path: Path) -> None:
    payload = _payload(repo, "validate")
    payload["scientific_eligible"] = True
    _write(tmp_path, "validate", payload)
    with pytest.raises(ProfileError, match="L.2 makes exactly 'full' eligible"):
        load_profile("validate", repo=tmp_path)


def test_smoke_cannot_be_granted_winner_selection(repo: Path, tmp_path: Path) -> None:
    payload = _payload(repo, "smoke")
    payload["may_select_scientific_winner"] = True
    _write(tmp_path, "smoke", payload)
    with pytest.raises(ProfileError, match="winner selection only under 'full'"):
        load_profile("smoke", repo=tmp_path)


def test_full_cannot_be_given_an_engineering_budget(repo: Path, tmp_path: Path) -> None:
    payload = _payload(repo, "full")
    payload["engineering_budget"] = {"max_epochs": 1}
    _write(tmp_path, "full", payload)
    with pytest.raises(ProfileError, match="L.12 forbids shrinking a scientific budget"):
        load_profile("full", repo=tmp_path)


def test_smoke_cannot_be_granted_a_live_provider(repo: Path, tmp_path: Path) -> None:
    payload = _payload(repo, "smoke")
    payload["compute_policy"]["live_provider"] = True
    _write(tmp_path, "smoke", payload)
    with pytest.raises(ProfileError, match="live provider under non-eligible profile"):
        load_profile("smoke", repo=tmp_path)


def test_smoke_cannot_be_granted_target_labels(repo: Path, tmp_path: Path) -> None:
    payload = _payload(repo, "smoke")
    payload["compute_policy"]["target_label_access"] = "everything"
    _write(tmp_path, "smoke", payload)
    with pytest.raises(ProfileError, match="target label access under non-eligible profile"):
        load_profile("smoke", repo=tmp_path)


def test_a_profile_cannot_borrow_another_profiles_namespace(
        repo: Path, tmp_path: Path) -> None:
    payload = _payload(repo, "smoke")
    payload["reports_namespace"] = "reports/full"
    _write(tmp_path, "smoke", payload)
    with pytest.raises(ProfileError, match="L.2 gives profile 'smoke'"):
        load_profile("smoke", repo=tmp_path)


def test_a_profile_cannot_share_a_run_tree(repo: Path, tmp_path: Path) -> None:
    payload = _payload(repo, "smoke")
    payload["runs_namespace"] = "runs/full"
    _write(tmp_path, "smoke", payload)
    with pytest.raises(ProfileError, match="share a run tree"):
        load_profile("smoke", repo=tmp_path)


def test_unknown_profile_is_refused(repo: Path) -> None:
    with pytest.raises(ProfileError, match="unknown execution profile"):
        load_profile("production", repo=repo)


def test_stamp_carries_both_mandatory_l2_fields(repo: Path) -> None:
    for name in PROFILE_NAMES:
        stamp = load_profile(name, repo=repo).stamp()
        assert stamp["execution_profile"] == name
        assert "scientific_eligible" in stamp
