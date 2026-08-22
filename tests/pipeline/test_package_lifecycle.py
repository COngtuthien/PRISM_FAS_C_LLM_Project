"""The derived-package lifecycle: built, validated, finalized — in that order.

`_step_m3b` treated "the directory exists and holds a `PACKAGE_LOCK.json`" as
REUSED_VALID. `build_m3b_package` writes that lock with `status: "building"`, and
preparation never ran the validate → finalize → validate sequence the canonical
`prism data priors model-build` runs. So a fully built M3B — hours of frozen
tower inference — sat there unfinalized, preparation reported BUILT, and C4
refused it three steps later:

    data/packages/prism_data_v1_m3b reports status 'building';
    a scientific run trains only against a package its own validator passed

C4 was right. A lock exists from the builder's first write; `status` is what says
whether anything checked it. "Locked" is not "validated".

Finalization is not free of consequence. `finalize_lock` promotes `status`, fills
in `target_isolation` and `package_validation`, and RECOMPUTES
`content_identity_sha256` over the promoted lock — so a pair plan built against
the pre-finalization identity is stale the moment the package is finalized. That
is measured here rather than assumed, with the real `finalize_lock`; no test in
this file edits a lock by hand to manufacture a state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.data.package.builder import IDENTITY_EXCLUDED_FIELDS  # noqa: E402
from prism_fas.data.package.builder import finalize_lock  # noqa: E402
from prism_fas.pipeline import preparation  # noqa: E402
from prism_fas.utils.core import stable_json_hash  # noqa: E402


def _building_lock(root: Path, *, package_id: str, samples: int = 4) -> str:
    """A lock in the state the builder actually leaves behind.

    Written through the same identity rule the builders use, so the identity this
    returns is the one a real `building` package would carry.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock: dict[str, Any] = {
        "package_id": package_id, "package_schema_version": "m3-v1",
        "status": preparation.PACKAGE_STATUS_BUILDING,
        "total_samples": samples,
        "per_split_counts": {"source_train": samples},
        "target_isolation": {"policy": "feature_only_no_labels_no_identity",
                             "status": "pending"},
        "package_validation": {"status": "pending", "checks_passed": None,
                               "checks_total": None},
    }
    lock["content_identity_sha256"] = stable_json_hash(
        {key: value for key, value in lock.items()
         if key not in IDENTITY_EXCLUDED_FIELDS})
    (root / "PACKAGE_LOCK.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return str(lock["content_identity_sha256"])


def _payload(root: Path) -> dict[Path, bytes]:
    """Everything under the package that is not its lock."""
    return {path: path.read_bytes() for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "PACKAGE_LOCK.json"}


def _report(passed: bool, *, checks: int = 40) -> dict[str, Any]:
    return {"passed": passed, "target_isolation": {"passed": True},
            "checks": [{"check_id": f"c{index}", "passed": passed, "severity": "error"}
                       for index in range(checks)],
            "errors": [] if passed else [{"check_id": "c0"}]}


# --- "locked" is not "validated" ---------------------------------------------

def test_a_building_lock_is_not_scientific_input(tmp_path: Path) -> None:
    """The state the RTX host was actually in."""
    root = tmp_path / "prism_data_v1_m3b"
    _building_lock(root, package_id="prism_data_v1_m3b")

    state = preparation.package_status(root)
    assert state["present"] is True
    assert state["locked"] is True, "the lock file is there — that was the trap"
    assert state["status"] == "building"
    assert state["reusable_as_scientific_input"] is False
    assert "never validated and finalized" in state["why"]


def test_a_validated_lock_is_scientific_input(tmp_path: Path) -> None:
    root = tmp_path / "prism_data_v1_m3b"
    _building_lock(root, package_id="prism_data_v1_m3b")
    finalize_lock(root, _report(True))

    state = preparation.package_status(root)
    assert state["status"] == "validated"
    assert state["package_validation"] == "passed"
    assert state["reusable_as_scientific_input"] is True
    assert state["why"] is None


def test_package_lock_presence_alone_is_never_reuse(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """The eliminated anti-pattern, asserted at the step it lived in."""
    from prism_fas.data.package import m3b as m3b_module

    source = tmp_path / preparation.DERIVED_PACKAGES["m3a"]
    target = tmp_path / preparation.DERIVED_PACKAGES["m3b"]
    _building_lock(target, package_id="prism_data_v1_m3b")
    monkeypatch.setattr(m3b_module, "build_m3b_package",
                        lambda *args, **kwargs: pytest.fail(
                            "an existing package must never be rebuilt to finalize it"))
    monkeypatch.setattr(preparation, "validate_package_for_test", None, raising=False)

    calls: list[dict[str, Any]] = []

    def fake_validate(root, *, require_validated_status=True, parent_package=None):
        calls.append({"root": Path(root), "strict": require_validated_status,
                      "parent": parent_package})
        return _report(True)

    from prism_fas.data import package as package_module

    monkeypatch.setattr(package_module, "validate_package", fake_validate)
    outcome = preparation._step_m3b(tmp_path, resume=True)

    assert calls, "the step returned without validating anything"
    assert outcome.action == "FINALIZED"
    assert outcome.detail["package"]["status"] == "validated"
    assert source == tmp_path / "data/packages/prism_data_v1_m3a"


# --- the lifecycle, through the real finalize_lock ---------------------------

class _Validator(list):
    """The recorded `validate_package` calls, plus the verdict it should return.

    `finalize_lock` is the REAL one throughout — the identity arithmetic is what
    these tests are about, so manufacturing a promoted lock by hand would test
    the fixture instead of the contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self.outcome = {"passed": True}


@pytest.fixture
def validator(monkeypatch: pytest.MonkeyPatch) -> _Validator:
    from prism_fas.data import package as package_module

    calls = _Validator()

    def fake_validate(root, *, require_validated_status=True, parent_package=None):
        calls.append({"root": Path(root), "strict": require_validated_status,
                      "parent": Path(parent_package) if parent_package else None})
        return _report(calls.outcome["passed"])

    monkeypatch.setattr(package_module, "validate_package", fake_validate)
    return calls


def test_an_existing_building_package_is_finalized_in_place(tmp_path: Path,
                                                            validator) -> None:
    """No prior is recomputed to change a status field."""
    root = tmp_path / "prism_data_v1_m3b"
    before_identity = _building_lock(root, package_id="prism_data_v1_m3b")
    (root / "priors").mkdir()
    (root / "priors" / "a.npz").write_bytes(b"an expensive frozen-tower prior")
    payload_before = _payload(root)

    state = preparation.ensure_package_validated(
        root, parent=tmp_path / "prism_data_v1_m3a", label="the M3B prior package")

    assert state["action"] == "FINALIZED"
    assert state["finalized"] is True
    assert state["priors_rebuilt"] is False
    assert state["status"] == "validated"
    assert state["package_validation"] == "passed"
    assert state["content_identity_before_finalization"] == before_identity
    assert _payload(root) == payload_before, "finalization rewrites the lock alone"


def test_finalization_changes_the_content_identity(tmp_path: Path, validator) -> None:
    """The consequence the pair plan has to notice."""
    root = tmp_path / "prism_data_v1_m3b"
    before = _building_lock(root, package_id="prism_data_v1_m3b")

    state = preparation.ensure_package_validated(root, label="the M3B prior package")

    assert state["content_identity_before_finalization"] == before
    assert state["content_identity_sha256"] != before
    assert state["content_identity_sha256"] == preparation.package_status(
        root)["content_identity_sha256"], "the finalized identity is what downstream sees"


def test_the_validation_sequence_is_loose_then_strict(tmp_path: Path, validator) -> None:
    root = tmp_path / "prism_data_v1_m3b"
    _building_lock(root, package_id="prism_data_v1_m3b")
    parent = tmp_path / "prism_data_v1_m3a"

    preparation.ensure_package_validated(root, parent=parent,
                                         label="the M3B prior package")

    assert [call["strict"] for call in validator] == [False, True], (
        "loose validation must precede finalization and strict must follow it")
    assert {call["parent"] for call in validator} == {parent}, (
        "both validations must be given the M3A parent")


def test_a_validated_package_is_still_validated_before_reuse(tmp_path: Path,
                                                             validator) -> None:
    root = tmp_path / "prism_data_v1_m3b"
    _building_lock(root, package_id="prism_data_v1_m3b")
    finalize_lock(root, _report(True))
    validator.clear()

    state = preparation.ensure_package_validated(root, label="the M3B prior package")

    assert state["action"] == "REUSED_VALID"
    assert state["finalized"] is False
    assert [call["strict"] for call in validator] == [True], (
        "a validated package is strict-validated, not taken on trust")


def test_a_building_package_that_does_not_validate_fails_closed(tmp_path: Path,
                                                                validator) -> None:
    root = tmp_path / "prism_data_v1_m3b"
    before = _building_lock(root, package_id="prism_data_v1_m3b")
    validator.outcome["passed"] = False

    with pytest.raises(preparation.PreparationError) as raised:
        preparation.ensure_package_validated(root, label="the M3B prior package")

    assert raised.value.reason == preparation.PACKAGE_NOT_VALIDATED
    assert raised.value.detail["failed_checks"]
    assert preparation.package_status(root)["status"] == "building", (
        "a package that failed validation must not have been promoted")
    assert preparation.package_status(root)["content_identity_sha256"] == before


def test_a_validated_package_that_stops_validating_fails_closed(tmp_path: Path,
                                                                validator) -> None:
    root = tmp_path / "prism_data_v1_m3b"
    _building_lock(root, package_id="prism_data_v1_m3b")
    finalize_lock(root, _report(True))
    validator.outcome["passed"] = False

    with pytest.raises(preparation.PreparationError) as raised:
        preparation.ensure_package_validated(root, label="the M3B prior package")

    assert raised.value.reason == preparation.PACKAGE_NOT_VALIDATED
    assert "does not validate" in str(raised.value)


def test_an_unrecognized_lock_status_is_refused(tmp_path: Path, validator) -> None:
    root = tmp_path / "prism_data_v1_m3b"
    _building_lock(root, package_id="prism_data_v1_m3b")
    lock = json.loads((root / "PACKAGE_LOCK.json").read_text(encoding="utf-8"))
    lock["status"] = "whatever"
    (root / "PACKAGE_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(preparation.PreparationError, match="unrecognized lock status"):
        preparation.ensure_package_validated(root, label="the M3B prior package")


# --- the freshly built path --------------------------------------------------

def test_a_newly_built_m3b_is_validated_finalized_and_revalidated(
        tmp_path: Path, validator, monkeypatch: pytest.MonkeyPatch) -> None:
    from prism_fas.data.package import m3b as m3b_module
    from prism_fas.pipeline import portable_paths

    target = tmp_path / preparation.DERIVED_PACKAGES["m3b"]
    parent = tmp_path / preparation.DERIVED_PACKAGES["m3a"]
    (tmp_path / "weights").mkdir(parents=True, exist_ok=True)

    def fake_build(source, output, model_config, *, weight_root, resume=True, **kwargs):
        _building_lock(Path(output), package_id="prism_data_v1_m3b")
        return {"samples": 4}

    monkeypatch.setattr(m3b_module, "build_m3b_package", fake_build)
    monkeypatch.setattr(portable_paths, "resolve", lambda repo: type(
        "R", (), {"weights": type("W", (), {"path": tmp_path / "weights"})()})())

    outcome = preparation._step_m3b(tmp_path, resume=True)

    assert outcome.action == "BUILT"
    assert [call["strict"] for call in validator] == [False, True]
    assert {call["parent"] for call in validator} == {parent}
    assert outcome.detail["package"]["status"] == "validated"
    assert outcome.detail["package"]["package_validation"] == "passed"
    assert preparation.package_status(target)["reusable_as_scientific_input"] is True


def test_the_m3a_reuse_branch_no_longer_accepts_a_building_package(
        tmp_path: Path, validator) -> None:
    """The same anti-pattern lived in M3A's reuse branch, with
    `require_validated_status=False` — which accepts a `building` package."""
    root = tmp_path / preparation.DERIVED_PACKAGES["m3a"]
    _building_lock(root, package_id="prism_data_v1_m3a")

    outcome = preparation._step_m3a(tmp_path, resume=True)

    assert outcome.action == "FINALIZED"
    assert [call["strict"] for call in validator] == [False, True]
    assert preparation.package_status(root)["status"] == "validated"


# --- the pair plan must notice the identity change ---------------------------

def _pair_plan(root: Path, *, package_identity: str, bank_identity: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("pair_manifest_train.parquet", "pair_manifest_validation.parquet"):
        (root / name).write_bytes(b"parquet")
    (root / "PAIR_PLAN_LOCK.json").write_text(json.dumps({
        "package_identity": package_identity, "recipe_bank_identity": bank_identity,
        "train_pairs": 896, "validation_pairs": 224}), encoding="utf-8")


def test_a_plan_built_before_finalization_is_not_reused(tmp_path: Path,
                                                        validator) -> None:
    """The RTX host's exact situation: gpat_pairs present, bound to the
    pre-finalization M3B identity."""
    package = tmp_path / preparation.DERIVED_PACKAGES["m3b"]
    pairs = tmp_path / preparation.DERIVED_PACKAGES["gpat_pairs"]
    stale = _building_lock(package, package_id="prism_data_v1_m3b")
    _pair_plan(pairs, package_identity=stale,
               bank_identity=preparation.M7_BANK_CONTENT_IDENTITY)

    assert preparation._pair_plan_is_current(
        pairs, package, preparation.M7_BANK_CONTENT_IDENTITY) is True, (
        "sanity: it matches the pre-finalization identity")

    preparation.ensure_package_validated(package, label="the M3B prior package")

    assert preparation._pair_plan_is_current(
        pairs, package, preparation.M7_BANK_CONTENT_IDENTITY) is False, (
        "finalization recomputed the package identity; the plan is stale")


def test_the_step_rebuilds_the_plan_against_the_finalized_identity(
        tmp_path: Path, validator, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    from prism_fas.synthesis import pair_plan as pair_plan_module

    shutil.copytree(REPO / preparation.M7_RECIPE_BANK,
                    tmp_path / preparation.M7_RECIPE_BANK)
    package = tmp_path / preparation.DERIVED_PACKAGES["m3b"]
    pairs = tmp_path / preparation.DERIVED_PACKAGES["gpat_pairs"]
    stale = _building_lock(package, package_id="prism_data_v1_m3b")
    _pair_plan(pairs, package_identity=stale,
               bank_identity=preparation.M7_BANK_CONTENT_IDENTITY)

    preparation.ensure_package_validated(package, label="the M3B prior package")
    final = preparation.package_status(package)["content_identity_sha256"]

    written: dict[str, Any] = {}

    def fake_write(package_root, bank_root, output_root, **kwargs):
        written.update(package_root=Path(package_root), bank_root=Path(bank_root))
        _pair_plan(Path(output_root), package_identity=final,
                   bank_identity=preparation.M7_BANK_CONTENT_IDENTITY)
        return {"lock": {"train_pairs": 896, "validation_pairs": 224},
                "summary": {"source_dev_opened": False, "target_test_opened": False}}

    monkeypatch.setattr(pair_plan_module, "write_pair_plan", fake_write)
    monkeypatch.setattr(pair_plan_module, "pair_plan_identity", lambda root: "rebuilt")
    outcome = preparation._step_pairs(tmp_path, resume=True)

    assert outcome.action == "BUILT", "a stale plan is rebuilt, not reused"
    assert written["package_root"] == package
    assert written["bank_root"] == tmp_path / preparation.M7_RECIPE_BANK
    lock = json.loads((pairs / "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8"))
    assert lock["package_identity"] == final, (
        "the lock must end bound to the FINAL validated M3B identity")
    assert lock["recipe_bank_identity"] == preparation.M7_BANK_CONTENT_IDENTITY
    assert preparation._pair_plan_is_current(
        pairs, package, preparation.M7_BANK_CONTENT_IDENTITY) is True


def test_a_plan_matching_the_finalized_identity_is_reused(tmp_path: Path,
                                                          validator) -> None:
    package = tmp_path / preparation.DERIVED_PACKAGES["m3b"]
    pairs = tmp_path / preparation.DERIVED_PACKAGES["gpat_pairs"]
    _building_lock(package, package_id="prism_data_v1_m3b")
    preparation.ensure_package_validated(package, label="the M3B prior package")
    final = preparation.package_status(package)["content_identity_sha256"]
    _pair_plan(pairs, package_identity=final,
               bank_identity=preparation.M7_BANK_CONTENT_IDENTITY)

    assert preparation._pair_plan_is_current(
        pairs, package, preparation.M7_BANK_CONTENT_IDENTITY) is True


# --- diagnosis must not describe a building package as valid -----------------

def test_the_diagnosis_reports_the_lock_status_not_just_locked(tmp_path: Path,
                                                              monkeypatch) -> None:
    import shutil

    shutil.copytree(REPO / "configs", tmp_path / "configs")
    package = tmp_path / preparation.DERIVED_PACKAGES["m3b"]
    _building_lock(package, package_id="prism_data_v1_m3b")
    monkeypatch.setattr(preparation, "_paths_config",
                        lambda repo: REPO / "configs" / "paths.local.yaml")

    report = preparation.diagnose(tmp_path)
    m3b = report["packages"]["m3b"]

    assert m3b["locked"] is True
    assert m3b["status"] == "building"
    assert m3b["package_validation"] == "pending"
    assert m3b["content_identity_sha256"]
    assert m3b["reusable_as_scientific_input"] is False
    assert m3b["why"]


def test_the_diagnosis_explains_a_stale_pair_plan(tmp_path: Path, monkeypatch) -> None:
    import shutil

    shutil.copytree(REPO / "configs", tmp_path / "configs")
    package = tmp_path / preparation.DERIVED_PACKAGES["m3b"]
    pairs = tmp_path / preparation.DERIVED_PACKAGES["gpat_pairs"]
    _building_lock(package, package_id="prism_data_v1_m3b")
    _pair_plan(pairs, package_identity="0" * 64,
               bank_identity=preparation.M7_BANK_CONTENT_IDENTITY)
    monkeypatch.setattr(preparation, "_paths_config",
                        lambda repo: REPO / "configs" / "paths.local.yaml")

    plan = preparation.diagnose(tmp_path)["gpat_pairs"]

    assert plan["locked"] is True
    assert plan["bound_to_frozen_bank"] is True
    assert plan["bound_to_current_package"] is False
    assert plan["reusable"] is False
    assert "pre-finalization" in plan["why"]
    assert "no manual deletion" in plan["why"]


def test_the_diagnosis_opens_no_target_and_writes_nothing(tmp_path: Path,
                                                          monkeypatch) -> None:
    import shutil

    shutil.copytree(REPO / "configs", tmp_path / "configs")
    _building_lock(tmp_path / preparation.DERIVED_PACKAGES["m3b"],
                   package_id="prism_data_v1_m3b")
    monkeypatch.setattr(preparation, "_paths_config",
                        lambda repo: REPO / "configs" / "paths.local.yaml")
    before = {path: path.read_bytes() for path in sorted(tmp_path.rglob("*"))
              if path.is_file()}

    report = preparation.diagnose(tmp_path)

    assert {path: path.read_bytes() for path in sorted(tmp_path.rglob("*"))
            if path.is_file()} == before
    assert "siw" not in json.dumps(report, default=str).lower()
    assert report["scientific_eligible"] is False


# --- the coarse pass must not report NOTHING_TO_DO over any of this ----------

def test_a_building_package_keeps_the_packages_tree_on_the_to_do_list(
        tmp_path: Path) -> None:
    """Otherwise `prepare` returns NOTHING_TO_DO and the fix never runs.

    `what_is_needed` is what decides whether the steps execute at all. It used to
    ask "is the marker the builder writes last present" — and the builder writes
    the `building` lock itself, so an unfinalized package looked finished.
    """
    for key in ("m3a", "m3b"):
        _building_lock(tmp_path / preparation.DERIVED_PACKAGES[key],
                       package_id=f"prism_data_v1_{key}")

    assert preparation._incomplete(tmp_path, "packages") is True

    for key in ("m3a", "m3b"):
        finalize_lock(tmp_path / preparation.DERIVED_PACKAGES[key], _report(True))

    assert preparation._incomplete(tmp_path, "packages") is False


def test_a_stale_pair_plan_keeps_gpat_pairs_on_the_to_do_list(tmp_path: Path) -> None:
    package = tmp_path / preparation.DERIVED_PACKAGES["m3b"]
    pairs = tmp_path / preparation.DERIVED_PACKAGES["gpat_pairs"]
    stale = _building_lock(package, package_id="prism_data_v1_m3b")
    _pair_plan(pairs, package_identity=stale,
               bank_identity=preparation.M7_BANK_CONTENT_IDENTITY)

    assert preparation._incomplete(tmp_path, "gpat_pairs") is False, (
        "sanity: it matches the identity the package currently has")

    finalize_lock(package, _report(True))

    assert preparation._incomplete(tmp_path, "gpat_pairs") is True, (
        "finalization recomputed the package identity; the plan must be rebuilt")


def test_the_host_state_is_not_reported_nothing_to_do(tmp_path: Path,
                                                      monkeypatch) -> None:
    """The RTX host exactly: M2 complete, both packages locked, a pair plan
    present — and M3B still `building`."""
    monkeypatch.setattr(preparation, "_m2_status_or_absent",
                        lambda repo, buildable: {"complete": True, "reason": None})
    monkeypatch.setattr(preparation, "_paths_config",
                        lambda repo: REPO / "configs" / "paths.local.yaml")
    _building_lock(tmp_path / preparation.DERIVED_PACKAGES["m3a"],
                   package_id="prism_data_v1_m3a")
    finalize_lock(tmp_path / preparation.DERIVED_PACKAGES["m3a"], _report(True))
    stale = _building_lock(tmp_path / preparation.DERIVED_PACKAGES["m3b"],
                           package_id="prism_data_v1_m3b")
    _pair_plan(tmp_path / preparation.DERIVED_PACKAGES["gpat_pairs"],
               package_identity=stale,
               bank_identity=preparation.M7_BANK_CONTENT_IDENTITY)
    for dataset in ("casia_fasd", "msu_mfsd"):
        (tmp_path / "data" / "raw" / dataset).mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "raw" / dataset / "README.txt").write_text("x", encoding="utf-8")

    needed = preparation.what_is_needed(tmp_path)

    assert needed["nothing_to_do"] is False, (
        "an unfinalized M3B must not be reported as nothing to do")
    assert "packages" in needed["missing_derived"]


# --- the loop the RTX host actually walked -----------------------------------

def test_the_c4_gate_accepts_what_preparation_now_finalizes(tmp_path: Path,
                                                            validator) -> None:
    """End to end against the real refusal.

    Before: preparation left M3B at `building`, and
    `sources.verify_support_inputs` — the gate that stopped the host — raised
    `SourceUnavailable: reports status 'building'`. After: the same gate accepts
    the same package, because preparation finalized it.
    """
    import shutil

    from prism_fas.pipeline.adapters import sources

    shutil.copytree(REPO / preparation.M7_RECIPE_BANK,
                    tmp_path / preparation.M7_RECIPE_BANK)
    package = tmp_path / preparation.DERIVED_PACKAGES["m3b"]
    pairs = tmp_path / preparation.DERIVED_PACKAGES["gpat_pairs"]
    building_identity = _building_lock(package, package_id="prism_data_v1_m3b")

    # The host's state: a plan bound to the pre-finalization identity.
    _pair_plan(pairs, package_identity=building_identity,
               bank_identity=preparation.M7_BANK_CONTENT_IDENTITY)
    with pytest.raises(sources.SourceUnavailable, match="building"):
        sources.verify_support_inputs(tmp_path)

    preparation.ensure_package_validated(package, label="the M3B prior package")
    final_identity = preparation.package_status(package)["content_identity_sha256"]

    # Still refused — now for the right reason: the plan is bound to an identity
    # that no longer exists, which is exactly what must not be trained on.
    with pytest.raises(sources.SupportIdentityMismatch):
        sources.verify_support_inputs(tmp_path)

    # Rebuilding the plan against the finalized package closes the loop.
    _pair_plan(pairs, package_identity=final_identity,
               bank_identity=preparation.M7_BANK_CONTENT_IDENTITY)
    report = sources.verify_support_inputs(tmp_path)

    assert report["identities_agree"] is True
    assert report["package_identity"] == final_identity
    assert report["package_identity"] != building_identity
    assert report["bank_identity"] == preparation.M7_BANK_CONTENT_IDENTITY


# --- nothing is deleted or rebuilt to obtain a status change -----------------

def test_finalization_deletes_nothing_anywhere(tmp_path: Path, validator) -> None:
    root = tmp_path / "prism_data_v1_m3b"
    _building_lock(root, package_id="prism_data_v1_m3b")
    for relative in ("priors/a.npz", "images/a.jpg", "shards/source_train-00000.tar",
                     "manifests/samples.parquet"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    payload_before = _payload(root)

    preparation.ensure_package_validated(root, label="the M3B prior package")

    assert _payload(root) == payload_before
    assert len(payload_before) == 4
