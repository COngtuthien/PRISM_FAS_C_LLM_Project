"""The C4 scientific support batch and the three artifacts it must agree with.

Two stale roots were in one call, and they failed in different ways.

`SOURCE_PACKAGE_ROOT` was `data/packages`, the parent directory, while
`SampleStore.open` reads `<package_root>/manifests/source_train.parquet` — a file
that has never existed one level up. And the recipe bank was
`assets/recipe_banks/c3`, the container of the three C3 treatment banks, which is
not an M7 frozen-bank root at all.

The bank is not a preference. `m8_pipeline.build_batch` does
`recipes[pair["recipe_id"]]` over whatever bank it is handed, so the support
batch has to resolve the same bank the pair plan drew its recipe ids from.
Preparation binds the plan to M3B + M7 from frozen Version-B evidence; this
consumer reads that binding rather than restating it.

Nothing here stubs a root. `SampleStore.open`, `load_pairs`, `resolve_bank` and
the real frozen M7 bank are all the shipping code — only `build_batch` and the
AdaFace registry are replaced, because neither is what this contract is about.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline import preparation  # noqa: E402
from prism_fas.pipeline.adapters import sources  # noqa: E402

pytest.importorskip("pyarrow")

PACKAGE_IDENTITY = "b" * 64
ADAFACE_SHA = "43bd2d570584d95d4a17ce81f26449034c45dbeed750afcab651872abc0e1496"


# --- the roots the production module resolves --------------------------------

def test_the_scientific_package_root_is_the_m3b_package_not_its_parent() -> None:
    """`SampleStore.open` reads `<root>/manifests/source_train.parquet`."""
    assert sources.SOURCE_PACKAGE_ROOT == "data/packages/prism_data_v1_m3b"
    assert sources.SOURCE_PACKAGE_ROOT != "data/packages"
    assert (REPO / "data" / "packages" / "manifests").exists() is False, (
        "the parent directory holds no manifests, which is why the old root "
        "could never have worked")


def test_the_scientific_bank_root_is_the_frozen_m7_bank() -> None:
    assert sources.RECIPE_BANK_ROOT == "assets/recipe_banks/prism_recipe_bank_m7_v1"
    assert "c3" not in Path(sources.RECIPE_BANK_ROOT).parts


def test_the_roots_come_from_the_producer_not_a_second_spelling() -> None:
    """One declaration, so a rename in preparation cannot leave C4 behind."""
    assert sources.SOURCE_PACKAGE_ROOT == preparation.DERIVED_PACKAGES[
        preparation.PAIR_PLAN_PACKAGE]
    assert sources.GPAT_PAIR_ROOT == preparation.DERIVED_PACKAGES["gpat_pairs"]
    assert sources.RECIPE_BANK_ROOT == preparation.M7_RECIPE_BANK


def test_c3_is_declared_as_the_rehearsal_conditioning_source_only() -> None:
    assert sources.REHEARSAL_CONDITIONING_SOURCE == "assets/recipe_banks/c3"
    assert sources.REHEARSAL_CONDITIONING_SOURCE != sources.RECIPE_BANK_ROOT


# --- a scientific repository, assembled from the real frozen bank ------------

class _Context:
    """The policy object `support_batch` asks, with the real invariant."""

    def __init__(self, scientific: bool) -> None:
        self.scientific_eligible = scientific
        self.fixtures_permitted = not scientific
        self.name = "SCIENTIFIC" if scientific else "REHEARSAL"

    @property
    def is_scientific(self) -> bool:
        return self.scientific_eligible


class _Registry:
    """Stands in for the pinned AdaFace registry; not what this file tests."""

    verified = {"identity": ADAFACE_SHA}

    @classmethod
    def resolve(cls, weight_root: Path, *, roles: tuple[str, ...]) -> "_Registry":
        return cls()

    def adaface(self, device: str) -> Any:
        return object()


def _source_manifest(path: Path, sample_ids: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict({
        "sample_id": sample_ids,
        "project_split": ["source_train"] * len(sample_ids),
        "dataset": ["casia_fasd"] * len(sample_ids),
        "label_live_spoof": ["live", "spoof"] * (len(sample_ids) // 2)}), path)


def _pair_rows(recipe_ids: list[str], bank_identity: str) -> list[dict[str, Any]]:
    return [{"pair_id": f"gpatpair_{index}", "partition": "train", "slot": index,
             "domain_relation": "same_domain",
             "live_sample_id": f"live_{index}", "live_dataset": "casia_fasd",
             "live_source_record_id": f"lrec_{index}",
             "spoof_sample_id": f"spoof_{index}", "spoof_dataset": "casia_fasd",
             "spoof_source_record_id": f"srec_{index}",
             "recipe_id": recipe_ids[index], "recipe_seed": 1,
             "different_subject_rule": "enforced",
             "package_identity": PACKAGE_IDENTITY,
             "recipe_bank_identity": bank_identity}
            for index in range(len(recipe_ids))]


@pytest.fixture
def scientific_repo(tmp_path: Path) -> dict[str, Any]:
    """A project carrying the REAL frozen M7 bank, a locked M3B package and a
    pair plan whose recipe ids are drawn from that exact bank."""
    from prism_fas.recipes.bank import load_bank
    from prism_fas.synthesis import pair_plan as pair_plan_module

    repo = tmp_path / "p"
    shutil.copytree(REPO / sources.RECIPE_BANK_ROOT, repo / sources.RECIPE_BANK_ROOT)
    bank = load_bank(repo / sources.RECIPE_BANK_ROOT)
    bank_identity = str(bank["lock"]["bank_content_identity_sha256"])
    recipe_ids = [recipe.recipe_id for recipe in bank["recipes"]][:4]

    package = repo / sources.SOURCE_PACKAGE_ROOT
    sample_ids = [f"{role}_{index}" for index in range(4) for role in ("live", "spoof")]
    _source_manifest(package / "manifests" / "source_train.parquet", sorted(sample_ids))
    # The two splits M8 must never open, written as bytes no parquet reader can
    # parse: opening one is a hard failure rather than a silent success.
    for forbidden in ("source_dev.parquet", "target_test_features.parquet"):
        (package / "manifests" / forbidden).write_bytes(b"opening this is a failure")
    (package / "PACKAGE_LOCK.json").write_text(json.dumps({
        "package_id": "prism_data_v1_m3b", "status": "validated",
        "content_identity_sha256": PACKAGE_IDENTITY,
        "package_validation": {"status": "passed", "checks_passed": 40,
                               "checks_total": 40}}), encoding="utf-8")

    pairs = repo / sources.GPAT_PAIR_ROOT
    rows = _pair_rows(recipe_ids, bank_identity)
    pair_plan_module._write_parquet(pairs / "pair_manifest_train.parquet", rows)
    pair_plan_module._write_parquet(pairs / "pair_manifest_validation.parquet", rows[:2])
    (pairs / "PAIR_PLAN_LOCK.json").write_text(json.dumps({
        "pair_plan_schema_version": "m8-pair-plan-v1", "seed": 20260806,
        "package_identity": PACKAGE_IDENTITY, "recipe_bank_identity": bank_identity,
        "train_pairs": 896, "validation_pairs": 224,
        "pair_plan_identity_sha256": "d" * 64}), encoding="utf-8")

    return {"repo": repo, "bank_identity": bank_identity, "recipe_ids": recipe_ids,
            "package_lock": package / "PACKAGE_LOCK.json",
            "plan_lock": pairs / "PAIR_PLAN_LOCK.json"}


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Records what the real resolvers were handed, without building tensors."""
    from prism_fas.synthesis import m8_pipeline, quality_models

    seen: dict[str, Any] = {}
    real_open, real_resolve = m8_pipeline.SampleStore.open, m8_pipeline.resolve_bank

    def watched_open(package_root, audit=None):
        seen["sample_store_root"] = Path(package_root)
        return real_open(package_root, audit)

    def watched_resolve(bank_root):
        seen["bank_root"] = Path(bank_root)
        return real_resolve(bank_root)

    def watched_build(store, pairs, bank, identity_model, *, device="cpu"):
        seen.update(build_pairs=list(pairs), build_bank=bank, build_store=store)
        return object()

    monkeypatch.setattr(m8_pipeline.SampleStore, "open", watched_open)
    monkeypatch.setattr(m8_pipeline, "resolve_bank", watched_resolve)
    monkeypatch.setattr(m8_pipeline, "build_batch", watched_build)
    monkeypatch.setattr(quality_models, "QualityModelRegistry", _Registry)
    return seen


def _run(repo: Path, size: int = 2) -> tuple[Any, dict[str, Any]]:
    return sources.support_batch(repo, size, _Context(scientific=True))


# --- what the production path hands the canonical resolvers ------------------

def test_the_sample_store_is_opened_on_the_m3b_package(scientific_repo, spy) -> None:
    _run(scientific_repo["repo"])

    assert spy["sample_store_root"] == scientific_repo["repo"] / sources.SOURCE_PACKAGE_ROOT
    assert spy["sample_store_root"].name == "prism_data_v1_m3b"


def test_the_sample_store_is_never_opened_on_the_packages_parent(
        scientific_repo, spy) -> None:
    """The old root, named so a regression to it is unmistakable."""
    _run(scientific_repo["repo"])

    assert spy["sample_store_root"] != scientific_repo["repo"] / "data" / "packages"
    assert (spy["sample_store_root"] / "manifests" / "source_train.parquet").is_file()


def test_the_bank_resolved_is_the_frozen_m7_bank(scientific_repo, spy) -> None:
    _run(scientific_repo["repo"])

    assert spy["bank_root"] == scientific_repo["repo"] / sources.RECIPE_BANK_ROOT
    assert spy["bank_root"].name == "prism_recipe_bank_m7_v1"


def test_no_c3_root_is_ever_resolved_under_science(scientific_repo, spy) -> None:
    _run(scientific_repo["repo"])
    repo = scientific_repo["repo"]

    assert "c3" not in spy["bank_root"].parts
    for arm in ("det", "llm", "rnd"):
        assert spy["bank_root"] != repo / "assets" / "recipe_banks" / "c3" / arm
    assert spy["bank_root"] != repo / "assets" / "recipe_banks" / "c3"


# --- the three identities agree ----------------------------------------------

def test_the_pair_plan_package_identity_matches_the_loaded_m3b(scientific_repo) -> None:
    report = sources.verify_support_inputs(scientific_repo["repo"])
    plan = json.loads(scientific_repo["plan_lock"].read_text(encoding="utf-8"))
    package = json.loads(scientific_repo["package_lock"].read_text(encoding="utf-8"))

    assert report["package_identity"] == package["content_identity_sha256"]
    assert report["package_identity"] == plan["package_identity"]
    assert report["identities_agree"] is True


def test_the_pair_plan_bank_identity_matches_the_loaded_m7(scientific_repo) -> None:
    report = sources.verify_support_inputs(scientific_repo["repo"])
    plan = json.loads(scientific_repo["plan_lock"].read_text(encoding="utf-8"))

    assert report["bank_identity"] == scientific_repo["bank_identity"]
    assert report["bank_identity"] == plan["recipe_bank_identity"]
    assert report["bank_identity"] == preparation.M7_BANK_CONTENT_IDENTITY
    assert report["bank_id"] == "prism_recipe_bank_m7_v1"
    assert report["bank_status"] == "frozen"


def test_build_batch_receives_recipe_ids_that_exist_in_that_bank(
        scientific_repo, spy) -> None:
    """The row-level consequence of the identity agreement, and the thing that
    would otherwise surface as a bare KeyError inside `build_batch`."""
    _run(scientific_repo["repo"], size=4)

    known = {recipe.recipe_id for recipe in spy["build_bank"]["recipes"]}
    used = {pair["recipe_id"] for pair in spy["build_pairs"]}
    assert used, "the spy saw no pairs"
    assert used <= known
    assert spy["build_bank"]["bank_id"] == "prism_recipe_bank_m7_v1"


def test_a_pair_plan_from_another_bank_is_refused(scientific_repo, spy) -> None:
    plan = json.loads(scientific_repo["plan_lock"].read_text(encoding="utf-8"))
    plan["recipe_bank_identity"] = "e" * 64
    scientific_repo["plan_lock"].write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(sources.SupportIdentityMismatch) as raised:
        _run(scientific_repo["repo"])

    assert raised.value.reason_code == "IDENTITY_MISMATCH"
    assert "build_store" not in spy, "nothing may be opened after a mismatch"


def test_a_pair_plan_from_another_package_is_refused(scientific_repo, spy) -> None:
    plan = json.loads(scientific_repo["plan_lock"].read_text(encoding="utf-8"))
    plan["package_identity"] = "f" * 64
    scientific_repo["plan_lock"].write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(sources.SupportIdentityMismatch):
        _run(scientific_repo["repo"])

    assert "build_store" not in spy


def test_an_unvalidated_package_is_refused(scientific_repo, spy) -> None:
    package = json.loads(scientific_repo["package_lock"].read_text(encoding="utf-8"))
    package["package_validation"] = {"status": "pending"}
    scientific_repo["package_lock"].write_text(json.dumps(package), encoding="utf-8")

    with pytest.raises(sources.SourceUnavailable, match="package_validation"):
        _run(scientific_repo["repo"])

    assert "build_store" not in spy


@pytest.mark.parametrize("absent", ["PAIR_PLAN_LOCK.json",
                                    "pair_manifest_train.parquet",
                                    "pair_manifest_validation.parquet"])
def test_an_incomplete_pair_plan_is_refused(scientific_repo, spy, absent: str) -> None:
    (scientific_repo["repo"] / sources.GPAT_PAIR_ROOT / absent).unlink()

    with pytest.raises(sources.SourceUnavailable, match="is absent"):
        _run(scientific_repo["repo"])

    assert "build_store" not in spy


def test_a_bank_that_fails_the_frozen_gate_is_refused(scientific_repo, spy) -> None:
    lock_path = scientific_repo["repo"] / sources.RECIPE_BANK_ROOT / "BANK_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["status"] = "building"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    with pytest.raises(sources.SourceUnavailable, match="frozen recipe bank"):
        _run(scientific_repo["repo"])

    assert "build_store" not in spy


def test_the_gate_writes_nothing(scientific_repo) -> None:
    repo = scientific_repo["repo"]
    before = {path: path.read_bytes() for path in sorted(repo.rglob("*")) if path.is_file()}

    sources.verify_support_inputs(repo)

    assert {path: path.read_bytes()
            for path in sorted(repo.rglob("*")) if path.is_file()} == before


# --- source-only, measured by the store itself -------------------------------

def test_only_the_source_train_manifest_is_opened(scientific_repo, spy) -> None:
    _, provenance = _run(scientific_repo["repo"])
    audit = provenance["source_only_audit"]

    assert audit["manifests_opened"] == ["manifests/source_train.parquet"]
    assert audit["source_train_opened"] is True
    assert audit["source_dev_opened"] is False
    assert audit["target_test_opened"] is False
    assert audit["target_label_artifact_opened"] is False
    assert audit["raw_dataset_path_opened"] is False


def test_the_forbidden_splits_stay_unreadable(scientific_repo, spy) -> None:
    """They are on disk as unparseable bytes, so a run that opened one could not
    have completed. It completed."""
    _run(scientific_repo["repo"])
    manifests = scientific_repo["repo"] / sources.SOURCE_PACKAGE_ROOT / "manifests"

    assert (manifests / "source_dev.parquet").read_bytes().startswith(b"opening")
    assert (manifests / "target_test_features.parquet").read_bytes().startswith(b"opening")


def test_no_target_artifact_is_named_anywhere_in_the_provenance(
        scientific_repo, spy) -> None:
    _, provenance = _run(scientific_repo["repo"])
    serialized = json.dumps(provenance, default=str).lower()

    assert "siw" not in serialized
    assert provenance["fixture_backed"] is False
    assert provenance["inputs"]["identities_agree"] is True


def test_the_audit_refuses_a_forbidden_open_by_construction() -> None:
    """The store's own guard, exercised directly: it raises rather than records."""
    from prism_fas.synthesis.m8_pipeline import PipelineError, SourceOnlyAudit

    audit = SourceOnlyAudit()
    for forbidden in ("manifests/source_dev.parquet",
                      "manifests/target_test_features.parquet",
                      "manifests/siw_target_features.parquet"):
        with pytest.raises(PipelineError):
            audit.record(forbidden)
    assert audit.opened == []


# --- the rehearsal branch is unchanged and cannot be reached by science -------

def test_the_rehearsal_still_uses_the_c3_fixture_conditioning(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from prism_fas.pipeline.adapters import c4

    monkeypatch.setattr(c4, "_fixture_batch",
                        lambda repo, size, *, seed: f"fixture-{size}")
    batch, provenance = sources.support_batch(tmp_path, 2, _Context(scientific=False))

    assert batch == "fixture-2"
    assert provenance["fixture_backed"] is True
    assert provenance["source"] == "deterministic_fixture"
    assert provenance["conditioning_source"] == "assets/recipe_banks/c3"
    assert c4.SUPPORT_RECIPE_SOURCE.startswith("assets/recipe_banks/c3/")


def test_a_scientific_context_can_never_take_the_fixture_branch() -> None:
    """Structural, not conventional: the flag is derived, never set.

    Built from the real `ExecutionContext`, so the guarantee is the shipping
    one rather than a property of this file's stand-in.
    """
    from dataclasses import dataclass

    from prism_fas.pipeline.execution import ExecutionContext

    @dataclass
    class _Profile:
        name: str = "full"
        scientific_eligible: bool = True
        reports_namespace: str = "full"
        runs_namespace: str = "full"

    context = ExecutionContext.for_profile(_Profile())
    assert context.is_scientific is True
    assert context.fixtures_permitted is False
    assert context.budget is None, "a scientific run may truncate nothing"

    # ...and the stand-in this file uses obeys the same invariant in both
    # directions, so the branch tests above are testing the real condition.
    for scientific in (True, False):
        assert _Context(scientific).fixtures_permitted is not _Context(scientific).is_scientific


def test_the_scientific_branch_has_no_fixture_fallback(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path) -> None:
    """With every real input absent, science raises. It does not degrade."""
    from prism_fas.pipeline.adapters import c4

    monkeypatch.setattr(c4, "_fixture_batch",
                        lambda *args, **kwargs: pytest.fail(
                            "a scientific run must never reach the fixture batch"))

    with pytest.raises(sources.SourceUnavailable):
        sources.support_batch(tmp_path, 2, _Context(scientific=True))
