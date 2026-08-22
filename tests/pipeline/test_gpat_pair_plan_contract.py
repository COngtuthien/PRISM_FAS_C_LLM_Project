"""The GPAT pair plan and the frozen recipe bank it is bound to.

The defect this file exists to stop from recurring: `_step_pairs` passed
`assets/recipe_banks/c3` to `write_pair_plan`, which reaches `load_bank`. That
directory is a CONTAINER of the three C3 scientific banks (`det/`, `llm/`,
`rnd/`), each holding `C3_BANK.json` + `recipes.jsonl`. It carries none of the
seven files `recipes.bank.BANK_FILES` requires, so the real RTX 5090 host — after
M2, M3A and M3B had all completed — stopped with

    BankError: assets/recipe_banks/c3 is not a frozen recipe bank; missing [...]

The C3 banks and the frozen M7 bank are different contracts. Nothing here
converts between them, and a test below asserts no C3 root is ever handed to
`load_bank`.

Two roots are bound by frozen Version-B evidence rather than chosen: the
pair-plan lock records `recipe_bank_identity` = fa989938…10cb (the M7 bank) and
`package_identity` = b1cf29b6…9dc6 (the M3B package). `_step_pairs` was passing
the M3A package, which is structurally loadable and would therefore have stamped
the wrong identity into every `pair_id` — silently.
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

from prism_fas.pipeline import preparation  # noqa: E402
from prism_fas.recipes import bank as bank_module  # noqa: E402
from prism_fas.synthesis import pair_plan as pair_plan_module  # noqa: E402

pytest.importorskip("pyarrow")

#: The frozen contract, as recorded in docs/c0/C0_VERSION_B_INTEGRITY.md §2.2 and
#: in the Version-B pair-plan lock. Spelled out here rather than imported from
#: the module under test — a test that reads its expectation from the code it
#: checks proves nothing.
FROZEN_BANK_ID = "prism_recipe_bank_m7_v1"
FROZEN_BANK_IDENTITY = "fa989938cafdc4887518cc45c35d559d00278358439dc68c2486da10309210cb"
FROZEN_TRAIN_PAIRS = 896
FROZEN_VALIDATION_PAIRS = 224


# --- the bank the production step actually resolves --------------------------

def test_the_step_resolves_the_frozen_m7_bank_not_the_c3_container() -> None:
    """The whole defect, named."""
    root = preparation.recipe_bank_root(REPO)

    assert root == REPO / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1"
    assert root.name == FROZEN_BANK_ID
    assert "c3" not in root.relative_to(REPO).parts, (
        "the C3 container is a different contract and is never a load_bank input")


def test_load_bank_succeeds_on_the_exact_shipped_root() -> None:
    bank = bank_module.load_bank(preparation.recipe_bank_root(REPO))

    assert bank["bank_id"] == FROZEN_BANK_ID
    assert len(bank["recipes"]) == 128
    assert bank["lock"]["status"] == bank_module.BANK_STATUS_FROZEN


def test_validate_bank_passes_on_the_shipped_root() -> None:
    report = bank_module.validate_bank(preparation.recipe_bank_root(REPO))

    assert report["passed"] is True, report.get("errors")
    assert report["errors"] == []


def test_every_bank_file_is_present() -> None:
    root = preparation.recipe_bank_root(REPO)
    missing = [name for name in bank_module.BANK_FILES if not (root / name).is_file()]

    assert missing == []


def test_the_c3_container_is_not_a_frozen_bank() -> None:
    """The measurement behind the claim, not an assumption about it."""
    with pytest.raises(bank_module.BankError, match="not a frozen recipe bank"):
        bank_module.load_bank(REPO / "assets" / "recipe_banks" / "c3")


@pytest.mark.parametrize("arm", ["det", "llm", "rnd"])
def test_no_c3_arm_can_be_silently_substituted(arm: str) -> None:
    """Pointing one directory deeper is the tempting near-miss. It also fails —
    a C3 arm is `C3_BANK.json` + `recipes.jsonl`, not the M7 contract."""
    arm_root = REPO / "assets" / "recipe_banks" / "c3" / arm
    assert arm_root.is_dir(), "the C3 banks must still be there, untouched"

    with pytest.raises(bank_module.BankError, match="not a frozen recipe bank"):
        bank_module.load_bank(arm_root)


def test_the_c3_banks_keep_their_own_contract() -> None:
    """C3 and M7 stay explicitly distinct: neither grew the other's files."""
    for arm in ("det", "llm", "rnd"):
        root = REPO / "assets" / "recipe_banks" / "c3" / arm
        assert (root / "C3_BANK.json").is_file()
        assert (root / "recipes.jsonl").is_file()
        assert not (root / "BANK_LOCK.json").exists(), (
            "a C3 bank must never be converted into the M7 contract")


# --- the validation gate -----------------------------------------------------

def test_the_gate_reports_the_frozen_identity() -> None:
    report = preparation.validate_recipe_bank(REPO)

    assert report["bank_id"] == FROZEN_BANK_ID
    assert report["status"] == "frozen"
    assert report["recipe_count"] == 128
    assert report["bank_content_identity_sha256"] == FROZEN_BANK_IDENTITY
    assert report["validated_by"].endswith("validate_bank"), (
        "the canonical validator must do the validating")


def test_a_missing_bank_fails_closed(tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preparation, "recipe_bank_root", lambda repo: tmp_path / "absent")

    with pytest.raises(preparation.PreparationError) as raised:
        preparation.validate_recipe_bank(REPO)

    assert raised.value.reason == preparation.RECIPE_BANK_INVALID
    assert "not a frozen recipe bank" in str(raised.value)
    assert set(raised.value.detail["missing"]) == set(bank_module.BANK_FILES)


def test_a_substituted_but_internally_consistent_bank_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`validate_bank` cannot catch this on its own — a different bank that
    re-derives its own hashes passes it. Only the pinned identity refuses it."""
    import shutil

    other = tmp_path / "other_bank"
    shutil.copytree(preparation.recipe_bank_root(REPO), other)
    lock = json.loads((other / "BANK_LOCK.json").read_text(encoding="utf-8"))
    lock["bank_content_identity_sha256"] = "0" * 64
    (other / "BANK_LOCK.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    monkeypatch.setattr(preparation, "recipe_bank_root", lambda repo: other)

    with pytest.raises(preparation.PreparationError) as raised:
        preparation.validate_recipe_bank(REPO)

    assert raised.value.reason == preparation.RECIPE_BANK_INVALID
    assert raised.value.detail["expected_identity"] == FROZEN_BANK_IDENTITY


def test_an_unfrozen_bank_is_refused(tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    draft = tmp_path / "draft_bank"
    shutil.copytree(preparation.recipe_bank_root(REPO), draft)
    lock = json.loads((draft / "BANK_LOCK.json").read_text(encoding="utf-8"))
    lock["status"] = "building"
    (draft / "BANK_LOCK.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    monkeypatch.setattr(preparation, "recipe_bank_root", lambda repo: draft)

    with pytest.raises(preparation.PreparationError) as raised:
        preparation.validate_recipe_bank(REPO)

    assert raised.value.reason == preparation.RECIPE_BANK_INVALID
    assert "status" in str(raised.value)


def test_the_gate_writes_nothing_and_repairs_nothing(tmp_path: Path) -> None:
    root = preparation.recipe_bank_root(REPO)
    before = {path: path.read_bytes() for path in sorted(root.iterdir()) if path.is_file()}

    preparation.validate_recipe_bank(REPO)

    assert {path: path.read_bytes()
            for path in sorted(root.iterdir()) if path.is_file()} == before


# --- the package the plan is bound to ----------------------------------------

def test_the_plan_is_bound_to_the_m3b_package() -> None:
    """Frozen Version-B evidence: the pair-plan lock records the M3B package
    identity, and both production M8 callsites pass the M3B root. M3A is
    structurally loadable, so naming it would have been silently wrong."""
    assert preparation.PAIR_PLAN_PACKAGE == "m3b"
    assert preparation.DERIVED_PACKAGES["m3b"].endswith("prism_data_v1_m3b")

    modal = (REPO / "modal_m8.py").read_text(encoding="utf-8")
    assert "prism_data_v1_m3b" in modal
    assert "assets\" / \"recipe_banks\" / \"prism_recipe_bank_m7_v1" in modal or \
           "prism_recipe_bank_m7_v1" in modal


# --- the frozen pair-count contract ------------------------------------------

def test_the_pair_counts_are_the_frozen_ones() -> None:
    """896 / 224, declared in two places that must agree."""
    import yaml

    config = yaml.safe_load(
        (REPO / "configs" / "synthesis" / "gpat_m8.yaml").read_text(encoding="utf-8"))

    assert pair_plan_module.EXPECTED_TRAIN_PAIRS == FROZEN_TRAIN_PAIRS
    assert pair_plan_module.EXPECTED_VALIDATION_PAIRS == FROZEN_VALIDATION_PAIRS
    assert config["pair_plan"]["expected_train_pairs"] == FROZEN_TRAIN_PAIRS
    assert config["pair_plan"]["expected_validation_pairs"] == FROZEN_VALIDATION_PAIRS
    assert config["pair_plan"]["seed"] == pair_plan_module.PAIR_PLAN_SEED


@pytest.mark.skipif(not (Path("D:/AI on IOT/Anti_spoofing/PRISM_FAS_B_Project")
                         / "reports" / "m8" / "pairs" / "PAIR_PLAN_LOCK.json").is_file(),
                    reason="the immutable Version-B repository is not on this machine")
def test_the_frozen_version_b_plan_records_these_identities() -> None:
    """Read-only against the immutable Version-B evidence, which is what binds
    both roots. Skipped where that repository is not present."""
    lock = json.loads(
        (Path("D:/AI on IOT/Anti_spoofing/PRISM_FAS_B_Project") / "reports" / "m8"
         / "pairs" / "PAIR_PLAN_LOCK.json").read_text(encoding="utf-8"))

    assert lock["recipe_bank_identity"] == FROZEN_BANK_IDENTITY
    assert lock["train_pairs"] == FROZEN_TRAIN_PAIRS
    assert lock["validation_pairs"] == FROZEN_VALIDATION_PAIRS
    assert lock["seed"] == pair_plan_module.PAIR_PLAN_SEED


# --- a real package fixture, for the split and write-order contracts ---------

def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "dataset", "source_record_id", "subject_id",
               "label_live_spoof", "project_split"]
    pq.write_table(pa.Table.from_pydict({name: [row[name] for row in rows]
                                         for name in columns}), path)


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """A package shaped like M3B: source_train, plus the two splits M8 must never
    open, written as unreadable bytes so opening one is a hard failure."""
    root = tmp_path / "prism_data_v1_m3b"
    rows: list[dict[str, Any]] = []
    for dataset in ("casia_fasd", "msu_mfsd"):
        for label in ("live", "spoof"):
            for index in range(10):
                record = f"{dataset}_{label}_{index}"
                rows.append({"sample_id": f"s_{record}", "dataset": dataset,
                             "source_record_id": record, "subject_id": record,
                             "label_live_spoof": label, "project_split": "source_train"})
    _write_manifest(root / "manifests" / "source_train.parquet", rows)
    for forbidden in ("source_dev.parquet", "target_test_features.parquet"):
        (root / "manifests" / forbidden).write_bytes(b"opening this is a test failure")
    (root / "PACKAGE_LOCK.json").write_text(json.dumps({
        "package_id": "prism_data_v1_m3b", "status": "validated",
        "content_identity_sha256": "b" * 64}), encoding="utf-8")
    return root


def test_only_the_source_train_manifest_is_opened(package: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Every parquet read is recorded, so `source_dev` and `target_test` being
    absent from the list is a measurement rather than a promise."""
    import pyarrow.parquet as pq

    opened: list[str] = []
    real = pq.read_table

    def watched(source, *args, **kwargs):
        opened.append(Path(str(source)).name)
        return real(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", watched)
    pair_plan_module.build_pair_plan(package, preparation.recipe_bank_root(REPO))

    assert opened == ["source_train.parquet"]
    assert "source_dev.parquet" not in opened
    assert "target_test_features.parquet" not in opened


def test_no_target_artifact_is_touched(package: Path) -> None:
    plan = pair_plan_module.build_pair_plan(package, preparation.recipe_bank_root(REPO))
    rows = plan["pairs"]["train"] + plan["pairs"]["validation"]

    assert rows
    assert {row["live_dataset"] for row in rows} <= {"casia_fasd", "msu_mfsd"}
    assert {row["spoof_dataset"] for row in rows} <= {"casia_fasd", "msu_mfsd"}
    assert all("siw" not in json.dumps(row).lower() for row in rows)
    assert plan["recipe_bank_identity"] == FROZEN_BANK_IDENTITY


def test_the_lock_is_written_only_after_a_successful_construction(
        package: Path, tmp_path: Path) -> None:
    """This fixture is deliberately too small for 896/224. `write_pair_plan`
    enforces the frozen counts and must leave nothing behind when it refuses."""
    output = tmp_path / "gpat_pairs"

    with pytest.raises(pair_plan_module.PairPlanError, match="pair counts"):
        pair_plan_module.write_pair_plan(package, preparation.recipe_bank_root(REPO), output)

    assert not (output / "PAIR_PLAN_LOCK.json").exists()
    assert not (output / "pair_manifest_train.parquet").exists()
    assert not (output / "pair_manifest_validation.parquet").exists()


# --- reuse is identity-aware -------------------------------------------------

def _plan_dir(root: Path, *, lock: dict[str, Any], manifests: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if manifests:
        for name in ("pair_manifest_train.parquet", "pair_manifest_validation.parquet"):
            (root / name).write_bytes(b"parquet")
    (root / "PAIR_PLAN_LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    return root


def test_a_plan_built_from_these_inputs_is_reusable(tmp_path: Path, package: Path) -> None:
    output = _plan_dir(tmp_path / "pairs", lock={
        "package_identity": "b" * 64, "recipe_bank_identity": FROZEN_BANK_IDENTITY})

    assert preparation._pair_plan_is_current(output, package, FROZEN_BANK_IDENTITY) is True


def test_a_plan_missing_its_manifests_is_not_reusable(tmp_path: Path,
                                                      package: Path) -> None:
    """A lock beside no manifests is an interrupted plan, not a finished one."""
    output = _plan_dir(tmp_path / "pairs", manifests=False, lock={
        "package_identity": "b" * 64, "recipe_bank_identity": FROZEN_BANK_IDENTITY})

    assert preparation._pair_plan_is_current(output, package, FROZEN_BANK_IDENTITY) is False


def test_a_plan_from_another_bank_is_not_reusable(tmp_path: Path, package: Path) -> None:
    output = _plan_dir(tmp_path / "pairs", lock={
        "package_identity": "b" * 64, "recipe_bank_identity": "c" * 64})

    assert preparation._pair_plan_is_current(output, package, FROZEN_BANK_IDENTITY) is False


def test_a_plan_from_another_package_is_not_reusable(tmp_path: Path,
                                                     package: Path) -> None:
    """The M3A-vs-M3B hazard: both are loadable, and only the identity tells them
    apart."""
    output = _plan_dir(tmp_path / "pairs", lock={
        "package_identity": "a" * 64, "recipe_bank_identity": FROZEN_BANK_IDENTITY})

    assert preparation._pair_plan_is_current(output, package, FROZEN_BANK_IDENTITY) is False


def test_no_plan_at_all_is_not_reusable(tmp_path: Path, package: Path) -> None:
    assert preparation._pair_plan_is_current(tmp_path / "absent", package,
                                             FROZEN_BANK_IDENTITY) is False


# --- the step itself ---------------------------------------------------------

def test_the_step_hands_the_builder_the_two_frozen_roots(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No stub chooses either root: both come from the production module."""
    from prism_fas.synthesis import pair_plan as module

    seen: dict[str, Path] = {}

    def spy(package_root, bank_root, output_root, **kwargs):
        seen.update(package_root=Path(package_root), bank_root=Path(bank_root),
                    output_root=Path(output_root))
        raise RuntimeError("stop after the arguments that matter")

    monkeypatch.setattr(module, "write_pair_plan", spy)
    with pytest.raises(RuntimeError, match="stop after"):
        preparation._step_pairs(REPO, resume=True)

    assert seen["bank_root"] == REPO / "assets" / "recipe_banks" / "prism_recipe_bank_m7_v1"
    assert seen["package_root"] == REPO / "data" / "packages" / "prism_data_v1_m3b"
    assert seen["output_root"] == REPO / "data" / "packages" / "gpat_pairs"
    assert "c3" not in seen["bank_root"].parts
    assert "prism_data_v1_m3a" not in seen["package_root"].as_posix()


def test_the_step_validates_the_bank_before_building_anything(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad bank must stop the step before it reaches the builder."""
    from prism_fas.synthesis import pair_plan as module

    monkeypatch.setattr(preparation, "recipe_bank_root",
                        lambda repo: REPO / "assets" / "recipe_banks" / "c3")
    monkeypatch.setattr(module, "write_pair_plan",
                        lambda *args, **kwargs: pytest.fail(
                            "the builder must not run against an unvalidated bank"))

    with pytest.raises(preparation.PreparationError) as raised:
        preparation._step_pairs(REPO, resume=True)

    assert raised.value.reason == preparation.RECIPE_BANK_INVALID
