"""The derived-data preparation orchestration, exercised for real against fixtures.

This module was the one part of the portable closure that was neither tested nor
run. `train.py` calls it with `dry_run=not plan.is_scientific`, so every laptop
rehearsal took the dry branch and the real orchestration — the branch that runs
first on the collaborator's GPU host — had never executed anywhere.

Two production defects were sitting in it, and both are the kind that only a real
call finds:

* `finalize_lock(root, config, report)` passed three arguments to a two-argument
  function, so every genuine M3A build died with a `TypeError`; and
* every builder was handed `configs/paths.local.yaml` directly. That file is
  Git-ignored, so a clone has none, and a copied folder carries one whose roots
  name the machine it left — the one-folder promise failed either way.

The tests here call the real `prepare()` entrypoint. What they replace is the
four canonical builders, and only those: the step order, the dependency
chaining, the resume decisions, the validation, the failure handling and the
report are all the shipping code. A stub that merely returned success would
prove nothing, so each one creates the tree its real counterpart would, which is
what lets the post-build completeness check mean something.

Nothing here touches a real dataset, a GPU, a provider or the target.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline import portable_paths, preparation  # noqa: E402

#: The order the module declares. Asserted against, never imported into, the
#: expectations below — a test that reads STEPS to check STEPS proves nothing.
EXPECTED_STEPS = ("m2_preprocess", "m3a_package", "m3b_priors", "gpat_pairs")


# --- the fixture project ------------------------------------------------------

def _write(path: Path, text: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A minimal portable folder: raw inputs and weights present, nothing derived."""
    repo = tmp_path / "PRISM_FAS_C_LLM_Project"
    for dataset in ("casia_fasd", "msu_mfsd", "siw_mv2"):
        _write(repo / "data" / "raw" / dataset / "README.txt", dataset)
    _write(repo / "weights" / "siglip2" / "model.bin", "weights")
    _write(repo / "assets" / "recipe_banks" / "c3" / "llm" / "bank.json", "{}")
    _write(repo / "configs" / "data" / "preprocess_m2.yaml", "version: 1\n")
    _write(repo / "configs" / "data" / "package_m3a.yaml", "version: 1\n")
    _write(repo / "configs" / "models" / "model_priors.yaml", "version: 1\n")
    return repo


class Builders:
    """Stand-ins for the four canonical builders, recording how they were called.

    Each one creates the tree its real counterpart creates. That is deliberate:
    `prepare()` re-reads the filesystem after the loop and fails when a tree is
    still absent, so a stub that only returned a dict would turn that check into
    a permanent failure and hide the behaviour under test.
    """

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.calls: list[dict[str, Any]] = []
        self.fail_at: str | None = None
        self.build_nothing_at: str | None = None
        self.validation_passes = True

    # -- helpers
    @property
    def order(self) -> list[str]:
        return [call["step"] for call in self.calls]

    def _record(self, step: str, **kwargs: Any) -> None:
        if self.fail_at == step:
            raise RuntimeError(f"stub builder failure at {step}")
        self.calls.append({"step": step, **kwargs})

    def _make(self, step: str, path: Path, lock: str | None = None) -> None:
        if self.build_nothing_at == step:
            return
        path.mkdir(parents=True, exist_ok=True)
        _write(path / "content.json", json.dumps({"step": step}))
        if lock:
            _write(path / lock, json.dumps({"status": "validated"}))

    # -- the four builders
    def m2_run(self, dataset: str, config_path: Path, preprocess_path: Path,
               limit_records: int = 3, dry_run: bool = False,
               resume: bool = False, force: bool = False) -> dict[str, Any]:
        self._record("m2_preprocess", dataset=dataset, config_path=Path(config_path),
                     preprocess_path=Path(preprocess_path), limit_records=limit_records,
                     dry_run=dry_run, resume=resume, force=force)
        self._make("m2_preprocess", self.repo / "data" / "processed" / dataset)
        return {"records": limit_records}

    def build_package(self, input_root: Path, package_root: Path, config: Any,
                      *, resume: bool = True, **kwargs: Any) -> dict[str, Any]:
        self._record("m3a_package", input_root=Path(input_root),
                     package_root=Path(package_root), resume=resume)
        self._make("m3a_package", Path(package_root), lock="PACKAGE_LOCK.json")
        return {"samples": 36}

    def validate_package(self, package_root: Path, *,
                         require_validated_status: bool = True,
                         **kwargs: Any) -> dict[str, Any]:
        return {"passed": self.validation_passes, "checks": [], "package_root": str(package_root)}

    def finalize_lock(self, package_root: Path, report: dict) -> dict:
        # Two parameters, exactly as the canonical builder declares them. The
        # regression this guards is a call with three.
        self._record("m3a_finalize", package_root=Path(package_root),
                     report_passed=report.get("passed"))
        return {"status": "validated"}

    def load_package_config(self, path: Path) -> Any:
        return {"config_path": str(path)}

    def build_m3b(self, input_package: Path, output_package: Path, model_config: Path,
                  *, weight_root: Path, resume: bool = True, **kwargs: Any) -> dict[str, Any]:
        self._record("m3b_priors", input_package=Path(input_package),
                     output_package=Path(output_package), model_config=Path(model_config),
                     weight_root=Path(weight_root), resume=resume)
        self._make("m3b_priors", Path(output_package), lock="PACKAGE_LOCK.json")
        return {"samples": 36}

    def write_pair_plan(self, package_root: Path, bank_root: Path, output_root: Path,
                        **kwargs: Any) -> dict[str, Any]:
        self._record("gpat_pairs", package_root=Path(package_root),
                     bank_root=Path(bank_root), output_root=Path(output_root))
        self._make("gpat_pairs", Path(output_root), lock="PAIR_PLAN_LOCK.json")
        return {}

    def pair_plan_identity(self, output_root: Path) -> str:
        return "pairplan-" + Path(output_root).name


@pytest.fixture
def builders(project: Path, monkeypatch: pytest.MonkeyPatch) -> Builders:
    """Patch the real builder modules, so a renamed builder fails these tests."""
    from prism_fas.data import m2_runner, package
    from prism_fas.data.package import m3b
    from prism_fas.synthesis import pair_plan

    stub = Builders(project)
    monkeypatch.setattr(m2_runner, "run", stub.m2_run)
    monkeypatch.setattr(package, "build_package", stub.build_package)
    monkeypatch.setattr(package, "validate_package", stub.validate_package)
    monkeypatch.setattr(package, "finalize_lock", stub.finalize_lock)
    monkeypatch.setattr(package, "load_package_config", stub.load_package_config)
    monkeypatch.setattr(m3b, "build_m3b_package", stub.build_m3b)
    monkeypatch.setattr(pair_plan, "write_pair_plan", stub.write_pair_plan)
    monkeypatch.setattr(pair_plan, "pair_plan_identity", stub.pair_plan_identity)
    # The record count comes from the real adapter over a real dataset root, which
    # a fixture has no copy of. A distinctive number is used so the assertion that
    # it reaches the builder cannot pass by coincidence.
    monkeypatch.setattr(preparation, "_record_count", lambda repo, dataset: 617)
    return stub


# --- A. the implementation contract ------------------------------------------

def test_the_declared_step_order_is_the_one_the_report_promised() -> None:
    assert preparation.STEPS == EXPECTED_STEPS


def test_train_py_calls_the_public_entrypoint_with_dry_run_tied_to_the_intent() -> None:
    """The wiring under test, read from train.py rather than assumed."""
    source = (REPO / "train.py").read_text(encoding="utf-8")
    assert "preparation.prepare(" in source
    assert "dry_run=not plan.is_scientific" in source, (
        "train.py must tie the dry-run branch to the scientific intent; a literal "
        "would make the rehearsal and the GPU host take the same branch")


# --- B. dry run ---------------------------------------------------------------

def test_dry_run_invokes_no_builder_and_writes_no_tree(project: Path,
                                                       builders: Builders) -> None:
    report = preparation.prepare(project, dry_run=True)

    assert report["outcome"] == "WOULD_BUILD"
    assert builders.calls == []
    assert not (project / "data" / "processed").exists()
    assert not (project / "data" / "packages").exists()


def test_dry_run_never_reports_preparation_complete(project: Path,
                                                    builders: Builders) -> None:
    report = preparation.prepare(project, dry_run=True)
    assert report["outcome"] != "PREPARED"
    assert report["scientific_eligible"] is False


def test_dry_run_names_what_it_would_build(project: Path, builders: Builders) -> None:
    report = preparation.prepare(project, dry_run=True)
    assert set(report["needed"]["missing_derived"]) == {"processed", "packages",
                                                        "gpat_pairs"}


# --- C. the non-dry fresh build ----------------------------------------------

def test_non_dry_run_builds_every_step_in_the_declared_order(project: Path,
                                                             builders: Builders) -> None:
    report = preparation.prepare(project, dry_run=False)

    assert report["outcome"] == "PREPARED"
    assert [item["step"] for item in report["steps"]] == list(EXPECTED_STEPS)

    # m2 is called once per source dataset, so the raw call list repeats it.
    # Collapsing runs of the same step is what makes this an order assertion
    # rather than a call-count one.
    build_order: list[str] = []
    for step in builders.order:
        if step in EXPECTED_STEPS and (not build_order or build_order[-1] != step):
            build_order.append(step)
    assert build_order == list(EXPECTED_STEPS)


def test_each_step_consumes_the_previous_step_output(project: Path,
                                                     builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    calls = {call["step"]: call for call in builders.calls}

    m3a_root = project / "data" / "packages" / "prism_data_v1_m3a"
    assert calls["m3a_package"]["input_root"] == project / "data" / "processed"
    assert calls["m3a_package"]["package_root"] == m3a_root
    assert calls["m3b_priors"]["input_package"] == m3a_root
    assert calls["m3b_priors"]["output_package"] == project / "data" / "packages" / "prism_data_v1_m3b"
    assert calls["gpat_pairs"]["package_root"] == m3a_root
    assert calls["gpat_pairs"]["bank_root"] == project / "assets" / "recipe_banks" / "c3"


def test_resume_is_propagated_to_every_builder(project: Path,
                                               builders: Builders) -> None:
    preparation.prepare(project, resume=True, dry_run=False)
    resumable = [call for call in builders.calls if "resume" in call]
    assert resumable, "no builder received a resume flag"
    assert all(call["resume"] is True for call in resumable)

    builders.calls.clear()
    for tree in ("processed", "packages"):
        _rmtree(project / "data" / tree)
    preparation.prepare(project, resume=False, dry_run=False)
    resumable = [call for call in builders.calls if "resume" in call]
    assert all(call["resume"] is False for call in resumable)


def test_the_scientific_corpus_is_not_truncated_to_the_builder_default(
        project: Path, builders: Builders) -> None:
    """`m2_runner.run` defaults `limit_records=3`; preparation must override it."""
    preparation.prepare(project, dry_run=False)
    m2_calls = [call for call in builders.calls if call["step"] == "m2_preprocess"]

    assert {call["dataset"] for call in m2_calls} == {"casia_fasd", "msu_mfsd"}
    for call in m2_calls:
        assert call["limit_records"] == 617, "the record count must reach the builder"
        assert call["limit_records"] != 3, "the smoke default would truncate the corpus"
        assert call["dry_run"] is False
        assert call["force"] is False


def test_the_target_dataset_is_never_preprocessed(project: Path,
                                                  builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    datasets = {call["dataset"] for call in builders.calls if call["step"] == "m2_preprocess"}
    assert "siw_mv2" not in datasets


# --- D. resume and idempotency ------------------------------------------------

def _rmtree(path: Path) -> None:
    import shutil

    if path.exists():
        shutil.rmtree(path)


def test_scenario_b_everything_present_is_nothing_to_do(project: Path,
                                                        builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    builders.calls.clear()

    report = preparation.prepare(project, dry_run=False)
    assert report["outcome"] == "NOTHING_TO_DO"
    assert builders.calls == [], "a complete tree must not be rebuilt"


def test_scenario_c_partial_completion_reuses_what_is_valid(project: Path,
                                                            builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    _rmtree(project / "data" / "packages" / "gpat_pairs")
    builders.calls.clear()

    report = preparation.prepare(project, dry_run=False)
    actions = {item["step"]: item["action"] for item in report["steps"]}

    assert actions["m2_preprocess"] == "REUSED_VALID"
    assert actions["m3a_package"] == "REUSED_VALID"
    assert actions["m3b_priors"] == "REUSED_VALID"
    assert actions["gpat_pairs"] == "BUILT"
    assert builders.order == ["gpat_pairs"], "only the missing step may run"


def test_scenario_d_an_interrupted_step_is_rebuilt(project: Path,
                                                   builders: Builders) -> None:
    """A package directory without its lock is interrupted work, not a valid tree."""
    preparation.prepare(project, dry_run=False)
    (project / "data" / "packages" / "prism_data_v1_m3a" / "PACKAGE_LOCK.json").unlink()
    builders.calls.clear()

    report = preparation.prepare(project, dry_run=False)
    actions = {item["step"]: item["action"] for item in report["steps"]}
    assert actions["m3a_package"] == "BUILT"


def test_scenario_e_a_stale_package_that_fails_validation_is_rebuilt(
        project: Path, builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    _rmtree(project / "data" / "packages" / "gpat_pairs")
    builders.calls.clear()
    builders.validation_passes = False

    # The M3A tree is present and locked, but its validator now refuses it. The
    # contract is to rebuild and revalidate, and to fail when validation still
    # refuses — never to accept the stale tree.
    builders.validation_passes = False
    with pytest.raises(preparation.PreparationError) as caught:
        preparation.prepare(project, dry_run=False)
    assert caught.value.reason == preparation.PREPARATION_FAILED


def test_scenario_f_a_failing_builder_raises_and_names_the_step(
        project: Path, builders: Builders) -> None:
    builders.fail_at = "m3a_package"

    with pytest.raises(preparation.PreparationError) as caught:
        preparation.prepare(project, dry_run=False)

    assert caught.value.reason == preparation.PREPARATION_FAILED
    assert "m3a_package" in str(caught.value)
    assert caught.value.detail["step"] == "m3a_package"
    assert caught.value.detail["completed"] == ["m2_preprocess"]


def test_a_rerun_after_completion_is_byte_identical_except_for_timing(
        project: Path, builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    first = preparation.prepare(project, dry_run=False)
    second = preparation.prepare(project, dry_run=False)

    for report in (first, second):
        report.pop("elapsed_seconds")
    assert first == second


# --- E. failure atomicity -----------------------------------------------------

def test_a_failure_stops_every_later_step(project: Path, builders: Builders) -> None:
    builders.fail_at = "m3a_package"
    with pytest.raises(preparation.PreparationError):
        preparation.prepare(project, dry_run=False)

    assert "m3b_priors" not in builders.order
    assert "gpat_pairs" not in builders.order
    assert not (project / "data" / "packages" / "gpat_pairs").exists()


def test_a_failed_preparation_is_resumable_from_the_failed_step(
        project: Path, builders: Builders) -> None:
    builders.fail_at = "m3b_priors"
    with pytest.raises(preparation.PreparationError):
        preparation.prepare(project, dry_run=False)

    builders.fail_at = None
    builders.calls.clear()
    report = preparation.prepare(project, dry_run=False)

    assert report["outcome"] == "PREPARED"
    assert builders.order == ["m3b_priors", "gpat_pairs"], (
        "the completed steps must be reused, not rebuilt")


def test_no_silent_success_when_a_tree_is_still_absent(project: Path,
                                                       builders: Builders) -> None:
    """A builder that returns success without producing its tree is a failure."""
    builders.build_nothing_at = "gpat_pairs"

    with pytest.raises(preparation.PreparationError) as caught:
        preparation.prepare(project, dry_run=False)

    assert "gpat_pairs" in caught.value.detail["still_missing"]


# --- F. missing raw data ------------------------------------------------------

@pytest.mark.parametrize("absent", ["casia_fasd", "msu_mfsd"])
def test_missing_raw_data_blocks_before_any_builder_runs(project: Path,
                                                         builders: Builders,
                                                         absent: str) -> None:
    _rmtree(project / "data" / "raw" / absent)

    report = preparation.prepare(project, dry_run=False)

    assert report["outcome"] == "BLOCKED"
    assert report["reason_code"] == preparation.MISSING_RAW_DATA
    assert builders.calls == [], "nothing may be built without its raw source"


def test_a_blocked_preparation_fabricates_nothing(project: Path,
                                                  builders: Builders) -> None:
    _rmtree(project / "data" / "raw" / "casia_fasd")
    preparation.prepare(project, dry_run=False)

    assert not (project / "data" / "processed").exists()
    assert not (project / "data" / "packages").exists()


def test_the_absent_target_dataset_does_not_block_preparation(project: Path,
                                                              builders: Builders) -> None:
    """SiW is the held-out target and derives nothing; its absence is irrelevant here."""
    _rmtree(project / "data" / "raw" / "siw_mv2")

    report = preparation.prepare(project, dry_run=False)
    assert report["outcome"] == "PREPARED"


# --- G. the paths config the builders are handed ------------------------------

def test_a_folder_with_no_paths_config_gets_one_describing_itself(
        project: Path, builders: Builders) -> None:
    """A clone carries no paths.local.yaml; preparation must not need the operator."""
    assert not (project / "configs" / "paths.local.yaml").exists()

    report = preparation.prepare(project, dry_run=False)

    assert report["outcome"] == "PREPARED"
    assert report["paths_config"]["action"] == portable_paths.WRITTEN
    written = yaml.safe_load(
        (project / "configs" / "paths.local.yaml").read_text(encoding="utf-8"))
    assert Path(written["project_root"]) == project
    assert Path(written["processed_root"]) == project / "data" / "processed"


def test_a_config_naming_another_machine_is_replaced(project: Path,
                                                     builders: Builders) -> None:
    """The folder-copy case: the config travelled, its roots did not."""
    _write(project / "configs" / "paths.local.yaml", yaml.safe_dump({
        "workspace_root": "D:/elsewhere",
        "project_root": "D:/elsewhere/PRISM_FAS_C_LLM_Project",
        "raw_datasets": {"casia_fasd": "D:/elsewhere/casia",
                         "msu_mfsd": "D:/elsewhere/msu",
                         "siw_mv2": "D:/elsewhere/siw"},
        "model_cache": "D:/elsewhere/model_cache",
        "work_root": "D:/elsewhere/work",
        "processed_root": "D:/elsewhere/processed",
        "package_root": "D:/elsewhere/packages",
        "runs_root": "D:/elsewhere/runs",
        "reports_root": "D:/elsewhere/reports"}))

    report = preparation.prepare(project, dry_run=False)

    assert report["paths_config"]["action"] == portable_paths.REWRITTEN
    written = yaml.safe_load(
        (project / "configs" / "paths.local.yaml").read_text(encoding="utf-8"))
    assert Path(written["project_root"]) == project
    assert "elsewhere" not in json.dumps(written), (
        "no path from the machine the folder left may survive")


def test_a_config_that_already_names_this_folder_is_left_alone(project: Path) -> None:
    authored = _write(project / "configs" / "paths.local.yaml", yaml.safe_dump({
        "workspace_root": str(project.parent),
        "project_root": str(project),
        "raw_datasets": {"casia_fasd": "/mnt/big/casia",
                         "msu_mfsd": "/mnt/big/msu",
                         "siw_mv2": "/mnt/big/siw"},
        "model_cache": "/mnt/big/models",
        "work_root": str(project / "data" / "work"),
        "processed_root": str(project / "data" / "processed"),
        "package_root": str(project / "data" / "packages"),
        "runs_root": str(project / "runs"),
        "reports_root": str(project / "reports")}))
    before = authored.read_text(encoding="utf-8")

    result = portable_paths.ensure_local_paths(project)

    assert result["action"] == portable_paths.REUSED
    assert authored.read_text(encoding="utf-8") == before, (
        "an operator's deliberate external roots must survive")


def test_the_builders_are_handed_the_generated_config(project: Path,
                                                      builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    m2_calls = [call for call in builders.calls if call["step"] == "m2_preprocess"]
    assert all(call["config_path"] == project / "configs" / "paths.local.yaml"
               for call in m2_calls)


# --- H. relocation ------------------------------------------------------------

def _prepare_at(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build the same fixture project under a different absolute root."""
    repo = root / "PRISM_FAS_C_LLM_Project"
    for dataset in ("casia_fasd", "msu_mfsd", "siw_mv2"):
        _write(repo / "data" / "raw" / dataset / "README.txt", dataset)
    _write(repo / "weights" / "siglip2" / "model.bin", "weights")
    _write(repo / "assets" / "recipe_banks" / "c3" / "llm" / "bank.json", "{}")

    from prism_fas.data import m2_runner, package
    from prism_fas.data.package import m3b
    from prism_fas.synthesis import pair_plan

    stub = Builders(repo)
    monkeypatch.setattr(m2_runner, "run", stub.m2_run)
    monkeypatch.setattr(package, "build_package", stub.build_package)
    monkeypatch.setattr(package, "validate_package", stub.validate_package)
    monkeypatch.setattr(package, "finalize_lock", stub.finalize_lock)
    monkeypatch.setattr(package, "load_package_config", stub.load_package_config)
    monkeypatch.setattr(m3b, "build_m3b_package", stub.build_m3b)
    monkeypatch.setattr(pair_plan, "write_pair_plan", stub.write_pair_plan)
    monkeypatch.setattr(pair_plan, "pair_plan_identity", stub.pair_plan_identity)
    monkeypatch.setattr(preparation, "_record_count", lambda repo, dataset: 617)
    return {"repo": repo, "report": preparation.prepare(repo, dry_run=False),
            "builders": stub}


def test_the_same_fixture_under_two_roots_prepares_identically(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _prepare_at(tmp_path / "root_one", monkeypatch)
    second = _prepare_at(tmp_path / "root_two", monkeypatch)

    def comparable(report: dict[str, Any]) -> dict[str, Any]:
        return {"outcome": report["outcome"],
                "steps": [{"step": item["step"], "action": item["action"]}
                          for item in report["steps"]],
                "missing": sorted(report["needed"]["missing_derived"])}

    assert comparable(first["report"]) == comparable(second["report"])


def test_no_absolute_host_path_is_recorded_as_identity_relevant(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepare_at(tmp_path / "root_three", monkeypatch)
    resolution = portable_paths.resolve(prepared["repo"]).as_dict()

    for group in ("raw_datasets", "derived"):
        for root in resolution[group].values():
            assert root["identity_relevant"] is False


def test_relocation_moves_the_output_roots_with_the_folder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepare_at(tmp_path / "root_four", monkeypatch)
    repo = prepared["repo"]

    for call in prepared["builders"].calls:
        for key, value in call.items():
            if isinstance(value, Path) and key != "bank_root":
                assert repo in value.parents or value == repo, (
                    f"{key}={value} escaped the relocated folder")


# --- I. the one-folder auto-build contract ------------------------------------

def test_raw_present_and_derived_absent_triggers_the_auto_build(
        project: Path, builders: Builders) -> None:
    needed = preparation.what_is_needed(project)

    assert needed["can_rebuild"] is True
    assert needed["blocked"] is False
    assert needed["nothing_to_do"] is False

    report = preparation.prepare(project, dry_run=False)
    assert report["outcome"] == "PREPARED"


def test_manual_derived_data_command_is_not_required(project: Path,
                                                     builders: Builders) -> None:
    """The whole contract: one call, from raw to every derived tree."""
    preparation.prepare(project, dry_run=False)

    for tree in ("processed", "packages"):
        assert (project / "data" / tree).is_dir()
    assert (project / "data" / "packages" / "gpat_pairs" / "PAIR_PLAN_LOCK.json").is_file()
    assert preparation.what_is_needed(project)["nothing_to_do"] is True


# --- J. the target firewall ---------------------------------------------------

def test_preparation_needs_no_target_labels(project: Path, builders: Builders) -> None:
    labels = project / "data" / "evaluation_only"
    preparation.prepare(project, dry_run=False)
    assert not labels.exists(), "preparation must not create or require a label tree"


def test_no_builder_is_pointed_at_the_target_raw_root(project: Path,
                                                      builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    target_root = project / "data" / "raw" / "siw_mv2"

    for call in builders.calls:
        for value in call.values():
            if isinstance(value, Path):
                assert target_root not in value.parents and value != target_root


def test_preparation_does_not_read_inside_the_target_root(project: Path,
                                                          builders: Builders,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Statting the root is portability; opening a file inside it is not."""
    target_root = (project / "data" / "raw" / "siw_mv2").resolve()
    opened: list[str] = []
    real_open = Path.open

    def watched(self: Path, *args: Any, **kwargs: Any):
        resolved = Path(self).resolve()
        if target_root in resolved.parents:
            opened.append(str(resolved))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", watched)
    preparation.prepare(project, dry_run=False)

    assert opened == []


# --- K. completion markers: an interrupted tree is not a finished one ---------

def test_a_package_without_its_lock_is_not_reported_as_nothing_to_do(
        project: Path, builders: Builders) -> None:
    """The defect this guards: a half-written package looked complete.

    Presence was `the directory has something in it`, which is equally true of a
    build that died mid-write. Preparation reported NOTHING_TO_DO and C4 would
    have trained against the partial tree.
    """
    preparation.prepare(project, dry_run=False)
    (project / "data" / "packages" / "prism_data_v1_m3a" / "PACKAGE_LOCK.json").unlink()

    needed = preparation.what_is_needed(project)
    assert needed["nothing_to_do"] is False
    assert "packages" in needed["missing_derived"]


def test_a_pair_plan_without_its_lock_is_rebuilt(project: Path,
                                                 builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    (project / "data" / "packages" / "gpat_pairs" / "PAIR_PLAN_LOCK.json").unlink()
    builders.calls.clear()

    report = preparation.prepare(project, dry_run=False)
    actions = {item["step"]: item["action"] for item in report["steps"]}
    assert actions["gpat_pairs"] == "BUILT"
    assert builders.order == ["gpat_pairs"]


def test_a_finished_tree_is_still_nothing_to_do(project: Path,
                                                builders: Builders) -> None:
    """The marker check must not make a complete tree look unfinished."""
    preparation.prepare(project, dry_run=False)
    assert preparation.what_is_needed(project)["nothing_to_do"] is True


def test_an_absent_derived_root_is_missing_rather_than_incomplete(
        project: Path) -> None:
    """The `root is not a directory` branch: absent and half-built are different."""
    assert preparation._incomplete(project, "packages") is False
    needed = preparation.what_is_needed(project)
    assert "packages" in needed["missing_derived"]


def test_unresolvable_weights_block_the_prior_package(project: Path,
                                                      builders: Builders) -> None:
    """M3B priors come from the frozen towers; there is no substitute for them."""
    _rmtree(project / "weights")

    with pytest.raises(preparation.PreparationError) as caught:
        preparation.prepare(project, dry_run=False)

    assert caught.value.reason == preparation.MISSING_RAW_DATA
    assert "weights" in str(caught.value)
    assert "gpat_pairs" not in builders.order


def test_finalize_lock_is_called_with_the_two_arguments_it_declares(
        project: Path, builders: Builders) -> None:
    """Regression: this used to pass three, which was a TypeError on every build.

    The stub mirrors the canonical signature exactly, so a third argument fails
    here rather than on the collaborator's machine.
    """
    import inspect

    from prism_fas.data.package import builder as package_builder

    signature = inspect.signature(package_builder.finalize_lock)
    assert list(signature.parameters) == ["package_root", "report"]

    preparation.prepare(project, dry_run=False)
    finalize = [call for call in builders.calls if call["step"] == "m3a_finalize"]
    assert len(finalize) == 1
    assert finalize[0]["report_passed"] is True


def test_record_count_matches_the_canonical_adapter_contract() -> None:
    """`_record_count` is stubbed in these tests, so its call shape is asserted here.

    It has no branches — it delegates and returns a length — so the only way it
    can fail is the way `finalize_lock` did: a signature that moved underneath it.
    Every name and parameter it binds is checked against the real modules.
    """
    import inspect

    from prism_fas.config.models import load_paths
    from prism_fas.data.m2_runner import DatasetDefinition, adapter_for

    assert list(inspect.signature(load_paths).parameters) == ["path"]
    assert list(inspect.signature(adapter_for).parameters) == ["definition", "root"]
    assert hasattr(DatasetDefinition, "model_validate")
    assert hasattr(adapter_for(
        DatasetDefinition(dataset="casia_fasd", adapter_version="v1"),
        Path(".")), "records")

    source = inspect.getsource(preparation._record_count)
    assert "adapter_for(" in source and ".records()" in source
    for dataset in ("casia_fasd", "msu_mfsd"):
        assert (REPO / "configs" / "data" / f"{dataset}.yaml").is_file(), (
            "the config _record_count reads must exist in the shipped folder")


# --- L. train.py wiring: the scientific path takes the non-dry branch ---------

class _Plan:
    """The fields `_zero_argument` reads from a RunPlan."""

    def __init__(self, *, scientific: bool) -> None:
        self.intent = "GPU_SCIENTIFIC_FULL" if scientific else "CPU_FULL_REHEARSAL"
        self.profile_name = "full" if scientific else "rehearsal"
        self.is_scientific = scientific
        self.ready = True
        self.first_stage = "C4" if scientific else "C0"
        self.last_stage = "C13"


@pytest.fixture
def train_module(project: Path, monkeypatch: pytest.MonkeyPatch):
    """`train.py`, with its repo redirected and every side effect stubbed."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import bootstrap as boot
    import train as train_module

    from prism_fas.pipeline import gpu_preflight, orchestrator, runner

    calls: dict[str, Any] = {"order": [], "prepare_kwargs": None, "run_kwargs": None}

    monkeypatch.setattr(train_module, "REPO", project)
    monkeypatch.setattr(train_module, "_git_identity", lambda: {"commit": "test"})
    monkeypatch.setattr(boot, "ensure_environment", lambda **kwargs: {"gpu": {}})
    monkeypatch.setattr(runner, "preflight_summary",
                        lambda *args, **kwargs: "preflight")

    def fake_preflight(repo, *, strict):
        calls["order"].append("gpu_preflight")
        calls["preflight_strict"] = strict
        return {"applicable": False, "probes_run": 0, "elapsed_seconds": 0.0}

    def fake_prepare(repo, **kwargs):
        calls["order"].append("preparation")
        calls["prepare_kwargs"] = kwargs
        return {"outcome": "NOTHING_TO_DO", "summary": "", "steps": [],
                "reason_code": None}

    def fake_run(**kwargs):
        calls["order"].append("orchestrator")
        calls["run_kwargs"] = kwargs
        return _Result()

    class _Result:
        outcome = "PASS"
        stages = []

    monkeypatch.setattr(gpu_preflight, "run_preflight", fake_preflight)
    monkeypatch.setattr(gpu_preflight, "write_report", lambda repo, report: project)
    monkeypatch.setattr(preparation, "prepare", fake_prepare)
    monkeypatch.setattr(preparation, "write_report", lambda repo, report: project)
    monkeypatch.setattr(orchestrator, "run", fake_run)
    monkeypatch.setattr(train_module, "_print_stage_table", lambda result: None)
    monkeypatch.setattr(train_module, "_summarize", lambda *args, **kwargs: None,
                        raising=False)
    return train_module, calls


def _zero_argument(train_module, scientific: bool, monkeypatch: pytest.MonkeyPatch):
    import argparse

    from prism_fas.pipeline import runner

    monkeypatch.setattr(runner, "resolve",
                        lambda repo, environment: _Plan(scientific=scientific))
    args = argparse.Namespace(preflight_only=False)
    return train_module._zero_argument(args)


def test_the_scientific_intent_takes_the_non_dry_preparation_branch(
        train_module, monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls = train_module
    _zero_argument(module, scientific=True, monkeypatch=monkeypatch)

    assert calls["prepare_kwargs"]["dry_run"] is False, (
        "GPU_SCIENTIFIC_FULL must run the real preparation, not report on it")
    assert calls["prepare_kwargs"]["resume"] is True


def test_preparation_runs_before_the_orchestrator_reaches_c4(
        train_module, monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls = train_module
    _zero_argument(module, scientific=True, monkeypatch=monkeypatch)

    assert calls["order"] == ["gpu_preflight", "preparation", "orchestrator"]
    assert calls["run_kwargs"]["first_stage"] == "C4"


def test_the_rehearsal_intent_keeps_the_safe_dry_branch(
        train_module, monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls = train_module
    _zero_argument(module, scientific=False, monkeypatch=monkeypatch)

    assert calls["prepare_kwargs"]["dry_run"] is True, (
        "a CPU rehearsal must not spend hours building a tree it never reads")
    assert calls["run_kwargs"]["profile_name"] == "rehearsal"


def test_a_blocked_preparation_stops_before_the_orchestrator(
        train_module, monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls = train_module

    def blocked(repo, **kwargs):
        calls["order"].append("preparation")
        return {"outcome": "BLOCKED", "reason_code": preparation.MISSING_RAW_DATA,
                "summary": "raw data is absent", "steps": []}

    monkeypatch.setattr(preparation, "prepare", blocked)
    exit_code = _zero_argument(module, scientific=True, monkeypatch=monkeypatch)

    assert "orchestrator" not in calls["order"], (
        "no scientific stage may start after preparation blocked")
    assert exit_code == module.EXIT_BLOCKED


def test_a_preparation_error_stops_before_the_orchestrator(
        train_module, monkeypatch: pytest.MonkeyPatch) -> None:
    module, calls = train_module

    def failing(repo, **kwargs):
        calls["order"].append("preparation")
        raise preparation.PreparationError(
            preparation.PREPARATION_FAILED, "the stub builder failed")

    monkeypatch.setattr(preparation, "prepare", failing)
    exit_code = _zero_argument(module, scientific=True, monkeypatch=monkeypatch)

    assert "orchestrator" not in calls["order"]
    assert exit_code == module.EXIT_BLOCKED


# --- M. the report is never scientific evidence -------------------------------

def test_every_report_declares_itself_scientifically_ineligible(
        project: Path, builders: Builders) -> None:
    reports = [preparation.prepare(project, dry_run=True)]
    reports.append(preparation.prepare(project, dry_run=False))
    reports.append(preparation.prepare(project, dry_run=False))

    # BLOCKED needs a folder with nothing derived yet. Removing the raw source
    # from a fully prepared folder is correctly NOTHING_TO_DO: with every tree
    # already built there is no longer anything to derive from it.
    _rmtree(project / "data")
    for dataset in ("msu_mfsd", "siw_mv2"):
        _write(project / "data" / "raw" / dataset / "README.txt", dataset)
    reports.append(preparation.prepare(project, dry_run=False))

    assert {report["outcome"] for report in reports} == {
        "WOULD_BUILD", "PREPARED", "NOTHING_TO_DO", "BLOCKED"}
    for report in reports:
        assert report["scientific_eligible"] is False


def test_the_report_is_written_outside_every_scientific_namespace(
        project: Path, builders: Builders) -> None:
    report = preparation.prepare(project, dry_run=False)
    path = preparation.write_report(project, report)

    assert path == project / "reports" / "preflight" / "DERIVED_DATA_PREPARATION.json"
    assert json.loads(path.read_text(encoding="utf-8"))["scientific_eligible"] is False


def test_a_fixture_preparation_cannot_become_a_scientific_ancestor(
        project: Path, builders: Builders) -> None:
    """Nothing preparation writes is named by a C4-C13 scientific required input."""
    from prism_fas.pipeline.adapters import registry as adapter_registry

    preparation.prepare(project, dry_run=False)
    produced = {path.relative_to(project).as_posix()
                for path in project.rglob("*") if path.is_file()}

    registry = adapter_registry.build_registry()
    inherited = [item.relative_path
                 for stage in ("C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
                               "C12", "C13")
                 for item in registry[stage].required_inputs()
                 if item.relative_path.startswith(("reports/", "runs/"))]

    assert inherited, "no stage declares an inherited artifact"
    for relative in inherited:
        assert relative.startswith(("reports/full", "runs/full"))
        assert relative not in produced
