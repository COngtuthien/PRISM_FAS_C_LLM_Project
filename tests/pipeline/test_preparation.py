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
    import shutil

    repo = tmp_path / "PRISM_FAS_C_LLM_Project"
    for dataset in ("casia_fasd", "msu_mfsd", "siw_mv2"):
        _write(repo / "data" / "raw" / dataset / "README.txt", dataset)
    _write(repo / "weights" / "siglip2" / "model.bin", "weights")
    _write(repo / "assets" / "recipe_banks" / "c3" / "llm" / "bank.json", "{}")
    _write(repo / "configs" / "data" / "package_m3a.yaml", "version: 1\n")
    _write(repo / "configs" / "models" / "model_priors.yaml", "version: 1\n")
    # The M2 configs are the real ones. `m2_output_root` derives the canonical
    # namespace from `preprocessing_version` and `config_hash`, so a placeholder
    # here would leave the fixture free to invent its own layout — which is the
    # class of mistake this whole module now exists to prevent.
    (repo / "configs" / "data").mkdir(parents=True, exist_ok=True)
    for name in ("preprocess_m2.yaml", "m2_run_profiles.yaml"):
        shutil.copyfile(REPO / "configs" / "data" / name, repo / "configs" / "data" / name)
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
        self.m2_complete = False

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
    #
    # M2 is stubbed one level higher than the other three: at `_step_m2` rather
    # than at the runner underneath it, because a fixture cannot cheaply produce
    # crops whose SHA-256s satisfy the canonical validator. What it must never do
    # is choose where M2 output lands — the stub asks `m2_output_root` for that,
    # so a consumer that reads anywhere else still fails here. The unstubbed
    # producer/consumer contract lives in test_m2_m3a_contract.py.
    def step_m2(self, repo: Path, *, resume: bool) -> Any:
        root = preparation.m2_output_root(repo)
        if self.m2_complete:
            # The real step's own first branch: a complete, validated tree is
            # reused and nothing is reprocessed.
            return preparation.StepOutcome(
                "m2_preprocess", "REUSED_VALID", "stub reused a complete M2 tree",
                {"m2_root": root.as_posix()})
        self._record("m2_preprocess", resume=resume, m2_root=root,
                     config_path=preparation._paths_config(repo),
                     datasets=list(preparation.SOURCE_DATASETS))
        if self.build_nothing_at == "m2_preprocess":
            return preparation.StepOutcome("m2_preprocess", "BUILT", "stub built nothing")
        # Completion is recorded in memory rather than as fixture parquet files:
        # the canonical M2 root nests a 64-character config hash, and on Windows
        # a pytest tmp path plus that hash exceeds MAX_PATH. The location itself
        # is still asserted, from `m2_root` above.
        self.m2_complete = True
        return preparation.StepOutcome(
            "m2_preprocess", "BUILT", "stub preprocessed the source corpora",
            {"m2_root": root.as_posix(),
             "per_dataset": {name: {"records_total": 617, "records_walked": 617}
                             for name in preparation.SOURCE_DATASETS}})

    def m2_status(self, repo: Path, *, deep: bool = False,
                  require_marker: bool = True) -> dict[str, Any]:
        root = preparation.m2_output_root(repo)
        complete = self.m2_complete
        return {"profile": preparation.M2_RUN_PROFILE, "root": root.as_posix(),
                "manifests_root": (root / "manifests").as_posix(),
                "manifests_present": {name: complete for name in preparation.M2_MANIFESTS},
                "counts": {name: (1 if complete and name.startswith("source") else 0)
                           for name in preparation.M2_MANIFESTS},
                "outstanding_records": {}, "marker": None,
                "validation": {"passed": True} if complete and deep else None,
                "complete": complete,
                "reason": None if complete else "MANIFESTS_ABSENT"}

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
    from prism_fas.data import package
    from prism_fas.data.package import m3b
    from prism_fas.synthesis import pair_plan

    stub = Builders(project)
    monkeypatch.setattr(preparation, "_step_m2", stub.step_m2)
    monkeypatch.setattr(preparation, "m2_status", stub.m2_status)
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
    monkeypatch.setattr(preparation, "_records", lambda repo, dataset: [])
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
    assert calls["m3a_package"]["input_root"] == preparation.m2_output_root(project), (
        "M3A must consume the canonical M2 output root, not data/processed")
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
    builders.m2_complete = False
    for tree in ("processed", "packages"):
        _rmtree(project / "data" / tree)
    preparation.prepare(project, resume=False, dry_run=False)
    resumable = [call for call in builders.calls if "resume" in call]
    assert all(call["resume"] is False for call in resumable)


def test_the_scientific_corpus_is_not_truncated(project: Path,
                                               builders: Builders) -> None:
    """L.12 forbids shrinking a scientific input to fit the machine.

    The producer asks for every canonical record: `all_records=True` with no
    record limit. The legacy helper this replaced defaulted `limit_records=3`,
    which would have silently preprocessed three videos per dataset.
    """
    source = (REPO / "src" / "prism_fas" / "pipeline" / "preparation.py").read_text(
        encoding="utf-8")
    assert "all_records=True" in source
    assert "limit_records=None" in source
    assert "m2_runner.run(" not in source, (
        "the legacy m2a runner writes where M3A cannot read")

    preparation.prepare(project, dry_run=False)
    m2_calls = [call for call in builders.calls if call["step"] == "m2_preprocess"]
    assert m2_calls and m2_calls[0]["datasets"] == ["casia_fasd", "msu_mfsd"]


def test_the_target_dataset_is_never_preprocessed(project: Path,
                                                  builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    datasets = {name for call in builders.calls
                for name in call.get("datasets", ())}
    assert "siw_mv2" not in datasets
    assert "siw_mv2" not in preparation.SOURCE_DATASETS


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


def _external_config(project: Path) -> Path:
    return _write(project / "configs" / "paths.local.yaml", yaml.safe_dump({
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


def test_an_in_folder_copy_supersedes_an_external_root(project: Path) -> None:
    """The point of copying the assets in: the in-folder copy must win.

    A config naming this folder used to be treated as authoritative outright. That
    left a folder whose datasets had been copied inside still reading them from the
    machine they came from, which is the whole defect this closure removes.
    """
    _external_config(project)          # the `project` fixture has in-folder raw+weights

    result = portable_paths.ensure_local_paths(project)

    assert result["action"] == portable_paths.REWRITTEN
    written = yaml.safe_load(
        (project / "configs" / "paths.local.yaml").read_text(encoding="utf-8"))
    assert Path(written["raw_datasets"]["casia_fasd"]) == project / "data/raw/casia_fasd"
    assert Path(written["model_cache"]) == project / "weights"
    assert "/mnt/big" not in json.dumps(written)


def test_an_external_root_survives_when_no_in_folder_copy_exists(project: Path) -> None:
    """The override is still honoured where it is the only answer.

    A machine that legitimately keeps its corpora elsewhere must keep working;
    in-folder only wins when there is something in the folder to win with.
    """
    external = project.parent / "elsewhere" / "casia"
    _write(external / "README.txt", "external")
    _rmtree(project / "data" / "raw" / "casia_fasd")
    _rmtree(project / "weights")
    _write(project / "configs" / "paths.local.yaml", yaml.safe_dump({
        "workspace_root": str(project.parent), "project_root": str(project),
        "raw_datasets": {"casia_fasd": str(external),
                         "msu_mfsd": str(project / "data/raw/msu_mfsd"),
                         "siw_mv2": str(project / "data/raw/siw_mv2")},
        "model_cache": str(project / "weights"),
        "work_root": str(project / "data" / "work"),
        "processed_root": str(project / "data" / "processed"),
        "package_root": str(project / "data" / "packages"),
        "runs_root": str(project / "runs"),
        "reports_root": str(project / "reports")}))

    portable_paths.ensure_local_paths(project)

    written = yaml.safe_load(
        (project / "configs" / "paths.local.yaml").read_text(encoding="utf-8"))
    assert Path(written["raw_datasets"]["casia_fasd"]) == external, (
        "with no in-folder copy, the declared external root is the only answer")


def test_every_write_root_is_forced_inside_the_project(project: Path) -> None:
    """A write root left outside would put this run's outputs on the old machine."""
    _write(project / "configs" / "paths.local.yaml", yaml.safe_dump({
        "workspace_root": "/old", "project_root": str(project),
        "raw_datasets": {"casia_fasd": "/old/casia", "msu_mfsd": "/old/msu",
                         "siw_mv2": "/old/siw"},
        "model_cache": "/old/models", "work_root": "/old/work",
        "processed_root": "/old/processed", "package_root": "/old/packages",
        "runs_root": "/old/runs", "reports_root": "/old/reports"}))

    portable_paths.ensure_local_paths(project)

    written = yaml.safe_load(
        (project / "configs" / "paths.local.yaml").read_text(encoding="utf-8"))
    for key in ("work_root", "processed_root", "package_root", "runs_root",
                "reports_root"):
        assert Path(written[key]).is_relative_to(project), f"{key} escaped the project"


def test_the_builders_are_handed_the_generated_config(project: Path,
                                                      builders: Builders) -> None:
    preparation.prepare(project, dry_run=False)
    m2_calls = [call for call in builders.calls if call["step"] == "m2_preprocess"]
    assert all(call["config_path"] == project / "configs" / "paths.local.yaml"
               for call in m2_calls)


# --- G2. the SCRFD path inside a hashed config --------------------------------

def test_the_m2_config_hash_does_not_move_when_the_detector_is_found_elsewhere() -> None:
    """The declared path is inside config_hash, so only the lookup may move.

    `scrfd_model_path` is an absolute path from the authoring machine and it is
    hashed into `config_hash`, which names the M2 work tree and is stamped into
    every manifest row. Repointing the string would change a frozen identity, so
    resolution falls back to the in-folder copy instead and the string stays put.
    """
    from prism_fas.data.preprocess_m2 import M2Config, load_m2_config

    declared = load_m2_config(REPO / "configs" / "data" / "preprocess_m2.yaml")
    raw = yaml.safe_load(
        (REPO / "configs" / "data" / "preprocess_m2.yaml").read_text(encoding="utf-8"))
    raw["scrfd_model_path"] = "D:/no-such-machine/face_detectors/scrfd_10g_bnkps.onnx"
    relocated = M2Config.model_validate(raw)

    # The hash follows the DECLARED string, which is the point: it is an identity,
    # not a location, and this test would fail if resolution leaked into it.
    assert relocated.config_hash != declared.config_hash, (
        "sanity: a different declared string is a different config")
    assert declared.config_hash == load_m2_config(
        REPO / "configs" / "data" / "preprocess_m2.yaml").config_hash

    resolved = relocated.resolved_scrfd_model_path
    if (REPO / "weights" / "face_detectors" / "scrfd_10g_bnkps.onnx").is_file():
        assert resolved.is_file(), "an absent declared path must fall back in-folder"
        assert resolved.parent == REPO / "weights" / "face_detectors"
    else:
        pytest.skip("no in-folder detector copy in this checkout")


def test_nothing_that_hashes_the_config_uses_the_resolved_path() -> None:
    """The resolved path must never reach an identity. Checked in the source."""
    import inspect

    from prism_fas.data import preprocess_m2

    source = inspect.getsource(preprocess_m2.M2Config)
    hash_body = source[source.index("def config_hash"):]
    hash_body = hash_body[:hash_body.index("\n    @property", 1)
                          if "\n    @property" in hash_body[1:] else len(hash_body)]
    assert "resolved_scrfd_model_path" not in hash_body


# --- H. relocation ------------------------------------------------------------

def _prepare_at(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build the same fixture project under a different absolute root."""
    repo = root / "PRISM_FAS_C_LLM_Project"
    for dataset in ("casia_fasd", "msu_mfsd", "siw_mv2"):
        _write(repo / "data" / "raw" / dataset / "README.txt", dataset)
    _write(repo / "weights" / "siglip2" / "model.bin", "weights")
    _write(repo / "assets" / "recipe_banks" / "c3" / "llm" / "bank.json", "{}")
    import shutil

    (repo / "configs" / "data").mkdir(parents=True, exist_ok=True)
    for name in ("preprocess_m2.yaml", "m2_run_profiles.yaml"):
        shutil.copyfile(REPO / "configs" / "data" / name, repo / "configs" / "data" / name)

    from prism_fas.data import package
    from prism_fas.data.package import m3b
    from prism_fas.synthesis import pair_plan

    stub = Builders(repo)
    monkeypatch.setattr(preparation, "_step_m2", stub.step_m2)
    monkeypatch.setattr(preparation, "m2_status", stub.m2_status)
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
    report = preparation.prepare(project, dry_run=False)

    assert (project / "data" / "packages").is_dir()
    assert (project / "data" / "packages" / "gpat_pairs" / "PAIR_PLAN_LOCK.json").is_file()
    assert preparation.what_is_needed(project)["nothing_to_do"] is True
    # The preprocessed tree is the M2 profile namespace, not `data/processed`;
    # the report says where it is rather than leaving the operator to guess.
    m2_root = Path(next(item for item in report["steps"]
                        if item["step"] == "m2_preprocess")["m2_root"])
    assert m2_root == preparation.m2_output_root(project)
    assert project in m2_root.parents


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

    from prism_fas.config.models import DatasetDefinition, load_paths
    from prism_fas.data.adapters import adapter_for

    assert list(inspect.signature(load_paths).parameters) == ["path"]
    assert list(inspect.signature(adapter_for).parameters) == ["definition", "root"]
    assert hasattr(DatasetDefinition, "model_validate")
    assert hasattr(adapter_for(
        DatasetDefinition(dataset="casia_fasd", adapter_version="v1"),
        Path(".")), "records")

    source = inspect.getsource(preparation._records)
    assert "adapter_for(" in source and ".records()" in source
    assert inspect.getsource(preparation._record_count).count("_records(") == 1, (
        "the count must come from the record list the producer actually walks")
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


@pytest.mark.parametrize("reason", [preparation.M2_INCOMPLETE,
                                    preparation.TARGET_IN_SOURCE_TREE])
def test_the_new_preparation_reason_codes_stop_before_c4(
        train_module, monkeypatch: pytest.MonkeyPatch, capsys, reason: str) -> None:
    """`M2_INCOMPLETE` and `TARGET_IN_SOURCE_TREE` are new to the runner. They
    must reach the operator by name and end the run before any stage starts."""
    module, calls = train_module

    def failing(repo, **kwargs):
        calls["order"].append("preparation")
        raise preparation.PreparationError(reason, "the M2 tree is not usable")

    monkeypatch.setattr(preparation, "prepare", failing)
    exit_code = _zero_argument(module, scientific=True, monkeypatch=monkeypatch)

    assert "orchestrator" not in calls["order"]
    assert exit_code == module.EXIT_BLOCKED
    stderr = capsys.readouterr().err
    assert f"[{reason}]" in stderr
    assert "Stopped BEFORE C4" in stderr


def test_an_autograd_failure_stops_before_c4(
        train_module, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The RTX 5090 case. The probe raising must end the run before preparation
    and before any stage, with the reason code and the stop line the operator
    reads from the terminal."""
    from prism_fas.pipeline import gpu_preflight

    module, calls = train_module

    def failing(repo, *, strict):
        calls["order"].append("gpu_preflight")
        raise gpu_preflight.GPUPreflightError(
            gpu_preflight.AUTOGRAD_FAILED,
            "the representative model could not complete a forward/backward step")

    monkeypatch.setattr(gpu_preflight, "run_preflight", failing)
    exit_code = _zero_argument(module, scientific=True, monkeypatch=monkeypatch)

    assert calls["order"] == ["gpu_preflight"], (
        "neither preparation nor any scientific stage may start after AUTOGRAD_FAILED")
    assert exit_code == module.EXIT_BLOCKED
    stderr = capsys.readouterr().err
    assert "[AUTOGRAD_FAILED]" in stderr
    assert "Stopped BEFORE C4" in stderr


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
