"""The portable one-command runner: bootstrap, intent resolution and portability.

The property this suite protects is that `python train.py` behaves correctly on a
machine nobody tested it on. That machine is by definition unavailable, so every
test here constructs the condition instead of waiting for it: a fake RTX 5090, a
driver too old, an unknown accelerator, an unsupported interpreter, a relocated
project root.

The safety boundary gets the most attention. A CPU host must resolve to
CPU_FULL_REHEARSAL and must not be able to reach the scientific branch by any
route, and a rehearsal must write only to its own namespace. Those two together
are what stop a laptop from producing a Version-C P3 number.
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bootstrap as boot  # noqa: E402
from prism_fas.pipeline import assets, portability, runner  # noqa: E402


@pytest.fixture(scope="module")
def contract() -> dict:
    return boot.read_contract()


# --- the bootstrap must not need the thing it bootstraps ---------------------

def test_bootstrap_imports_only_the_standard_library() -> None:
    """A single third-party import here breaks the entrypoint on a bare host."""
    tree = ast.parse((REPO / "bootstrap.py").read_text(encoding="utf-8"))
    stdlib = set(sys.stdlib_module_names)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    assert imported <= stdlib, f"bootstrap.py imports non-stdlib {sorted(imported - stdlib)}"


def test_train_py_top_level_imports_are_stdlib_only() -> None:
    """Everything above the re-exec point runs before the environment exists."""
    tree = ast.parse((REPO / "train.py").read_text(encoding="utf-8"))
    stdlib = set(sys.stdlib_module_names)
    top_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            top_level.add(node.module.split(".")[0])
    assert top_level <= stdlib, f"train.py imports {sorted(top_level - stdlib)} at module level"


# --- the contract ------------------------------------------------------------

def test_the_contract_parses_without_pyyaml(contract: dict) -> None:
    assert contract["schema_version"] == "prism-environment-contract-v1"
    assert set(contract["profiles"]) >= {"cpu", "cuda-cu129", "cuda-cu126"}
    assert contract["python"]["minimum"] == "3.11"


def test_only_cuda_profiles_may_do_science(contract: dict) -> None:
    profiles = contract["profiles"]
    assert profiles["cpu"]["supports_scientific_execution"] is False
    for name in ("cuda-cu130", "cuda-cu129", "cuda-cu126"):
        assert profiles[name]["supports_scientific_execution"] is True


def test_cuda_profiles_are_declared_not_claimed_validated(contract: dict) -> None:
    """Honesty check: no CUDA wheel has been validated on this machine."""
    for name in ("cuda-cu130", "cuda-cu129", "cuda-cu126"):
        assert contract["profiles"][name]["status"] == "DECLARED_NOT_VALIDATED_HERE"
    assert contract["profiles"]["cpu"]["status"] == "VALIDATED"


def test_the_current_interpreter_is_supported(contract: dict) -> None:
    assert boot.check_python(contract)["supported"] is True


def test_an_unsupported_interpreter_is_refused_with_the_numbers() -> None:
    fake = {"python": {"minimum": "3.99", "maximum_exclusive": "4.0"}}
    with pytest.raises(boot.BootstrapError) as caught:
        boot.check_python(fake)
    assert caught.value.reason == boot.UNSUPPORTED_PYTHON
    assert "3.99" in str(caught.value)
    assert "never installs or replaces" in str(caught.value)


# --- hardware to profile, without searching ---------------------------------

@pytest.mark.parametrize("platform_tag,name,driver,capability,expected", [
    # Linux keeps exactly the plan it had.
    ("linux_x86_64", "NVIDIA GeForce RTX 5090", "580.88", "12.0", "cuda-cu129"),
    ("linux_x86_64", "NVIDIA GeForce RTX 5080", "572.00", "12.0", "cuda-cu129"),
    ("linux_x86_64", "NVIDIA H100 80GB HBM3", "575.00", "9.0", "cuda-cu129"),
    ("linux_x86_64", "NVIDIA GeForce RTX 4090", "550.54", "8.9", "cuda-cu126"),
    ("linux_x86_64", "NVIDIA A100-SXM4-80GB", "535.10", "8.0", "cuda-cu126"),
    # Windows cannot use cu129: that index publishes no win_amd64 wheel for the
    # pinned torch, so a Blackwell card resolves to the CUDA 13.0 profile and an
    # Ada or Ampere card to CUDA 12.6.
    ("win_amd64", "NVIDIA GeForce RTX 5090", "580.88", "12.0", "cuda-cu130"),
    ("win_amd64", "NVIDIA H100 80GB HBM3", "580.00", "9.0", "cuda-cu130"),
    ("win_amd64", "NVIDIA GeForce RTX 4090", "550.54", "8.9", "cuda-cu126"),
    ("win_amd64", "NVIDIA A100-SXM4-80GB", "535.10", "8.0", "cuda-cu126"),
])
def test_a_declared_gpu_selects_its_declared_profile(contract: dict,
                                                     platform_tag: str, name: str,
                                                     driver: str, capability: str,
                                                     expected: str) -> None:
    gpu = {"available": True, "name": name, "driver_version": driver,
           "compute_capability": capability, "memory_total_mb": 32768}
    selection = boot.select_profile(contract, gpu, platform_tag=platform_tag)
    assert selection["profile_id"] == expected
    assert selection["supports_scientific_execution"] is True
    assert platform_tag in contract["profiles"][expected]["platforms"]


def test_a_5090_on_an_old_driver_is_refused_rather_than_downgraded(contract: dict) -> None:
    """The failure this prevents happens after the data is loaded, not at import."""
    gpu = {"available": True, "name": "NVIDIA GeForce RTX 5090",
           "driver_version": "520.00", "compute_capability": "12.0"}
    with pytest.raises(boot.BootstrapError) as caught:
        boot.select_profile(contract, gpu, platform_tag=boot.LINUX_X86_64)
    assert caught.value.reason == boot.CUDA_NOT_VALIDATED
    assert "will NOT guess a CUDA wheel" in str(caught.value)
    assert "will NOT fall back to CPU" in str(caught.value)


def test_an_unknown_accelerator_matches_nothing(contract: dict) -> None:
    gpu = {"available": True, "name": "Some Unreleased Accelerator",
           "driver_version": "999.0", "compute_capability": "13.0"}
    with pytest.raises(boot.BootstrapError) as caught:
        boot.select_profile(contract, gpu, platform_tag=boot.LINUX_X86_64)
    assert caught.value.detail["family"] == "UNRECOGNISED_NAME"


def test_no_gpu_selects_the_cpu_profile_and_refuses_science(contract: dict) -> None:
    selection = boot.select_profile(contract, {"available": False})
    assert selection["profile_id"] == "cpu"
    assert selection["supports_scientific_execution"] is False


def test_family_inference_is_provenance_and_never_a_gate(contract: dict) -> None:
    """The family names the architecture for the operator; it decides nothing.

    Two non-matches are kept distinct because they mean different things: no
    name was reported at all, versus a name this contract has not seen. Neither
    blocks selection, which is decided by compute capability and driver.
    """
    assert boot.gpu_family("NVIDIA GeForce RTX 5090", contract) == "Blackwell"
    assert boot.gpu_family("NVIDIA GeForce RTX 4090", contract) == "Ada"
    assert boot.gpu_family(None, contract) == "UNKNOWN"
    assert boot.gpu_family("Totally Made Up", contract) == "UNRECOGNISED_NAME"
    assert contract["selection"]["model_name_is_a_gate"] is False


def test_a_card_whose_name_is_unlisted_is_still_selected_on_capability(
        contract: dict) -> None:
    """§12: do not reject an otherwise compatible GPU over its marketing name."""
    gpu = {"available": True, "name": "NVIDIA Some-New-Datacenter-Part",
           "driver_version": "560.00", "compute_capability": "8.9",
           "memory_total_mb": 49152}
    selection = boot.select_profile(contract, gpu, platform_tag=boot.LINUX_X86_64)
    assert selection["profile_id"] == "cuda-cu126"
    assert selection["supports_scientific_execution"] is True
    assert selection["family"] == "UNRECOGNISED_NAME"
    assert selection["grade"] == boot.COMPATIBLE_DECLARED_PROFILE


def test_a_capability_inside_the_range_but_unlisted_is_a_candidate(
        contract: dict) -> None:
    """Plausible is reported as plausible, not promoted to validated."""
    gpu = {"available": True, "name": "NVIDIA Whatever", "driver_version": "575.00",
           "compute_capability": "8.7", "memory_total_mb": 24576}
    selection = boot.select_profile(contract, gpu, platform_tag=boot.LINUX_X86_64)
    assert selection["grade"] == boot.UNVALIDATED_COMPATIBLE_CANDIDATE


# --- environment identity ----------------------------------------------------

def test_environment_identity_covers_every_included_requirement_file(contract: dict) -> None:
    profile = contract["profiles"]["cpu"]
    identity = boot.environment_identity("cpu", profile, "3.13.11")
    assert set(identity["files"]) == {"requirements/cpu.txt", "requirements/base.txt",
                                      "requirements/constraints.txt"}
    assert len(identity["identity"]) == 64


def test_environment_identity_is_stable_and_path_independent(contract: dict,
                                                             tmp_path: Path) -> None:
    """Moving the folder must not force a reinstall."""
    profile = contract["profiles"]["cpu"]
    first = boot.environment_identity("cpu", profile, "3.13.11")["identity"]
    second = boot.environment_identity("cpu", profile, "3.13.11")["identity"]
    assert first == second
    material = boot.environment_identity("cpu", profile, "3.13.11")["material"]
    assert "hostname" not in json.dumps(material)
    assert str(REPO) not in json.dumps(material)


def test_a_different_profile_gives_a_different_identity(contract: dict) -> None:
    cpu = boot.environment_identity("cpu", contract["profiles"]["cpu"], "3.13.11")
    cuda = boot.environment_identity("cuda-cu129", contract["profiles"]["cuda-cu129"],
                                     "3.13.11")
    assert cpu["identity"] != cuda["identity"]


def test_a_corrupt_manifest_reads_as_absent_rather_than_ready(tmp_path: Path) -> None:
    """An interrupted write must cost a reinstall, never a false ready claim."""
    path = tmp_path / "ENVIRONMENT_MANIFEST.json"
    path.write_text('{"environment_identity": "abc"', encoding="utf-8")
    assert boot.read_manifest(path) is None


def test_the_manifest_write_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ENVIRONMENT_MANIFEST.json"
    boot.write_manifest({"environment_identity": "x" * 64}, path)
    assert boot.read_manifest(path)["environment_identity"] == "x" * 64
    assert not list(path.parent.glob(".*tmp"))


# --- intent resolution -------------------------------------------------------

def test_a_cpu_host_resolves_to_rehearsal_and_cannot_reach_science() -> None:
    plan = runner.resolve(REPO, {"profile_supports_scientific_execution": False,
                                 "profile_id": "cpu", "gpu": {"available": False}})
    assert plan.intent == runner.CPU_FULL_REHEARSAL
    assert plan.profile_name == "rehearsal"
    assert plan.is_scientific is False
    assert plan.as_dict()["scientific_eligible"] is False


def test_a_cuda_host_resolves_to_science_at_the_first_incomplete_stage() -> None:
    """Proven with a mock. No GPU is allocated and no science is started."""
    plan = runner.resolve(REPO, {"profile_supports_scientific_execution": True,
                                 "profile_id": "cuda-cu129",
                                 "gpu": {"available": True,
                                         "name": "NVIDIA GeForce RTX 5090",
                                         "driver_version": "580.88",
                                         "compute_capability": "12.0",
                                         "memory_total_mb": 32768}})
    assert plan.intent == runner.GPU_SCIENTIFIC_FULL
    assert plan.profile_name == "full"
    assert plan.first_stage == "C4", "C0-C3 are complete; C4 is the resume point"
    assert plan.last_stage == "C13"
    assert plan.resume is True


def test_c0_to_c3_are_recognised_as_scientifically_complete() -> None:
    completion = runner.scientific_completion(REPO)
    for stage in ("C0", "C1", "C2", "C3"):
        assert completion[stage] is True, stage
    for stage in ("C4", "C13"):
        assert completion[stage] is False, stage


def test_zero_argument_always_resumes() -> None:
    for scientific in (True, False):
        plan = runner.resolve(REPO, {"profile_supports_scientific_execution": scientific,
                                     "profile_id": "x", "gpu": {"available": scientific}})
        assert plan.resume is True, "the user must never have to type --resume"


def test_the_preflight_summary_names_the_intent_and_the_firewall() -> None:
    plan = runner.resolve(REPO, {"profile_supports_scientific_execution": False,
                                 "profile_id": "cpu", "gpu": {"available": False}})
    text = runner.preflight_summary(REPO, plan, git_identity="abc1234")
    assert "CPU_FULL_REHEARSAL" in text
    assert "Target firewall     ARMED" in text
    assert "reports/rehearsal/" in text


def _scientific_plan(repo=REPO):
    return runner.resolve(repo, {"profile_supports_scientific_execution": True,
                                 "profile_id": "cuda-cu129",
                                 "gpu": {"available": True,
                                         "name": "NVIDIA GeForce RTX 5090",
                                         "driver_version": "580.88"}})


def test_absent_derived_trees_do_not_block_a_scientific_plan() -> None:
    """They are what the run BUILDS, so requiring them up front never starts it.

    This assertion used to be the opposite, written before preparation existed:
    the plan refused because data/processed and data/packages were absent. On the
    destination machine they are absent every time, and the step that creates them
    runs after this gate — so the old rule blocked the run from ever producing
    what the rule demanded.
    """
    plan = _scientific_plan()
    generated = {"preprocessed_source_data", "source_packages", "gpat_pair_plan",
                 "target_label_artifact"}
    assert generated & {item["logical_name"]
                        for item in plan.bundle["produced_by_the_run"]} == generated
    assert plan.ready, plan.blockers
    assert not plan.blockers


def test_a_missing_operator_supplied_asset_still_blocks(tmp_path) -> None:
    """The guarantee the old test was protecting, in its still-true form.

    Preparation can derive data/processed from the raw corpora; it cannot invent
    the raw corpora or a frozen weight. Those must still stop the run at the gate
    rather than hours in.
    """
    empty = tmp_path / "PRISM_FAS_C_LLM_Project"
    (empty / "configs").mkdir(parents=True)
    (empty / "data").mkdir()
    for name in ("runs", "reports", "state"):
        (empty / name).mkdir()

    plan = _scientific_plan(empty)
    assert not plan.ready
    assert plan.blockers
    missing = {item["logical_name"] for item in plan.bundle["missing"]}
    assert missing, "a folder with no raw data and no weights must name what it lacks"
    text = runner.preflight_summary(empty, plan)
    assert "BLOCKED — nothing was executed" in text
    assert "MISSING" in text


# --- the rehearsal boundary --------------------------------------------------

def test_the_rehearsal_profile_cannot_declare_eligibility() -> None:
    from prism_fas.pipeline.profiles import load_profile

    profile = load_profile("rehearsal", repo=REPO)
    assert profile.scientific_eligible is False
    assert profile.may_select_scientific_winner is False
    assert profile.reports_namespace == "reports/rehearsal"
    assert profile.runs_namespace == "runs/rehearsal"


def test_the_rehearsal_may_not_reach_a_live_provider_or_a_target_label() -> None:
    from prism_fas.pipeline.profiles import load_profile

    policy = load_profile("rehearsal", repo=REPO).compute_policy
    assert policy.live_provider is False
    assert policy.target_label_access is False
    assert policy.target_metric_access is False
    assert policy.raw.get("target_package") == "fixture_only"


def test_exactly_one_profile_is_scientifically_eligible() -> None:
    from prism_fas.pipeline.profiles import PROFILE_NAMES, load_profile

    eligible = [name for name in PROFILE_NAMES
                if load_profile(name, repo=REPO).scientific_eligible]
    assert eligible == ["full"]


# --- the portable bundle -----------------------------------------------------

def test_the_asset_manifest_is_built_fresh_rather_than_read() -> None:
    """A stale committed manifest must not be able to claim an asset is present."""
    manifest = assets.load_manifest(REPO)
    assert manifest["item_count"] > 20
    assert manifest["project_relative"] is True
    assert manifest["policy"]["no_invented_hashes"] is True
    assert manifest["policy"]["no_automatic_dataset_download"] is True


def test_target_assets_are_required_only_from_c10() -> None:
    """A rehearsal must never be blocked by — or reach for — the SiW package."""
    items = {item["logical_name"]: item for item in assets.load_manifest(REPO)["items"]}
    siw = items["raw_dataset_siw_mv2"]
    assert siw["required_for_cpu_rehearsal"] is False
    assert siw["required_for_real_target_only"] is True
    assert siw["required_stage"] == "C10"
    assert siw["access"] == assets.EVALUATION_ONLY


def test_the_bundle_verdicts_are_the_declared_ones() -> None:
    cpu = portability.bundle_readiness(REPO, intent="CPU_FULL_REHEARSAL")
    gpu = portability.bundle_readiness(REPO, intent="GPU_SCIENTIFIC_FULL")
    assert cpu["verdict"] == "PORTABLE_BUNDLE_READY_FOR_CPU_REHEARSAL"
    assert gpu["verdict"] == "PORTABLE_BUNDLE_READY_FOR_GPU_SCIENCE"
    assert cpu["ready"] is True, cpu["missing"]
    assert gpu["required_count"] > cpu["required_count"]


def test_every_missing_item_carries_an_actionable_remedy() -> None:
    report = portability.bundle_readiness(REPO, intent="GPU_SCIENTIFIC_FULL")
    for item in report["missing"]:
        assert item["expected_path"]
        assert item["how_to_obtain"], item["logical_name"]


# --- relocation --------------------------------------------------------------

def test_a_relocated_project_keeps_its_scientific_identities(tmp_path: Path) -> None:
    """Copying the folder must not move a frozen identity."""
    destination = tmp_path / "moved project with spaces"
    for relative in ("assets", "configs", "reports/c3", "src"):
        source = REPO / relative
        if source.exists():
            shutil.copytree(source, destination / relative,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                          "raw_responses"))

    from prism_fas.pipeline.checks import check_c3_scientific_banks_frozen

    here = check_c3_scientific_banks_frozen(REPO)
    there = check_c3_scientific_banks_frozen(destination)
    assert there.ok, there.detail["problems"]
    assert there.detail["lock_identity_recomputed"] == here.detail["lock_identity_recomputed"]
    for arm, row in here.detail["arms"].items():
        assert there.detail["arms"][arm]["bank_identity_recomputed"] == \
            row["bank_identity_recomputed"]


def test_a_relocated_project_rebuilds_the_same_search_plans(tmp_path: Path) -> None:
    import yaml

    from prism_fas.search.lr_decision import load_decision
    from prism_fas.search.plan import gpat_search_plan

    destination = tmp_path / "elsewhere"
    (destination / "configs" / "search").mkdir(parents=True)
    (destination / "configs" / "synthesis").mkdir(parents=True)
    shutil.copy2(REPO / "configs/search/lr_anchor_decision.yaml",
                 destination / "configs/search/lr_anchor_decision.yaml")
    shutil.copy2(REPO / "configs/synthesis/gpat_m8.yaml",
                 destination / "configs/synthesis/gpat_m8.yaml")

    config = yaml.safe_load(
        (REPO / "configs/synthesis/gpat_m8.yaml").read_text(encoding="utf-8"))
    here, _ = gpat_search_plan(config, lr_decision=load_decision(REPO).for_component("C4"))
    there, _ = gpat_search_plan(
        config, lr_decision=load_decision(destination).for_component("C4"))
    assert here.identity == there.identity


def test_paths_in_the_manifest_are_project_relative_where_they_can_be() -> None:
    items = assets.load_manifest(REPO)["items"]
    in_git = [item for item in items if item["origin"] == assets.IN_GIT]
    assert in_git
    for item in in_git:
        assert not Path(item["expected_path"]).is_absolute(), item["logical_name"]


# --- the approved learning-rate decision -------------------------------------

def test_the_lr_decision_is_approved_and_preserves_every_ratio() -> None:
    from prism_fas.search.lr_decision import load_decision

    record = load_decision(REPO)
    assert record.approved
    c4 = record.for_component("C4")
    assert c4.interpretation == "B_common_multiplier"
    assert c4.candidates == (0.5, 1.0, 2.0)
    assert c4.lr_for_groups(1.0) == {"encoder_lr": 2.0e-4, "recipe_lr": 1.0e-4,
                                     "generator_lr": 2.0e-4}
    for multiplier in c4.candidates:
        assert c4.ratio_preserved(multiplier), multiplier


def test_track_g_needs_no_decision_but_is_still_searched() -> None:
    """UNIQUE_INHERITED_ANCHOR is about the ANCHOR, not about the search.

    This test previously asserted `track_g.candidates == ()` under the name
    "carries no multiplier because it needs none". It was encoding a defect: the
    interpretation records that exactly one inherited LR scalar is applicable, so
    no USER DECISION was needed to choose one. It does not follow that the
    coordinate is skipped, and 15.2.2 puts `learning_rate` first in the frozen
    order with candidates anchor x {0.5, 1.0, 2.0} for every component that has
    an applicable anchor. Track G was omitting its own learning rate.

    Corrected before the first C7 scientific trial; the full contract lives in
    tests/pipeline/test_lr_track_g_coordinate.py and the identity move is
    recorded in reports/handoff/LR_ANCHOR_DECISION_CORRECTION.json.
    """
    from prism_fas.search.lr_decision import load_decision

    track_g = load_decision(REPO).for_component("C7_TRACK_G")

    # How the anchor was resolved: unique, so no user decision was required.
    assert track_g.interpretation == "UNIQUE_INHERITED_ANCHOR"
    assert track_g.compliance_class == "ALREADY_IMPLIED_BY_FROZEN_SPEC"
    assert dict(track_g.anchor_vector) == {"head_lr": 1.0e-4}
    # ...and it expands over one group, so there is no inherited ratio to hold.
    assert track_g.searches_a_multiplier is False

    # Whether the frozen coordinate is searched: yes, like every other component.
    assert track_g.searches_the_learning_rate is True
    assert track_g.candidates == (0.5, 1.0, 2.0)
    assert track_g.lr_for_groups(1.0) == {"head_lr": 1.0e-4}


def test_an_unapproved_decision_record_is_refused(tmp_path: Path) -> None:
    """A search plan must never be built from a decision nobody approved."""
    from prism_fas.search.lr_decision import LRDecisionError, load_decision

    (tmp_path / "configs" / "search").mkdir(parents=True)
    text = (REPO / "configs/search/lr_anchor_decision.yaml").read_text(encoding="utf-8")
    (tmp_path / "configs/search/lr_anchor_decision.yaml").write_text(
        text.replace("decision_status: APPROVED",
                     "decision_status: AWAITING_USER_APPROVAL"), encoding="utf-8")
    with pytest.raises(LRDecisionError, match="not APPROVED"):
        load_decision(tmp_path)


# --- reporting cannot train --------------------------------------------------

def test_no_reporting_module_imports_a_trainer() -> None:
    """A figure must never be able to trigger the job it describes."""
    forbidden = ("prism_fas.detector.trainer", "prism_fas.synthesis.gpat_trainer",
                 "prism_fas.train.trainer", "modal")
    for path in (REPO / "src" / "prism_fas" / "reporting").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not any(name.startswith(item) for item in forbidden), \
                    f"{path.name} imports {name}"


def test_every_declared_table_is_written_even_when_empty(tmp_path: Path) -> None:
    from prism_fas.reporting import tables

    report = tables.generate_all({}, tmp_path)
    assert len(report["tables"]) == len(tables.DECLARED_TABLES)
    for item in report["tables"]:
        assert (tmp_path / item["csv"]).exists()
        assert item["empty_reason"], "an empty table must say why it is empty"


def test_plots_report_what_they_could_not_draw(tmp_path: Path) -> None:
    from prism_fas.reporting import plots

    report = plots.generate_all({}, tmp_path)
    assert report["written_count"] == 0
    assert report["skipped_count"] == len(plots.DECLARED_FIGURES)
    for item in report["skipped"]:
        assert item["needs"], "a skipped figure must name the evidence it needed"


def test_the_report_marks_a_rehearsal_as_not_scientific_evidence() -> None:
    from prism_fas.reporting import report

    html = report.render({"execution_intent": "CPU_FULL_REHEARSAL", "scientific": False})
    assert "REHEARSAL — NOT SCIENTIFIC EVIDENCE" in html
    assert "completes no milestone" in html
    assert "<!doctype html>" in html.lower()


def test_the_report_never_invents_a_target_number() -> None:
    from prism_fas.reporting import report

    html = report.render({"execution_intent": "GPU_SCIENTIFIC_FULL", "scientific": True})
    assert "no scientific results exist" in html
    assert "no target protocol evidence" in html


def test_complexity_never_enters_winner_selection() -> None:
    from prism_fas.reporting import complexity

    import torch

    model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.GELU(),
                                torch.nn.Linear(4, 1))
    profile = complexity.profile_model(model, torch.randn(2, 8), name="probe")
    assert profile["selection_input"] is False
    assert profile["total_parameters"] == 41
    assert profile["complexity"]["status"] == complexity.COMPLETE
    assert profile["complexity"]["flops"] == profile["complexity"]["macs"] * 2


def test_an_uncountable_module_downgrades_to_partial_rather_than_lying() -> None:
    from prism_fas.reporting import complexity

    import torch

    class Exotic(torch.nn.Module):
        def forward(self, x):                                # noqa: ANN001, ANN201
            return x * 2

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), Exotic())
    profile = complexity.profile_model(model, torch.randn(2, 4), name="probe")
    assert profile["complexity"]["status"] == complexity.PARTIAL
    assert "Exotic" in profile["complexity"]["unsupported_operations"]
    assert any("LOWER BOUND" in warning for warning in profile["complexity"]["warnings"])


# --- rerunning the same command must produce the same rehearsal --------------
#
# Found by running `python train.py` three times on the laptop. C8 sampled its
# rehearsal rows from the PENDING remainder, so each rerun exercised two
# different arms, the complexity table changed between runs, and after enough
# reruns the sample would have been empty — a rehearsal reporting PASS while
# executing nothing. The stage-level rollup compounded it by being written
# inside the per-row loop, so it named whichever arm finished last.

def test_the_rehearsal_sample_is_drawn_from_the_plan_not_the_remainder() -> None:
    """Plan order, so a rerun exercises the same arms."""
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c8.py").read_text(
        encoding="utf-8")
    assert "pending[:SMOKE_ROWS]" not in source, (
        "sampling the pending remainder slides the window forward on every rerun")
    assert "list(zip(plan.rows, decisions, directories))[:count]" in source


def test_the_stage_complexity_rollup_covers_every_arm_not_the_last_one() -> None:
    """The rollup is written once, after the loop, over all executed rows."""
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c8.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    run_one = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "_run_one")
    written = {node.value for node in ast.walk(run_one)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "C8_MODEL_COMPLEXITY.json" not in written, (
        "_run_one writes the stage rollup per row, so the last row wins")
    assert "C8_COMPUTE_RESOURCES.json" not in written


def test_a_multi_model_rollup_is_read_as_every_model() -> None:
    """The collector must not read only the first entry of a rollup."""
    from prism_fas.reporting import profiled_entries

    rollup = {"models": [{"model": "detector_row_a"}, {"model": "detector_row_b"}]}
    assert [entry["model"] for entry in profiled_entries(rollup, "models")] == [
        "detector_row_a", "detector_row_b"]

    flat = {"model": "gpat_residual_generator", "total_parameters": 910538}
    assert profiled_entries(flat, "models") == [flat]

    assert profiled_entries({}, "models") == []
    assert profiled_entries({"models": [1, "two", None]}, "models") == []


def test_two_sampled_rows_cannot_collide_in_the_complexity_table() -> None:
    """Rows differing only by protocol or seed must get distinct names."""
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c8.py").read_text(
        encoding="utf-8")
    assert 'name=f"detector_{row.row_id}"' in source
    assert 'name=f"detector_{row.experiment_id}"' not in source, (
        "experiment_id repeats across protocols and seeds")


def test_a_resume_validated_row_is_reported_from_what_it_stored() -> None:
    """Reused rows still appear, so the table does not shrink on the second run."""
    from prism_fas.pipeline.adapters import c8

    assert hasattr(c8.C8Adapter, "_reuse_one")
    source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c8.py").read_text(
        encoding="utf-8")
    assert "SKIP_VALID_COMPLETE" in source and "_reuse_one" in source
