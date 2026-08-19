"""The dependency contract: one pin per package, and every pin installable.

`onnxruntime==1.24.0` sat in three requirement files for months. It was never
installed on any machine — this project's own `.venv` has no onnxruntime at all —
and it does not exist: the 1.24 family on PyPI begins at 1.24.1. A fresh Windows
deployment discovered that by failing at it, after resolving 2.5 GB of torch.

These tests are the check that would have caught it: a pin is only real if it is
declared consistently everywhere AND there is recorded index evidence that a
wheel exists for every interpreter the project says it supports.

Offline by construction. The index evidence is the committed artifact
`reports/handoff/ONNXRUNTIME_PIN_EVIDENCE.json`, produced by an authenticated
query and carrying the observation it is asked about, so the suite never needs a
socket to answer "is this installable".
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bootstrap as boot  # noqa: E402

REQUIREMENTS = REPO / "requirements"
PROFILE_FILES = ("cpu.txt", "cuda-cu126.txt", "cuda-cu129.txt")
EVIDENCE = REPO / "reports" / "handoff" / "ONNXRUNTIME_PIN_EVIDENCE.json"

PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s#]+)")


@pytest.fixture(scope="module")
def contract() -> dict:
    return boot.read_contract()


@pytest.fixture(scope="module")
def evidence() -> dict:
    assert EVIDENCE.exists(), (
        "the ONNX Runtime pin evidence is missing; it is the record of why this "
        "version was chosen and is required to keep this suite offline")
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def pins_in(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PIN.match(line.strip())
        if match:
            found[match.group("name").lower()] = match.group("version")
    return found


# --- the pin is one value, everywhere ----------------------------------------

def test_onnxruntime_is_pinned_to_the_same_version_in_every_declaration(
        contract: dict) -> None:
    declared = {name: pins_in(REQUIREMENTS / name).get("onnxruntime")
                for name in (*PROFILE_FILES, "constraints.txt")}
    assert None not in declared.values(), declared
    assert len(set(declared.values())) == 1, declared
    assert set(declared.values()) == {contract["dependencies"]["onnxruntime"]["pin"]}


def test_the_pin_that_never_existed_is_gone_from_the_whole_requirements_tree() -> None:
    for path in REQUIREMENTS.glob("*.txt"):
        assert "onnxruntime==1.24.0" not in path.read_text(encoding="utf-8"), path.name


def test_the_contract_records_why_the_previous_pin_was_replaced(
        contract: dict) -> None:
    block = contract["dependencies"]["onnxruntime"]
    assert block["previous_pin"] == "1.24.0"
    assert block["previous_pin_classification"] == "INTENDED_BUT_UNINSTALLABLE_PIN"
    assert block["result_affecting"] is True, "SCRFD preprocessing runs through ORT"
    assert "smallest installable patch" in block["selection_rule"]


# --- the pin is installable for every interpreter the project declares -------

def test_the_replaced_pin_is_recorded_as_absent_from_the_index(
        evidence: dict) -> None:
    availability = evidence["index_availability"]
    assert availability["1.24.0_present"] is False
    assert availability["1.24.0_file_count"] == 0
    assert availability["1.24_family_published"][0] == "1.24.1"


def test_the_selected_pin_is_the_smallest_patch_in_the_same_family(
        evidence: dict, contract: dict) -> None:
    availability = evidence["index_availability"]
    assert availability["selected"] == contract["dependencies"]["onnxruntime"]["pin"]
    assert availability["selected"] == availability["smallest_installable_in_family"]
    family = availability["1.24_family_published"]
    assert availability["selected"] != family[-1], (
        "the rule is smallest installable in the family, never latest")


def _declared_minors(contract: dict) -> list[str]:
    minimum = tuple(int(part) for part in contract["python"]["minimum"].split("."))
    maximum = tuple(int(part)
                    for part in contract["python"]["maximum_exclusive"].split("."))
    return [f"{minimum[0]}.{minor}" for minor in range(minimum[1], maximum[1])]


def test_a_windows_wheel_exists_for_every_supported_python(contract: dict,
                                                           evidence: dict) -> None:
    """The declared Windows profile, which is what the GPU laptop runs."""
    tags = set(evidence["index_availability"]["selected_windows_cp_tags"])
    required = {"cp" + minor.replace(".", "") for minor in _declared_minors(contract)}
    assert required <= tags, sorted(required - tags)


def test_a_linux_wheel_exists_for_every_supported_python(contract: dict,
                                                         evidence: dict) -> None:
    """Windows is not fixed by breaking the Linux host the folder may move to."""
    tags = set(evidence["index_availability"]["selected_linux_cp_tags"])
    required = {"cp" + minor.replace(".", "") for minor in _declared_minors(contract)}
    assert required <= tags, sorted(required - tags)


def test_the_preferred_interpreters_are_all_covered_by_the_pin(contract: dict,
                                                               evidence: dict) -> None:
    tags = set(evidence["index_availability"]["selected_windows_cp_tags"])
    for minor in contract["python"]["preferred_minors"]:
        assert "cp" + minor.replace(".", "") in tags, minor


# --- the runtime change is scientifically accounted for ----------------------

def test_the_runtime_change_carries_a_measured_equivalence_verdict(
        evidence: dict) -> None:
    assert evidence["classification"] == (
        "NUMERICALLY_EQUIVALENT_WITHIN_DECLARED_TOLERANCE")
    comparison = evidence["ab_comparison"]
    observed, tolerance = comparison["observed"], comparison["declared_tolerances"]
    assert observed["raw_tensor_max_abs"] <= tolerance["raw_tensor_max_abs"]
    assert observed["box_max_abs_pixels"] <= tolerance["box_max_abs_pixels"]
    assert observed["landmark_max_abs_pixels"] <= tolerance["landmark_max_abs_pixels"]
    assert observed["crop_box_max_abs_pixels"] <= tolerance["crop_box_max_abs_pixels"]
    assert observed["detection_count_mismatches"] == 0
    assert observed["selected_face_mismatches"] == 0


def test_the_equivalence_evidence_used_source_fixtures_and_no_target(
        evidence: dict) -> None:
    comparison = evidence["ab_comparison"]
    assert comparison["domain"] == "SOURCE"
    assert comparison["target_data_touched"] is False
    assert comparison["target_metrics_consulted"] is False
    assert comparison["fixture_count"] >= 20
    assert comparison["limitation"], "the unusable pin cannot be A/B'd; say so"


def test_the_equivalence_evidence_names_the_frozen_detector_it_ran(
        evidence: dict) -> None:
    comparison = evidence["ab_comparison"]
    model = REPO / comparison["model"]
    assert model.exists(), comparison["model"]
    assert len(comparison["model_sha256"]) == 64


# --- profile resolution -------------------------------------------------------

def test_the_cpu_profile_resolves_to_its_full_requirement_closure(
        contract: dict) -> None:
    files = boot.requirement_files(contract["profiles"]["cpu"])
    names = {path.name for path in files}
    assert names == {"cpu.txt", "base.txt", "constraints.txt"}
    assert "onnxruntime" in pins_in(REQUIREMENTS / "cpu.txt")


@pytest.mark.parametrize("profile_id", ["cuda-cu126", "cuda-cu129"])
def test_a_cuda_profile_resolves_consistently_with_the_contract(contract: dict,
                                                                profile_id: str) -> None:
    profile = contract["profiles"][profile_id]
    files = boot.requirement_files(profile)
    names = {path.name for path in files}
    assert names == {Path(profile["requirements"]).name, "base.txt", "constraints.txt"}

    text = (REPO / profile["requirements"]).read_text(encoding="utf-8")
    pins = pins_in(REPO / profile["requirements"])
    assert pins["torch"] == profile["torch"]
    assert profile["torch_index"] in text, "the CUDA wheel index must be declared"
    assert "onnxruntime" in pins


def test_torch_and_torchvision_agree_across_every_hardware_profile() -> None:
    """One torch minor for the project; only the CUDA tag differs."""
    versions = {name: pins_in(REQUIREMENTS / name) for name in PROFILE_FILES}
    assert len({pins["torch"] for pins in versions.values()}) == 1, versions
    assert len({pins["torchvision"] for pins in versions.values()}) == 1, versions


def test_torch_is_never_pinned_in_the_shared_constraints() -> None:
    """Pinning it there would make one host's wheel bind another's."""
    shared = pins_in(REQUIREMENTS / "constraints.txt")
    assert "torch" not in shared and "torchvision" not in shared


def test_no_requirement_line_is_left_unpinned() -> None:
    """A range resolved at install time is a different environment every time."""
    unpinned: list[str] = []
    for name in (*PROFILE_FILES, "constraints.txt", "dev.txt"):
        for line in (REQUIREMENTS / name).read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith(("#", "-")):
                continue
            if not PIN.match(text):
                unpinned.append(f"{name}: {text}")
    assert not unpinned, unpinned


# --- bootstrap tooling policy ------------------------------------------------

def test_the_pip_policy_is_declared_and_bounded_on_both_sides(contract: dict) -> None:
    policy = boot.pip_policy(contract)
    assert policy["upgrade_policy"] == "BOUNDED_MINIMUM_ONLY"
    assert policy["minimum"] and policy["maximum_exclusive"]
    assert boot._version_tuple(policy["minimum"]) < boot._version_tuple(
        policy["maximum_exclusive"])
    assert policy["upgrade_setuptools"] is False
    assert policy["upgrade_wheel"] is False


def test_pip_is_left_alone_when_it_already_satisfies_the_floor(
        contract: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boot, "_pip_version", lambda _python: "26.2.1")
    monkeypatch.setattr(boot, "_run", lambda *_a, **_k: pytest.fail(
        "pip was upgraded although it already satisfied the policy"))
    report = boot.ensure_pip_tooling(Path("python"), contract, quiet=True)
    assert report["action"] == "KEPT"


def test_the_pip_the_deployment_shipped_with_is_left_exactly_where_it_was(
        contract: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """The deployment log showed pip 24.0 -> 26.2.1, chosen by nothing.

    24.0 is the declared floor, so the upgrade bought nothing and silently made
    the resolver a different program than the one this project was pinned under.
    """
    monkeypatch.setattr(boot, "_pip_version", lambda _python: "24.0")
    monkeypatch.setattr(boot, "_run", lambda *_a, **_k: pytest.fail(
        "pip 24.0 already meets the declared floor and must not be upgraded"))
    assert boot.ensure_pip_tooling(Path("python"), contract,
                                   quiet=True)["action"] == "KEPT"


def test_a_pip_below_the_floor_is_upgraded_only_within_the_declared_window(
        contract: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(boot, "_pip_version", lambda _python: "23.2.1")
    monkeypatch.setattr(boot, "_run",
                        lambda command, **_k: commands.append(command))
    report = boot.ensure_pip_tooling(Path("python"), contract, quiet=True)
    assert report["action"] == "UPGRADED_TO_POLICY_FLOOR"
    policy = boot.pip_policy(contract)
    assert report["requirement"] == f"pip>={policy['minimum']},<{policy['maximum_exclusive']}"
    assert len(commands) == 1
    assert commands[0][-1] == report["requirement"]
    assert not any("setuptools" in part or "wheel" in part
                   for command in commands for part in command)


def test_the_bootstrap_never_issues_an_unbounded_upgrade() -> None:
    source = (REPO / "bootstrap.py").read_text(encoding="utf-8")
    assert '"--upgrade", "pip"' not in source, (
        "an unbounded `pip install --upgrade pip` makes the resolver that chose "
        "this project's dependency set whichever pip was newest that day")
    upgrades = [line for line in source.splitlines() if '"--upgrade"' in line]
    assert len(upgrades) == 1, upgrades


def test_no_application_dependency_is_installed_without_its_requirement_file(
        contract: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Everything installed comes from a pinned file, never from an argument."""
    commands: list[list[str]] = []
    monkeypatch.setattr(boot, "_run", lambda command, **_k: commands.append(command))
    monkeypatch.setattr(boot, "ensure_pip_tooling",
                        lambda *_a, **_k: {"action": "KEPT"})
    boot.install_requirements(Path("python"), contract["profiles"]["cpu"],
                              quiet=True, contract=contract)
    installs = [command for command in commands if "install" in command]
    assert installs
    for command in installs:
        assert "-r" in command or "-e" in command, command


# --- the environment is not called ready while it is not ---------------------

def test_a_scientific_profile_requires_the_onnx_import_before_adoption(
        contract: dict) -> None:
    """A rehearsal never opens an ONNX session; a scientific run does."""
    scientific = boot.import_groups(contract, scientific=True)
    rehearsal = boot.import_groups(contract, scientific=False)
    assert "science_only" in scientific
    assert "science_only" not in rehearsal
    assert "onnxruntime" in contract["import_checks"]["science_only"]


def test_an_environment_missing_a_declared_import_is_not_recorded_as_ready(
        contract: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Adoption is decided by measurement, and a failed measurement blocks."""
    monkeypatch.setattr(boot, "VENV", tmp_path / ".venv")
    monkeypatch.setattr(boot, "MANIFEST", tmp_path / "ENVIRONMENT_MANIFEST.json")
    monkeypatch.setattr(boot, "running_inside_project_venv", lambda: False)
    monkeypatch.setattr(boot, "read_manifest", lambda *_a, **_k: None)
    monkeypatch.setattr(boot, "detect_gpu", lambda: {"available": False})
    monkeypatch.setattr(boot, "classify_venv",
                        lambda *_a, **_k: {"state": boot.VENV_VALID,
                                           "action": boot.VENV_REUSE, "why": "stub"})
    monkeypatch.setattr(boot, "verify_imports",
                        lambda *_a, **_k: {"ok": False, "missing": ["onnxruntime"],
                                           "modules": [], "groups": []})
    monkeypatch.setattr(boot, "install_requirements",
                        lambda *_a, **_k: {"offline_wheelhouse_used": False})

    with pytest.raises(boot.BootstrapError) as caught:
        boot.ensure_environment(quiet=True, allow_install=True)
    assert "onnxruntime" in str(caught.value)
    assert not (tmp_path / "ENVIRONMENT_MANIFEST.json").exists()


# --- the platform a profile can actually be installed on ---------------------
#
# Auditing the CUDA plan turned up a third defect, latent behind the two the
# deployment reported: the cu129 index publishes torch 2.13.0 for Linux only.
# On Windows that profile would not have failed loudly, because the CUDA
# requirement files carry --extra-index-url and PyPI stays in the resolution
# set: pip would have installed a different build while the manifest went on
# naming the CUDA index it never used.

CUDA_EVIDENCE = REPO / "reports" / "handoff" / "CUDA_DEPENDENCY_PLAN_EVIDENCE.json"


@pytest.fixture(scope="module")
def cuda_evidence() -> dict:
    assert CUDA_EVIDENCE.exists(), (
        "the CUDA dependency-plan evidence is missing "
        "(scripts/audit_cuda_dependency_plan.py)")
    return json.loads(CUDA_EVIDENCE.read_text(encoding="utf-8"))


def test_every_profile_declares_the_platforms_it_publishes_for(contract: dict) -> None:
    for profile_id, profile in contract["profiles"].items():
        assert profile.get("platforms"), profile_id


def test_the_declared_platforms_match_the_measured_index(contract: dict,
                                                         cuda_evidence: dict) -> None:
    """The contract may not claim a wheel the index does not publish."""
    published = contract["platform_wheels"]["published"]
    for profile_id, profile in contract["profiles"].items():
        measured = cuda_evidence["profiles"][profile_id][
            "platforms_with_a_declared_wheel"]
        declared = set(profile["platforms"])
        assert declared <= set(measured), (profile_id, sorted(declared), measured)
        for platform, claimed in published[profile_id].items():
            assert claimed == (platform in declared), (profile_id, platform)


def test_the_windows_gap_that_started_this_is_recorded_not_papered_over(
        contract: dict, cuda_evidence: dict) -> None:
    assert contract["platform_wheels"]["published"]["cuda-cu129"]["win_amd64"] is False
    assert "win_amd64" not in contract["profiles"]["cuda-cu129"]["platforms"]
    assert "win_amd64" in cuda_evidence["profiles"]["cuda-cu129"][
        "platforms_missing_a_declared_wheel"]


def test_a_profile_with_no_wheel_for_the_host_is_incompatible(contract: dict) -> None:
    verdict = boot.classify_candidate(contract["profiles"]["cuda-cu129"],
                                      capability="12.0", driver=(580, 88),
                                      platform_tag=boot.WIN_AMD64)
    assert verdict["grade"] == boot.INCOMPATIBLE
    assert "no win_amd64 wheel" in verdict["why"]
    assert verdict["platform_satisfied"] is False
    # ... and the same card on Linux is fine, so the gate is about the platform.
    assert boot.classify_candidate(contract["profiles"]["cuda-cu129"],
                                   capability="12.0", driver=(580, 88),
                                   platform_tag=boot.LINUX_X86_64
                                   )["grade"] != boot.INCOMPATIBLE


def test_a_windows_blackwell_host_resolves_to_a_profile_that_has_a_windows_wheel(
        contract: dict) -> None:
    gpu = {"available": True, "name": "NVIDIA GeForce RTX 5090",
           "driver_version": "580.88", "compute_capability": "12.0"}
    selection = boot.select_profile(contract, gpu, platform_tag=boot.WIN_AMD64)
    assert selection["profile_id"] == "cuda-cu130"
    assert boot.WIN_AMD64 in contract["profiles"]["cuda-cu130"]["platforms"]
    assert selection["supports_scientific_execution"] is True


def test_the_linux_plan_is_exactly_what_it_was_before_the_platform_gate(
        contract: dict) -> None:
    """Windows was not fixed by moving Linux."""
    for driver, capability, expected in (("580.88", "12.0", "cuda-cu129"),
                                         ("572.00", "12.0", "cuda-cu129"),
                                         ("575.00", "9.0", "cuda-cu129"),
                                         ("550.54", "8.9", "cuda-cu126"),
                                         ("535.10", "8.0", "cuda-cu126")):
        gpu = {"available": True, "name": "NVIDIA Test", "driver_version": driver,
               "compute_capability": capability}
        selection = boot.select_profile(contract, gpu,
                                        platform_tag=boot.LINUX_X86_64)
        assert selection["profile_id"] == expected, (driver, capability)


def test_a_windows_host_with_no_installable_profile_blocks_and_says_why(
        contract: dict) -> None:
    """Blackwell on a driver too old for CUDA 13 has nowhere to go on Windows."""
    gpu = {"available": True, "name": "NVIDIA GeForce RTX 5070",
           "driver_version": "575.00", "compute_capability": "12.0"}
    with pytest.raises(boot.BootstrapError) as caught:
        boot.select_profile(contract, gpu, platform_tag=boot.WIN_AMD64)
    assert caught.value.reason == boot.CUDA_NOT_VALIDATED
    assert "win_amd64" in str(caught.value)
    assert "publishes no win_amd64 wheel" in str(caught.value)


@pytest.mark.parametrize("sysconfig_platform,expected", [
    ("win-amd64", "win_amd64"),
    ("linux-x86_64", "linux_x86_64"),
    ("linux-aarch64", "linux_aarch64"),
    ("macosx-14.0-arm64", "macosx_arm64"),
    ("mingw_x86_64_ucrt", "unknown_platform"),
])
def test_the_host_platform_is_read_from_the_wheel_platform_string(
        sysconfig_platform: str, expected: str) -> None:
    """MSYS2 reports its own build platform, and it is not a wheel target here."""
    assert boot.host_platform_tag(
        {"sysconfig_platform": sysconfig_platform}) == expected


def test_the_cuda_requirement_file_and_the_contract_name_the_same_index(
        cuda_evidence: dict) -> None:
    for profile_id, entry in cuda_evidence["profiles"].items():
        assert entry["index_matches_contract"], profile_id
        assert entry["torch_matches_contract"], profile_id


# --- what was installed, not what was asked for ------------------------------

TORCH_PROBE_OUTPUT = '{"version": "%s", "cuda": "%s"}\n'


def test_a_torch_build_carrying_the_declared_tag_is_accepted(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boot.subprocess, "check_output",
                        lambda *_a, **_k: TORCH_PROBE_OUTPUT % ("2.13.0+cu130", "13.0"))
    report = boot.verify_torch_build(Path("python"), {"cuda_tag": "cu130"})
    assert report["verdict"] == "MATCHES_DECLARED_PROFILE"
    assert report["local_version_label"] == "cu130"


def test_a_silently_substituted_pypi_wheel_is_caught(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The PyPI wheel carries no local version label; the profile's build does."""
    monkeypatch.setattr(boot.subprocess, "check_output",
                        lambda *_a, **_k: TORCH_PROBE_OUTPUT % ("2.13.0", "13.0"))
    report = boot.verify_torch_build(Path("python"), {"cuda_tag": "cu129"})
    assert report["verdict"] == "SUBSTITUTED_BUILD"
    assert "cu129" in report["why"]


def test_a_wheel_from_the_wrong_cuda_index_is_caught(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boot.subprocess, "check_output",
                        lambda *_a, **_k: TORCH_PROBE_OUTPUT % ("2.13.0+cu126", "12.6"))
    assert boot.verify_torch_build(Path("python"), {"cuda_tag": "cu130"}
                                   )["verdict"] == "SUBSTITUTED_BUILD"


def test_the_cpu_profile_has_no_build_tag_to_verify() -> None:
    assert boot.verify_torch_build(Path("python"), {})["verdict"] == "NOT_APPLICABLE"


def test_a_substituted_build_blocks_the_run_rather_than_being_recorded(
        contract: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(boot, "VENV", tmp_path / ".venv")
    monkeypatch.setattr(boot, "MANIFEST", tmp_path / "ENVIRONMENT_MANIFEST.json")
    monkeypatch.setattr(boot, "running_inside_project_venv", lambda: False)
    monkeypatch.setattr(boot, "read_manifest", lambda *_a, **_k: None)
    monkeypatch.setattr(boot, "detect_gpu", lambda: {
        "available": True, "name": "NVIDIA GeForce RTX 5090",
        "driver_version": "580.88", "compute_capability": "12.0"})
    monkeypatch.setattr(boot, "classify_venv",
                        lambda *_a, **_k: {"state": boot.VENV_VALID,
                                           "action": boot.VENV_REUSE, "why": "stub"})
    monkeypatch.setattr(boot, "verify_imports", lambda *_a, **_k: {
        "ok": True, "missing": [], "modules": [], "groups": []})
    monkeypatch.setattr(boot, "verify_torch_build", lambda *_a, **_k: {
        "verdict": "SUBSTITUTED_BUILD", "declared_cuda_tag": "cu130",
        "installed_version": "2.13.0", "installed_cuda": "13.0", "why": "stub"})

    with pytest.raises(boot.BootstrapError) as caught:
        boot.ensure_environment(quiet=True, allow_install=False)
    assert caught.value.reason == boot.CUDA_NOT_VALIDATED
    assert "not the build this profile declares" in str(caught.value)
    assert not (tmp_path / "ENVIRONMENT_MANIFEST.json").exists()
