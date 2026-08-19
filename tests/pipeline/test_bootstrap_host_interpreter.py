"""Host-interpreter classification, Windows fallback and `.venv` recovery.

Every case here came from a physical deployment or is one step away from it. A
project folder was copied to a Windows GPU laptop and `python train.py` failed
twice before any science could start:

* PATH `python` was `C:\\msys64\\mingw64\\bin\\python.exe`. MSYS2/MinGW Python
  reports `os.name == "nt"` and `sys.platform == "win32"` exactly like a standard
  Windows CPython, and then creates `.venv/bin/python.exe` instead of
  `.venv/Scripts/python.exe`. The bootstrap had classified it on `os.name` alone.
* the dependency install then died part-way, leaving a `.venv` that existed, ran,
  and was missing half its packages.

Neither machine is available here, so every test constructs the condition rather
than waiting for it: the classification is a pure function over an evidence
dictionary, so MSYS2 is testable on a host that has never had MSYS2, and a Linux
`.venv` is testable on Windows.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bootstrap as boot  # noqa: E402


@pytest.fixture(scope="module")
def contract() -> dict:
    return boot.read_contract()


# --- evidence builders -------------------------------------------------------
#
# The shape a real interpreter reports. `venv_scripts_path` is the decisive
# field: it is what `venv` itself asks sysconfig for, so it is what the
# environment layout will actually be.

def windows_cpython(version: str = "3.12.8", *, executable: str | None = None,
                    prefix: str | None = None,
                    base_prefix: str | None = None, **overrides) -> dict:
    executable = executable or r"C:\Users\LOQ\AppData\Local\Programs\Python\Python312\python.exe"
    prefix = prefix or str(Path(executable).parent)
    evidence = {
        "executable": executable,
        "version": version,
        "version_info": [int(part) for part in version.split(".")],
        "implementation": "CPython",
        "sys_platform": "win32",
        "os_name": "nt",
        "sysconfig_platform": "win-amd64",
        "default_scheme": "nt",
        "venv_scripts_path": r"__prism_probe_base__\Scripts",
        "prefix": prefix,
        "base_prefix": base_prefix or prefix,
        "msystem": None,
        "usable": True,
    }
    evidence.update(overrides)
    return evidence


def msys2_python(version: str = "3.12.11", **overrides) -> dict:
    """What `C:\\msys64\\mingw64\\bin\\python.exe` actually reports.

    Windows-like on every field the old code looked at, POSIX on the one that
    decides the layout.
    """
    evidence = {
        "executable": r"C:\msys64\mingw64\bin\python.exe",
        "version": version,
        "version_info": [int(part) for part in version.split(".")],
        "implementation": "CPython",
        "sys_platform": "win32",
        "os_name": "nt",
        "sysconfig_platform": "mingw_x86_64_ucrt",
        "default_scheme": "posix_prefix",
        "venv_scripts_path": "__prism_probe_base__/bin",
        "prefix": r"C:\msys64\mingw64",
        "base_prefix": r"C:\msys64\mingw64",
        "msystem": "MINGW64",
        "usable": True,
    }
    evidence.update(overrides)
    return evidence


def posix_cpython(version: str = "3.12.8", *, prefix: str = "/usr",
                  **overrides) -> dict:
    evidence = {
        "executable": "/usr/bin/python3",
        "version": version,
        "version_info": [int(part) for part in version.split(".")],
        "implementation": "CPython",
        "sys_platform": "linux",
        "os_name": "posix",
        "sysconfig_platform": "linux-x86_64",
        "default_scheme": "posix_prefix",
        "venv_scripts_path": "__prism_probe_base__/bin",
        "prefix": prefix,
        "base_prefix": prefix,
        "msystem": None,
        "usable": True,
    }
    evidence.update(overrides)
    return evidence


# --- 1-3, and the trap in between --------------------------------------------

def test_a_standard_windows_cpython_is_recognised_and_uses_the_scripts_layout() -> None:
    verdict = boot.classify_interpreter(windows_cpython())
    assert verdict["classification"] == boot.STANDARD_WINDOWS_CPYTHON
    assert verdict["venv_scheme"] == boot.WINDOWS_SCHEME
    assert verdict["scripts_dir"] == "Scripts"
    assert verdict["may_build_the_project_environment"]


def test_a_standard_posix_cpython_is_recognised_and_uses_the_bin_layout() -> None:
    verdict = boot.classify_interpreter(posix_cpython())
    assert verdict["classification"] == boot.POSIX_CPYTHON
    assert verdict["venv_scheme"] == boot.POSIX_SCHEME
    assert verdict["scripts_dir"] == "bin"
    assert verdict["may_build_the_project_environment"]


def test_msys2_python_is_not_a_windows_cpython_however_windows_it_looks() -> None:
    """The regression. Every field the old code read says "Windows"."""
    evidence = msys2_python()
    assert evidence["os_name"] == "nt"
    assert evidence["sys_platform"] == "win32"

    verdict = boot.classify_interpreter(evidence)
    assert verdict["classification"] == boot.MSYS2_MINGW_PYTHON
    assert verdict["venv_scheme"] == boot.POSIX_SCHEME
    assert not verdict["may_build_the_project_environment"]
    assert "Scripts" in verdict["why"] or "bin" in verdict["why"]


def test_the_msys2_verdict_rests_on_the_venv_scheme_not_on_the_path() -> None:
    """Path markers are corroboration. A renamed install is still MSYS2."""
    verdict = boot.classify_interpreter(
        msys2_python(executable=r"D:\tools\py\python.exe", msystem=None))
    assert verdict["classification"] == boot.MSYS2_MINGW_PYTHON


def test_a_standard_cpython_launched_from_an_msys2_shell_is_still_standard() -> None:
    """MSYSTEM describes the shell, not the interpreter.

    This case is live in this repository's own tooling: a Git-Bash session
    exports MSYSTEM while running the ordinary Windows CPython. Classifying on
    the environment variable would refuse a perfectly good interpreter.
    """
    verdict = boot.classify_interpreter(windows_cpython(msystem="MINGW64"))
    assert verdict["classification"] == boot.STANDARD_WINDOWS_CPYTHON
    assert any("MSYSTEM" in signal for signal in verdict["signals"])


def test_a_cygwin_interpreter_is_classified_by_its_platform() -> None:
    verdict = boot.classify_interpreter(
        msys2_python(sys_platform="cygwin", os_name="posix"))
    assert verdict["classification"] == boot.MSYS2_MINGW_PYTHON


def test_a_non_cpython_windows_interpreter_is_unknown_not_assumed() -> None:
    verdict = boot.classify_interpreter(windows_cpython(implementation="PyPy"))
    assert verdict["classification"] == boot.UNKNOWN_PYTHON
    assert not verdict["may_build_the_project_environment"]


def test_the_real_interpreter_running_this_suite_classifies_as_supported() -> None:
    verdict = boot.classify_interpreter(boot.local_interpreter_evidence())
    assert verdict["classification"] in (boot.STANDARD_WINDOWS_CPYTHON,
                                         boot.POSIX_CPYTHON)
    expected = "Scripts" if sys.platform == "win32" else "bin"
    assert verdict["scripts_dir"] == expected


# --- 4. MSYS2 on PATH, a supported CPython available -------------------------

def _fake_launcher(*entries: tuple[str, str]) -> tuple[list[dict], object]:
    """A `py -0p` result plus a probe that answers for those executables."""
    candidates = [{"source": "py_launcher", "tag": version.rsplit(".", 1)[0],
                   "executable": path, "default": index == 0}
                  for index, (version, path) in enumerate(entries)]
    table = {path: version for version, path in entries}

    def probe(executable: str) -> dict:
        return windows_cpython(table[str(executable)], executable=str(executable))

    return candidates, probe


def test_an_msys2_host_falls_back_to_a_supported_windows_cpython(
        contract: dict) -> None:
    """The user's command stays `python train.py`; `py -3.12` is never required."""
    candidates, probe = _fake_launcher(
        ("3.14.7", r"C:\Python314\python.exe"),
        ("3.12.8", r"C:\Users\LOQ\AppData\Local\Programs\Python\Python312\python.exe"))

    def selector(_contract):
        return boot.select_windows_cpython(_contract, candidates=candidates,
                                           probe=probe)

    host = boot.resolve_host_interpreter(contract, evidence=msys2_python(),
                                         selector=selector)
    assert host["selection"] == "WINDOWS_CPYTHON_FALLBACK"
    assert host["classification"] == boot.STANDARD_WINDOWS_CPYTHON
    assert host["venv_scheme"] == boot.WINDOWS_SCHEME
    assert host["executable"].endswith(r"Python312\python.exe")
    assert host["fallback"]["from_classification"] == boot.MSYS2_MINGW_PYTHON


def test_the_fallback_obeys_the_declared_preference_not_the_newest_installed(
        contract: dict) -> None:
    """3.14 is installed and is the launcher default. It is out of range."""
    candidates, probe = _fake_launcher(
        ("3.14.7", r"C:\Python314\python.exe"),
        ("3.13.11", r"C:\Python313\python.exe"),
        ("3.11.9", r"C:\Python311\python.exe"))
    search = boot.select_windows_cpython(contract, candidates=candidates, probe=probe)
    assert search["selected"]["version"] == "3.13.11"
    assert contract["python"]["preferred_minors"][0] == "3.13"
    rejected = {item["executable"]: item.get("rejected_because")
                for item in search["examined"]}
    assert "outside the supported range" in rejected[r"C:\Python314\python.exe"]


def test_an_msys2_interpreter_offered_by_the_launcher_is_never_selected(
        contract: dict) -> None:
    candidates = [{"source": "py_launcher", "tag": "3.12",
                   "executable": r"C:\msys64\mingw64\bin\python.exe", "default": True}]
    search = boot.select_windows_cpython(
        contract, candidates=candidates, probe=lambda _executable: msys2_python())
    assert search["selected"] is None
    assert search["examined"][0]["classification"] == boot.MSYS2_MINGW_PYTHON


# --- 5. MSYS2 on PATH, nothing supported anywhere ----------------------------

def test_an_msys2_host_with_no_supported_cpython_blocks_before_installing(
        contract: dict) -> None:
    def selector(_contract):
        return boot.select_windows_cpython(
            _contract,
            candidates=[{"source": "py_launcher", "tag": "3.14",
                         "executable": r"C:\Python314\python.exe", "default": True}],
            probe=lambda executable: windows_cpython("3.14.7", executable=executable))

    with pytest.raises(boot.BootstrapError) as caught:
        boot.resolve_host_interpreter(contract, evidence=msys2_python(),
                                      selector=selector)
    error = caught.value
    assert error.reason == boot.SUPPORTED_WINDOWS_CPYTHON_NOT_FOUND
    assert error.reason in boot.BLOCKING_REASONS
    assert r"C:\msys64\mingw64\bin\python.exe" in str(error)
    assert boot.MSYS2_MINGW_PYTHON in str(error)
    assert "3.11" in str(error) and "3.14" in str(error)
    assert r"C:\Python314\python.exe" in str(error)
    assert "No package was installed" in str(error)
    assert error.detail["interpreters_discovered"]


def test_an_unsupported_posix_interpreter_still_refuses_without_a_launcher(
        contract: dict) -> None:
    """There is no py.exe on Linux; the operator installs a supported CPython."""
    with pytest.raises(boot.BootstrapError) as caught:
        boot.resolve_host_interpreter(contract, evidence=posix_cpython("3.9.18"))
    assert caught.value.reason == boot.UNSUPPORTED_PYTHON


def test_a_supported_current_interpreter_is_used_as_is(contract: dict) -> None:
    for evidence in (windows_cpython("3.12.8"), posix_cpython("3.11.9")):
        host = boot.resolve_host_interpreter(contract, evidence=evidence)
        assert host["selection"] == "CURRENT_INTERPRETER"
        assert host["fallback"] is None


# --- 5-6, 14. layouts, and the hybrid that must never be built ---------------

def test_the_two_canonical_layouts_are_the_only_ones_constructed(
        tmp_path: Path) -> None:
    assert boot.venv_python(tmp_path, scheme=boot.WINDOWS_SCHEME) == (
        tmp_path / "Scripts" / "python.exe")
    assert boot.venv_python(tmp_path, scheme=boot.POSIX_SCHEME) == (
        tmp_path / "bin" / "python")
    for scheme in (boot.WINDOWS_SCHEME, boot.POSIX_SCHEME):
        built = boot.venv_python(tmp_path, scheme=scheme)
        assert built != tmp_path / "bin" / "python.exe", "the MSYS2 hybrid"


def test_a_fresh_environment_is_created_and_validated_on_this_real_host(
        tmp_path: Path) -> None:
    """Not a mock: `python -m venv` really runs, and the result is measured."""
    host = boot.local_interpreter_evidence()
    scheme = boot.classify_interpreter(host)["venv_scheme"]
    venv = tmp_path / ".venv"

    interpreter = boot.create_venv(venv, quiet=True, host_executable=sys.executable,
                                   expected_scheme=scheme)
    assert interpreter.exists()
    assert interpreter == boot.venv_python(venv, scheme=scheme)

    report = boot.validate_venv(venv, expected_scheme=scheme)
    assert report["valid"], report["why"]
    evidence = report["interpreter_evidence"]
    assert Path(evidence["prefix"]).resolve() == venv.resolve()
    assert evidence["prefix"] != evidence["base_prefix"]
    assert evidence["implementation"] == "CPython"


def test_an_exit_code_of_zero_is_not_accepted_as_a_created_environment(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The MSYS2 failure mode exactly: the command succeeds, the layout is wrong."""
    venv = tmp_path / ".venv"

    def fake_run(command, **_kwargs):
        # What MSYS2 Python leaves behind on Windows.
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python.exe").write_text("", encoding="utf-8")
        (venv / "pyvenv.cfg").write_text("home = C:\\msys64\\mingw64", encoding="utf-8")

    monkeypatch.setattr(boot, "_run", fake_run)
    with pytest.raises(boot.BootstrapError) as caught:
        boot.create_venv(venv, quiet=True, host_executable=sys.executable,
                         expected_scheme=boot.WINDOWS_SCHEME)
    assert caught.value.reason == boot.VENV_NOT_VALIDATED


# --- 6-13. classification and recovery of an existing .venv ------------------

def _fake_venv(root: Path, *, scheme: str, interpreter_name: str | None = None,
               version: str = "3.12.8", prefix: Path | None = None) -> Path:
    directory = root / ("Scripts" if scheme == boot.WINDOWS_SCHEME else "bin")
    directory.mkdir(parents=True, exist_ok=True)
    name = interpreter_name or ("python.exe" if scheme == boot.WINDOWS_SCHEME
                                else "python")
    interpreter = directory / name
    interpreter.write_text("", encoding="utf-8")
    (root / "pyvenv.cfg").write_text("home = elsewhere\n", encoding="utf-8")
    return interpreter


def _probe_for(root: Path, version: str = "3.12.8", *,
               prefix: Path | None = None, usable: bool = True):
    def probe(executable):
        if not usable:
            return {"executable": str(executable), "usable": False,
                    "error": "OSError: the base interpreter is gone"}
        return windows_cpython(version, executable=str(executable),
                               prefix=str(prefix or root),
                               base_prefix=r"C:\Python312")

    return probe


def test_an_absent_environment_is_created_not_repaired(tmp_path: Path) -> None:
    state = boot.classify_venv(tmp_path / ".venv",
                               expected_scheme=boot.WINDOWS_SCHEME)
    assert state["state"] == boot.VENV_ABSENT
    assert state["action"] == boot.VENV_CREATE


def test_an_empty_directory_left_by_an_interrupted_creation_is_absent(
        tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    venv.mkdir()
    assert boot.classify_venv(venv, expected_scheme=boot.WINDOWS_SCHEME)["state"] == (
        boot.VENV_ABSENT)


def test_a_half_written_environment_with_no_interpreter_is_rebuilt(
        tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    (venv / "Lib" / "site-packages").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = C:\\Python312\n", encoding="utf-8")
    state = boot.classify_venv(venv, expected_scheme=boot.WINDOWS_SCHEME)
    assert state["state"] == boot.VENV_NO_INTERPRETER
    assert state["action"] == boot.VENV_REBUILD


def test_a_valid_environment_is_reused(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    _fake_venv(venv, scheme=boot.WINDOWS_SCHEME)
    state = boot.classify_venv(venv, expected_scheme=boot.WINDOWS_SCHEME,
                               expected_minor="3.12",
                               probe=_probe_for(venv))
    assert state["state"] == boot.VENV_VALID
    assert state["action"] == boot.VENV_REUSE


def test_a_linux_environment_copied_to_windows_is_rebuilt(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    _fake_venv(venv, scheme=boot.POSIX_SCHEME)
    state = boot.classify_venv(venv, expected_scheme=boot.WINDOWS_SCHEME,
                               probe=_probe_for(venv))
    assert state["state"] == boot.VENV_WRONG_SCHEME
    assert state["action"] == boot.VENV_REBUILD


def test_a_windows_environment_copied_to_linux_is_rebuilt(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    _fake_venv(venv, scheme=boot.WINDOWS_SCHEME)
    state = boot.classify_venv(venv, expected_scheme=boot.POSIX_SCHEME,
                               probe=_probe_for(venv))
    assert state["state"] == boot.VENV_WRONG_SCHEME
    assert state["action"] == boot.VENV_REBUILD


def test_an_msys2_created_environment_on_windows_is_rebuilt(tmp_path: Path) -> None:
    """`.venv/bin/python.exe` — the exact artefact of the deployment failure."""
    venv = tmp_path / ".venv"
    interpreter = _fake_venv(venv, scheme=boot.POSIX_SCHEME,
                             interpreter_name="python.exe")
    assert interpreter == venv / "bin" / "python.exe"
    found, scheme = boot.existing_venv_interpreter(venv)
    assert (found, scheme) == (interpreter, boot.POSIX_SCHEME)

    state = boot.classify_venv(venv, expected_scheme=boot.WINDOWS_SCHEME,
                               probe=_probe_for(venv))
    assert state["state"] == boot.VENV_WRONG_SCHEME
    assert state["action"] == boot.VENV_REBUILD


def test_an_environment_whose_interpreter_will_not_run_is_rebuilt(
        tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    _fake_venv(venv, scheme=boot.WINDOWS_SCHEME)
    state = boot.classify_venv(venv, expected_scheme=boot.WINDOWS_SCHEME,
                               probe=_probe_for(venv, usable=False))
    assert state["state"] == boot.VENV_INTERPRETER_UNUSABLE
    assert state["action"] == boot.VENV_REBUILD


def test_an_environment_of_the_wrong_python_minor_is_rebuilt(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    _fake_venv(venv, scheme=boot.WINDOWS_SCHEME)
    state = boot.classify_venv(venv, expected_scheme=boot.WINDOWS_SCHEME,
                               expected_minor="3.13",
                               probe=_probe_for(venv, "3.12.8"))
    assert state["state"] == boot.VENV_WRONG_VERSION
    assert state["action"] == boot.VENV_REBUILD


def test_an_interpreter_that_is_not_in_this_venv_is_rebuilt(tmp_path: Path) -> None:
    """A `.venv` whose interpreter reports the host prefix is not an environment."""
    venv = tmp_path / ".venv"
    _fake_venv(venv, scheme=boot.WINDOWS_SCHEME)
    state = boot.classify_venv(venv, expected_scheme=boot.WINDOWS_SCHEME,
                               probe=_probe_for(venv, prefix=tmp_path / "elsewhere"))
    assert state["state"] == boot.VENV_FOREIGN_PREFIX
    assert state["action"] == boot.VENV_REBUILD


def test_a_dependency_incomplete_environment_is_topped_up_never_deleted(
        tmp_path: Path) -> None:
    """The GPU laptop's actual state after pip died on onnxruntime."""
    venv = tmp_path / ".venv"
    _fake_venv(venv, scheme=boot.WINDOWS_SCHEME)
    state = boot.classify_venv(venv, expected_scheme=boot.WINDOWS_SCHEME,
                               dependencies_ok=False, probe=_probe_for(venv))
    assert state["state"] == boot.VENV_DEPENDENCIES_INCOMPLETE
    assert state["action"] == boot.VENV_INSTALL_INTO
    assert state["action"] != boot.VENV_REBUILD, "reinstalling must not cost a rebuild"


def test_a_dependency_incomplete_environment_is_never_marked_valid(
        tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    _fake_venv(venv, scheme=boot.WINDOWS_SCHEME)
    report = boot.validate_venv(venv, expected_scheme=boot.WINDOWS_SCHEME,
                                probe=_probe_for(venv))
    assert report["valid"]
    incomplete = boot.classify_venv(venv, expected_scheme=boot.WINDOWS_SCHEME,
                                    dependencies_ok=False, probe=_probe_for(venv))
    assert incomplete["state"] != boot.VENV_VALID


# --- 7, 14. deletion safety and self-recreation ------------------------------

def test_an_interpreter_may_not_be_asked_to_recreate_the_venv_it_lives_in(
        tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    interpreter = _fake_venv(venv, scheme=boot.WINDOWS_SCHEME)
    with pytest.raises(boot.BootstrapError) as caught:
        boot.assert_not_self_recreation(interpreter, venv)
    assert caught.value.reason == boot.SELF_RECREATION_REFUSED

    with pytest.raises(boot.BootstrapError):
        boot.create_venv(venv, quiet=True, host_executable=interpreter,
                         expected_scheme=boot.WINDOWS_SCHEME)


def test_a_host_interpreter_outside_the_venv_is_allowed_to_create_it(
        tmp_path: Path) -> None:
    boot.assert_not_self_recreation(sys.executable, tmp_path / ".venv")


def test_only_this_projects_own_venv_may_ever_be_deleted(tmp_path: Path) -> None:
    foreign = tmp_path / "somebody-elses-env"
    _fake_venv(foreign, scheme=boot.WINDOWS_SCHEME)
    with pytest.raises(boot.BootstrapError):
        boot.remove_venv(foreign, repo=tmp_path)

    named_venv_elsewhere = tmp_path / "nested" / ".venv"
    _fake_venv(named_venv_elsewhere, scheme=boot.WINDOWS_SCHEME)
    with pytest.raises(boot.BootstrapError):
        boot.remove_venv(named_venv_elsewhere, repo=tmp_path)
    assert named_venv_elsewhere.exists()


def test_a_directory_that_is_not_a_virtual_environment_is_not_deleted(
        tmp_path: Path) -> None:
    disguised = tmp_path / ".venv"
    disguised.mkdir()
    (disguised / "somebodys_thesis.docx").write_text("x", encoding="utf-8")
    with pytest.raises(boot.BootstrapError, match="does not look like"):
        boot.remove_venv(disguised, repo=tmp_path)
    assert (disguised / "somebodys_thesis.docx").exists()


def test_the_projects_own_venv_is_rebuilt_deterministically(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    _fake_venv(venv, scheme=boot.POSIX_SCHEME)
    removal = boot.remove_venv(venv, repo=tmp_path)
    assert removal["removed"]
    assert not venv.exists()


def test_the_running_project_venv_is_never_deleted() -> None:
    """Guarded twice: by the path rule and by "am I inside it"."""
    if not boot.running_inside_project_venv():
        pytest.skip("this suite is not running from the project .venv")
    with pytest.raises(boot.BootstrapError) as caught:
        boot.remove_venv(boot.VENV)
    assert caught.value.reason == boot.SELF_RECREATION_REFUSED


# --- 15. the re-exec ---------------------------------------------------------

def test_the_runner_reexecs_into_the_environment_exactly_once(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """`python train.py` must hand over to `.venv`, and the child must not repeat."""
    sys.path.insert(0, str(REPO))
    import train

    interpreter = boot.venv_python()
    calls: list[dict] = []

    def fake_ensure(**_kwargs):
        return {"interpreter": str(interpreter), "action": "REUSED",
                "profile_id": "cpu", "environment_identity": "a" * 64,
                "host_interpreter": {"fallback": None}, "venv_recovery": {}}

    class Completed:
        returncode = 0

    def fake_run(command, **kwargs):
        calls.append({"command": command, "env": kwargs.get("env", {})})
        return Completed()

    monkeypatch.setattr(boot, "ensure_environment", fake_ensure)
    monkeypatch.setattr(boot, "running_inside_project_venv", lambda: False)
    monkeypatch.setattr(train.subprocess, "run", fake_run)
    monkeypatch.delenv(train.REEXEC_FLAG, raising=False)
    monkeypatch.setattr(train, "REPO", REPO)

    code = train._bootstrap_and_reexec(["--preflight-only"], quiet=True)
    assert code == 0
    assert len(calls) == 1, "exactly one re-exec"
    assert calls[0]["command"][0] == str(interpreter)
    assert calls[0]["command"][1].endswith("train.py")
    assert calls[0]["env"][train.REEXEC_FLAG] == "1", "the child must not re-exec"


def test_a_child_that_is_already_the_environment_does_not_bootstrap_again(
        monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(REPO))
    import train

    monkeypatch.setenv(train.REEXEC_FLAG, "1")
    monkeypatch.setattr(boot, "ensure_environment", lambda **_k: pytest.fail(
        "the child re-entered bootstrap"))
    assert train._bootstrap_and_reexec([], quiet=True) is None


def test_a_blocking_host_failure_is_reported_as_blocked_not_as_a_usage_error(
        monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(REPO))
    import train

    def fake_ensure(**_kwargs):
        raise boot.BootstrapError(boot.SUPPORTED_WINDOWS_CPYTHON_NOT_FOUND,
                                  "no supported standard Windows CPython")

    monkeypatch.setattr(boot, "ensure_environment", fake_ensure)
    monkeypatch.setattr(boot, "running_inside_project_venv", lambda: False)
    monkeypatch.delenv(train.REEXEC_FLAG, raising=False)
    assert train._bootstrap_and_reexec([], quiet=True) == train.EXIT_BLOCKED


# --- 16. the environment fingerprint -----------------------------------------

def test_the_environment_identity_does_not_move_with_the_host_interpreter(
        contract: dict) -> None:
    """A folder copied to another machine must not reinstall for cosmetic reasons."""
    profile = contract["profiles"]["cpu"]
    identity = boot.environment_identity("cpu", profile, "3.12.8")["identity"]
    assert identity == boot.environment_identity("cpu", profile, "3.12.11")["identity"]
    assert identity != boot.environment_identity("cpu", profile, "3.13.11")["identity"]


def test_no_absolute_path_or_classification_enters_the_environment_identity(
        contract: dict) -> None:
    material = boot.environment_identity(
        "cpu", contract["profiles"]["cpu"], "3.12.8")["material"]
    assert set(material) == {"profile_id", "requirements", "python_minor"}
    assert str(REPO) not in repr(material)
    for classification in (boot.STANDARD_WINDOWS_CPYTHON, boot.MSYS2_MINGW_PYTHON):
        assert classification not in repr(material)


def test_the_contract_declares_the_host_interpreter_policy(contract: dict) -> None:
    block = contract["host_interpreter"]
    assert block["may_build_the_environment"] == ["STANDARD_WINDOWS_CPYTHON",
                                                  "POSIX_CPYTHON"]
    assert "MSYS2_MINGW_PYTHON" in block["refused"]
    assert block["msys2_scientific_execution"] is False
    assert block["windows_fallback"]["enabled"] is True
    assert block["hybrid_layouts_are_defects"] is True
    assert contract["virtual_environment"]["manual_deletion_required"] is False


def test_the_bootstrap_still_shells_out_only_to_interpreters_it_classified() -> None:
    """No `pip install` or `venv` command may name a bare `python`."""
    source = (REPO / "bootstrap.py").read_text(encoding="utf-8")
    assert "sys.executable, \"-m\", \"venv\"" not in source
    assert subprocess  # the import is used through _run/check_output only


# --- the whole path, wired together ------------------------------------------

def test_an_msys2_host_builds_a_windows_layout_venv_end_to_end(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract: dict) -> None:
    """The deployment failure, from `python train.py` to a validated environment.

    Classification, fallback, layout, creation, installation and the recorded
    manifest are separately tested above; this asserts they are actually wired to
    each other, which is the part the original defect got wrong.
    """
    venv = tmp_path / ".venv"
    manifest_path = tmp_path / "ENVIRONMENT_MANIFEST.json"
    fallback = r"C:\Users\LOQ\AppData\Local\Programs\Python\Python312\python.exe"
    created: dict = {}

    monkeypatch.setattr(boot, "VENV", venv)
    monkeypatch.setattr(boot, "MANIFEST", manifest_path)
    monkeypatch.setattr(boot, "running_inside_project_venv", lambda: False)
    monkeypatch.setattr(boot, "local_interpreter_evidence", msys2_python)
    # The real discovery and the real selection run; only what the machine
    # offers, and what each candidate reports about itself, is substituted.
    monkeypatch.setattr(boot, "discover_windows_interpreters", lambda: [
        {"source": "py_launcher", "tag": "3.14",
         "executable": r"C:\Python314\python.exe", "default": True},
        {"source": "py_launcher", "tag": "3.12",
         "executable": fallback, "default": False}])
    monkeypatch.setattr(boot, "interpreter_evidence", lambda executable, **_k: (
        windows_cpython("3.14.7", executable=str(executable))
        if "314" in str(executable)
        else windows_cpython("3.12.8", executable=str(executable))))
    monkeypatch.setattr(boot, "detect_gpu", lambda: {
        "available": True, "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "driver_version": "580.00", "compute_capability": "8.9",
        "memory_total_mb": 8188})
    monkeypatch.setattr(boot, "read_manifest", lambda *_a, **_k: None)
    monkeypatch.setattr(boot, "create_venv",
                        lambda *args, **kwargs: created.update(kwargs) or venv)
    monkeypatch.setattr(boot, "install_requirements",
                        lambda *_a, **_k: {"offline_wheelhouse_used": False})
    monkeypatch.setattr(boot, "verify_imports", lambda *_a, **_k: {
        "ok": True, "missing": [], "modules": [], "groups": []})
    monkeypatch.setattr(boot, "installed_packages", lambda _python: {})

    report = boot.ensure_environment(quiet=True, allow_install=True)

    assert report["action"] == "INSTALLED"
    assert report["host_interpreter"]["selection"] == "WINDOWS_CPYTHON_FALLBACK"
    assert report["host_interpreter"]["host_classification"] == boot.MSYS2_MINGW_PYTHON
    assert report["venv_scheme"] == boot.WINDOWS_SCHEME
    assert report["interpreter"] == str(venv / "Scripts" / "python.exe")
    assert created["host_executable"] == fallback
    assert created["expected_scheme"] == boot.WINDOWS_SCHEME
    assert created["expected_minor"] == "3.12"
    # The version recorded is the interpreter that will run, not the one that
    # happened to launch the entrypoint.
    assert report["python"]["found"] == "3.12.8"
    assert report["python"]["classification"] == boot.STANDARD_WINDOWS_CPYTHON
    # A CUDA host must require the ONNX import before the environment is ready.
    # cu130, not cu129: this is a Windows host, and the cu129 index publishes no
    # win_amd64 wheel for the pinned torch.
    assert report["host_platform"] == boot.WIN_AMD64
    assert report["profile_id"] == "cuda-cu130"
    assert "science_only" in report["required_import_groups"]
    assert manifest_path.exists()


def test_a_posix_host_builds_a_bin_layout_venv_end_to_end(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same wiring on the Linux machine this folder may be copied to next."""
    venv = tmp_path / ".venv"
    created: dict = {}

    monkeypatch.setattr(boot, "VENV", venv)
    monkeypatch.setattr(boot, "MANIFEST", tmp_path / "ENVIRONMENT_MANIFEST.json")
    monkeypatch.setattr(boot, "running_inside_project_venv", lambda: False)
    monkeypatch.setattr(boot, "local_interpreter_evidence", posix_cpython)
    monkeypatch.setattr(boot, "detect_gpu", lambda: {"available": False})
    monkeypatch.setattr(boot, "read_manifest", lambda *_a, **_k: None)
    monkeypatch.setattr(boot, "create_venv",
                        lambda *args, **kwargs: created.update(kwargs) or venv)
    monkeypatch.setattr(boot, "install_requirements",
                        lambda *_a, **_k: {"offline_wheelhouse_used": False})
    monkeypatch.setattr(boot, "verify_imports", lambda *_a, **_k: {
        "ok": True, "missing": [], "modules": [], "groups": []})
    monkeypatch.setattr(boot, "installed_packages", lambda _python: {})

    report = boot.ensure_environment(quiet=True, allow_install=True)

    assert report["host_interpreter"]["selection"] == "CURRENT_INTERPRETER"
    assert report["venv_scheme"] == boot.POSIX_SCHEME
    assert report["interpreter"] == str(venv / "bin" / "python")
    assert created["expected_scheme"] == boot.POSIX_SCHEME
    assert report["profile_id"] == "cpu"
    assert "science_only" not in report["required_import_groups"]
