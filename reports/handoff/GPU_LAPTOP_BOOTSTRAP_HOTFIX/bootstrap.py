"""Stdlib-only environment bootstrap. Runs before the project environment exists.

This module is imported by `train.py` on a machine that may have nothing but a
system Python. It therefore imports **only the standard library** — no yaml, no
pydantic, no torch — and a test asserts that, because a single third-party import
here would make the zero-argument entrypoint fail on exactly the host it exists
to serve.

What it does, in order:

1. check the interpreter against the declared version range;
2. detect the hardware and pick an already-declared dependency profile;
3. create `.venv/` if absent;
4. install the profile's locked requirements;
5. write `state/ENVIRONMENT_MANIFEST.json` recording what was installed;
6. hand the caller the interpreter to re-exec with.

Two rules shape the design.

**No open-ended version search.** The hardware is inspected, then matched against
profiles declared in `configs/environment/environment_contract.yaml`. A host that
matches nothing stops with `CUDA_ENVIRONMENT_NOT_VALIDATED` and a remediation
report — it never guesses a wheel, and it never quietly falls back to CPU for
scientific work.

**Installation is idempotent by identity.** The manifest records a hash over the
requirement file bytes, the profile and the interpreter. If that identity still
matches, nothing is installed and no package index is contacted. A second
`python train.py` is therefore offline and fast.

The contract file is parsed with a deliberately small YAML reader below rather
than PyYAML, for the same reason the rest of this module is stdlib-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path(__file__).resolve().parent
CONTRACT = REPO / "configs" / "environment" / "environment_contract.yaml"
VENV = REPO / ".venv"
MANIFEST = REPO / "state" / "ENVIRONMENT_MANIFEST.json"
WHEELHOUSE = REPO / "vendor" / "wheels"

SCHEMA_VERSION = "prism-environment-manifest-v1"

#: Exit reasons the caller may need to distinguish.
UNSUPPORTED_PYTHON = "UNSUPPORTED_PYTHON"
CUDA_NOT_VALIDATED = "CUDA_ENVIRONMENT_NOT_VALIDATED"
BOOTSTRAP_FAILED = "BOOTSTRAP_FAILED"
SUPPORTED_WINDOWS_CPYTHON_NOT_FOUND = "SUPPORTED_WINDOWS_CPYTHON_NOT_FOUND"
SELF_RECREATION_REFUSED = "SELF_RECREATION_REFUSED"
VENV_NOT_VALIDATED = "VENV_NOT_VALIDATED"

#: Reasons the caller should treat as "this host cannot run the project", as
#: opposed to "this invocation was wrong".
BLOCKING_REASONS = (CUDA_NOT_VALIDATED, SUPPORTED_WINDOWS_CPYTHON_NOT_FOUND)


class BootstrapError(RuntimeError):
    """The environment cannot be prepared. Carries a reason code."""

    def __init__(self, reason: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.reason = reason
        self.detail = detail or {}


# --- a very small YAML reader ------------------------------------------------
#
# The contract is a flat-ish mapping of scalars, lists and nested maps written by
# this project. Parsing it with 60 lines of stdlib is a smaller risk than making
# the bootstrap depend on the package manager it is supposed to bootstrap.

def _coerce(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_coerce(part) for part in _split_inline(inner)] if inner else []
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "~", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _split_inline(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current)
    return parts


def read_contract(path: Path = CONTRACT) -> dict[str, Any]:
    """Parse the environment contract without PyYAML.

    Supports the subset the contract uses: nested mappings by indentation,
    inline lists, block lists of scalars, block scalars (``>-``) and comments.
    Anything richer would be a sign the contract had grown into something that
    should be code.
    """
    if not path.exists():
        raise BootstrapError(BOOTSTRAP_FAILED,
                             f"the environment contract is missing at {path}")
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_block: tuple[int, dict[str, Any], str] | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        if pending_block is not None:
            indent, holder, key = pending_block
            if raw.strip() and (len(raw) - len(raw.lstrip())) > indent:
                holder[key] = (str(holder[key]) + " " + raw.strip()).strip()
                continue
            pending_block = None
        line = raw.split(" #")[0].rstrip() if " #" in raw else raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()

        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]

        if body.startswith("- "):
            item = _coerce(body[2:])
            if isinstance(parent, list):
                parent.append(item)
            continue

        if ":" not in body:
            continue
        key, _, value = body.partition(":")
        key, value = key.strip(), value.strip()
        if value in (">-", ">", "|", "|-"):
            parent[key] = ""
            pending_block = (indent, parent, key)
            continue
        if value == "":
            child: Any = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        parent[key] = _coerce(value)

    # A key whose children turned out to be list items becomes a list.
    def normalize(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: normalize(item) for key, item in node.items()}
        return node

    return normalize(root)


# --- host interpreter classification -----------------------------------------
#
# `os.name == "nt"` is NOT enough to know what a Windows interpreter will do.
# MSYS2/MinGW Python is a native Windows executable that reports `os.name == "nt"`
# and `sys.platform == "win32"`, and then creates a POSIX-scheme virtual
# environment: `.venv/bin/python.exe` instead of `.venv/Scripts/python.exe`. A
# real deployment hit exactly that, because PATH `python` resolved to
# `C:\msys64\mingw64\bin\python.exe`.
#
# The decisive fact is not the platform string but the scheme `venv` itself uses.
# Since 3.11 `venv` asks sysconfig for `get_path("scripts", scheme="venv", ...)`,
# so asking the same question is a measurement of what the interpreter would
# actually do rather than an inference from what it calls itself.

STANDARD_WINDOWS_CPYTHON = "STANDARD_WINDOWS_CPYTHON"
MSYS2_MINGW_PYTHON = "MSYS2_MINGW_PYTHON"
POSIX_CPYTHON = "POSIX_CPYTHON"
UNKNOWN_PYTHON = "UNKNOWN_PYTHON"

#: The two canonical virtual-environment layouts. There is no third, and a
#: hybrid such as `.venv\bin\python.exe` is a defect, never a layout.
WINDOWS_SCHEME = "windows"        # .venv/Scripts/python.exe
POSIX_SCHEME = "posix"            # .venv/bin/python
SCRIPTS_DIR = {WINDOWS_SCHEME: "Scripts", POSIX_SCHEME: "bin"}

#: Path segments that appear only inside an MSYS2/MinGW/Cygwin installation.
#: Corroborating evidence only — never the deciding fact, because a standard
#: CPython launched *from* an MSYS2 shell is still a standard CPython.
MSYS_PATH_MARKERS = ("msys64", "msys32", "mingw64", "mingw32", "ucrt64",
                     "clang64", "clang32", "clangarm64", "cygwin64", "cygwin")

POSIX_PLATFORM_PREFIXES = ("linux", "darwin", "freebsd", "openbsd", "netbsd",
                           "aix", "sunos")

_PROBE_BASE = "__prism_probe_base__"

#: Emitted by a candidate interpreter so the classification measures *it* rather
#: than whichever interpreter happens to be running this module.
_EVIDENCE_PROBE = """
import json, os, platform, sys, sysconfig
base = "@BASE@"
try:
    scripts = sysconfig.get_path("scripts", scheme="venv", vars={
        "base": base, "platbase": base,
        "installed_base": base, "installed_platbase": base})
except Exception as error:
    scripts = "UNAVAILABLE: %s" % error
print(json.dumps({
    "executable": sys.executable,
    "version": platform.python_version(),
    "version_info": list(sys.version_info[:3]),
    "implementation": platform.python_implementation(),
    "sys_platform": sys.platform,
    "os_name": os.name,
    "sysconfig_platform": sysconfig.get_platform(),
    "default_scheme": sysconfig.get_default_scheme(),
    "venv_scripts_path": scripts,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "msystem": os.environ.get("MSYSTEM"),
}))
"""


def _venv_scripts_path(base: str = _PROBE_BASE) -> str:
    """What `python -m venv <base>` would call its scripts directory, here."""
    try:
        return sysconfig.get_path("scripts", scheme="venv", vars={
            "base": base, "platbase": base,
            "installed_base": base, "installed_platbase": base})
    except Exception as error:                               # noqa: BLE001 - reported
        return f"UNAVAILABLE: {error}"


def local_interpreter_evidence() -> dict[str, Any]:
    """Facts about the interpreter executing this function."""
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "sys_platform": sys.platform,
        "os_name": os.name,
        "sysconfig_platform": sysconfig.get_platform(),
        "default_scheme": sysconfig.get_default_scheme(),
        "venv_scripts_path": _venv_scripts_path(),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "msystem": os.environ.get("MSYSTEM"),
        "probe": "in-process",
        "usable": True,
    }


def interpreter_evidence(executable: str | Path | None = None, *,
                         timeout: int = 60) -> dict[str, Any]:
    """Facts about any interpreter, measured by running it.

    The local interpreter is measured in process; anything else is measured by a
    subprocess, because a claim about another interpreter that was not obtained
    from that interpreter is a guess.
    """
    if executable is None:
        return local_interpreter_evidence()
    candidate = Path(executable)
    try:
        if candidate.resolve() == Path(sys.executable).resolve():
            return local_interpreter_evidence()
    except OSError:
        pass
    probe = _EVIDENCE_PROBE.replace("@BASE@", _PROBE_BASE)
    try:
        output = subprocess.check_output([str(candidate), "-c", probe], text=True,
                                         stderr=subprocess.DEVNULL, timeout=timeout)
        payload = json.loads(output.strip().splitlines()[-1])
    except Exception as error:                               # noqa: BLE001 - reported
        return {"executable": str(candidate), "probe": "subprocess",
                "usable": False, "error": f"{type(error).__name__}: {error}"}
    payload["probe"] = "subprocess"
    payload["usable"] = True
    return payload


def classify_interpreter(evidence: dict[str, Any]) -> dict[str, Any]:
    """Name what kind of Python this is, and what venv layout it would produce.

    Pure: it reads a dictionary, so the MSYS2 case is testable on a machine that
    has never had MSYS2 installed. The order of the checks is the argument —
    `sys.platform` and `os.name` are consulted last, because they are exactly the
    fields MSYS2 Python answers like a standard Windows CPython.
    """
    executable = str(evidence.get("executable") or "")
    sys_platform = str(evidence.get("sys_platform") or "")
    os_name = str(evidence.get("os_name") or "")
    implementation = str(evidence.get("implementation") or "")
    sysconfig_platform = str(evidence.get("sysconfig_platform") or "")
    scripts_path = str(evidence.get("venv_scripts_path") or "")
    scripts_dir = PurePosixPath(scripts_path.replace("\\", "/")).name if scripts_path else ""

    signals: list[str] = []
    parts = {part.lower() for part in re.split(r"[\\/]+", executable) if part}
    markers = sorted(parts & set(MSYS_PATH_MARKERS))
    if markers:
        signals.append(f"installation path contains {markers}")
    if evidence.get("msystem"):
        signals.append(f"MSYSTEM={evidence['msystem']} in the environment")
    if sysconfig_platform.lower().startswith(("mingw", "msys", "cygwin")):
        signals.append(f"sysconfig platform {sysconfig_platform!r}")
    if sys_platform in ("msys", "cygwin"):
        signals.append(f"sys.platform={sys_platform!r}")

    windows_like = os_name == "nt" or sys_platform == "win32"
    posix_like = os_name == "posix" or any(
        sys_platform.startswith(prefix) for prefix in POSIX_PLATFORM_PREFIXES)

    def result(classification: str, scheme: str, why: str) -> dict[str, Any]:
        return {"classification": classification, "venv_scheme": scheme,
                "scripts_dir": SCRIPTS_DIR[scheme],
                "measured_scripts_dir": scripts_dir or None,
                "why": why, "signals": signals, "evidence": evidence,
                "may_build_the_project_environment":
                    classification in (STANDARD_WINDOWS_CPYTHON, POSIX_CPYTHON)}

    if sys_platform in ("msys", "cygwin"):
        return result(MSYS2_MINGW_PYTHON, POSIX_SCHEME,
                      f"sys.platform is {sys_platform!r}, which is never a standard "
                      "Windows CPython")

    if windows_like:
        # The scheme `venv` itself would use is the deciding fact. A standard
        # Windows CPython answers "Scripts"; MSYS2/MinGW answers "bin", which is
        # precisely the defect that produced `.venv/bin/python.exe` on Windows.
        mingw_build = sysconfig_platform.lower().startswith(("mingw", "msys", "cygwin"))
        if scripts_dir and scripts_dir.lower() != "scripts":
            return result(MSYS2_MINGW_PYTHON, POSIX_SCHEME,
                          "a Windows-like interpreter whose venv scheme puts scripts "
                          f"in {scripts_dir!r}, not 'Scripts'")
        if mingw_build:
            return result(MSYS2_MINGW_PYTHON, POSIX_SCHEME,
                          "sysconfig reports the MinGW build platform "
                          f"{sysconfig_platform!r}")
        if not scripts_dir:
            return result(UNKNOWN_PYTHON, WINDOWS_SCHEME,
                          "the interpreter did not report a venv scripts scheme")
        if implementation != "CPython":
            return result(UNKNOWN_PYTHON, WINDOWS_SCHEME,
                          f"implementation is {implementation!r}, not CPython")
        return result(STANDARD_WINDOWS_CPYTHON, WINDOWS_SCHEME,
                      "a CPython whose venv scheme is the standard Windows layout "
                      "(Scripts/)")

    if posix_like:
        if implementation != "CPython":
            return result(UNKNOWN_PYTHON, POSIX_SCHEME,
                          f"implementation is {implementation!r}, not CPython")
        if scripts_dir and scripts_dir.lower() != "bin":
            return result(UNKNOWN_PYTHON, POSIX_SCHEME,
                          "a POSIX host whose venv scheme puts scripts in "
                          f"{scripts_dir!r}, not 'bin'")
        return result(POSIX_CPYTHON, POSIX_SCHEME,
                      "a CPython on a POSIX host with the standard bin/ layout")

    return result(UNKNOWN_PYTHON, POSIX_SCHEME if os_name == "posix" else WINDOWS_SCHEME,
                  f"unrecognised host: os.name={os_name!r}, "
                  f"sys.platform={sys_platform!r}")


# --- finding a standard Windows CPython --------------------------------------

def _py_launcher_candidates() -> list[dict[str, Any]]:
    """Ask the Windows Python Launcher what is installed (`py -0p`)."""
    launcher = shutil.which("py")
    if launcher is None:
        return []
    for arguments in (["-0p"], ["--list-paths"]):
        try:
            output = subprocess.check_output([launcher, *arguments], text=True,
                                             stderr=subprocess.STDOUT, timeout=60)
        except Exception:                                    # noqa: BLE001
            continue
        found: list[dict[str, Any]] = []
        for line in output.splitlines():
            text = line.strip()
            if not text or text.lower().startswith(("installed", "no python")):
                continue
            match = re.search(r"(?P<path>[A-Za-z]:[\\/].*?python(?:w)?\.exe)", text)
            if match is None:
                continue
            tag = re.match(r"^-(?:V:)?(?P<tag>\d+\.\d+)", text)
            head = text.split(match.group("path"))[0]
            found.append({"source": "py_launcher",
                          "tag": tag.group("tag") if tag else None,
                          "executable": match.group("path").strip(),
                          "default": "*" in head})
        if found:
            return found
    return []


def _well_known_candidates() -> list[dict[str, Any]]:
    """Standard CPython install locations, for a host with no `py.exe`."""
    roots: list[Path] = []
    for variable in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value) / "Programs" / "Python")
            roots.append(Path(value))
    roots.append(Path("C:/"))
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        try:
            if not root.is_dir():
                continue
            entries = sorted(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not re.fullmatch(r"Python3\d+", entry.name, flags=re.IGNORECASE):
                continue
            candidate = entry / "python.exe"
            key = str(candidate).lower()
            if candidate.is_file() and key not in seen:
                seen.add(key)
                found.append({"source": "well_known_install_root", "tag": None,
                              "executable": str(candidate), "default": False})
    return found


def discover_windows_interpreters() -> list[dict[str, Any]]:
    """Every plausible standard Windows CPython this host offers, deduplicated."""
    candidates = _py_launcher_candidates() + _well_known_candidates()
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(Path(candidate["executable"])).lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def _supported_range(contract: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    block = dict(contract.get("python") or {})
    return (_version_tuple(block.get("minimum", "3.11")),
            _version_tuple(block.get("maximum_exclusive", "3.14")))


def _preferred_minors(contract: dict[str, Any]) -> list[str]:
    block = dict(contract.get("python") or {})
    declared = block.get("preferred_minors") or []
    if isinstance(declared, str):
        declared = [declared]
    return [str(item) for item in declared]


def _minor(version_info: Any) -> str:
    return ".".join(str(part) for part in tuple(version_info)[:2])


def select_windows_cpython(contract: dict[str, Any], *,
                           candidates: list[dict[str, Any]] | None = None,
                           probe: Any = None) -> dict[str, Any]:
    """Pick one supported standard Windows CPython, deterministically.

    Newest-wins is explicitly not the rule. The contract declares a preference
    order, so a host that has 3.14 installed (outside the supported range) and
    3.12 installed resolves to 3.12 rather than failing or guessing.
    """
    probe = probe or interpreter_evidence
    candidates = discover_windows_interpreters() if candidates is None else candidates
    minimum, maximum = _supported_range(contract)
    preferred = _preferred_minors(contract)

    examined: list[dict[str, Any]] = []
    usable: list[dict[str, Any]] = []
    for candidate in candidates:
        evidence = probe(candidate["executable"])
        if not evidence.get("usable", True):
            examined.append({**candidate, "classification": UNKNOWN_PYTHON,
                             "rejected_because": evidence.get(
                                 "error", "could not be launched")})
            continue
        verdict = classify_interpreter(evidence)
        version_info = tuple(evidence.get("version_info") or ())
        entry = {**candidate, "version": evidence.get("version"),
                 "classification": verdict["classification"],
                 "venv_scheme": verdict["venv_scheme"]}
        if verdict["classification"] != STANDARD_WINDOWS_CPYTHON:
            entry["rejected_because"] = verdict["why"]
        elif not (minimum <= version_info < maximum):
            entry["rejected_because"] = (
                f"Python {evidence.get('version')} is outside the supported range")
        else:
            entry["minor"] = _minor(version_info)
            entry["evidence"] = evidence
            usable.append(entry)
        examined.append({key: value for key, value in entry.items()
                         if key != "evidence"})

    result = {"selected": None, "examined": examined,
              "supported_range": [".".join(map(str, minimum)),
                                  ".".join(map(str, maximum))],
              "preferred_minors": preferred}
    if not usable:
        return result

    def rank(entry: dict[str, Any]) -> tuple[int, tuple[int, ...]]:
        minor = entry.get("minor") or ""
        position = preferred.index(minor) if minor in preferred else len(preferred)
        # Within an undeclared minor, lower wins: the project has never been run
        # on the newest thing a host happens to have installed.
        return (position, tuple(entry["evidence"]["version_info"]))

    usable.sort(key=rank)
    result["selected"] = usable[0]
    return result


def resolve_host_interpreter(contract: dict[str, Any], *,
                             evidence: dict[str, Any] | None = None,
                             selector: Any = None) -> dict[str, Any]:
    """Decide which interpreter is allowed to build the project environment.

    Returns the executable to create `.venv` with, its classification, the venv
    scheme it will produce, and how it was chosen. Raises rather than proceeding
    when a Windows host offers nothing but MSYS2 Python.
    """
    evidence = local_interpreter_evidence() if evidence is None else evidence
    verdict = classify_interpreter(evidence)
    classification = verdict["classification"]
    minimum, maximum = _supported_range(contract)
    version_info = tuple(evidence.get("version_info") or ())
    in_range = bool(version_info) and minimum <= version_info < maximum

    if classification in (STANDARD_WINDOWS_CPYTHON, POSIX_CPYTHON) and in_range:
        return {"executable": str(evidence.get("executable")),
                "classification": classification,
                "venv_scheme": verdict["venv_scheme"],
                "scripts_dir": verdict["scripts_dir"],
                "selection": "CURRENT_INTERPRETER",
                "why": verdict["why"], "signals": verdict["signals"],
                "evidence": evidence, "host_classification": classification,
                "fallback": None}

    windows_like = (str(evidence.get("os_name")) == "nt"
                    or str(evidence.get("sys_platform")) == "win32"
                    or classification == MSYS2_MINGW_PYTHON)
    if not windows_like:
        # A POSIX host has no launcher to fall back to; the operator installs a
        # supported interpreter. This is the pre-existing behaviour, unchanged.
        raise BootstrapError(
            UNSUPPORTED_PYTHON,
            f"Python {evidence.get('version')} at {evidence.get('executable')} is "
            f"classified {classification} and is outside the supported range "
            f"[{'.'.join(map(str, minimum))}, {'.'.join(map(str, maximum))}). "
            "Install a supported interpreter and run `python train.py` again. "
            "This project never installs or replaces the host Python.",
            {"classification": classification, "evidence": evidence})

    selector = selector or select_windows_cpython
    search = selector(contract)
    best = search.get("selected")
    if best is None:
        discovered = "\n".join(
            f"    {item.get('executable')}  "
            f"{item.get('version') or 'unknown version'}  "
            f"{item.get('classification', UNKNOWN_PYTHON)}"
            + (f"  - {item['rejected_because']}" if item.get("rejected_because") else "")
            for item in search.get("examined", [])) or "    (none discovered)"
        raise BootstrapError(
            SUPPORTED_WINDOWS_CPYTHON_NOT_FOUND,
            "no supported standard Windows CPython could be found, and this host's "
            "PATH Python cannot build the project environment.\n"
            f"    detected interpreter   {evidence.get('executable')}\n"
            f"    classification         {classification}\n"
            f"    reason                 {verdict['why']}\n"
            f"    supported Python       [{'.'.join(map(str, minimum))}, "
            f"{'.'.join(map(str, maximum))})  preferred "
            f"{search.get('preferred_minors')}\n"
            "  Interpreters discovered:\n" + discovered +
            "\n  MSYS2/MinGW Python creates a POSIX-scheme environment on Windows and "
            "is\n  not supported for scientific execution. Install a standard Windows "
            "CPython\n  from python.org (a version in the range above) and run "
            "`python train.py` again.\n"
            "  No package was installed and no environment was created.",
            {"detected_interpreter": evidence.get("executable"),
             "classification": classification,
             "supported_range": search.get("supported_range"),
             "preferred_minors": search.get("preferred_minors"),
             "interpreters_discovered": search.get("examined", [])})

    return {"executable": best["executable"],
            "classification": best["classification"],
            "venv_scheme": best["venv_scheme"],
            "scripts_dir": SCRIPTS_DIR[best["venv_scheme"]],
            "selection": "WINDOWS_CPYTHON_FALLBACK",
            "why": f"PATH python is {classification}; the Windows Python Launcher "
                   f"offered a supported standard CPython {best.get('version')}",
            "signals": verdict["signals"],
            "evidence": best.get("evidence", {}),
            "host_classification": classification,
            "fallback": {"from": str(evidence.get("executable")),
                         "from_classification": classification,
                         "to": best["executable"],
                         "to_version": best.get("version"),
                         "examined": search.get("examined", [])}}


# --- python version ----------------------------------------------------------

def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", str(text))[:3])


def check_python(contract: dict[str, Any],
                 evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Refuse an interpreter outside the declared range, with the exact numbers.

    `evidence` names WHICH interpreter is being judged. With none it judges the
    running one, which is what a bare `python train.py` starts with; the
    bootstrap passes the interpreter it actually resolved, so the recorded
    version is the one that will build the environment rather than the one that
    happened to launch the entrypoint.
    """
    block = dict(contract.get("python") or {})
    minimum = _version_tuple(block.get("minimum", "3.11"))
    maximum = _version_tuple(block.get("maximum_exclusive", "3.14"))
    evidence = local_interpreter_evidence() if evidence is None else evidence
    current = tuple(evidence.get("version_info") or ())
    verdict = classify_interpreter(evidence)

    ok = bool(current) and minimum <= current < maximum
    detail = {
        "found": evidence.get("version"),
        "found_tuple": list(current),
        "required_minimum": block.get("minimum"),
        "required_maximum_exclusive": block.get("maximum_exclusive"),
        "preferred_minors": _preferred_minors(contract),
        "tested_on": block.get("tested_on"),
        "implementation": evidence.get("implementation"),
        "executable": evidence.get("executable"),
        "classification": verdict["classification"],
        "venv_scheme": verdict["venv_scheme"],
        "supported": ok,
    }
    if not ok:
        raise BootstrapError(
            UNSUPPORTED_PYTHON,
            f"Python {evidence.get('version')} is outside the supported range "
            f"[{block.get('minimum')}, {block.get('maximum_exclusive')}). "
            f"Install a supported interpreter and run `python train.py` again. "
            f"This project never installs or replaces the host Python.",
            detail)
    return detail


# --- hardware ----------------------------------------------------------------

def detect_gpu() -> dict[str, Any]:
    """Read non-secret GPU facts from nvidia-smi. Absent tooling is not an error."""
    info: dict[str, Any] = {"available": False, "name": None, "driver_version": None,
                            "memory_total_mb": None, "compute_capability": None,
                            "source": None, "query_error": None}
    binary = shutil.which("nvidia-smi")
    if binary is None:
        info["query_error"] = "nvidia-smi not found on PATH"
        return info
    try:
        output = subprocess.check_output(
            [binary, "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=30).strip()
    except Exception as error:                              # noqa: BLE001 - reported
        info["query_error"] = f"{type(error).__name__}: {error}"
        return info
    if not output:
        info["query_error"] = "nvidia-smi returned no GPU"
        return info
    first = output.splitlines()[0]
    parts = [item.strip() for item in first.split(",")]
    info.update({
        "available": True, "source": "nvidia-smi",
        "name": parts[0] if parts else None,
        "driver_version": parts[1] if len(parts) > 1 else None,
        "memory_total_mb": _to_int(parts[2]) if len(parts) > 2 else None,
        "compute_capability": parts[3] if len(parts) > 3 and parts[3] else None,
        "all_gpus": [line.strip() for line in output.splitlines()],
    })
    return info


def _to_int(text: str) -> int | None:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def gpu_family(name: str | None, contract: dict[str, Any]) -> str:
    """Name the architecture family, for the operator's benefit only.

    This is PROVENANCE, not a gate. An earlier version made the family a
    matching condition, which meant a perfectly compatible datacenter card
    would be refused for the sole reason that its marketing string was not in a
    list somebody had typed. Compute capability is the hardware fact that
    actually determines whether a wheel's kernels will run; the name is how a
    human recognises the machine.
    """
    if not name:
        return "UNKNOWN"
    patterns = dict(contract.get("family_patterns") or {})
    upper = name.upper()
    for family, needles in patterns.items():
        for needle in (needles if isinstance(needles, list) else [needles]):
            if str(needle).upper() in upper:
                return family
    return "UNRECOGNISED_NAME"


#: Wheel platforms this project can be installed on. A profile that publishes
#: nothing for the host's tag cannot be selected on it, whatever its GPU says.
WIN_AMD64 = "win_amd64"
LINUX_X86_64 = "linux_x86_64"
LINUX_AARCH64 = "linux_aarch64"
MACOS_ARM64 = "macosx_arm64"
UNKNOWN_PLATFORM = "unknown_platform"


def host_platform_tag(evidence: dict[str, Any] | None = None) -> str:
    """Name the wheel platform the host actually needs a build for.

    Derived from the interpreter's own `sysconfig.get_platform()` rather than
    from `sys.platform`, because that is the string the wheel tags are built
    from — and, on Windows, because an MSYS2 interpreter reports `win32` while
    needing a completely different set of artifacts.
    """
    evidence = local_interpreter_evidence() if evidence is None else evidence
    platform_name = str(evidence.get("sysconfig_platform") or "").lower()
    machine = platform_name.rsplit("-", 1)[-1]
    if platform_name.startswith("win"):
        return WIN_AMD64 if machine in ("amd64", "x86_64") else UNKNOWN_PLATFORM
    if platform_name.startswith("linux"):
        if machine in ("x86_64", "amd64"):
            return LINUX_X86_64
        if machine in ("aarch64", "arm64"):
            return LINUX_AARCH64
        return UNKNOWN_PLATFORM
    if platform_name.startswith("macosx"):
        return MACOS_ARM64 if machine in ("arm64", "universal2") else UNKNOWN_PLATFORM
    return UNKNOWN_PLATFORM


#: How well a host matches a declared profile.
VALIDATED_PROFILE = "VALIDATED_PROFILE"
COMPATIBLE_DECLARED_PROFILE = "COMPATIBLE_DECLARED_PROFILE"
UNVALIDATED_COMPATIBLE_CANDIDATE = "UNVALIDATED_COMPATIBLE_CANDIDATE"
INCOMPATIBLE = "INCOMPATIBLE"


def _capability_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(value).strip().split("."))
    except (TypeError, ValueError):
        return ()


def classify_candidate(profile: dict[str, Any], *, capability: str,
                       driver: tuple[int, ...],
                       platform_tag: str | None = None) -> dict[str, Any]:
    """Grade one declared profile against the detected hardware.

    The host's wheel platform is checked first, then compute capability, then
    the driver floor. Platform comes first because it is the only one of the
    three that cannot be argued with: a profile whose index publishes no build
    for this platform cannot be installed from it at all, and — since the CUDA
    requirement files carry --extra-index-url — would not even fail loudly. pip
    would fall through to the PyPI wheel while the manifest went on naming the
    CUDA index it never used. That is what happened to cuda-cu129 on Windows.

    A capability inside the profile's declared range but not in its enumerated
    list is a CANDIDATE rather than a match: the kernels are almost certainly
    present, but nobody has run it, and the difference between "should work" and
    "has worked" is the difference this vocabulary exists to keep.
    """
    declared = [str(item) for item in (profile.get("compute_capabilities") or [])]
    minimum_driver = _version_tuple(profile.get("minimum_driver_version") or "0")
    driver_ok = driver >= minimum_driver
    platforms = [str(item) for item in (profile.get("platforms") or [])]
    platform_ok = not platforms or not platform_tag or platform_tag in platforms

    detail = {"profile_id": profile.get("id"),
              "declared_capabilities": declared,
              "detected_capability": capability or "unreported",
              "minimum_driver_version": profile.get("minimum_driver_version"),
              "driver_satisfied": driver_ok,
              "declared_platforms": platforms,
              "host_platform": platform_tag,
              "platform_satisfied": platform_ok}

    if not platform_ok:
        return {**detail, "grade": INCOMPATIBLE,
                "why": f"this profile's index publishes no {platform_tag} wheel "
                       f"(it declares {platforms})"}

    if not driver_ok:
        return {**detail, "grade": INCOMPATIBLE,
                "why": f"driver below the profile floor "
                       f"{profile.get('minimum_driver_version')}"}

    if not capability:
        # Unreported capability cannot be graded upward. Refusing here is safer
        # than assuming, because the failure it prevents happens at the first
        # kernel launch, long after the data has loaded.
        return {**detail, "grade": INCOMPATIBLE,
                "why": "the host did not report a compute capability"}

    if capability in declared:
        grade = (VALIDATED_PROFILE if profile.get("status") == "VALIDATED"
                 else COMPATIBLE_DECLARED_PROFILE)
        return {**detail, "grade": grade,
                "why": f"compute capability {capability} is declared by this profile"}

    detected = _capability_tuple(capability)
    supported = [_capability_tuple(item) for item in declared]
    supported = [item for item in supported if item]
    if supported and min(supported) <= detected <= max(supported):
        return {**detail, "grade": UNVALIDATED_COMPATIBLE_CANDIDATE,
                "why": f"compute capability {capability} falls inside this profile's "
                       f"declared range {min(declared)}-{max(declared)} but is not "
                       "itself enumerated"}

    return {**detail, "grade": INCOMPATIBLE,
            "why": f"compute capability {capability} is outside this profile's "
                   f"declared range"}


def select_profile(contract: dict[str, Any], gpu: dict[str, Any], *,
                   platform_tag: str | None = None) -> dict[str, Any]:
    """Match the host to an already-declared profile. Never search for one.

    Returns the selected profile plus the reasoning, so a refusal can explain
    itself without the caller re-deriving anything.
    """
    profiles = dict(contract.get("profiles") or {})
    selection = dict(contract.get("selection") or {})
    order = selection.get("order") or []
    if isinstance(order, str):
        order = [order]
    platform_tag = platform_tag or host_platform_tag()

    if not gpu.get("available"):
        cpu = dict(profiles.get("cpu") or {})
        return {"profile_id": "cpu", "profile": cpu, "reason": "no CUDA GPU detected",
                "gpu": gpu, "family": "NONE", "supports_scientific_execution": False,
                "host_platform": platform_tag, "candidates_considered": []}

    family = gpu_family(gpu.get("name"), contract)
    driver = _version_tuple(gpu.get("driver_version") or "0")
    capability = str(gpu.get("compute_capability") or "")
    considered: list[dict[str, Any]] = []

    for profile_id in order:
        profile = dict(profiles.get(profile_id) or {})
        if not profile:
            continue
        considered.append(classify_candidate(profile, capability=capability,
                                             driver=driver,
                                             platform_tag=platform_tag))

    # An exactly-declared capability wins over a merely-plausible one, whichever
    # order the profiles are listed in. The GPU's model name is recorded but
    # never consulted: a card this contract has never heard of is accepted when
    # its capability and driver satisfy a profile.
    rank = {VALIDATED_PROFILE: 0, COMPATIBLE_DECLARED_PROFILE: 1,
            UNVALIDATED_COMPATIBLE_CANDIDATE: 2}
    ranked = sorted((item for item in considered if item["grade"] != INCOMPATIBLE),
                    key=lambda item: (rank[item["grade"]],
                                      order.index(item["profile_id"])))

    if ranked:
        best = ranked[0]
        profile = dict(profiles.get(best["profile_id"]) or {})
        return {"profile_id": best["profile_id"], "profile": profile,
                "grade": best["grade"],
                "reason": f"compute capability {capability or 'unreported'} with driver "
                          f"{gpu.get('driver_version')}: {best['why']} ({best['grade']})",
                "gpu": gpu, "family": family, "host_platform": platform_tag,
                "supports_scientific_execution":
                    bool(profile.get("supports_scientific_execution")),
                "candidates_considered": considered}

    lines = "\n".join(
        f"    {item['profile_id']:<14} {item['grade']:<32} {item['why']}"
        for item in considered)
    raise BootstrapError(
        CUDA_NOT_VALIDATED,
        "the detected GPU satisfies no declared environment profile.\n"
        f"    host platform       {platform_tag}\n"
        f"    GPU                 {gpu.get('name')}  (name is provenance, not a gate)\n"
        f"    driver              {gpu.get('driver_version')}\n"
        f"    compute capability  {gpu.get('compute_capability') or 'unreported'}\n"
        f"    architecture        {family}\n"
        "  Declared profiles, and why each was rejected:\n" + lines +
        "\n  Compatibility is decided by compute capability and driver version, not\n"
        "  by the GPU's model name, so an unlisted card is fine when its capability\n"
        "  is covered. Either install a driver that satisfies a declared profile, or\n"
        "  extend configs/environment/environment_contract.yaml deliberately.\n"
        "  This runner will NOT guess a CUDA wheel and will NOT fall back to CPU\n"
        "  for scientific execution.",
        {"gpu": gpu, "family": family, "host_platform": platform_tag,
         "candidates_considered": considered})


# --- environment identity ----------------------------------------------------

def requirement_files(profile: dict[str, Any]) -> list[Path]:
    """The requirement file and everything it includes, in a stable order."""
    entry = REPO / str(profile.get("requirements", "requirements/cpu.txt"))
    seen: list[Path] = []

    def walk(path: Path) -> None:
        if not path.exists() or path in seen:
            return
        seen.append(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            for flag in ("-r ", "--requirement ", "-c ", "--constraint "):
                if text.startswith(flag):
                    walk((path.parent / text[len(flag):].strip()).resolve())

    walk(entry.resolve())
    return seen


def environment_identity(profile_id: str, profile: dict[str, Any],
                         python_version: str) -> dict[str, Any]:
    """A hash over exactly what determines the installed set.

    Requirement bytes, the profile id and the interpreter's minor version. Not
    the hostname, not the absolute path — moving the folder must not force a
    reinstall.
    """
    files = requirement_files(profile)
    digests = {
        path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files}
    material = {"profile_id": profile_id, "requirements": digests,
                "python_minor": ".".join(python_version.split(".")[:2])}
    identity = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"identity": identity, "material": material,
            "files": [path.relative_to(REPO).as_posix() for path in files]}


def venv_python(venv: Path = VENV, *, scheme: str | None = None) -> Path:
    """The canonical interpreter path for one layout.

    Two layouts exist and no third is ever constructed. `.venv\\bin\\python.exe`
    — what MSYS2 Python produces on Windows — is a defect, not a layout, and is
    recognised only in order to be rejected.
    """
    scheme = scheme or (WINDOWS_SCHEME if os.name == "nt" else POSIX_SCHEME)
    if scheme == WINDOWS_SCHEME:
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


#: What an existing `.venv` turned out to be.
VENV_ABSENT = "ABSENT"
VENV_VALID = "VALID"
VENV_NO_INTERPRETER = "PARTIAL_NO_INTERPRETER"
VENV_INTERPRETER_UNUSABLE = "INTERPRETER_WILL_NOT_RUN"
VENV_WRONG_SCHEME = "WRONG_SCHEME_FOR_HOST"
VENV_WRONG_VERSION = "INCOMPATIBLE_PYTHON_VERSION"
VENV_FOREIGN_PREFIX = "NOT_THIS_PROJECT_VENV"
VENV_DEPENDENCIES_INCOMPLETE = "DEPENDENCIES_INCOMPLETE"

#: What to do about it. REBUILD is the only one that deletes anything, and it
#: deletes only this project's own `.venv`.
VENV_CREATE = "CREATE"
VENV_REUSE = "REUSE"
VENV_INSTALL_INTO = "INSTALL_INTO"
VENV_REBUILD = "REBUILD"


def existing_venv_interpreter(venv: Path = VENV) -> tuple[Path | None, str | None]:
    """Find whatever interpreter an existing `.venv` actually contains.

    Both canonical layouts are probed, plus the hybrid `bin/python.exe` that
    MSYS2 Python leaves on Windows, because a folder that arrived from another
    machine is exactly the case this function exists for.
    """
    for scheme in (WINDOWS_SCHEME, POSIX_SCHEME):
        candidate = venv_python(venv, scheme=scheme)
        if candidate.exists():
            return candidate, scheme
    hybrid = venv / "bin" / "python.exe"
    if hybrid.exists():
        return hybrid, POSIX_SCHEME
    return None, None


def _is_empty(directory: Path) -> bool:
    try:
        return not any(directory.iterdir())
    except OSError:
        return False


def classify_venv(venv: Path = VENV, *, expected_scheme: str,
                  expected_minor: str | None = None,
                  dependencies_ok: bool | None = None,
                  probe: Any = None) -> dict[str, Any]:
    """Grade an existing `.venv` and say what should be done about it.

    Every state below was seen or is reachable on a real destination machine: a
    half-written environment from an interrupted install, a Linux `.venv` inside
    a copied folder, an MSYS2-scheme `.venv` on Windows, an interpreter whose
    base Python has been uninstalled. The operator should never have to delete
    anything by hand, so each state carries the action that repairs it.
    """
    probe = probe or interpreter_evidence
    detail: dict[str, Any] = {"path": str(venv), "expected_scheme": expected_scheme,
                              "expected_python_minor": expected_minor}

    if not venv.exists() or _is_empty(venv):
        return {**detail, "state": VENV_ABSENT, "action": VENV_CREATE,
                "why": "no virtual environment is present"}

    interpreter, scheme = existing_venv_interpreter(venv)
    if interpreter is None:
        return {**detail, "state": VENV_NO_INTERPRETER, "action": VENV_REBUILD,
                "why": "a .venv directory exists but contains no interpreter, which "
                       "is what an interrupted creation leaves behind"}
    detail["found_interpreter"] = str(interpreter)
    detail["found_scheme"] = scheme

    if scheme != expected_scheme:
        return {**detail, "state": VENV_WRONG_SCHEME, "action": VENV_REBUILD,
                "why": f"the environment uses the {scheme} layout "
                       f"({interpreter.name} under {interpreter.parent.name}/) but "
                       f"this host needs the {expected_scheme} layout; a .venv "
                       "cannot be carried between the two"}

    evidence = probe(interpreter)
    detail["interpreter_evidence"] = evidence
    if not evidence.get("usable", True):
        return {**detail, "state": VENV_INTERPRETER_UNUSABLE, "action": VENV_REBUILD,
                "why": f"the interpreter will not run: {evidence.get('error')}"}

    try:
        prefix_matches = Path(str(evidence.get("prefix"))).resolve() == venv.resolve()
    except OSError:
        prefix_matches = False
    if not prefix_matches or evidence.get("prefix") == evidence.get("base_prefix"):
        return {**detail, "state": VENV_FOREIGN_PREFIX, "action": VENV_REBUILD,
                "why": f"sys.prefix is {evidence.get('prefix')!r}, which is not this "
                       "project's .venv; the interpreter is not the environment it "
                       "appears to be in"}

    found_minor = _minor(evidence.get("version_info") or ())
    detail["found_python_minor"] = found_minor
    if expected_minor and found_minor != expected_minor:
        return {**detail, "state": VENV_WRONG_VERSION, "action": VENV_REBUILD,
                "why": f"the environment is Python {found_minor} but the host "
                       f"interpreter is {expected_minor}; the ABI differs and the "
                       "installed wheels would not load"}

    if dependencies_ok is False:
        return {**detail, "state": VENV_DEPENDENCIES_INCOMPLETE,
                "action": VENV_INSTALL_INTO,
                "why": "the environment is structurally sound but its declared "
                       "dependencies do not all import, which is what an interrupted "
                       "pip install leaves behind"}

    return {**detail, "state": VENV_VALID, "action": VENV_REUSE,
            "why": "the environment matches this host and its interpreter runs"}


def validate_venv(venv: Path = VENV, *, expected_scheme: str,
                  expected_minor: str | None = None,
                  probe: Any = None) -> dict[str, Any]:
    """Prove an environment is usable. Exit code 0 from `venv` is not proof.

    A subprocess that returned 0 says the command ran. This says the interpreter
    exists where this platform puts it, launches, and reports a `sys.prefix`
    inside the project environment rather than the host Python it was built from.
    """
    report = classify_venv(venv, expected_scheme=expected_scheme,
                           expected_minor=expected_minor, probe=probe)
    report["valid"] = report["state"] == VENV_VALID
    report["expected_interpreter"] = str(venv_python(venv, scheme=expected_scheme))
    return report


def assert_not_self_recreation(host_executable: str | Path,
                               venv: Path = VENV) -> None:
    """A project venv interpreter must never be asked to recreate its own venv."""
    try:
        host = Path(host_executable).resolve()
        target = venv.resolve()
    except OSError:
        return
    if host == target or target in host.parents:
        raise BootstrapError(
            SELF_RECREATION_REFUSED,
            f"refusing to recreate {target} using {host}, which lives inside it. "
            "An environment cannot rebuild itself in place: the files being "
            "replaced are the ones executing. Re-run `python train.py` with the "
            "host interpreter, or delete the environment while nothing is using it.",
            {"host_executable": str(host), "venv": str(target)})


def remove_venv(venv: Path = VENV, *, repo: Path = REPO) -> dict[str, Any]:
    """Delete this project's own `.venv`, and refuse to delete anything else.

    Three guards, because the blast radius of a wrong answer here is somebody's
    system Python or a colleague's environment: the path must be `<repo>/.venv`,
    it must look like a virtual environment, and the interpreter running this
    code must not be inside it.
    """
    resolved = venv.resolve()
    if resolved.name != ".venv" or resolved.parent != repo.resolve():
        raise BootstrapError(
            BOOTSTRAP_FAILED,
            f"refusing to delete {resolved}: only this project's own .venv "
            f"({repo.resolve() / '.venv'}) may be rebuilt.")
    try:
        inside = (Path(sys.prefix).resolve() == resolved
                  or resolved in Path(sys.executable).resolve().parents)
    except OSError:
        inside = False
    if inside:
        raise BootstrapError(
            SELF_RECREATION_REFUSED,
            f"refusing to delete {resolved} while executing from inside it.")
    if _is_empty(resolved):
        return {"removed": False, "reason": "the directory was already empty",
                "path": str(resolved)}
    looks_like_venv = (resolved / "pyvenv.cfg").exists() or any(
        (resolved / name).exists()
        for name in ("Scripts", "bin", "Lib", "lib", "Include", "include"))
    if not looks_like_venv:
        raise BootstrapError(
            BOOTSTRAP_FAILED,
            f"refusing to delete {resolved}: it does not look like a virtual "
            "environment (no pyvenv.cfg and no interpreter layout). Inspect it by "
            "hand rather than letting the bootstrap remove unknown files.")
    shutil.rmtree(resolved)
    return {"removed": True, "path": str(resolved),
            "reason": "rebuilt because the environment did not match this host"}


def read_manifest(path: Path = MANIFEST) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(payload: dict[str, Any], path: Path = MANIFEST) -> None:
    """Atomic write: an interrupted bootstrap must not leave a half manifest.

    A truncated manifest would fail to parse, which `read_manifest` turns into
    "no manifest" — so the worst case of an interrupted write is a reinstall,
    never a false claim that the environment is ready.
    """
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


# --- installation ------------------------------------------------------------

def _run(command: list[str], *, quiet: bool) -> None:
    result = subprocess.run(command, cwd=str(REPO),
                            capture_output=quiet, text=True)
    if result.returncode != 0:
        tail = ""
        if quiet and result.stderr:
            tail = "\n" + "\n".join(result.stderr.strip().splitlines()[-15:])
        raise BootstrapError(
            BOOTSTRAP_FAILED,
            f"command failed with exit {result.returncode}: {' '.join(command[:4])}...{tail}")


def create_venv(venv: Path = VENV, *, quiet: bool = False,
                host_executable: str | Path | None = None,
                expected_scheme: str | None = None,
                expected_minor: str | None = None) -> Path:
    """Create `.venv` with a named host interpreter, then prove it works.

    The host interpreter is passed in rather than assumed to be `sys.executable`,
    because on Windows the interpreter that launched `train.py` may be MSYS2
    Python — which would produce `.venv/bin/python.exe` and leave every later
    step looking for a file that is not there.
    """
    host_executable = str(host_executable or sys.executable)
    assert_not_self_recreation(host_executable, venv)
    if expected_scheme is None:
        expected_scheme = classify_interpreter(
            interpreter_evidence(host_executable))["venv_scheme"]

    if not venv_python(venv, scheme=expected_scheme).exists():
        _run([host_executable, "-m", "venv", str(venv)], quiet=quiet)

    validation = validate_venv(venv, expected_scheme=expected_scheme,
                               expected_minor=expected_minor)
    if not validation["valid"]:
        raise BootstrapError(
            VENV_NOT_VALIDATED,
            f"the virtual environment at {venv} was created but did not validate: "
            f"{validation['why']}. Expected interpreter "
            f"{validation['expected_interpreter']}.",
            validation)
    return venv_python(venv, scheme=expected_scheme)


def _pip_version(python: Path) -> str | None:
    try:
        output = subprocess.check_output([str(python), "-m", "pip", "--version"],
                                         text=True, stderr=subprocess.DEVNULL,
                                         timeout=120)
    except Exception:                                        # noqa: BLE001
        return None
    match = re.search(r"pip\s+(\d+(?:\.\d+)*)", output)
    return match.group(1) if match else None


def pip_policy(contract: dict[str, Any]) -> dict[str, Any]:
    """The declared bootstrap-tooling policy, read rather than assumed."""
    block = dict((contract.get("virtual_environment") or {}).get("pip") or {})
    maximum = block.get("maximum_exclusive")
    return {"minimum": str(block.get("minimum", "24.0")),
            "maximum_exclusive": str(maximum) if maximum else None,
            "upgrade_policy": str(block.get("upgrade_policy",
                                            "BOUNDED_MINIMUM_ONLY")),
            "upgrade_setuptools": bool(block.get("upgrade_setuptools", False)),
            "upgrade_wheel": bool(block.get("upgrade_wheel", False))}


def ensure_pip_tooling(python: Path, contract: dict[str, Any], *,
                       quiet: bool = False) -> dict[str, Any]:
    """Bring pip up to the declared floor, and no further.

    The previous behaviour was an unbounded `pip install --upgrade pip`, which
    meant the resolver that chose this project's dependency set was whichever
    pip happened to be newest on the day. A deterministic bootstrap cannot have
    an undeclared component, so the upgrade is bounded on both sides and only
    happens when pip is actually below the floor. setuptools and wheel are never
    touched opportunistically.
    """
    policy = pip_policy(contract)
    before = _pip_version(python)
    report = {"policy": policy, "pip_before": before, "pip_after": before,
              "action": "KEPT", "requirement": None}
    if before and _version_tuple(before) >= _version_tuple(policy["minimum"]):
        report["why"] = (f"pip {before} already satisfies the declared floor "
                         f"{policy['minimum']}")
        return report
    requirement = f"pip>={policy['minimum']}"
    if policy["maximum_exclusive"]:
        requirement += f",<{policy['maximum_exclusive']}"
    _run([str(python), "-m", "pip", "install", "--upgrade", requirement], quiet=quiet)
    report.update({"action": "UPGRADED_TO_POLICY_FLOOR", "requirement": requirement,
                   "pip_after": _pip_version(python),
                   "why": f"pip {before} is below the declared floor "
                          f"{policy['minimum']}"})
    return report


def install_requirements(python: Path, profile: dict[str, Any], *,
                         quiet: bool = False,
                         contract: dict[str, Any] | None = None) -> dict[str, Any]:
    """Install the profile's locked set, offline from a wheelhouse when present."""
    requirements = REPO / str(profile.get("requirements"))
    offline = WHEELHOUSE.exists() and any(WHEELHOUSE.glob("*.whl"))
    command = [str(python), "-m", "pip", "install", "-r", str(requirements)]
    if offline:
        command += ["--no-index", "--find-links", str(WHEELHOUSE)]
    tooling = ensure_pip_tooling(python, contract or read_contract(), quiet=quiet)
    _run(command, quiet=quiet)
    # The project itself, so `import prism_fas` resolves without PYTHONPATH.
    _run([str(python), "-m", "pip", "install", "-e", ".", "--no-deps"], quiet=quiet)
    return {"requirements": requirements.relative_to(REPO).as_posix(),
            "offline_wheelhouse_used": offline,
            "bootstrap_tooling": tooling}


def verify_imports(python: Path, contract: dict[str, Any], *,
                   groups: list[str]) -> dict[str, Any]:
    """Check that the named import groups actually resolve in an environment.

    A requirement hash proves what was asked for. This proves what is there —
    which is the thing that fails at run time, hours in, on the host nobody
    tested. Run as a subprocess so it measures the target interpreter rather than
    whichever one happens to be executing this function.
    """
    checks = dict(contract.get("import_checks") or {})
    modules: list[str] = []
    for group in groups:
        value = checks.get(group) or []
        modules.extend(value if isinstance(value, list) else [value])
    if not modules:
        return {"groups": groups, "modules": [], "missing": [], "ok": True}

    probe = "\n".join([
        "import importlib, json",
        "missing = []",
        f"for name in {modules!r}:",
        "    try:",
        "        importlib.import_module(name)",
        "    except Exception:",
        "        missing.append(name)",
        "print(json.dumps(missing))",
    ])
    try:
        output = subprocess.check_output([str(python), "-c", probe], text=True,
                                         stderr=subprocess.DEVNULL, timeout=300)
        missing = json.loads(output.strip() or "[]")
    except Exception as error:                               # noqa: BLE001
        return {"groups": groups, "modules": modules, "missing": modules, "ok": False,
                "error": f"{type(error).__name__}: {error}"}
    return {"groups": groups, "modules": modules, "missing": missing,
            "ok": not missing}


def verify_torch_build(python: Path, profile: dict[str, Any]) -> dict[str, Any]:
    """Read the torch build that is actually installed, and judge it.

    A requirement file names an index; it cannot make pip use it. With
    `--extra-index-url` in play, PyPI stays in the resolution set, so a profile
    whose own index lacks a wheel for the host is satisfied silently by a
    different build. The manifest would then record a CUDA tag that nothing had
    ever verified. This asks the installed torch what it is.
    """
    tag = str(profile.get("cuda_tag") or "")
    report = {"declared_cuda_tag": tag or None, "checked": bool(tag)}
    if not tag:
        report["verdict"] = "NOT_APPLICABLE"
        return report

    probe = ("import json, torch; print(json.dumps({'version': torch.__version__, "
             "'cuda': torch.version.cuda}))")
    try:
        output = subprocess.check_output([str(python), "-c", probe], text=True,
                                         stderr=subprocess.DEVNULL, timeout=600)
        payload = json.loads(output.strip().splitlines()[-1])
    except Exception as error:                               # noqa: BLE001 - reported
        report.update({"verdict": "UNREADABLE",
                       "error": f"{type(error).__name__}: {error}"})
        return report

    version = str(payload.get("version") or "")
    local = version.partition("+")[2]
    report.update({"installed_version": version, "installed_cuda": payload.get("cuda"),
                   "local_version_label": local or None})
    if local == tag:
        report["verdict"] = "MATCHES_DECLARED_PROFILE"
        return report
    report["verdict"] = "SUBSTITUTED_BUILD"
    report["why"] = (
        f"the installed torch is {version!r}, whose build label is "
        f"{local or '(none)'!r}, but the selected profile declares {tag!r}. "
        "pip satisfied the pin from somewhere other than the profile's index.")
    return report


def installed_packages(python: Path) -> dict[str, str]:
    try:
        output = subprocess.check_output(
            [str(python), "-m", "pip", "list", "--format=json"],
            text=True, stderr=subprocess.DEVNULL, timeout=120)
        return {item["name"]: item["version"] for item in json.loads(output)}
    except Exception:                                        # noqa: BLE001
        return {}


# --- the entry point train.py calls -----------------------------------------

def import_groups(contract: dict[str, Any], *, scientific: bool) -> list[str]:
    """Which declared import groups must resolve before an environment is ready.

    A CPU rehearsal substitutes a fixture tower and never opens an ONNX session,
    so it does not need `science_only`. A scientific profile does — and that is
    precisely the environment where a half-finished pip install must NOT be
    adopted as complete.
    """
    checks = dict(contract.get("import_checks") or {})
    key = "required_for_science" if scientific else "required_for_rehearsal"
    groups = checks.get(key) or ["core", "reporting"]
    return list(groups) if isinstance(groups, list) else [str(groups)]


def ensure_environment(*, quiet: bool = False, allow_install: bool = True
                       ) -> dict[str, Any]:
    """Prepare the environment and return what the caller needs to re-exec.

    Returns a report with `interpreter` (the interpreter to use), `action`
    (`REUSED` / `ADOPTED` / `INSTALLED` / `INSTALL_REQUIRED`) and the full
    provenance that goes into the run's environment record.

    The order matters. The host interpreter is classified and, on Windows,
    replaced by a standard CPython BEFORE anything is created, because the
    interpreter decides the venv layout; an existing `.venv` is then graded
    against that decision rather than assumed to fit it.
    """
    contract = read_contract()
    inside = running_inside_project_venv()
    if inside:
        # Already the project environment: it is the answer, not a candidate.
        evidence = local_interpreter_evidence()
        verdict = classify_interpreter(evidence)
        host = {"executable": sys.executable,
                "classification": verdict["classification"],
                "venv_scheme": verdict["venv_scheme"],
                "scripts_dir": verdict["scripts_dir"],
                "selection": "PROJECT_VENV", "why": verdict["why"],
                "signals": verdict["signals"], "evidence": evidence,
                "host_classification": verdict["classification"], "fallback": None}
    else:
        host = resolve_host_interpreter(contract)

    python_report = check_python(contract, host["evidence"])
    platform_tag = host_platform_tag(host["evidence"])
    gpu = detect_gpu()
    selection = select_profile(contract, gpu, platform_tag=platform_tag)
    profile_id, profile = selection["profile_id"], selection["profile"]
    identity = environment_identity(profile_id, profile,
                                    str(python_report["found"]))
    scientific = bool(profile.get("supports_scientific_execution"))
    groups = import_groups(contract, scientific=scientific)

    scheme = host["venv_scheme"]
    interpreter = venv_python(VENV, scheme=scheme)
    expected_minor = None if inside else _minor(
        host["evidence"].get("version_info") or ())

    manifest = read_manifest(MANIFEST)
    identity_matches = bool(manifest
                            and manifest.get("environment_identity")
                            == identity["identity"])

    venv_state = classify_venv(VENV, expected_scheme=scheme,
                               expected_minor=expected_minor)
    action = "REUSED"
    install: dict[str, Any] = {}
    imports: dict[str, Any] = {}
    recovery: dict[str, Any] = {"state": venv_state["state"],
                                "action": venv_state["action"],
                                "why": venv_state["why"],
                                "rebuilt": False}

    if identity_matches and venv_state["state"] == VENV_VALID:
        # The recorded identity still describes this environment. Nothing is
        # installed and no package index is contacted.
        action = "REUSED"
    elif venv_state["action"] == VENV_REUSE:
        # An existing .venv with no matching manifest is the ordinary case for a
        # folder that was prepared by hand or copied mid-flight. Rather than
        # reinstalling on every invocation, verify what actually imports and
        # adopt it when it is complete - section 8's "verify its environment
        # identity before using it", answered by measurement rather than a hash.
        imports = verify_imports(interpreter, contract, groups=groups)
        if imports.get("ok"):
            action = "ADOPTED"
        elif not allow_install:
            action = "INSTALL_REQUIRED"
            recovery.update({"state": VENV_DEPENDENCIES_INCOMPLETE,
                             "action": VENV_INSTALL_INTO,
                             "why": "declared imports are missing: "
                                    f"{imports.get('missing')}"})
        else:
            # Structurally sound, dependencies incomplete: install into it. An
            # interrupted `pip install` costs the remaining packages, not a
            # rebuild, and never asks the operator to delete anything.
            recovery.update({"state": VENV_DEPENDENCIES_INCOMPLETE,
                             "action": VENV_INSTALL_INTO,
                             "why": "declared imports are missing: "
                                    f"{imports.get('missing')}"})
            install = install_requirements(interpreter, profile, quiet=quiet,
                                           contract=contract)
            imports = verify_imports(interpreter, contract, groups=groups)
            action = "INSTALLED"
    elif not allow_install:
        action = "INSTALL_REQUIRED"
    else:
        if venv_state["action"] == VENV_REBUILD:
            recovery["removed"] = remove_venv(VENV)
            recovery["rebuilt"] = True
        create_venv(VENV, quiet=quiet, host_executable=host["executable"],
                    expected_scheme=scheme, expected_minor=expected_minor)
        install = install_requirements(interpreter, profile, quiet=quiet,
                                       contract=contract)
        imports = verify_imports(interpreter, contract, groups=groups)
        action = "INSTALLED"

    torch_build: dict[str, Any] = {}
    if scientific and action in ("INSTALLED", "ADOPTED"):
        torch_build = verify_torch_build(interpreter, profile)
        if torch_build.get("verdict") == "SUBSTITUTED_BUILD":
            raise BootstrapError(
                CUDA_NOT_VALIDATED,
                "the installed PyTorch is not the build this profile declares.\n"
                f"    profile             {profile_id}  "
                f"({profile.get('torch_index')})\n"
                f"    declared build      {torch_build['declared_cuda_tag']}\n"
                f"    installed           {torch_build.get('installed_version')}  "
                f"(CUDA {torch_build.get('installed_cuda')})\n"
                "  The requirement file names an index; it cannot make pip use it. "
                "A wheel\n  that came from somewhere else may run and may not, and "
                "the environment\n  manifest would record a CUDA tag nobody "
                "verified. Extend\n  configs/environment/environment_contract.yaml "
                "deliberately, or install a\n  driver that satisfies a profile whose "
                "index publishes a wheel for this host.",
                {"profile_id": profile_id, "torch_build": torch_build,
                 "host_platform": platform_tag})

    if action in ("INSTALLED", "ADOPTED") and not imports.get("ok", True):
        # Never record an environment as ready when the thing that fails at run
        # time - the import - still fails.
        raise BootstrapError(
            BOOTSTRAP_FAILED,
            f"the environment at {VENV} is still incomplete after installation: "
            f"{imports.get('missing')} did not import. "
            f"Requirements: {profile.get('requirements')}.",
            {"imports": imports, "install": install, "venv": venv_state["state"]})

    report = {
        "schema_version": SCHEMA_VERSION,
        "environment_identity": identity["identity"],
        "identity_material": identity["material"],
        "requirement_files": identity["files"],
        "profile_id": profile_id,
        "profile_status": profile.get("status"),
        "profile_supports_scientific_execution": scientific,
        "selection_reason": selection["reason"],
        "candidates_considered": selection.get("candidates_considered", []),
        "python": python_report,
        "host_interpreter": {key: value for key, value in host.items()
                             if key != "evidence"},
        "host_interpreter_evidence": host["evidence"],
        "venv_scheme": scheme,
        "venv_recovery": recovery,
        "host_platform": platform_tag,
        "torch_build": torch_build,
        "gpu": gpu,
        "gpu_family": selection.get("family"),
        "venv": str(VENV.relative_to(REPO)) if VENV.is_relative_to(REPO) else str(VENV),
        "interpreter": str(interpreter),
        "action": action,
        "install": install,
        "required_import_groups": groups,
        "science_imports": None,
        "network_contacted": action == "INSTALLED" and not install.get(
            "offline_wheelhouse_used", False),
    }
    report["imports"] = imports
    if action in ("INSTALLED", "ADOPTED"):
        report["installed_packages"] = installed_packages(interpreter)
        write_manifest(report, MANIFEST)
    elif identity_matches and manifest:
        report["installed_packages"] = manifest.get("installed_packages", {})
    return report


def running_inside_project_venv() -> bool:
    """True when the current interpreter already IS the project environment.

    Compares prefixes rather than executable paths, so it answers correctly
    whichever layout the environment uses and whichever alias (python, python3,
    python3.12) started it.
    """
    try:
        return Path(sys.prefix).resolve() == VENV.resolve()
    except OSError:
        return False


__all__ = ["REPO", "CONTRACT", "VENV", "MANIFEST", "WHEELHOUSE", "SCHEMA_VERSION",
           "UNSUPPORTED_PYTHON", "CUDA_NOT_VALIDATED", "BOOTSTRAP_FAILED",
           "SUPPORTED_WINDOWS_CPYTHON_NOT_FOUND", "SELF_RECREATION_REFUSED",
           "VENV_NOT_VALIDATED", "BLOCKING_REASONS",
           "STANDARD_WINDOWS_CPYTHON", "MSYS2_MINGW_PYTHON", "POSIX_CPYTHON",
           "UNKNOWN_PYTHON", "WINDOWS_SCHEME", "POSIX_SCHEME", "SCRIPTS_DIR",
           "VENV_ABSENT", "VENV_VALID", "VENV_NO_INTERPRETER",
           "VENV_INTERPRETER_UNUSABLE", "VENV_WRONG_SCHEME", "VENV_WRONG_VERSION",
           "VENV_FOREIGN_PREFIX", "VENV_DEPENDENCIES_INCOMPLETE",
           "VENV_CREATE", "VENV_REUSE", "VENV_INSTALL_INTO", "VENV_REBUILD",
           "BootstrapError", "read_contract", "check_python", "detect_gpu",
           "gpu_family", "select_profile", "requirement_files", "environment_identity",
           "local_interpreter_evidence", "interpreter_evidence", "classify_interpreter",
           "discover_windows_interpreters", "select_windows_cpython",
           "resolve_host_interpreter", "venv_python", "existing_venv_interpreter",
           "classify_venv", "validate_venv", "assert_not_self_recreation",
           "remove_venv", "pip_policy", "ensure_pip_tooling", "import_groups",
           "host_platform_tag", "verify_torch_build", "classify_candidate",
           "WIN_AMD64", "LINUX_X86_64", "LINUX_AARCH64", "MACOS_ARM64",
           "UNKNOWN_PLATFORM",
           "read_manifest", "write_manifest", "create_venv",
           "install_requirements", "installed_packages", "ensure_environment",
           "running_inside_project_venv"]
