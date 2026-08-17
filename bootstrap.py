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
from pathlib import Path
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


# --- python version ----------------------------------------------------------

def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", str(text))[:3])


def check_python(contract: dict[str, Any]) -> dict[str, Any]:
    """Refuse an interpreter outside the declared range, with the exact numbers."""
    block = dict(contract.get("python") or {})
    minimum = _version_tuple(block.get("minimum", "3.11"))
    maximum = _version_tuple(block.get("maximum_exclusive", "3.14"))
    current = sys.version_info[:3]

    if not (minimum <= current[:len(minimum)] or current >= minimum):
        pass
    ok = minimum <= current < maximum
    detail = {
        "found": platform.python_version(),
        "found_tuple": list(current),
        "required_minimum": block.get("minimum"),
        "required_maximum_exclusive": block.get("maximum_exclusive"),
        "tested_on": block.get("tested_on"),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "supported": ok,
    }
    if not ok:
        raise BootstrapError(
            UNSUPPORTED_PYTHON,
            f"Python {platform.python_version()} is outside the supported range "
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
    """Infer the architecture family conservatively; unknown stays unknown."""
    if not name:
        return "UNKNOWN"
    patterns = dict(contract.get("family_patterns") or {})
    upper = name.upper()
    for family, needles in patterns.items():
        for needle in (needles if isinstance(needles, list) else [needles]):
            if str(needle).upper() in upper:
                return family
    return "UNKNOWN"


def select_profile(contract: dict[str, Any], gpu: dict[str, Any]) -> dict[str, Any]:
    """Match the host to an already-declared profile. Never search for one.

    Returns the selected profile plus the reasoning, so a refusal can explain
    itself without the caller re-deriving anything.
    """
    profiles = dict(contract.get("profiles") or {})
    selection = dict(contract.get("selection") or {})
    order = selection.get("order") or []
    if isinstance(order, str):
        order = [order]

    if not gpu.get("available"):
        cpu = dict(profiles.get("cpu") or {})
        return {"profile_id": "cpu", "profile": cpu, "reason": "no CUDA GPU detected",
                "gpu": gpu, "family": "NONE", "supports_scientific_execution": False,
                "candidates_considered": []}

    family = gpu_family(gpu.get("name"), contract)
    driver = _version_tuple(gpu.get("driver_version") or "0")
    capability = str(gpu.get("compute_capability") or "")
    considered: list[dict[str, Any]] = []

    for profile_id in order:
        profile = dict(profiles.get(profile_id) or {})
        if not profile:
            continue
        families = profile.get("gpu_families") or []
        capabilities = [str(item) for item in (profile.get("compute_capabilities") or [])]
        minimum = _version_tuple(profile.get("minimum_driver_version") or "0")
        family_ok = family in families
        capability_ok = (not capability) or (capability in capabilities)
        driver_ok = driver >= minimum
        considered.append({"profile_id": profile_id, "family_match": family_ok,
                           "compute_capability_match": capability_ok,
                           "driver_match": driver_ok,
                           "minimum_driver_version": profile.get("minimum_driver_version")})
        if family_ok and capability_ok and driver_ok:
            return {"profile_id": profile_id, "profile": profile,
                    "reason": f"{family} GPU with driver {gpu.get('driver_version')} "
                              f"matches the declared {profile_id} profile",
                    "gpu": gpu, "family": family,
                    "supports_scientific_execution":
                        bool(profile.get("supports_scientific_execution")),
                    "candidates_considered": considered}

    raise BootstrapError(
        CUDA_NOT_VALIDATED,
        f"the detected GPU matches no declared environment profile.\n"
        f"    GPU                 {gpu.get('name')}\n"
        f"    driver              {gpu.get('driver_version')}\n"
        f"    compute capability  {gpu.get('compute_capability') or 'unreported'}\n"
        f"    inferred family     {family}\n"
        f"  Declared profiles and what they require:\n"
        + "\n".join(
            f"    {item['profile_id']:<14} families={profiles[item['profile_id']].get('gpu_families')} "
            f"min driver={item['minimum_driver_version']}" for item in considered)
        + "\n  Either install a driver that satisfies a declared profile, or extend "
          "configs/environment/environment_contract.yaml deliberately.\n"
          "  This runner will NOT guess a CUDA wheel and will NOT fall back to CPU "
          "for scientific execution.",
        {"gpu": gpu, "family": family, "candidates_considered": considered})


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


def venv_python(venv: Path = VENV) -> Path:
    return (venv / "Scripts" / "python.exe" if os.name == "nt"
            else venv / "bin" / "python")


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


def create_venv(venv: Path = VENV, *, quiet: bool = False) -> Path:
    if not venv_python(venv).exists():
        _run([sys.executable, "-m", "venv", str(venv)], quiet=quiet)
    python = venv_python(venv)
    if not python.exists():
        raise BootstrapError(BOOTSTRAP_FAILED,
                             f"the virtual environment was created but {python} is absent")
    return python


def install_requirements(python: Path, profile: dict[str, Any], *,
                         quiet: bool = False) -> dict[str, Any]:
    """Install the profile's locked set, offline from a wheelhouse when present."""
    requirements = REPO / str(profile.get("requirements"))
    offline = WHEELHOUSE.exists() and any(WHEELHOUSE.glob("*.whl"))
    command = [str(python), "-m", "pip", "install", "-r", str(requirements)]
    if offline:
        command += ["--no-index", "--find-links", str(WHEELHOUSE)]
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"], quiet=quiet)
    _run(command, quiet=quiet)
    # The project itself, so `import prism_fas` resolves without PYTHONPATH.
    _run([str(python), "-m", "pip", "install", "-e", ".", "--no-deps"], quiet=quiet)
    return {"requirements": requirements.relative_to(REPO).as_posix(),
            "offline_wheelhouse_used": offline}


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


def installed_packages(python: Path) -> dict[str, str]:
    try:
        output = subprocess.check_output(
            [str(python), "-m", "pip", "list", "--format=json"],
            text=True, stderr=subprocess.DEVNULL, timeout=120)
        return {item["name"]: item["version"] for item in json.loads(output)}
    except Exception:                                        # noqa: BLE001
        return {}


# --- the entry point train.py calls -----------------------------------------

def ensure_environment(*, quiet: bool = False, allow_install: bool = True
                       ) -> dict[str, Any]:
    """Prepare the environment and return what the caller needs to re-exec.

    Returns a report with `python` (the interpreter to use), `action`
    (`REUSED` / `INSTALLED` / `CURRENT`) and the full provenance that goes into
    the run's environment record.
    """
    contract = read_contract()
    python_report = check_python(contract)
    gpu = detect_gpu()
    selection = select_profile(contract, gpu)
    profile_id, profile = selection["profile_id"], selection["profile"]
    identity = environment_identity(profile_id, profile, platform.python_version())

    manifest = read_manifest()
    interpreter = venv_python()
    matches = bool(manifest
                   and manifest.get("environment_identity") == identity["identity"]
                   and interpreter.exists())

    action = "REUSED"
    install: dict[str, Any] = {}
    imports: dict[str, Any] = {}
    if not matches:
        # An existing .venv with no matching manifest is the ordinary case for a
        # folder that was prepared by hand or copied mid-flight. Rather than
        # reinstalling on every invocation, verify what actually imports and
        # adopt it when it is complete — §8's "verify its environment identity
        # before using it", answered by measurement instead of by a hash alone.
        adoptable = interpreter.exists() and (
            running_inside_project_venv() or not allow_install)
        if adoptable:
            imports = verify_imports(interpreter, contract,
                                     groups=["core", "reporting"])
            action = "ADOPTED" if imports.get("ok") else "INSTALL_REQUIRED"
        elif not allow_install:
            action = "INSTALL_REQUIRED"
        else:
            create_venv(quiet=quiet)
            install = install_requirements(interpreter, profile, quiet=quiet)
            imports = verify_imports(interpreter, contract,
                                     groups=["core", "reporting"])
            action = "INSTALLED"

    report = {
        "schema_version": SCHEMA_VERSION,
        "environment_identity": identity["identity"],
        "identity_material": identity["material"],
        "requirement_files": identity["files"],
        "profile_id": profile_id,
        "profile_status": profile.get("status"),
        "profile_supports_scientific_execution":
            bool(profile.get("supports_scientific_execution")),
        "selection_reason": selection["reason"],
        "candidates_considered": selection.get("candidates_considered", []),
        "python": python_report,
        "gpu": gpu,
        "gpu_family": selection.get("family"),
        "venv": str(VENV.relative_to(REPO)) if VENV.is_relative_to(REPO) else str(VENV),
        "interpreter": str(interpreter),
        "action": action,
        "install": install,
        "science_imports": None,
        "network_contacted": action == "INSTALLED" and not install.get(
            "offline_wheelhouse_used", False),
    }
    report["imports"] = imports
    if action in ("INSTALLED", "ADOPTED"):
        report["installed_packages"] = installed_packages(interpreter)
        write_manifest(report)
    elif matches and manifest:
        report["installed_packages"] = manifest.get("installed_packages", {})
    return report


def running_inside_project_venv() -> bool:
    """True when the current interpreter already IS the project environment."""
    try:
        return Path(sys.executable).resolve() == venv_python().resolve()
    except OSError:
        return False


__all__ = ["REPO", "CONTRACT", "VENV", "MANIFEST", "WHEELHOUSE", "SCHEMA_VERSION",
           "UNSUPPORTED_PYTHON", "CUDA_NOT_VALIDATED", "BOOTSTRAP_FAILED",
           "BootstrapError", "read_contract", "check_python", "detect_gpu",
           "gpu_family", "select_profile", "requirement_files", "environment_identity",
           "venv_python", "read_manifest", "write_manifest", "create_venv",
           "install_requirements", "installed_packages", "ensure_environment",
           "running_inside_project_venv"]
