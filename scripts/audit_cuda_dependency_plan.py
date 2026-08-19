"""Check every declared hardware profile against the wheel index it names.

A profile is a promise that a specific torch build exists for the host it will
be selected on. Nothing had ever checked that promise, and one of them was false:
the CUDA 12.9 index publishes torch 2.13.0 for Linux only, so on Windows the
`--extra-index-url` would quietly fall through to the PyPI wheel — a different
CUDA build from the one the manifest would then claim was installed.

This queries the indices, records what is actually published per platform, and
writes reports/handoff/CUDA_DEPENDENCY_PLAN_EVIDENCE.json.

    python scripts/audit_cuda_dependency_plan.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import bootstrap as boot  # noqa: E402

PYPI = "https://pypi.org/pypi/{package}/{version}/json"
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}/{package}/"

PLATFORM_PATTERNS = {
    "win_amd64": "win_amd64",
    "linux_x86_64": "manylinux",          # narrowed below
    "linux_aarch64": "aarch64",
    "macosx_arm64": "macosx",
}


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as stream:
        return stream.read().decode("utf-8", "replace")


def index_wheels(tag: str, package: str, version: str) -> list[str]:
    try:
        html = fetch(TORCH_INDEX.format(tag=tag, package=package))
    except Exception as error:                               # noqa: BLE001
        return [f"INDEX_ERROR: {type(error).__name__}: {error}"]
    names = sorted(set(re.findall(r">([^<]+\.whl)<", html)))
    return [name for name in names if name.startswith(f"{package}-{version}+{tag}-")]


def pypi_wheels(package: str, version: str) -> list[str]:
    payload = json.loads(fetch(PYPI.format(package=package, version=version)))
    return sorted(item["filename"] for item in payload["urls"])


def platforms_of(filenames: list[str]) -> dict[str, list[str]]:
    found: dict[str, set[str]] = {key: set() for key in PLATFORM_PATTERNS}
    for name in filenames:
        if not name.endswith(".whl"):
            continue
        parts = name.split("-")
        tag = parts[2] if len(parts) > 2 else ""
        if "win_amd64" in name:
            found["win_amd64"].add(tag)
        elif "aarch64" in name:
            found["linux_aarch64"].add(tag)
        elif "manylinux" in name or "linux_x86_64" in name:
            found["linux_x86_64"].add(tag)
        elif "macosx" in name:
            found["macosx_arm64"].add(tag)
    return {key: sorted(value) for key, value in found.items() if value}


def main() -> int:
    contract = boot.read_contract()
    minimum = contract["python"]["minimum"]
    maximum = contract["python"]["maximum_exclusive"]
    required = ["cp" + f"3{minor}" for minor in
                range(int(minimum.split(".")[1]), int(maximum.split(".")[1]))]

    profiles = {}
    for profile_id, profile in contract["profiles"].items():
        requirements = (REPO / str(profile["requirements"])).read_text(
            encoding="utf-8")
        pins = dict(re.findall(r"^([A-Za-z0-9._-]+)==([^\s#]+)", requirements,
                               flags=re.MULTILINE))
        index = re.search(r"--extra-index-url\s+(\S+)", requirements)
        tag = str(profile.get("cuda_tag") or "cpu")
        entry = {
            "requirements": profile["requirements"],
            "declared_index": index.group(1) if index else None,
            "contract_index": profile.get("torch_index"),
            "index_matches_contract": bool(index)
            and index.group(1) == profile.get("torch_index"),
            "cuda_tag": tag,
            "torch": pins.get("torch"),
            "torchvision": pins.get("torchvision"),
            "torch_matches_contract": pins.get("torch") == str(profile.get("torch")),
            "declared_platforms": profile.get("platforms"),
        }
        for package, key in (("torch", "torch"), ("torchvision", "torchvision")):
            version = pins.get(package)
            if not version:
                continue
            names = index_wheels(tag, package, version)
            entry[f"{key}_index_wheels"] = len(names)
            entry[f"{key}_index_platforms"] = platforms_of(names)
        profiles[profile_id] = entry

        # What the host would actually get for every platform the profile claims.
        published = entry.get("torch_index_platforms", {})
        entry["platforms_with_a_declared_wheel"] = sorted(
            platform for platform, tags in published.items()
            if set(required) <= set(tags))
        entry["platforms_missing_a_declared_wheel"] = sorted(
            set(PLATFORM_PATTERNS) - set(entry["platforms_with_a_declared_wheel"]))

    pypi_torch = pypi_wheels("torch", "2.13.0")
    findings = {
        "schema_version": "prism-cuda-dependency-plan-evidence-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "question": "does every declared hardware profile have a real wheel for "
                    "every platform it can be selected on?",
        "supported_python_tags": required,
        "profiles": profiles,
        "pypi_fallback": {
            "why_it_matters": "the CUDA requirement files use --extra-index-url, so "
                              "PyPI stays in the resolution set. A profile whose own "
                              "index has no wheel for the host platform therefore "
                              "does not fail loudly: pip installs the PyPI build "
                              "instead, and the environment manifest goes on naming "
                              "the CUDA index it did not use.",
            "pypi_torch_2_13_0_platforms": platforms_of(pypi_torch),
        },
        "verdict_per_profile": {
            profile_id: ("OK" if entry["platforms_with_a_declared_wheel"] else
                         "NO_WHEEL_FOR_ANY_PLATFORM")
            for profile_id, entry in profiles.items()},
    }

    out = REPO / "reports" / "handoff" / "CUDA_DEPENDENCY_PLAN_EVIDENCE.json"
    out.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")

    for profile_id, entry in profiles.items():
        print(f"{profile_id:12s} torch={entry.get('torch')} "
              f"index_ok={entry['index_matches_contract']} "
              f"has_wheel={entry['platforms_with_a_declared_wheel']} "
              f"missing={entry['platforms_missing_a_declared_wheel']}")
    print("wrote", out.relative_to(REPO).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
