"""Orchestrator integration and the suite's own firewall (required tests 22-30).

The firewall tests at the end are the ones that make the rest trustworthy. Every
other test asserts that the code *chose* not to call a provider; these assert it
*could* not — no module-level import reaches a vendor SDK, a GPU runtime or the
target-evaluation path, and the one lazy import that can is unreachable without
passing the live gate first.
"""
from __future__ import annotations

import ast
import json
import socket
from pathlib import Path

import pytest

from prism_fas.pipeline.orchestrator import run

from conftest_adapters import make_sandbox

PIPELINE_DIR = Path(__file__).resolve().parents[2] / "src" / "prism_fas" / "pipeline"


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    return make_sandbox(tmp_path / "repo")


# --- 22, 23. state and index across runs -------------------------------------

def test_pipeline_state_records_the_stage_range_and_c3_cursor(sandbox: Path) -> None:
    run(repo=sandbox, profile_name="smoke", first_stage="C0", last_stage="C3")
    state = json.loads((sandbox / "state/PIPELINE_STATE.json").read_text(encoding="utf-8"))

    assert state["stage_range"] == ["C0", "C3"]
    assert state["substage"] == "C3"
    assert state["execution_profile"] == "smoke"
    assert state["scientific_eligible"] is False
    assert state["c3_logical_request_state"]["total_requests"] == 12
    assert state["c3_logical_request_state"]["all_complete"] is True
    assert "engineering_status_scope" in state


def test_pipeline_state_is_rewritten_atomically_between_runs(sandbox: Path) -> None:
    run(repo=sandbox, profile_name="validate", first_stage="C0", last_stage="C1")
    run(repo=sandbox, profile_name="validate", first_stage="C0", last_stage="C3")
    state = json.loads((sandbox / "state/PIPELINE_STATE.json").read_text(encoding="utf-8"))
    assert state["stage_range"] == ["C0", "C3"]
    assert not list((sandbox / "state").glob("*.tmp"))
    assert not list((sandbox / "state").glob(".*"))


def test_index_records_a_row_per_substage(sandbox: Path) -> None:
    run(repo=sandbox, profile_name="smoke", first_stage="C0", last_stage="C3")
    index = json.loads((sandbox / "state/MASTER_RUN_INDEX.json").read_text(encoding="utf-8"))
    substages = {row["substage"] for row in index["runs"]}
    assert {"C0", "C1", "C2", "C2B", "C2C", "C3"} <= substages


def test_a_later_run_never_drops_an_earlier_row(sandbox: Path) -> None:
    """L.8: a FAIL or BLOCKED row stays addressable after a later success."""
    run(repo=sandbox, profile_name="smoke")                      # C0-C13: blocks C4+
    first = json.loads((sandbox / "state/MASTER_RUN_INDEX.json").read_text(encoding="utf-8"))
    blocked_ids = {row["run_id"] for row in first["runs"] if row["status"] == "BLOCKED"}
    assert blocked_ids

    run(repo=sandbox, profile_name="smoke", first_stage="C0", last_stage="C3")
    second = json.loads((sandbox / "state/MASTER_RUN_INDEX.json").read_text(encoding="utf-8"))
    surviving = {row["run_id"] for row in second["runs"]}
    assert blocked_ids <= surviving, "a blocked row was dropped by a later successful run"


# --- 24, 25. eligibility of smoke and blocked rows ---------------------------

def test_every_smoke_row_is_not_scientifically_eligible(sandbox: Path) -> None:
    run(repo=sandbox, profile_name="smoke", first_stage="C0", last_stage="C3")
    index = json.loads((sandbox / "state/MASTER_RUN_INDEX.json").read_text(encoding="utf-8"))
    assert index["runs"]
    for row in index["runs"]:
        assert row["execution_profile"] == "smoke"
        assert row["scientific_eligible"] is False


def test_every_blocked_row_is_not_scientifically_eligible(sandbox: Path) -> None:
    run(repo=sandbox, profile_name="full")
    index = json.loads((sandbox / "state/MASTER_RUN_INDEX.json").read_text(encoding="utf-8"))
    blocked = [row for row in index["runs"] if row["status"] == "BLOCKED"]
    assert blocked
    assert all(row["scientific_eligible"] is False for row in blocked)


def test_smoke_artifacts_serialize_both_l2_fields(sandbox: Path) -> None:
    run(repo=sandbox, profile_name="smoke", first_stage="C0", last_stage="C3")
    for stage_id in ("C0", "C1", "C2", "C3"):
        payload = json.loads(
            (sandbox / "reports/smoke" / stage_id.lower()
             / f"{stage_id}_SMOKE.json").read_text(encoding="utf-8"))
        assert payload["execution_profile"] == "smoke"
        assert payload["scientific_eligible"] is False


# --- 26. full C3 cannot start if the pre-live gate fails ---------------------

def test_full_c3_cannot_start_when_the_pre_live_gate_fails(sandbox: Path) -> None:
    """Break an ancestor, then confirm C3 never reaches a generating mode."""
    (sandbox / "reports/c2c/C2C_ACCEPTANCE.json").unlink()

    result = run(repo=sandbox, profile_name="full", first_stage="C3", last_stage="C3")
    assert result.outcome in {"FAIL", "BLOCKED"}
    assert result.provider_calls == 0

    c3 = result.outcomes[0]
    modes = {item.mode for item in c3.adapter_results}
    assert modes == {"PRE_LIVE_VERIFY"}, "C3 entered a generating mode despite a failed gate"


def test_full_makes_zero_provider_calls_without_authorization(sandbox: Path) -> None:
    result = run(repo=sandbox, profile_name="full", first_stage="C0", last_stage="C3")
    assert result.provider_calls == 0


# --- 27, 28. frozen identities and locks unchanged ---------------------------

def _hashes(repo: Path) -> dict[str, str]:
    import hashlib

    targets = [
        "reports/c3/C3_BANK_LOCK.json",
        "reports/c3/v15_selection_contract/C3_BANK_CONTRACT_LOCK.json",
        "configs/recipes/ontology_m7.yaml",
        "configs/version_c/llm/c2c_route_policy.yaml",
        "configs/version_c/llm/c3_selection_contract.yaml",
        "src/prism_fas/recipes/selection.py",
    ]
    return {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in targets}


def test_running_every_profile_leaves_frozen_artifacts_byte_identical(
        sandbox: Path) -> None:
    before = _hashes(sandbox)
    run(repo=sandbox, profile_name="validate", first_stage="C0", last_stage="C3")
    run(repo=sandbox, profile_name="smoke", first_stage="C0", last_stage="C3")
    run(repo=sandbox, profile_name="full", first_stage="C0", last_stage="C3")
    assert _hashes(sandbox) == before


def test_no_offline_profile_writes_c3_generation_state_or_archives(
        sandbox: Path) -> None:
    """An offline profile must not touch the scientific C3 live namespace.

    Since the real C3 generation ran, that namespace now legitimately holds a
    state file and 12 archives. So the assertion is not absence — it is that
    validate and smoke leave those bytes exactly as they found them, and put
    their own rehearsal state somewhere else entirely.
    """
    import hashlib

    live = sandbox / "reports/c3/live"

    def fingerprint() -> dict[str, str]:
        return {path.relative_to(live).as_posix():
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(live.rglob("*")) if path.is_file()}

    before = fingerprint()
    run(repo=sandbox, profile_name="validate", first_stage="C0", last_stage="C3")
    run(repo=sandbox, profile_name="smoke", first_stage="C0", last_stage="C3")

    assert fingerprint() == before, "an offline profile modified the C3 scientific namespace"
    # The smoke rehearsal's own state went to the smoke namespace instead.
    assert (sandbox / "reports/smoke/c3/live/C3_LIVE_GENERATION_STATE.json").exists()


def test_the_c3_generation_prohibition_still_holds_after_a_smoke_run(
        sandbox: Path) -> None:
    """A smoke rehearsal must not look like C3 generation to the pre-live check."""
    from prism_fas.pipeline.checks import check_c3_generation_not_started

    run(repo=sandbox, profile_name="smoke", first_stage="C0", last_stage="C3")
    result = check_c3_generation_not_started(sandbox)
    assert result.ok, result.detail["found"]


# --- 29. the suite itself cannot reach a network -----------------------------

def test_socket_connections_are_blocked_in_this_suite() -> None:
    with pytest.raises(AssertionError, match="attempted a network connection"):
        socket.create_connection(("example.invalid", 443))


def test_ambient_credentials_are_deleted_in_this_suite() -> None:
    import os

    assert os.environ.get("GEMINI_API_KEY") is None
    assert os.environ.get("GOOGLE_API_KEY") is None


# --- 30. no provider / GPU / Modal / target import is reachable --------------

FORBIDDEN_MODULES = (
    "prism_fas.cloud",
    "prism_fas.data.target_eval",
    "prism_fas.evaluation.target_prediction",
    "google.genai",
    "modal",
    "torch",
)

#: The single gated exception. `_build_provider` imports the Gemini provider
#: only after `assert_binding_permitted` has already accepted a LIVE binding, so
#: the import is unreachable unless the profile, the stage and an explicit human
#: authorization all agree. It is named here so adding a second one has to be a
#: deliberate edit to this list.
GATED_LAZY_IMPORTS = {
    ("adapters/c3.py", "prism_fas.llm.providers.gemini"),
}


def _imports(path: Path) -> set[tuple[str, bool]]:
    """(module, is_module_level) for every import in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level = {id(node) for node in tree.body}
    found: set[tuple[str, bool]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        found.update((name, id(node) in top_level) for name in names)
    return found


@pytest.mark.parametrize("module_path", sorted(PIPELINE_DIR.rglob("*.py")),
                         ids=lambda path: str(path.relative_to(PIPELINE_DIR)))
def test_no_pipeline_module_imports_a_provider_gpu_or_target_module(
        module_path: Path) -> None:
    relative = module_path.relative_to(PIPELINE_DIR).as_posix()
    offending = set()
    for name, _module_level in _imports(module_path):
        if not any(name == item or name.startswith(f"{item}.")
                   for item in FORBIDDEN_MODULES):
            continue
        if (relative, name) in GATED_LAZY_IMPORTS:
            continue
        offending.add(name)
    assert not offending, f"{relative} imports {sorted(offending)}"


def test_the_gemini_provider_is_never_imported_at_module_level() -> None:
    """The gated exception must stay lazy, or importing the package loads the SDK."""
    for module_path in PIPELINE_DIR.rglob("*.py"):
        for name, module_level in _imports(module_path):
            if name.startswith("prism_fas.llm.providers.gemini"):
                assert not module_level, (
                    f"{module_path.relative_to(PIPELINE_DIR)} imports the Gemini provider "
                    "at module level; it must stay behind the live-binding gate")


def test_importing_the_pipeline_package_does_not_load_a_vendor_sdk() -> None:
    """The structural version of the same claim, measured at runtime."""
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import prism_fas.pipeline.orchestrator;"
        "import prism_fas.pipeline.adapters.c3;"
        "import prism_fas.pipeline.adapters.registry;"
        "loaded=[m for m in sys.modules if m.startswith(('google.genai','modal'))];"
        "print(loaded)")
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        cwd=str(PIPELINE_DIR.parents[2]))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"a vendor SDK was loaded: {result.stdout}"
