"""Validate-profile checks (L.2) and the firewall around them.

Two jobs. First, the checks must actually measure the repository rather than
agree with a document — so each one is also run against a deliberately broken
copy and must fail there. A check that cannot fail is not a check.

Second, nothing on the validate path may reach a live provider, a GPU job or a
target label. That is asserted structurally: the modules are inspected for the
imports that would make it possible, not merely trusted not to use them.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from prism_fas.pipeline import checks
from prism_fas.pipeline.checks import CHECKS, run_check
from prism_fas.pipeline.stages import STAGES

PIPELINE_DIR = Path(__file__).resolve().parents[2] / "src" / "prism_fas" / "pipeline"


# --- the checks measure something -------------------------------------------

def test_every_declared_check_exists() -> None:
    """A stage cannot reference a check that was never implemented."""
    declared = {check_id for stage in STAGES for check_id in stage.validate_checks}
    assert declared <= set(CHECKS)


def test_every_implemented_check_is_declared_by_a_stage() -> None:
    declared = {check_id for stage in STAGES for check_id in stage.validate_checks}
    assert set(CHECKS) == declared


def test_all_checks_pass_against_the_live_repository(repo: Path) -> None:
    failures = {check_id: run_check(check_id, repo).summary
                for check_id in CHECKS if not run_check(check_id, repo).ok}
    assert failures == {}


def test_an_unknown_check_id_fails_rather_than_raising(repo: Path) -> None:
    result = run_check("no_such_check", repo)
    assert not result.ok
    assert "no such check" in result.summary


def test_a_check_that_raises_becomes_a_failed_measurement(
        repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(_: Path):
        raise RuntimeError("boom")

    monkeypatch.setitem(CHECKS, "spec_sha256", explode)
    result = run_check("spec_sha256", repo)
    assert not result.ok
    assert "RuntimeError" in result.summary


# --- the checks can fail ----------------------------------------------------

def test_a_wrong_spec_is_detected(tmp_path: Path) -> None:
    target = tmp_path / checks.SPEC_RELPATH
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the spec")
    result = checks.check_spec_sha256(tmp_path)
    assert not result.ok
    assert result.detail["expected"] == checks.EXPECTED_SPEC_SHA256
    assert result.detail["actual"] != checks.EXPECTED_SPEC_SHA256


def test_a_missing_spec_is_detected(tmp_path: Path) -> None:
    assert not checks.check_spec_sha256(tmp_path).ok


def test_a_missing_acceptance_file_is_detected(tmp_path: Path) -> None:
    assert not checks.check_c0_acceptance(tmp_path).ok


def test_a_drifted_contract_identity_is_detected(repo: Path, tmp_path: Path) -> None:
    """Rewrite the lock's expected value; the live code must disagree with it."""
    import shutil

    for relative in ("configs", "src"):
        shutil.copytree(repo / relative, tmp_path / relative,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    lock_source = repo / checks.PRELIMINARY_LOCK
    lock_target = tmp_path / checks.PRELIMINARY_LOCK
    lock_target.parent.mkdir(parents=True)
    payload = json.loads(lock_source.read_text(encoding="utf-8"))
    payload["components"]["ontology_identity"] = "0" * 64
    lock_target.write_text(json.dumps(payload), encoding="utf-8")

    result = checks.check_contract_identities(tmp_path)
    assert not result.ok
    assert "ontology_identity" in result.detail["drifted"]


def test_c3_scientific_banks_verify_in_the_real_repository(repo: Path) -> None:
    """The frozen banks are complete and every identity re-derives.

    This replaces the old `c3_generation_not_started` test. That check asserted
    that no C3 generation evidence existed, which stopped being true when the
    authorized live 12x32 run completed on 2026-08-16 — and it kept passing only
    because its globs pointed at `reports/c3/raw_responses/` while the archives
    were written to `reports/c3/live/raw_responses/`. The obligation moved from
    proving a prohibition to proving the frozen result is intact.
    """
    result = checks.check_c3_scientific_banks_frozen(repo)
    assert result.ok, result.detail["problems"]
    assert result.detail["logical_requests_completed"] == 12
    assert result.detail["execution_profile"] == "full"
    for arm, row in result.detail["arms"].items():
        assert row["raw_slots"] == 384, arm
        assert row["selected"] == 256, arm
        assert row["bank_identity_reproduces"], arm
        assert row["recipes_jsonl_lines"] == 256, arm
        assert row["recipes_jsonl_lf_only"], arm


def test_c3_bank_drift_would_be_detected(repo: Path, tmp_path: Path) -> None:
    """The check measures, so a moved identity must trip it."""
    import shutil

    for relative in ("reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json",
                     "reports/c3/live/C3_LIVE_GENERATION_STATE.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
    shutil.copytree(repo / "assets/recipe_banks/c3", tmp_path / "assets/recipe_banks/c3")
    assert checks.check_c3_scientific_banks_frozen(tmp_path).ok

    bank = tmp_path / "assets/recipe_banks/c3/llm/C3_BANK.json"
    payload = json.loads(bank.read_text(encoding="utf-8"))
    payload["selected_recipe_identities"] = payload["selected_recipe_identities"][:-1]
    bank.write_text(json.dumps(payload), encoding="utf-8")

    result = checks.check_c3_scientific_banks_frozen(tmp_path)
    assert not result.ok
    assert any("LLM" in problem for problem in result.detail["problems"])


def test_the_retired_generation_check_is_gone(repo: Path) -> None:
    """A check that asserts a false thing must not survive under its own name."""
    assert not hasattr(checks, "check_c3_generation_not_started")
    assert "c3_generation_not_started" not in checks.CHECKS


def test_version_b_is_at_the_frozen_commit(repo: Path) -> None:
    result = checks.check_version_b_integrity(repo)
    assert result.ok
    assert result.detail["head"] == checks.EXPECTED_VERSION_B_HEAD
    assert result.detail["tag_peeled_commit"] == checks.EXPECTED_VERSION_B_HEAD
    assert result.detail["clean"] is True


def test_a_missing_version_b_is_detected(monkeypatch: pytest.MonkeyPatch,
                                         repo: Path, tmp_path: Path) -> None:
    monkeypatch.setattr(checks, "VERSION_B_PATH", tmp_path / "absent")
    assert not checks.check_version_b_integrity(repo).ok


def test_the_route_contract_is_checked_for_equality_not_containment(repo: Path) -> None:
    result = checks.check_route_contract_exact(repo)
    assert result.ok
    assert result.detail["actual"] == ["physics", "gpat"]


def test_both_c3_locks_still_verify(repo: Path) -> None:
    result = checks.check_c3_locks_verify(repo)
    assert result.ok, result.detail["problems"]
    assert result.detail["preliminary"]["body_hash_reproduces"]
    assert result.detail["superseding"]["status"] == "PRE_SCIENTIFIC_SUPERSEDING_CONTRACT_LOCK"


def test_the_c3_identities_re_derive_from_live_code(repo: Path) -> None:
    result = checks.check_c3_contract_identities(repo)
    assert result.ok, result.detail["drifted"]
    assert len(result.detail["comparisons"]) == 4


# --- firewall ---------------------------------------------------------------

#: Modules that can reach a provider, a GPU job or the target labels. None of
#: them may be imported by a TOP-LEVEL pipeline module: those run under every
#: profile including validate, which L.2 defines as static readiness.
#:
#: `pipeline/adapters/` is deliberately out of scope here — an adapter is the
#: layer that legitimately drives the planner and a mock or replay provider, and
#: its stricter rules (module-level bans plus one named gated lazy import) are
#: enforced in test_adapters_integration.py.
#: Modules exempt from the torch ban, by name and with a reason. `gpu_preflight`
#: exists to prove a GPU can execute the pipeline, so importing torch is its
#: entire job. It is imported from the zero-argument runner AFTER the profile is
#: resolved and never from the validate path, which is what the ban protects.
TORCH_EXEMPT = {"gpu_preflight.py"}

FORBIDDEN_IMPORTS = (
    "prism_fas.llm.providers",
    "prism_fas.llm.pipeline",
    "prism_fas.cloud",
    "prism_fas.data.target_eval",
    "prism_fas.evaluation.target_prediction",
    "google.genai",
    "modal",
    "torch",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module_path", sorted(PIPELINE_DIR.glob("*.py")),
                         ids=lambda path: path.name)
def test_no_pipeline_module_imports_a_provider_gpu_or_target_module(
        module_path: Path) -> None:
    imported = _imported_modules(module_path)
    forbidden_here = tuple(item for item in FORBIDDEN_IMPORTS
                           if not (item == "torch"
                                   and module_path.name in TORCH_EXEMPT))
    offending = {name for name in imported
                 if any(name == forbidden or name.startswith(f"{forbidden}.")
                        for forbidden in forbidden_here)}
    assert not offending, f"{module_path.name} imports {sorted(offending)}"


def test_train_py_imports_no_provider_gpu_or_target_module(repo: Path) -> None:
    imported = _imported_modules(repo / "train.py")
    offending = {name for name in imported
                 if any(name == forbidden or name.startswith(f"{forbidden}.")
                        for forbidden in FORBIDDEN_IMPORTS)}
    assert not offending, f"train.py imports {sorted(offending)}"


def test_train_py_delegates_rather_than_implementing(repo: Path) -> None:
    """L.4: the entrypoint must not absorb recipe, GPAT, synthesis or lock logic.

    The zero-argument runner added argument handling, bootstrap dispatch and
    console formatting to this file — all of which L.4 permits, because none of
    them is science. What it forbids is the entrypoint OWNING scientific
    behaviour, so the assertion is about the names it defines rather than their
    number: anything that looks like a pipeline implementation must live under
    src/prism_fas/ and be called from here.
    """
    tree = ast.parse((repo / "train.py").read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    # `_diagnose_data` is console formatting over `preparation.diagnose`: it
    # measures nothing itself and owns no pipeline behaviour.
    allowed = {"build_parser", "main", "_bootstrap_and_reexec", "_zero_argument",
               "_explicit", "_print_stage_table", "_git_identity",
               "_diagnose_data"}
    assert functions <= allowed, f"train.py defines unexpected {sorted(functions - allowed)}"
    forbidden = ("train", "fit", "select", "score", "render", "compile", "freeze",
                 "evaluate", "calibrate", "generate")
    for name in functions:
        stem = name.lstrip("_").split("_")[0]
        assert stem not in forbidden, f"train.py defines {name}, which sounds like science"
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


#: Config paths the runtime names for artifacts a LATER stage produces. Each is
#: guarded at its call site and fails closed with its own message, so naming an
#: absent file is correct there rather than a typo waiting to fire.
CONFIGS_NOT_YET_BUILT = {
    "configs/data/target_layout.yaml":
        "declares where the sealed held-out target package lives. Built by the "
        "C10 target preparation, which has never run; `sources.py` checks "
        "`is_file()` and raises SourceUnavailable rather than opening it.",
}


def _literal_paths(tree: ast.AST) -> set[tuple[int, str]]:
    """Every `configs/...` or `assets/...` path this module spells out.

    Catches both forms the codebase uses: a whole path in one string, and a
    `repo / "configs" / "models" / "x.yaml"` division chain.
    """
    def chain(node: ast.AST) -> list[str]:
        parts: list[str] = []
        while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if not (isinstance(node.right, ast.Constant)
                    and isinstance(node.right.value, str)):
                return []            # a variable in the chain: not a literal path
            parts.append(node.right.value)
            node = node.left
        return list(reversed(parts))

    found: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        values: list[str] = []
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            parts = chain(node)
            if parts:
                values.append("/".join(parts))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append(node.value)
        for value in values:
            value = value.lstrip("./")
            if not value.startswith(("configs/", "assets/")) or "{" in value:
                continue
            # Prose that opens with a path — an error message naming the file it
            # could not find — is not itself a path reference.
            if any(character.isspace() for character in value):
                continue
            found.add((node.lineno, value))
    return found


def test_every_config_the_runtime_names_actually_exists(repo: Path) -> None:
    """The guard for the class of defect that stopped three GPU runs in a row.

    `_step_m3b` handed `build_m3b_package` a path to
    `configs/models/model_priors.yaml`. That file has never existed — the
    canonical config is `m3b_priors.yaml` — and the only thing standing between
    the typo and the operator was a fixture that wrote a placeholder under the
    wrong name. It surfaced after M2's hours of SCRFD and all of M3A had already
    completed on the real host.

    So the whole runtime tree is walked here, not just preparation: a path
    spelled in code and absent from the folder fails on this machine instead of
    on the third step of a multi-hour run.
    """
    sources = sorted((repo / "src" / "prism_fas").rglob("*.py")) + [repo / "train.py"]
    missing: list[str] = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, value in _literal_paths(tree):
            if value in CONFIGS_NOT_YET_BUILT:
                continue
            target = repo / value
            if not (target.is_file() or target.is_dir()):
                missing.append(f"{path.relative_to(repo).as_posix()}:{line} -> {value}")
    assert missing == [], (
        "these paths are named by runtime code and absent from the folder: "
        + "; ".join(sorted(missing)))


@pytest.mark.parametrize("relative", sorted(CONFIGS_NOT_YET_BUILT))
def test_a_config_that_does_not_exist_yet_is_guarded_not_opened(
        repo: Path, relative: str) -> None:
    """An exemption is only honest while the call site still checks first."""
    assert not (repo / relative).is_file(), (
        f"{relative} now exists; remove it from CONFIGS_NOT_YET_BUILT so the "
        "audit checks it like every other config")
    users = [path for path in (repo / "src" / "prism_fas").rglob("*.py")
             if relative in path.read_text(encoding="utf-8")]
    assert users, f"nothing references {relative}; drop the exemption"
    for path in users:
        source = path.read_text(encoding="utf-8")
        assert "is_file()" in source, (
            f"{path.name} names {relative} without checking it exists")


def test_a_validate_run_leaves_no_provider_credential_in_its_artifacts(
        repo: Path) -> None:
    """No artifact may serialize anything shaped like a key."""
    for path in (repo / "reports" / "validate").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "AIza" not in text
        assert "GEMINI_API_KEY" not in text
