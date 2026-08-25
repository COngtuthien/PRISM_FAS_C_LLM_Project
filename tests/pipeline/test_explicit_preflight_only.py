"""`--preflight-only` must mean what it says under the EXPLICIT `--profile` path.

The defect this file is the regression for: `train.py::_explicit` parsed
`--preflight-only` but never read `args.preflight_only` — it called
`orchestrator.run(...)` unconditionally, so

    python train.py --profile full --from C8 --to C8 --preflight-only

was not guaranteed to be read-only. It could reach `C8Adapter.workflow()` and
train a real detector. `_zero_argument` already got this right (it checks
`args.preflight_only` and returns BEFORE calling `run()` at all); this file
proves the explicit path now does too, and does so through the SAME canonical
precondition gate a real run uses — never a second, looser verifier.

Three tiers:

* CLI-boundary wiring (Tier 1) — `train.main()` with `orchestrator.run` mocked,
  proving the ARGUMENT is actually threaded through, which is the exact bug.
* `EngineeringAdapter`-level mechanics (Tier 2) — a minimal adapter whose
  `workflow()` explodes if ever called, proving the C4-C13 base class itself
  never reaches it under `preflight_only`.
* Real C8 end-to-end (Tier 3) — the actual `C8Adapter` over a sandbox with a
  REAL, verifiable C7 `DETECTOR_CONFIG_LOCK` (built by really running C7's
  scientific path, not a hand-written fixture), proving zero training calls,
  zero checkpoint writes, zero state/report mutation, and that a broken lock
  or a missing checkpoint still BLOCKS under preflight exactly as a real run
  would.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from conftest_adapters import make_sandbox, request_for  # noqa: E402
from prism_fas.pipeline.adapters import AdapterRequest, AdapterResult  # noqa: E402
from prism_fas.pipeline.adapters.common import EngineeringAdapter  # noqa: E402
from prism_fas.pipeline.status import DualStatus  # noqa: E402
from test_c7_scientific_path import _approve, _run, scientific  # noqa: E402,F401
from test_c8_scientific_path import with_c7_lock  # noqa: E402,F401

import train  # noqa: E402


# --- Tier 1: the exact reported bug, at the train.py CLI boundary ------------

def test_explicit_preflight_only_threads_the_flag_into_orchestrator_run(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug, stated directly: `_explicit` must pass `preflight_only=True`
    through to `orchestrator.run`, not merely parse and discard the flag."""
    calls: list[dict[str, Any]] = []

    class _FakeProfile:
        name = "full"
        config_path = "configs/profiles/full.yaml"
        profile_identity = "x" * 64
        scientific_eligible = True

    class _FakeResult:
        profile = _FakeProfile()
        run_id = "fake-run"
        phase = "preflight"
        outcomes: list[Any] = []
        written: list[str] = []
        blockers: list[str] = []
        outcome = "PASS"

    def fake_run(**kwargs: Any):
        calls.append(kwargs)
        return _FakeResult()

    monkeypatch.setattr(train, "REPO", REPO)
    monkeypatch.setattr("prism_fas.pipeline.orchestrator.run", fake_run)

    args = train.build_parser().parse_args(
        ["--profile", "full", "--from", "C8", "--to", "C8", "--preflight-only"])
    code = train._explicit(args)

    assert len(calls) == 1, "orchestrator.run must be called exactly once"
    assert calls[0]["preflight_only"] is True, (
        "the exact reported defect: --preflight-only was parsed but never passed "
        "to orchestrator.run")
    assert code == train.EXIT_PASS


def test_removing_preflight_only_passes_false_and_still_executes(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 15: the fix must not disable authorized explicit execution."""
    calls: list[dict[str, Any]] = []

    class _FakeProfile:
        name = "full"
        config_path = "configs/profiles/full.yaml"
        profile_identity = "x" * 64
        scientific_eligible = True

    class _FakeResult:
        profile = _FakeProfile()
        run_id = "fake-run"
        phase = "preflight"
        outcomes: list[Any] = []
        written: list[str] = []
        blockers: list[str] = []
        outcome = "PASS"

    def fake_run(**kwargs: Any):
        calls.append(kwargs)
        return _FakeResult()

    monkeypatch.setattr(train, "REPO", REPO)
    monkeypatch.setattr("prism_fas.pipeline.orchestrator.run", fake_run)

    args = train.build_parser().parse_args(
        ["--profile", "full", "--from", "C8", "--to", "C8"])
    code = train._explicit(args)

    assert len(calls) == 1
    assert calls[0]["preflight_only"] is False
    assert calls[0]["first_stage"] == "C8" and calls[0]["last_stage"] == "C8"
    assert code == train.EXIT_PASS


def test_zero_argument_preflight_only_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_zero_argument` already returned before calling `run()`; this proves the
    fix did not change that (it lives entirely in `_explicit` / the adapter
    base class)."""
    import inspect

    source = inspect.getsource(train._zero_argument)
    assert "if args.preflight_only:" in source
    # The preflight-only return must precede the call to `run(...)`.
    assert source.index("if args.preflight_only:") < source.index("result = run(")


# --- Tier 2: EngineeringAdapter mechanics, isolated from any real stage ------

@dataclass
class _ExplodingWorkflowAdapter(EngineeringAdapter):
    """`workflow()` raises if ever called — the assertion IS the test."""

    stage_id: str = "ZZ"
    substages: tuple[str, ...] = ("ZZ",)
    modes: tuple[str, ...] = ("VERIFY",)
    requires_gpu: bool = False
    blocked_precondition: bool = False

    def semantic_preconditions(self, request: AdapterRequest) -> list[dict[str, Any]]:
        if self.blocked_precondition:
            return [{"name": "planted_precondition", "path": "<test>", "present": False,
                     "blocking": True, "description": "deliberately unsatisfied"}]
        return []

    def workflow(self, request: AdapterRequest, context: Any) -> list[AdapterResult]:
        raise AssertionError("workflow() must never be called under preflight_only")


def _scientific_full_profile(repo: Path) -> Any:
    from prism_fas.pipeline.profiles import load_profile

    return load_profile("full", repo=repo)


def test_preflight_only_never_calls_workflow_when_gate_passes(tmp_path: Path) -> None:
    repo = make_sandbox(tmp_path / "repo")
    request = AdapterRequest(repo=repo, profile=_scientific_full_profile(repo),
                             preflight_only=True)
    results = _ExplodingWorkflowAdapter().run(request)

    assert len(results) == 1
    result = results[0]
    assert result.mode == "PREFLIGHT_ONLY"
    assert result.status == "PASS"
    assert result.status_axes.scientific == "NOT_RUN"
    assert result.detail["preflight_only"] is True


def test_preflight_only_never_calls_workflow_when_gate_blocks(tmp_path: Path) -> None:
    """The gate's own BLOCKED result is returned unchanged — no fabricated
    'preflight blocked' verdict, no second verifier."""
    repo = make_sandbox(tmp_path / "repo")
    adapter = _ExplodingWorkflowAdapter(blocked_precondition=True)
    request = AdapterRequest(repo=repo, profile=_scientific_full_profile(repo),
                             preflight_only=True)
    results = adapter.run(request)

    assert len(results) == 1
    result = results[0]
    assert result.mode == "FULL_PRECONDITION_GATE"
    assert result.status == "BLOCKED"
    assert "planted_precondition" in result.summary
    # The identical result a REAL (non-preflight) scientific run would get.
    real_request = AdapterRequest(repo=repo, profile=_scientific_full_profile(repo),
                                  preflight_only=False)
    real_gate = adapter.full_precondition_gate(real_request)
    assert real_gate is not None
    assert real_gate.summary == result.summary
    assert real_gate.checks == result.checks


def test_preflight_only_false_still_calls_workflow(tmp_path: Path) -> None:
    """Sanity: the normal path is unaffected when the flag is not set."""
    repo = make_sandbox(tmp_path / "repo")
    request = AdapterRequest(repo=repo, profile=_scientific_full_profile(repo),
                             preflight_only=False)
    with pytest.raises(AssertionError, match="workflow.* must never be called"):
        _ExplodingWorkflowAdapter().run(request)


def test_preflight_only_under_a_non_scientific_profile_also_skips_workflow(
        tmp_path: Path) -> None:
    """Smoke/rehearsal never reach `full_precondition_gate`; preflight must
    still stop before `workflow()` rather than exercising a fixture."""
    from prism_fas.pipeline.profiles import load_profile

    repo = make_sandbox(tmp_path / "repo")
    request = AdapterRequest(repo=repo, profile=load_profile("smoke", repo=repo),
                             preflight_only=True)
    results = _ExplodingWorkflowAdapter().run(request)

    assert results[0].mode == "PREFLIGHT_ONLY"
    assert results[0].status == "PASS"


# --- Tier 3: the real C8 adapter, a real frozen C7 lock, no mocked verifier --

def _mark_data_packages_present(repo: Path) -> None:
    """C8's `required_inputs` presence-checks the CANONICAL roots
    (`sources.SOURCE_PACKAGE_ROOT`, `sources.WEIGHT_ROOT`) by existence only;
    the sandbox never copies real package bytes (they are large), so a
    genuinely-satisfied preflight test creates just the directories. See
    `tests/pipeline/test_c8_precondition_root_drift.py` for the regression
    that closes the root-drift defect these two paths used to have
    (`data/packages` and `data/packages/pretrained` — neither is a real
    scientific root)."""
    from prism_fas.pipeline.adapters import sources

    (repo / sources.SOURCE_PACKAGE_ROOT).mkdir(parents=True, exist_ok=True)
    (repo / sources.WEIGHT_ROOT).mkdir(parents=True, exist_ok=True)


def _assert_nothing_scientific_was_written(repo: Path) -> None:
    assert not (repo / "runs" / "full" / "c8").exists(), "a C8 row directory was created"
    assert not (repo / "reports" / "full" / "c8").exists(), "a C8 report was written"
    assert not (repo / "state" / "PIPELINE_STATE.json").exists(), (
        "state/PIPELINE_STATE.json was written by a preflight-only run")
    assert not (repo / "state" / "MASTER_RUN_INDEX.json").exists(), (
        "state/MASTER_RUN_INDEX.json was written by a preflight-only run")


def test_c8_preflight_only_passes_without_training_when_the_lock_is_valid(
        with_c7_lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prism_fas.detector import trainer as trainer_module
    from prism_fas.pipeline.orchestrator import run

    calls = {"trainer_init": 0, "flow": 0, "optimizer_steps": 0, "checkpoint_writes": 0}

    class _ExplodingTrainer:
        def __init__(self, **_kwargs: Any) -> None:
            calls["trainer_init"] += 1
            raise AssertionError("M9Trainer must never be instantiated under preflight-only")

    def _exploding_flow(_trainer: Any, **_kwargs: Any) -> Any:
        calls["flow"] += 1
        raise AssertionError("run_source_only_flow must never be called under preflight-only")

    monkeypatch.setattr(trainer_module, "M9Trainer", _ExplodingTrainer)
    monkeypatch.setattr(trainer_module, "run_source_only_flow", _exploding_flow)
    _mark_data_packages_present(with_c7_lock)
    # This laptop has no CUDA device; stub the hardware probe so the PASS
    # verdict this test proves is about the C6/C7 evidence checks, not about
    # whether a GPU happens to be plugged into the test runner. The probe
    # itself (`_accelerator_available`) is exercised by its own tests.
    monkeypatch.setattr("prism_fas.pipeline.adapters.common._accelerator_available",
                        lambda: (True, {"device": "stub-gpu-for-this-test"}))

    result = run(repo=with_c7_lock, profile_name="full", first_stage="C8", last_stage="C8",
                 preflight_only=True)

    assert calls == {"trainer_init": 0, "flow": 0, "optimizer_steps": 0,
                     "checkpoint_writes": 0}
    assert result.outcome == "PASS", [
        (o.stage.stage_id, ar.mode, ar.status, ar.summary,
         [c["check_id"] for c in ar.checks if not c["ok"]])
        for o in result.outcomes for ar in o.adapter_results]
    c8_outcome = next(o for o in result.outcomes if o.stage.stage_id == "C8")
    assert c8_outcome.adapter_results[0].mode == "PREFLIGHT_ONLY"
    # The C6 evidence and the C7 lock were genuinely re-verified, not skipped.
    checks = c8_outcome.adapter_results[0].checks
    verified = {c["check_id"] for c in checks}
    assert "c8_input_c6_matched_banks" in verified
    assert "c8_input_c7_config_lock" in verified
    assert all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]
    _assert_nothing_scientific_was_written(with_c7_lock)


def test_c8_preflight_only_blocks_on_a_broken_c7_lock(
        with_c7_lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 11: an intentionally broken C7 lock BLOCKS preflight,
    through C8's OWN semantic_preconditions -> verify_detector_config_lock —
    the same verifier a real run would use, not a looser stand-in."""
    from prism_fas.detector import trainer as trainer_module
    from prism_fas.pipeline.orchestrator import run

    def _explodes(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("nothing scientific may run once the lock is broken")

    monkeypatch.setattr(trainer_module, "M9Trainer", _explodes)
    monkeypatch.setattr(trainer_module, "run_source_only_flow", _explodes)
    _mark_data_packages_present(with_c7_lock)

    from prism_fas.pipeline.adapters import c7

    path = with_c7_lock / c7.SCIENTIFIC_CONFIG_LOCK_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tracks"]["R"]["winner_checkpoint_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = run(repo=with_c7_lock, profile_name="full", first_stage="C8", last_stage="C8",
                 preflight_only=True)

    assert result.outcome == "BLOCKED"
    c8_outcome = next(o for o in result.outcomes if o.stage.stage_id == "C8")
    gate_result = c8_outcome.adapter_results[0]
    assert gate_result.mode == "FULL_PRECONDITION_GATE"
    assert gate_result.status == "BLOCKED"
    assert "c7_config_lock" in gate_result.summary
    _assert_nothing_scientific_was_written(with_c7_lock)


def test_c8_preflight_only_blocks_on_a_missing_winner_checkpoint(
        with_c7_lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 12: canonical C8 preconditions demand the winner checkpoint
    exist and hash to what the lock recorded; deleting it must BLOCK."""
    from prism_fas.detector import trainer as trainer_module
    from prism_fas.pipeline.orchestrator import run

    def _explodes(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("nothing scientific may run once the checkpoint is gone")

    monkeypatch.setattr(trainer_module, "M9Trainer", _explodes)
    monkeypatch.setattr(trainer_module, "run_source_only_flow", _explodes)
    _mark_data_packages_present(with_c7_lock)

    from prism_fas.pipeline.adapters import c7

    path = with_c7_lock / c7.SCIENTIFIC_CONFIG_LOCK_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = with_c7_lock / payload["tracks"]["G"]["winner_checkpoint"]
    assert checkpoint.is_file()
    checkpoint.unlink()

    result = run(repo=with_c7_lock, profile_name="full", first_stage="C8", last_stage="C8",
                 preflight_only=True)

    assert result.outcome == "BLOCKED"
    _assert_nothing_scientific_was_written(with_c7_lock)


def test_c8_preflight_only_resolves_zero_target_paths_and_labels(
        with_c7_lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No row manifest (the only place C8 would ever record a target count)
    exists, and the preflight route stayed entirely inside the two modes that
    cannot resolve a target — never `_run_scientific_row`."""
    from prism_fas.detector import trainer as trainer_module
    from prism_fas.pipeline.orchestrator import run

    monkeypatch.setattr(trainer_module, "M9Trainer",
                        lambda **_k: (_ for _ in ()).throw(
                            AssertionError("no training under preflight")))
    _mark_data_packages_present(with_c7_lock)

    result = run(repo=with_c7_lock, profile_name="full", first_stage="C8", last_stage="C8",
                 preflight_only=True)

    for outcome in result.outcomes:
        for adapter_result in outcome.adapter_results:
            assert adapter_result.mode in ("PREFLIGHT_ONLY", "FULL_PRECONDITION_GATE")
    assert not list((with_c7_lock / "runs" / "full" / "c8").rglob("run_manifest.json"))


def test_c8_preflight_only_c7_lock_bytes_are_unchanged(with_c7_lock: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 7/8: the C7 lock is read/verified, never rewritten, by a
    C8 preflight-only pass."""
    from prism_fas.detector import trainer as trainer_module
    from prism_fas.pipeline.orchestrator import run
    from prism_fas.pipeline.adapters import c7

    monkeypatch.setattr(trainer_module, "M9Trainer",
                        lambda **_k: (_ for _ in ()).throw(
                            AssertionError("no training under preflight")))
    _mark_data_packages_present(with_c7_lock)

    lock_path = with_c7_lock / c7.SCIENTIFIC_CONFIG_LOCK_PATH
    before = lock_path.read_bytes()
    bank_paths = sorted((with_c7_lock / "reports/full/c6").glob("*.json"))
    before_banks = {path: path.read_bytes() for path in bank_paths}

    run(repo=with_c7_lock, profile_name="full", first_stage="C8", last_stage="C8",
       preflight_only=True)

    assert lock_path.read_bytes() == before, "C8 preflight rewrote the C7 lock"
    for path, content in before_banks.items():
        assert path.read_bytes() == content, f"C8 preflight rewrote {path}"
