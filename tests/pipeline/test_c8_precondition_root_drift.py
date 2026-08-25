"""C8's precondition roots must be the SAME roots C7 scientifically trained
against — not a second, independently-spelled set that can (and did) drift.

The defect: `C8Adapter.required_inputs()` declared

    pretrained_weights  ->  data/packages/pretrained   (nothing ever writes this)
    source_packages     ->  data/packages               (the PARENT of the real
                                                           package, not the package)

reproduced live on the GPU host: the corrected, genuinely read-only
`--preflight-only` (see `tests/pipeline/test_explicit_preflight_only.py`)
correctly ran no workflow and wrote nothing, and BLOCKED — truthfully, but on
an invented path. C7's own `required_inputs()` already used the correct
roots (`weights`, `data/packages/prism_data_v1_m3b`) and C7 scientifically
completed against exactly those bytes.

The fix has two parts, both tested here:

1. `required_inputs()` for both C7 and C8 now import the canonical root
   constants (`sources.WEIGHT_ROOT`, `sources.SOURCE_PACKAGE_ROOT`,
   `c6_evidence.C6_REPORTS`, `c7.SCIENTIFIC_CONFIG_LOCK_PATH`) instead of
   re-spelling them — a fast, presence-only first layer that cannot drift
   from the module that owns each root.
2. `C8Adapter.semantic_preconditions()` now ALSO calls
   `sources.verify_detector_inputs` — the SAME canonical, SHA-verifying check
   (pinned SigLIP2/ConvNeXt by content hash, the frozen recipe text cache by
   file SHA-256 and re-derived semantic identity, the validated M3B package,
   the frozen M7 recipe bank, the C5 candidate tree) C7's own
   `_scientific_prepare` already required before its first trial. A directory
   merely existing was never equivalent to this, and nothing here reimplements
   any part of it a second, looser way.

Three tiers:

* Part 1 — path/import audit, no fixture needed.
* Part 2 — the real SHA-verification mechanism in `detector.pretrained`,
  exercised with small fixture weight bytes and a monkeypatched pin (the
  actual multi-GB pinned files cannot live in a test), proving "missing",
  "corrupt" and "correct" each do what they should — a gap no existing test
  closed for the pretrained weights (the recipe text cache equivalent already
  exists in `test_c7_global_input_preflight.py`).
* Part 3 — C8's own gate, end to end, reacting to the canonical verifier.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from conftest_adapters import make_sandbox, request_for  # noqa: E402
from prism_fas.pipeline.adapters import c7, c8  # noqa: E402
from test_c7_scientific_path import _approve, _run, scientific  # noqa: E402,F401
from test_c8_scientific_path import with_c7_lock  # noqa: E402,F401


# ==============================================================================
# Part 1 — path/import audit (no C6/C7 fixture needed)
# ==============================================================================

def test_c8_no_longer_requires_data_packages_pretrained() -> None:
    paths = {ri.name: ri.relative_path for ri in c8.C8Adapter().required_inputs()}
    assert paths["pretrained_weights"] != "data/packages/pretrained"
    assert "data/packages/pretrained" not in paths.values()


def test_c8_resolves_the_canonical_weight_root_used_by_detector_science() -> None:
    from prism_fas.pipeline.adapters import sources

    paths = {ri.name: ri.relative_path for ri in c8.C8Adapter().required_inputs()}
    assert paths["pretrained_weights"] == sources.WEIGHT_ROOT == "weights"


def test_c7_and_c8_cannot_drift_to_different_pretrained_roots() -> None:
    """Not just equal strings by coincidence: both `required_inputs()`
    functions read the constant off `sources`, verified from source."""
    import inspect

    from prism_fas.pipeline.adapters import sources

    c7_paths = {ri.name: ri.relative_path for ri in c7.C7Adapter().required_inputs()}
    c8_paths = {ri.name: ri.relative_path for ri in c8.C8Adapter().required_inputs()}
    assert c7_paths["pretrained_weights"] == c8_paths["pretrained_weights"] == \
        sources.WEIGHT_ROOT
    assert c7_paths["source_package"] == c8_paths["source_package"] == \
        sources.SOURCE_PACKAGE_ROOT

    c7_source = inspect.getsource(c7.C7Adapter.required_inputs)
    c8_source = inspect.getsource(c8.C8Adapter.required_inputs)
    for source in (c7_source, c8_source):
        assert "sources.WEIGHT_ROOT" in source
        assert "sources.SOURCE_PACKAGE_ROOT" in source
        # Neither hand-spells the roots any more; a future edit that
        # reintroduces a literal is the drift this test exists to catch.
        assert '"weights"' not in source
        assert '"data/packages' not in source


def test_c8_source_package_check_resolves_the_canonical_m3b_root(tmp_path: Path) -> None:
    """Not merely accepting an unrelated parent directory. The old defect's
    twin: `data/packages` existing must not satisfy the check — only the
    actual M3B package root may."""
    from prism_fas.pipeline.adapters import sources

    repo = make_sandbox(tmp_path / "repo")
    source_package_input = next(ri for ri in c8.C8Adapter().required_inputs()
                                if ri.name == "source_package")
    assert source_package_input.relative_path == sources.SOURCE_PACKAGE_ROOT
    assert source_package_input.relative_path == "data/packages/prism_data_v1_m3b"

    # The parent existing alone must still be reported absent.
    (repo / "data" / "packages").mkdir(parents=True)
    assert source_package_input.resolve(repo)["present"] is False

    # Only the actual package root satisfies it.
    (repo / sources.SOURCE_PACKAGE_ROOT).mkdir(parents=True)
    assert source_package_input.resolve(repo)["present"] is True


# ==============================================================================
# Part 2 — the real SHA-verification mechanism (detector.pretrained), with
# small fixture bytes under a monkeypatched pin. No multi-GB file is faked;
# the pin itself is temporarily pointed at what the test actually wrote, so
# the SAME comparison code (`sha256_file(path) != pinned`) runs for real.
# ==============================================================================

def _write_fixture_pins(weight_root: Path, monkeypatch: pytest.MonkeyPatch,
                        *, corrupt: str | None = None) -> None:
    """A small, internally-consistent SigLIP2 + ConvNeXt pin, written to disk
    and installed as the CURRENT pin so `SigLIP2Artifacts.resolve` /
    `resolve_convnext_weight` verify real bytes against a real expected hash.

    `corrupt`, when given a required SigLIP2 filename or `"convnext"`,
    overwrites that one file with different bytes AFTER the pin is computed,
    so the pin still expects the original hash and the file no longer matches.
    """
    from prism_fas.detector import pretrained as pretrained_module

    siglip_dir = weight_root / "pretrained" / "m9" / "siglip2"
    siglip_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name in pretrained_module.SIGLIP2_PIN["required_files"]:
        content = f"fixture-bytes-for-{name}".encode()
        (siglip_dir / name).write_bytes(content)
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}

    convnext_dir = weight_root / "pretrained" / "m9" / "convnextv2_atto"
    convnext_dir.mkdir(parents=True, exist_ok=True)
    convnext_content = b"fixture-convnext-weight-bytes"
    (convnext_dir / "model.safetensors").write_bytes(convnext_content)
    convnext_sha = hashlib.sha256(convnext_content).hexdigest()

    fixture_siglip_pin = {**pretrained_module.SIGLIP2_PIN, "files": files}
    fixture_convnext_pin = {**pretrained_module.CONVNEXT_PIN, "weight_sha256": convnext_sha}
    monkeypatch.setattr(pretrained_module, "SIGLIP2_PIN", fixture_siglip_pin)
    monkeypatch.setattr(pretrained_module, "CONVNEXT_PIN", fixture_convnext_pin)

    if corrupt == "convnext":
        (convnext_dir / "model.safetensors").write_bytes(b"tampered convnext bytes")
    elif corrupt is not None:
        (siglip_dir / corrupt).write_bytes(b"tampered siglip bytes")


def test_missing_canonical_weights_blocks(tmp_path: Path) -> None:
    from prism_fas.pipeline.adapters import sources

    repo = tmp_path / "repo"
    (repo / sources.WEIGHT_ROOT).mkdir(parents=True)

    with pytest.raises(sources.DetectorInputsUnavailable, match="pinned SigLIP2 not found"):
        sources._pinned_detector_weights(repo)


def test_corrupt_pinned_weights_blocks_through_the_canonical_verifier(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prism_fas.pipeline.adapters import sources

    repo = tmp_path / "repo"
    _write_fixture_pins(repo / sources.WEIGHT_ROOT, monkeypatch, corrupt="config.json")

    with pytest.raises(sources.DetectorInputsUnavailable, match="SHA .* != pinned"):
        sources._pinned_detector_weights(repo)


def test_corrupt_convnext_weight_blocks_through_the_canonical_verifier(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prism_fas.pipeline.adapters import sources

    repo = tmp_path / "repo"
    _write_fixture_pins(repo / sources.WEIGHT_ROOT, monkeypatch, corrupt="convnext")

    with pytest.raises(sources.DetectorInputsUnavailable, match="ConvNeXt weight SHA"):
        sources._pinned_detector_weights(repo)


def test_correct_canonical_weights_pass_the_pretrained_check(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prism_fas.pipeline.adapters import sources

    repo = tmp_path / "repo"
    _write_fixture_pins(repo / sources.WEIGHT_ROOT, monkeypatch)

    resolved = sources._pinned_detector_weights(repo)
    assert resolved["stub_substituted"] is False
    assert resolved["downloaded_during_run"] is False
    assert resolved["global_tower"]["identity_sha256"]
    assert resolved["local_backbone"]["weight_sha256"]


# ==============================================================================
# Part 3 — C8's own gate, end to end. `scientific` / `with_c7_lock` stub
# `sources.verify_detector_inputs` to the shape it already had before this
# task (its own SHA mechanics are Part 2 above and `test_c7_global_input_
# preflight.py`); what these tests prove is that C8's GATE now calls it and
# reacts correctly to both a failure and a success, through the real
# `orchestrator.run(..., preflight_only=True)` path.
# ==============================================================================

def _stub_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-op: `with_c7_lock` already installs a passing stub. Named for
    readability at each call site."""


def test_c8_semantic_preconditions_calls_the_canonical_verifier(with_c7_lock: Path) -> None:
    """Structural: the call exists, and there is no second implementation of
    what it checks."""
    import inspect

    body = inspect.getsource(c8.C8Adapter.semantic_preconditions)
    assert "sources.verify_detector_inputs(request.repo)" in body
    assert "c8_scientific_inputs_verified" in body

    request = request_for(with_c7_lock, "full")
    rows = {row["name"]: row for row in c8.C8Adapter().semantic_preconditions(request)}
    assert "c8_scientific_inputs_verified" in rows
    assert rows["c8_scientific_inputs_verified"]["present"] is True
    assert rows["c8_scientific_inputs_verified"]["blocking"] is False
    assert rows["c8_scientific_inputs_verified"]["verifier"] == (
        "prism_fas.pipeline.adapters.sources.verify_detector_inputs")


def test_c8_preflight_blocks_when_the_canonical_verifier_fails(
        with_c7_lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact live-GPU scenario, reproduced: `verify_detector_inputs`
    finds a real problem (here: simulated), and preflight BLOCKS on it —
    never on an invented duplicate path."""
    from prism_fas.pipeline.adapters import sources
    from prism_fas.pipeline.orchestrator import run

    def _fails(_repo: Path, **_kwargs: Any) -> Any:
        raise sources.DetectorInputsUnavailable(
            "the pinned detector backbones are not resolvable under weights: "
            "planted failure")

    monkeypatch.setattr(sources, "verify_detector_inputs", _fails)
    monkeypatch.setattr("prism_fas.pipeline.adapters.common._accelerator_available",
                        lambda: (True, {"device": "stub-gpu-for-this-test"}))

    result = run(repo=with_c7_lock, profile_name="full", first_stage="C8", last_stage="C8",
                preflight_only=True)

    assert result.outcome == "BLOCKED"
    c8_outcome = next(o for o in result.outcomes if o.stage.stage_id == "C8")
    gate_result = c8_outcome.adapter_results[0]
    assert gate_result.mode == "FULL_PRECONDITION_GATE"
    problem = next(c for c in gate_result.checks
                   if c["check_id"] == "c8_input_c8_scientific_inputs_verified")
    assert problem["ok"] is False
    assert "planted failure" in json.dumps(problem)


def test_c8_preflight_passes_when_the_canonical_verifier_succeeds(
        with_c7_lock: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prism_fas.detector import trainer as trainer_module
    from prism_fas.pipeline.orchestrator import run

    monkeypatch.setattr(trainer_module, "M9Trainer",
                        lambda **_k: (_ for _ in ()).throw(
                            AssertionError("no training under preflight")))
    monkeypatch.setattr("prism_fas.pipeline.adapters.common._accelerator_available",
                        lambda: (True, {"device": "stub-gpu-for-this-test"}))
    # The canonical roots, present — this is what the fix means by "correct":
    # required_inputs' fast presence layer must also agree.
    from prism_fas.pipeline.adapters import sources
    (with_c7_lock / sources.SOURCE_PACKAGE_ROOT).mkdir(parents=True, exist_ok=True)
    (with_c7_lock / sources.WEIGHT_ROOT).mkdir(parents=True, exist_ok=True)

    result = run(repo=with_c7_lock, profile_name="full", first_stage="C8", last_stage="C8",
                preflight_only=True)

    assert result.outcome == "PASS", [
        (o.stage.stage_id, ar.mode, ar.status, ar.summary,
         [c["check_id"] for c in ar.checks if not c["ok"]])
        for o in result.outcomes for ar in o.adapter_results]


def test_c8_preflight_blocks_on_invalid_c6_closure(with_c7_lock: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from prism_fas.detector import trainer as trainer_module
    from prism_fas.pipeline.orchestrator import run
    from prism_fas.pipeline.adapters import sources

    def _explodes(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("nothing scientific may run once C6 is invalid")

    monkeypatch.setattr(trainer_module, "M9Trainer", _explodes)
    # This test isolates the pre-existing `c6_closure_verified` check (via the
    # non-raising `c6_evidence.evidence_report`) from the NEW
    # `c8_scientific_inputs_verified` check added by this fix: the fixture's
    # own `verify_detector_inputs` stub calls the REAL, raising
    # `verify_c6_evidence` internally and would abort the test on the same
    # corruption for an unrelated reason, so it is held passing here.
    monkeypatch.setattr(sources, "verify_detector_inputs",
                        lambda _repo, **_k: {"pretrained": {}, "recipe_text_cache": {}})
    (with_c7_lock / sources.SOURCE_PACKAGE_ROOT).mkdir(parents=True, exist_ok=True)
    (with_c7_lock / sources.WEIGHT_ROOT).mkdir(parents=True, exist_ok=True)

    bank_lock = with_c7_lock / "reports/full/c6/C6_BANK_LOCK_RND.json"
    payload = json.loads(bank_lock.read_text(encoding="utf-8"))
    payload["provenance_closure"] = {"closed": False, "unaccounted": ["x"]}
    bank_lock.write_text(json.dumps(payload), encoding="utf-8")

    result = run(repo=with_c7_lock, profile_name="full", first_stage="C8", last_stage="C8",
                preflight_only=True)

    assert result.outcome == "BLOCKED"
    c8_outcome = next(o for o in result.outcomes if o.stage.stage_id == "C8")
    checks = c8_outcome.adapter_results[0].checks
    assert not next(c for c in checks
                    if c["check_id"] == "c8_input_c6_closure_verified")["ok"]


def test_the_normal_non_preflight_c8_path_remains_reachable(with_c7_lock: Path,
                                                             monkeypatch: pytest.MonkeyPatch
                                                             ) -> None:
    """Once every real precondition is satisfied, `full_precondition_gate`
    returns None — the signal `EngineeringAdapter.run()` uses to proceed into
    `workflow()`. This is the requirement that the fix must not, in the
    process of correcting the paths, make C8 permanently unreachable."""
    from prism_fas.pipeline.adapters import sources

    monkeypatch.setattr("prism_fas.pipeline.adapters.common._accelerator_available",
                        lambda: (True, {"device": "stub-gpu-for-this-test"}))
    (with_c7_lock / sources.SOURCE_PACKAGE_ROOT).mkdir(parents=True, exist_ok=True)
    (with_c7_lock / sources.WEIGHT_ROOT).mkdir(parents=True, exist_ok=True)

    request = request_for(with_c7_lock, "full")
    gate = c8.C8Adapter().full_precondition_gate(request)

    assert gate is None, "a fully-satisfied gate must not block the real run"
