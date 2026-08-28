"""Regression test for a NameError in `prism_fas.cli.m10_target.extract()`.

A real GPU run of `prism m10 target-package extract` failed before detector
construction with `NameError: name 'resolve_detector_path' is not defined`:
the function's local import pulled in `SCRFDDetector, load_m2_config` but
called `resolve_detector_path` (used both to build the detector and to bind
`detector_model_path`/`detector_model_sha256` into the run context) without
ever importing it.

This test proves `extract()` reaches and executes the
`resolve_detector_path(cfg.scrfd_model_path)` call without raising
`NameError`. It never runs real SCRFD (the detector class is replaced with a
fake) and never runs real target preprocessing (the pending-record set is
forced empty, so the processing loop that would call `run_preprocessing`
never executes).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from prism_fas.cli import m10_target


def _fake_record(video_id: str = "vid-0") -> SimpleNamespace:
    return SimpleNamespace(video_id=video_id)


def test_extract_resolves_detector_path_without_nameerror(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {"detector_constructed": 0}

    class _FakeDetector:
        def __init__(self, model_path, input_size, provider):
            calls["detector_constructed"] = calls["detector_constructed"] + 1
            calls["model_path"] = model_path
            calls["input_size"] = input_size
            calls["provider"] = provider

    fake_paths = SimpleNamespace(
        raw_datasets=SimpleNamespace(siw_mv2=tmp_path / "raw"),
        project_root=tmp_path, work_root=tmp_path)

    monkeypatch.setattr(m10_target, "load_paths", lambda config: fake_paths)
    monkeypatch.setattr(m10_target, "_layout",
                        lambda: SimpleNamespace(dataset_definition=lambda: object()))
    monkeypatch.setattr(m10_target, "_audit",
                        lambda paths, layout: SimpleNamespace(passed=True, report=lambda: {}))
    # One record but zero selected: len(selected) != len(records), so the
    # --confirm-full-run guard does not trigger; pending stays empty, so the
    # real-processing loop (run_preprocessing) never runs.
    monkeypatch.setattr(m10_target, "_select", lambda records, audit, **kwargs: [])
    monkeypatch.setattr(m10_target, "_profile_root",
                        lambda paths, cfg: (SimpleNamespace(output_namespace="target_eval_v2"), tmp_path))

    import prism_fas.data.adapters as adapters_module
    monkeypatch.setattr(adapters_module, "adapter_for",
                        lambda dataset_definition, root: SimpleNamespace(
                            inference_records=lambda: [_fake_record()]))

    import prism_fas.data.preprocess_m2 as preprocess_m2_module
    monkeypatch.setattr(preprocess_m2_module, "SCRFDDetector", _FakeDetector)

    m10_target.extract(
        config=Path("configs/paths.local.yaml"),
        preprocess_config=Path("configs/data/preprocess_m2.yaml"),
        confirm=False, limit_records=None, families=None, include_live=0,
        chunk=25, resume=True, dry_run=False)

    assert calls["detector_constructed"] == 1
    assert calls["model_path"] is not None


def test_extract_source_imports_resolve_detector_path_in_the_same_scope_it_is_used() -> None:
    """A cheaper, purely static companion check: the local import inside
    `extract()` must name `resolve_detector_path` alongside `SCRFDDetector`
    and `load_m2_config` — this is exactly the line that was missing it."""
    import inspect

    source = inspect.getsource(m10_target.extract)
    assert "from prism_fas.data.preprocess_m2 import" in source
    import_line = next(line for line in source.splitlines()
                       if "from prism_fas.data.preprocess_m2 import" in line)
    assert "resolve_detector_path" in import_line
    assert "SCRFDDetector" in import_line
    assert "load_m2_config" in import_line
