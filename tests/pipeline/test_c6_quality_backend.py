"""The C6 quality-backend construction boundary, executed rather than inspected.

`_fit_nominal_calibration` called `QualityBackends.resolve(...)`, which does not
exist — `QualityBackends` is constructed, and the class with a `resolve`
classmethod is `QualityModelRegistry`. The GPU run reached C6, blocked at
CALIBRATION_UNAVAILABLE, and printed only "the NOMINAL calibration could not be
fitted"; the `AttributeError` sat inside a check detail nobody saw.

The existing C6 tests proved the wiring by reading source text, so a call to a
method that was never defined looked exactly like a call to one that was. These
tests run the production method and intercept the real constructor boundary, so
an API that does not exist fails here instead of on the GPU host.

Nothing is calibrated. The models are never loaded: the constructor is replaced
AT the boundary under test, which is the point after which the work is expensive
and before which the wiring is what matters.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.pipeline.adapters import AdapterRequest  # noqa: E402
from prism_fas.pipeline.adapters import c6 as c6_module  # noqa: E402
from prism_fas.pipeline.adapters.c6 import C6Adapter  # noqa: E402
from prism_fas.pipeline.profiles import load_profile  # noqa: E402
from prism_fas.synthesis import quality_calibration  # noqa: E402


# --- 1. the API that does not exist ------------------------------------------

def test_quality_backends_has_no_resolve_classmethod() -> None:
    """The standing API-drift guard, stated as the fact it is."""
    assert not hasattr(quality_calibration.QualityBackends, "resolve")
    assert {"detect", "embed", "parse", "manifest"} <= {
        name for name in dir(quality_calibration.QualityBackends)
        if not name.startswith("_")}


def test_the_constructor_is_the_canonical_api() -> None:
    signature = inspect.signature(quality_calibration.QualityBackends.__init__)

    assert list(signature.parameters) == ["self", "weight_root", "device"]
    assert signature.parameters["device"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_class_that_does_have_resolve_is_the_registry() -> None:
    """Where the confusion came from: one of the two classes really does."""
    from prism_fas.synthesis.quality_models import QualityModelRegistry

    assert hasattr(QualityModelRegistry, "resolve")
    # ...and `QualityBackends.__init__` is what calls it.
    source = inspect.getsource(quality_calibration.QualityBackends.__init__)
    assert "QualityModelRegistry.resolve(" in source


def test_no_adapter_calls_the_nonexistent_backend_resolve() -> None:
    for stage in ("c5", "c6"):
        source = (REPO / "src" / "prism_fas" / "pipeline" / "adapters"
                  / f"{stage}.py").read_text(encoding="utf-8")
        assert "QualityBackends.resolve" not in source, stage


# --- the production call ------------------------------------------------------

def _request(root: Path) -> AdapterRequest:
    return AdapterRequest(repo=root, profile=load_profile("full", repo=REPO))


def _state(root: Path) -> dict[str, Any]:
    return {"package_root": root / "package", "package_identity": "b" * 64}


class _Recorder:
    """Stands in for `QualityBackends` AT the constructor boundary under test."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.device = "recorded"

    def __call__(self, weight_root: Any, **kwargs: Any) -> "_Recorder":
        self.calls.append((weight_root, dict(kwargs)))
        return self


def _run(root: Path, monkeypatch, *, device: str | None = "cpu",
         binding_ok: bool = True, backend=None, fit=None) -> Any:
    """Drive `_fit_nominal_calibration` with the expensive parts replaced."""
    reports = root / "reports" / "full" / "c6"
    reports.mkdir(parents=True, exist_ok=True)
    (root / "configs" / "synthesis").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(c6_module, "FROZEN_QUALITY_BACKEND_DEVICE", device)
    monkeypatch.setattr(c6_module, "_verify_quality_model_binding",
                        lambda weight_root: {"ok": binding_ok,
                                             "roles": ["identity", "parsing", "detector"],
                                             "error_type": None if binding_ok else "FileNotFoundError",
                                             "error": "" if binding_ok else "detector weight absent"})
    monkeypatch.setattr(quality_calibration, "load_quality_config",
                        lambda path: {"benign": {"gaussian_noise_std": 0.001}})
    if backend is not None:
        monkeypatch.setattr(quality_calibration, "QualityBackends", backend)
    science = c6_module.science_module()
    if fit is not None:
        monkeypatch.setattr(science, "fit_nominal_calibration", fit)

    state = _state(root)
    result = C6Adapter()._fit_nominal_calibration(_request(root), state, reports)
    return result, state, reports


# --- 2, 3, 4, 5. the happy path reaches the real boundary --------------------

def test_the_production_method_constructs_the_backend_with_the_weight_root(
        tmp_path: Path, monkeypatch) -> None:
    recorder = _Recorder()
    fitted = {"thresholds": {"tau_fd": 0.5}, "device": "cpu"}

    result, state, _ = _run(tmp_path, monkeypatch, backend=recorder,
                            fit=lambda package, config, backends: fitted)

    assert recorder.calls, "the production method really did construct the backend"
    weight_root, kwargs = recorder.calls[0]
    assert Path(weight_root) == tmp_path / "weights"
    assert list(kwargs) == ["device"], "the device is passed explicitly"
    assert kwargs["device"] == "cpu"
    assert result.status_axes.engineering != "BLOCKED"
    assert state["backends"] is recorder


def test_the_constructed_backend_is_what_the_calibrator_receives(
        tmp_path: Path, monkeypatch) -> None:
    recorder = _Recorder()
    seen: dict[str, Any] = {}

    def fit(package_root, config, backends):
        seen.update({"package_root": package_root, "backends": backends,
                     "config": config})
        return {"thresholds": {"tau_fd": 0.5}}

    _run(tmp_path, monkeypatch, backend=recorder, fit=fit)

    assert seen["backends"] is recorder, "the same object, not a second one"
    assert seen["package_root"] == tmp_path / "package"
    assert "benign" in seen["config"]


def test_a_successful_fit_writes_the_calibration_as_a_c6_output(
        tmp_path: Path, monkeypatch) -> None:
    fitted = {"thresholds": {"tau_fd": 0.5, "tau_lm": 0.2}, "device": "cpu"}

    result, state, reports = _run(tmp_path, monkeypatch, backend=_Recorder(),
                                  fit=lambda package, config, backends: fitted)
    payload = json.loads((reports / "QUALITY_CALIBRATION.json")
                         .read_text(encoding="utf-8"))

    assert payload["thresholds"] == fitted["thresholds"]
    assert payload["split"] == "source_train"
    assert payload["is_scientific_lock"] is True
    assert state["calibration_path"] == reports / "QUALITY_CALIBRATION.json"
    assert any(item["check_id"] == "c6_calibration_is_an_output_not_an_input"
               and item["ok"] for item in result.checks)


# --- 6, 7. fail closed --------------------------------------------------------

def test_a_constructor_failure_blocks_and_names_the_exception(
        tmp_path: Path, monkeypatch) -> None:
    """The shape of the original defect: the boundary raises."""
    def exploding(weight_root, **kwargs):
        raise AttributeError("type object 'QualityBackends' has no attribute 'resolve'")

    result, _, reports = _run(tmp_path, monkeypatch, backend=exploding)

    assert result.status == "BLOCKED"
    assert result.mode == "CALIBRATION_UNAVAILABLE"
    assert "AttributeError" in result.summary, (
        "the console line names the exception now, not just 'could not be fitted'")
    failure = json.loads((reports / "C6_CALIBRATION_FAILURE.json")
                         .read_text(encoding="utf-8"))
    assert failure["error_type"] == "AttributeError"
    assert failure["substage"] == "FIT_NOMINAL_CALIBRATION"
    assert "no attribute 'resolve'" in failure["sanitized_reason"]


def test_a_calibration_failure_blocks_and_names_the_exception(
        tmp_path: Path, monkeypatch) -> None:
    def exploding(package_root, config, backends):
        raise RuntimeError("source_train must supply both live and spoof populations")

    result, _, reports = _run(tmp_path, monkeypatch, backend=_Recorder(),
                              fit=exploding)
    failure = json.loads((reports / "C6_CALIBRATION_FAILURE.json")
                         .read_text(encoding="utf-8"))

    assert result.status == "BLOCKED"
    assert failure["error_type"] == "RuntimeError"
    assert "live and spoof populations" in failure["sanitized_reason"]


def test_a_missing_pinned_model_blocks_before_anything_is_measured(
        tmp_path: Path, monkeypatch) -> None:
    recorder = _Recorder()

    result, _, reports = _run(tmp_path, monkeypatch, binding_ok=False,
                              backend=recorder)

    assert result.status == "BLOCKED"
    assert result.mode == "QUALITY_MODEL_BINDING_INVALID"
    assert recorder.calls == [], "no backend was constructed"
    binding = next(item for item in result.checks
                   if item["check_id"] == "c6_quality_models_bind")
    assert binding["ok"] is False
    assert binding["detail"]["roles"] == ["identity", "parsing", "detector"]


def test_the_binding_check_uses_the_canonical_registry() -> None:
    source = inspect.getsource(c6_module._verify_quality_model_binding)

    assert "QualityModelRegistry.resolve(" in source
    assert 'roles=roles' in source
    for forbidden in ("download", "urlretrieve", "fallback", "substitute"):
        assert forbidden not in source, forbidden


def test_a_host_path_never_reaches_the_failure_artifact(
        tmp_path: Path, monkeypatch) -> None:
    def exploding(weight_root, **kwargs):
        raise FileNotFoundError(r"missing D:\weights\adaface\model.ckpt")

    _, _, reports = _run(tmp_path, monkeypatch, backend=exploding)
    failure = json.loads((reports / "C6_CALIBRATION_FAILURE.json")
                         .read_text(encoding="utf-8"))

    assert "[redacted-path]" in failure["sanitized_reason"]
    assert "D:\\weights" not in failure["sanitized_reason"]


# --- 3. the device is frozen or the stage refuses ----------------------------

def test_an_undetermined_device_blocks_rather_than_guessing(
        tmp_path: Path, monkeypatch) -> None:
    recorder = _Recorder()

    result, _, reports = _run(tmp_path, monkeypatch, device=None, backend=recorder)

    assert result.status == "BLOCKED"
    assert result.mode == "C6_QUALITY_BACKEND_DEVICE_NEEDS_SCIENTIFIC_DECISION"
    assert recorder.calls == [], "no device was picked and no backend was built"
    device_check = next(item for item in result.checks
                        if item["check_id"] == "c6_quality_backend_device_is_frozen")
    assert device_check["ok"] is False
    assert len(device_check["detail"]["audited"]) >= 6


def test_the_device_is_never_defaulted_from_the_constructor_signature() -> None:
    """`device='cpu'` is a Python default, not a scientific contract."""
    source = inspect.getsource(C6Adapter._fit_nominal_calibration)

    assert "_quality_backend_device(request)" in source
    assert "device=device" in source
    assert 'device="cpu"' not in source and "device='cpu'" not in source


def test_the_device_is_never_inferred_from_availability() -> None:
    source = inspect.getsource(c6_module._quality_backend_device)

    for forbidden in ("cuda.is_available", "resolve_device", "torch"):
        assert forbidden not in source, forbidden
    assert "FROZEN_QUALITY_BACKEND_DEVICE" in source


def test_the_frozen_device_is_passed_straight_through(
        tmp_path: Path, monkeypatch) -> None:
    recorder = _Recorder()

    _run(tmp_path, monkeypatch, device="cuda", backend=recorder,
         fit=lambda package, config, backends: {"thresholds": {}})

    assert recorder.calls[0][1]["device"] == "cuda", (
        "whatever the decision freezes is what reaches the constructor")


# --- 8, 9, 10. the firewall ---------------------------------------------------

def test_no_engineering_fixture_reaches_the_calibration_substage() -> None:
    source = inspect.getsource(C6Adapter._fit_nominal_calibration)

    for forbidden in ("ENGINEERING_NOMINAL", "gate_metrics",
                      "SMOKE_CANDIDATES_PER_ARM"):
        assert forbidden not in source, forbidden


def test_the_calibration_substage_opens_no_source_dev_and_no_target() -> None:
    source = inspect.getsource(C6Adapter._fit_nominal_calibration)

    for forbidden in ('"source_dev"', "source_dev.parquet", "target_test",
                      "siw", "SiW", "label_live_spoof"):
        assert forbidden not in source, forbidden


def test_the_calibrator_reads_source_train_only() -> None:
    """The split is chosen inside the canonical calibrator, not here."""
    source = inspect.getsource(quality_calibration.calibrate)

    assert "SourceOnlyAudit()" in source
    assert "SampleStore.open(package_root, audit)" in source
