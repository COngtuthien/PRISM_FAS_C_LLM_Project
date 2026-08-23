"""The written calibration must load. Producer to real consumer, end to end.

The §11.4 reconciliation replaced `thresholds` with the assembled inherited
NOMINAL but left `threshold_sha256` as the calibrator's hash of the map it had
just superseded. `FrozenCalibration.load` recomputes the hash from the
thresholds it is about to hand the evaluator, found the disagreement and refused
— correctly. The GPU run got as far as `_evaluate_generated_candidates` and
stopped there, so no candidate was measured and no acceptance count existed.

Every existing test stopped at the artifact. None of them fed the written file to
the consumer that reads it, which is the only place the two fields meet. That is
the gap here: the production writer runs, and the REAL `FrozenCalibration.load`
reads what it wrote.

The consumer stays strict. The producer was wrong.
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
from prism_fas.synthesis.c6_threshold_inheritance import (  # noqa: E402
    INHERITED_NOMINAL, VERSION_B_THRESHOLD_SHA256)
from prism_fas.synthesis.quality_gate import Thresholds  # noqa: E402
from prism_fas.synthesis.synthetic_bank import (FrozenCalibration,  # noqa: E402
                                                SyntheticBankError)

#: The calibrator's own fitted output — deliberately the superseded v1 values, so
#: the fitted map and the inherited map genuinely differ, as they did on the GPU.
FITTED = {"tau_fd": 0.5, "tau_id": 0.9995203357934952,
          "tau_lm": 0.002135227532959269, "tau_parse": 0.8747814437904173,
          "tau_out": 0.0, "tau_fp": 5.687657785453908}


def _calibrator_payload() -> dict[str, Any]:
    """What `quality_calibration.calibrate` returns, in the shape it returns it."""
    thresholds = Thresholds.from_dict(FITTED)
    return {
        "thresholds": thresholds.as_dict(),
        "threshold_sha256": thresholds.sha256(),
        "calibration_config_sha256": "c" * 64,
        "fingerprint": {"references": {"casia_fasd": {"mean": [0.1]}},
                        "reference_sha256": "f" * 64, "tau_fp": FITTED["tau_fp"]},
        "quality_models": {"models": {}},
        "populations": {"live": 10, "spoof": 10},
        "device": "cuda",
    }


class _Backends:
    device = "cuda"

    def __call__(self, weight_root: Any, **kwargs: Any) -> "_Backends":
        return self

    def manifest(self) -> dict[str, Any]:
        return {"models": {}}


@pytest.fixture
def written(tmp_path: Path, monkeypatch) -> tuple[Path, dict[str, Any]]:
    """Run the production writer and return the artifact it produced."""
    reports = tmp_path / "reports" / "full" / "c6"
    reports.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(c6_module, "FROZEN_QUALITY_BACKEND_DEVICE", "cuda")
    monkeypatch.setattr(c6_module, "_cuda_present", lambda: True)
    monkeypatch.setattr(c6_module, "_verify_quality_model_binding",
                        lambda weight_root: {"ok": True, "error_type": None,
                                             "roles": ["identity", "parsing",
                                                       "detector"]})
    monkeypatch.setattr(quality_calibration, "load_quality_config",
                        lambda path: {"benign": {"gaussian_noise_std": 0.001}})
    monkeypatch.setattr(quality_calibration, "QualityBackends", _Backends())
    science = c6_module.science_module()
    monkeypatch.setattr(science, "fit_nominal_calibration",
                        lambda package, config, backends: _calibrator_payload())

    request = AdapterRequest(repo=tmp_path, profile=load_profile("full", repo=REPO))
    state: dict[str, Any] = {"package_root": tmp_path / "package",
                             "package_identity": "b" * 64}
    result = C6Adapter()._fit_nominal_calibration(request, state, reports)
    assert result.status_axes.engineering != "BLOCKED", result.summary

    path = reports / "QUALITY_CALIBRATION.json"
    return path, {"artifact": json.loads(path.read_text(encoding="utf-8")),
                  "state": state, "result": result}


# --- 1-6. the two identities, kept apart --------------------------------------

def test_the_artifact_thresholds_are_the_inherited_nominal(written) -> None:
    _, bundle = written

    assert bundle["artifact"]["thresholds"] == INHERITED_NOMINAL
    assert bundle["artifact"]["thresholds"] != FITTED


def test_the_threshold_hash_is_the_hash_of_those_thresholds(written) -> None:
    """The invariant the consumer enforces, checked on the written bytes."""
    _, bundle = written
    artifact = bundle["artifact"]

    assert artifact["threshold_sha256"] == Thresholds.from_dict(
        artifact["thresholds"]).sha256()


def test_the_threshold_hash_equals_the_recorded_nominal_identity(written) -> None:
    _, bundle = written
    artifact = bundle["artifact"]

    assert artifact["threshold_sha256"] == artifact["nominal_identity_sha256"]


def test_the_threshold_hash_equals_the_version_b_threshold_hash(written) -> None:
    """Under the current all-inherited contract the two must coincide."""
    _, bundle = written

    assert bundle["artifact"]["threshold_sha256"] == VERSION_B_THRESHOLD_SHA256


def test_the_calibrator_fitted_map_and_its_own_hash_are_preserved(written) -> None:
    _, bundle = written
    artifact = bundle["artifact"]

    assert artifact["calibrator_fitted_thresholds"] == FITTED
    assert artifact["calibrator_fitted_threshold_sha256"] == Thresholds.from_dict(
        FITTED).sha256()
    assert artifact["calibrator_fitted_thresholds_are_provenance_only"] is True


def test_the_fitted_hash_really_differs_from_the_final_hash(written) -> None:
    """Otherwise this whole fixture would prove nothing."""
    _, bundle = written
    artifact = bundle["artifact"]

    assert (artifact["calibrator_fitted_threshold_sha256"]
            != artifact["threshold_sha256"])


# --- 7, 8, 9. the real consumer accepts it ------------------------------------

def test_the_real_frozen_calibration_loads_the_written_artifact(written) -> None:
    """The exact call that raised on the GPU host."""
    path, _ = written

    calibration = FrozenCalibration.load(path)

    assert calibration.threshold_sha256 == VERSION_B_THRESHOLD_SHA256


def test_the_loaded_thresholds_are_the_inherited_ones(written) -> None:
    path, _ = written

    calibration = FrozenCalibration.load(path)

    assert calibration.thresholds.as_dict() == INHERITED_NOMINAL
    assert calibration.thresholds.tau_id == INHERITED_NOMINAL["tau_id"]
    assert calibration.thresholds.tau_id != FITTED["tau_id"]


def test_the_evaluator_therefore_receives_the_inherited_thresholds(written) -> None:
    """`CandidateEvaluator` holds the calibration, so what loads is what gates."""
    from prism_fas.synthesis.synthetic_bank import CandidateEvaluator

    path, _ = written
    evaluator = CandidateEvaluator(_Backends(), FrozenCalibration.load(path))

    assert evaluator.calibration.thresholds.as_dict() == INHERITED_NOMINAL


# --- the negative: the consumer must keep refusing ----------------------------

def test_swapping_in_the_calibrator_hash_is_rejected(written, tmp_path) -> None:
    """The original defect, reconstructed. The consumer stays strict."""
    _, bundle = written
    broken = {**bundle["artifact"],
              "threshold_sha256": bundle["artifact"]["calibrator_fitted_threshold_sha256"]}
    path = tmp_path / "BROKEN_CALIBRATION.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(SyntheticBankError, match="threshold hash does not match"):
        FrozenCalibration.load(path)


def test_the_consumer_guard_was_not_weakened() -> None:
    source = inspect.getsource(FrozenCalibration.load)

    assert 'thresholds.sha256() != payload["threshold_sha256"]' in source
    assert "raise SyntheticBankError" in source
    for forbidden in ("nominal_identity_sha256", "calibrator_fitted", "C6"):
        assert forbidden not in source, (
            f"the consumer must not special-case anything: {forbidden}")


# --- state and artifact describe the same thresholds --------------------------

def test_the_in_memory_state_and_the_artifact_agree(written) -> None:
    _, bundle = written
    state, artifact = bundle["state"]["calibration"], bundle["artifact"]

    assert state["thresholds"] == artifact["thresholds"]
    assert state["threshold_sha256"] == artifact["threshold_sha256"]
    assert Thresholds.from_dict(state["thresholds"]).sha256() == state["threshold_sha256"]
    assert state["threshold_sha256"] == bundle["state"][
        "threshold_provenance"]["nominal_identity_sha256"]


def test_one_payload_builds_both(written) -> None:
    source = inspect.getsource(C6Adapter._fit_nominal_calibration)

    assert "_final_calibration_payload(" in source
    assert "**calibration," in source, "the artifact is written from that payload"
    assert 'state["calibration"] = calibration' in source
    # The old two-structure shape is gone.
    assert '{**payload, "thresholds": nominal}' not in source


def test_the_builder_refuses_a_self_inconsistent_payload() -> None:
    """It checks the consumer's invariant before writing, not after failing."""
    provenance = {"nominal_identity_sha256": "0" * 64}

    with pytest.raises(c6_module.ThresholdIdentityMismatch):
        c6_module._final_calibration_payload(
            _calibrator_payload(), INHERITED_NOMINAL, provenance,
            device="cuda", provenance={}, backends=_Backends(),
            package_identity="b" * 64)


# --- the profile label is provenance, not identity ----------------------------

def test_the_nominal_source_label_reflects_the_assembled_provenance() -> None:
    assert "§11.4 assembled NOMINAL" in c6_module.NOMINAL_SOURCE_LABEL
    assert "Version-B inherited" in c6_module.NOMINAL_SOURCE_LABEL
    source = inspect.getsource(C6Adapter._build_common_profiles)
    assert "nominal_source=NOMINAL_SOURCE_LABEL" in source
    assert 'nominal_source="source_train NOMINAL fitted at C6"' not in source


def test_the_label_enters_no_threshold_identity() -> None:
    """`threshold_identity` hashes the threshold VALUES, so the label is free."""
    from prism_fas.synthesis.c6_scientific import (build_common_profiles,
                                                   threshold_identity)

    one = build_common_profiles(INHERITED_NOMINAL, nominal_source="label A")
    two = build_common_profiles(INHERITED_NOMINAL, nominal_source="label B")

    for name in ("STRICT", "NOMINAL", "PERMISSIVE"):
        assert (threshold_identity(one[name].thresholds)
                == threshold_identity(two[name].thresholds))


def test_the_nominal_profile_matches_the_calibration_to_the_frozen_rounding() -> None:
    """The two NOMINAL identities are close but NOT equal, and that is expected.

    `derive_profile` returns `round(value, 12)`, which is frozen behaviour that
    predates this milestone, so the NOMINAL PROFILE carries values rounded at the
    twelfth decimal while the calibration artifact carries the inherited values
    unrounded. The difference is ~1e-13 and below any measurement resolution, but
    it means `quality_threshold_identity` and the calibration's
    `threshold_sha256` are different hashes of different objects. Recorded here
    so nobody later assumes they must match; the rounding is not changed.
    """
    from prism_fas.synthesis.c6_scientific import (build_common_profiles,
                                                   threshold_identity)

    profiles = build_common_profiles(INHERITED_NOMINAL,
                                     nominal_source=c6_module.NOMINAL_SOURCE_LABEL)
    profile_nominal = dict(profiles["NOMINAL"].thresholds)

    for name, value in INHERITED_NOMINAL.items():
        assert profile_nominal[name] == pytest.approx(value, abs=1e-12), name
    assert profile_nominal == {name: round(value, 12) + 0.0
                               for name, value in INHERITED_NOMINAL.items()}
    assert threshold_identity(profiles["NOMINAL"].thresholds) != VERSION_B_THRESHOLD_SHA256


# --- no downstream consumer binds the fitted identity -------------------------

def test_no_downstream_identity_binds_the_calibrator_fitted_hash() -> None:
    """Bank locks, selector identity and profile identity all use the profile
    thresholds, never the calibration payload's hash."""
    for relative in ("src/prism_fas/synthesis/c6_scientific.py",
                     "src/prism_fas/synthesis/c6_matched_bank.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        assert "calibrator_fitted" not in source, relative
        assert 'payload["threshold_sha256"]' not in source, relative

    adapter = (REPO / "src" / "prism_fas" / "pipeline" / "adapters" / "c6.py"
               ).read_text(encoding="utf-8")
    bank = inspect.getsource(C6Adapter._build_matched_banks)
    assert 'state["threshold_identities"][profile]' in bank
    assert "calibrator_fitted" not in bank
