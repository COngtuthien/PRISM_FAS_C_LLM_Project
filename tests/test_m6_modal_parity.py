"""M6 focused tests. No network access: real Modal results are asserted from
the ignored report artifacts produced by the explicit integration commands."""
from __future__ import annotations
import ast
import re
import json
from pathlib import Path

import numpy as np
import pytest

from prism_fas.cloud.config import (assert_no_absolute_local_paths, assert_remote_path_is_safe, assert_upload_is_safe,
                                    load_cloud_config, redact)
from prism_fas.cloud.parity import (ParityError, assert_target_isolated, compare_decisions, compare_exact,
                                    compare_features, compare_numeric)

ROOT = Path(__file__).parents[1]
CONFIG = load_cloud_config(ROOT / "configs" / "cloud" / "modal_m6.yaml")
REPORTS = ROOT / "reports" / "m6"


def _rows(n=4, offset=0.0):
    return [{"sample_id": f"s{i}", "source_record_id": f"r{i}", "true_label": "live", "true_target": 0,
             "crop_sha256": "a" * 64, "prior_sha256": "b" * 64, "spoof_logit": 0.5 + offset,
             "p_spoof_calibrated": 0.4 + offset, "decision": "live"} for i in range(n)]


# --- config -----------------------------------------------------------------

def test_cloud_config_parses_and_pins_expectations():
    assert CONFIG.cloud_schema_version == "m6-modal-v1" and CONFIG.app_name == "prism-fas-b-m6"
    assert {v.name for v in CONFIG.volumes.values()} == {"prism-fas-b-data", "prism-fas-b-models", "prism-fas-b-runs"}
    assert CONFIG.volume("data").mode == "read_only" and CONFIG.volume("runs").mode == "read_write"
    assert CONFIG.package["expected_content_identity_sha256"] == "b1cf29b69a165ed5d9e074fc8127c17fbf057723edf9e272048ec3a564eb9dc6"
    assert CONFIG.package["expected_split_counts"] == {"source_train": 1440, "source_dev": 2079, "target_test": 3140}
    text = (ROOT / "configs" / "cloud" / "modal_m6.yaml").read_text(encoding="utf-8")
    assert "D:" not in text and "C:\\" not in text
    # No credential values or credential-bearing keys (a comment saying "no token" is fine).
    assert not re.search(r"\b(ak|as|st)-[A-Za-z0-9]{8,}", text)
    assert not re.search(r"^\s*(token|token_id|token_secret|api_key|secret)\s*:", text, re.MULTILINE)


def test_mount_paths_and_remote_paths_are_absolute_and_safe():
    for spec in CONFIG.volumes.values():
        assert spec.mount.startswith("/vol/")
    for remote in CONFIG.remote_paths.values():
        assert assert_remote_path_is_safe(remote) == remote
    for bad in ("packages/x", "/vol/../etc", "C:/vol/data"):
        with pytest.raises(ValueError):
            assert_remote_path_is_safe(bad)


@pytest.mark.parametrize("path", ["Dataset/casia-fasd", "data/work/m2/x", "configs/paths.local.yaml",
                                  "D:/AI on IOT/Anti_spoofing/Dataset", "model_cache/backbones"])
def test_raw_dataset_and_secret_paths_are_refused(path):
    with pytest.raises(PermissionError, match="refusing to upload"):
        assert_upload_is_safe(path)


def test_processed_package_upload_is_allowed():
    assert assert_upload_is_safe("data/processed/prism_data_v1_m3b").name == "prism_data_v1_m3b"


def test_gpu_allow_list_is_enforced():
    assert CONFIG.resolve_gpu(None) == "L4" and CONFIG.resolve_gpu("T4") == "T4"
    for bad in ("H100", "H200", "B200", "A100-80GB"):
        with pytest.raises(ValueError, match="allow-list"):
            CONFIG.resolve_gpu(bad)


def test_portable_metadata_rejects_absolute_paths():
    assert assert_no_absolute_local_paths({"run": "b00", "package": "packages/x"})
    for bad in ({"p": "D:\\AI on IOT\\x"}, {"p": "/home/user/x"}, {"p": "/Users/admin/x"}):
        with pytest.raises(ValueError, match="absolute local path"):
            assert_no_absolute_local_paths(bad)


def test_secrets_are_redacted_in_reports():
    assert redact("ak-SECRETVALUE123").startswith("ak-") and "SECRETVALUE" not in redact("ak-SECRETVALUE123")
    assert redact("prism-fas-b-data") == "prism-fas-b-data"


# --- import isolation --------------------------------------------------------

def test_trainer_core_never_imports_modal():
    offenders = []
    for path in (ROOT / "src" / "prism_fas" / "train").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(name.split(".")[0] == "modal" for name in names):
                offenders.append(path.name)
    assert offenders == [], f"TrainerCore must stay backend-neutral: {offenders}"


def test_modal_app_uses_single_trainer_and_safe_inputs():
    source = (ROOT / "modal_app.py").read_text(encoding="utf-8")
    assert "from prism_fas.train.trainer import train_b00" in source
    assert "def train_b00(" not in source, "must not define a second trainer"
    assert 'GPU_ALLOW_LIST = ("L4", "T4", "A10G")' in source
    assert ".add_local_dir(\"src\"" in source and "data/" not in source.split("image = (")[1].split(")")[0]


# --- parity comparators ------------------------------------------------------

def test_exact_comparator_matches_and_detects_drift():
    local = _rows(); remote = [dict(row) for row in local]
    assert compare_exact(local, remote, ("sample_id", "true_target", "crop_sha256"))["mismatches"] == 0
    remote[2]["true_target"] = 1
    with pytest.raises(ParityError, match="exact field mismatch"):
        compare_exact(local, remote, ("sample_id", "true_target"))
    with pytest.raises(ParityError, match="row count differs"):
        compare_exact(local, remote[:3], ("sample_id",))


def test_numeric_comparator_passes_inside_and_fails_outside_tolerance():
    tolerances = CONFIG.parity["tolerances"]
    local = np.array([0.10, -0.20, 3.5]); close = local + 1e-5
    result = compare_numeric(local, close, name="logits", max_abs=tolerances["logit_max_abs_diff"],
                             mean_abs=tolerances["logit_mean_abs_diff"])
    assert result["passed"] and result["max_abs_diff"] <= tolerances["logit_max_abs_diff"]
    with pytest.raises(ParityError, match="max abs diff"):
        compare_numeric(local, local + 1e-2, name="logits", max_abs=tolerances["logit_max_abs_diff"])
    with pytest.raises(ParityError, match="shape differs"):
        compare_numeric(local, local[:2], name="logits", max_abs=1.0)


def test_feature_cosine_comparator():
    rng = np.random.default_rng(0); local = rng.normal(size=(8, 320))
    assert compare_features(local, local + 1e-7, min_cosine=0.99999)["passed"]
    with pytest.raises(ParityError, match="mean cosine"):
        compare_features(local, rng.normal(size=(8, 320)), min_cosine=0.99999)


def test_near_threshold_decisions_are_ambiguous_not_failures():
    threshold = CONFIG.parity["tolerances"]["decision_ambiguous_band"], 0.3414527626344137
    band, thr = threshold[0], threshold[1]
    local = [{"sample_id": "s1", "p_spoof_calibrated": thr - 1e-5, "decision": "live"}]
    remote = [{"sample_id": "s1", "p_spoof_calibrated": thr + 1e-5, "decision": "spoof"}]
    result = compare_decisions(local, remote, threshold=thr, ambiguous_band=band)
    assert result["disagreements"] == 0 and result["numerically_ambiguous"] == 1
    far_local = [{"sample_id": "s2", "p_spoof_calibrated": 0.05, "decision": "live"}]
    far_remote = [{"sample_id": "s2", "p_spoof_calibrated": 0.95, "decision": "spoof"}]
    with pytest.raises(ParityError, match="outside the ambiguity band"):
        compare_decisions(far_local, far_remote, threshold=thr, ambiguous_band=band)


def test_target_fixture_isolation():
    clean = [{"sample_id": "t1", "source_record_id": "siw_a", "spoof_logit": 0.2, "decision": "live"}]
    assert assert_target_isolated(clean)["labels_present"] is False
    with pytest.raises(ParityError, match="forbidden fields"):
        assert_target_isolated([{**clean[0], "true_target": 1}])


# --- local reference fixture -------------------------------------------------

reference_required = pytest.mark.skipif(not (REPORTS / "local_parity_reference.json").is_file(),
                                        reason="local parity reference not built")


@reference_required
def test_local_parity_reference_contract():
    reference = json.loads((REPORTS / "local_parity_reference.json").read_text(encoding="utf-8"))
    assert reference["device"] == "cpu"
    assert len(reference["source"]) == CONFIG.parity["source_dev_samples"]
    assert len(reference["target"]) == CONFIG.parity["target_samples"]
    assert reference["package_content_identity"] == CONFIG.package["expected_content_identity_sha256"]
    assert assert_target_isolated(reference["target"])["forbidden_fields"] == []
    text = json.dumps(reference)
    assert "D:\\" not in text and "C:\\" not in text and "/Users/" not in text
    assert all(row["true_label"] in {"live", "spoof"} for row in reference["source"])


# --- real Modal result contracts (from ignored report artifacts) --------------

def _remote(name):
    path = REPORTS / f"remote_{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@pytest.mark.skipif(_remote("probe") is None, reason="remote probe report not present")
def test_real_remote_environment_contract():
    probe = _remote("probe")
    assert probe["package_present"] and probe["model_present"] and probe["parity_inputs_present"]
    assert probe["package_identity"] == CONFIG.package["expected_content_identity_sha256"]
    assert probe["timm"] == "1.0.28"


@pytest.mark.skipif(_remote("verify") is None, reason="remote verify report not present")
def test_real_remote_package_validation_contract():
    verify = _remote("verify")
    assert verify["validation_passed"] and verify["errors"] == 0
    assert verify["identity_matches_expected"] and verify["weight_sha_matches"]
    assert verify["per_split_counts"] == CONFIG.package["expected_split_counts"]
    assert verify["target_isolation"] is True


@pytest.mark.skipif(_remote("smoke") is None, reason="remote smoke report not present")
def test_real_training_smoke_contract():
    smoke = _remote("smoke")
    assert smoke["gpu"]["gpu_name"] and smoke["steps_first"] == 5
    assert smoke["resume_continued"] and smoke["steps_after_resume"] >= 6
    assert all(np.isfinite(value) for value in smoke["losses"] + smoke["grad_norms"])
    for composition in smoke["batch_compositions"]:
        assert composition == {"casia_fasd/live": 8, "casia_fasd/spoof": 8, "msu_mfsd/live": 8, "msu_mfsd/spoof": 8}
    assert smoke["package_content_identity"] == CONFIG.package["expected_content_identity_sha256"]


@pytest.mark.skipif(not (REPORTS / "forward_parity.json").is_file(), reason="forward parity report not present")
def test_real_forward_parity_contract():
    parity = json.loads((REPORTS / "forward_parity.json").read_text(encoding="utf-8"))
    assert parity["passed"] is True
    assert parity["logits"]["max_abs_diff"] <= CONFIG.parity["tolerances"]["logit_max_abs_diff"]
    assert parity["probabilities"]["max_abs_diff"] <= CONFIG.parity["tolerances"]["probability_max_abs_diff"]
    assert parity["decisions"]["disagreements"] == 0


@pytest.mark.skipif(not (REPORTS / "checkpoint_portability.json").is_file(), reason="portability report not present")
def test_real_checkpoint_portability_contract():
    report = json.loads((REPORTS / "checkpoint_portability.json").read_text(encoding="utf-8"))
    assert report["loaded_on_cpu"] and report["global_step"] >= 6
    assert report["package_content_identity"] == CONFIG.package["expected_content_identity_sha256"]
    assert report["forward_finite"] is True
