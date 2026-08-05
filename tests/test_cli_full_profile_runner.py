import json

import pytest
from typer.testing import CliRunner

import prism_fas.cli.main as cli
from prism_fas.cli.main import app
from prism_fas.data.m2_runner import RunExecutionResult

CONFIG = ["--config", "configs/paths.local.yaml", "--preprocess-config", "configs/data/preprocess_m2.yaml"]


def _result(dataset, role):
    return RunExecutionResult(run_profile="full_preprocessing", run_id=f"full-{dataset}", dataset=dataset, dataset_role=role, output_root="out", canonical_records_total=1, canonical_records_selected=1, canonical_records_attempted=1, samples_selected=1, samples_successful=1, samples_failed=0, frames_read=1, detector_calls=1, crops_written=1, failures_by_code={}, manifest_counts={"source_frames": 0, "source_crops": 0, "target_frames": 0, "target_crops": 0, "preprocessing_failures": 0}, dry_run=False, partial=False, status="completed", started_at="t0", finished_at="t1")


@pytest.mark.parametrize("dataset,role", [("casia_fasd", "source"), ("siw_mv2", "target")])
def test_full_profile_cli_invokes_context_aware_runner_with_correct_role(monkeypatch, tmp_path, dataset, role):
    calls, legacy = [], []
    monkeypatch.setattr(cli, "SCRFDDetector", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "run_m2a", lambda *a, **k: legacy.append(a) or {})
    monkeypatch.setattr(cli, "migrate_m2a", lambda *a, **k: legacy.append(a) or {})

    def fake_run_preprocessing(context, records, **kwargs):
        calls.append((context, list(records), kwargs))
        return _result(dataset, role)
    monkeypatch.setattr(cli, "run_preprocessing", fake_run_preprocessing)

    result = CliRunner().invoke(app, ["data", "preprocess", "run", "--dataset", dataset, *CONFIG, "--run-profile", "full_preprocessing", "--confirm-full-run", "--allow-partial-full-profile", "--limit-records", "2", "--output-dir-name", "cli_unit_smoke"])
    assert result.exit_code == 0, result.output

    assert len(calls) == 1 and legacy == []
    context, records, kwargs = calls[0]
    assert context.run_profile == "full_preprocessing" and context.dataset == dataset and context.dataset_role == role
    assert context.dry_run is False and context.preprocessing_config_hash == "8f1e68ef5bc646a24f5b636261c7741c08b79bc9ba46904e3490f111d348c5dd"
    assert context.detector_model_sha256 == "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
    assert context.detector_input_size == 320 and context.detector_threshold == .5
    assert "cli_unit_smoke" in str(context.output_root) and "m2a" not in context.output_root.parts
    assert len(records) == 2 and kwargs["detector"] is not None
    assert json.loads(result.output.strip().splitlines()[-1])["dataset"] == dataset


def test_full_profile_dry_run_does_not_execute_the_runner(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_preprocessing", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(cli, "run_m2a", lambda *a, **k: calls.append(a) or {})
    result = CliRunner().invoke(app, ["data", "preprocess", "run", "--dataset", "msu_mfsd", *CONFIG, "--run-profile", "full_preprocessing", "--confirm-full-run", "--all-records", "--dry-run"])
    assert result.exit_code == 0 and calls == []
    assert json.loads(result.output.strip())["execution"] == "dry_run"
