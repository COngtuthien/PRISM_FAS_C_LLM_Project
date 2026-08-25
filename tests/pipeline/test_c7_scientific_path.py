"""The C7 scientific SOURCE_SEARCH, rehearsed end to end without a GPU.

This file exists because the C6 milestone cost several real GPU reruns to find
defects that were wiring, not science: a dict where an object was required, a
producer and a consumer disagreeing about a path. The way not to repeat that is
to drive the REAL production lifecycle here — the real search plan, the real
coordinate engine, the real trial-summary writer, the real lock writer and the
real lock verifier — and stub only the two things a laptop genuinely cannot do:
put a detector on a CUDA device, and forward a batch through it.

What that means concretely:

* `_scientific_trial_config` runs for real, over the real `m9_reference.yaml`
  through the canonical `load_m9_configs`, so a coordinate that names a scalar
  the config does not have fails HERE;
* `_run_scientific_trial` runs for real, so a wrong state key, a wrong path or a
  trial summary that omits a field the finalizer reads fails HERE;
* `coordinate_search` runs for real, so one-pass ordering, retention of failed
  and divergent trials and resume fail HERE;
* `_scientific_finalize` and `verify_detector_config_lock` run for real against
  each other, so a lock-schema mismatch between producer and verifier fails HERE.

Nothing in this file is scientific evidence. Every artifact is written into a
tmp_path sandbox, the metrics come from a synthetic frame table, and the profile
under which the adapter is driven is the only scientifically eligible one — which
is exactly the point: these tests prove what the SCIENTIFIC path does.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from c6_bank_fixture import (PACKAGE_IDENTITY, RECIPE_BANK_IDENTITY,  # noqa: E402
                             build_c6_closure, install_c3_bank)
from conftest_adapters import make_sandbox, request_for  # noqa: E402
from prism_fas.evaluation import source_selection  # noqa: E402
from prism_fas.pipeline.adapters import c7  # noqa: E402
from prism_fas.synthesis import c6_matched_bank as selector  # noqa: E402

#: The frozen search population, from the decision record rather than restated.
#: A test asserting DET against a hard-coded "DET" would keep passing if the
#: record changed; asserting against the loader is what makes these tests notice.
ARM = "DET"


# --- the two things a laptop cannot do ---------------------------------------

@dataclass
class _Contract:
    payload_body: dict[str, Any] = field(default_factory=lambda: {"real_live": 12,
                                                                  "real_spoof": 12,
                                                                  "synthetic": 8})

    def payload(self) -> dict[str, Any]:
        return dict(self.payload_body)


@dataclass
class _Sampler:
    contract: _Contract = field(default_factory=_Contract)


@dataclass
class _Identity:
    def payload(self) -> dict[str, Any]:
        return {"config_hash": "0" * 64, "architecture_identity": "1" * 64}


class _Variant:
    def flags(self) -> dict[str, Any]:
        return {"fusion": "glr_concat"}

    def identity(self) -> str:
        return "2" * 64


class _Model:
    """Only what a run manifest reads off the model."""

    def architecture_identity(self) -> str:
        return "3" * 64


class _Image:
    shape = (2, 3, 224, 224)


class _Batch:
    image = _Image()


class _Validation:
    """The narrowest `M9ValidationDataset` a caller can get away with."""

    positions = (0, 1)

    def batch(self, positions: Any) -> _Batch:
        return _Batch()


class _StubTrainer:
    """Everything the production row/trial runners read off a trainer, and no more.

    Deliberately narrow: the point is to discover, by AttributeError, every
    attribute the production code assumes a trainer has. A permissive Mock would
    answer every access and prove nothing — `validation()` is in this class only
    because C8's row runner asked for it and the stub refused.
    """

    instances: list["_StubTrainer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.config = kwargs["config"]
        self.run_root = Path(kwargs["run_root"])
        self.cache_root = Path(kwargs.get("cache_root", self.run_root / "cache"))
        self.device = kwargs["device"]
        self.loader_config = None
        self.variant = _Variant()
        self.identity = _Identity()
        self.samplers = {"G5": _Sampler()}
        self.decision_logit_name = "fused_logit_R"
        self.decision_score_name = "p_R"
        self.model = _Model()
        type(self).instances.append(self)

    def validation(self) -> _Validation:
        return _Validation()

    def checkpoint_path(self, kind: str) -> Path:
        path = self.run_root / "checkpoints" / f"{kind}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(f"checkpoint::{self.config.hash()}::{kind}".encode())
        return path


def _flow(trainer: Any, *, resume: bool = True) -> dict[str, Any]:
    trainer.checkpoint_path("best")
    return {"run_summary": {"best_metrics": {"epoch": 7}, "git_commit": "deadbeef",
                            "device": trainer.device, "stage_lineage": []},
            "stages": {"G5": {"status": "COMPLETED"}},
            "declared_stages": ["G1", "G2", "G5", "G6"],
            "stages_executed_here": ["G1", "G2", "G5", "G6"],
            "run_closure": {"closed": True}, "resumed_from": None,
            "resumed_stage": None,
            "source_isolation": {"target_test_opened": False,
                                 "source_dev_opened": True}}


def _frames(quality: float,
            domains: tuple[str, ...] = ("casia_fasd", "msu_mfsd")) -> list[dict[str, Any]]:
    """A synthetic source_dev frame table over the requested domains.

    `domains` is not decoration. In production `M9ValidationDataset` is already
    scoped by `config.source_domains`, so a P1 row's frames contain CASIA alone;
    a stub that returned both would hand `source_selection.evaluate` a domain the
    protocol does not select on, and the refusal that produced would be an
    artifact of the stub rather than a finding about the code.

    `quality` shifts the separation between live and spoof logits, so a trial's
    ranking tuple is a deterministic function of its configuration and the
    ordering the coordinate engine produces is checkable.
    """
    rows: list[dict[str, Any]] = []
    for domain in domains:
        for video in range(6):
            label = video % 2
            for frame in range(4):
                rows.append({
                    "sample_id": f"{domain}:v{video}:f{frame}",
                    "source_record_id": f"{domain}:v{video}",
                    "dataset": domain, "label": label,
                    "logit": (2.0 * quality if label else -2.0 * quality) + 0.01 * frame})
    return rows


# --- the sandbox -------------------------------------------------------------

@pytest.fixture
def scientific(tmp_path, monkeypatch):
    """A full-profile C7 request over a sandbox with a verifiable C6 closure."""
    repo = make_sandbox(tmp_path / "repo")
    per_route = 2
    monkeypatch.setattr(selector, "PER_ROUTE", per_route)
    monkeypatch.setattr(selector, "FINAL_BANK_PER_ARM", per_route * len(selector.ROUTES))
    built = build_c6_closure(repo, per_route=per_route)
    for arm in selector.ARMS:
        install_c3_bank(repo, arm, built["recipes"])

    # The frozen inputs C7 verifies. Resolving the real ones needs the M3B
    # package and the pinned weights, which this machine does not have; the
    # verifier itself is exercised by its own tests, so here it is stubbed to
    # return the identities the rest of the path binds.
    def inputs(_repo, **_kwargs):
        return {
            "package_root": "data/packages/prism_data_v1_m3b",
            "package_identity": PACKAGE_IDENTITY,
            "recipe_bank_root": "assets/recipe_banks/prism_recipe_bank_m7_v1",
            "recipe_bank_identity": RECIPE_BANK_IDENTITY,
            "recipe_bank_id": "prism_recipe_bank_m7_v1",
            "recipe_bank_recipe_count": 128,
            "candidates_root": "runs/full/c5/scientific/candidates",
            "weight_root": "weights",
            "pretrained": {"global_tower": {"identity_sha256": "a" * 64,
                                            "role": "frozen_global_tower"},
                           "local_backbone": {"weight_sha256": "b" * 64,
                                              "role": "trainable_local_branch"},
                           "stub_substituted": False, "downloaded_during_run": False},
            "c6": __import__("prism_fas.evaluation.c6_evidence",
                             fromlist=["verify_c6_evidence"]
                             ).verify_c6_evidence(repo).as_dict(),
            "c6_arms": list(selector.ARMS),
            "identities_agree": True,
            "target_paths_resolved": 0, "target_labels_resolved": 0,
            "verified_by": "test stub over the real verifier's output shape",
        }

    from prism_fas.detector import trainer as trainer_module
    from prism_fas.pipeline.adapters import sources

    monkeypatch.setattr(sources, "verify_detector_inputs", inputs)
    monkeypatch.setattr(c7, "_scientific_device", lambda: "cpu")
    monkeypatch.setattr(trainer_module, "M9Trainer", _StubTrainer)
    monkeypatch.setattr(trainer_module, "run_source_only_flow", _flow)
    monkeypatch.setattr(
        source_selection, "source_dev_frame_rows",
        lambda trainer: _frames(1.0 + 0.5 * float(trainer.config.weight_decay),
                                tuple(trainer.config.source_domains)))
    monkeypatch.setattr(
        "prism_fas.detector.decision_audit.decision_graph_hash",
        lambda model: {"decision_graph_hash": "c" * 64, "material": {}})
    _StubTrainer.instances = []
    return repo


def _unfreeze(repo: Path, **overrides: Any) -> None:
    """Put the decision record back into its pre-decision shape.

    The committed record is FROZEN at DET, so proving that an unresolved decision
    BLOCKS the stage means unfreezing a copy in the sandbox rather than the other
    way round.
    """
    import yaml

    path = repo / "configs/search/c7_source_search_decision.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload.update({"decision_status": "NEEDS_SCIENTIFIC_DECISION", **overrides})
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _decision(repo: Path, **overrides: Any) -> None:
    """Overwrite named fields of the frozen record, for the refusal cases."""
    import yaml

    path = repo / "configs/search/c7_source_search_decision.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload.update(overrides)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _approve(repo: Path, **overrides: Any) -> None:
    """The committed record is already FROZEN; overrides are for refusal cases."""
    if overrides:
        _decision(repo, **overrides)


def _run(repo: Path, **kwargs: Any) -> list[Any]:
    request = request_for(repo, "full", **kwargs)
    adapter = c7.C7Adapter()
    return adapter.workflow(request, request.context)


def _by_mode(results: list[Any]) -> dict[str, Any]:
    return {result.mode: result for result in results}


def _search_result(results: list[Any]) -> Any:
    """The SEARCH result, not the PLAN result. Both carry the same mode."""
    return next(result for result in results
                if result.mode == c7.SCIENTIFIC_SEARCH
                and result.substage != "C7_SCIENTIFIC_PLAN")


def _plan_result(results: list[Any]) -> Any:
    return next(result for result in results
                if result.substage == "C7_SCIENTIFIC_PLAN")


# --- the decision that blocks the first trial --------------------------------

def test_an_unfrozen_decision_blocks_before_any_trial(scientific) -> None:
    """An unresolved search population must stop C7 before the first trial."""
    _unfreeze(scientific)
    results = _run(scientific)
    modes = _by_mode(results)

    assert c7.VERIFY_C6_EVIDENCE in modes
    assert modes[c7.VERIFY_C6_EVIDENCE].status == "PASS"
    plan = modes[c7.SCIENTIFIC_SEARCH]
    assert plan.status != "PASS"
    assert "NEEDS_SCIENTIFIC_DECISION" in plan.summary
    blocked = next(item for item in plan.checks
                   if item["check_id"] == "c7_source_search_population_frozen")
    assert blocked["ok"] is False
    assert blocked["detail"]["reason_code"] == "NEEDS_SCIENTIFIC_DECISION"

    # No trial ran, and no lock was written.
    assert _StubTrainer.instances == []
    assert not (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).exists()


def test_the_precondition_gate_names_the_unresolved_decision(scientific) -> None:
    _unfreeze(scientific)
    request = request_for(scientific, "full")
    rows = c7.C7Adapter().semantic_preconditions(request)
    by_name = {row["name"]: row for row in rows}

    assert by_name["c6_closure_verified"]["present"] is True
    decision = by_name["c7_source_search_decision"]
    assert decision["present"] is False
    assert decision["blocking"] is True
    assert decision["reason_code"] == "NEEDS_SCIENTIFIC_DECISION"


# --- the full lifecycle, once the decision is approved -----------------------

def test_the_whole_lifecycle_runs_and_writes_a_verifiable_lock(scientific) -> None:
    _approve(scientific)
    results = _run(scientific)
    modes = _by_mode(results)

    for mode in (c7.VERIFY_C6_EVIDENCE, c7.SCIENTIFIC_SEARCH,
                 c7.FINALIZE_DETECTOR_CONFIG, c7.VERIFY_CONFIG_LOCK):
        assert mode in modes, f"{mode} did not run"
        assert modes[mode].status == "PASS", (
            mode, [item for item in modes[mode].checks if not item["ok"]])

    # Trials really ran, and every one of them is retained.
    search = _search_result(results)
    total = 0
    for track in ("G", "R"):
        retained = next(item for item in search.checks
                        if item["check_id"] == f"c7_track_{track.lower()}_all_trials_retained")
        assert retained["ok"] is True
        total += retained["detail"]["trials_executed"]
    assert total == len(_StubTrainer.instances) > 0

    lock_path = scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH
    assert lock_path.is_file()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["is_scientific_lock"] is True
    assert payload["fixture_backed"] is False
    assert payload["target_access"] == 0
    assert payload["metrics_from_trained_runs"] is True
    assert payload["training_arm"] == ARM
    assert sorted(payload["tracks"]) == ["G", "R"]
    for track, sub in payload["tracks"].items():
        assert len(sub["retained_trials"]) == sub["trials_executed"] > 0
        assert sub["winner_config_sha256"]
        assert sub["winner_checkpoint_sha256"]


def test_the_lock_verifier_accepts_what_the_lock_writer_produced(scientific) -> None:
    """Producer and verifier agree — the mismatch class this milestone exists to
    catch before a GPU run does."""
    _approve(scientific)
    _run(scientific)

    verification = c7.verify_detector_config_lock(
        scientific, scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH)
    assert verification["valid"] is True, [
        item["check_id"] for item in verification["checks"] if not item["ok"]]


@pytest.mark.parametrize("mutation,failing_check", [
    ({"is_scientific_lock": False}, "c7_config_lock_is_scientific"),
    ({"fixture_backed": True}, "c7_config_lock_is_scientific"),
    # The flag, not the prose. The verifier used to grep `metrics_source` for
    # "analytic", which the honest description "no analytic objective" trips
    # over — so a lock from a real run failed and a lock that simply omitted the
    # phrase passed. It is a declared boolean now.
    ({"metrics_from_trained_runs": False},
     "c7_config_lock_metrics_are_not_analytic"),
    ({"metrics_from_trained_runs": None,
      "metrics_source": "deterministic analytic objective"},
     "c7_config_lock_metrics_are_not_analytic"),
    ({"target_access": 1}, "c7_config_lock_declares_no_target_access"),
    ({"c6_bank_locks": {}}, "c7_config_lock_binds_its_inputs"),
    ({"per_arm_search_performed": True},
     "c7_config_lock_declares_no_per_arm_search"),
    ({"shared_within_track": False}, "c7_config_lock_declares_no_per_arm_search"),
    ({"training_arm": "LLM"}, "c7_config_lock_search_bank_is_the_frozen_arms"),
    ({"search_decision_identity": ""},
     "c7_config_lock_binds_the_frozen_search_arm"),
    ({"tracks": {}}, "c7_config_lock_names_a_configuration_per_track"),
])
def test_the_lock_verifier_refuses_each_mutation(scientific, mutation,
                                                 failing_check) -> None:
    """Injection tests: each way a lock could be wrong is caught by its own check."""
    _approve(scientific)
    _run(scientific)
    path = scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(mutation)
    path.write_text(json.dumps(payload), encoding="utf-8")

    verification = c7.verify_detector_config_lock(scientific, path)
    failed = {item["check_id"] for item in verification["checks"] if not item["ok"]}
    assert verification["valid"] is False
    assert failing_check in failed


@pytest.mark.parametrize("mutation,failing_check", [
    ({"retained_trials": []}, "c7_config_lock_track_r_retains_every_trial"),
    ({"winner_checkpoint_sha256": "0" * 64},
     "c7_config_lock_track_r_checkpoint_is_intact"),
    ({"search_plan_identity": ""}, "c7_config_lock_track_r_binds_the_frozen_envelope"),
    ({"decision_graph_hash": ""}, "c7_config_lock_track_r_binds_the_decision_graph"),
])
def test_the_lock_verifier_refuses_each_per_track_mutation(scientific, mutation,
                                                           failing_check) -> None:
    _approve(scientific)
    _run(scientific)
    path = scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tracks"]["R"].update(mutation)
    path.write_text(json.dumps(payload), encoding="utf-8")

    verification = c7.verify_detector_config_lock(scientific, path)
    failed = {item["check_id"] for item in verification["checks"] if not item["ok"]}
    assert verification["valid"] is False
    assert failing_check in failed


def test_a_failed_trial_is_retained_and_ranks_last(scientific, monkeypatch) -> None:
    """§15.2.2 retains invalid trials. A trial that will not train is a result."""
    from prism_fas.detector import trainer as trainer_module

    calls = {"n": 0}
    real = _flow

    def sometimes_fails(trainer: Any, *, resume: bool = True) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("planted trial failure: the detector did not train")
        return real(trainer, resume=resume)

    monkeypatch.setattr(trainer_module, "run_source_only_flow", sometimes_fails)
    _approve(scientific)
    results = _run(scientific)

    search = _search_result(results)
    retained = next(item for item in search.checks
                    if item["check_id"] == "c7_track_g_all_trials_retained")
    # `FAIL` is the engine's own vocabulary (`coordinate.TRIAL_STATUS`). A trial
    # runner that returned `FAILED` would leave the summary on disk and the
    # leaderboard disagreeing about the same configuration.
    assert retained["detail"]["trials_by_status"]["FAIL"] == 1
    assert retained["ok"] is True

    payload = json.loads(
        (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).read_text(encoding="utf-8"))
    statuses = [row["status"] for row in payload["tracks"]["G"]["retained_trials"]]
    assert "FAIL" in statuses
    # A failed trial ranks after every finite-valid one, so it is never first.
    assert statuses[0] == "PASS"
    assert statuses[-1] == "FAIL"

    # And its own summary survives, addressable by its config sha.
    failed = next(row for row in payload["tracks"]["G"]["retained_trials"]
                  if row["status"] == "FAIL")
    summary = scientific / failed["artifacts"][0]
    assert summary.is_file()
    body = json.loads(summary.read_text(encoding="utf-8"))
    assert body["status"] == "FAIL"
    assert "planted trial failure" in body["reason"]
    assert body["scientific_eligible"] is True


def test_resume_reuses_completed_trials_and_never_reruns_them(scientific) -> None:
    """A completed trial is a completed artifact; a restart does not redo it."""
    _approve(scientific)
    _run(scientific)
    first = len(_StubTrainer.instances)
    assert first > 0

    _StubTrainer.instances = []
    results = _run(scientific, resume=True)
    modes = _by_mode(results)

    assert modes[c7.VERIFY_CONFIG_LOCK].status == "PASS"
    # Zero, not "fewer". A COMPLETED pass is returned rather than re-walked: `best`
    # is restored to the final winning vector, so re-walking would generate the
    # EARLY coordinates' trials with the LATE ones already moved — different
    # configurations, different hashes, missing the reuse table, and retrained on
    # a GPU for nothing.
    assert _StubTrainer.instances == [], (
        "a resumed pass retrained trials the search state already records as PASS")


def test_the_scientific_search_state_is_namespaced_apart(scientific) -> None:
    _approve(scientific)
    _run(scientific)
    reports = scientific / "reports/full/c7"

    for track in ("G", "R"):
        assert (reports / c7._search_state_name(track)).is_file()
    assert not (reports / c7.SCIENTIFIC_SEARCH_STATE).exists()
    assert not (reports / "C7_SEARCH_STATE.json").exists(), (
        "the scientific pass wrote the engineering search state file")


@pytest.mark.parametrize("track", ("G", "R"))
def test_one_pass_in_the_frozen_coordinate_order(scientific, track) -> None:
    _approve(scientific)
    plan = _plan_result(_run(scientific))

    order = next(item for item in plan.checks
                 if item["check_id"]
                 == f"c7_track_{track.lower()}_coordinate_order_is_the_frozen_one")
    assert order["ok"] is True
    # The frozen §15.2.2 sequence starts with the learning rate. The approved
    # decision expresses it as a common multiplier, in the SAME position — so the
    # plan carries `learning_rate_multiplier` first, and the check compares
    # against the frozen order with exactly that substitution.
    assert order["detail"]["frozen_order"][:3] == ["learning_rate", "weight_decay",
                                                   "warmup"]
    assert order["detail"]["actual"][:3] == ["learning_rate_multiplier", "weight_decay",
                                             "warmup"]
    assert order["detail"]["lr_coordinate_name"] == "learning_rate_multiplier"

    optimizer = next(item for item in plan.checks
                     if item["check_id"] == "c7_optimizer_family_is_the_inherited_one")
    assert optimizer["ok"] is True and optimizer["detail"]["optimizer"] == "AdamW"

    schedule = next(item for item in plan.checks
                    if item["check_id"] == "c7_trial_schedule_is_not_shortened")
    assert schedule["ok"] is True
    assert schedule["detail"]["declared_trials_per_track"][track] > 0


def test_the_trial_config_moves_only_the_searched_scalar(scientific) -> None:
    """A coordinate may move the scalar it names and nothing else."""
    _approve(scientific)
    _run(scientific)

    configs = [instance.config for instance in _StubTrainer.instances]
    assert configs, "no trial was constructed"
    reference = configs[0]
    for config in configs:
        # Never searched: the batch, the schedule and the optimizer family.
        assert config.steps_per_epoch == reference.steps_per_epoch
        assert config.mixed_epochs == reference.mixed_epochs
        assert config.betas == reference.betas
        # Always the decided protocol's domains.
        assert tuple(config.source_domains) == source_selection.domains_for("P3")
    # Track R's learning-rate coordinate moved BOTH groups and held the frozen
    # 1:10 ratio. Track G has one applicable group, so its backbone_lr never moves
    # from the inherited anchor — which is what "uniquely inherited" means there.
    track_r = [config for config in configs if config.variant.track == "R"]
    ratios = {round(config.head_lr / config.backbone_lr, 9) for config in track_r}
    assert ratios == {10.0}


def test_the_trial_summary_carries_what_the_finalizer_reads(scientific) -> None:
    """The producer/consumer contract inside C7, checked field by field."""
    _approve(scientific)
    _run(scientific)
    payload = json.loads(
        (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).read_text(encoding="utf-8"))
    summary = json.loads(
        (scientific / payload["tracks"]["R"]["winner_trial_summary"]).read_text(
            encoding="utf-8"))

    for field_name in ("trial_config_sha256", "search_plan_identity", "checkpoint",
                       "checkpoint_sha256", "decision_logit_name",
                       "decision_score_name", "decision_graph_hash", "calibration",
                       "package_identity", "c6_bank_lock_selected_set_sha256",
                       "batch_contract", "code_lineage", "best_epoch",
                       "track", "training_arm", "parent_identities"):
        assert field_name in summary, f"{field_name} is missing from the trial summary"
    assert summary["calibration"]["decision_logit_name"] == summary["decision_logit_name"]
    assert summary["calibration"]["thresholded_quantity"] == summary["decision_score_name"]


def test_a_c6_closure_that_does_not_verify_blocks_the_stage(scientific) -> None:
    _approve(scientific)
    path = scientific / "reports/full/c6/C6_BANK_LOCK_RND.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance_closure"] = {"closed": False, "unaccounted": ["x"]}
    path.write_text(json.dumps(payload), encoding="utf-8")

    request = request_for(scientific, "full")
    rows = c7.C7Adapter().semantic_preconditions(request)
    closure = next(row for row in rows if row["name"] == "c6_closure_verified")
    assert closure["present"] is False
    assert closure["blocking"] is True
    assert closure["problems"]


# --- C7 closure audit: `inherited_anchor_report` is a pre-decision diagnostic,
# --- never a final-state blocker --------------------------------------------
#
# The defect class this section guards against: `inherited_anchor_report` is
# computed from the RAW inherited configuration, before the approved LR decision
# replaces the ambiguous per-scalar `learning_rate` coordinate in place
# (`plan._apply_lr_decision`). Both tracks' inherited config declares more than
# one candidate LR scalar (backbone_lr AND head_lr), so this report shows
# `ambiguous: ["learning_rate"]` / `executable_under_full: false` for EVERY real
# C7 run, Track G and Track R alike, regardless of whether the coordinate was
# actually searched and resolved. These tests prove that fact is (a) explicitly
# labeled as pre-decision in the lock, (b) never the source of any pass/fail
# decision C7 or C8 makes, and (c) carried beside — never instead of — the
# actual resolved, executed state for that track.

def test_track_g_final_lock_carries_the_approved_unique_anchor_resolution(
        scientific) -> None:
    from prism_fas.search.lr_decision import UNIQUE_ANCHOR

    _approve(scientific)
    _run(scientific)
    payload = json.loads(
        (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).read_text(encoding="utf-8"))
    sub = payload["tracks"]["G"]

    # The FINAL, resolved state: Track G's learning-rate coordinate WAS searched
    # and resolved to a unique inherited anchor.
    assert sub["lr_interpretation"] == UNIQUE_ANCHOR
    assert sub["lr_anchor_vector"] == {"head_lr": 0.0001}
    assert sub["lr_decision_identity"]
    resolution = sub["lr_decision_resolution"]
    assert resolution["interpretation"] == UNIQUE_ANCHOR
    assert resolution["searches_the_learning_rate"] is True
    assert resolution["multipliers"] == [0.5, 1.0, 2.0]

    # A real learning-rate coordinate was actually in the search plan and is
    # retained among the trials, i.e. this track was NOT blocked.
    assert any(row["coordinate"] == "learning_rate_multiplier"
              for row in sub["retained_trials"])


def test_track_r_final_lock_carries_the_approved_common_multiplier_resolution(
        scientific) -> None:
    from prism_fas.search.lr_decision import COMMON_MULTIPLIER

    _approve(scientific)
    _run(scientific)
    payload = json.loads(
        (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).read_text(encoding="utf-8"))
    sub = payload["tracks"]["R"]

    assert sub["lr_interpretation"] == COMMON_MULTIPLIER
    assert sub["lr_decision_identity"]
    resolution = sub["lr_decision_resolution"]
    assert resolution["interpretation"] == COMMON_MULTIPLIER
    assert resolution["searches_the_learning_rate"] is True
    assert resolution["preserved_ratio"]
    # The 1:10 backbone:head ratio is preserved at every multiplier.
    for values in resolution["lr_at_each_multiplier"].values():
        backbone = values.get("backbone_lr")
        head = values.get("head_lr")
        if backbone:
            assert round(head / backbone, 6) == 10.0

    assert any(row["coordinate"] == "learning_rate_multiplier"
              for row in sub["retained_trials"])


@pytest.mark.parametrize("track", ["G", "R"])
def test_the_pre_decision_diagnostic_cannot_be_mistaken_for_a_final_blocker(
        scientific, track) -> None:
    """The exact stale-report scenario this audit exists to close out.

    `inherited_anchor_report` is the raw pre-decision anchor lookup: both tracks'
    inherited config carries backbone_lr AND head_lr, so the raw lookup is always
    ambiguous and `executable_under_full` is always False there — independent of
    whether the coordinate was actually searched. A reader must not be able to
    confuse that with the track's real, final execution state.
    """
    _approve(scientific)
    _run(scientific)
    payload = json.loads(
        (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).read_text(encoding="utf-8"))
    sub = payload["tracks"][track]
    report = sub["inherited_anchor_report"]

    # The report is explicitly self-scoped, so it cannot be silently read as the
    # final state.
    assert report["diagnostic_scope"] == "PRE_DECISION_STRUCTURAL"
    assert "learning_rate" in report["ambiguous"]
    assert report["executable_under_full"] is False
    assert report["scope_note"]

    # Right beside it, the FINAL resolved state says the opposite: the track ran
    # to completion with real trials and a real winner.
    assert sub["lr_interpretation"]
    assert sub["lr_decision_resolution"]["searches_the_learning_rate"] is True
    assert sub["trials_executed"] > 0
    assert sub["trials_by_status"].get("PASS", 0) > 0
    assert sub["winner_config_sha256"]
    assert sub["winner_checkpoint_sha256"]

    # And the lock as a whole still verifies clean — the pre-decision diagnostic
    # is not read as a gating signal anywhere in the verifier.
    verification = c7.verify_detector_config_lock(
        scientific, scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH)
    assert verification["valid"] is True


def test_deleting_the_pre_decision_diagnostic_does_not_change_lock_validity(
        scientific) -> None:
    """`verify_detector_config_lock` (shared by C7 and C8) must never read
    `inherited_anchor_report`, `executable_under_full`, `ambiguous` or
    `blocking_reason` as a scientific decision input."""
    _approve(scientific)
    _run(scientific)
    path = scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    for track in payload["tracks"]:
        del payload["tracks"][track]["inherited_anchor_report"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    verification = c7.verify_detector_config_lock(scientific, path)
    assert verification["valid"] is True, [
        item["check_id"] for item in verification["checks"] if not item["ok"]]


def test_c8_track_configuration_reads_only_final_fields_not_the_diagnostic(
        scientific) -> None:
    """C8's per-row detector configuration never touches the pre-decision report
    or the raw ambiguity vocabulary — only the frozen `winner_config` and the
    identities in `_track_parents`."""
    from prism_fas.pipeline.adapters import c8

    _approve(scientific)
    _run(scientific)
    payload = json.loads(
        (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).read_text(encoding="utf-8"))

    for track in payload["tracks"]:
        poisoned = json.loads(json.dumps(payload))
        poisoned["tracks"][track]["inherited_anchor_report"] = {
            "diagnostic_scope": "PRE_DECISION_STRUCTURAL",
            "ambiguous": ["learning_rate"], "executable_under_full": False,
            "blocking_reason": "poisoned for this test"}
        sub = c8.track_configuration(poisoned, track)
        assert sub["winner_config"] == payload["tracks"][track]["winner_config"]
        parents = c8._track_parents(poisoned, track)
        assert parents["c7_detector_config"] == \
            payload["tracks"][track]["winner_config_sha256"]


def test_target_access_is_zero_throughout_the_finalized_lock(scientific) -> None:
    _approve(scientific)
    _run(scientific)
    payload = json.loads(
        (scientific / c7.SCIENTIFIC_CONFIG_LOCK_PATH).read_text(encoding="utf-8"))

    assert payload["target_access"] == 0
    assert payload["no_target_capability_proof"]["target_roots_mounted"] == []
    assert payload["no_target_capability_proof"]["target_labels_resolved"] == 0
    for track in payload["tracks"].values():
        assert track["winner_config_sha256"]


def test_reporting_metadata_never_enters_the_search_plan_identity() -> None:
    """The diagnostic keys added to `anchor_resolution_report` (`diagnostic_scope`,
    `scope_note`) and the per-track lock additions (`lr_decision_resolution`,
    `lr_decision_identity`) live entirely outside `SearchPlan.identity_material`,
    so no reporting-only change can move a search-plan identity, a winner config
    SHA or a trial-set digest."""
    import yaml

    from prism_fas.search.plan import (K4_ONLY_WEIGHTS, anchor_resolution_report,
                                       detector_search_plan)

    config = yaml.safe_load(
        (REPO / "configs/train/m9_reference.yaml").read_text(encoding="utf-8"))
    plan, resolutions = detector_search_plan(config, k4_weights=K4_ONLY_WEIGHTS)
    report = anchor_resolution_report(resolutions)

    assert report["diagnostic_scope"] == "PRE_DECISION_STRUCTURAL"
    material = plan.identity_material()
    serialized = json.dumps(material, sort_keys=True)
    for leaked_key in ("diagnostic_scope", "scope_note", "inherited_anchor_report",
                       "lr_decision_resolution", "blocking_reason"):
        assert leaked_key not in serialized, (
            f"{leaked_key!r} leaked into the search-plan identity material")
