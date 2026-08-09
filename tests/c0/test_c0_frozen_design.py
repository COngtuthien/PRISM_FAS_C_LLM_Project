"""C0 regression guards for PRISM-FAS-C-LLM.

These tests do not train, do not call a provider and do not read a dataset. They
assert that the frozen Version-C design constants stay as C0 recorded them, that the
Version-B lineage claim is internally consistent, and that the secret-handling policy
cannot be violated by a committed file.

A silent edit to `configs/version_c/c0_frozen_design.yaml` fails here rather than
surfacing as an unexplained change in a later scientific result.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
DESIGN_PATH = REPO / "configs" / "version_c" / "c0_frozen_design.yaml"
ACCEPTANCE_PATH = REPO / "reports" / "c0" / "C0_ACCEPTANCE.json"
SNAPSHOT_PATH = REPO / "reports" / "c0" / "VERSION_B_INTEGRITY_SNAPSHOT.json"

VERSION_B_CHECKPOINT = "7799f7decd35db6987ce4578824e5bd8d9eab4ae"
VERSION_B_TAG = "m10-blind-evaluation-checkpoint"
VERSION_C_SPEC_SHA256 = "d9edbfefa6829f29bb075e2f3d12073bb6517be57c0debcf93c92c4346d2e2df"


@pytest.fixture(scope="module")
def design() -> dict:
    return yaml.safe_load(DESIGN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def acceptance() -> dict:
    return json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


# --- datasets and protocols ------------------------------------------------

def test_exactly_three_datasets_and_no_fourth(design):
    datasets = design["datasets"]
    assert datasets["allowed"] == ["casia_fasd", "msu_mfsd", "siw_mv2"]
    assert datasets["count"] == 3 == len(datasets["allowed"])
    assert set(datasets["roles"]) == set(datasets["allowed"])


def test_siw_is_never_described_as_a_never_seen_blind_target(design):
    disclosure = design["datasets"]["siw_mv2_disclosure"].lower()
    assert "historically known" in disclosure
    assert "must not" in disclosure and "never-before-seen" in disclosure
    assert "procedural" in disclosure


def test_protocol_roles_and_seed_counts(design):
    protocols = design["protocols"]
    assert protocols["p1"]["train"] == ["casia_fasd"]
    assert protocols["p1"]["cross_domain_test"].startswith("msu_mfsd")
    assert protocols["p2"]["train"] == ["msu_mfsd"]
    assert protocols["p2"]["cross_domain_test"].startswith("casia_fasd")
    assert protocols["p3"]["train"] == ["casia_fasd", "msu_mfsd"]
    assert protocols["p3"]["cross_domain_test"] == "siw_mv2 v2"
    # P3 is the MAIN protocol and carries the 5-seed primary rows.
    assert protocols["p1"]["track_g_seeds"] == 3
    assert protocols["p2"]["track_g_seeds"] == 3
    assert protocols["p3"]["track_g_seeds"] == 5
    population = protocols["p3"]["target_population"]
    assert population["live"] + population["spoof"] == population["videos"] == 1700


# --- generator arms and budgets --------------------------------------------

def test_three_matched_generator_arms_of_256_recipes(design):
    arms = design["generator_arms"]
    assert arms["arms"] == ["RND", "DET", "LLM"]
    assert arms["final_recipe_count_per_arm"] == 256


def test_llm_candidate_budget_384_to_256(design):
    bank = design["llm"]["bank"]
    assert bank["pilot_candidates"] == 32
    assert bank["scientific_candidate_slots"] == 384
    assert bank["final_recipe_count"] == 256
    assert bank["request_schedule"] == "12x32"
    # 12 calls x 32 objects is exactly the 384-slot budget.
    calls, per_call = (int(part) for part in bank["request_schedule"].split("x"))
    assert calls * per_call == bank["scientific_candidate_slots"]
    # The selection pool floor sits between the final bank and the raw budget.
    floor = bank["minimum_valid_unique_pool_before_selection"]
    assert bank["final_recipe_count"] < floor == 320 < bank["scientific_candidate_slots"]


def test_synthesis_budget_2048_candidates_and_1024_accepted_per_arm(design):
    synthesis = design["synthesis"]
    arms = design["generator_arms"]["final_recipe_count_per_arm"]
    assert synthesis["renders_per_recipe"] == 8
    assert synthesis["physics_renders_per_recipe"] == 4
    assert synthesis["gpat_renders_per_recipe"] == 4
    assert (synthesis["physics_renders_per_recipe"]
            + synthesis["gpat_renders_per_recipe"]) == synthesis["renders_per_recipe"]
    assert arms * synthesis["renders_per_recipe"] == synthesis["candidate_renders_per_arm"] == 2048
    assert synthesis["candidate_renders_all_arms"] == 2048 * 3 == 6144
    assert synthesis["accepted_per_arm"] == 1024
    assert synthesis["accepted_physics_per_arm"] == 512
    assert synthesis["accepted_gpat_per_arm"] == 512
    assert (synthesis["accepted_physics_per_arm"]
            + synthesis["accepted_gpat_per_arm"]) == synthesis["accepted_per_arm"]
    assert "C6 FAILS" in synthesis["gate_failure_policy"]


def test_q_is_a_weight_and_never_a_label(design):
    q = design["synthesis"]["q_semantics"].lower()
    assert "never a class label" in q
    assert "only the synthetic loss bracket" in q


# --- LLM provider contract --------------------------------------------------

def test_llm_provider_and_model_are_frozen(design):
    llm = design["llm"]
    assert llm["provider"] == "google_gemini_developer_api"
    assert llm["model_id"] == "gemini-3.6-flash"
    assert llm["generation"]["thinking_level"] == "medium"
    assert llm["generation"]["response_mime_type"] == "application/json"
    assert llm["generation"]["modality"] == "TEXT_ONLY"


def test_deprecated_sampling_controls_are_forbidden(design):
    forbidden = design["llm"]["generation"]["forbidden_sampling_controls"]
    assert set(forbidden) == {"temperature", "top_p", "top_k"}


def test_every_provider_capability_that_could_leak_is_disabled(design):
    disabled = design["llm"]["disabled_capabilities"]
    expected = {"tools", "google_search_grounding", "url_context", "code_execution",
                "file_search", "image_input", "audio_input", "video_input"}
    assert set(disabled) == expected
    assert all(value is False for value in disabled.values())


def test_no_dataset_or_target_information_may_reach_the_provider(design):
    leakage = design["llm"]["leakage"]
    assert all(value is False for value in leakage.values())
    assert "allow_source_images_to_llm" in leakage
    assert "allow_target_images_to_llm" in leakage
    assert "allow_siw_taxonomy" in leakage
    assert "allow_siw_metrics" in leakage


def test_billing_starts_free_and_is_never_auto_enabled(design):
    billing = design["llm"]["billing"]
    assert billing["start_tier"] == "free"
    assert billing["auto_enable_paid"] is False
    assert design["llm"]["retry"]["quota_exhausted_action"] == "checkpoint_and_stop"
    assert design["llm"]["retry"]["semantic_max_retries"] == 2


def test_c0_made_zero_live_provider_requests(design):
    assert design["llm"]["c0_live_requests_made"] == 0


# --- GPAT neutrality, tracks, replication, statistics -----------------------

def test_gpat_c_is_generator_neutral_and_shared(design):
    gpat = design["gpat_c"]
    assert gpat["requirement"] == "generator-neutral"
    assert gpat["trained_once"] is True
    assert gpat["frozen_and_shared_by_all_arms"] is True
    assert "INDEPENDENT" in gpat["support_bank"]
    assert "own recipe bank" in gpat["normative"]


def test_track_g_is_architecturally_minimal(design):
    track_g = design["tracks"]["track_g"]
    assert track_g["regions"] is False          # YAML `off` parses to False
    assert track_g["prompt_head"] is False
    assert track_g["manifold"] is False
    assert "PRIMARY" in track_g["role"]


def test_track_r_keeps_the_manifold_off_in_its_primary_variant(design):
    track_r = design["tracks"]["track_r"]
    assert track_r["manifold"] is False
    assert track_r["manifold_secondary_k"] == 4
    assert track_r["prompt_head"] is True
    assert track_r["regions"] == 9
    assert "Atto" in track_r["local_branch"]
    assert "NOT Tiny" in track_r["local_branch"]


def test_replication_policy(design):
    replication = design["replication"]
    assert replication["seed_family_5"] == [20260806, 20260807, 20260808, 20260809, 20260810]
    assert replication["seed_family_3"] == replication["seed_family_5"][:3]
    assert replication["p3_track_g_primary_seeds"] == 5
    assert replication["p1_p2_track_g_seeds"] == 3
    assert replication["track_r_and_prompt_head_seeds"] == 3
    assert replication["best_seed_reporting"] == "forbidden"
    assert "diagnostic only" in replication["single_seed_rows"]


def test_statistical_protocol(design):
    stats = design["statistics"]
    assert stats["bootstrap_unit"] == "VIDEO"
    assert stats["resamples"] == 10000
    assert stats["bootstrap_seed"] == 20260810
    assert stats["confidence_interval"] == "95% percentile"
    assert stats["multiple_comparison_correction"] == "Holm-Bonferroni"
    assert stats["target_hypothesis_family"] == ["C-H1", "C-H2", "C-H3", "C-H5"]
    assert stats["reported_separately"] == ["C-H4"]
    # C-H4 is source-only and must never be mixed into the target family.
    assert not set(stats["reported_separately"]) & set(stats["target_hypothesis_family"])


def test_compute_is_backend_neutral(design):
    compute = design["compute"]
    assert compute["backend_ids"] == ["modal_l4", "ssh_lab"]
    assert compute["backend_is_scientific_factor"] is False
    assert compute["ssh_lab"]["mandatory_before"] == "C8"
    assert "UNKNOWN" in compute["ssh_lab"]["gpu_model"]
    assert "never silently change precision" in compute["precision_rule"].lower()


# --- Version-B lineage ------------------------------------------------------

def test_version_b_lineage_points_at_the_frozen_checkpoint(design):
    lineage = design["lineage"]
    assert lineage["version_b_checkpoint"] == VERSION_B_CHECKPOINT
    assert lineage["version_b_tag"] == VERSION_B_TAG
    assert "immutable" in lineage["version_b_role"]


def test_version_b_snapshot_is_internally_consistent(snapshot):
    git = snapshot["git"]
    assert git["head"] == VERSION_B_CHECKPOINT
    assert git["main"] == VERSION_B_CHECKPOINT
    assert git["origin_main"] == VERSION_B_CHECKPOINT
    assert git["tag_m10_blind_evaluation_checkpoint_peeled"] == VERSION_B_CHECKPOINT
    assert git["expected_checkpoint"] == VERSION_B_CHECKPOINT
    assert git["status_clean"] is True


def test_version_b_recipe_bank_proves_no_external_llm_was_invoked(snapshot):
    """The single fact that justifies the Version-C LLM arm being a new factor."""
    evidence = snapshot["llm_gap_evidence"]
    assert evidence["m7_generator_external_llm_invoked"] is False
    assert evidence["m7_generator_model_id"] == "deterministic-source-only-recipe-generator"
    assert evidence["m7_generator_provider"] == "deterministic_local"
    assert snapshot["m7_recipe_bank"]["status"] == "frozen"
    assert snapshot["m7_recipe_bank"]["recipe_count"] == 128


def test_no_expected_version_b_artifact_is_missing(snapshot):
    assert "NOT FOUND" not in json.dumps(snapshot)


def test_version_b_test_baseline_is_recorded(snapshot):
    baseline = snapshot["version_b_test_baseline"]
    assert baseline["passed"] == 1081
    assert baseline["failed"] == 0
    assert baseline["skipped"] == 0


# --- secret handling --------------------------------------------------------

def test_env_example_declares_names_without_values():
    lines = (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
    assignments = [line for line in lines
                   if "=" in line and not line.lstrip().startswith("#")]
    assert assignments, "the example file must declare at least one variable name"
    assert any(line.startswith("GEMINI_API_KEY=") for line in assignments)
    for line in assignments:
        name, _, value = line.partition("=")
        assert value == "", f"{name} carries a value; .env.example must be names only"


def test_gitignore_keeps_secrets_and_raw_data_out_but_admits_c0_reports():
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    for rule in (".env", "!.env.example", "data/evaluation_only/", "model_cache/",
                 "*.pem", "*.key", "data/recipe_banks/", "data/synthetic_banks/",
                 "data/llm_raw_responses/"):
        assert rule in gitignore, f"missing .gitignore rule: {rule}"
    # The Version-C milestone report roots must be committable, or C0 cannot ship
    # its acceptance artifact.
    assert "!reports/c0/" in gitignore
    assert "reports/*" in gitignore


def test_no_committed_c0_artifact_contains_an_api_key(acceptance, snapshot, design):
    """A key must never reach Git, a log, a report or a provenance record."""
    blob = json.dumps([acceptance, snapshot, design])
    # Google API keys are 39 characters beginning with this prefix.
    assert "AIza" not in blob
    for token in ("GEMINI_API_KEY=", "api_key\":", "Authorization:"):
        assert token not in blob


# --- C0 acceptance ----------------------------------------------------------

def test_c0_acceptance_declares_no_execution_happened(acceptance):
    forbidden = acceptance["no_execution_proof"]
    assert forbidden["gemini_api_calls"] == 0
    assert forbidden["gpu_training_jobs"] == 0
    assert forbidden["modal_jobs"] == 0
    assert forbidden["ssh_lab_jobs"] == 0
    assert forbidden["synthetic_renders"] == 0
    assert forbidden["siw_label_reads"] == 0
    assert forbidden["target_metrics_computed"] == 0
    assert forbidden["version_b_artifacts_modified"] == 0


def test_c0_acceptance_has_no_failed_mandatory_check(acceptance):
    failed = [name for name, passed in acceptance["checks"].items() if passed is not True]
    assert failed == [], f"failed C0 acceptance checks: {failed}"
    assert acceptance["result"] == "PASS"


def test_c0_records_the_authoritative_spec_fingerprint(acceptance):
    spec = acceptance["specs"]["version_c_authority"]
    assert spec["sha256"] == VERSION_C_SPEC_SHA256
    assert spec["size_bytes"] == 455241


def test_reconciliation_counts_sum_to_the_audited_total(acceptance):
    reconciliation = acceptance["spec_reconciliation"]
    statuses = reconciliation["status_counts"]
    assert sum(statuses.values()) == reconciliation["requirements_audited"] == 37
    assert set(statuses) == {"EXACT", "PARTIAL", "DEVIATED", "OMITTED",
                             "SUPERSEDED_WITH_JUSTIFICATION", "NOT_ESTABLISHED"}


def test_every_required_c0_document_exists(acceptance):
    for relative in acceptance["artifacts"]["documents"]:
        assert (REPO / relative).is_file(), f"missing C0 artifact: {relative}"
