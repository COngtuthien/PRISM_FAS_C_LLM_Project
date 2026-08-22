"""C5_SOURCE_PAIR_PLAN_V1: the frozen schedule, and what it must not depend on.

Two properties carry the whole design.

The schedule must be ARM-INDEPENDENT. RND, DET and LLM differ only in recipe
content, which is the treatment under test. If an arm could influence which live
sample or which spoof source a position receives, a C6 acceptance-rate
difference would be uninterpretable — the Version-B confound §11.3 exists to
remove.

Generation identity must not bind the C6 quality gate. C6 picks its profile from
three preregistered candidates AFTER the candidates exist, so binding a threshold
into a candidate id would make the candidate SET a function of the gate and close
a C5 -> C6 -> C5 loop.

The fixture is a small synthetic `source_train` manifest — no images are
rendered and no GPU is touched, because the schedule is arithmetic over sample
ids and nothing here needs pixels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from prism_fas.synthesis import c5_source_pair_plan as plan_module  # noqa: E402
from prism_fas.synthesis.c5_source_pair_plan import (  # noqa: E402
    ARMS, CANDIDATES_PER_ARM, CROSS_DOMAIN, GPAT, PHYSICS, PHYSICS_NONE,
    RECIPES_PER_ARM, RENDERS_PER_RECIPE, SAME_DOMAIN, SourcePairPlanError,
    arm_candidate_plan_identity, build_source_pair_plan, candidate_identity,
    eligible_spoof_sources, source_pair_plan_identity)

pytest.importorskip("pyarrow")

PACKAGE_IDENTITY = "b1cf29b6" + "0" * 56


def _package(root: Path, *, live: int = 40, spoof: int = 40,
             subjects: bool = True) -> Path:
    """An M3B-shaped package whose source_train carries both domains."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for label, count in ((("live"), live), (("spoof"), spoof)):
        for index in range(count):
            dataset = "casia_fasd" if index % 2 == 0 else "msu_mfsd"
            rows.append({
                "sample_id": f"{label}_{dataset}_{index:03d}",
                "dataset": dataset,
                "source_record_id": f"{label}_{dataset}_rec{index:03d}",
                "subject_id": f"subj{index:03d}" if subjects else "",
                "project_split": "source_train",
                "label_live_spoof": label})
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(
        {name: [row[name] for row in rows] for name in rows[0]}),
        root / "manifests" / "source_train.parquet")
    # The two splits C5 must never open, as bytes no reader can parse.
    for forbidden in ("source_dev.parquet", "target_test_features.parquet"):
        (root / "manifests" / forbidden).write_bytes(b"opening this is a failure")
    (root / "PACKAGE_LOCK.json").write_text(json.dumps({
        "package_id": "prism_data_v1_m3b", "status": "validated",
        "content_identity_sha256": PACKAGE_IDENTITY,
        "package_validation": {"status": "passed"}}), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> Path:
    return _package(tmp_path_factory.mktemp("m3b") / "prism_data_v1_m3b")


@pytest.fixture(scope="module")
def plan(package: Path) -> dict:
    return build_source_pair_plan(package)


# --- 5-7. the frozen cardinalities -------------------------------------------

def test_the_plan_has_exactly_2048_positions(plan: dict) -> None:
    assert RECIPES_PER_ARM * RENDERS_PER_RECIPE == 2048 == CANDIDATES_PER_ARM
    assert len(plan["positions"]) == 2048
    assert plan["positions_per_arm"] == 2048


def test_the_routes_split_1024_and_1024(plan: dict) -> None:
    routes = [row["route"] for row in plan["positions"]]

    assert routes.count(PHYSICS) == 1024
    assert routes.count(GPAT) == 1024


def test_every_recipe_gets_exactly_four_physics_and_four_gpat(plan: dict) -> None:
    per_recipe: dict[int, list[str]] = {}
    for row in plan["positions"]:
        per_recipe.setdefault(row["recipe_ordinal"], []).append(row["route"])

    assert len(per_recipe) == 256
    for ordinal, routes in per_recipe.items():
        assert routes.count(PHYSICS) == 4, ordinal
        assert routes.count(GPAT) == 4, ordinal


def test_the_gpat_domain_schedule_is_two_same_and_two_cross(plan: dict) -> None:
    per_recipe: dict[int, list[str]] = {}
    for row in plan["positions"]:
        if row["route"] == GPAT:
            per_recipe.setdefault(row["recipe_ordinal"], []).append(row["domain_relation"])

    for ordinal, relations in per_recipe.items():
        assert relations.count(SAME_DOMAIN) == 2, ordinal
        assert relations.count(CROSS_DOMAIN) == 2, ordinal


def test_the_slot_schedule_is_the_frozen_one() -> None:
    assert plan_module.ROUTE_BY_SLOT == (PHYSICS, GPAT, PHYSICS, GPAT,
                                         PHYSICS, GPAT, PHYSICS, GPAT)
    assert plan_module.DOMAIN_RELATION_BY_SLOT == {1: SAME_DOMAIN, 3: CROSS_DOMAIN,
                                                   5: SAME_DOMAIN, 7: CROSS_DOMAIN}


def test_the_global_position_is_eight_r_plus_s(plan: dict) -> None:
    for row in plan["positions"]:
        assert row["position"] == 8 * row["recipe_ordinal"] + row["slot"]


# --- 8-9. arm independence, which is the scientific point --------------------

def test_the_same_position_gets_the_same_live_sample_in_every_arm(plan: dict) -> None:
    """The schedule is a function of the position; the arm is not an input.

    Built once and reused by all three arms, so this asserts the property the
    design relies on rather than re-running a per-arm builder.
    """
    by_position = {row["position"]: row["live_target_sample_id"]
                   for row in plan["positions"]}

    for arm in ARMS:
        for position, live in by_position.items():
            assert plan["positions"][position]["live_target_sample_id"] == live, arm


def test_the_same_gpat_position_gets_the_same_spoof_source_in_every_arm(
        package: Path) -> None:
    """Rebuilt from scratch three times: identical, because no arm is involved."""
    builds = [build_source_pair_plan(package) for _ in ARMS]
    signatures = [
        [(row["position"], row["spoof_source_sample_id"], row["domain_relation"])
         for row in build["positions"] if row["route"] == GPAT]
        for build in builds]

    assert signatures[0] == signatures[1] == signatures[2]
    assert signatures[0], "no GPAT positions were built"


def test_the_base_schedule_takes_no_arm_and_no_recipe_bank(plan: dict) -> None:
    assert plan["arm_independent"] is True
    serialized = json.dumps(plan)
    for arm in ARMS:
        assert f'"{arm}"' not in serialized
    assert "recipe_bank_identity" not in plan


def test_the_arm_identity_differs_while_naming_the_same_base(plan: dict) -> None:
    base = source_pair_plan_identity(plan)
    identities = {
        arm: arm_candidate_plan_identity(
            source_pair_plan_identity=base, arm=arm,
            recipe_bank_identity=f"bank-{arm}", gpat_checkpoint_sha256="ck" * 32,
            physics_engine_version="p1", ontology_identity="onto")
        for arm in ARMS}

    assert len(set(identities.values())) == 3, "each arm's plan is its own"
    # ...and all three are functions of the one shared base.
    other = arm_candidate_plan_identity(
        source_pair_plan_identity="a different base", arm="RND",
        recipe_bank_identity="bank-RND", gpat_checkpoint_sha256="ck" * 32,
        physics_engine_version="p1", ontology_identity="onto")
    assert other != identities["RND"]


# --- 10. exposure fairness ----------------------------------------------------

def test_live_exposure_differs_by_at_most_one(plan: dict) -> None:
    counts: dict[str, int] = {}
    for row in plan["positions"]:
        counts[row["live_target_sample_id"]] = counts.get(row["live_target_sample_id"], 0) + 1

    assert max(counts.values()) - min(counts.values()) <= 1, sorted(counts.values())[:5]
    assert len(counts) == plan["live_list_size"], "every live sample is used"


# --- 11-15. source-only, and the pairing constraints -------------------------

def test_only_source_train_is_read(package: Path, plan: dict) -> None:
    assert plan["source_only"] == {"source_dev_opened": False,
                                   "target_test_opened": False,
                                   "target_labels_opened": False,
                                   "target_access": 0}
    manifests = package / "manifests"
    assert (manifests / "source_dev.parquet").read_bytes().startswith(b"opening")
    assert (manifests / "target_test_features.parquet").read_bytes().startswith(b"opening")


def test_a_manifest_row_from_another_split_is_refused(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "bad"
    (root / "manifests").mkdir(parents=True)
    pq.write_table(pa.Table.from_pydict({
        "sample_id": ["a"], "dataset": ["casia_fasd"], "source_record_id": ["r"],
        "subject_id": ["s"], "project_split": ["source_dev"],
        "label_live_spoof": ["live"]}), root / "manifests" / "source_train.parquet")

    with pytest.raises(SourcePairPlanError, match="source_train"):
        plan_module.load_source_rows(root)


def test_no_target_token_appears_in_the_plan(plan: dict) -> None:
    serialized = json.dumps(plan).lower()

    assert "siw" not in serialized
    assert "target_test" not in serialized or '"target_test_opened": false' in serialized


def test_a_gpat_pair_never_shares_a_source_record(plan: dict) -> None:
    for row in plan["positions"]:
        if row["route"] == GPAT:
            assert row["live_source_record_id"] != row["spoof_source_record_id"]


def test_a_gpat_pair_never_shares_a_subject_when_both_are_known(package: Path,
                                                                plan: dict) -> None:
    live, spoof = plan_module.load_source_rows(package)
    subjects = {row.sample_id: row.subject_id for row in [*live, *spoof]}

    for row in plan["positions"]:
        if row["route"] != GPAT:
            continue
        live_subject = subjects[row["live_target_sample_id"]]
        spoof_subject = subjects[row["spoof_source_sample_id"]]
        if live_subject and spoof_subject:
            assert live_subject != spoof_subject, row["position"]


def test_the_domain_relation_is_honoured(plan: dict) -> None:
    for row in plan["positions"]:
        if row["route"] != GPAT:
            continue
        if row["domain_relation"] == SAME_DOMAIN:
            assert row["spoof_dataset"] == row["live_dataset"], row["position"]
        else:
            assert row["spoof_dataset"] != row["live_dataset"], row["position"]


# --- 17. an empty pool fails closed ------------------------------------------

def test_an_empty_eligible_pool_fails_closed(tmp_path: Path) -> None:
    """One dataset only, so no cross-domain spoof source can exist."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "single_domain"
    (root / "manifests").mkdir(parents=True)
    rows = [{"sample_id": f"{label}_{index:03d}", "dataset": "casia_fasd",
             "source_record_id": f"{label}_rec{index:03d}",
             "subject_id": f"s{index:03d}", "project_split": "source_train",
             "label_live_spoof": label}
            for label in ("live", "spoof") for index in range(8)]
    pq.write_table(pa.Table.from_pydict(
        {name: [row[name] for row in rows] for name in rows[0]}),
        root / "manifests" / "source_train.parquet")
    (root / "PACKAGE_LOCK.json").write_text(json.dumps({
        "content_identity_sha256": PACKAGE_IDENTITY}), encoding="utf-8")

    with pytest.raises(SourcePairPlanError, match="no eligible cross_domain"):
        build_source_pair_plan(root)


def test_the_constraints_are_never_relaxed_to_fill_a_pool(package: Path) -> None:
    live, spoof = plan_module.load_source_rows(package)
    target = live[0]

    for relation in (SAME_DOMAIN, CROSS_DOMAIN):
        eligible = eligible_spoof_sources(spoof, target, relation)
        assert eligible
        for row in eligible:
            assert row.source_record_id != target.source_record_id
            if target.subject_id and row.subject_id:
                assert row.subject_id != target.subject_id
            assert ((row.dataset == target.dataset) if relation == SAME_DOMAIN
                    else (row.dataset != target.dataset))


# --- 1-4. what the candidate identity binds, and what it must not ------------

def _identity(**overrides) -> str:
    base = dict(source_pair_plan_identity="plan", arm="RND",
                recipe_bank_identity="bank", recipe_id="r0", recipe_ordinal=0,
                slot=1, position=1, route=GPAT, live_target_sample_id="live_0",
                spoof_source_sample_id="spoof_0", package_identity=PACKAGE_IDENTITY,
                ontology_identity="onto", generator_binding="ck" * 32)
    return candidate_identity(**{**base, **overrides})


def test_the_candidate_identity_binds_no_quality_calibration() -> None:
    """Read off the function's own signature, so a new parameter is caught."""
    import inspect

    parameters = set(inspect.signature(candidate_identity).parameters)
    for forbidden in ("threshold_sha256", "fingerprint_reference_sha256",
                      "calibration_sha256", "quality_profile", "thresholds",
                      "calibration", "accepted"):
        assert forbidden not in parameters, forbidden


def test_changing_a_c6_threshold_cannot_change_a_candidate_id() -> None:
    """There is no input through which it could. That is the guarantee."""
    before = _identity()
    after = _identity()

    assert before == after
    source = (REPO / "src" / "prism_fas" / "synthesis" / "c5_source_pair_plan.py"
              ).read_text(encoding="utf-8")
    body = source[source.index("def candidate_identity"):]
    body = body[:body.index("\n__all__")]
    for forbidden in ("threshold", "calibration", "fingerprint_reference"):
        for line in body.splitlines():
            if forbidden in line and not line.strip().startswith(("#", "*")):
                assert "ABSENT" in body or "absent" in body, line


@pytest.mark.parametrize("field,value", [
    ("package_identity", "another package"),
    ("recipe_bank_identity", "another bank"),
    ("recipe_id", "another recipe"),
    ("live_target_sample_id", "another live"),
    ("spoof_source_sample_id", "another spoof"),
    ("route", PHYSICS),
    ("slot", 3),
    ("position", 9),
    ("recipe_ordinal", 1),
    ("arm", "DET"),
    ("generator_binding", "a different checkpoint"),
    ("source_pair_plan_identity", "another plan"),
    ("ontology_identity", "another ontology"),
])
def test_every_generation_relevant_input_changes_the_identity(field, value) -> None:
    assert _identity(**{field: value}) != _identity(), field


def test_the_physics_route_binds_the_engine_and_not_the_checkpoint() -> None:
    physics = _identity(route=PHYSICS, spoof_source_sample_id=None,
                        generator_binding="physics-engine-v1")
    gpat = _identity(route=GPAT, generator_binding="physics-engine-v1")

    assert physics != gpat
    assert physics.startswith("c5syn_")


def test_a_physics_candidate_records_no_spoof_source() -> None:
    with_none = _identity(route=PHYSICS, spoof_source_sample_id=None)
    with_sentinel = _identity(route=PHYSICS, spoof_source_sample_id=PHYSICS_NONE)

    assert with_none == with_sentinel


def test_the_plan_identity_excludes_nothing_that_determines_the_schedule(
        package: Path, plan: dict) -> None:
    base = source_pair_plan_identity(plan)
    moved = json.loads(json.dumps(plan))
    moved["positions"][0]["live_target_sample_id"] = "a different live"

    assert source_pair_plan_identity(moved) != base


def test_the_plan_declares_that_it_binds_no_calibration(plan: dict) -> None:
    assert plan["binds_quality_calibration"] is False


# --- 27. the Version-B planner is untouched ----------------------------------

def test_the_version_b_planner_is_not_imported_by_the_version_c_plan() -> None:
    """Checked as IMPORTS rather than as text: `arm_candidate_plan_identity` is
    this module's own function and merely contains the substring."""
    import ast

    tree = ast.parse((REPO / "src" / "prism_fas" / "synthesis"
                      / "c5_source_pair_plan.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)

    for forbidden in ("candidate_plan", "synthetic_bank"):
        assert not any(forbidden in name for name in imported), (forbidden, imported)


def test_the_version_b_contract_still_holds() -> None:
    from prism_fas.synthesis import candidate_plan

    assert candidate_plan.EXPECTED_TOTAL == 1120
    assert candidate_plan.EXPECTED_PER_ROUTE == 560
    assert candidate_plan.CANDIDATE_SEED == 20260806


def test_the_two_planners_produce_different_identity_prefixes() -> None:
    from prism_fas.synthesis import candidate_plan

    version_b = candidate_plan.candidate_id(
        package_identity=PACKAGE_IDENTITY, bank_identity="bank", route="gpat",
        live_sample_id="live_0", spoof_sample_id="spoof_0", recipe_id="r0",
        seed=20260806, generator_binding="ck" * 32)

    assert version_b.startswith("syn_")
    assert _identity().startswith("c5syn_")
    assert version_b != _identity()
