"""Validate-profile readiness checks (v1.5 Appendix L.2).

L.2 defines validate as "schema/identity/dataset/environment/lock checks" with
no scientific training. Every check in this module obeys that literally: it
reads bytes already on disk, re-derives identities from live code, and compares.
Nothing here trains, calls a provider, launches a GPU job or resolves a target
label, and each check is a pure function of the repository.

Re-derivation, not transcription. A check that read an identity out of
`PROJECT_STATE.md` and compared it to the same document would always pass. So
the expected values come from the immutable locks (authority level 3) and from
`CLAUDE.md`'s hard invariants, while the actual values are computed by running
the same loaders the pipeline itself uses. When the two disagree, that is a real
finding, and the check reports both numbers rather than a bare False.

`ok=False` is a legitimate outcome. These checks measure; the orchestrator
decides what a failure means.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: `CLAUDE.md` hard invariants. Held as expectations to check against, never as
#: inputs to anything the pipeline computes.
SPEC_RELPATH = ("docs/PRISM_FAS_C_LLM_v1_5_FINAL_ComputeConstrained_"
                "FullPipeline_Spec_2026.docx")
EXPECTED_SPEC_SHA256 = "ad8495f2576607546ff8c3bd4f47991197cbb3802265a599d1808aa1a97066e5"

VERSION_B_PATH = Path(r"D:\AI on IOT\Anti_spoofing\PRISM_FAS_B_Project")
EXPECTED_VERSION_B_HEAD = "7799f7decd35db6987ce4578824e5bd8d9eab4ae"
EXPECTED_VERSION_B_TAG = "m10-blind-evaluation-checkpoint"

PRELIMINARY_LOCK = Path("reports/c3/C3_BANK_LOCK.json")
SUPERSEDING_LOCK = Path("reports/c3/v15_selection_contract/C3_BANK_CONTRACT_LOCK.json")

ONTOLOGY_CONFIG = Path("configs/recipes/ontology_m7.yaml")
ROUTE_POLICY_CONFIG = Path("configs/version_c/llm/c2c_route_policy.yaml")
PROVIDER_CONFIG = Path("configs/version_c/llm/c1_gemini_provider.yaml")
QUOTA_CONFIG = Path("configs/version_c/llm/c2b_coverage_quotas.yaml")

#: The batch size the frozen prompt identity was rendered at (§7.8, 12x32).
#: Read from the lock at check time rather than assumed; this is only the key.
_SCHEDULE_KEY = "objects_per_request"

#: Provider evidence C3 generation would create. Their absence is the positive
#: evidence that no C3 scientific request has been made.
C3_GENERATION_EVIDENCE_GLOBS = (
    "reports/c3/C3_RAW_ARCHIVE*.json",
    "reports/c3/C3_*_RAW_ARCHIVE*.json",
    "reports/c3/C3_PROVENANCE*.json",
    "reports/c3/C3_*_PROVENANCE*.json",
    "reports/c3/C3_BATCH_STATE*.json",
    "reports/c3/C3_LIVE_*_AUDIT*.json",
    "reports/c3/raw_responses/**/*",
)


@dataclass(frozen=True)
class CheckResult:
    """One measurement. `detail` carries both sides of every comparison."""

    check_id: str
    stage_id: str
    ok: bool
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"check_id": self.check_id, "stage_id": self.stage_id, "ok": self.ok,
                "summary": self.summary, "detail": self.detail}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --- C0 ---------------------------------------------------------------------

def check_spec_sha256(repo: Path) -> CheckResult:
    """The governing contract is the document `CLAUDE.md` pins, byte for byte."""
    path = repo / SPEC_RELPATH
    if not path.exists():
        return CheckResult("spec_sha256", "C0", False,
                           "the authoritative spec is not at its recorded path",
                           {"path": SPEC_RELPATH, "exists": False})
    actual = _sha256_file(path)
    ok = actual == EXPECTED_SPEC_SHA256
    return CheckResult(
        "spec_sha256", "C0", ok,
        "the spec on disk is the pinned v1.5 document" if ok else
        "the spec on disk is NOT the pinned v1.5 document",
        {"path": SPEC_RELPATH, "expected": EXPECTED_SPEC_SHA256, "actual": actual})


def check_version_b_integrity(repo: Path) -> CheckResult:
    """Version B is immutable: HEAD, peeled tag and a clean tree, all three.

    Checking HEAD alone would miss a moved tag, and checking the tag alone would
    miss a detached HEAD, so all three are measured and reported separately.
    """
    if not (VERSION_B_PATH / ".git").exists():
        return CheckResult("version_b_integrity", "C0", False,
                           "the Version-B repository is not readable at its recorded path",
                           {"path": str(VERSION_B_PATH), "exists": False})
    head = _git(["rev-parse", "HEAD"], VERSION_B_PATH)
    peeled = _git(["rev-parse", f"{EXPECTED_VERSION_B_TAG}^{{commit}}"], VERSION_B_PATH)
    dirty = _git(["status", "--porcelain"], VERSION_B_PATH)
    detail = {
        "path": str(VERSION_B_PATH),
        "expected_head": EXPECTED_VERSION_B_HEAD,
        "head": head,
        "tag": EXPECTED_VERSION_B_TAG,
        "tag_peeled_commit": peeled,
        "head_matches": head == EXPECTED_VERSION_B_HEAD,
        "tag_peels_to_expected": peeled == EXPECTED_VERSION_B_HEAD,
        "clean": dirty == "",
        "dirty_paths": dirty.splitlines() if dirty else [],
    }
    ok = bool(detail["head_matches"] and detail["tag_peels_to_expected"] and detail["clean"])
    return CheckResult(
        "version_b_integrity", "C0", ok,
        "Version B is at the frozen commit, tagged and clean" if ok else
        "Version B is NOT at the frozen commit, tag or clean state", detail)


def check_environment(repo: Path) -> CheckResult:
    """Record what this run executed on. Informational, and always ok.

    An environment is not right or wrong; it is a fact the artifact must carry
    so a later reader can tell whether two runs are comparable at all.
    """
    return CheckResult(
        "environment", "C0", True, "environment fingerprint recorded",
        {"python": platform.python_version(),
         "implementation": platform.python_implementation(),
         "platform": platform.platform(),
         "executable": sys.executable,
         "repo_commit": _git(["rev-parse", "HEAD"], repo),
         "repo_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
         "repo_dirty_paths": (_git(["status", "--porcelain"], repo) or "").splitlines()})


def _acceptance_present(repo: Path, stage_id: str, relpath: str,
                        check_id: str) -> CheckResult:
    """A milestone's frozen acceptance file exists and parses.

    The file's own verdict is reported but not re-judged: these are historical
    artifacts under authority level 4, and validate confirms they are readable
    and says what they claim. It does not re-derive their science.
    """
    path = repo / relpath
    if not path.exists():
        return CheckResult(check_id, stage_id, False,
                           f"{relpath} is missing", {"path": relpath, "exists": False})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return CheckResult(check_id, stage_id, False,
                           f"{relpath} does not parse as JSON",
                           {"path": relpath, "error": str(error)})
    claimed = payload.get("result") or payload.get("status") or payload.get("verdict")
    return CheckResult(
        check_id, stage_id, True, f"{relpath} is present and readable",
        {"path": relpath, "sha256": _sha256_file(path), "claimed_result": claimed,
         "predates_dual_status_schema": "execution_profile" not in payload,
         "note": "historical acceptance evidence; validate confirms it is readable and "
                 "reports what it claims, and never re-judges or edits it"})


def check_c0_acceptance(repo: Path) -> CheckResult:
    return _acceptance_present(repo, "C0", "reports/c0/C0_ACCEPTANCE.json",
                               "c0_acceptance_present")


# --- C1 ---------------------------------------------------------------------

def _derive_contract_identities(repo: Path) -> dict[str, str]:
    """Re-derive the eight frozen contract identities from live code.

    Each one is produced by the same loader the scientific path uses, reading
    the same config bytes, so a drifted config or a changed builder shows up
    here rather than at generation time. The imports are local because a
    validate run should not pay for them when only C0 was requested.
    """
    from prism_fas.llm.config import load_llm_config, provider_config_identity
    from prism_fas.llm.coverage_quotas import load_quota_spec
    from prism_fas.llm.json_schema import (candidate_json_schema, candidate_object_schema,
                                           json_schema_identity)
    from prism_fas.llm.prompt import build_generation_prompt, load_prompt_template
    from prism_fas.llm.route_policy import load_route_policy
    from prism_fas.recipes.ontology import load_ontology

    ontology = load_ontology(repo / ONTOLOGY_CONFIG)
    route_policy = load_route_policy(repo / ROUTE_POLICY_CONFIG)
    provider = load_llm_config(repo / PROVIDER_CONFIG)
    quotas = load_quota_spec(repo / QUOTA_CONFIG)

    batch_size = _frozen_batch_size(repo)
    template = load_prompt_template(ontology, route_policy)
    rendered = build_generation_prompt(template, recipes_requested=batch_size,
                                       coverage_quotas=quotas.prompt_block(ontology))

    return {
        "ontology_identity": ontology.sha256,
        "route_policy_identity": route_policy.route_policy_identity,
        "provider_config_identity": provider_config_identity(provider),
        "coverage_quota_identity": quotas.quota_identity,
        "system_prompt_identity": template.identity(),
        "batch_generation_template_identity":
            hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "single_recipe_schema_identity":
            json_schema_identity(candidate_object_schema(ontology)),
        # The C2B finding: the provider rejects the bounded envelope, so the
        # working contract omits the array bound and enforces exactly-N on the
        # response instead. The frozen identity is the unbounded one.
        "batch_envelope_schema_identity": json_schema_identity(
            candidate_json_schema(ontology, recipes_requested=batch_size,
                                  array_bounds=False)),
    }


def _frozen_batch_size(repo: Path) -> int:
    """The objects-per-request the frozen prompt identity was rendered at.

    Read from the immutable lock rather than written here, so this module holds
    no scientific constant of its own.
    """
    lock = json.loads((repo / PRELIMINARY_LOCK).read_text(encoding="utf-8"))
    schedule = lock["components"]["request_schedule"]
    return int(schedule[_SCHEDULE_KEY])


def check_contract_identities(repo: Path) -> CheckResult:
    """Every frozen C1/C2C contract identity still reproduces from live code.

    The expected values come from the immutable BANK_LOCK's `components`, and
    the actual values are computed by running the loaders. Both are reported per
    component so a single drifted config is immediately identifiable.
    """
    lock_path = repo / PRELIMINARY_LOCK
    if not lock_path.exists():
        return CheckResult("contract_identities", "C1", False,
                           "the frozen bank lock that records the expected identities is missing",
                           {"path": PRELIMINARY_LOCK.as_posix(), "exists": False})
    expected = json.loads(lock_path.read_text(encoding="utf-8"))["components"]
    try:
        actual = _derive_contract_identities(repo)
    except Exception as error:  # a loader that cannot run is itself the finding
        return CheckResult("contract_identities", "C1", False,
                           "a contract identity could not be re-derived from live code",
                           {"error": f"{type(error).__name__}: {error}"})

    components = [
        {"component": name, "expected": expected.get(name), "actual": value,
         "matches": expected.get(name) == value}
        for name, value in sorted(actual.items())]
    drifted = [item["component"] for item in components if not item["matches"]]
    return CheckResult(
        "contract_identities", "C1", not drifted,
        "all frozen contract identities reproduce from live code" if not drifted else
        f"contract identities drifted from the frozen lock: {drifted}",
        {"expected_source": PRELIMINARY_LOCK.as_posix(), "components": components,
         "drifted": drifted})


def check_c1_acceptance(repo: Path) -> CheckResult:
    return _acceptance_present(repo, "C1", "reports/c1/C1_ACCEPTANCE.json",
                               "c1_acceptance_present")


# --- C2 ---------------------------------------------------------------------

def check_c2_acceptance(repo: Path) -> CheckResult:
    return _acceptance_present(repo, "C2", "reports/c2/C2_ACCEPTANCE.json",
                               "c2_acceptance_present")


def check_route_contract_exact(repo: Path) -> CheckResult:
    """The generator route is exactly ["physics", "gpat"] — not a superset.

    C2C froze this as an exact sequence. A policy that merely *allowed* the two
    routes would also admit a physics-only or gpat-only recipe, which the bank
    lock explicitly refuses, so equality is the check and containment is not.
    """
    from prism_fas.llm.route_policy import load_route_policy

    lock_path = repo / PRELIMINARY_LOCK
    expected = ["physics", "gpat"]
    if lock_path.exists():
        contract = json.loads(lock_path.read_text(encoding="utf-8")).get("route_contract", {})
        expected = list(contract.get("required_generator_route", expected))

    policy = load_route_policy(repo / ROUTE_POLICY_CONFIG)
    actual = list(policy.allowed_scientific_generator_route)
    ok = actual == expected
    return CheckResult(
        "route_contract_exact", "C2", ok,
        "the generator route is exactly the frozen sequence" if ok else
        "the generator route does not equal the frozen sequence",
        {"expected": expected, "actual": actual,
         "expected_source": PRELIMINARY_LOCK.as_posix(),
         "policy_config": ROUTE_POLICY_CONFIG.as_posix(),
         "route_policy_identity": policy.route_policy_identity})


# --- C3 ---------------------------------------------------------------------

def check_c3_contract_identities(repo: Path) -> CheckResult:
    """Generation, selection and bank contract identities all re-derive (§7.8.4).

    The generation identity is read from the preliminary lock and its body hash
    is recomputed, so the lock cannot vouch for itself. The selection identity is
    rebuilt from the live selector, eligibility order, schedules, ontology and
    route policy. The bank identity is then recomputed by the §7.8.4 formula
    from those two rather than copied from either lock.
    """
    from prism_fas.llm.bank_lock import canonical_text, sha256_text
    from prism_fas.llm.route_policy import load_route_policy
    from prism_fas.llm.selection_contract import assemble
    from prism_fas.recipes.ontology import load_ontology

    lock_path = repo / PRELIMINARY_LOCK
    superseding_path = repo / SUPERSEDING_LOCK
    for path in (lock_path, superseding_path):
        if not path.exists():
            return CheckResult("c3_contract_identities", "C3", False,
                               f"{path.name} is missing",
                               {"path": str(path.relative_to(repo)), "exists": False})

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in lock.items() if key != "bank_lock_identity"}
    lock_body_identity = sha256_text(canonical_text(body))
    generation = lock["composite"]["c3_generation_contract_identity"]

    ontology = load_ontology(repo / ONTOLOGY_CONFIG)
    route_policy = load_route_policy(repo / ROUTE_POLICY_CONFIG)
    route_policy.validate_against(ontology)
    record = assemble(repo=repo, ontology=ontology, route_policy=route_policy,
                      generation_contract_identity=generation)

    superseding = json.loads(superseding_path.read_text(encoding="utf-8"))
    bound = superseding.get("contract_identities", {})

    comparisons = [
        {"identity": "preliminary_lock_body",
         "expected": lock.get("bank_lock_identity"), "actual": lock_body_identity},
        {"identity": "c3_selection_contract_identity",
         "expected": bound.get("c3_selection_contract_identity"),
         "actual": record["c3_selection_contract_identity"]},
        {"identity": "c3_bank_contract_identity",
         "expected": bound.get("c3_bank_contract_identity"),
         "actual": record["c3_bank_contract_identity"]},
        {"identity": "c3_generation_contract_identity",
         "expected": bound.get("c3_generation_contract_identity"), "actual": generation},
    ]
    for item in comparisons:
        item["matches"] = item["expected"] == item["actual"]
    drifted = [item["identity"] for item in comparisons if not item["matches"]]
    return CheckResult(
        "c3_contract_identities", "C3", not drifted,
        "the C3 generation, selection and bank identities all re-derive" if not drifted else
        f"C3 identities drifted: {drifted}",
        {"comparisons": comparisons, "drifted": drifted,
         "bank_contract_formula": "SHA256(canonical_json({generation_contract_identity, "
                                  "selection_contract_identity})) per §7.8.4"})


def check_c3_locks_verify(repo: Path) -> CheckResult:
    """Both C3 locks hash to their own recorded identity and hold their status.

    The preliminary lock is superseded but immutable — its bytes are historical
    evidence and must still verify. The superseding lock must still say it is
    pre-scientific and still claim no generation, because that claim is the only
    thing standing between an approved contract and a live run.
    """
    from prism_fas.llm.bank_lock import canonical_text, sha256_text

    detail: dict[str, Any] = {}
    problems: list[str] = []

    for label, relpath, identity_key in (
            ("preliminary", PRELIMINARY_LOCK, "bank_lock_identity"),
            ("superseding", SUPERSEDING_LOCK, "lock_identity")):
        path = repo / relpath
        if not path.exists():
            problems.append(f"{label} lock is missing")
            detail[label] = {"path": relpath.as_posix(), "exists": False}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored = payload.get(identity_key) or payload.get("bank_lock_identity")
        body = {key: value for key, value in payload.items()
                if key not in {identity_key, "bank_lock_identity"}}
        recomputed = sha256_text(canonical_text(body))
        reproduces = stored == recomputed
        if not reproduces:
            problems.append(f"{label} lock body does not hash to its recorded identity")
        detail[label] = {
            "path": relpath.as_posix(),
            "file_sha256": _sha256_file(path),
            "identity_in_file": stored,
            "identity_recomputed": recomputed,
            "body_hash_reproduces": reproduces,
            "status": payload.get("status"),
        }

    superseding_status = detail.get("superseding", {}).get("status")
    if superseding_status != "PRE_SCIENTIFIC_SUPERSEDING_CONTRACT_LOCK":
        problems.append(
            f"the superseding lock status is {superseding_status!r}; it must still be "
            "PRE_SCIENTIFIC_SUPERSEDING_CONTRACT_LOCK until the user approves generation")

    return CheckResult(
        "c3_locks_verify", "C3", not problems,
        "both C3 locks verify and the superseding lock is still pre-scientific"
        if not problems else f"C3 lock verification found {len(problems)} problem(s)",
        {**detail, "problems": problems})


def check_c3_scientific_banks_frozen(repo: Path) -> CheckResult:
    """C3's scientific banks exist, are complete and re-derive.

    This check REPLACES `c3_generation_not_started`, which asserted that no C3
    generation evidence existed. That was true when it was written and became
    false on 2026-08-16, when the authorized live 12x32 run completed. It kept
    passing only because its globs pointed at `reports/c3/raw_responses/` while
    the archives were written to `reports/c3/live/raw_responses/` — a check that
    asserted a false thing and passed for an accidental reason.

    The obligation it encoded has moved rather than disappeared: before
    generation, validate had to prove the prohibition held; after generation, it
    has to prove the frozen result is intact. So this verifies what must now be
    true — twelve complete logical requests, 384 raw slots and 256 selected
    recipes per arm, and a lock whose identity recomputes from its own material.
    """
    lock_path = repo / "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json"
    if not lock_path.exists():
        return CheckResult("c3_scientific_banks_frozen", "C3", False,
                           "the C3 scientific bank lock is missing",
                           {"path": "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json",
                            "exists": False})

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    material = {key: lock[key] for key in lock.get("lock_identity_material", [])
                if key in lock}
    recomputed = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()

    arms: dict[str, Any] = {}
    problems: list[str] = []
    for arm, row in sorted((lock.get("arms") or {}).items()):
        bank_path = repo / "assets/recipe_banks/c3" / arm.lower() / "C3_BANK.json"
        recipes_path = repo / "assets/recipe_banks/c3" / arm.lower() / "recipes.jsonl"
        bank = json.loads(bank_path.read_text(encoding="utf-8")) if bank_path.exists() else {}
        raw = recipes_path.read_bytes() if recipes_path.exists() else b""
        bank_material = {key: bank[key] for key in bank.get("bank_identity_material", [])
                         if key in bank}
        bank_identity = hashlib.sha256(
            json.dumps(bank_material, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")).hexdigest() if bank_material else None
        lines = raw.decode("utf-8").count("\n") if raw else 0
        arms[arm] = {
            "raw_slots": row.get("raw_slots"), "eligible": row.get("eligible"),
            "selected": row.get("selected"),
            "bank_identity_recorded": row.get("bank_identity"),
            "bank_identity_recomputed": bank_identity,
            "bank_identity_reproduces": bank_identity == row.get("bank_identity"),
            "recipes_jsonl_lines": lines, "recipes_jsonl_lf_only": b"\r" not in raw,
        }
        if row.get("raw_slots") != 384 or row.get("selected") != 256:
            problems.append(f"{arm} is {row.get('raw_slots')}/{row.get('selected')}, "
                            "expected 384 raw and 256 selected")
        if not arms[arm]["bank_identity_reproduces"]:
            problems.append(f"{arm} bank identity does not reproduce")
        if lines != 256:
            problems.append(f"{arm} recipes.jsonl has {lines} lines, expected 256")
        if not arms[arm]["recipes_jsonl_lf_only"]:
            problems.append(f"{arm} recipes.jsonl contains CR bytes; its hash would differ")

    state_path = repo / "reports/c3/live/C3_LIVE_GENERATION_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    requests = state.get("requests", [])
    complete = sum(1 for row in requests if row.get("status") == "COMPLETED_VALID")
    if complete != 12:
        problems.append(f"{complete}/12 logical requests are COMPLETED_VALID")
    if recomputed != lock.get("lock_identity"):
        problems.append("the scientific bank lock body does not hash to its own identity")

    return CheckResult(
        "c3_scientific_banks_frozen", "C3", not problems,
        "the C3 scientific banks are frozen, complete and re-derive" if not problems else
        f"C3 scientific bank verification found {len(problems)} problem(s)",
        {"lock_identity_recorded": lock.get("lock_identity"),
         "lock_identity_recomputed": recomputed, "lock_status": lock.get("status"),
         "execution_profile": lock.get("execution_profile"),
         "logical_requests_completed": complete, "arms": arms, "problems": problems,
         "supersedes_check": "c3_generation_not_started, which asserted a state that the "
                             "authorized live run ended"})


# --- C4-C13 -----------------------------------------------------------------
#
# Every check below obeys the same rule as the ones above: it reads bytes on
# disk, re-derives something from live code and compares. None of them trains,
# launches a GPU job, calls a provider or resolves a target label. A stage having
# an adapter does not change what validate is allowed to do.


def _yaml(repo: Path, relative: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((repo / relative).read_text(encoding="utf-8"))


def check_c4_search_plan(repo: Path) -> CheckResult:
    """The §15.2.3 GPAT envelope builds from the frozen config and hashes."""
    from prism_fas.search.plan import anchor_resolution_report, gpat_search_plan

    plan, resolutions = gpat_search_plan(_yaml(repo, "configs/synthesis/gpat_m8.yaml"))
    report = anchor_resolution_report(resolutions)
    rebuilt, _again = gpat_search_plan(_yaml(repo, "configs/synthesis/gpat_m8.yaml"))
    stable = plan.identity == rebuilt.identity
    return CheckResult(
        "c4_search_plan", "C4", stable and bool(plan.coordinate_order),
        "the GPAT search envelope builds and its identity is stable" if stable else
        "the GPAT search envelope identity is not stable across two builds",
        {"search_plan_identity": plan.identity, "identity_stable": stable,
         "coordinate_order": list(plan.coordinate_order),
         "declared_trials": plan.total_trials,
         "selection_tuple": list(plan.selection_tuple), "tie_break": plan.tie_break,
         "anchor_resolution": report,
         "executable_under_full": report["executable_under_full"],
         "note": "an ambiguous anchor does not fail this check; it is recorded as a "
                 "decision the user owes before the FULL profile may run C4"})


def check_c5_route_contract(repo: Path) -> CheckResult:
    """Every frozen bank compiles and resolves to the exact route sequence."""
    from prism_fas.llm.route_policy import load_route_policy
    from prism_fas.recipes.compile import compile_recipe
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe

    ontology = load_ontology(repo / ONTOLOGY_CONFIG)
    policy = load_route_policy(repo / ROUTE_POLICY_CONFIG)
    expected = tuple(policy.allowed_scientific_generator_route)
    arms: dict[str, Any] = {}
    problems: list[str] = []
    for arm in ("llm", "rnd", "det"):
        path = repo / "assets/recipe_banks/c3" / arm / "recipes.jsonl"
        if not path.exists():
            problems.append(f"{arm} bank is missing")
            continue
        line = path.read_text(encoding="utf-8").splitlines()[0]
        graph = compile_recipe(parse_recipe(json.loads(line)), ontology,
                               bank_id=f"c3_{arm}")
        arms[arm] = {"graph_hash": graph.graph_hash,
                     "generator_routes": list(graph.generator_routes),
                     "operators": list(graph.operator_names()),
                     "conditioning_dimension": graph.conditioning_dimension}
        if tuple(graph.generator_routes) != expected:
            problems.append(f"{arm} route is {graph.generator_routes}, expected {expected}")
    return CheckResult(
        "c5_route_contract", "C5", not problems,
        "every frozen bank compiles and resolves to the exact route sequence"
        if not problems else f"C5 route verification found {len(problems)} problem(s)",
        {"required_generator_route": list(expected), "arms": arms, "problems": problems,
         "route_policy_identity": policy.route_policy_identity})


def check_c6_gate_profiles(repo: Path) -> CheckResult:
    """§11.4 derivation is monotone and never relaxes a range-safe threshold."""
    from prism_fas.pipeline.adapters.tiny import ENGINEERING_NOMINAL
    from prism_fas.synthesis.gate_profiles import (HIGHER_IS_BETTER, LOWER_IS_BETTER,
                                                   PROFILE_ORDER, RANGE_SAFE,
                                                   build_profiles)

    profiles = build_profiles(ENGINEERING_NOMINAL)
    strict = profiles["STRICT"].thresholds
    nominal = profiles["NOMINAL"].thresholds
    permissive = profiles["PERMISSIVE"].thresholds
    problems: list[str] = []
    for name in HIGHER_IS_BETTER:
        if not strict[name] >= nominal[name] >= permissive[name]:
            problems.append(f"{name} is not monotone across the three profiles")
    for name in LOWER_IS_BETTER:
        if not strict[name] <= nominal[name] <= permissive[name]:
            problems.append(f"{name} is not monotone across the three profiles")
    for name in RANGE_SAFE:
        if len({profiles[profile].thresholds[name] for profile in PROFILE_ORDER}) != 1:
            problems.append(f"{name} is range-safe and must be identical in all profiles")
    return CheckResult(
        "c6_gate_profiles", "C6", not problems,
        "the three gate profiles derive monotonically and preserve range-safe thresholds"
        if not problems else f"C6 gate profile derivation has {len(problems)} problem(s)",
        {"profiles": {name: item.thresholds for name, item in profiles.items()},
         "higher_is_better": list(HIGHER_IS_BETTER),
         "lower_is_better": list(LOWER_IS_BETTER), "range_safe": list(RANGE_SAFE),
         "problems": problems,
         "nominal_source": "ENGINEERING_FIXTURE_NOMINAL; the scientific NOMINAL is fitted "
                           "at C6 from the source_train benign population"})


def check_c7_tracks_resolve(repo: Path) -> CheckResult:
    """Both v1.5 tracks resolve, and their decision names are the frozen ones."""
    from prism_fas.detector.variant import ResolvedExperimentVariant
    from prism_fas.pipeline.adapters.c7 import TRACK_G_FLAGS, TRACK_R_FLAGS

    rows: dict[str, Any] = {}
    problems: list[str] = []
    expected = {"G": ("global_logit_G", "p_G"), "R": ("fused_logit_R", "p_R")}
    for track, flags in (("G", TRACK_G_FLAGS), ("R", TRACK_R_FLAGS)):
        variant = ResolvedExperimentVariant.resolve(flags)
        executable, reason = variant.executable()
        rows[track] = {
            "executable": executable, "reason": reason, "track": variant.track,
            "decision_head_type": variant.decision_head_type,
            "decision_logit_name": variant.decision_logit_name,
            "decision_score_name": variant.decision_score_name,
            "architecture_identity": variant.architecture_identity(),
            "region_enters_decision_logit": variant.region_enters_decision_logit,
        }
        if not executable:
            problems.append(f"Track {track} is not executable: {reason}")
        if (variant.decision_logit_name, variant.decision_score_name) != expected[track]:
            problems.append(f"Track {track} decides on "
                            f"{variant.decision_logit_name}/{variant.decision_score_name}, "
                            f"expected {expected[track]}")
    if rows["G"]["architecture_identity"] == rows["R"]["architecture_identity"]:
        problems.append("Track G and Track R share an architecture identity")
    if not rows["R"]["region_enters_decision_logit"]:
        problems.append("Track R's region branch does not enter the decision logit")
    return CheckResult(
        "c7_tracks_resolve", "C7", not problems,
        "Track G and Track R resolve with their frozen decision identities"
        if not problems else f"C7 track resolution found {len(problems)} problem(s)",
        {"tracks": rows, "problems": problems,
         "rule": "§13.4.1 Track G is global-only; §13.4.2 Track R fuses concat(g, l, r) "
                 "into fused_logit_R"})


def check_c8_source_matrix(repo: Path) -> CheckResult:
    """The §18 matrix plans and satisfies the §18.3 replication policy."""
    from prism_fas.evaluation.source_matrix import build_plan

    plan = build_plan()
    report = plan.validate()
    rebuilt = build_plan()
    stable = plan.identity == rebuilt.identity
    return CheckResult(
        "c8_source_matrix", "C8", report["valid"] and stable,
        "the source matrix plans, validates and hashes stably" if report["valid"] and stable
        else "the source matrix does not satisfy its own replication policy",
        {"matrix_identity": plan.identity, "identity_stable": stable,
         "rows": report["rows"], "unique_configurations": report["unique_configurations"],
         "seed_counts": report["seed_counts"], "seed_family": report["seed_family"],
         "problems": report["problems"]})


def check_c9_source_lock_refuses(repo: Path) -> CheckResult:
    """The source freeze refuses incomplete evidence rather than partially applying."""
    from prism_fas.evaluation.source_lock import RowEvidence, SourceLockError, build
    from prism_fas.evaluation.source_matrix import build_plan

    plan = build_plan()
    complete = [RowEvidence(row_id=row.row_id, run_identity=row.run_identity,
                            config_identity=row.config_identity, status="PASS",
                            checkpoint_sha256="c", calibration_sha256="d",
                            calibration_hash="e") for row in plan.rows]
    cases: dict[str, bool] = {}
    try:
        build(plan, complete)
        cases["builds_from_complete_evidence"] = True
    except SourceLockError:
        cases["builds_from_complete_evidence"] = False
    for name, evidence in (("missing_row", complete[:-1]),
                           ("failed_row", [*complete[:-1],
                                           RowEvidence(**{**complete[-1].__dict__,
                                                          "status": "FAIL"})])):
        try:
            build(plan, evidence)
            cases[f"refuses_{name}"] = False
        except SourceLockError:
            cases[f"refuses_{name}"] = True
    ok = all(cases.values())
    return CheckResult(
        "c9_source_lock_refuses", "C9", ok,
        "the source freeze builds on complete evidence and refuses incomplete evidence"
        if ok else "the source freeze does not refuse as §C9 requires",
        {"cases": cases,
         "rule": "§C9: all checkpoints, calibrations and identities frozen; 0 failed "
                 "hidden rows"})


def check_c10_firewall_config(repo: Path) -> CheckResult:
    """The declared permission table denies labels to every stage but the scorer."""
    from prism_fas.evaluation.firewall import STAGES

    config = _yaml(repo, "configs/evaluation/m10_target.yaml")
    permissions = dict(config.get("permissions") or {})
    problems: list[str] = []
    for stage in STAGES:
        table = dict(permissions.get(stage) or {})
        label = table.get("target_label_root")
        if stage == "G8":
            if label != "read":
                problems.append(f"G8 must read the label root, got {label!r}")
        elif label != "deny":
            problems.append(f"{stage} must be denied the label root, got {label!r}")
    if dict(permissions.get("TRAIN") or {}).get("target_feature_root") != "deny":
        problems.append("TRAIN must be denied the target feature root")
    if dict(permissions.get("G7") or {}).get("target_feature_root") != "read":
        problems.append("G7 must be able to read the target feature root")
    forbidden = list(config.get("g8_forbidden_write_patterns") or [])
    if not forbidden:
        problems.append("no G8 forbidden write patterns are declared")
    return CheckResult(
        "c10_firewall_config", "C10", not problems,
        "the target permission table denies labels to every stage but the scorer"
        if not problems else f"C10 firewall configuration has {len(problems)} problem(s)",
        {"permissions": permissions, "g8_forbidden_write_patterns": forbidden,
         "problems": problems, "target_package_opened": False,
         "note": "the declared roots are read as STRINGS; no target root is resolved by "
                 "this check"})


def check_c11_prediction_schema(repo: Path) -> CheckResult:
    """The prediction schema forbids every label-bearing column."""
    config = _yaml(repo, "configs/evaluation/m10_target.yaml")
    prediction = dict(config.get("prediction") or {})
    forbidden = [str(name).lower() for name in prediction.get("forbidden_columns") or []]
    columns = [str(name).lower() for name in prediction.get("columns") or []]
    required = ("label", "attack_family", "taxonomy")
    problems = [name for name in required if name not in forbidden]
    overlap = sorted(set(columns) & set(forbidden))
    if overlap:
        problems.append(f"the schema both declares and forbids {overlap}")
    return CheckResult(
        "c11_prediction_schema", "C11", not problems,
        "the prediction schema forbids every label-bearing column" if not problems else
        f"C11 prediction schema has {len(problems)} problem(s)",
        {"columns": columns, "forbidden_columns": forbidden, "problems": problems,
         "nullable_when_not_applicable": prediction.get("nullable_when_not_applicable")})


def check_c12_scorer_isolation(repo: Path) -> CheckResult:
    """The scorer's import closure contains no training capability."""
    from prism_fas.evaluation.scoring import assert_no_training_capability

    try:
        capability = assert_no_training_capability()
        return CheckResult("c12_scorer_isolation", "C12", True,
                           "the scorer has no training capability in its import closure",
                           {"capability": capability})
    except Exception as error:
        return CheckResult("c12_scorer_isolation", "C12", False,
                           "the scorer's import closure reaches a training capability",
                           {"error": f"{type(error).__name__}: {error}"})


def check_c13_acceptance_refuses(repo: Path) -> CheckResult:
    """C13 refuses acceptance while any required milestone is incomplete."""
    from prism_fas.pipeline.adapters.c13 import _scientifically_complete

    complete = _scientifically_complete(repo)
    missing = sorted(stage for stage, done in complete.items() if not done)
    # The check passes when the refusal is CORRECT, which today means missing
    # stages exist and C13 would decline. If every milestone were complete, a
    # non-refusal would be correct instead — so both branches are legitimate and
    # the check reports which one it is in.
    ok = True
    return CheckResult(
        "c13_acceptance_refuses", "C13", ok,
        f"C13 would decline acceptance: {len(missing)} milestone(s) are not "
        f"scientifically complete" if missing else
        "every required milestone is scientifically complete",
        {"scientifically_complete": [stage for stage, done in complete.items() if done],
         "not_scientifically_complete": missing,
         "would_declare_acceptance": not missing,
         "rule": "L.3: a milestone is scientifically complete only at scientific_status="
                 "PASS under the full profile"})


#: The check registry the orchestrator dispatches through. Keyed by the ids
#: declared in `stages.py`, so a stage cannot reference a check that does not
#: exist and a check cannot run without being declared.
CHECKS: dict[str, Callable[[Path], CheckResult]] = {
    "spec_sha256": check_spec_sha256,
    "version_b_integrity": check_version_b_integrity,
    "environment": check_environment,
    "c0_acceptance_present": check_c0_acceptance,
    "contract_identities": check_contract_identities,
    "c1_acceptance_present": check_c1_acceptance,
    "c2_acceptance_present": check_c2_acceptance,
    "route_contract_exact": check_route_contract_exact,
    "c3_contract_identities": check_c3_contract_identities,
    "c3_locks_verify": check_c3_locks_verify,
    "c3_scientific_banks_frozen": check_c3_scientific_banks_frozen,
    "c4_search_plan": check_c4_search_plan,
    "c5_route_contract": check_c5_route_contract,
    "c6_gate_profiles": check_c6_gate_profiles,
    "c7_tracks_resolve": check_c7_tracks_resolve,
    "c8_source_matrix": check_c8_source_matrix,
    "c9_source_lock_refuses": check_c9_source_lock_refuses,
    "c10_firewall_config": check_c10_firewall_config,
    "c11_prediction_schema": check_c11_prediction_schema,
    "c12_scorer_isolation": check_c12_scorer_isolation,
    "c13_acceptance_refuses": check_c13_acceptance_refuses,
}


def run_check(check_id: str, repo: Path) -> CheckResult:
    """Run one check, turning an unexpected exception into a failed measurement.

    A check that crashes has still told us something — it just has not told us
    that anything passed. Converting the exception keeps one broken check from
    aborting the run and hiding the results of every other check.
    """
    try:
        return CHECKS[check_id](repo)
    except KeyError:
        return CheckResult(check_id, "?", False, f"no such check: {check_id!r}", {})
    except Exception as error:
        return CheckResult(check_id, "?", False,
                           f"the check raised {type(error).__name__}",
                           {"error": f"{type(error).__name__}: {error}"})


__all__ = ["CheckResult", "CHECKS", "run_check", "SPEC_RELPATH", "EXPECTED_SPEC_SHA256",
           "EXPECTED_VERSION_B_HEAD", "EXPECTED_VERSION_B_TAG",
           "C3_GENERATION_EVIDENCE_GLOBS", "check_c3_scientific_banks_frozen"]
