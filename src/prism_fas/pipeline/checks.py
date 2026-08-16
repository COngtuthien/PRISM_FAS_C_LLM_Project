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


def check_c3_generation_not_started(repo: Path) -> CheckResult:
    """No C3 scientific generation has occurred. Measured, not assumed.

    C3 generation is prohibited until explicitly authorized, so validate proves
    the prohibition still holds by looking for the artifacts a generation run
    would necessarily have written. Finding any of them is a STOP, not a note.
    """
    found: list[str] = []
    for pattern in C3_GENERATION_EVIDENCE_GLOBS:
        found.extend(sorted(str(path.relative_to(repo).as_posix())
                            for path in repo.glob(pattern) if path.is_file()))
    ok = not found
    return CheckResult(
        "c3_generation_not_started", "C3", ok,
        "no C3 generation evidence exists; the prohibition still holds" if ok else
        "C3 generation evidence is present; generation is NOT in the state this "
        "check expects",
        {"patterns": list(C3_GENERATION_EVIDENCE_GLOBS), "found": found,
         "c3_scientific_requests_expected": 0})


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
    "c3_generation_not_started": check_c3_generation_not_started,
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
           "C3_GENERATION_EVIDENCE_GLOBS"]
