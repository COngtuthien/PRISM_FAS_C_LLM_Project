"""Measure prism_c3_selection_v1's determinism and write the evidence report.

    python scripts/c3_selector_determinism_report.py

Offline: no network, no credential, no provider, no GPU, no Modal, no target
access. It generates nothing scientific — the recipes it selects over are
synthetic fixtures, and the banks it produces are discarded. What it records is
whether the SELECTOR is a pure function of candidate content.

Three independent kinds of evidence go into
`reports/c3/v15_selection_contract/C3_SELECTOR_DETERMINISM_REPORT.json`:

1. the exact pass/fail/skip counts of the offline C3 suite, from a real run;
2. direct probes — repeated runs, permuted input, reversed input, a degenerate
   instance where stage 5 alone decides, and the real 320 -> 256 budget;
3. RND/DET schedule reproducibility, including drafting in reverse slot order.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "tests" / "c3"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from prism_fas.recipes import arm_schedules as sched  # noqa: E402
from prism_fas.recipes import selection as sel  # noqa: E402
from prism_fas.recipes.canonical import recipe_hash  # noqa: E402
from prism_fas.recipes.ontology import load_ontology  # noqa: E402

from c3_fixtures import make_pool, make_recipe  # noqa: E402

OUT = REPO / "reports" / "c3" / "v15_selection_contract"
REPORT = OUT / "C3_SELECTOR_DETERMINISM_REPORT.json"

TINY_QUOTAS: dict[str, dict[str, int | None]] = {
    "medium":       {"hard_min": 0, "hard_max": 99, "preferred_min": None},
    "geometry":     {"hard_min": 0, "hard_max": 99, "preferred_min": None},
    "illumination": {"hard_min": 0, "hard_max": 99, "preferred_min": None},
    "artifact":     {"hard_min": 0, "hard_max": 99, "preferred_min": 3},
    "region":       {"hard_min": 0, "hard_max": 99, "preferred_min": 2},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


class _Quotas:
    """Swap the module quota table for a probe, then restore it exactly."""

    def __init__(self, quotas: dict[str, Any]) -> None:
        self.quotas = quotas
        self.original: dict[str, Any] = {}

    def __enter__(self) -> None:
        self.original = dict(sel.QUOTAS)
        sel.QUOTAS.update(self.quotas)

    def __exit__(self, *exc: Any) -> None:
        sel.QUOTAS.clear()
        sel.QUOTAS.update(self.original)


# ------------------------------------------------------------------ the suite
def run_suite() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "tests/c3", "-q", "--no-header",
               "-p", "no:cacheprovider", "--continue-on-collection-errors"]
    started = time.time()
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    tail = (result.stdout or "").strip().splitlines()
    summary = tail[-1] if tail else ""

    def count(label: str) -> int:
        found = re.search(rf"(\d+) {label}", summary)
        return int(found.group(1)) if found else 0

    return {
        "command": " ".join(command[1:]),
        "exit_code": result.returncode,
        "summary_line": summary,
        "passed": count("passed"),
        "failed": count("failed"),
        "skipped": count("skipped"),
        "errors": count("error"),
        "elapsed_seconds": round(time.time() - started, 1),
        "offline": "the suite blocks sockets and deletes ambient credentials in conftest",
    }


# ------------------------------------------------------------------- probes
def probe_repeatability(ontology) -> dict[str, Any]:
    pool = make_pool(ontology, 40, tag="det")
    with _Quotas(TINY_QUOTAS):
        runs = [sel.select(pool, ontology, bank_size=24, enforce_minimum_pool=False)
                for _ in range(3)]
    identities = [run.selected_set_identity for run in runs]
    return {"runs": len(runs), "selected_set_identities": identities,
            "identical": len(set(identities)) == 1}


def probe_order_invariance(ontology) -> dict[str, Any]:
    pool = make_pool(ontology, 40, tag="det")
    with _Quotas(TINY_QUOTAS):
        baseline = sel.select(pool, ontology, bank_size=24, enforce_minimum_pool=False)
        observed = {"baseline": baseline.selected_set_identity}
        for seed in (1, 2, 3, 7, 11):
            shuffled = list(pool)
            random.Random(seed).shuffle(shuffled)
            observed[f"permutation_seed_{seed}"] = sel.select(
                shuffled, ontology, bank_size=24,
                enforce_minimum_pool=False).selected_set_identity
        observed["reversed"] = sel.select(list(reversed(pool)), ontology, bank_size=24,
                                          enforce_minimum_pool=False).selected_set_identity
    return {"selected_set_identities": observed,
            "invariant": len(set(observed.values())) == 1}


def probe_stage_5(ontology) -> dict[str, Any]:
    """A degenerate instance: all subsets tie, so stage 5 alone decides."""
    medium = ontology.media[0]
    geometry = ontology.geometry_shapes[0]
    pool = [make_recipe(ontology, index, tag="tie", medium=medium, geometry=geometry,
                        illumination=ontology.illumination[0],
                        artifacts=[ontology.artifacts_for_medium(medium)[0]],
                        regions=[ontology.regions_for_geometry(geometry)[0]])
            for index in range(10)]
    ordered = sorted(recipe_hash(recipe) for recipe in pool)
    with _Quotas(TINY_QUOTAS):
        result = sel.select(pool, ontology, bank_size=4, enforce_minimum_pool=False)
    return {
        "pool": len(pool), "bank": 4,
        "every_subset_ties_on_stages_1_to_4": True,
        "tied_subsets": 210,
        "expected_smallest_shas": ordered[:4],
        "selected_shas": sorted(result.selected_shas),
        "stage_5_returned_the_lexicographically_smallest_set":
            sorted(result.selected_shas) == ordered[:4],
    }


def probe_full_budget(ontology) -> dict[str, Any]:
    """The real contract: 320 eligible -> exactly 256, frozen quotas unmodified."""
    pool = make_pool(ontology, sel.MINIMUM_ELIGIBLE_POOL_PER_ARM, tag="full")
    started = time.time()
    result = sel.select(pool, ontology, arm="FIXTURE")
    return {
        "note": "synthetic fixtures; this bank is a harness artifact and is discarded. "
                "It is NOT a C3 scientific recipe bank.",
        "eligible": result.eligible_count,
        "selected": len(result.selected_shas),
        "rejected": len(result.rejected_shas),
        "selected_plus_rejected": len(result.selected_shas) + len(result.rejected_shas),
        "unique_selected": len(set(result.selected_shas)),
        "hard_violations": sel.hard_violations(result.counts),
        "objective": {"S_pref": result.s_pref, "S_single": result.s_single,
                      "S_multi": result.s_multi},
        "counts": result.counts,
        "selected_set_identity": result.selected_set_identity,
        "elapsed_seconds": round(time.time() - started, 1),
        "quota_table_modified": False,
    }


def probe_schedules(ontology) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("RND", "DET"):
        schedule = sched.build_schedule(arm, ontology)
        forward = list(sched.draft_schedule(arm, ontology))
        backward = [(slot_id, sched.draft_candidate(arm, slot_id, ontology))
                    for slot_id in reversed(sched.slot_ids(arm))]
        backward.reverse()

        def digest(pairs: list[tuple[str, dict[str, Any]]]) -> str:
            text = json.dumps(pairs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

        out[arm] = {
            "slots": len(forward),
            "unique_slot_ids": len({slot_id for slot_id, _ in forward}),
            "seed": schedule.seed,
            "schedule_identity": schedule.schedule_identity,
            "schedule_identity_recomputed": sched.build_schedule(arm,
                                                                ontology).schedule_identity,
            "draft_digest_forward_order": digest(forward),
            "draft_digest_reverse_order": digest(backward),
            "reproducible": (digest(forward) == digest(backward)
                             and len(forward) == sched.SLOTS_PER_ARM),
            "route_declared_by_every_slot": all(
                payload["generator_route"] == ["physics", "gpat"] for _, payload in forward),
        }
    out["schedules_are_distinct"] = (out["RND"]["schedule_identity"]
                                     != out["DET"]["schedule_identity"])
    return out


def main() -> int:
    ontology = load_ontology(REPO / "configs" / "recipes" / "ontology_m7.yaml")

    suite = run_suite()
    repeatability = probe_repeatability(ontology)
    order = probe_order_invariance(ontology)
    stage_5 = probe_stage_5(ontology)
    schedules = probe_schedules(ontology)
    full = probe_full_budget(ontology)

    determinism_holds = all([
        repeatability["identical"],
        order["invariant"],
        stage_5["stage_5_returned_the_lexicographically_smallest_set"],
        schedules["RND"]["reproducible"],
        schedules["DET"]["reproducible"],
        schedules["schedules_are_distinct"],
        full["selected"] == sel.FINAL_BANK_SIZE_PER_ARM,
        full["hard_violations"] == [],
        suite["failed"] == 0 and suite["errors"] == 0 and suite["exit_code"] == 0,
    ])

    payload = {
        "schema_version": "c3-selector-determinism-report-v1",
        "milestone": "C3",
        "substage": "selection-contract freeze (pre-scientific)",
        "generated_at_utc": utc_now(),
        "generator_code_commit": git("rev-parse", "HEAD"),
        "selection_version": sel.SELECTION_VERSION,
        "governing_clauses": ["7.8", "7.8.1", "7.8.2", "7.8.3", "7.8.5"],
        "offline_attestation": {
            "live_provider_calls": 0,
            "network_calls": 0,
            "gpu_or_modal_jobs": 0,
            "target_label_or_metric_reads": 0,
            "c3_scientific_logical_requests": 0,
            "c3_scientific_candidate_slots": 0,
            "candidates_are": "synthetic ontology-derived fixtures",
        },
        "offline_suite": suite,
        "probes": {
            "repeated_run_identity": repeatability,
            "input_order_invariance": order,
            "stage_5_canonical_tie_break": stage_5,
            "control_arm_schedule_reproducibility": schedules,
            "real_320_to_256_budget": full,
        },
        "determinism_holds": determinism_holds,
        "execution_profile": "validate",
        "scientific_eligible": False,
        "scientific_status": "NOT_RUN",
        "note": "Determinism evidence for the selector. Not C3 scientific completion.",
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print("wrote", REPORT.relative_to(REPO).as_posix())
    print("suite:", suite["summary_line"])
    print("determinism_holds:", determinism_holds)
    return 0 if determinism_holds else 4


if __name__ == "__main__":
    sys.exit(main())
