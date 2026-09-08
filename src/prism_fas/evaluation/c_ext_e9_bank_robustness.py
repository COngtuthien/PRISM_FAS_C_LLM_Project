"""PRISM-FAS-C EXT-Q1Q2 -- E9: conditional bank-selection robustness.

Section 16 (E9) of PRISM_FAS_C_EXT_Q1Q2_Detailed_Spec_v1_0.docx.

Scope
-----
E9 measures ONLY the sensitivity of the exact frozen 256-recipe bank selection
to deterministic perturbations of each arm's frozen 384-candidate pool. For every
arm in {RND, DET, LLM} and every perturbation realization in {P1, P2, P3} it:

  1. deterministically masks a fixed fraction of the 384-candidate pool using a
     versioned SHA-256 rule (NO Python RNG),
  2. re-runs the EXACT existing MILP selector
     (``prism_fas.recipes.selection.select``, ``prism_c3_selection_v1``) on the
     retained pool,
  3. records the selected 256-recipe bank plus the selector's own optimization
     trace, and
  4. computes descriptive stability metrics against the historical selected-256
     reference bank.

Hard rules honoured by this module
----------------------------------
* The selector is REUSED, never reimplemented. There is exactly one call to
  ``prism_fas.recipes.selection.select`` and no alternative objective / solver /
  constraint / bank-size code path anywhere in this file.
* Bank size is never lowered. A realization that cannot yield a valid
  256-recipe bank is terminal ``BLOCKED`` with the selector's failure reason.
* No LLM call, no network, no GPU, no detector training.
* No target label / target prediction / target metric / E8 target score is read.
  This module imports only the stdlib, ``prism_fas.recipes.*`` (dataset-agnostic
  recipe schema / ontology / canonical hash / selector) and
  ``prism_fas.evaluation.c_ext_common``.
* Every write is routed through ``c_ext_common.assert_ext_write_path`` restricted
  to ``reports/c_ext_q1q2_v1/e9_bank_robustness/``; no Flow-1 / Flow-2 / C3
  artifact can be touched.

Protocol amendment (additive, frozen before any E9 outcome)
----------------------------------------------------------
The governing v1.0 document fixes the three perturbation seeds but does not
numerically fix the retained fraction. This module freezes, purely
combinatorially:

    original_pool_count = 384
    retained_count      = 320       (retained_fraction = 320/384 = 5/6)
    masked_count        =  64
    selected_count      = 256       (=> exactly 64 candidates of MILP slack)

identical for RND, DET and LLM. The value was not chosen from any E8 ACER,
target label, target prediction, target metric or E9 result -- see
``E9_PROTOCOL_AMENDMENT.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism_fas.evaluation.c_ext_common import (  # noqa: E402
    canonical_json_bytes, repo_root, sha256_bytes, sha256_file, sha256_json,
    write_json_atomic, write_text_atomic,
)

# --------------------------------------------------------------------------- #
# Frozen constants -- never change without a new dated amendment
# --------------------------------------------------------------------------- #

E9_ID = "e9_bank_robustness"
OUTPUT_SUBTREE = f"reports/c_ext_q1q2_v1/{E9_ID}"

#: The exact base commit this milestone is pinned to (audit reference only).
#: The frozen PARENT commit E9 development began from. This value is permanent
#: and is NOT required to equal HEAD at scientific-execution time -- the reviewed
#: E9 implementation is expected to land as a CHILD of this commit. Preflight
#: proves this commit is an ancestor of HEAD (E9 was derived from it); the
#: separate E9_EXECUTION_BINDING.json pins the exact implementation commit that
#: actually runs the nine selections.
E9_BASE_COMMIT = "6f0642a1d05c35e4c1778d329fb547f1b22a4115"

#: This module and its test, repo-relative, so E9 can prove its own scientific
#: implementation is committed and unmodified before it runs any selection.
E9_MODULE_RELPATH = "src/prism_fas/evaluation/c_ext_e9_bank_robustness.py"
E9_TEST_RELPATH = "tests/pipeline/test_c_ext_e9_bank_robustness.py"

#: The frozen selector implementation file and its historically-proven sha256
#: (re-verified here: select() reproduced all three C3 selected_set_identity
#: values from the committed 256-recipe banks).
SELECTOR_MODULE_RELPATH = "src/prism_fas/recipes/selection.py"
SELECTOR_MODULE_SHA256 = "084b3d94237f2b2e7583676a46c7560e8de6fa30a93b5cd43961630e0bb883a1"

#: Protected historical artifacts that must be byte-identical to HEAD before any
#: E9 scientific selection runs (fail-closed list).
PROTECTED_HISTORICAL_RELPATHS: tuple[str, ...] = (
    "src/prism_fas/recipes/selection.py",
    "assets/recipe_banks/c3/rnd/recipes.jsonl",
    "assets/recipe_banks/c3/det/recipes.jsonl",
    "assets/recipe_banks/c3/llm/recipes.jsonl",
    "assets/recipe_banks/c3/rnd/C3_BANK.json",
    "assets/recipe_banks/c3/det/C3_BANK.json",
    "assets/recipe_banks/c3/llm/C3_BANK.json",
    "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json",
    "reports/c3/v15_selection_contract/C3_RND_SCHEDULE_CONTRACT.json",
    "reports/c3/v15_selection_contract/C3_DET_SCHEDULE_CONTRACT.json",
    "configs/recipes/ontology_m7.yaml",
)

EXECUTION_BINDING_RELPATH = f"{OUTPUT_SUBTREE}/E9_EXECUTION_BINDING.json"
CLOSURE_RELPATH = f"{OUTPUT_SUBTREE}/E9_CLOSURE.json"

#: Reported implementation-source states of the E9 module file.
IMPL_UNCOMMITTED_REVIEW_STATE = "UNCOMMITTED_REVIEW_STATE"   # module untracked
IMPL_DIRTY_TRACKED = "DIRTY_TRACKED"                         # tracked, differs from HEAD
IMPL_COMMITTED_CLEAN = "COMMITTED_CLEAN"                     # tracked, byte-identical to HEAD

#: Versioned perturbation-string tag. Bytes are
#: ``E9-POOL-PERTURB-v1|<arm>|<seed>|<candidate_id>`` UTF-8, then SHA-256.
PERTURB_STRING_VERSION = "E9-POOL-PERTURB-v1"

ARMS: tuple[str, ...] = ("RND", "DET", "LLM")

#: Perturbation realizations and their frozen seeds (Section 16).
PERTURBATIONS: dict[str, int] = {
    "P1": 20260921,
    "P2": 20260922,
    "P3": 20260923,
}

ORIGINAL_POOL_COUNT = 384
RETAINED_COUNT = 320
MASKED_COUNT = ORIGINAL_POOL_COUNT - RETAINED_COUNT  # 64
SELECTED_COUNT = 256
RETAINED_FRACTION_NUM = 320
RETAINED_FRACTION_DEN = 384

#: The frozen selector -- resolved, never reimplemented.
SELECTOR_MODULE = "prism_fas.recipes.selection"
SELECTOR_CALLABLE = "prism_fas.recipes.selection.select"
SELECTOR_VERSION = "prism_c3_selection_v1"

#: Frozen ontology used by C3 selection (verified in preflight).
ONTOLOGY_RELPATH = "configs/recipes/ontology_m7.yaml"
ONTOLOGY_IDENTITY = "90694441c2ef1477ca8f6c4dd724a4997a3e166cbf5a067d52c101892f952bbd"

#: Historical C3 scientific bank lock -- the reference identities E9 compares to.
C3_SCIENTIFIC_BANK_LOCK_RELPATH = "reports/c3/scientific/C3_SCIENTIFIC_BANK_LOCK.json"
C3_BANK_RELPATHS: dict[str, str] = {
    "RND": "assets/recipe_banks/c3/rnd/C3_BANK.json",
    "DET": "assets/recipe_banks/c3/det/C3_BANK.json",
    "LLM": "assets/recipe_banks/c3/llm/C3_BANK.json",
}
#: Frozen selected-256 recipe files (read-only reference; NOT the 384 pool).
C3_SELECTED_RECIPES_RELPATHS: dict[str, str] = {
    "RND": "assets/recipe_banks/c3/rnd/recipes.jsonl",
    "DET": "assets/recipe_banks/c3/det/recipes.jsonl",
    "LLM": "assets/recipe_banks/c3/llm/recipes.jsonl",
}

#: Where E9 expects each arm's FROZEN 384-candidate pool + its hash-lock. These
#: are additive inputs the operator must materialize from the authorized C3
#: evidence (raw provider archive for LLM + the frozen deterministic control
#: schedules for RND/DET) before E9 can execute. E9 never regenerates them.
POOL_INPUT_RELPATHS: dict[str, str] = {
    arm: f"{OUTPUT_SUBTREE}/frozen_pools/{arm}_POOL_384.jsonl" for arm in ARMS
}
POOL_BINDING_RELPATH = f"{OUTPUT_SUBTREE}/E9_INPUT_BINDING.json"

#: Deterministic E9 INPUT artifacts expected to be TRACKED and CLEAN at the
#: execution commit, so the scientific runtime commit carries the canonical input
#: representation, not just code. (The historical raw provider archives are NOT
#: in this list -- they are source evidence, not committed by E9.)
INPUT_ARTIFACT_RELPATHS: tuple[str, ...] = (
    POOL_BINDING_RELPATH,
    *(POOL_INPUT_RELPATHS[a] for a in ARMS),
)

#: Frozen C3 eligible-pool identities. A reconstructed 384-pool MUST reproduce
#: these exactly -- checked independently of, and cross-checked against, the value
#: in each arm's C3_BANK.json. Source: C3_SCIENTIFIC_BANK_LOCK.json.
C3_ELIGIBLE_POOL_IDENTITY: dict[str, str] = {
    "RND": "aa2422860ac50c544decf25c0dd41a1c43b5cb9b82678dba27cef599979bded5",
    "DET": "96e32642180dff842e4e4870650be8cc3e2ceab797bca2efcc0b4014ae1f03f2",
    "LLM": "4032a7f8708a27d1545a84277d2b439767ae253c04113f3812ac30c16255c978",
}

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class E9Error(RuntimeError):
    """E9 refused an operation. Never partially applied, never silently relaxed."""


class E9Blocked(E9Error):
    """A required frozen input cannot be proven; E9 stops with BLOCKED."""


# --------------------------------------------------------------------------- #
# Deterministic perturbation rule
# --------------------------------------------------------------------------- #

def perturbation_string(arm: str, seed: int, candidate_id: str) -> str:
    """The exact versioned canonical string the SHA-256 is taken over."""
    if arm not in ARMS:
        raise E9Error(f"unknown arm {arm!r}; expected one of {ARMS}")
    return f"{PERTURB_STRING_VERSION}|{arm}|{seed}|{candidate_id}"


def perturbation_digest(arm: str, seed: int, candidate_id: str) -> str:
    """SHA-256 hex over the UTF-8 bytes of :func:`perturbation_string`.

    A pure hash of an explicitly versioned string -- no ``random`` module, no
    library-version-dependent behaviour, so the scientific identity is stable.
    """
    text = perturbation_string(arm, seed, candidate_id)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def partition_pool(
    arm: str,
    seed: int,
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    """Deterministically split one arm's 384-candidate pool for one seed.

    Returns a dict with ``retained`` (exactly 320 candidate ids, in retained
    sort order), ``masked`` (exactly 64) and ``rows`` (the full 384-row
    membership manifest, one entry per candidate). The result is independent of
    the order of ``candidate_ids`` on input: candidates are ordered by
    ``(perturbation_sha256_hex, candidate_id)`` ascending.
    """
    ids = list(candidate_ids)
    if len(ids) != ORIGINAL_POOL_COUNT:
        raise E9Blocked(
            f"arm {arm!r}: pool has {len(ids)} candidates, not the required "
            f"{ORIGINAL_POOL_COUNT}; E9 fails closed"
        )
    if len(set(ids)) != len(ids):
        raise E9Blocked(f"arm {arm!r}: pool contains duplicate candidate ids; E9 fails closed")

    keyed = [
        {
            "candidate_id": cid,
            "perturbation_sha256": perturbation_digest(arm, seed, cid),
        }
        for cid in ids
    ]
    keyed.sort(key=lambda r: (r["perturbation_sha256"], r["candidate_id"]))

    retained_ids = [r["candidate_id"] for r in keyed[:RETAINED_COUNT]]
    masked_ids = [r["candidate_id"] for r in keyed[RETAINED_COUNT:]]
    if len(retained_ids) != RETAINED_COUNT or len(masked_ids) != MASKED_COUNT:
        raise E9Error(
            f"arm {arm!r}: partition produced {len(retained_ids)}/{len(masked_ids)} "
            f"(expected {RETAINED_COUNT}/{MASKED_COUNT})"
        )
    retained_set = set(retained_ids)

    rows = []
    for rank, r in enumerate(keyed):
        rows.append(
            {
                "arm": arm,
                "perturbation_seed": seed,
                "perturb_string_version": PERTURB_STRING_VERSION,
                "candidate_id": r["candidate_id"],
                "perturbation_sha256": r["perturbation_sha256"],
                "sort_rank": rank,
                "membership": "retained" if r["candidate_id"] in retained_set else "masked",
            }
        )
    return {
        "arm": arm,
        "perturbation_seed": seed,
        "retained": retained_ids,
        "masked": masked_ids,
        "rows": rows,
        "membership_identity": sha256_json(
            {
                "arm": arm,
                "seed": seed,
                "perturb_string_version": PERTURB_STRING_VERSION,
                "retained": sorted(retained_ids),
                "masked": sorted(masked_ids),
            }
        ),
    }


# --------------------------------------------------------------------------- #
# Frozen candidate pool (read-only)
# --------------------------------------------------------------------------- #

def _canonical_candidate_id(recipe: Any) -> str:
    """The canonical Version-C candidate identity: the canonical recipe SHA-256
    (``prism_fas.recipes.canonical.recipe_hash``). This is the identity C3 uses
    in ``selected_recipe_identities`` and in the selector tie-break trace; it is
    NOT invented here."""
    from prism_fas.recipes.canonical import recipe_hash

    return recipe_hash(recipe)


def load_frozen_pool(path: Path) -> list[dict[str, Any]]:
    """Read-only load of one arm's frozen 384-candidate pool.

    The file is a JSONL of v1.1 recipe payloads (the same schema as the frozen
    ``recipes.jsonl``). Returns a list of
    ``{"candidate_id", "recipe_id", "recipe"}`` dicts. Fails closed on a bad
    row count, a schema-invalid recipe, or a duplicate canonical identity.
    """
    from prism_fas.recipes.schema import parse_recipe

    if not path.exists():
        raise E9Blocked(f"frozen candidate pool not found: {path}")
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                recipe = parse_recipe(payload)
            except Exception as exc:  # noqa: BLE001 - fail closed, keep the reason
                raise E9Blocked(f"{path}:{lineno}: not a valid v1.1 recipe: {exc}") from exc
            out.append(
                {
                    "candidate_id": _canonical_candidate_id(recipe),
                    "recipe_id": getattr(recipe, "recipe_id", None),
                    "recipe": recipe,
                }
            )
    if len(out) != ORIGINAL_POOL_COUNT:
        raise E9Blocked(
            f"{path}: {len(out)} candidates, not the required {ORIGINAL_POOL_COUNT}"
        )
    if len({c["candidate_id"] for c in out}) != len(out):
        raise E9Blocked(f"{path}: duplicate canonical candidate identities")
    return out


def pool_identity(candidates: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent identity of a candidate pool: SHA-256 over the sorted
    list of canonical candidate ids."""
    return sha256_json(sorted(c["candidate_id"] for c in candidates))


# --------------------------------------------------------------------------- #
# Selector -- REUSED, never reimplemented
# --------------------------------------------------------------------------- #

def selector_provenance() -> dict[str, Any]:
    """Prove which selector implementation E9 will call."""
    from prism_fas.recipes import selection as _sel

    module_file = Path(_sel.__file__).resolve()
    return {
        "selector_module": SELECTOR_MODULE,
        "selector_callable": SELECTOR_CALLABLE,
        "selection_version": _sel.SELECTION_VERSION,
        "selector_module_file": module_file.name,
        "selector_module_sha256": sha256_file(module_file),
        "raw_candidate_slots_per_arm": _sel.RAW_CANDIDATE_SLOTS_PER_ARM,
        "minimum_eligible_pool_per_arm": _sel.MINIMUM_ELIGIBLE_POOL_PER_ARM,
        "final_bank_size_per_arm": _sel.FINAL_BANK_SIZE_PER_ARM,
        "reused_not_reimplemented": True,
    }


def run_realization(
    arm: str,
    perturbation_id: str,
    seed: int,
    pool: Sequence[Mapping[str, Any]],
    ontology: Any,
) -> dict[str, Any]:
    """Perturb one pool for one seed and re-run the frozen MILP selector once.

    Returns a fully-populated result row. A selector infeasibility (or any
    deviation from an exact 256-recipe bank) is recorded as terminal
    ``BLOCKED`` with the selector's own reason -- it is NEVER retried and the
    retained fraction / bank size are NEVER relaxed in response.
    """
    from prism_fas.recipes.selection import (SelectionInfeasible, select)

    by_id = {c["candidate_id"]: c for c in pool}
    part = partition_pool(arm, seed, list(by_id.keys()))
    retained_recipes = [by_id[cid]["recipe"] for cid in part["retained"]]

    base: dict[str, Any] = {
        "arm": arm,
        "perturbation_id": perturbation_id,
        "perturbation_seed": seed,
        "perturb_string_version": PERTURB_STRING_VERSION,
        "original_pool_identity": pool_identity(pool),
        "original_pool_count": len(pool),
        "membership_identity": part["membership_identity"],
        "retained_candidate_ids": list(part["retained"]),
        "retained_count": len(part["retained"]),
        "masked_candidate_ids": list(part["masked"]),
        "masked_count": len(part["masked"]),
        "selector": selector_provenance(),
        "selector_objective_config": {
            "selection_version": SELECTOR_VERSION,
            "bank_size": SELECTED_COUNT,
            "minimum_pool": RETAINED_COUNT,
            "lexicographic_stages": ["hard_feasibility", "S_pref", "S_single",
                                     "S_multi", "canonical_tie_break"],
        },
        "target_access": False,
        "target_labels_accessed": False,
        "target_features_accessed": False,
        "llm_calls": 0,
    }

    try:
        result = select(retained_recipes, ontology, arm=arm)
    except SelectionInfeasible as exc:
        base.update(
            {
                "solver_status": "INFEASIBLE",
                "selected_count": 0,
                "selected_candidate_ids": [],
                "selected_bank_identity": None,
                "objective": None,
                "selector_trace": {"infeasible_reason": str(exc)},
                "status": "BLOCKED",
                "blocked_reason": f"selector infeasible on perturbed pool: {exc}",
            }
        )
        return base

    if len(result.selected_shas) != SELECTED_COUNT:
        base.update(
            {
                "solver_status": "NON_CONFORMANT",
                "selected_count": len(result.selected_shas),
                "selected_candidate_ids": list(result.selected_shas),
                "selected_bank_identity": result.selected_set_identity,
                "objective": {"S_pref": result.s_pref, "S_single": result.s_single,
                              "S_multi": result.s_multi},
                "selector_trace": {"tie_break_trace": result.tie_break_trace},
                "status": "BLOCKED",
                "blocked_reason": (
                    f"selector returned {len(result.selected_shas)} recipes, not "
                    f"{SELECTED_COUNT}; bank size is never lowered"
                ),
            }
        )
        return base

    base.update(
        {
            "solver_status": "SELECTED_256_VERIFIED",
            "selected_count": len(result.selected_shas),
            "selected_candidate_ids": list(result.selected_shas),
            "selected_bank_identity": result.selected_set_identity,
            "objective": {
                "S_pref": result.s_pref,
                "S_single": result.s_single,
                "S_multi": result.s_multi,
                "counts": result.counts,
            },
            "selector_trace": {
                "eligible_count": result.eligible_count,
                "rejected_count": len(result.rejected_shas),
                "tie_break_trace": result.tie_break_trace,
            },
            "status": "OK",
            "blocked_reason": None,
        }
    )
    return base


# --------------------------------------------------------------------------- #
# Descriptive stability metrics (no p-values)
# --------------------------------------------------------------------------- #

def _jaccard(a: Iterable[str], b: Iterable[str]) -> dict[str, Any]:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return {
        "intersection": inter,
        "union": union,
        "jaccard": (inter / union) if union else 0.0,
    }


def stability_for_arm(
    arm: str,
    reference_selected: Sequence[str],
    realizations: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Descriptive selection-stability summary for one arm.

    ``realizations`` maps ``P1``/``P2``/``P3`` to that realization's selected
    candidate id list (empty if the realization was BLOCKED).
    """
    ref = list(reference_selected)
    per_realization: dict[str, Any] = {}
    jaccards: list[float] = []
    for pid, selected in realizations.items():
        j = _jaccard(selected, ref)
        j["reference_retention_fraction"] = (
            j["intersection"] / len(ref) if ref else 0.0
        )
        j["selected_count"] = len(list(selected))
        per_realization[pid] = j
        if j["selected_count"]:
            jaccards.append(j["jaccard"])

    pairwise: dict[str, Any] = {}
    pids = list(realizations.keys())
    for i in range(len(pids)):
        for k in range(i + 1, len(pids)):
            a, b = pids[i], pids[k]
            pairwise[f"{a}|{b}"] = _jaccard(realizations[a], realizations[b])

    freq: dict[str, int] = {}
    for selected in realizations.values():
        for cid in selected:
            freq[cid] = freq.get(cid, 0) + 1

    return {
        "arm": arm,
        "reference_selected_count": len(ref),
        "per_realization_vs_reference": per_realization,
        "pairwise_between_realizations": pairwise,
        "per_candidate_selection_frequency": dict(sorted(freq.items())),
        "descriptive_jaccard_vs_reference": {
            "mean": (sum(jaccards) / len(jaccards)) if jaccards else None,
            "min": min(jaccards) if jaccards else None,
            "max": max(jaccards) if jaccards else None,
            "n_non_blocked": len(jaccards),
        },
        "note": (
            "Descriptive only. Structural selection stability is NOT a claim of "
            "downstream detector superiority and this is NOT independent LLM "
            "generation replication."
        ),
    }


# --------------------------------------------------------------------------- #
# Protocol amendment + lock (frozen before any E9 outcome)
# --------------------------------------------------------------------------- #

def build_protocol_amendment() -> dict[str, Any]:
    return {
        "schema_version": "e9-protocol-amendment-v1",
        "milestone": "E9",
        "title": "E9 retained-fraction implementation-resolution amendment",
        "governing_source": "PRISM_FAS_C_EXT_Q1Q2_Detailed_Spec_v1_0.docx, Section 16 (E9)",
        "base_commit": E9_BASE_COMMIT,
        "amendment_kind": "additive_implementation_resolution",
        "resolves": {
            "gap": (
                "Section 16 fixes the three perturbation seeds (P1=20260921, "
                "P2=20260922, P3=20260923) but does not numerically fix the "
                "retained fraction of the 384-candidate pool."
            ),
            "resolution": {
                "original_pool_count": ORIGINAL_POOL_COUNT,
                "retained_count": RETAINED_COUNT,
                "retained_fraction": f"{RETAINED_FRACTION_NUM}/{RETAINED_FRACTION_DEN}",
                "retained_fraction_decimal": RETAINED_FRACTION_NUM / RETAINED_FRACTION_DEN,
                "masked_count": MASKED_COUNT,
                "selected_count": SELECTED_COUNT,
                "milp_selection_slack_above_required": RETAINED_COUNT - SELECTED_COUNT,
            },
            "rationale": [
                "Purely combinatorial: exactly 64 candidates removed leaves the "
                "MILP exactly 64 candidates of selection slack above the required 256.",
                "Identical rule and identical count for RND, DET and LLM.",
                "retained_fraction = 320/384 = 5/6.",
            ],
        },
        "disclosure": {
            "not_numerically_specified_in_v1_0": True,
            "frozen_after_e8_existed": True,
            "frozen_before_any_e9_perturbation_or_selection_outcome": True,
            "no_e8_acer_used": True,
            "no_target_labels_used": True,
            "no_target_predictions_used": True,
            "no_target_metrics_used": True,
            "no_e9_result_used": True,
            "selected_from_e8_target_performance": False,
        },
        "perturbation_rule": {
            "string_version": PERTURB_STRING_VERSION,
            "canonical_string": f"{PERTURB_STRING_VERSION}|<arm>|<seed>|<candidate_id>",
            "hash": "SHA-256 over UTF-8 bytes of the canonical string",
            "sort_key": ["sha256_hex", "candidate_id"],
            "retain": f"first {RETAINED_COUNT} ascending",
            "mask": f"remaining {MASKED_COUNT}",
            "candidate_id": (
                "canonical Version-C recipe identity "
                "(prism_fas.recipes.canonical.recipe_hash); the same identity C3 "
                "uses in selected_recipe_identities and the selector tie-break trace"
            ),
            "no_python_random": True,
        },
        "arms": list(ARMS),
        "perturbation_seeds": dict(PERTURBATIONS),
        "selector": {
            "module": SELECTOR_MODULE,
            "callable": SELECTOR_CALLABLE,
            "version": SELECTOR_VERSION,
            "reused_not_reimplemented": True,
            "bank_size_never_lowered": True,
        },
        "target_access": False,
        "llm_calls": 0,
    }


def _amendment_identity(amendment: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(amendment))


def build_protocol_lock(amendment: Mapping[str, Any]) -> dict[str, Any]:
    ident = _amendment_identity(amendment)
    return {
        "schema_version": "e9-protocol-lock-v1",
        "milestone": "E9",
        "status": "FROZEN",
        "base_commit": E9_BASE_COMMIT,
        "e9_protocol_amendment_identity": ident,
        "amendment_canonical_form": (
            "json.dumps(amendment, sort_keys=True, separators=(',',':'), "
            "ensure_ascii=True, allow_nan=False) then SHA-256 over UTF-8 bytes"
        ),
        "retained_count": RETAINED_COUNT,
        "original_pool_count": ORIGINAL_POOL_COUNT,
        "masked_count": MASKED_COUNT,
        "selected_count": SELECTED_COUNT,
        "retained_fraction": f"{RETAINED_FRACTION_NUM}/{RETAINED_FRACTION_DEN}",
        "perturbation_seeds": dict(PERTURBATIONS),
        "arms": list(ARMS),
        "selector_version": SELECTOR_VERSION,
        "immutability": {
            "rewrite_permitted": False,
            "rule": (
                "identical recomputation is an idempotent no-op; a build that "
                "would produce different bytes raises rather than overwriting"
            ),
        },
    }


def _write_idempotent(repo: Path, relpath: str, obj: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``obj`` as pretty JSON iff absent or byte-identical; else fail closed."""
    abs_path = repo / relpath
    new_text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True,
                          allow_nan=False) + "\n"
    if abs_path.exists():
        old_text = abs_path.read_text(encoding="utf-8")
        if old_text != new_text:
            raise E9Error(
                f"REFUSED: {relpath} already exists with different bytes "
                f"(existing sha256={sha256_bytes(old_text.encode())}, "
                f"new sha256={sha256_bytes(new_text.encode())}); E9 never overwrites "
                "a scientific artifact"
            )
        return {"path": relpath, "action": "idempotent_noop",
                "sha256": sha256_bytes(new_text.encode("utf-8"))}
    write_text_atomic(abs_path, new_text, root=repo)
    return {"path": relpath, "action": "written",
            "sha256": sha256_bytes(new_text.encode("utf-8"))}


def freeze_protocol(repo: Path) -> dict[str, Any]:
    """Create-or-verify the additive E9 amendment + lock. Metadata only; no
    perturbation and no selection outcome is computed here."""
    amendment = build_protocol_amendment()
    lock = build_protocol_lock(amendment)
    a = _write_idempotent(repo, f"{OUTPUT_SUBTREE}/E9_PROTOCOL_AMENDMENT.json", amendment)
    b = _write_idempotent(repo, f"{OUTPUT_SUBTREE}/E9_PROTOCOL_LOCK.json", lock)
    return {
        "amendment": a,
        "lock": b,
        "e9_protocol_amendment_identity": lock["e9_protocol_amendment_identity"],
    }


# --------------------------------------------------------------------------- #
# git HEAD (subprocess-only, no torch/trainer import)
# --------------------------------------------------------------------------- #

def resolve_git_head(repo: Path) -> str:
    from prism_fas.utils.core import git_commit as _git_commit

    commit = _git_commit(repo)
    if not commit or not _GIT_SHA_RE.match(commit):
        raise E9Error(f"could not resolve a 40-hex git HEAD for {repo} (got {commit!r})")
    return commit


def _git(repo: Path, args: list[str]) -> tuple[int, str]:
    """Subprocess-only git. Returns (returncode, stdout.strip()). No torch/trainer
    import is reachable from here."""
    import subprocess

    try:
        proc = subprocess.run(["git", *args], cwd=repo, text=True,
                              capture_output=True, check=False)
    except OSError as exc:  # git not on PATH
        return 127, f"{exc}"
    return proc.returncode, (proc.stdout or "").strip()


def _git_head(repo: Path) -> str | None:
    rc, out = _git(repo, ["rev-parse", "HEAD"])
    return out if rc == 0 and _GIT_SHA_RE.match(out) else None


def _commit_in_history(repo: Path, commit: str) -> bool:
    return _git(repo, ["cat-file", "-e", f"{commit}^{{commit}}"])[0] == 0


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return _git(repo, ["merge-base", "--is-ancestor", ancestor, descendant])[0] == 0


def git_file_provenance(repo: Path, relpath: str) -> dict[str, Any]:
    """Prove whether one file is tracked and byte-identical to HEAD.

    * ``tracked``    -- ``git ls-files --error-unmatch`` succeeds.
    * ``worktree_sha256`` -- sha256 of the file on disk (or None if absent).
    * ``head_blob_sha256`` -- sha256 of the file's committed content at HEAD
      (``git show HEAD:<relpath>``), or None if not committed at HEAD.
    * ``clean_vs_head`` -- tracked AND worktree_sha256 == head_blob_sha256 AND
      ``git status --porcelain -- <relpath>`` is empty.
    """
    abs_path = repo / relpath
    worktree_sha = sha256_file(abs_path) if abs_path.exists() else None

    tracked = _git(repo, ["ls-files", "--error-unmatch", "--", relpath])[0] == 0

    rc, _ = _git(repo, ["cat-file", "-e", f"HEAD:{relpath}"])
    head_blob_sha = None
    if rc == 0:
        import subprocess
        try:
            blob = subprocess.run(["git", "show", f"HEAD:{relpath}"], cwd=repo,
                                  capture_output=True, check=False)
            if blob.returncode == 0:
                head_blob_sha = sha256_bytes(blob.stdout)
        except OSError:
            head_blob_sha = None

    porcelain = _git(repo, ["status", "--porcelain", "--", relpath])[1]
    diff_rc = _git(repo, ["diff", "--quiet", "HEAD", "--", relpath])[0]
    clean_vs_head = bool(
        tracked
        and worktree_sha is not None
        and head_blob_sha is not None
        and worktree_sha == head_blob_sha
        and porcelain == ""
        and diff_rc == 0
    )
    return {
        "path": relpath,
        "tracked": tracked,
        "worktree_sha256": worktree_sha,
        "head_blob_sha256": head_blob_sha,
        "status_porcelain": porcelain,
        "diff_vs_head_clean": diff_rc == 0,
        "clean_vs_head": clean_vs_head,
    }


def _last_commit_touching(repo: Path, relpath: str) -> str | None:
    rc, out = _git(repo, ["log", "-1", "--format=%H", "--", relpath])
    return out if rc == 0 and _GIT_SHA_RE.match(out) else None


def code_provenance(repo: Path) -> dict[str, Any]:
    """E9 code provenance with THREE distinct commits, never conflated:

    * ``base_commit``                -- frozen E8 parent, permanent, proven to be
      an ANCESTOR of HEAD (never required to equal HEAD).
    * ``implementation_source_commit`` -- the last commit that changed the E9
      module file; the reviewed implementation lands here.
    * ``execution_commit``           -- the current clean committed HEAD, i.e. the
      authoritative scientific runtime commit. It may equal
      ``implementation_source_commit`` or be a descendant that only ADDS frozen
      E9 input artifacts.
    """
    head = _git_head(repo)
    e9 = git_file_provenance(repo, E9_MODULE_RELPATH)
    e9_test = git_file_provenance(repo, E9_TEST_RELPATH)
    selector = git_file_provenance(repo, SELECTOR_MODULE_RELPATH)

    if not e9["tracked"]:
        impl_status = IMPL_UNCOMMITTED_REVIEW_STATE
    elif not e9["clean_vs_head"]:
        impl_status = IMPL_DIRTY_TRACKED
    else:
        impl_status = IMPL_COMMITTED_CLEAN

    base_present = _commit_in_history(repo, E9_BASE_COMMIT) if head else False
    base_is_ancestor = (
        _is_ancestor(repo, E9_BASE_COMMIT, head) if (head and base_present) else False
    )
    impl_source = _last_commit_touching(repo, E9_MODULE_RELPATH) if e9["tracked"] else None
    impl_source_is_ancestor = (
        _is_ancestor(repo, impl_source, head) if (impl_source and head) else False
    )

    scientific_code_tracked_and_clean = bool(
        e9["clean_vs_head"] and e9_test["clean_vs_head"]
        and selector["clean_vs_head"]
        and selector["worktree_sha256"] == SELECTOR_MODULE_SHA256
    )

    return {
        "base_commit": E9_BASE_COMMIT,
        "base_commit_present_in_history": base_present,
        "base_commit_is_ancestor_of_head": base_is_ancestor,
        "implementation_source_commit": impl_source,
        "implementation_source_commit_is_ancestor_of_head": impl_source_is_ancestor,
        "execution_commit": head,           # authoritative scientific runtime commit
        "current_code_commit": head,        # alias, kept for readability
        "head_is_base_commit": head == E9_BASE_COMMIT,
        "implementation_commit_status": impl_status,
        "scientific_code_tracked_and_clean": scientific_code_tracked_and_clean,
        "e9_module": e9,
        "e9_test": e9_test,
        "selector_module": {
            **selector,
            "expected_sha256": SELECTOR_MODULE_SHA256,
            "matches_frozen_selector_sha256": selector["worktree_sha256"] == SELECTOR_MODULE_SHA256,
        },
        "scientific_execution_allowed_from_this_state": bool(
            impl_status == IMPL_COMMITTED_CLEAN
            and scientific_code_tracked_and_clean
            and base_present and base_is_ancestor
            and impl_source and impl_source_is_ancestor
        ),
    }


# --------------------------------------------------------------------------- #
# Execution binding -- frozen runtime evidence pinning the EXECUTION commit
# --------------------------------------------------------------------------- #
#
# Lifecycle (corrected -- no self-reference):
#   * E9_EXECUTION_BINDING.json is FROZEN AFTER the final clean committed
#     execution worktree exists. It records execution_commit == that HEAD.
#   * It does NOT need to be committed before the scientific run. During
#     execution it is immutable runtime evidence living under the ignored E9
#     reports namespace; its bytes/identity are pinned into E9_EVIDENCE.sha256
#     and E9_CLOSURE.json.
#   * The scientific --run gate requires  current HEAD == binding.execution_commit
#     (NOT that the binding file is committed at HEAD).
#   * It may be archived in a LATER additive evidence commit; that commit is not
#     the execution commit and must never trigger a scientific rerun.

def _protocol_lock_identity(repo: Path) -> tuple[str, str]:
    p = repo / f"{OUTPUT_SUBTREE}/E9_PROTOCOL_LOCK.json"
    if not p.exists():
        raise E9Blocked("E9_PROTOCOL_LOCK.json is missing; run --preflight first")
    lock = json.loads(p.read_text(encoding="utf-8"))
    ident = lock.get("e9_protocol_amendment_identity")
    if not ident or not _HEX64_RE.match(ident):
        raise E9Blocked("E9_PROTOCOL_LOCK.json has no valid e9_protocol_amendment_identity")
    return ident, sha256_file(p)


def _input_binding_view(repo: Path) -> dict[str, Any]:
    """Parse + validate E9_INPUT_BINDING.json. Fails closed unless all three arms
    are MATERIALIZED and every pool_identity equals the frozen C3
    eligible_pool_identity for that arm."""
    p = repo / POOL_BINDING_RELPATH
    if not p.exists():
        raise E9Blocked("E9_INPUT_BINDING.json is missing; run --materialize-frozen-pools first")
    binding = json.loads(p.read_text(encoding="utf-8"))
    arms = binding.get("arms", {})
    if not binding.get("all_pools_materialized") or set(arms) != set(ARMS) or any(
        arms.get(a, {}).get("status") != "MATERIALIZED" for a in ARMS
    ):
        raise E9Blocked(
            "E9_INPUT_BINDING.json does not have all three arms (RND, DET, LLM) "
            "MATERIALIZED; a full three-arm input binding is mandatory before an "
            "execution binding can be frozen"
        )
    pool_identity_by_arm: dict[str, str] = {}
    for a in ARMS:
        pid = arms[a].get("pool_identity")
        if pid != C3_ELIGIBLE_POOL_IDENTITY[a]:
            raise E9Blocked(
                f"{a}: E9_INPUT_BINDING pool_identity {pid} != frozen C3 "
                f"eligible_pool_identity {C3_ELIGIBLE_POOL_IDENTITY[a]}"
            )
        pool_identity_by_arm[a] = pid
    return {
        "pool_identity_by_arm": pool_identity_by_arm,
        "identity": sha256_json(pool_identity_by_arm),
        "file_sha256": sha256_file(p),
    }


def input_artifacts_report(repo: Path) -> dict[str, Any]:
    """git state of the deterministic E9 INPUT artifacts that must be TRACKED and
    CLEAN at the execution commit."""
    entries = {}
    not_clean = []
    for rel in INPUT_ARTIFACT_RELPATHS:
        fp = git_file_provenance(repo, rel)
        entries[rel] = {"tracked": fp["tracked"],
                        "worktree_sha256": fp["worktree_sha256"],
                        "head_blob_sha256": fp["head_blob_sha256"],
                        "clean_vs_head": fp["clean_vs_head"]}
        if not fp["clean_vs_head"]:
            not_clean.append(rel)
    return {"artifacts": entries, "not_clean": not_clean,
            "all_tracked_and_clean": not not_clean}


def build_execution_binding(repo: Path) -> dict[str, Any]:
    """Assemble (not write) the E9 execution binding for the CURRENT HEAD as the
    execution commit. Fails closed unless every scientific-code and input-artifact
    provenance precondition holds."""
    prov = code_provenance(repo)
    e9, e9_test = prov["e9_module"], prov["e9_test"]
    sel = prov["selector_module"]

    if prov["implementation_commit_status"] != IMPL_COMMITTED_CLEAN:
        raise E9Blocked(
            "refusing to freeze an execution binding: E9 implementation module is "
            f"{prov['implementation_commit_status']} (needs {IMPL_COMMITTED_CLEAN}); "
            "commit the reviewed E9 module + tests first"
        )
    if not e9_test["clean_vs_head"]:
        raise E9Blocked(f"{E9_TEST_RELPATH} is not tracked+clean at HEAD")
    if not sel["matches_frozen_selector_sha256"] or not sel["clean_vs_head"]:
        raise E9Blocked(
            f"selector is not the frozen, committed {SELECTOR_MODULE_RELPATH} "
            f"(sha256 {SELECTOR_MODULE_SHA256})"
        )
    if not prov["base_commit_present_in_history"]:
        raise E9Blocked(f"base commit {E9_BASE_COMMIT} is not in this repository's history")
    if not prov["base_commit_is_ancestor_of_head"]:
        raise E9Blocked(
            f"base commit {E9_BASE_COMMIT} is not an ancestor of HEAD; E9 must be "
            "DERIVED from the base commit"
        )
    if not prov["implementation_source_commit"]:
        raise E9Blocked("could not resolve implementation_source_commit for the E9 module")
    if not prov["implementation_source_commit_is_ancestor_of_head"]:
        raise E9Blocked("implementation_source_commit is not an ancestor of HEAD")

    protocol_lock_identity, protocol_lock_sha256 = _protocol_lock_identity(repo)
    input_view = _input_binding_view(repo)

    inputs = input_artifacts_report(repo)
    if not inputs["all_tracked_and_clean"]:
        raise E9Blocked(
            "deterministic E9 input artifacts are not all tracked+clean at HEAD: "
            + ", ".join(inputs["not_clean"])
            + " -- force-add and commit them into the execution commit first"
        )

    head = prov["execution_commit"]
    material = {
        "base_commit": E9_BASE_COMMIT,
        "implementation_source_commit": prov["implementation_source_commit"],
        "execution_commit": head,
        "e9_module_sha256": e9["head_blob_sha256"],
        "e9_test_sha256": e9_test["head_blob_sha256"],
        "selector_module_sha256": sel["head_blob_sha256"],
        "protocol_lock_identity": protocol_lock_identity,
        "input_binding_identity": input_view["identity"],
        "pool_identity_by_arm": input_view["pool_identity_by_arm"],
        "ontology_identity": ONTOLOGY_IDENTITY,
    }
    frozen_pools = {}
    for a in ARMS:
        fp = git_file_provenance(repo, POOL_INPUT_RELPATHS[a])
        frozen_pools[a] = {
            "path": POOL_INPUT_RELPATHS[a],
            "file_sha256": fp["worktree_sha256"],
            "head_blob_sha256": fp["head_blob_sha256"],
            "pool_identity": input_view["pool_identity_by_arm"][a],
            "tracked_at_execution_commit": fp["tracked"],
        }
    binding = {
        "schema_version": "e9-execution-binding-v2",
        "milestone": "E9",
        "status": "FROZEN",
        "purpose": (
            "Frozen runtime evidence: pins execution_commit (the clean committed "
            "HEAD from which the nine scientific selections run) plus every code / "
            "input identity. base_commit and implementation_source_commit are "
            "proven ANCESTORS of execution_commit, never required to equal it. This "
            "file need NOT be committed before --run; it is immutable evidence and "
            "may be archived in a later additive evidence commit."
        ),
        **material,
        "e9_module_path": E9_MODULE_RELPATH,
        "e9_test_path": E9_TEST_RELPATH,
        "selector_module_path": SELECTOR_MODULE_RELPATH,
        "protocol_lock_path": f"{OUTPUT_SUBTREE}/E9_PROTOCOL_LOCK.json",
        "protocol_lock_file_sha256": protocol_lock_sha256,
        "input_binding_path": POOL_BINDING_RELPATH,
        "input_binding_file_sha256": input_view["file_sha256"],
        "frozen_pools": frozen_pools,
        "base_commit_is_ancestor_of_execution_commit": True,
        "implementation_source_commit_is_ancestor_of_execution_commit": True,
        "execution_binding_identity": sha256_json(material),
        "immutability": {
            "rewrite_permitted": False,
            "rule": ("immutable once frozen; identical recomputation is an idempotent "
                     "no-op; different bytes fail closed and are never silently rewritten"),
        },
        "post_run_archive_rule": (
            "E9_EXECUTION_BINDING.json, membership, selection results, stability and "
            "closure may be committed in a LATER additive evidence commit; that "
            "commit is NOT execution_commit and must not cause a scientific rerun."
        ),
    }
    return binding


def freeze_execution_binding(repo: Path) -> dict[str, Any]:
    binding = build_execution_binding(repo)
    result = _write_idempotent(repo, EXECUTION_BINDING_RELPATH, binding)
    return {
        "execution_binding": result,
        "execution_binding_identity": binding["execution_binding_identity"],
        "execution_binding_file_sha256": sha256_file(repo / EXECUTION_BINDING_RELPATH),
        "execution_commit": binding["execution_commit"],
        "implementation_source_commit": binding["implementation_source_commit"],
        "base_commit": binding["base_commit"],
    }


def verify_execution_binding(repo: Path) -> dict[str, Any]:
    """Fail-closed gate for scientific --run. Recomputes live git state and checks
    it against the frozen E9_EXECUTION_BINDING.json. Never mutates anything.

    The master check is: rebuilding the binding from live state must reproduce the
    frozen ``execution_binding_identity`` exactly. Itemized checks follow so a
    failure report is legible.
    """
    failures: list[str] = []
    p = repo / EXECUTION_BINDING_RELPATH
    if not p.exists():
        return {"ok": False, "failures": [
            "E9_EXECUTION_BINDING.json is absent; freeze it at the final clean "
            "execution commit via --freeze-execution-binding"],
            "binding": None, "code_provenance": code_provenance(repo)}
    binding = json.loads(p.read_text(encoding="utf-8"))
    binding_file_sha256 = sha256_file(p)
    prov = code_provenance(repo)
    head = prov["execution_commit"]
    e9, e9_test, sel = prov["e9_module"], prov["e9_test"], prov["selector_module"]

    # --- master check: live recomputation must reproduce the frozen identity ---
    recomputed = None
    try:
        recomputed = build_execution_binding(repo)
        if recomputed["execution_binding_identity"] != binding.get("execution_binding_identity"):
            failures.append(
                "live recomputation of the execution binding differs from the frozen "
                f"identity ({recomputed['execution_binding_identity']} != "
                f"{binding.get('execution_binding_identity')})")
    except E9Error as exc:
        failures.append(f"execution binding cannot be recomputed from live state: {exc}")

    # --- itemized checks ---
    if not e9["tracked"]:
        failures.append("E9 implementation module is UNTRACKED")
    elif not e9["clean_vs_head"]:
        failures.append("E9 implementation module differs from HEAD (dirty/uncommitted)")
    if not e9_test["clean_vs_head"]:
        failures.append(f"{E9_TEST_RELPATH} is not tracked+clean at HEAD")
    if prov["implementation_commit_status"] != IMPL_COMMITTED_CLEAN:
        failures.append(
            f"implementation_commit_status is {prov['implementation_commit_status']}, "
            f"not {IMPL_COMMITTED_CLEAN}")
    if not prov["scientific_code_tracked_and_clean"]:
        failures.append("scientific code (E9 module/test + selector) is not all tracked+clean")

    if binding.get("base_commit") != E9_BASE_COMMIT:
        failures.append(
            f"execution binding base_commit {binding.get('base_commit')!r} != frozen "
            f"{E9_BASE_COMMIT}")
    if not prov["base_commit_present_in_history"]:
        failures.append(f"base commit {E9_BASE_COMMIT} not in history")
    if not prov["base_commit_is_ancestor_of_head"]:
        failures.append("base commit is not an ancestor of the execution commit")

    if binding.get("execution_commit") != head:
        failures.append(
            f"execution binding execution_commit {binding.get('execution_commit')!r} "
            f"!= current HEAD {head!r} (the run must execute from execution_commit)")
    if binding.get("implementation_source_commit") != prov["implementation_source_commit"]:
        failures.append("execution binding implementation_source_commit != live value")
    if not prov["implementation_source_commit_is_ancestor_of_head"]:
        failures.append("implementation_source_commit is not an ancestor of the execution commit")

    if binding.get("e9_module_sha256") != e9["worktree_sha256"]:
        failures.append("E9 module worktree sha256 != execution binding e9_module_sha256")
    if binding.get("e9_module_sha256") != e9["head_blob_sha256"]:
        failures.append("E9 module committed-blob sha256 != execution binding e9_module_sha256")
    if binding.get("selector_module_sha256") != sel["worktree_sha256"]:
        failures.append("selector worktree sha256 != execution binding selector_module_sha256")
    if sel["worktree_sha256"] != SELECTOR_MODULE_SHA256:
        failures.append(
            f"selector worktree sha256 != frozen selector binding {SELECTOR_MODULE_SHA256}")
    if not sel["clean_vs_head"]:
        failures.append("selector module differs from HEAD (dirty)")

    try:
        protocol_lock_identity, protocol_lock_sha256 = _protocol_lock_identity(repo)
        if binding.get("protocol_lock_identity") != protocol_lock_identity:
            failures.append("E9_PROTOCOL_LOCK identity != execution binding")
        if binding.get("protocol_lock_file_sha256") != protocol_lock_sha256:
            failures.append("E9_PROTOCOL_LOCK file sha256 != execution binding")
    except E9Blocked as exc:
        failures.append(str(exc))

    try:
        input_view = _input_binding_view(repo)
        if binding.get("input_binding_identity") != input_view["identity"]:
            failures.append("E9_INPUT_BINDING identity != execution binding")
        if binding.get("input_binding_file_sha256") != input_view["file_sha256"]:
            failures.append("E9_INPUT_BINDING file sha256 != execution binding")
        if binding.get("pool_identity_by_arm") != input_view["pool_identity_by_arm"]:
            failures.append("per-arm pool_identity != execution binding")
    except E9Blocked as exc:
        failures.append(str(exc))

    inputs = input_artifacts_report(repo)
    if not inputs["all_tracked_and_clean"]:
        failures.append(
            "deterministic E9 input artifacts not tracked+clean at HEAD: "
            + ", ".join(inputs["not_clean"]))

    protected = protected_historical_report(repo)
    if not protected["all_clean_vs_head"]:
        failures.append(
            "protected historical artifact(s) changed vs HEAD: "
            + ", ".join(protected["changed"]))

    # --- post-run tamper detection: a written closure pins the binding bytes ---
    closure_path = repo / CLOSURE_RELPATH
    if closure_path.exists():
        try:
            closure = json.loads(closure_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            closure = {}
        pinned = closure.get("execution_binding_file_sha256")
        if pinned and pinned != binding_file_sha256:
            failures.append(
                "E9_EXECUTION_BINDING.json bytes changed since E9_CLOSURE.json was written")

    return {"ok": not failures, "failures": failures, "binding": binding,
            "binding_file_sha256": binding_file_sha256,
            "code_provenance": prov, "protected": protected,
            "input_artifacts": inputs}


def protected_historical_report(repo: Path) -> dict[str, Any]:
    entries = {}
    changed = []
    for rel in PROTECTED_HISTORICAL_RELPATHS:
        fp = git_file_provenance(repo, rel)
        entries[rel] = {
            "tracked": fp["tracked"],
            "worktree_sha256": fp["worktree_sha256"],
            "head_blob_sha256": fp["head_blob_sha256"],
            "clean_vs_head": fp["clean_vs_head"],
        }
        if not fp["clean_vs_head"]:
            changed.append(rel)
    return {"artifacts": entries, "changed": changed, "all_clean_vs_head": not changed}


def _module_imports(path: Path) -> set[str]:
    """Every module name this file imports, from a static AST parse -- including
    the lazy function-local imports E9 uses to keep its top level light."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names.add(node.module)
    return names


# --------------------------------------------------------------------------- #
# Historical reference identities
# --------------------------------------------------------------------------- #

def load_reference_identities(repo: Path) -> dict[str, Any]:
    """Read the historical C3 selected-256 identities E9 compares against.

    Read-only. Pulls per-arm ``selected_set_identity`` /
    ``selected_recipe_identities`` from the frozen C3 scientific bank lock and
    the per-arm ``C3_BANK.json`` files.
    """
    lock_path = repo / C3_SCIENTIFIC_BANK_LOCK_RELPATH
    if not lock_path.exists():
        raise E9Blocked(f"missing frozen C3 scientific bank lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    arms_block = lock.get("arms", {})
    out: dict[str, Any] = {
        "c3_scientific_bank_lock_sha256": sha256_file(lock_path),
        "lock_identity": lock.get("lock_identity"),
        "arms": {},
    }
    for arm in ARMS:
        entry = arms_block.get(arm, {})
        bank_path = repo / C3_BANK_RELPATHS[arm]
        bank = json.loads(bank_path.read_text(encoding="utf-8")) if bank_path.exists() else {}
        selected = list(bank.get("selected_recipe_identities", []))
        out["arms"][arm] = {
            "reference_selected_set_identity": entry.get("selected_set_identity"),
            "reference_bank_identity": entry.get("bank_identity"),
            "reference_eligible_pool_identity": entry.get("eligible_pool_identity"),
            "reference_eligible": entry.get("eligible"),
            "reference_selected": entry.get("selected"),
            "c3_bank_json_sha256": sha256_file(bank_path) if bank_path.exists() else None,
            "reference_selected_recipe_identities": selected,
            "reference_selected_recipe_identities_count": len(selected),
        }
    return out


# --------------------------------------------------------------------------- #
# Frozen 384-candidate pool materialization (input preparation, not science)
# --------------------------------------------------------------------------- #
#
# The authorized C3 run never persisted a 384-row candidate-pool file: the LLM
# raw provider archives are git-ignored and the RND/DET raw pools live only in
# the frozen deterministic schedule. This step re-derives each arm's frozen
# 384-candidate pool the SAME way milestone E1 does (accepted precedent) and
# proves it before writing:
#
#   RND / DET : re-materialize the 384-slot schedule from
#               ``prism_fas.recipes.arm_schedules.draft_schedule`` + the frozen
#               ontology. GATE: live ``ArmSchedule.schedule_identity`` must equal
#               the frozen ``C3_<ARM>_SCHEDULE_CONTRACT.json`` value AND every one
#               of the 256 frozen selected recipes must reproduce (canonical
#               ``recipe_hash``) as a member of the reconstructed pool.
#   LLM       : parse the 12 frozen provider responses under
#               ``reports/c3/live/raw_responses/``. GATE: exactly 384 candidates
#               and the 256 frozen selected identities are a subset. If those
#               archives are absent (they are git-ignored) LLM is BLOCKED here --
#               E9 never calls a provider to recreate them.
#
# No new scientific bank is generated: this only reconstructs the already-frozen
# pool and fails closed on any provenance mismatch.

def _positional_recipe_id(slot_index: int) -> str:
    return f"R-{slot_index:06d}"


def _canonical_recipe_line(recipe: Any) -> str:
    from prism_fas.recipes.canonical import canonical_json

    return canonical_json(recipe)


def _reconstruct_control_pool(repo: Path, arm: str) -> dict[str, Any]:
    """RND/DET: deterministic 384-slot reconstruction with full provenance gate."""
    from prism_fas.recipes.arm_schedules import build_schedule, draft_schedule
    from prism_fas.recipes.canonical import recipe_hash
    from prism_fas.recipes.ontology import load_ontology
    from prism_fas.recipes.schema import parse_recipe

    contract_path = repo / f"reports/c3/v15_selection_contract/C3_{arm}_SCHEDULE_CONTRACT.json"
    if not contract_path.exists():
        raise E9Blocked(f"{arm}: missing frozen schedule contract {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ontology = load_ontology(repo / ONTOLOGY_RELPATH)
    if ontology.sha256 != contract["ontology_identity"]:
        raise E9Blocked(
            f"{arm}: live ontology sha256 {ontology.sha256} != frozen "
            f"{contract['ontology_identity']}"
        )
    schedule = build_schedule(arm, ontology)
    if schedule.schedule_identity != contract["schedule_identity"]:
        raise E9Blocked(
            f"{arm}: live schedule_identity {schedule.schedule_identity} != frozen "
            f"{contract['schedule_identity']}"
        )

    recipes: list[Any] = []
    for index, (_slot_id, payload) in enumerate(draft_schedule(arm, ontology)):
        row = dict(payload)
        row["recipe_id"] = _positional_recipe_id(index)
        recipes.append(parse_recipe(row))
    if len(recipes) != ORIGINAL_POOL_COUNT:
        raise E9Blocked(f"{arm}: reconstructed {len(recipes)} slots, expected {ORIGINAL_POOL_COUNT}")

    pool_ids = {recipe_hash(r) for r in recipes}
    _assert_pool_identity(arm, pool_ids, repo)
    selected_path = repo / C3_SELECTED_RECIPES_RELPATHS[arm]
    selected = [parse_recipe(json.loads(l)) for l in
                selected_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    selected_ids = {recipe_hash(r) for r in selected}
    if not selected_ids <= pool_ids:
        raise E9Blocked(
            f"{arm}: {len(selected_ids - pool_ids)} of {len(selected_ids)} frozen selected "
            "recipes are not members of the reconstructed 384-pool; provenance fails"
        )
    bank = json.loads((repo / C3_BANK_RELPATHS[arm]).read_text(encoding="utf-8"))
    if set(bank.get("selected_recipe_identities", [])) != selected_ids:
        raise E9Blocked(f"{arm}: C3_BANK.selected_recipe_identities disagree with recipes.jsonl")

    return {
        "arm": arm,
        "method": "deterministic_schedule_reconstruction",
        "recipes": recipes,
        "provenance": {
            "schedule_identity": schedule.schedule_identity,
            "frozen_schedule_identity": contract["schedule_identity"],
            "ontology_identity": ontology.sha256,
            "eligible_pool_identity": sha256_json(sorted(pool_ids)),
            "frozen_c3_eligible_pool_identity": C3_ELIGIBLE_POOL_IDENTITY[arm],
            "selected_256_reproduced_as_pool_members": f"{len(selected_ids)}/{len(selected_ids)}",
            "schedule_contract_path": f"reports/c3/v15_selection_contract/C3_{arm}_SCHEDULE_CONTRACT.json",
            "schedule_contract_sha256": sha256_file(contract_path),
        },
    }


def _assert_pool_identity(arm: str, pool_ids: set[str], repo: Path) -> None:
    """A reconstructed 384-pool must reproduce the frozen C3 eligible_pool_identity
    for its arm -- checked against the hard constant AND against the value on disk
    in C3_BANK.json / C3_SCIENTIFIC_BANK_LOCK.json."""
    if len(pool_ids) != ORIGINAL_POOL_COUNT:
        raise E9Blocked(f"{arm}: reconstructed pool has {len(pool_ids)} unique ids, "
                        f"expected {ORIGINAL_POOL_COUNT}")
    got = sha256_json(sorted(pool_ids))
    if got != C3_ELIGIBLE_POOL_IDENTITY[arm]:
        raise E9Blocked(
            f"{arm}: reconstructed eligible_pool_identity {got} != frozen "
            f"{C3_ELIGIBLE_POOL_IDENTITY[arm]}; refusing an unfaithful pool"
        )
    on_disk = json.loads(
        (repo / C3_BANK_RELPATHS[arm]).read_text(encoding="utf-8")
    ).get("eligible_pool_identity")
    if on_disk != C3_ELIGIBLE_POOL_IDENTITY[arm]:
        raise E9Blocked(
            f"{arm}: C3_BANK.json eligible_pool_identity {on_disk} != frozen constant "
            f"{C3_ELIGIBLE_POOL_IDENTITY[arm]}"
        )


def _reconstruct_llm_pool(repo: Path) -> dict[str, Any]:
    """LLM: parse the 12 frozen provider responses into the 384 raw candidates."""
    from prism_fas.recipes.canonical import recipe_hash
    from prism_fas.recipes.schema import parse_recipe

    raw_dir = repo / "reports/c3/live/raw_responses"
    files = sorted(raw_dir.glob("c3-llm-req-*.json"))
    if len(files) != 12:
        raise E9Blocked(
            f"LLM: expected 12 frozen provider archives under {raw_dir}, found {len(files)}. "
            "These are git-ignored; restore them from the authorized C3 run. E9 never "
            "calls a provider to recreate them."
        )
    payloads: list[dict[str, Any]] = []
    for f in files:
        outer = json.loads(f.read_text(encoding="utf-8"))
        inner = json.loads(outer["raw_response"]) if "raw_response" in outer else outer
        payloads.extend(inner["recipes"])
    if len(payloads) != ORIGINAL_POOL_COUNT:
        raise E9Blocked(f"LLM: parsed {len(payloads)} raw candidates, expected {ORIGINAL_POOL_COUNT}")
    recipes = [parse_recipe(p) for p in payloads]
    pool_ids = {recipe_hash(r) for r in recipes}
    # HARD gate: reconstructed LLM eligible_pool_identity must be exactly
    # 4032a7f8708a27d1545a84277d2b439767ae253c04113f3812ac30c16255c978.
    _assert_pool_identity("LLM", pool_ids, repo)
    selected_path = repo / C3_SELECTED_RECIPES_RELPATHS["LLM"]
    selected_ids = {
        recipe_hash(parse_recipe(json.loads(l)))
        for l in selected_path.read_text(encoding="utf-8").splitlines() if l.strip()
    }
    if not selected_ids <= pool_ids:
        raise E9Blocked(
            f"LLM: {len(selected_ids - pool_ids)} of {len(selected_ids)} frozen selected "
            "recipes are not members of the parsed 384-pool; provenance fails"
        )
    return {
        "arm": "LLM",
        "method": "frozen_provider_archive_parse",
        "recipes": recipes,
        "provenance": {
            "raw_response_files": [f.name for f in files],
            "raw_response_tree_sha256": sha256_json(
                {f.name: sha256_file(f) for f in files}
            ),
            "raw_response_file_sha256": {f.name: sha256_file(f) for f in files},
            "eligible_pool_identity": sha256_json(sorted(pool_ids)),
            "frozen_c3_eligible_pool_identity": C3_ELIGIBLE_POOL_IDENTITY["LLM"],
            "selected_256_reproduced_as_pool_members": f"{len(selected_ids)}/{len(selected_ids)}",
        },
    }


def materialize_frozen_pools(repo: Path) -> dict[str, Any]:
    """Create-or-verify each arm's frozen 384-candidate pool JSONL + binding.

    Reconstruction only -- no scientific generation, no perturbation, no MILP.
    Idempotent: an existing pool file with different bytes fails closed.
    """
    from prism_fas.recipes.canonical import recipe_hash

    binding: dict[str, Any] = {
        "schema_version": "e9-input-binding-v1",
        "milestone": "E9",
        "base_commit": E9_BASE_COMMIT,
        "principle": (
            "Frozen Version-C RND/DET/LLM candidate pools are reconstructed and "
            "PROVEN, never regenerated. No LLM call, no new scientific bank."
        ),
        "candidate_id_definition": (
            "canonical Version-C recipe identity prism_fas.recipes.canonical.recipe_hash "
            "(recipe_id assigned positionally: R-<6-digit slot index>)"
        ),
        "arms": {},
        "target_access": False,
        "llm_calls": 0,
    }
    any_blocked = False
    for arm in ARMS:
        rel = POOL_INPUT_RELPATHS[arm]
        try:
            recon = (_reconstruct_llm_pool(repo) if arm == "LLM"
                     else _reconstruct_control_pool(repo, arm))
        except E9Blocked as exc:
            any_blocked = True
            binding["arms"][arm] = {"pool_path": rel, "status": "BLOCKED", "reason": str(exc)}
            continue
        recipes = recon["recipes"]
        # Fidelity gate BEFORE any write: the reconstructed pool's identity must
        # equal the frozen C3 eligible_pool_identity for this arm -- checked
        # against the hard constant AND the value on disk.
        cand_ids = [recipe_hash(r) for r in recipes]
        this_pool_identity = sha256_json(sorted(cand_ids))
        c3_bank = json.loads((repo / C3_BANK_RELPATHS[arm]).read_text(encoding="utf-8"))
        c3_eligible_pool_identity = c3_bank.get("eligible_pool_identity")
        if this_pool_identity != C3_ELIGIBLE_POOL_IDENTITY[arm] \
                or this_pool_identity != c3_eligible_pool_identity:
            raise E9Error(
                f"{arm}: reconstructed pool_identity {this_pool_identity} != frozen C3 "
                f"eligible_pool_identity (constant {C3_ELIGIBLE_POOL_IDENTITY[arm]}, "
                f"on-disk {c3_eligible_pool_identity}); refusing to write/bind an "
                "unfaithful pool"
            )
        lines = "".join(_canonical_recipe_line(r) + "\n" for r in recipes)
        abs_path = repo / rel
        if abs_path.exists():
            if abs_path.read_text(encoding="utf-8") != lines:
                raise E9Error(
                    f"REFUSED: {rel} exists with different bytes; E9 never overwrites a "
                    "materialized pool. Delete it deliberately if a re-materialization is intended."
                )
            action = "idempotent_noop"
        else:
            write_text_atomic(abs_path, lines, root=repo)
            action = "written"
        binding["arms"][arm] = {
            "pool_path": rel,
            "status": "MATERIALIZED",
            "action": action,
            "pool_count": len(recipes),
            "pool_file_sha256": sha256_bytes(lines.encode("utf-8")),
            "pool_identity": this_pool_identity,
            "c3_eligible_pool_identity": c3_eligible_pool_identity,
            "matches_c3_eligible_pool_identity": True,
            "method": recon["method"],
            "provenance": recon["provenance"],
        }
    binding["all_pools_materialized"] = not any_blocked
    _write_idempotent_allow_update(repo, POOL_BINDING_RELPATH, binding)
    return binding


def _write_idempotent_allow_update(repo: Path, relpath: str, obj: Mapping[str, Any]) -> None:
    """The input binding may legitimately change as arms move BLOCKED->MATERIALIZED,
    so it is a normal guarded atomic write (never a scientific-outcome artifact)."""
    write_json_atomic(repo / relpath, obj, root=repo)


# --------------------------------------------------------------------------- #
# Preflight (read-only w.r.t. science; writes only the additive amendment/lock)
# --------------------------------------------------------------------------- #

def preflight(repo: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(cid: str, ok: bool, summary: str, **detail: Any) -> None:
        checks.append({"check_id": cid, "ok": bool(ok), "summary": summary, "detail": detail})

    # base commit binding + code provenance --------------------------------
    # base_commit is the frozen development parent. It is NOT required to equal
    # HEAD; the scientific implementation is expected to land as a child of it.
    cprov = code_provenance(repo)
    head = cprov["execution_commit"]
    check(
        "e9_base_commit_binding",
        E9_BASE_COMMIT == "6f0642a1d05c35e4c1778d329fb547f1b22a4115"
        and cprov["base_commit_present_in_history"],
        f"base_commit pinned at {E9_BASE_COMMIT}; present_in_history="
        f"{cprov['base_commit_present_in_history']}, ancestor_of_head="
        f"{cprov['base_commit_is_ancestor_of_head']}",
        base_commit=E9_BASE_COMMIT,
        base_commit_present_in_history=cprov["base_commit_present_in_history"],
        base_commit_is_ancestor_of_head=cprov["base_commit_is_ancestor_of_head"],
        head_is_base_commit=cprov["head_is_base_commit"],
    )
    # Descriptive, never fails the engineering preflight: pre-review the E9
    # module is legitimately UNCOMMITTED_REVIEW_STATE. Scientific --run enforces
    # COMMITTED_CLEAN + a valid execution binding independently.
    check(
        "e9_code_provenance",
        True,
        f"implementation_commit_status={cprov['implementation_commit_status']}, "
        f"implementation_source_commit={cprov['implementation_source_commit']}, "
        f"execution_commit={head}, "
        f"selector_sha_frozen={cprov['selector_module']['matches_frozen_selector_sha256']}",
        base_commit=E9_BASE_COMMIT,
        implementation_source_commit=cprov["implementation_source_commit"],
        execution_commit=head,
        implementation_commit_status=cprov["implementation_commit_status"],
        scientific_code_tracked_and_clean=cprov["scientific_code_tracked_and_clean"],
        e9_module=cprov["e9_module"],
        e9_test=cprov["e9_test"],
        selector_module=cprov["selector_module"],
        scientific_execution_allowed_from_this_state=cprov[
            "scientific_execution_allowed_from_this_state"],
    )

    # protocol amendment + lock (idempotent) -------------------------------
    protocol = None
    try:
        protocol = freeze_protocol(repo)
        check("e9_protocol_lock", True,
              f"amendment/lock frozen ({protocol['amendment']['action']}/"
              f"{protocol['lock']['action']})",
              **protocol)
    except E9Error as exc:
        check("e9_protocol_lock", False, str(exc))

    # seeds ---------------------------------------------------------------
    check(
        "e9_perturbation_seeds",
        PERTURBATIONS == {"P1": 20260921, "P2": 20260922, "P3": 20260923},
        f"perturbation seeds {PERTURBATIONS}",
        seeds=dict(PERTURBATIONS),
    )

    # counts ------------------------------------------------------------
    counts_ok = (
        ORIGINAL_POOL_COUNT == 384
        and RETAINED_COUNT == 320
        and MASKED_COUNT == 64
        and SELECTED_COUNT == 256
        and RETAINED_COUNT - SELECTED_COUNT == 64
    )
    check(
        "e9_fixed_counts",
        counts_ok,
        f"pool={ORIGINAL_POOL_COUNT} retained={RETAINED_COUNT} masked={MASKED_COUNT} "
        f"selected={SELECTED_COUNT} slack={RETAINED_COUNT - SELECTED_COUNT}",
        original_pool_count=ORIGINAL_POOL_COUNT, retained_count=RETAINED_COUNT,
        masked_count=MASKED_COUNT, selected_count=SELECTED_COUNT,
    )

    # selector resolution + proof -------------------------------------
    prov = None
    try:
        prov = selector_provenance()
        from prism_fas.recipes.selection import select as _sel_fn  # noqa: F401
        ok = (
            prov["selection_version"] == SELECTOR_VERSION
            and prov["final_bank_size_per_arm"] == SELECTED_COUNT
            and prov["minimum_eligible_pool_per_arm"] == RETAINED_COUNT
            and prov["raw_candidate_slots_per_arm"] == ORIGINAL_POOL_COUNT
            and _HEX64_RE.match(prov["selector_module_sha256"]) is not None
        )
        check("e9_selector_resolved", ok,
              f"selector {SELECTOR_CALLABLE} ({prov['selection_version']}) resolved and reused",
              **prov)
    except Exception as exc:  # noqa: BLE001
        check("e9_selector_resolved", False, f"selector unresolved: {exc}")

    # ontology identity ---------------------------------------------
    ont_path = repo / ONTOLOGY_RELPATH
    if ont_path.exists():
        ont_sha = sha256_file(ont_path)
        check("e9_ontology_identity", ont_sha == ONTOLOGY_IDENTITY,
              f"ontology sha256 {ont_sha}", path=ONTOLOGY_RELPATH,
              observed=ont_sha, expected=ONTOLOGY_IDENTITY)
    else:
        check("e9_ontology_identity", False, f"missing ontology: {ont_path}")

    # historical reference identities -----------------------------
    refs = None
    try:
        refs = load_reference_identities(repo)
        arms_ok = all(
            refs["arms"][a]["reference_selected_set_identity"]
            and refs["arms"][a]["reference_eligible"] == 384
            and refs["arms"][a]["reference_selected"] == 256
            for a in ARMS
        )
        check("e9_reference_256_bank_identities", arms_ok,
              "historical selected-256 identities resolved for RND/DET/LLM",
              arms={a: refs["arms"][a]["reference_selected_set_identity"] for a in ARMS})
    except E9Error as exc:
        check("e9_reference_256_bank_identities", False, str(exc))

    # frozen 384 candidate pools ---------------------------------
    pool_report: dict[str, Any] = {}
    pools_ok = True
    binding_path = repo / POOL_BINDING_RELPATH
    binding = json.loads(binding_path.read_text(encoding="utf-8")) if binding_path.exists() else {}
    for arm in ARMS:
        rel = POOL_INPUT_RELPATHS[arm]
        p = repo / rel
        if not p.exists():
            pools_ok = False
            pool_report[arm] = {"path": rel, "present": False,
                                "reason": "frozen 384-candidate pool artifact absent; "
                                "run --materialize-frozen-pools"}
            continue
        try:
            cand = load_frozen_pool(p)
            ref_ids = set(
                (refs or {}).get("arms", {}).get(arm, {})
                .get("reference_selected_recipe_identities", [])
            )
            pool_ids = {c["candidate_id"] for c in cand}
            selected_subset = bool(ref_ids) and ref_ids <= pool_ids
            bound = (binding.get("arms", {}) or {}).get(arm, {})
            bound_sha_ok = (
                bound.get("pool_file_sha256") == sha256_file(p)
                if bound.get("pool_file_sha256") else None
            )
            this_pool_identity = pool_identity(cand)
            c3_eligible_pool_identity = (
                (refs or {}).get("arms", {}).get(arm, {}).get("reference_eligible_pool_identity")
            )
            eligible_identity_ok = (
                this_pool_identity == c3_eligible_pool_identity
                if c3_eligible_pool_identity else None
            )
            pool_report[arm] = {
                "path": rel,
                "present": True,
                "count": len(cand),
                "count_is_384": len(cand) == ORIGINAL_POOL_COUNT,
                "pool_identity": this_pool_identity,
                "c3_eligible_pool_identity": c3_eligible_pool_identity,
                "matches_c3_eligible_pool_identity": eligible_identity_ok,
                "file_sha256": sha256_file(p),
                "binding_sha256_matches": bound_sha_ok,
                "binding_status": bound.get("status"),
                "binding_method": bound.get("method"),
                "reference_selected_256_subset_of_pool": selected_subset,
            }
            if (len(cand) != ORIGINAL_POOL_COUNT or not selected_subset
                    or bound_sha_ok is False or eligible_identity_ok is False):
                pools_ok = False
        except E9Error as exc:
            pools_ok = False
            pool_report[arm] = {"path": rel, "present": True, "reason": str(exc)}
    check("e9_frozen_384_pools", pools_ok,
          "each arm has a frozen pool of exactly 384 canonical candidates"
          if pools_ok else
          "one or more frozen 384-candidate pools cannot be proven; E9 is BLOCKED",
          arms=pool_report,
          remediation=(
              "materialize each arm's frozen 384-candidate pool as a hash-locked "
              f"JSONL under {OUTPUT_SUBTREE}/frozen_pools/ from the AUTHORIZED C3 "
              "evidence (LLM: the 12 raw provider archives; RND/DET: the frozen "
              "deterministic control schedules) and record it in E9_INPUT_BINDING.json. "
              "E9 never regenerates recipes."
          ))

    # no target / no LLM / no network / no torch dependency (structural) ----
    imported = _module_imports(Path(__file__))
    forbidden_prefixes = (
        "torch", "google.genai", "google.generativeai", "genai",
        "prism_fas.evaluation.firewall", "prism_fas.evaluation.target_prediction",
        "prism_fas.detector", "prism_fas.data",
        "requests", "urllib.request", "urllib3", "http.client", "socket",
        "modal",
    )
    hits = sorted(
        name for name in imported
        if any(name == p or name.startswith(p + ".") for p in forbidden_prefixes)
    )
    check("e9_no_target_or_llm_dependency", not hits,
          "E9 module imports no target-prediction / LLM / network / torch / "
          "detector / dataset surface",
          scanned_module=Path(__file__).name,
          imported_top_level=sorted({n.split('.')[0] for n in imported}),
          suspicious=hits)

    ok = all(c["ok"] for c in checks)
    blocked = any(c["check_id"] == "e9_frozen_384_pools" and not c["ok"] for c in checks)
    status = "PASS" if ok else ("BLOCKED" if blocked else "FAIL")

    # Scientific-execution readiness is a SEPARATE, stricter gate than the
    # engineering preflight status. It requires a committed, unmodified E9
    # implementation and a frozen execution binding.
    sci_blockers: list[str] = []
    if status != "PASS":
        sci_blockers.append(f"engineering preflight status is {status}")
    if cprov["implementation_commit_status"] != IMPL_COMMITTED_CLEAN:
        sci_blockers.append(
            f"E9 implementation is {cprov['implementation_commit_status']} "
            f"(needs {IMPL_COMMITTED_CLEAN})")
    if not cprov["scientific_code_tracked_and_clean"]:
        sci_blockers.append("scientific code (E9 module/test + selector) not all tracked+clean")
    if not cprov["base_commit_is_ancestor_of_head"]:
        sci_blockers.append("base commit is not an ancestor of the execution commit")
    if not cprov["implementation_source_commit_is_ancestor_of_head"]:
        sci_blockers.append("implementation_source_commit is not an ancestor of the execution commit")
    eb = verify_execution_binding(repo)
    if not eb["ok"]:
        sci_blockers.append("execution binding invalid/absent: " + "; ".join(eb["failures"]))
    scientific_execution_ready = not sci_blockers

    body = {
        "schema_version": "e9-preflight-v3",
        "milestone": "E9",
        "status": status,
        "engineering_status": "SMOKE_PASS" if ok else "BLOCKED",
        "scientific_status": "NOT_RUN",
        "base_commit": E9_BASE_COMMIT,
        "implementation_source_commit": cprov["implementation_source_commit"],
        "execution_commit": head,
        "implementation_commit_status": cprov["implementation_commit_status"],
        "scientific_execution_ready": scientific_execution_ready,
        "scientific_execution_blockers": sci_blockers,
        "code_provenance": cprov,
        "execution_binding_check": eb,
        "checks": checks,
        "checks_run": len(checks),
        "checks_failed": sum(1 for c in checks if not c["ok"]),
        "protocol": protocol,
        "reference_identities": refs,
        "selector_provenance": prov,
        "writes_scientific_selected_banks": False,
        "runs_milp": False,
        "creates_perturbation_selection_outcomes": False,
        "target_access": False,
        "target_labels_accessed": False,
        "target_features_accessed": False,
        "llm_calls": 0,
    }
    write_json_atomic(repo / f"{OUTPUT_SUBTREE}/E9_PREFLIGHT.json", body, root=repo)
    return body


# --------------------------------------------------------------------------- #
# Scientific execution (guarded)
# --------------------------------------------------------------------------- #

def run_bank_perturbation(repo: Path, *, authorized: bool) -> dict[str, Any]:
    if not authorized:
        raise E9Error(
            "E9 scientific execution requires --run --authorize-e9-bank-perturbation"
        )

    # --- FAIL-CLOSED provenance gate, BEFORE any pool load / membership freeze /
    #     selection. Recomputes live git state; never mutates anything.
    gate = verify_execution_binding(repo)
    if not gate["ok"]:
        raise E9Blocked(
            "E9 scientific execution REFUSED -- not a clean, committed, "
            "execution-bound state:\n  - "
            + "\n  - ".join(gate["failures"])
            + "\nTransition: review -> commit E9 module+tests+protocol -> restore "
            "LLM archives -> materialize/prove pools -> commit E9 input artifacts "
            "-> --freeze-execution-binding -> --preflight -> --run."
        )
    binding = gate["binding"]

    # --- idempotency: a prior closure for THIS execution binding is a no-op; a
    #     closure for a DIFFERENT binding fails closed. A later archive commit
    #     moves HEAD, so re-running from it is already refused by the gate above.
    closure_path = repo / CLOSURE_RELPATH
    if closure_path.exists():
        prior = json.loads(closure_path.read_text(encoding="utf-8"))
        if prior.get("execution_binding_identity") == binding["execution_binding_identity"] \
                and prior.get("execution_commit") == binding["execution_commit"]:
            return {"preflight": "PASS", "closure": prior, "idempotent_noop": True}
        raise E9Blocked(
            "E9_CLOSURE.json already exists for a DIFFERENT execution binding "
            f"({prior.get('execution_binding_identity')}); refusing to overwrite a "
            "scientific closure. Investigate before proceeding."
        )

    pre = preflight(repo)
    if pre["status"] != "PASS":
        raise E9Blocked(
            f"E9 preflight status is {pre['status']}; refusing to run the 9 MILP "
            "perturbation selections until every preflight check passes"
        )
    if not pre["scientific_execution_ready"]:
        raise E9Blocked(
            "E9 preflight scientific_execution_ready=false: "
            + "; ".join(pre["scientific_execution_blockers"])
        )

    from prism_fas.recipes.ontology import load_ontology

    ontology = load_ontology(repo / ONTOLOGY_RELPATH)
    refs = load_reference_identities(repo)

    # 1) freeze all 9 memberships first --------------------------------
    pools: dict[str, list[dict[str, Any]]] = {
        arm: load_frozen_pool(repo / POOL_INPUT_RELPATHS[arm]) for arm in ARMS
    }
    membership_rows: list[dict[str, Any]] = []
    membership_index: dict[str, Any] = {}
    for arm in ARMS:
        ids = [c["candidate_id"] for c in pools[arm]]
        for pid, seed in PERTURBATIONS.items():
            part = partition_pool(arm, seed, ids)
            for row in part["rows"]:
                row["perturbation_id"] = pid
                membership_rows.append(row)
            membership_index[f"{arm}|{pid}"] = {
                "seed": seed,
                "membership_identity": part["membership_identity"],
                "retained_count": len(part["retained"]),
                "masked_count": len(part["masked"]),
            }
    membership_jsonl = "".join(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for r in membership_rows
    )
    write_text_atomic(repo / f"{OUTPUT_SUBTREE}/E9_PERTURBATION_MEMBERSHIP.jsonl",
                      membership_jsonl, root=repo)

    # 2) run the frozen selector once per realization -----------------
    results: list[dict[str, Any]] = []
    realized_selected: dict[str, dict[str, list[str]]] = {a: {} for a in ARMS}
    for arm in ARMS:
        for pid, seed in PERTURBATIONS.items():
            row = run_realization(arm, pid, seed, pools[arm], ontology)
            row["reference_selected_set_identity"] = \
                refs["arms"][arm]["reference_selected_set_identity"]
            results.append(row)
            realized_selected[arm][pid] = list(row.get("selected_candidate_ids", []))

    results_jsonl = "".join(
        json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        for r in results
    )
    write_text_atomic(repo / f"{OUTPUT_SUBTREE}/E9_SELECTION_RESULTS.jsonl",
                      results_jsonl, root=repo)

    # 3) descriptive stability ---------------------------------------
    stability: dict[str, Any] = {}
    csv_lines = ["arm,perturbation_id,seed,status,selected_count,intersection_with_reference,"
                 "jaccard_vs_reference,reference_retention_fraction"]
    for arm in ARMS:
        ref_selected = refs["arms"][arm]["reference_selected_recipe_identities"]
        stability[arm] = stability_for_arm(arm, ref_selected, realized_selected[arm])
        for pid, seed in PERTURBATIONS.items():
            per = stability[arm]["per_realization_vs_reference"][pid]
            r = next(x for x in results if x["arm"] == arm and x["perturbation_id"] == pid)
            csv_lines.append(
                f"{arm},{pid},{seed},{r['status']},{per['selected_count']},"
                f"{per['intersection']},{per['jaccard']:.6f},"
                f"{per['reference_retention_fraction']:.6f}"
            )
    write_text_atomic(repo / f"{OUTPUT_SUBTREE}/E9_BANK_STABILITY.csv",
                      "\n".join(csv_lines) + "\n", root=repo)
    stability_body = {
        "schema_version": "e9-bank-stability-v1",
        "milestone": "E9",
        "arms": stability,
        "target_access": False,
        "llm_calls": 0,
    }
    write_json_atomic(repo / f"{OUTPUT_SUBTREE}/E9_BANK_STABILITY.json",
                      stability_body, root=repo)

    # 4) evidence + closure ---------------------------------------
    any_blocked = any(r["status"] == "BLOCKED" for r in results)
    evidence_files = [
        "E9_INPUT_AUDIT.json",
        "E9_PROTOCOL_AMENDMENT.json", "E9_PROTOCOL_LOCK.json",
        "E9_INPUT_BINDING.json",
        "frozen_pools/RND_POOL_384.jsonl", "frozen_pools/DET_POOL_384.jsonl",
        "frozen_pools/LLM_POOL_384.jsonl",
        "E9_EXECUTION_BINDING.json", "E9_PREFLIGHT.json",
        "E9_PERTURBATION_MEMBERSHIP.jsonl", "E9_SELECTION_RESULTS.jsonl",
        "E9_BANK_STABILITY.csv", "E9_BANK_STABILITY.json",
    ]
    sha_lines = []
    for name in evidence_files:
        p = repo / OUTPUT_SUBTREE / name
        if p.exists():
            sha_lines.append(f"{sha256_file(p)}  {OUTPUT_SUBTREE}/{name}")
    write_text_atomic(repo / f"{OUTPUT_SUBTREE}/E9_EVIDENCE.sha256",
                      "\n".join(sha_lines) + "\n", root=repo)

    eb = gate["binding"]
    closure = {
        "schema_version": "e9-closure-v3",
        "milestone": "E9",
        "status": "CLOSED_WITH_BLOCKED_REALIZATIONS" if any_blocked else "CLOSED",
        "base_commit": E9_BASE_COMMIT,
        "implementation_source_commit": eb["implementation_source_commit"],
        "execution_commit": eb["execution_commit"],
        "base_commit_is_ancestor_of_execution_commit": True,
        "implementation_source_commit_is_ancestor_of_execution_commit": True,
        "e9_module_sha256": eb["e9_module_sha256"],
        "e9_test_sha256": eb["e9_test_sha256"],
        "selector_module_sha256": eb["selector_module_sha256"],
        "protocol_lock_identity": eb["protocol_lock_identity"],
        "e9_protocol_amendment_identity": pre["protocol"]["e9_protocol_amendment_identity"],
        "input_binding_identity": eb["input_binding_identity"],
        "pool_identity_by_arm": eb["pool_identity_by_arm"],
        "execution_binding_identity": eb["execution_binding_identity"],
        "execution_binding_file_sha256": gate["binding_file_sha256"],
        "arms": list(ARMS),
        "perturbation_seeds": dict(PERTURBATIONS),
        "realizations_total": len(results),
        "realizations_ok": sum(1 for r in results if r["status"] == "OK"),
        "realizations_blocked": sum(1 for r in results if r["status"] == "BLOCKED"),
        "selector": selector_provenance(),
        "bank_size_ever_lowered": False,
        "retained_fraction_changed_after_outcomes": False,
        "any_realization_retried": False,
        "target_access": False,
        "target_labels_accessed": False,
        "target_features_accessed": False,
        "llm_calls": 0,
        "evidence_sha256": f"{OUTPUT_SUBTREE}/E9_EVIDENCE.sha256",
        "post_run_archive_rule": (
            "E9_EXECUTION_BINDING.json, E9_PERTURBATION_MEMBERSHIP.jsonl, "
            "E9_SELECTION_RESULTS.jsonl, E9_BANK_STABILITY.*, E9_EVIDENCE.sha256 and "
            "this closure may be committed in a LATER additive evidence commit. That "
            "commit is NOT execution_commit and must not trigger a scientific rerun "
            "(the run gate refuses any HEAD != execution_commit)."
        ),
    }
    write_json_atomic(repo / f"{OUTPUT_SUBTREE}/E9_CLOSURE.json", closure, root=repo)
    return {"preflight": pre["status"], "closure": closure,
            "results": results, "stability": stability_body}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "E9 -- conditional bank-selection robustness from frozen candidate "
            "pools. No LLM, no network, no GPU, no detector training, no target "
            "label/prediction/metric access."
        )
    )
    parser.add_argument(
        "--materialize-frozen-pools", action="store_true",
        help=("Input preparation (not science): reconstruct and PROVE each arm's "
              "frozen 384-candidate pool (RND/DET from the frozen schedule, LLM "
              "from the frozen provider archives) and write frozen_pools/*.jsonl "
              "+ E9_INPUT_BINDING.json. No perturbation, no MILP, no LLM call."),
    )
    parser.add_argument(
        "--code-provenance", action="store_true",
        help="Read-only: print base_commit / implementation_source_commit / "
             "execution_commit / implementation_commit_status and the E9 + "
             "selector git state.",
    )
    parser.add_argument(
        "--freeze-execution-binding", action="store_true",
        help=("Input preparation, run AFTER the final clean execution commit "
              "(scientific code + E9 input artifacts all committed): freeze "
              "E9_EXECUTION_BINDING.json pinning execution_commit + every code/"
              "input identity. Immutable evidence; need NOT be committed before "
              "--run. Fails closed unless everything is tracked+clean."),
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help=("Read-only w.r.t. science. Freezes the additive protocol "
              "amendment/lock, validates base-commit binding, code provenance, "
              "seeds, counts, selector, ontology, historical 256-bank identities "
              "and the frozen 384 pools. Runs no MILP and writes no selected bank."),
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Scientific execution: freeze 9 memberships, then run the frozen "
             "MILP selector once per realization. Requires the authorization flag.",
    )
    parser.add_argument(
        "--authorize-e9-bank-perturbation", action="store_true",
        help="Explicit authorization required by --run.",
    )
    args = parser.parse_args(argv)
    repo = repo_root()

    if getattr(args, "code_provenance", False) and not args.run:
        print(json.dumps(code_provenance(repo), indent=2, default=str))
        return 0

    if getattr(args, "materialize_frozen_pools", False) and not args.run:
        try:
            body = materialize_frozen_pools(repo)
        except E9Error as exc:
            print(f"E9 materialize refused: {exc}")
            return 1
        print(json.dumps(body, indent=2, default=str))
        return 0 if body.get("all_pools_materialized") else 0

    if getattr(args, "freeze_execution_binding", False) and not args.run:
        try:
            body = freeze_execution_binding(repo)
        except E9Error as exc:
            print(f"E9 freeze-execution-binding refused: {exc}")
            return 1
        print(json.dumps(body, indent=2, default=str))
        return 0

    if args.preflight and not args.run:
        body = preflight(repo)
        print(json.dumps(body, indent=2, default=str))
        return 0 if body["status"] in ("PASS", "BLOCKED") else 1

    if args.run:
        try:
            out = run_bank_perturbation(
                repo, authorized=args.authorize_e9_bank_perturbation
            )
        except E9Error as exc:
            print(f"E9 run refused: {exc}")
            return 1
        print(json.dumps({"preflight": out["preflight"],
                          "closure": out["closure"]}, indent=2, default=str))
        return 0

    print("Pass --preflight (read-only) or "
          "--run --authorize-e9-bank-perturbation (scientific).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
